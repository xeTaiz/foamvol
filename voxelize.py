"""Voxelize a trained CTScene into a regular 3D grid.

For each voxel center, finds the nearest Voronoi site and assigns that cell's
density. This is exact nearest-neighbor interpolation (correct for Voronoi).

Usage:
    python voxelize.py --model model.pt --resolution 128 --output volume.npy
"""

import argparse
import os

import numpy as np
import torch
import radfoam
import torch.nn.functional as F


def voxelize(model_path, resolution, output_path, extent=None):
    device = torch.device("cuda")

    scene_data = torch.load(model_path)
    points = scene_data["xyz"].to(device)
    density_raw = scene_data["density"].to(device)
    activation_scale = 1.0

    # Apply the same activation as CTScene.get_primal_density()
    density = activation_scale * F.softplus(density_raw, beta=10)

    # Build AABB tree for nearest-neighbor queries
    aabb_tree = radfoam.build_aabb_tree(points)

    # Determine bounding box
    if extent is None:
        pts_min = points.min(dim=0).values
        pts_max = points.max(dim=0).values
        center = (pts_min + pts_max) / 2
        half_extent = (pts_max - pts_min).max() / 2 * 1.1
        grid_min = center - half_extent
        grid_max = center + half_extent
    else:
        grid_min = torch.tensor([-extent, -extent, -extent], device=device)
        grid_max = torch.tensor([extent, extent, extent], device=device)

    # Generate grid query points
    coords = torch.linspace(0, 1, resolution, device=device)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    grid_points = grid_min + grid_points * (grid_max - grid_min)

    # Query nearest neighbors in batches
    batch_size = 1_000_000
    volume = torch.zeros(grid_points.shape[0], device=device)

    for start in range(0, grid_points.shape[0], batch_size):
        end = min(start + batch_size, grid_points.shape[0])
        batch_queries = grid_points[start:end]
        nn_indices = radfoam.nn(points, aabb_tree, batch_queries).long()
        volume[start:end] = density[nn_indices].squeeze(-1)

    volume = volume.reshape(resolution, resolution, resolution)
    volume_np = volume.cpu().numpy()

    # Save as .npy
    np.save(output_path, volume_np)
    print(f"Saved volume {volume_np.shape} to {output_path}")
    print(f"  min={volume_np.min():.4f}, max={volume_np.max():.4f}, mean={volume_np.mean():.4f}")

    # Optionally save as NIfTI
    if output_path.endswith(".npy"):
        nifti_path = output_path.replace(".npy", ".nii.gz")
    else:
        nifti_path = output_path + ".nii.gz"

    try:
        import nibabel as nib
        voxel_size = ((grid_max - grid_min) / resolution).cpu().numpy()
        affine = np.diag([*voxel_size, 1.0])
        affine[:3, 3] = grid_min.cpu().numpy()
        img = nib.Nifti1Image(volume_np, affine)
        nib.save(img, nifti_path)
        print(f"Saved NIfTI to {nifti_path}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Voxelize a CT reconstruction")
    parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution per axis")
    parser.add_argument("--output", type=str, default=None, help="Output file path (.npy), defaults to volume.npy next to model")
    parser.add_argument("--extent", type=float, default=None, help="Half-extent of the grid (auto if not set)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.model), "volume.npy")

    voxelize(args.model, args.resolution, output, args.extent)


if __name__ == "__main__":
    main()
