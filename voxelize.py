"""Voxelize a trained CTScene into a regular 3D grid.

By default, uses supersampling (k^3 uniformly spaced sub-points per voxel)
to approximate the volume-weighted average density within each voxel.
Use --supersample 1 to fall back to single center-point lookup.

Usage:
    python voxelize.py --model model.pt --resolution 128 --output volume.npy
    python voxelize.py --model model.pt --resolution 256 --supersample 4
"""

import argparse
import os

import numpy as np
import torch
import radfoam
import torch.nn.functional as F


def gaussian_blur_3d(volume, kernel_size=3, sigma=1.0):
    """Apply 3D Gaussian blur to a volume tensor (X, Y, Z) on GPU."""
    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=volume.device) - pad
    gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size, kernel_size)

    vol_5d = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)
    blurred = F.conv3d(vol_5d, kernel, padding=pad)
    return blurred.squeeze(0).squeeze(0)


def voxelize(model_path, resolution, output_path, extent=None, blur_sigma=0.0,
             supersample=3):
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

    # Generate voxel center coordinates
    coords = torch.linspace(0, 1, resolution, device=device)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    voxel_centers = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    voxel_centers = grid_min + voxel_centers * (grid_max - grid_min)

    k = supersample
    num_voxels = voxel_centers.shape[0]
    volume = torch.zeros(num_voxels, device=device)

    if k <= 1:
        # Single center-point lookup (original behavior)
        batch_size = 4_000_000
        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            nn_indices = radfoam.nn(points, aabb_tree, voxel_centers[start:end]).long()
            volume[start:end] = density[nn_indices].squeeze(-1)
    else:
        # Supersample: k^3 uniformly spaced sub-points per voxel
        voxel_size = (grid_max - grid_min) / resolution  # (3,)
        sub_coords = torch.linspace(-0.5 + 0.5 / k, 0.5 - 0.5 / k, k, device=device)
        ox, oy, oz = torch.meshgrid(sub_coords, sub_coords, sub_coords, indexing="ij")
        offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)  # (k^3, 3)
        offsets = offsets * voxel_size  # scale to world units

        samples_per_voxel = k ** 3
        batch_size = max(1, 4_000_000 // samples_per_voxel)
        print(f"Supersampling: {k}^3 = {samples_per_voxel} samples/voxel, "
              f"{batch_size} voxels/batch")

        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            centers = voxel_centers[start:end]  # (B, 3)
            sub_points = centers.unsqueeze(1) + offsets.unsqueeze(0)  # (B, k^3, 3)
            sub_points = sub_points.reshape(-1, 3)  # (B*k^3, 3)
            nn_indices = radfoam.nn(points, aabb_tree, sub_points).long()
            sub_densities = density[nn_indices].squeeze(-1)  # (B*k^3,)
            volume[start:end] = sub_densities.reshape(-1, samples_per_voxel).mean(dim=1)

    volume = volume.reshape(resolution, resolution, resolution)
    if blur_sigma > 0:
        volume = gaussian_blur_3d(volume, kernel_size=3, sigma=blur_sigma)
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
    parser.add_argument("--extent", type=float, default=1.0, help="Half-extent of the grid (auto if not set)")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="Gaussian blur sigma (0 = disabled)")
    parser.add_argument("--supersample", type=int, default=3, help="Sub-samples per axis per voxel (k^3 total, 1 = center-only)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.model), "volume.npy")

    voxelize(args.model, args.resolution, output, args.extent, args.blur_sigma,
             args.supersample)


if __name__ == "__main__":
    main()
