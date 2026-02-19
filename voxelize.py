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


def compute_cell_gradients(centers, values, adj_indices, adj_offsets, device):
    """Fit a per-cell linear gradient via weighted least squares on Voronoi neighbors.

    Args:
        centers: (N, 3) cell centers
        values: (N,) scalar values per cell
        adj_indices: (E,) flattened CSR neighbor indices
        adj_offsets: (N+1,) CSR row offsets
        device: torch device

    Returns:
        (N, 3) gradient vector per cell
    """
    N = centers.shape[0]

    # Compute neighbor counts and cap at max_k
    max_k = 30
    counts = adj_offsets[1:] - adj_offsets[:-1]  # (N,)
    K = min(int(counts.max().item()), max_k)

    # Build padded (N, K) neighbor index tensor with validity mask
    padded_idx = torch.zeros(N, K, dtype=torch.long, device=device)
    mask = torch.zeros(N, K, dtype=torch.bool, device=device)

    cell_ids = torch.arange(N, device=device)
    base_offsets = adj_offsets[:-1]  # (N,) start offset per cell
    for k in range(K):
        has_k = counts > k
        idx = cell_ids[has_k]
        padded_idx[idx, k] = adj_indices[base_offsets[idx] + k]
        mask[idx, k] = True

    # dc: (N, K, 3) displacement vectors from cell center to neighbors
    neighbor_centers = centers[padded_idx]  # (N, K, 3)
    dc = neighbor_centers - centers[:, None, :]  # (N, K, 3)

    # dv: (N, K) value differences
    neighbor_vals = values[padded_idx]  # (N, K)
    dv = neighbor_vals - values[:, None]  # (N, K)

    # Inverse-distance weights, masked
    dist = dc.norm(dim=-1)  # (N, K)
    eps = 1e-8
    w = torch.zeros_like(dist)
    w[mask] = 1.0 / (dist[mask] + eps)

    # Zero out invalid entries
    dc[~mask] = 0.0
    dv[~mask] = 0.0

    # Weighted least squares: solve A @ grad = rhs per cell
    # A = D^T W D  (3x3), rhs = D^T W dv (3,)
    w_dc = dc * w.unsqueeze(-1)  # (N, K, 3) weighted displacements
    A = torch.einsum("nki,nkj->nij", w_dc, dc)  # (N, 3, 3)
    rhs = torch.einsum("nki,nk->ni", w_dc, dv)  # (N, 3)

    # Regularize for boundary/low-neighbor cells
    A += 1e-6 * torch.eye(3, device=device).unsqueeze(0)

    # Solve batched 3x3 systems
    gradients = torch.linalg.solve(A, rhs)  # (N, 3)

    return gradients


def sample_interpolated(query_points, nn_indices, centers, values, gradients):
    """Evaluate the piecewise-linear field: f(x) = a[idx] + grad[idx] . (x - center[idx]).

    Args:
        query_points: (M, 3) sample positions
        nn_indices: (M,) nearest cell index per query point
        centers: (N, 3) cell centers
        values: (N,) scalar values per cell
        gradients: (N, 3) gradient per cell

    Returns:
        (M,) interpolated values, clamped to min=0
    """
    a = values[nn_indices]  # (M,)
    c = centers[nn_indices]  # (M, 3)
    g = gradients[nn_indices]  # (M, 3)
    dx = query_points - c  # (M, 3)
    result = a + (g * dx).sum(dim=-1)  # (M,)
    return result.clamp(min=0.0)


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
             supersample=3, interpolate=True):
    device = torch.device("cuda")

    scene_data = torch.load(model_path)
    points = scene_data["xyz"].to(device)
    density_raw = scene_data["density"].to(device)
    activation_scale = 1.0

    # Apply the same activation as CTScene.get_primal_density()
    density = activation_scale * F.softplus(density_raw, beta=10)

    # Compute per-cell gradients for linear interpolation
    density_flat = density.squeeze(-1)  # (N,)
    gradients = None
    if interpolate:
        if "adjacency" in scene_data and "adjacency_offsets" in scene_data:
            adj_indices = scene_data["adjacency"].to(device).long()
            adj_offsets = scene_data["adjacency_offsets"].to(device).long()
            print("Computing per-cell gradients for linear interpolation...")
            gradients = compute_cell_gradients(points, density_flat, adj_indices, adj_offsets, device)
            print(f"  gradient norms: min={gradients.norm(dim=-1).min():.4f}, "
                  f"max={gradients.norm(dim=-1).max():.4f}, "
                  f"mean={gradients.norm(dim=-1).mean():.4f}")
        else:
            print("Warning: adjacency data not found in checkpoint, falling back to constant lookup")
            interpolate = False

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
        # Single center-point lookup
        batch_size = 4_000_000
        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            query = voxel_centers[start:end]
            nn_indices = radfoam.nn(points, aabb_tree, query).long()
            if interpolate and gradients is not None:
                volume[start:end] = sample_interpolated(query, nn_indices, points, density_flat, gradients)
            else:
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
            if interpolate and gradients is not None:
                sub_densities = sample_interpolated(sub_points, nn_indices, points, density_flat, gradients)
            else:
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
    parser.add_argument("--interpolate", action=argparse.BooleanOptionalAction, default=True,
                        help="Use per-cell linear interpolation (default: True, use --no-interpolate to disable)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.model), "volume.npy")

    voxelize(args.model, args.resolution, output, args.extent, args.blur_sigma,
             args.supersample, args.interpolate)


if __name__ == "__main__":
    main()
