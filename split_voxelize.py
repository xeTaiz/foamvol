"""Split-aware voxelization for the two-sided thin-surface sub-cell partition.

Each Voronoi cell optionally carries an oriented internal surface (quaternion
frame + K-texel soft-Voronoi height field) that splits the cell into two
regions with densities mu_plus and mu_minus. This module evaluates that split
at arbitrary query points -- mirroring the CUDA forward kernel
(ct_thinsurface_forward in src/tracing/pipeline.cu) -- and voxels a trained
CTScene into a regular 3D grid with optional supersampling.

Skip meshing. For voxelization each sample point is assigned to its owning
Voronoi cell (nearest primal site) and evaluated against that cell's internal
surface; voxels straddling the surface average both sides via supersampling.

IMPORTANT -- what this metric is and is not:
  This is a *split-NN query-field* voxelization, not an exact inverse of the
  CUDA ray renderer. Each voxel sample takes the density of whichever side of
  its owning cell's learned internal surface the sample point falls on (nearest
  cell = the Voronoi owner). It does NOT line-integrate along rays, so it is a
  volumetric field estimate, not a projection-consistent reconstruction.
  Compare baseline vs thin-surface runs with the SAME script so the query-field
  convention is held constant across arms.

Side selection / smoothing:
  By default the side is HARD (no blend): each sample gets mu_plus or mu_minus
  by sign(signed_dist). Pass blend_eps > 0 (or --blend_eps) to linearly blend
  across |signed_dist| < blend_eps; this is a voxelization-time smoothing that
  can silently inflate PSNR against a smooth GT, so it is OFF by default and
  must be opt-in.

CLI mirrors voxelize.py:
    python split_voxelize.py --model model.pt --resolution 256 --supersample 4
    python split_voxelize.py --model model.pt --gt data/vol_gt.npy
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import radfoam
from radfoam_model.scene import assert_supported_thin_K
# Reuse the existing volume-eval helpers so metrics/slices/NIfTI match voxelize.
from voxelize import (
    compute_volume_psnr,
    compute_volume_ssim,
    gaussian_blur_3d,
    save_slices,
)


# ---------------------------------------------------------------------------
# Quaternion -> orthonormal frame (matches quat_to_frame in pipeline.cu)
# ---------------------------------------------------------------------------

def quat_to_frame(q: torch.Tensor):
    """q: (..., 4) [w, x, y, z] -> (n, t, b) each (..., 3).

    Column 0 is the surface normal n; columns 1-2 are in-plane axes t, b.
    """
    w, x, y, z = q.unbind(-1)
    inv_norm = torch.rsqrt(w * w + x * x + y * y + z * z + 1e-12)
    w = w * inv_norm
    x = x * inv_norm
    y = y * inv_norm
    z = z * inv_norm

    n = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)
    t = torch.stack([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dim=-1)
    b = torch.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)
    return n, t, b


# ---------------------------------------------------------------------------
# Split-cell point query (no radfoam dependency -- usable on CPU for tests)
# ---------------------------------------------------------------------------

def split_cell_query(
    query,
    points,
    nn_idx,
    density,            # raw density_base (N,) or (N,1)
    density_delta,      # (N,1)
    quaternions,        # (N,4)
    texel_sites_2d,     # (N,K,2)
    texel_heights,      # (N,K)
    cell_radius,        # (N,)
    thin_temp: float = 10.0,
    activation_scale: float = 1.0,
    blend_eps: float = 0.0,
):
    """Evaluate the two-sided thin-surface density at a batch of query points.

    For each query q with owning cell i = nn_idx[b]:
      mu_bar = activation_scale * softplus(density[i])
      delta = density_delta[i]
      mu_plus  = max(mu_bar + delta, 0),  mu_minus = max(mu_bar - delta, 0)
      frame (n, t, b) <- quat[i]
      in-plane projection p = cp + (t.(q-cp)) t + (b.(q-cp)) b
      soft-Voronoi height h = sum_k w_k * r*heights[k] / sum_k w_k,
        w_k = exp(-thin_temp * |p - site3_k|^2 / r^2),
        site3_k = cp + r*(s2d[k,0]*t + s2d[k,1]*b)
      signed distance s = n.(q - cp) - h
      value = mu_plus if s > 0 else mu_minus  (hard side by default; linear
      blend across |s| < blend_eps only when blend_eps > 0)

    Args:
        query:        (B, 3) sample positions.
        points:       (N, 3) Voronoi sites (cell centers).
        nn_idx:       (B,) long -- owning cell index per query (e.g. radfoam.nn).
        density:      (N,) or (N,1) raw density_base.
        density_delta, quaternions, texel_sites_2d, texel_heights, cell_radius:
            the four thin-surface params + per-cell radius
            (N,1)/(N,4)/(N,K,2)/(N,K)/(N,).

    Returns:
        value:       (B,) evaluated density.
        side:        (B,) +1 for mu_plus side, -1 for mu_minus side, 0 in blend band.
        signed_dist: (B,) signed distance to the internal surface (scene units).
    """
    if density.dim() == 2:
        density = density.squeeze(-1)
    if density_delta.dim() == 2:
        delta = density_delta.squeeze(-1)        # (N,)
    else:
        delta = density_delta
    cr = cell_radius.reshape(-1).clamp_min(1e-12)  # (N,)

    cp = points[nn_idx]                           # (B, 3)
    rel = query - cp                              # (B, 3)
    n, t, b = quat_to_frame(quaternions[nn_idx])  # each (B, 3)
    r = cr[nn_idx]                                # (B,)

    # In-plane projection of q onto the tangent plane through cp.
    tn = (t * rel).sum(-1)                        # (B,)
    tb = (b * rel).sum(-1)                        # (B,)
    p = cp + tn.unsqueeze(-1) * t + tb.unsqueeze(-1) * b   # (B, 3)

    # Soft-Voronoi height field over K texels.
    s2d = texel_sites_2d[nn_idx]                  # (B, K, 2)
    # sites: (B, K, 3) = cp + r*(s2d.t + s2d.b)
    sites = cp.unsqueeze(1) + (r.unsqueeze(-1).unsqueeze(-1)) * (
        s2d[..., :1] * t.unsqueeze(1) + s2d[..., 1:] * b.unsqueeze(1)
    )                                             # (B, K, 3)
    d2 = ((p.unsqueeze(1) - sites) ** 2).sum(-1) / (r.unsqueeze(-1) ** 2 + 1e-20)
    w = torch.exp(-thin_temp * d2)                # (B, K)
    h_k = texel_heights[nn_idx]                   # (B, K)
    w_sum = w.sum(-1).clamp_min(1e-20)
    h_eval = (w * (r.unsqueeze(-1) * h_k)).sum(-1) / w_sum   # (B,)

    signed_dist = (n * rel).sum(-1) - h_eval      # (B,)

    # Densities
    raw = density[nn_idx]                         # (B,)
    mu_bar = F.softplus(raw, beta=10.0) * activation_scale
    d_val = delta[nn_idx]                         # (B,)
    mu_p = torch.clamp(mu_bar + d_val, min=0.0)
    mu_n = torch.clamp(mu_bar - d_val, min=0.0)

    # Side selection. +n side (s > 0) -> mu_plus ; -n side (s < 0) -> mu_minus.
    # blend_eps == 0 (default): HARD side -- each sample takes one side's
    # density, no voxelization-time smoothing (avoids silently inflating PSNR).
    # blend_eps > 0: linear blend across |s| < blend_eps for smoothness.
    s = signed_dist
    if blend_eps and blend_eps > 0.0:
        alpha = torch.clamp(0.5 + s / (2.0 * blend_eps), 0.0, 1.0)
        side = torch.where(s > blend_eps, torch.ones_like(s),
                           torch.where(s < -blend_eps, -torch.ones_like(s),
                                       torch.zeros_like(s)))
    else:
        alpha = (s > 0).float()
        side = torch.where(s > 0, torch.ones_like(s),
                           torch.where(s < 0, -torch.ones_like(s),
                                       torch.zeros_like(s)))
    value = alpha * mu_p + (1.0 - alpha) * mu_n
    return value, side, signed_dist


# ---------------------------------------------------------------------------
# Voxelization driver (mirrors voxelize.voxelize but split-aware)
# ---------------------------------------------------------------------------

def voxelize_split(
    model_path,
    resolution,
    output_path,
    extent=1.0,
    blur_sigma=0.0,
    supersample=3,
    thin_temp=10.0,
    activation_scale=1.0,
    gt_path=None,
    side_map_path=None,
    blend_eps=0.0,
):
    """Voxelize a thin-surface checkpoint into a regular 3D grid.

    Falls back to scalar softplus density if the checkpoint has no thin-surface
    state (so the same script works on baseline checkpoints for comparison).
    """
    device = torch.device("cuda")

    scene_data = torch.load(model_path)
    points = scene_data["xyz"].to(device)
    density_flat = scene_data["density"].to(device).squeeze(-1)
    # CSR adjacency must be uint32 -- the C++/CUDA pipeline reads it as
    # uint32_t column indices / row pointers. int32 would misinterpret on
    # large graphs and is the dtype the trained CTScene stores.
    adjacency = scene_data["adjacency"].to(device).to(torch.uint32)
    adjacency_offsets = scene_data["adjacency_offsets"].to(device).to(torch.uint32)
    aabb_tree = radfoam.build_aabb_tree(points)
    _, cell_radius = radfoam.farthest_neighbor(points, adjacency, adjacency_offsets)

    ts_meta = scene_data.get("thin_surface")
    has_ts = ts_meta is not None and ts_meta.get("active", False)
    if has_ts:
        K = int(ts_meta.get("K", 4))
        assert_supported_thin_K(K)
        density_delta = scene_data["density_delta"].to(device)
        quaternions = scene_data["quaternions"].to(device)
        texel_sites_2d = scene_data["texel_sites_2d"].to(device)
        texel_heights = scene_data["texel_heights"].to(device)
        print(f"[split-voxelize] thin-surface ON, K={K}, N={points.shape[0]}")
    else:
        K = 0
        density_delta = quaternions = texel_sites_2d = texel_heights = None
        print(f"[split-voxelize] no thin-surface state; scalar softplus density "
              f"(N={points.shape[0]})")

    grid_min = torch.tensor([-extent, -extent, -extent], device=device)
    grid_max = torch.tensor([extent, extent, extent], device=device)

    # Voxel CENTERS (not endpoints): sample at (i+0.5)/res so each voxel is
    # represented by its centroid and the grid covers [grid_min, grid_max)
    # without double-counting the boundary at grid_max.
    coords = (torch.arange(resolution, device=device) + 0.5) / resolution
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    voxel_centers = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    voxel_centers = grid_min + voxel_centers * (grid_max - grid_min)

    def _eval(query_pts):
        nn_idx = radfoam.nn(points, aabb_tree, query_pts).long()
        if has_ts:
            val, _side, _s = split_cell_query(
                query_pts, points, nn_idx, density_flat, density_delta,
                quaternions, texel_sites_2d, texel_heights, cell_radius,
                thin_temp=thin_temp, activation_scale=activation_scale,
                blend_eps=blend_eps,
            )
            return torch.nan_to_num(val), _side
        else:
            mu = F.softplus(density_flat, beta=10.0) * activation_scale
            return torch.nan_to_num(mu[nn_idx]), None

    k = supersample
    num_voxels = voxel_centers.shape[0]
    volume = torch.zeros(num_voxels, device=device)
    side_acc = torch.zeros(num_voxels, device=device) if (has_ts and side_map_path) else None

    if k <= 1:
        batch_size = 2_000_000
        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            v, sd = _eval(voxel_centers[start:end])
            volume[start:end] = v
            if side_acc is not None and sd is not None:
                side_acc[start:end] = sd.float()
    else:
        voxel_size = (grid_max - grid_min) / resolution
        sub_coords = torch.linspace(-0.5 + 0.5 / k, 0.5 - 0.5 / k, k, device=device)
        ox, oy, oz = torch.meshgrid(sub_coords, sub_coords, sub_coords, indexing="ij")
        offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3) * voxel_size
        samples_per_voxel = k ** 3
        batch_size = max(1, 2_000_000 // samples_per_voxel)
        print(f"Supersampling: {k}^3 = {samples_per_voxel} samples/voxel, "
              f"{batch_size} voxels/batch")
        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            centers = voxel_centers[start:end]
            sub_points = (centers.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
            v, sd = _eval(sub_points)
            volume[start:end] = v.reshape(-1, samples_per_voxel).mean(dim=1)
            if side_acc is not None and sd is not None:
                side_acc[start:end] = sd.reshape(-1, samples_per_voxel).float().mean(dim=1)

    volume = volume.reshape(resolution, resolution, resolution)
    if blur_sigma > 0:
        volume = gaussian_blur_3d(volume, kernel_size=3, sigma=blur_sigma)
    volume_np = volume.cpu().numpy()
    np.save(output_path, volume_np)
    print(f"Saved volume {volume_np.shape} to {output_path}")
    print(f"  min={volume_np.min():.4f}, max={volume_np.max():.4f}, "
          f"mean={volume_np.mean():.4f}")

    if side_acc is not None:
        side_np = side_acc.reshape(resolution, resolution, resolution).cpu().numpy()
        np.save(side_map_path, side_np)
        print(f"Saved side map to {side_map_path} "
              f"(frac mu_plus side = {(side_np > 0).mean():.3f})")

    gt_volume = None
    if gt_path is not None and os.path.exists(gt_path):
        gt_volume = np.load(gt_path).astype(np.float32)
        psnr = compute_volume_psnr(volume_np, gt_volume)
        ssim = compute_volume_ssim(volume_np, gt_volume)
        print(f"  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")

    save_slices(volume_np, gt_volume, output_path, extent)

    if output_path.endswith(".npy"):
        nifti_path = output_path.replace(".npy", ".nii.gz")
    else:
        nifti_path = output_path + ".nii.gz"
    try:
        import nibabel as nib
        voxel_size_np = ((grid_max - grid_min) / resolution).cpu().numpy()
        affine = np.diag([*voxel_size_np, 1.0])
        affine[:3, 3] = grid_min.cpu().numpy()
        nib.save(nib.Nifti1Image(volume_np, affine), nifti_path)
        print(f"Saved NIfTI to {nifti_path}")
    except ImportError:
        pass

    return volume_np


def main():
    parser = argparse.ArgumentParser(
        description="Voxelize a thin-surface (split-cell) CT reconstruction")
    parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--extent", type=float, default=1.0)
    parser.add_argument("--blur_sigma", type=float, default=0.0)
    parser.add_argument("--supersample", type=int, default=3,
                        help="Sub-samples per axis per voxel (k^3 total, 1=center)")
    parser.add_argument("--thin_temp", type=float, default=10.0,
                        help="Soft-Voronoi Gaussian bandwidth (must match training)")
    parser.add_argument("--activation_scale", type=float, default=1.0)
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--side_map", type=str, default=None,
                        help="Optional output path for the +/-1 side-map .npy")
    parser.add_argument("--blend_eps", type=float, default=0.0,
                        help="Linear blend band half-width around the surface "
                             "(0 = hard side, default; >0 softens voxels "
                             "straddling the surface -- can inflate PSNR)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.model), "volume_split.npy")
    voxelize_split(args.model, args.resolution, output, args.extent,
                   args.blur_sigma, args.supersample, args.thin_temp,
                   args.activation_scale, args.gt, args.side_map,
                   args.blend_eps)


if __name__ == "__main__":
    main()
