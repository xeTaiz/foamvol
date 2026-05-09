"""Voxelize a trained CTScene into a regular 3D grid using IDW interpolation.

Uses bilateral IDW (matching the CUDA tracing kernel) to evaluate density at
voxel centers. Use --supersample k to average k^3 sub-points per voxel.
Use --supersample 1 to fall back to single center-point lookup.

If --gt is provided, computes volume PSNR/SSIM against the ground-truth volume
and saves a grid of axial/coronal/sagittal slices next to the output.

Usage:
    python voxelize.py --model model.pt --resolution 128 --output volume.npy
    python voxelize.py --model model.pt --resolution 256 --supersample 4
    python voxelize.py --model model.pt --gt data/vol_gt.npy
"""

import argparse
import os

import numpy as np
import torch
import radfoam
import torch.nn.functional as F

from radfoam_model.scene import idw_query


# Kept for vis_foam.py compatibility.
def sample_interpolated(query_points, nn_indices, centers, values, gradients, max_slope):
    a = values[nn_indices]
    mu_base = F.softplus(a, beta=10)
    c = centers[nn_indices]
    g = gradients[nn_indices]
    dx = query_points - c
    if max_slope is not None:
        slope = max_slope * torch.tanh(g)
    else:
        slope = g
    return (mu_base + (slope * dx).sum(dim=-1)).clamp(min=0)


def gaussian_blur_3d(volume, kernel_size=3, sigma=1.0):
    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=volume.device) - pad
    gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size, kernel_size)
    vol_5d = volume.unsqueeze(0).unsqueeze(0)
    return F.conv3d(vol_5d, kernel, padding=pad).squeeze(0).squeeze(0)


def compute_volume_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_volume_ssim(pred, gt, window_size=11):
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return 1.0
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2
    mu1 = F.conv2d(pred_t, kernel, padding=pad)
    mu2 = F.conv2d(gt_t, kernel, padding=pad)
    sigma1_sq = F.conv2d(pred_t ** 2, kernel, padding=pad) - mu1 ** 2
    sigma2_sq = F.conv2d(gt_t ** 2, kernel, padding=pad) - mu2 ** 2
    sigma12 = F.conv2d(pred_t * gt_t, kernel, padding=pad) - mu1 * mu2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def save_slices(volume, gt_volume, output_path, extent):
    """Save a 3×3 grid of axial/coronal/sagittal slice comparisons."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping slice visualization")
        return

    res = volume.shape[0]
    mid = res // 2
    axes = ["X", "Y", "Z"]
    indices = [mid, mid, mid]

    fig, axes_arr = plt.subplots(
        3 if gt_volume is not None else 1, 3,
        figsize=(12, 4 * (3 if gt_volume is not None else 1)),
    )
    if gt_volume is not None and axes_arr.ndim == 1:
        axes_arr = axes_arr[None, :]

    vmax = float(np.percentile(volume, 99))
    vmin = 0.0

    slices_pred = [
        volume[mid, :, :],
        volume[:, mid, :],
        volume[:, :, mid],
    ]
    titles = ["Axial (X)", "Coronal (Y)", "Sagittal (Z)"]

    if gt_volume is not None:
        gt_res = gt_volume.shape[0]
        gt_mid = gt_res // 2
        # Resize GT to match prediction resolution via nearest-neighbor
        scale = res / gt_res
        gt_slices = [
            gt_volume[gt_mid, :, :],
            gt_volume[:, gt_mid, :],
            gt_volume[:, :, gt_mid],
        ]
        gt_vmax = float(np.percentile(gt_volume, 99))

        for col, (pred_sl, gt_sl, title) in enumerate(zip(slices_pred, gt_slices, titles)):
            axes_arr[0, col].imshow(pred_sl.T, origin="lower", cmap="gray",
                                    vmin=vmin, vmax=vmax)
            axes_arr[0, col].set_title(f"Pred {title}")
            axes_arr[0, col].axis("off")

            axes_arr[1, col].imshow(gt_sl.T, origin="lower", cmap="gray",
                                    vmin=0, vmax=gt_vmax)
            axes_arr[1, col].set_title(f"GT {title}")
            axes_arr[1, col].axis("off")

            diff = pred_sl - gt_sl if pred_sl.shape == gt_sl.shape else pred_sl - pred_sl
            axes_arr[2, col].imshow(diff.T, origin="lower", cmap="bwr",
                                    vmin=-vmax * 0.5, vmax=vmax * 0.5)
            axes_arr[2, col].set_title(f"Diff {title}")
            axes_arr[2, col].axis("off")
    else:
        for col, (pred_sl, title) in enumerate(zip(slices_pred, titles)):
            axes_arr[col].imshow(pred_sl.T, origin="lower", cmap="gray",
                                 vmin=vmin, vmax=vmax)
            axes_arr[col].set_title(title)
            axes_arr[col].axis("off")

    plt.tight_layout()
    slice_path = output_path.replace(".npy", "_slices.png")
    if not slice_path.endswith(".png"):
        slice_path += "_slices.png"
    plt.savefig(slice_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved slices to {slice_path}")


def voxelize(model_path, resolution, output_path, extent=1.0, blur_sigma=0.0,
             supersample=3, sigma=0.7, sigma_v=0.35, per_cell_sigma=True,
             gt_path=None, hop=1):
    device = torch.device("cuda")

    scene_data = torch.load(model_path)
    points = scene_data["xyz"].to(device)
    density_flat = scene_data["density"].to(device).squeeze(-1)
    adjacency = scene_data["adjacency"].to(device).to(torch.int32)
    adjacency_offsets = scene_data["adjacency_offsets"].to(device).to(torch.int32)
    aabb_tree = radfoam.build_aabb_tree(points)
    _, cell_radius = radfoam.farthest_neighbor(points, adjacency, adjacency_offsets)

    activated = F.softplus(density_flat, beta=10)
    adj_off = adjacency_offsets.long()
    global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())

    grid_min = torch.tensor([-extent, -extent, -extent], device=device)
    grid_max = torch.tensor([extent, extent, extent], device=device)

    coords = torch.linspace(0, 1, resolution, device=device)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    voxel_centers = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    voxel_centers = grid_min + voxel_centers * (grid_max - grid_min)

    def _idw(query_pts):
        res = idw_query(
            query_pts, points, adjacency, adjacency_offsets, aabb_tree,
            activated, sigma, sigma_v,
            global_max_k=global_max_k,
            per_cell_sigma=per_cell_sigma,
            cell_radius=cell_radius,
            hop=hop,
        )
        return torch.nan_to_num(res.idw_result)

    k = supersample
    num_voxels = voxel_centers.shape[0]
    volume = torch.zeros(num_voxels, device=device)

    if k <= 1:
        batch_size = 2_000_000
        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            volume[start:end] = _idw(voxel_centers[start:end])
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
            volume[start:end] = _idw(sub_points).reshape(-1, samples_per_voxel).mean(dim=1)

    volume = volume.reshape(resolution, resolution, resolution)
    if blur_sigma > 0:
        volume = gaussian_blur_3d(volume, kernel_size=3, sigma=blur_sigma)
    volume_np = volume.cpu().numpy()

    np.save(output_path, volume_np)
    print(f"Saved volume {volume_np.shape} to {output_path}")
    print(f"  min={volume_np.min():.4f}, max={volume_np.max():.4f}, "
          f"mean={volume_np.mean():.4f}")

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
        img = nib.Nifti1Image(volume_np, affine)
        nib.save(img, nifti_path)
        print(f"Saved NIfTI to {nifti_path}")
    except ImportError:
        pass

    return volume_np


def main():
    parser = argparse.ArgumentParser(description="Voxelize a CT reconstruction")
    parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Grid resolution per axis")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npy path (default: volume.npy next to model)")
    parser.add_argument("--extent", type=float, default=1.0,
                        help="Half-extent of the grid")
    parser.add_argument("--blur_sigma", type=float, default=0.0,
                        help="Gaussian blur sigma (0 = disabled)")
    parser.add_argument("--supersample", type=int, default=3,
                        help="Sub-samples per axis per voxel (k^3 total, 1 = center-only)")
    parser.add_argument("--sigma", type=float, default=0.7,
                        help="IDW spatial sigma scale")
    parser.add_argument("--sigma_v", type=float, default=0.35,
                        help="IDW bilateral value sigma")
    parser.add_argument("--per_cell_sigma", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Adaptive sigma per cell (default: True)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Path to ground-truth volume (.npy) for PSNR/SSIM")
    parser.add_argument("--hop", type=int, default=1, choices=[1, 2],
                        help="IDW neighborhood depth (1=1-hop, 2=include neighbors-of-neighbors)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.model), "volume.npy")

    voxelize(args.model, args.resolution, output, args.extent, args.blur_sigma,
             args.supersample, args.sigma, args.sigma_v, args.per_cell_sigma,
             args.gt, args.hop)


if __name__ == "__main__":
    main()
