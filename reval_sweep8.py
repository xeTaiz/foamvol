"""Re-evaluate all sweep8 runs with updated IDW (weight floor).

Loads each saved checkpoint, recomputes slice visualizations and volume
metrics, and logs them to the run's existing TensorBoard at step 10001
so they can be compared side-by-side with the original step-10000 results.
"""

import os
import glob
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.tensorboard import SummaryWriter

import radfoam
from vis_foam import (load_density_field, query_density, sample_idw,
                      sample_idw_diagnostic, visualize_idw_diagnostics,
                      supersample_slice, make_slice_coords,
                      compute_cell_density_slice, visualize_slices,
                      load_gt_volume, sample_gt_slice, voxelize_volumes)

# All runs use the same dataset; the GT volume lives at this absolute path
GT_DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"


def compute_volume_psnr(pred, gt):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    pred, gt = pred.float(), gt.float()
    pixel_max = gt.max()
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(pixel_max ** 2 / mse)).item()


@torch.no_grad()
def compute_volume_ssim(pred, gt, window_size=11):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    pred, gt = pred.float(), gt.float()

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
    if pred.is_cuda:
        kernel = kernel.cuda()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    axis_ssims = []
    for axis in range(3):
        n_slices = pred.shape[axis]
        ssim_sum = 0.0
        count = 0
        for i in range(n_slices):
            if axis == 0:
                s_pred, s_gt = pred[i, :, :], gt[i, :, :]
            elif axis == 1:
                s_pred, s_gt = pred[:, i, :], gt[:, i, :]
            else:
                s_pred, s_gt = pred[:, :, i], gt[:, :, i]
            if s_gt.max() <= 0:
                continue
            img1 = s_pred.unsqueeze(0).unsqueeze(0)
            img2 = s_gt.unsqueeze(0).unsqueeze(0)
            mu1 = F.conv2d(img1, kernel, padding=pad)
            mu2 = F.conv2d(img2, kernel, padding=pad)
            sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=pad) - mu1 ** 2
            sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=pad) - mu2 ** 2
            sigma12 = F.conv2d(img1 * img2, kernel, padding=pad) - mu1 * mu2
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
            )
            ssim_sum += ssim_map.mean().item()
            count += 1
        axis_ssims.append(ssim_sum / count if count > 0 else 0.0)

    return float(np.mean(axis_ssims)), axis_ssims


def eval_run(out_dir, gt_volume, step=10001):
    """Re-evaluate a single run and log to its TensorBoard."""
    run_name = os.path.relpath(out_dir, "output")  # e.g. "sweep8/P2-combined-128k"
    model_path = os.path.join(out_dir, "model.pt")
    config_path = os.path.join(out_dir, "config.yaml")

    if not os.path.exists(model_path):
        print(f"  SKIP (no model.pt)")
        return
    if not os.path.exists(config_path):
        print(f"  SKIP (no config.yaml)")
        return

    # Read per-run config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    interp_sigma_scale = cfg.get("interp_sigma_scale", 0.55)
    interp_sigma_v = cfg.get("interp_sigma_v", 0.2)

    print(f"  Loading model...")
    field = load_density_field(model_path)

    # Compute interp_sigma from cell radius
    _, cell_radius = radfoam.farthest_neighbor(
        field["points"],
        field["adjacency"].int(),
        field["adjacency_offsets"].int(),
    )
    interp_sigma = interp_sigma_scale * cell_radius.median().item()
    print(f"  sigma={interp_sigma:.6f}, sigma_v={interp_sigma_v}")

    writer = SummaryWriter(out_dir)

    # --- Slice visualization ---
    print(f"  Slices (2x SS)...")
    axes = [0, 1, 2]
    slice_coords = [-0.2, 0.0, 0.2]
    density_slices, idw_slices, cell_density_slices, gt_slices_final = [], [], [], []
    for a in axes:
        for c in slice_coords:
            density_slices.append(supersample_slice(query_density, field, a, c, 256, 1.0, ss=2))
            idw_slices.append(supersample_slice(sample_idw, field, a, c, 256, 1.0, ss=2,
                                                sigma=interp_sigma, sigma_v=interp_sigma_v))
            cell_density_slices.append(
                compute_cell_density_slice(field["points"], a, c, 64, 1.0))
            gt_slices_final.append(sample_gt_slice(gt_volume, a, c, 256, 1.0))

    log_fig = partial(writer.add_figure, f"slices/{run_name}", global_step=step)
    log_fig_il = partial(writer.add_figure, f"slices_interleaved/{run_name}", global_step=step)
    slice_metrics = visualize_slices(
        density_slices, idw_slices, cell_density_slices,
        gt_slices=gt_slices_final, writer_fn=log_fig,
        writer_fn_interleaved=log_fig_il,
    )
    if slice_metrics is not None:
        for key, val in slice_metrics.items():
            tag = f"slice_{key.split('_')[1]}/{key.split('_')[0]}"
            writer.add_scalar(tag, val, step)
        print(f"  Slice IDW PSNR: {slice_metrics['idw_psnr']:.4f}")

    # --- IDW diagnostics ---
    print(f"  IDW diagnostics...")
    diag_coords = make_slice_coords(axis=2, coord=0.0, resolution=256, extent=1.0)
    diag = sample_idw_diagnostic(field, diag_coords,
                                 sigma=interp_sigma, sigma_v=interp_sigma_v)
    diag_writer = partial(writer.add_figure, f"idw_diagnostics/{run_name}", global_step=step)
    visualize_idw_diagnostics(diag, writer_fn=diag_writer)

    # --- 3D volume metrics ---
    if gt_volume is not None:
        vol_res = gt_volume.shape[0]
        print(f"  Voxelizing {vol_res}³...")
        raw_vol, idw_vol = voxelize_volumes(
            field, resolution=vol_res, extent=1.0,
            sigma=interp_sigma, sigma_v=interp_sigma_v,
        )

        vol_gt_t = torch.from_numpy(gt_volume).float().cuda()
        raw_vol_t = torch.from_numpy(raw_vol).float().cuda()
        idw_vol_t = torch.from_numpy(idw_vol).float().cuda()

        raw_psnr_3d = compute_volume_psnr(raw_vol_t, vol_gt_t)
        raw_ssim_3d, raw_ssim_ax = compute_volume_ssim(raw_vol_t, vol_gt_t)
        idw_psnr_3d = compute_volume_psnr(idw_vol_t, vol_gt_t)
        idw_ssim_3d, idw_ssim_ax = compute_volume_ssim(idw_vol_t, vol_gt_t)

        print(f"  Vol Raw  PSNR: {raw_psnr_3d:.4f}, SSIM: {raw_ssim_3d:.6f}")
        print(f"  Vol IDW  PSNR: {idw_psnr_3d:.4f}, SSIM: {idw_ssim_3d:.6f}")

        writer.add_scalar("test/vol_raw_psnr", raw_psnr_3d, step)
        writer.add_scalar("test/vol_raw_ssim", raw_ssim_3d, step)
        writer.add_scalar("test/vol_idw_psnr", idw_psnr_3d, step)
        writer.add_scalar("test/vol_idw_ssim", idw_ssim_3d, step)
        for ax_i, ax_name in enumerate(["x", "y", "z"]):
            writer.add_scalar(f"test/vol_raw_ssim_{ax_name}", raw_ssim_ax[ax_i], step)
            writer.add_scalar(f"test/vol_idw_ssim_{ax_name}", idw_ssim_ax[ax_i], step)

    writer.close()
    # Free GPU memory between runs
    del field
    torch.cuda.empty_cache()


def main():
    sweep_dir = "output/sweep8"

    # Load GT volume once (shared across all runs)
    print("Loading GT volume...")
    gt_volume = load_gt_volume(GT_DATA_PATH, "r2_gaussian")
    print(f"GT volume: shape={gt_volume.shape}" if gt_volume is not None else "No GT volume")

    # Discover all runs with model.pt
    run_dirs = sorted(glob.glob(os.path.join(sweep_dir, "*")))
    run_dirs = [d for d in run_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "model.pt"))]

    print(f"\nFound {len(run_dirs)} runs with checkpoints.\n")

    for i, out_dir in enumerate(run_dirs):
        name = os.path.basename(out_dir)
        print(f"[{i+1}/{len(run_dirs)}] {name}")
        try:
            eval_run(out_dir, gt_volume)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("All done!")


if __name__ == "__main__":
    main()
