#!/usr/bin/env python3
"""Push split-voxelized slice comparison panels into existing TensorBoard runs.

The "raw" column (and the interleaved panel's primary image) is read directly
from output/<run>/volume_hard_ss4.npy -- the exact 256^3 hard-side-selected,
4x supersampled grid produced by split_voxelize.py that generated the
volume_psnr/sobel_psnr/dice/... numbers in volume_hard_ss4_metrics.json.
What's pictured here is byte-identical to what was scored, so it's the right
artifact for visually auditing reconstruction quality per run.

The "IDW" column is the real flat (non-split-aware) bilateral natural-neighbor
field via sample_idw() -- the same computation and sigma/sigma_v the live
training slices_interleaved panel uses -- NOT a duplicate of the raw column.
visualize_slices() renders a genuine "IDW" panel, GT-minus-IDW diff, and
idw_psnr/idw_ssim metrics in rows 3-5 regardless of whether R2 data is also
supplied (R2 only replaces the *top-row* raw/IDW slot), so this must be a real
independent field, not a stand-in, or those panels/metrics would silently be
fabricated duplicates of the raw column.

Writes new image tags into the SAME output/<run> TensorBoard log directory
(new event file, additive -- does not touch or remove existing runs' data),
under slices_interleaved_split/<run> and slices_sobel_split/<run>, so they
show up alongside the live panels in an already-running TensorBoard instance
pointed at --logdir=output (next reload_interval tick picks them up).

Usage:
    python experiments/face_continuity_v1/push_split_slices.py \\
        --runs output/FC64_control output/FC64_unfrozen_control ... \\
        --gt r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone/vol_gt.npy \\
        --r2 <precomputed R2-Gaussian reference volume, optional> \\
        --step 10000
"""
from __future__ import annotations

import argparse
import os
from functools import partial

import numpy as np
import yaml

from vis_foam import (
    compute_cell_density_slice,
    compute_voronoi_edges,
    load_density_field,
    make_slice_coords,
    sample_gt_slice,
    sample_idw,
    visualize_slices,
)

AXES = [0, 1, 2]
COORDS = [-0.2, 0.0, 0.2]
RESOLUTION = 256
CELL_DENSITY_RESOLUTION = 64
EXTENT = 1.0


def _config_path_for_run(run_dir: str, configs_dir: str) -> str:
    run_name = os.path.basename(os.path.normpath(run_dir))
    return os.path.join(configs_dir, f"{run_name}.yaml")


def _resolve_idw_sigma(config_path: str, cell_radius_median: float):
    """Mirror train.py's sigma resolution: interp_sigma_abs overrides scale*radius."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sigma_abs = float(cfg.get("interp_sigma_abs", 0.0))
    sigma_scale = float(cfg.get("interp_sigma_scale", 1.0))
    sigma = sigma_abs if sigma_abs > 0 else sigma_scale * cell_radius_median
    sigma_v = cfg.get("interp_sigma_v", None)
    sigma_v = float(sigma_v) if sigma_v is not None else None
    return sigma, sigma_v


def build_slices(run_dir: str, config_path: str, gt_volume: np.ndarray,
                  r2_volume: np.ndarray | None):
    field = load_density_field(os.path.join(run_dir, "model.pt"))
    if not field.get("thin_surface_active", False):
        print(f"[{run_dir}] note: checkpoint has no active thin-surface state "
              f"(baseline/non-split run) -- raw column falls back to flat "
              f"per-cell density, same as vol_raw_psnr.")

    vol_path = os.path.join(run_dir, "volume_hard_ss4.npy")
    split_vol = np.load(vol_path).astype(np.float32)
    if split_vol.shape[0] != RESOLUTION:
        raise ValueError(
            f"{vol_path}: volume is {split_vol.shape}, expected "
            f"{RESOLUTION}^3 -- re-run split_voxelize.py with --resolution "
            f"{RESOLUTION} for this run first."
        )

    sigma, sigma_v = _resolve_idw_sigma(
        config_path, float(field["cell_radius"].median().item()))

    d_slices, idw_slices, cd_slices = [], [], []
    gt_slices, r2_slices, ve_slices = [], [], []
    for axis in AXES:
        for coord in COORDS:
            d_slices.append(sample_gt_slice(split_vol, axis, coord, RESOLUTION, EXTENT))
            coords_2d = make_slice_coords(axis, coord, RESOLUTION, EXTENT)
            idw_slices.append(sample_idw(field, coords_2d, sigma=sigma, sigma_v=sigma_v))
            cd_slices.append(compute_cell_density_slice(
                field["points"], axis, coord, CELL_DENSITY_RESOLUTION, EXTENT))
            gt_slices.append(sample_gt_slice(gt_volume, axis, coord, RESOLUTION, EXTENT))
            r2_slices.append(
                sample_gt_slice(r2_volume, axis, coord, RESOLUTION, EXTENT)
                if r2_volume is not None else None
            )
            ve_slices.append(compute_voronoi_edges(field, axis, coord, RESOLUTION, EXTENT))
    return d_slices, idw_slices, cd_slices, gt_slices, r2_slices, ve_slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                         help="output/<run> directories (need model.pt + volume_hard_ss4.npy)")
    parser.add_argument("--configs-dir", default="experiments/face_continuity_v1/configs",
                         help="directory holding <run_name>.yaml for each run "
                              "(used only to resolve the real IDW sigma/sigma_v)")
    parser.add_argument("--gt", required=True, help="path to vol_gt.npy")
    parser.add_argument("--r2", default=None, help="optional precomputed R2-Gaussian reference volume")
    parser.add_argument("--step", type=int, default=10000, help="TensorBoard global_step to log at")
    parser.add_argument("--tag-prefix", default="slices_interleaved_split")
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    gt_volume = np.load(args.gt).astype(np.float32)
    r2_volume = np.load(args.r2).astype(np.float32) if args.r2 else None

    for run_dir in args.runs:
        run_name = os.path.basename(os.path.normpath(run_dir))
        config_path = _config_path_for_run(run_dir, args.configs_dir)
        print(f"[{run_name}] loading checkpoint + volume_hard_ss4.npy "
              f"(config={config_path}) ...")
        d_slices, idw_slices, cd_slices, gt_slices, r2_slices, ve_slices = build_slices(
            run_dir, config_path, gt_volume, r2_volume)

        writer = SummaryWriter(log_dir=run_dir)
        log_fig_il = partial(writer.add_figure, f"{args.tag_prefix}/{run_name}",
                              global_step=args.step)
        log_fig_sobel = partial(writer.add_figure, f"slices_sobel_split/{run_name}",
                                 global_step=args.step)
        try:
            metrics = visualize_slices(
                d_slices, idw_slices, cd_slices,
                gt_slices=gt_slices,
                r2_slices=r2_slices,
                writer_fn_interleaved=log_fig_il,
                writer_fn_sobel=log_fig_sobel,
                voronoi_edges=ve_slices,
            )
            if metrics:
                for key, val in metrics.items():
                    try:
                        writer.add_scalar(f"slice_split/{key}", float(val), args.step)
                    except (TypeError, ValueError):
                        pass
        finally:
            writer.close()
        print(f"[{run_name}] pushed {args.tag_prefix}/{run_name} "
              f"and slices_sobel_split/{run_name} @ step {args.step}")


if __name__ == "__main__":
    main()
