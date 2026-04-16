#!/usr/bin/env python3
"""Quantify the softplus floor bias on air voxels.

GT clamps HU=-1000 to 0 (hard zero), but our density = softplus(raw) with
beta=10 means raw values that stagnate around -1 to -3 give nonzero
activations (~1e-5 to ~1e-10). This may leave a systematic PSNR gap.

For each provided run:
  1. Voxelize to vol_gt resolution (IDW + raw).
  2. Mask voxels where gt < air_thresh.
  3. Report min/mean/max/percentiles of predicted density in those voxels.
  4. Report PSNR with and without air voxels (so we see the gap's magnitude).
  5. Same for R2's vol_r2.npy as a reference.

Usage:
    python check_air_floor.py --run output/sweep25_grid/D04-n1h-1e2-si50_chest
    python check_air_floor.py --run <dir> --thresh 1e-3 --gt-name vol_gt.npy
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_foam import load_density_field, voxelize_volumes


def psnr(pred, gt, data_range=None):
    dr = data_range if data_range is not None else float(gt.max() - gt.min())
    mse = float(((pred - gt) ** 2).mean())
    if mse <= 0:
        return float("inf")
    return 20 * np.log10(dr) - 10 * np.log10(mse)


def stats(arr, name):
    a = np.asarray(arr).ravel()
    if a.size == 0:
        print(f"  {name}: empty")
        return
    q = np.quantile(a, [0.01, 0.5, 0.99])
    print(f"  {name:18s}  N={a.size:>9d}  "
          f"min={a.min():.3e}  p01={q[0]:.3e}  med={q[1]:.3e}  "
          f"p99={q[2]:.3e}  max={a.max():.3e}  mean={a.mean():.3e}")


def analyze(run_dir, thresh, gt_name, data_path_override=None):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(run_dir, "sweep_config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_path = data_path_override or cfg["data_path"]
    gt_path = os.path.join(data_path, gt_name)
    gt = np.load(gt_path).astype(np.float32)  # (D,H,W) stored as (X,Y,Z)
    print(f"\n=== {run_dir} ===")
    print(f"GT from: {gt_path}  shape={gt.shape}  min={gt.min():.3e}  max={gt.max():.3e}")

    model_path = os.path.join(run_dir, "model.pt")
    field = load_density_field(model_path)

    # Replicate the interp sigma used in final eval
    sigma = cfg["interp_sigma_scale"] * float(field["cell_radius"].median().item())
    sigma_v = cfg["interp_sigma_v"]

    raw_vol, idw_vol = voxelize_volumes(
        field, resolution=gt.shape[0], extent=1.0,
        sigma=sigma, sigma_v=sigma_v,
    )

    # Air mask (GT-defined)
    air = gt < thresh
    solid = ~air
    print(f"\nAir voxels (GT < {thresh}):  {air.sum():,} / {gt.size:,}  "
          f"({100*air.mean():.1f}%)")

    print("\n-- Predicted density stats on AIR voxels --")
    stats(gt[air],      "GT       (air)")
    stats(raw_vol[air], "Pred raw (air)")
    stats(idw_vol[air], "Pred IDW (air)")

    print("\n-- On SOLID voxels (sanity) --")
    stats(gt[solid],      "GT       (solid)")
    stats(idw_vol[solid], "Pred IDW (solid)")

    # Global and split PSNR
    dr = float(gt.max() - gt.min())
    psnr_all = psnr(idw_vol, gt, dr)
    psnr_solid = psnr(idw_vol[solid], gt[solid], dr)
    # Air-only PSNR uses GT range of the full volume (compare on same scale)
    air_mse = float(((idw_vol[air] - gt[air]) ** 2).mean())
    psnr_air = 20 * np.log10(dr) - 10 * np.log10(air_mse) if air_mse > 0 else float("inf")

    # How much does air drag down global PSNR? Substitute air -> 0 and recompute.
    idw_zeroed = idw_vol.copy()
    idw_zeroed[air] = 0.0
    psnr_zeroed = psnr(idw_zeroed, gt, dr)

    print(f"\n-- PSNR decomposition (range={dr:.4f}) --")
    print(f"  IDW PSNR full volume:            {psnr_all:.3f} dB")
    print(f"  IDW PSNR solid only:             {psnr_solid:.3f} dB")
    print(f"  IDW PSNR air only:               {psnr_air:.3f} dB")
    print(f"  If we forced air -> 0:           {psnr_zeroed:.3f} dB  "
          f"(gain vs full: +{psnr_zeroed - psnr_all:.3f})")

    # R2 reference, if present
    r2_path = os.path.join(data_path, "vol_r2.npy")
    if os.path.exists(r2_path):
        r2 = np.load(r2_path).astype(np.float32)
        print(f"\n-- R2 reference at {r2_path} --")
        stats(r2[air],   "R2       (air)")
        stats(r2[solid], "R2       (solid)")
        r2_all = psnr(r2, gt, dr)
        r2_solid = psnr(r2[solid], gt[solid], dr)
        r2_air_mse = float(((r2[air] - gt[air]) ** 2).mean())
        r2_air_psnr = 20 * np.log10(dr) - 10 * np.log10(r2_air_mse) if r2_air_mse > 0 else float("inf")
        r2_zeroed = r2.copy()
        r2_zeroed[air] = 0.0
        r2_all_zeroed = psnr(r2_zeroed, gt, dr)
        print(f"  R2 PSNR full:                    {r2_all:.3f} dB")
        print(f"  R2 PSNR solid only:              {r2_solid:.3f} dB")
        print(f"  R2 PSNR air only:                {r2_air_psnr:.3f} dB")
        print(f"  R2 if we forced air -> 0:        {r2_all_zeroed:.3f} dB  "
              f"(gain: +{r2_all_zeroed - r2_all:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, nargs="+",
                   help="One or more run directories (containing config.yaml and model.pt)")
    p.add_argument("--thresh", type=float, default=1e-3, help="Air threshold on GT")
    p.add_argument("--gt-name", default="vol_gt.npy")
    p.add_argument("--data-path", default=None,
                   help="Override data_path from config (useful if trained on a different machine)")
    args = p.parse_args()

    for run in args.run:
        analyze(run, args.thresh, args.gt_name, args.data_path)


if __name__ == "__main__":
    main()
