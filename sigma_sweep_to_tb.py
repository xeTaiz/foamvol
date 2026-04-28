#!/usr/bin/env python3
"""Score sigma_sweep.csv and write best-sigma scalars + heatmaps to a TensorBoard log dir.

For each (sigma_s, sigma_v) grid cell, computes relative improvement vs raw baseline:

    rel_imp(m) = sign(m) * (idw_metric - raw_metric) / |raw_metric + eps|

Three composite scores (mean over subsets):
    score_volume  — PSNR + SSIM
    score_surface — vol CD + vol H95 + vol F1@1
    score_combined — all 5

Winner = argmax(score_combined).

Logged to --tb-dir (appended to the run's existing event files):
    best/score_combined, best/score_volume, best/score_surface
    best/sigma_s, best/sigma_v
    best/vol_idw_psnr, best/vol_idw_ssim, best/vol_idw_f1_1v,
    best/vol_idw_hausdorff_95, best/vol_idw_chamfer
    best/heatmap_volume, best/heatmap_surface, best/heatmap_combined

Usage:
    python sigma_sweep_to_tb.py --csv output/sweep37/chest-x/sigma_sweep.csv \\
                                 --tb-dir output/sweep37/chest-x
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


EPS = 1e-8

# (key, idw_col, raw_col, sign: +1=higher-better, -1=lower-better)
METRICS = [
    ("psnr", "vol_idw_psnr",         "vol_raw_psnr",         +1),
    ("ssim", "vol_idw_ssim",         "vol_raw_ssim",         +1),
    ("cd",   "vol_idw_chamfer",      "vol_raw_chamfer",      -1),
    ("h95",  "vol_idw_hausdorff_95", "vol_raw_hausdorff_95", -1),
    ("f1",   "vol_idw_f1_1v",        "vol_raw_f1_1v",        +1),
]
VOL_KEYS  = ["psnr", "ssim"]
SURF_KEYS = ["cd", "h95", "f1"]
COMB_KEYS = VOL_KEYS + SURF_KEYS


def load_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def add_scores(rows):
    for row in rows:
        for key, col_idw, col_raw, sign in METRICS:
            baseline = abs(row[col_raw]) + EPS
            row[f"rel_{key}"] = sign * (row[col_idw] - row[col_raw]) / baseline
        row["score_volume"]   = float(np.mean([row[f"rel_{k}"] for k in VOL_KEYS]))
        row["score_surface"]  = float(np.mean([row[f"rel_{k}"] for k in SURF_KEYS]))
        row["score_combined"] = float(np.mean([row[f"rel_{k}"] for k in COMB_KEYS]))
    return rows


def make_heatmap(rows, score_key, title, sigma_s_vals, sigma_v_vals):
    grid = np.full((len(sigma_s_vals), len(sigma_v_vals)), np.nan)
    score_map = {(r["sigma_s"], r["sigma_v"]): r[score_key] for r in rows}
    for i, ss in enumerate(sigma_s_vals):
        for j, sv in enumerate(sigma_v_vals):
            v = score_map.get((ss, sv))
            if v is not None:
                grid[i, j] = v

    best_i, best_j = np.unravel_index(np.nanargmax(grid), grid.shape)
    vabs = max(np.nanmax(np.abs(grid)), 1e-6)

    fig, ax = plt.subplots(figsize=(max(4, len(sigma_v_vals) * 1.4 + 1),
                                     max(3, len(sigma_s_vals) * 1.1 + 1)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, label="mean relative improvement")

    ax.set_xticks(range(len(sigma_v_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in sigma_v_vals])
    ax.set_yticks(range(len(sigma_s_vals)))
    ax.set_yticklabels([f"{s:.3f}" for s in sigma_s_vals])
    ax.set_xlabel("σ_v")
    ax.set_ylabel("σ_s")
    ax.set_title(title)

    for i in range(len(sigma_s_vals)):
        for j in range(len(sigma_v_vals)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black",
                        fontweight="bold" if (i == best_i and j == best_j) else "normal")

    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                linewidth=3, edgecolor="blue", facecolor="none"))
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to sigma_sweep.csv")
    parser.add_argument("--tb-dir", required=True, help="Tensorboard log directory (run output dir)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ABORT] CSV not found: {args.csv}")
        sys.exit(1)

    rows = load_csv(args.csv)
    if not rows:
        print("[ABORT] CSV is empty")
        sys.exit(1)

    rows = add_scores(rows)

    sigma_s_vals = sorted(set(r["sigma_s"] for r in rows))
    sigma_v_vals = sorted(set(r["sigma_v"] for r in rows))
    best = max(rows, key=lambda r: r["score_combined"])

    print(f"Grid: σ_s={sigma_s_vals}  σ_v={sigma_v_vals}  ({len(rows)} combos)")
    print(f"Winner: σ_s={best['sigma_s']:.3f}  σ_v={best['sigma_v']:.2f}"
          f"  combined={best['score_combined']:.4f}"
          f"  vol={best['score_volume']:.4f}  surf={best['score_surface']:.4f}")
    print(f"  PSNR={best['vol_idw_psnr']:.4f}  SSIM={best['vol_idw_ssim']:.6f}"
          f"  F1={best['vol_idw_f1_1v']:.4f}"
          f"  H95={best['vol_idw_hausdorff_95']:.4f}  CD={best['vol_idw_chamfer']:.4f}")

    writer = SummaryWriter(args.tb_dir)

    writer.add_scalar("best/score_combined",        best["score_combined"],        0)
    writer.add_scalar("best/score_volume",          best["score_volume"],          0)
    writer.add_scalar("best/score_surface",         best["score_surface"],         0)
    writer.add_scalar("best/sigma_s",               best["sigma_s"],               0)
    writer.add_scalar("best/sigma_v",               best["sigma_v"],               0)
    writer.add_scalar("best/vol_idw_psnr",          best["vol_idw_psnr"],          0)
    writer.add_scalar("best/vol_idw_ssim",          best["vol_idw_ssim"],          0)
    writer.add_scalar("best/vol_idw_f1_1v",         best["vol_idw_f1_1v"],         0)
    writer.add_scalar("best/vol_idw_hausdorff_95",  best["vol_idw_hausdorff_95"],  0)
    writer.add_scalar("best/vol_idw_chamfer",       best["vol_idw_chamfer"],       0)

    for score_key, title, tag in [
        ("score_volume",   "Volume (PSNR + SSIM)",                "best/heatmap_volume"),
        ("score_surface",  "Surface (vol CD + H95 + F1)",         "best/heatmap_surface"),
        ("score_combined", "Combined (volume + surface, 5 total)", "best/heatmap_combined"),
    ]:
        fig = make_heatmap(rows, score_key, title, sigma_s_vals, sigma_v_vals)
        writer.add_figure(tag, fig, global_step=0)
        plt.close(fig)
        print(f"  Logged heatmap → {tag}")

    writer.close()
    print(f"Wrote best/* tags → {args.tb_dir}")


if __name__ == "__main__":
    main()
