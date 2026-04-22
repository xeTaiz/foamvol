"""Standalone script: plot cells-per-voxel vs. GT gradient magnitude.

Usage:
    micromamba run -n radfoam python analysis/cell_vs_gradient.py \
        --config output/<run>/config.yaml [--out <dir>]

Produces:
    <out>/cells_vs_gradient.png   — figure
    <out>/cells_vs_gradient.csv   — binned data (bin_center, mean, ci_95, n_voxels)
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch

# Repo root on path so vis_foam / radfoam_model imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vis_foam import load_density_field, load_gt_volume, visualize_cells_vs_gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to output/<run>/config.yaml")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same dir as config)")
    parser.add_argument("--n-bins", type=int, default=32,
                        help="Number of gradient-magnitude bins (default: 32)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(config_path)
    out_dir = os.path.abspath(args.out) if args.out else run_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data_path", "")
    dataset   = cfg.get("dataset", "r2_gaussian")

    # Load GT volume
    gt_volume = load_gt_volume(data_path, dataset)
    if gt_volume is None:
        print("No GT volume found — cannot compute gradient. Exiting.")
        sys.exit(1)
    print(f"GT volume: {gt_volume.shape}, range [{gt_volume.min():.4f}, {gt_volume.max():.4f}]")

    # Load model checkpoint
    model_pt = os.path.join(run_dir, "model.pt")
    if not os.path.exists(model_pt):
        print(f"model.pt not found at {model_pt}. Exiting.")
        sys.exit(1)
    print(f"Loading checkpoint: {model_pt}")
    field = load_density_field(model_pt)
    points = field["points"]  # (N, 3) CUDA float32
    print(f"Points: {points.shape[0]:,}")

    # Generate figure + binned stats
    fig, stats = visualize_cells_vs_gradient(
        points, gt_volume, n_bins=args.n_bins
    )
    C = stats["count_res"]
    print(f"Count grid: {C}³   mean cells/voxel: {points.shape[0] / C**3:.2f}")
    print(f"Spearman ρ = {stats['spearman_rho']:.4f}  (p = {stats['spearman_pval']:.2e})")

    import matplotlib.pyplot as plt
    png_path = os.path.join(out_dir, "cells_vs_gradient.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = os.path.join(out_dir, "cells_vs_gradient.csv")
    with open(csv_path, "w") as f:
        f.write("bin_center_log1p,mean_cells_per_voxel,ci_95\n")
        for bc, m, ci in zip(stats["bin_centers"], stats["bin_means"], stats["bin_cis"]):
            f.write(f"{bc:.6f},{m:.6f},{ci:.6f}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
