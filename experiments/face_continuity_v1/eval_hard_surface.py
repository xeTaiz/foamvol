#!/usr/bin/env python3
"""Chamfer/Hausdorff/F1 surface metrics for a hard split-aware voxelization.

Companion to split_voxelize.py's own volume/air metrics: reuses train.py's
compute_surface_metrics (multi-threshold marching-cubes Chamfer/HD95/F1)
between the hard-eval volume (volume_hard_ss4.npy) and the dataset GT volume,
matching the ``surface_hard_ss4_metrics.json`` naming/schema used in prior
face-continuity runs.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train import compute_surface_metrics
from vis_foam import load_gt_volume


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", required=True, help="Path to volume_hard_ss4.npy")
    p.add_argument("--data_path", required=True, help="Dataset directory (for GT volume)")
    p.add_argument("--dataset", default="r2_gaussian")
    p.add_argument("--output", required=True, help="Output metrics json path")
    args = p.parse_args()

    pred = np.load(args.volume).astype(np.float32)
    gt = load_gt_volume(args.data_path, args.dataset)
    if gt is None:
        raise RuntimeError(f"no GT volume found under {args.data_path}")
    if gt.shape != pred.shape:
        raise RuntimeError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    metrics = compute_surface_metrics(pred, gt)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {args.output}: {metrics}")


if __name__ == "__main__":
    main()
