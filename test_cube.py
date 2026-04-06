#!/usr/bin/env python3
"""Sanity check: can the Voronoi representation perfectly reconstruct simple cube scenes?

Tests:
  1a. Single cube, 7 manually-placed points (no densification) — proves representability
  1b. Single cube, random init + densification — proves optimizer convergence
  2a. 2×2×2 cubes, manually-placed points (no densification)
  2b. 2×2×2 cubes, random init + densification

Usage:
    python test_cube.py              # run all 4 tests
    python test_cube.py --test 1a    # run specific test
    python test_cube.py --list       # show test names
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import yaml


OUT_DIR = "output/cube_sanity"


# ---------------------------------------------------------------------------
# Manual point placement for cube scenes
# ---------------------------------------------------------------------------

MIN_POINTS = 32  # Delaunay triangulation minimum


def pad_to_min(pts, scale=1.05):
    """Pad point set to minimum triangulation size with far-away filler points."""
    if pts.shape[0] >= MIN_POINTS:
        return pts
    n_pad = MIN_POINTS - pts.shape[0]
    # Place filler points on a shell at scale (outside the scene)
    filler = torch.randn(n_pad, 3) * 0.1
    filler = filler / filler.norm(dim=-1, keepdim=True) * scale
    # Add small jitter to avoid degeneracies
    filler += torch.randn_like(filler) * 1e-4
    return torch.cat([pts, filler], dim=0)


def single_cube_points():
    """7 points that represent a single cube (side 0.5, centered at origin).

    Voronoi cell interfaces land at ±0.25 when the center point is at origin
    and surrounding points are at ±0.5 on each axis.
    The interface between two points is the perpendicular bisector plane,
    which for points at 0 and ±0.5 is at ±0.25.
    """
    jitter = 1e-4
    pts = torch.tensor([
        [0.0, 0.0, 0.0],       # center (high density)
        [0.5, 0.0, 0.0],       # +x
        [-0.5, 0.0, 0.0],      # -x
        [0.0, 0.5, 0.0],       # +y
        [0.0, -0.5, 0.0],      # -y
        [0.0, 0.0, 0.5],       # +z
        [0.0, 0.0, -0.5],      # -z
    ], dtype=torch.float32)
    pts += torch.randn_like(pts) * jitter
    return pad_to_min(pts)


def cube_2x2x2_points():
    """Points for 2×2×2 cube scene: 8 centers + 24 face points = 32.

    Each sub-cube has side = block_half = 0.25, so half-side = 0.125.
    Sub-cube centers are at (±0.0625, ±0.0625, ±0.0625).

    For the outer boundary at x=+0.25: a face point at x_outer such that
    midpoint(center_x, x_outer) = 0.25. For center_x = 0.0625:
      x_outer = 2*0.25 - 0.0625 = 0.4375

    Each of the 6 faces of the block has 4 sub-faces (one per adjacent sub-cube),
    so we need 4 points per face = 24 face points total.
    """
    jitter = 1e-4
    block_half = 0.25
    ch = block_half / 2  # 0.125 — center half-offset

    # 8 centers of sub-cubes (±ch, ±ch, ±ch)
    centers = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                centers.append([sx * ch, sy * ch, sz * ch])

    # 24 face points: for each of 6 faces, 4 points mirroring the 4 sub-cubes
    # that touch that face.
    # Face normal +x: sub-cubes with center_x = +ch need mirror at x = 2*block_half - ch
    outer = 2 * block_half - ch  # 0.4375
    face_points = []
    for sign in [-1, 1]:
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                # +x/-x face
                face_points.append([sign * outer, sa * ch, sb * ch])
                # +y/-y face
                face_points.append([sa * ch, sign * outer, sb * ch])
                # +z/-z face
                face_points.append([sa * ch, sb * ch, sign * outer])

    pts = torch.tensor(centers + face_points, dtype=torch.float32)
    pts += torch.randn_like(pts) * jitter
    return pad_to_min(pts)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def base_config(scene_type, out_name, init_points, final_points, iterations,
                densify=False):
    """Build a config dict for a cube test run."""
    cfg = {
        "dataset": "ct_cube",
        "data_path": scene_type,
        "num_angles": 180,
        "detector_size": 128,
        "iterations": iterations,
        "rays_per_batch": 500000,
        "init_points": init_points,
        "final_points": final_points,
        "activation_scale": 1.0,
        "init_scale": 1.05,
        "init_type": "random",
        "init_density": 0.0,
        "loss_type": "l1",
        "debug": False,
        "viewer": False,
        "save_volume": False,
        # Logging — less frequent for simple test cases
        "log_percent": 50,
        "diag_percent": 50,
        # Optimization
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "density_grad_clip": 10.0,
        # TV OFF
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        # Interpolation OFF
        "interpolation_start": -1,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": True,
        "per_neighbor_sigma": True,
        # BF OFF
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        # Gaussians OFF
        "gaussian_start": -1,
        "freeze_base_at_gaussian": False,
        "joint_finetune_start": -1,
        "peak_lr_init": 1e-2,
        "peak_lr_final": 1e-3,
        "offset_lr_init": 1e-3,
        "offset_lr_final": 1e-4,
        "cov_lr_init": 1e-2,
        "cov_lr_final": 1e-3,
        # Linear gradient OFF
        "gradient_start": -1,
        "gradient_lr_init": 1e-2,
        "gradient_lr_final": 1e-3,
        "gradient_warmup": 500,
        "gradient_max_slope": 5.0,
        "gradient_freeze_points": 500,
        # Pruning OFF
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.0,
        # Targeted sampling OFF
        "targeted_fraction": 0.0,
        "targeted_start": -1,
        # Contrast OFF
        "contrast_alpha": 0.0,
    }

    if densify:
        cfg["densify_from"] = 500
        cfg["densify_until"] = int(iterations * 0.6)
        cfg["densify_factor"] = 1.15
        cfg["gradient_fraction"] = 1.0
        cfg["idw_fraction"] = 0.0
        cfg["entropy_fraction"] = 0.0
        cfg["freeze_points"] = int(iterations * 0.95)
    else:
        # No densification — densify range set so warmup completes quickly
        # but no points are actually added (densify_until <= densify_from)
        cfg["densify_from"] = 100
        cfg["densify_until"] = 100
        cfg["densify_factor"] = 1.0
        cfg["gradient_fraction"] = 1.0
        cfg["idw_fraction"] = 0.0
        cfg["entropy_fraction"] = 0.0
        cfg["freeze_points"] = int(iterations * 0.95)

    return cfg


def save_init_points(pts, path):
    """Save initial point positions for manual override."""
    torch.save(pts, path)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS = {
    "1a": {
        "name": "single_cube_manual",
        "desc": "Single cube, 7+padding manually-placed points, no densification",
        "scene": "cube_single",
        "manual_points": single_cube_points,
        "init_points": MIN_POINTS,  # overridden by init_points_file
        "final_points": MIN_POINTS,
        "iterations": 10000,
        "densify": False,
    },
    "1b": {
        "name": "single_cube_random",
        "desc": "Single cube, random init → 512 cells, with densification",
        "scene": "cube_single",
        "manual_points": None,
        "init_points": 64,
        "final_points": 512,
        "iterations": 10000,
        "densify": True,
    },
    "2a": {
        "name": "cube_2x2x2_manual",
        "desc": "2×2×2 cubes, 32 manually-placed points, no densification",
        "scene": "cube_2x2x2",
        "manual_points": cube_2x2x2_points,
        "init_points": MIN_POINTS,  # overridden by init_points_file
        "final_points": MIN_POINTS,
        "iterations": 10000,
        "densify": False,
    },
    "2b": {
        "name": "cube_2x2x2_random",
        "desc": "2×2×2 cubes, random init → 512 cells, with densification",
        "scene": "cube_2x2x2",
        "manual_points": None,
        "init_points": 64,
        "final_points": 512,
        "iterations": 10000,
        "densify": True,
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(test_id):
    t = TESTS[test_id]
    out_name = t["name"]
    out_dir = os.path.join(OUT_DIR, out_name)

    # Check if already done
    metrics_path = os.path.join(out_dir, "metrics.txt")
    if os.path.exists(metrics_path):
        print(f"[SKIP] {test_id} ({out_name}) — already completed")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # Build and save config
    cfg = base_config(
        scene_type=t["scene"],
        out_name=out_name,
        init_points=t["init_points"],
        final_points=t["final_points"],
        iterations=t["iterations"],
        densify=t["densify"],
    )

    config_file = os.path.join(out_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save manual points if needed
    if t["manual_points"] is not None:
        pts = t["manual_points"]()
        pts_file = os.path.join(out_dir, "init_points.pt")
        save_init_points(pts, pts_file)
        extra_args = ["--init_points_file", pts_file]
    else:
        extra_args = []

    # Run training
    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"cube_sanity/{out_name}",
    ] + extra_args
    print(f"[RUN] {test_id}: {t['desc']}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"[FAIL] {test_id} exited with code {result.returncode}")
        return False

    if os.path.exists(metrics_path):
        print(f"[DONE] {test_id}: {out_name}")
        return True
    else:
        print(f"[WARN] {test_id} finished but no metrics.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description="Cube sanity checks")
    parser.add_argument("--test", nargs="+", choices=list(TESTS.keys()),
                        help="Run specific tests")
    parser.add_argument("--list", action="store_true",
                        help="List all tests and exit")
    args = parser.parse_args()

    if args.list:
        for tid, t in TESTS.items():
            print(f"  {tid}: {t['desc']}")
        return

    tests = args.test or list(TESTS.keys())
    os.makedirs(OUT_DIR, exist_ok=True)

    for tid in tests:
        run_test(tid)


if __name__ == "__main__":
    main()
