#!/usr/bin/env python3
"""Sanity check: can the Voronoi representation reconstruct simple cube scenes?

Tests:
  1a. Single cube, 7 manually-placed points (no densification) -- representability
  1b. Single cube, random init + densification -- optimizer convergence
  2a. 2x2x2 cubes, manually-placed points (no densification)
  2b. 2x2x2 cubes, random init + densification

Thin-surface (split-cell) variant: pass --thin-surface to inject the K=4
two-sided sub-cell partition config (thin_surface_start, K=4, delta/height
regularization, and a boundary-alignment warm-start that populates
_last_top_eigvec before activation). This is the cube smoke route for the
thin-surface P1 runs.

Usage (from repo root):
    python old/test_cube.py                  # run all 4 scalar tests
    python old/test_cube.py --test 1a       # run specific test
    python old/test_cube.py --list          # show test names
    python old/test_cube.py --thin-surface              # all tests, thin-surface K=4
    python old/test_cube.py --test 1a --thin-surface    # thin variant of 1a
    python old/test_cube.py --thin-surface --thin-start 4000

Operational unblocker (M1):
    python old/test_cube.py --test 1b --thin-surface --run-tag R0
        # writes to output/cube_sanity/single_cube_random_thin_R0/
        # and uses --experiment_name cube_sanity/single_cube_random_thin_R0
        # so reruns (R0/R1/R2/...) do not collide with historical results.
    python old/test_cube.py --test 1b --run-tag R0
        # (no --thin-surface) => output/cube_sanity/single_cube_random_R0/

The default behaviour (no --run-tag) is unchanged: outputs land in the
historical single_cube_* / cube_2x2x2_* directories, with the standard
non-suffixed --experiment_name.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

# Repo root = parent of this file's directory (old/). All paths and the
# train.py subprocess cwd are anchored here so the script is invokable from
# anywhere (esp. `python old/test_cube.py` from repo root on a worker).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "output", "cube_sanity")


# ---------------------------------------------------------------------------
# Manual point placement for cube scenes
# ---------------------------------------------------------------------------

MIN_POINTS = 32  # Delaunay triangulation minimum


def pad_to_min(pts, scale=1.05):
    """Pad point set to minimum triangulation size with far-away filler points."""
    if pts.shape[0] >= MIN_POINTS:
        return pts
    n_pad = MIN_POINTS - pts.shape[0]
    filler = torch.randn(n_pad, 3) * 0.1
    filler = filler / filler.norm(dim=-1, keepdim=True) * scale
    filler += torch.randn_like(filler) * 1e-4
    return torch.cat([pts, filler], dim=0)


def single_cube_points():
    """7 points representing a single cube (side 0.5, centered at origin)."""
    jitter = 1e-4
    pts = torch.tensor([
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.5],
        [0.0, 0.0, -0.5],
    ], dtype=torch.float32)
    pts += torch.randn_like(pts) * jitter
    return pad_to_min(pts)


def cube_2x2x2_points():
    """Points for 2x2x2 cube scene: 8 centers + 24 face points = 32."""
    jitter = 1e-4
    block_half = 0.25
    ch = block_half / 2  # 0.125
    centers = [[sx * ch, sy * ch, sz * ch]
               for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]]
    outer = 2 * block_half - ch  # 0.4375
    face_points = []
    for sign in [-1, 1]:
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                face_points.append([sign * outer, sa * ch, sb * ch])
                face_points.append([sa * ch, sign * outer, sb * ch])
                face_points.append([sa * ch, sb * ch, sign * outer])
    pts = torch.tensor(centers + face_points, dtype=torch.float32)
    pts += torch.randn_like(pts) * jitter
    return pad_to_min(pts)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def base_config(scene_type, out_name, init_points, final_points, iterations,
                densify=False, thin_surface=False, thin_start=6000):
    """Build a config dict for a cube test run.

    thin_surface=True injects the K=4 two-sided split-cell config: the surface
    activates at thin_start (after densification settles), with delta/height
    regularization and a boundary-alignment warm-start (top_eig_align) that
    populates _last_top_eigvec so initialize_thin_surface can warm-start the
    quaternions.
    """
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
        "log_percent": 50,
        "diag_percent": 50,
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "density_grad_clip": 10.0,
        "tv_weight": 0.0, "tv_start": 0, "tv_epsilon": 1e-4,
        "tv_area_weighted": False, "tv_border": False,
        "tv_anneal": False, "tv_on_raw": True,
        "interpolation_start": -1, "interp_ramp": False,
        "interp_sigma_scale": 0.7, "interp_sigma_v": 0.2,
        "per_cell_sigma": True, "per_neighbor_sigma": True,
        "bf_start": -1, "bf_until": 6000, "bf_period": 10,
        "bf_sigma_init": 2.0, "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0, "bf_sigma_v_final": 0.1,
        "gaussian_start": -1, "freeze_base_at_gaussian": False,
        "joint_finetune_start": -1,
        "peak_lr_init": 1e-2, "peak_lr_final": 1e-3,
        "offset_lr_init": 1e-3, "offset_lr_final": 1e-4,
        "cov_lr_init": 1e-2, "cov_lr_final": 1e-3,
        "gradient_start": -1, "gradient_lr_init": 1e-2,
        "gradient_lr_final": 1e-3, "gradient_warmup": 500,
        "gradient_max_slope": 5.0, "gradient_freeze_points": 500,
        "redundancy_threshold": 0.0, "redundancy_cap": 0.0,
        "targeted_fraction": 0.0, "targeted_start": -1,
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
        cfg["densify_from"] = 100
        cfg["densify_until"] = 100
        cfg["densify_factor"] = 1.0
        cfg["gradient_fraction"] = 1.0
        cfg["idw_fraction"] = 0.0
        cfg["entropy_fraction"] = 0.0
        cfg["freeze_points"] = int(iterations * 0.95)

    if thin_surface:
        # K=4 split-cell partition (P0 policy: K=4 only). Activates after
        # densification settles; delta/height regularization keeps the surface
        # sparse where unsupported by data.
        assert thin_start < cfg["freeze_points"], \
            "thin_start must precede freeze_points"
        cfg["thin_surface_start"] = thin_start
        cfg["thin_surface_K"] = 4
        cfg["thin_surface_delta_weight"] = 1e-3
        cfg["thin_surface_height_weight"] = 5e-4
        cfg["thin_surface_gate_tau"] = 0.01
        # Boundary-alignment warm-start: populates _last_top_eigvec in
        # [densify_from, thin_start) so initialize_thin_surface can orient the
        # quaternions. Gates off per-cell once the surface activates.
        cfg["top_eig_align_weight"] = 1e-2
        cfg["top_eig_align_start"] = cfg["densify_from"]
        cfg["top_eig_align_until"] = thin_start

    return cfg


def save_init_points(pts, path):
    torch.save(pts, path)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS = {
    "1a": {"name": "single_cube_manual", "desc": "Single cube, 7+pad manual points, no densify",
           "scene": "cube_single", "manual_points": single_cube_points,
           "init_points": MIN_POINTS, "final_points": MIN_POINTS,
           "iterations": 10000, "densify": False},
    "1b": {"name": "single_cube_random", "desc": "Single cube, random init -> 512, densify",
           "scene": "cube_single", "manual_points": None,
           "init_points": 64, "final_points": 512,
           "iterations": 10000, "densify": True},
    "2a": {"name": "cube_2x2x2_manual", "desc": "2x2x2 cubes, 32 manual points, no densify",
           "scene": "cube_2x2x2", "manual_points": cube_2x2x2_points,
           "init_points": MIN_POINTS, "final_points": MIN_POINTS,
           "iterations": 10000, "densify": False},
    "2b": {"name": "cube_2x2x2_random", "desc": "2x2x2 cubes, random init -> 512, densify",
           "scene": "cube_2x2x2", "manual_points": None,
           "init_points": 64, "final_points": 512,
           "iterations": 10000, "densify": True},
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(test_id, thin_surface=False, thin_start=6000, run_tag=None):
    t = TESTS[test_id]
    base_out_name = t["name"] + ("_thin" if thin_surface else "")
    # M1 operational unblocker: append a unique tag (e.g. R0/R1/...) so rescue
    # reruns do not collide with historical results and the [SKIP] check below
    # does not falsely skip a fresh run because the historical metrics.txt is
    # already in place.  When --run-tag is omitted the default name is
    # unchanged so existing callers see no behavioural change.
    tag_suffix = f"_{run_tag}" if run_tag else ""
    out_name = base_out_name + tag_suffix
    out_dir = os.path.join(OUT_DIR, out_name)

    metrics_path = os.path.join(out_dir, "metrics.txt")
    if os.path.exists(metrics_path):
        print(f"[SKIP] {test_id} ({out_name}) -- already completed")
        return True

    os.makedirs(out_dir, exist_ok=True)

    cfg = base_config(
        scene_type=t["scene"], out_name=out_name,
        init_points=t["init_points"], final_points=t["final_points"],
        iterations=t["iterations"], densify=t["densify"],
        thin_surface=thin_surface, thin_start=thin_start,
    )

    config_file = os.path.join(out_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    extra_args = []
    if t["manual_points"] is not None:
        pts = t["manual_points"]()
        pts_file = os.path.join(out_dir, "init_points.pt")
        save_init_points(pts, pts_file)
        extra_args = ["--init_points_file", pts_file]

    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "train.py"),
        "-c", config_file,
        "--experiment_name", f"cube_sanity/{out_name}",
    ] + extra_args
    print(f"[RUN] {test_id}: {t['desc']}"
          + (" [thin-surface K=4]" if thin_surface else "")
          + (f" [run_tag={run_tag}]" if run_tag else ""))
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"[FAIL] {test_id} exited with code {result.returncode}")
        return False
    if os.path.exists(metrics_path):
        print(f"[DONE] {test_id}: {out_name}")
        return True
    print(f"[WARN] {test_id} finished but no metrics.txt")
    return False


def main():
    parser = argparse.ArgumentParser(description="Cube sanity checks")
    parser.add_argument("--test", nargs="+", choices=list(TESTS.keys()),
                        help="Run specific tests")
    parser.add_argument("--list", action="store_true", help="List all tests and exit")
    parser.add_argument("--thin-surface", action="store_true",
                        help="Inject the K=4 two-sided split-cell (thin-surface) "
                             "config into every selected test")
    parser.add_argument("--thin-start", type=int, default=6000,
                        help="thin_surface_start iteration (default 6000; must be "
                             "< freeze_points)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Append a unique suffix to the output directory and "
                             "--experiment_name so reruns (e.g. R0/R1/R2 ...) "
                             "do not collide with historical results. "
                             "Default: no suffix (preserves prior behaviour).")
    args = parser.parse_args()

    if args.list:
        for tid, t in TESTS.items():
            print(f"  {tid}: {t['desc']}")
        return

    tests = args.test or list(TESTS.keys())
    os.makedirs(OUT_DIR, exist_ok=True)

    for tid in tests:
        run_test(tid, thin_surface=args.thin_surface,
                  thin_start=args.thin_start, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
