#!/usr/bin/env python3
"""Hyperparameter sweep for CT reconstruction.

Phase 1: Screen 4(G) x 5(T) x 5(C) = 100 runs at 64k points.
Phase 2: Validate top N configs from Phase 1 at 128k, 256k, and 512k points.

Usage:
    python sweep.py --phase 1
    python sweep.py --phase 2 --top_n 5
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile

import yaml

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

GRAD_CONFIGS = {
    "G0": {"gradient_start": -1},
    "G-5L0": {
        "gradient_start": 11000,
        "gradient_lr_init": 1e-5,
        "gradient_lr_final": 1e-7,
        "gradient_l2_weight": 0,
    },
    "G-4L-3": {
        "gradient_start": 11000,
        "gradient_lr_init": 1e-4,
        "gradient_lr_final": 1e-6,
        "gradient_l2_weight": 1e-3,
    },
    "G-5L-4": {
        "gradient_start": 11000,
        "gradient_lr_init": 1e-5,
        "gradient_lr_final": 1e-7,
        "gradient_l2_weight": 1e-4,
    },
}

TV_CONFIGS = {
    "T0": {"tv_weight": 0},
    "T-4a": {"tv_weight": 1e-4, "tv_start": 8000, "tv_area_weighted": True},
    "T-5a": {"tv_weight": 1e-5, "tv_start": 8000, "tv_area_weighted": True},
    "T-4": {"tv_weight": 1e-4, "tv_start": 8000, "tv_area_weighted": False},
    "T-5": {"tv_weight": 1e-5, "tv_start": 8000, "tv_area_weighted": False},
}

CONTRAST_CONFIGS = {
    "C00": {"contrast_fraction": 0.0},
    "C25": {"contrast_fraction": 0.25},
    "C50": {"contrast_fraction": 0.5},
    "C75": {"contrast_fraction": 0.75},
    "C90": {"contrast_fraction": 0.90},
}

POINTS_CONFIGS = {
    "P64": {"init_points": 4000, "final_points": 64000},
    "P128": {"init_points": 8000, "final_points": 128000},
    "P256": {"init_points": 16000, "final_points": 256000},
    "P512": {"init_points": 32000, "final_points": 512000},
}

# ---------------------------------------------------------------------------
# Fixed baseline parameters
# ---------------------------------------------------------------------------

BASELINE = {
    "iterations": 20000,
    "densify_from": 2000,
    "densify_until": 11000,
    "densify_factor": 1.15,
    "loss_type": "l2",
    "activation_scale": 1.0,
    "init_scale": 1.0,
    "init_type": "random",
    "points_lr_init": 2e-4,
    "points_lr_final": 5e-6,
    "density_lr_init": 1e-1,
    "density_lr_final": 1e-2,
    "freeze_points": 18000,
    "tv_epsilon": 1e-3,
    "gradient_warmup": 50,
    "gradient_clip": 0.01,
    "gradient_freeze_points": 1000,
    "dataset": "r2_gaussian",
    "data_path": "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
    "debug": False,
    "viewer": False,
}

SWEEP_NAM = "sweep2"
SWEEP_DIR = f"output/{SWEEP_NAM}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_config(g_id, t_id, c_id, p_id):
    """Merge baseline with the four axis configs."""
    cfg = dict(BASELINE)
    cfg.update(GRAD_CONFIGS[g_id])
    cfg.update(TV_CONFIGS[t_id])
    cfg.update(CONTRAST_CONFIGS[c_id])
    cfg.update(POINTS_CONFIGS[p_id])
    return cfg


def run_name(g_id, t_id, c_id, p_id):
    return f"{g_id}_{t_id}_{c_id}_{p_id}"


def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def parse_metrics(path):
    """Parse a metrics.txt file into a dict of floats."""
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                val = float(m.group(2))
                metrics[key] = val
    return metrics


def run_experiment(name, cfg):
    """Run a single training experiment via subprocess."""
    out_dir = os.path.join(SWEEP_DIR, name)
    mpath = metrics_path(name)

    if os.path.exists(mpath):
        print(f"[SKIP] {name} — metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # Write config YAML to a temp file
    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable,
        "train.py",
        "-c", config_file,
        "--experiment_name", f"{SWEEP_NAM}/{name}",
    ]
    print(f"[RUN]  {name}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"[FAIL] {name} exited with code {result.returncode}")
        return False

    if not os.path.exists(mpath):
        print(f"[WARN] {name} finished but metrics.txt not found")
        return False

    return True


def collect_summary(names, output_csv):
    """Read metrics.txt from each run and write a sorted summary CSV."""
    rows = []
    for name in names:
        mpath = metrics_path(name)
        if not os.path.exists(mpath):
            continue
        metrics = parse_metrics(mpath)
        rows.append({"name": name, **metrics})

    rows.sort(key=lambda r: r.get("test_psnr", 0), reverse=True)

    if not rows:
        print(f"[WARN] No completed runs to summarize")
        return rows

    fieldnames = ["name"] + [k for k in rows[0] if k != "name"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Summary written to {output_csv} ({len(rows)} runs)")
    return rows


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def phase1():
    """Screen all G x T x C combos at P64."""
    p_id = "P64"
    runs = []
    for g_id, t_id, c_id in itertools.product(
        GRAD_CONFIGS, TV_CONFIGS, CONTRAST_CONFIGS
    ):
        name = run_name(g_id, t_id, c_id, p_id)
        cfg = build_config(g_id, t_id, c_id, p_id)
        runs.append((name, cfg))

    print(f"Phase 1: {len(runs)} runs at {p_id}")
    for name, cfg in runs:
        run_experiment(name, cfg)

    names = [name for name, _ in runs]
    return collect_summary(names, os.path.join(SWEEP_DIR, "summary_phase1.csv"))


def phase2(top_n=5):
    """Validate top N configs from Phase 1 at P128, P256, and P512."""
    phase1_csv = os.path.join(SWEEP_DIR, "summary_phase1.csv")
    if not os.path.exists(phase1_csv):
        print(f"[ERROR] {phase1_csv} not found. Run --phase 1 first.")
        sys.exit(1)

    with open(phase1_csv) as f:
        reader = csv.DictReader(f)
        phase1_rows = list(reader)

    # Extract the G/T/C combo from the top N names
    top_combos = []
    for row in phase1_rows[:top_n]:
        name = row["name"]
        parts = name.split("_")  # e.g. G2_T1_C1_P64
        g_id, t_id, c_id = parts[0], parts[1], parts[2]
        top_combos.append((g_id, t_id, c_id))

    print(f"Phase 2: Top {len(top_combos)} combos × 3 point budgets")
    for g_id, t_id, c_id in top_combos:
        print(f"  {g_id}_{t_id}_{c_id}")

    runs = []
    for g_id, t_id, c_id in top_combos:
        for p_id in ("P128", "P256", "P512"):
            name = run_name(g_id, t_id, c_id, p_id)
            cfg = build_config(g_id, t_id, c_id, p_id)
            runs.append((name, cfg))

    for name, cfg in runs:
        run_experiment(name, cfg)

    names = [name for name, _ in runs]
    return collect_summary(names, os.path.join(SWEEP_DIR, "summary_phase2.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CT reconstruction hyperparameter sweep")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="1 = screen at 64k, 2 = validate top N at 128k/256k/512k")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of top configs to validate in Phase 2")
    args = parser.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)

    if args.phase == 1:
        phase1()
    else:
        phase2(top_n=args.top_n)


if __name__ == "__main__":
    main()
