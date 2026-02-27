#!/usr/bin/env python3
"""Hyperparameter sweep for CT reconstruction.

Sweep 7: Interpolation + Density Grad evaluation.
5 feature configs × 2 point budgets = 10 runs.

Usage:
    python sweep.py --phase 1
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

FEATURE_CONFIGS = {
    "base": {},
    "I-sharp": {
        "interpolation_start": 15000,
        "interp_sigma_scale": 0.25,
        "interp_sigma_v": 0.05,
    },
    "I-smooth": {
        "interpolation_start": 15000,
        "interp_sigma_scale": 0.4,
        "interp_sigma_v": 0.2,
    },
    "G-lr5": {
        "gradient_start": 15000,
        "gradient_lr_init": 1e-5,
        "gradient_lr_final": 1e-7,
        "gradient_freeze_points": 0,
    },
    "G-lr6": {
        "gradient_start": 15000,
        "gradient_lr_init": 1e-6,
        "gradient_lr_final": 1e-8,
        "gradient_freeze_points": 0,
    },
}

POINTS_CONFIGS = {
    "P128": {"init_points": 8000, "final_points": 128000},
    "P256": {"init_points": 16000, "final_points": 256000},
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
    "tv_weight": 1e-5,
    "tv_start": 8000,
    "tv_area_weighted": False,
    "tv_epsilon": 1e-3,
    "contrast_fraction": 0.5,
    "contrast_power": 0.5,
    "interpolation_start": -1,
    "interp_sigma_scale": 0.5,
    "interp_sigma_v": 0.1,
    "gradient_start": -1,
    "gradient_warmup": 50,
    "gradient_max_slope": 5.0,
    "gradient_freeze_points": 1000,
    "dataset": "r2_gaussian",
    "data_path": "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
    "debug": False,
    "viewer": False,
}

SWEEP_NAM = "sweep7"
SWEEP_DIR = f"output/{SWEEP_NAM}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_config(f_id, p_id):
    """Merge baseline with feature and points configs."""
    cfg = dict(BASELINE)
    cfg.update(FEATURE_CONFIGS[f_id])
    cfg.update(POINTS_CONFIGS[p_id])
    return cfg


def run_name(f_id, p_id):
    return f"{f_id}_{p_id}"


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
# Phase runner
# ---------------------------------------------------------------------------


def phase1():
    """Run all feature × points combos."""
    runs = []
    for f_id, p_id in itertools.product(FEATURE_CONFIGS, POINTS_CONFIGS):
        name = run_name(f_id, p_id)
        cfg = build_config(f_id, p_id)
        runs.append((name, cfg))

    print(f"Phase 1: {len(runs)} runs")
    for name, cfg in runs:
        run_experiment(name, cfg)

    names = [name for name, _ in runs]
    return collect_summary(names, os.path.join(SWEEP_DIR, "summary_phase1.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CT reconstruction hyperparameter sweep")
    parser.add_argument("--phase", type=int, default=1, choices=[1],
                        help="1 = run all feature × points combos")
    args = parser.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)

    phase1()


if __name__ == "__main__":
    main()
