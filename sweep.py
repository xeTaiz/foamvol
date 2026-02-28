#!/usr/bin/env python3
"""Hyperparameter sweep for CT reconstruction.

Sweep 9: Refinement around sweep 8 winners.
Combined base: density_lr_init=5e-2, interp_sigma_scale=0.7 (rest from r2fast.yaml).

Usage:
    python sweep.py                           # all 15 runs
    python sweep.py --runs E1-sv03 E2-sv04    # specific runs for worker splitting
    python sweep.py --list                    # print run names
    python sweep.py --summarize               # collect results only
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Baseline (combined best from sweep 8: density_lr_init=5e-2 + sigma_scale=0.7)
# ---------------------------------------------------------------------------

BASELINE = {
    # Pipeline
    "iterations": 10000,
    "densify_from": 1000,
    "densify_until": 6000,
    "densify_factor": 1.15,
    "contrast_fraction": 0.5,
    "loss_type": "l2",
    "debug": False,
    "viewer": False,
    "save_volume": False,
    "interpolation_start": 9000,
    "interp_sigma_scale": 0.7,
    "interp_sigma_v": 0.2,
    "redundancy_threshold": 0.01,
    "redundancy_cap": 0.05,
    # Model
    "init_points": 32000,
    "final_points": 128000,
    "activation_scale": 1.0,
    "init_scale": 1.0,
    "init_type": "random",
    # Optimization
    "points_lr_init": 2e-4,
    "points_lr_final": 5e-6,
    "density_lr_init": 5e-2,
    "density_lr_final": 1e-2,
    "freeze_points": 9500,
    "tv_weight": 1e-4,
    "tv_start": 5000,
    "tv_epsilon": 1e-4,
    "tv_area_weighted": False,
    "gradient_start": -1,
    # Dataset
    "dataset": "r2_gaussian",
    "data_path": "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
}

SWEEP_NAME = "sweep9"
SWEEP_DIR = f"output/{SWEEP_NAME}"

# ---------------------------------------------------------------------------
# All sweep 9 runs (flat dict — no phases)
# ---------------------------------------------------------------------------

SWEEP9_RUNS = {
    "baseline": {},
    # Group E: sigma_v exploration
    "E1-sv03":   {"interp_sigma_v": 0.3},
    "E2-sv04":   {"interp_sigma_v": 0.4},
    "E3-sv06":   {"interp_sigma_v": 0.6},
    # Group F: density_lr_final annealing
    "F1-dlf5e3": {"density_lr_final": 5e-3},
    "F2-dlf2e3": {"density_lr_final": 2e-3},
    "F3-dlf1e3": {"density_lr_final": 1e-3},
    # Group G: sigma_scale refinement
    "G1-ss06":   {"interp_sigma_scale": 0.6},
    "G2-ss08":   {"interp_sigma_scale": 0.8},
    # Group H: point budgets (1:4 init:final ratio)
    "H1-64k":    {"init_points": 16000, "final_points": 64000},
    "H2-80k":    {"init_points": 20000, "final_points": 80000},
    "H3-100k":   {"init_points": 25000, "final_points": 100000},
    "H4-140k":   {"init_points": 35000, "final_points": 140000},
    # Group X: spot-checks
    "X1-pli1e3": {"points_lr_init": 1e-3},
    "X2-tv5e5":  {"tv_weight": 5e-5},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_config(overrides):
    """Merge baseline with overrides."""
    cfg = dict(BASELINE)
    cfg.update(overrides)
    return cfg


def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def parse_metrics(path):
    """Parse a metrics.txt file into a dict of floats."""
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
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

    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable,
        "train.py",
        "-c", config_file,
        "--experiment_name", f"{SWEEP_NAME}/{name}",
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


def collect_summary(names, output_csv, sort_key="vol_idw_psnr"):
    """Read metrics.txt from each run and write a sorted summary CSV."""
    rows = []
    for name in names:
        mpath = metrics_path(name)
        if not os.path.exists(mpath):
            continue
        metrics = parse_metrics(mpath)
        rows.append({"name": name, **metrics})

    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    if not rows:
        print("[WARN] No completed runs to summarize")
        return rows

    fieldnames = ["name"] + [k for k in rows[0] if k != "name"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Summary written to {output_csv} ({len(rows)} runs)")
    return rows


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_sweep(runs=None, summarize=False):
    """Run all (or selected) sweep 9 experiments.

    Args:
        runs: Optional list of specific run IDs to execute.
        summarize: If True, only collect summary from existing results.
    """
    all_names = list(SWEEP9_RUNS.keys())

    if runs:
        selected = set(runs)
        names = [n for n in all_names if n in selected]
        unknown = selected - set(all_names)
        for u in unknown:
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    print(f"Sweep 9: {len(names)}/{len(all_names)} runs selected")

    if not summarize:
        for name in names:
            cfg = build_config(SWEEP9_RUNS[name])
            run_experiment(name, cfg)

    # Summary always covers all available results
    return collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CT reconstruction hyperparameter sweep (sweep 9)",
        epilog="Examples:\n"
               "  python sweep.py                           # all 15 runs\n"
               "  python sweep.py --runs E1-sv03 E2-sv04    # specific runs\n"
               "  python sweep.py --list                    # show run names\n"
               "  python sweep.py --summarize               # just collect results\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID",
                        help="Only run these specific run IDs")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip training, just collect existing results into summary CSV")
    parser.add_argument("--list", action="store_true",
                        help="Print all run names and exit")
    args = parser.parse_args()

    if args.list:
        print(f"\n{len(SWEEP9_RUNS)} sweep 9 runs:")
        for name in SWEEP9_RUNS:
            overrides = SWEEP9_RUNS[name]
            desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "(combined base)"
            print(f"  {name:12s}  {desc}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    run_sweep(runs=args.runs, summarize=args.summarize)


if __name__ == "__main__":
    main()
