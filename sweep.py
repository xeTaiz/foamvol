#!/usr/bin/env python3
"""Hyperparameter sweep for CT reconstruction.

Sweep 10: Point budgets, TV annealing, batch size, L1 loss.

Baseline: r2fast.yaml (sigma_v=0.35, density_lr_final=1e-3, init_scale=1.05,
          interp_sigma_scale=0.7).

Usage:
    python sweep.py                           # all 15 runs
    python sweep.py --runs A1-256k A2-512k    # specific runs for worker splitting
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
# Baseline (from r2fast.yaml — sweep 9 winners applied)
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
    "interp_sigma_v": 0.35,
    "redundancy_threshold": 0.01,
    "redundancy_cap": 0.05,
    # Model
    "init_points": 32000,
    "final_points": 128000,
    "activation_scale": 1.0,
    "init_scale": 1.05,
    "init_type": "random",
    # Optimization
    "points_lr_init": 2e-4,
    "points_lr_final": 5e-6,
    "density_lr_init": 5e-2,
    "density_lr_final": 1e-3,
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

SWEEP_NAME = "sweep10"
SWEEP_DIR = f"output/{SWEEP_NAME}"

# ---------------------------------------------------------------------------
# Sweep 10 runs
# ---------------------------------------------------------------------------

SWEEP10_RUNS = {
    # Group A: Point budgets (extend densify_until for larger budgets)
    "A1-256k":   {"init_points": 64000,  "final_points": 256000,  "densify_until": 7000, "freeze_points": 9500},
    "A2-512k":   {"init_points": 128000, "final_points": 512000,  "densify_until": 7500, "freeze_points": 9500},
    "A3-768k":   {"init_points": 192000, "final_points": 768000,  "densify_until": 8000, "freeze_points": 9500},
    "A4-1M":     {"init_points": 256000, "final_points": 1000000, "densify_until": 8000, "freeze_points": 9500},

    # Group B: Densification params at 512k
    "B1-df110":  {"init_points": 128000, "final_points": 512000, "densify_until": 7500, "densify_factor": 1.10},
    "B2-df120":  {"init_points": 128000, "final_points": 512000, "densify_until": 7500, "densify_factor": 1.20},
    "B3-du8500": {"init_points": 128000, "final_points": 512000, "densify_until": 8500, "densify_factor": 1.15},

    # Group C: TV annealing
    "C1-anneal":      {"tv_anneal": True},
    "C2-anneal-512k": {"init_points": 128000, "final_points": 512000, "densify_until": 7500, "tv_anneal": True},

    # Group D: L1 loss
    "D1-l1":      {"loss_type": "l1"},
    "D2-l1-512k": {"init_points": 128000, "final_points": 512000, "densify_until": 7500, "loss_type": "l1"},

    # Group E: Batch size (rays_per_batch)
    "E1-500k": {"rays_per_batch": 500000},
    "E2-1M":   {"rays_per_batch": 1000000},
    "E3-4M":   {"rays_per_batch": 4000000},

    # Reference
    "baseline": {},
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
    """Run all (or selected) sweep 10 experiments.

    Args:
        runs: Optional list of specific run IDs to execute.
        summarize: If True, only collect summary from existing results.
    """
    all_names = list(SWEEP10_RUNS.keys())

    if runs:
        selected = set(runs)
        names = [n for n in all_names if n in selected]
        unknown = selected - set(all_names)
        for u in unknown:
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    print(f"Sweep 10: {len(names)}/{len(all_names)} runs selected")

    if not summarize:
        for name in names:
            cfg = build_config(SWEEP10_RUNS[name])
            run_experiment(name, cfg)

    # Summary always covers all available results
    return collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CT reconstruction hyperparameter sweep (sweep 10)",
        epilog="Examples:\n"
               "  python sweep.py                           # all 15 runs\n"
               "  python sweep.py --runs A1-256k A2-512k    # specific runs\n"
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
        print(f"\n{len(SWEEP10_RUNS)} sweep 10 runs:")
        for name in SWEEP10_RUNS:
            overrides = SWEEP10_RUNS[name]
            desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "(baseline)"
            print(f"  {name:16s}  {desc}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    run_sweep(runs=args.runs, summarize=args.summarize)


if __name__ == "__main__":
    main()
