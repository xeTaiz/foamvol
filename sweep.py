#!/usr/bin/env python3
"""Hyperparameter sweep for CT reconstruction.

Sweep 13: Interpolation (sigma_scale × sigma_v), contrast, densification.
Baseline: 256k points, no TV regularization.

Usage:
    python sweep.py                           # all runs
    python sweep.py --runs A1-ss10-sv03 B2-cp10  # specific runs
    python sweep.py --worker 1 --of 4         # run 1st quarter
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
# Baseline (from r2fast.yaml)
# ---------------------------------------------------------------------------

BASELINE = {
    # Pipeline
    "iterations": 10000,
    "densify_from": 1000,
    "densify_until": 6000,
    "densify_factor": 1.15,
    "gradient_fraction": 0.4,
    "idw_fraction": 0.3,
    "contrast_fraction": 0.3,
    "contrast_power": 0.5,
    "contrast_alpha": 4.0,
    "loss_type": "l1",
    "debug": False,
    "viewer": False,
    "save_volume": False,
    "interpolation_start": 9000,
    "interp_ramp": False,
    "interp_sigma_scale": 0.7,
    "interp_sigma_v": 0.2,
    "per_cell_sigma": True,
    "per_neighbor_sigma": True,
    "redundancy_threshold": 0.01,
    "redundancy_cap": 0.05,
    "rays_per_batch": 1000000,
    # Model
    "init_points": 64000,
    "final_points": 256000,
    "activation_scale": 1.0,
    "init_scale": 1.05,
    "init_type": "random",
    "init_density": 2.0,
    "device": "cuda",
    # Optimization
    "points_lr_init": 2e-4,
    "points_lr_final": 5e-6,
    "density_lr_init": 5e-2,
    "density_lr_final": 1e-2,
    "freeze_points": 9500,
    "tv_on_raw": True,
    "tv_anneal": False,
    "tv_weight": 1e-2,
    "tv_start": 0,
    "tv_epsilon": 1e-4,
    "tv_area_weighted": False,
    "density_grad_clip": 10.0,
    "gradient_start": -1,
    # Dataset
    "dataset": "r2_gaussian",
    "data_path": "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
}

SWEEP_NAME = "sweep13"
SWEEP_DIR = f"output/{SWEEP_NAME}"

# ---------------------------------------------------------------------------
# Sweep 13 runs — 256k points, interpolation × contrast × densification
# ---------------------------------------------------------------------------

SWEEP13_RUNS = {
    # Group A: sigma_scale × sigma_v grid (sweep 12 winners were ss=1.0 and sv=0.3/0.5)
    "A1-ss10-sv02": {"interp_sigma_scale": 1.0},
    "A2-ss10-sv03": {"interp_sigma_scale": 1.0, "interp_sigma_v": 0.3},
    "A3-ss10-sv05": {"interp_sigma_scale": 1.0, "interp_sigma_v": 0.5},
    "A4-ss07-sv03": {"interp_sigma_v": 0.3},
    "A5-ss07-sv05": {"interp_sigma_v": 0.5},
    "A6-ss15-sv03": {"interp_sigma_scale": 1.5, "interp_sigma_v": 0.3},

    # Group B: Contrast power
    "B1-cp025": {"contrast_power": 0.25},
    "B2-cp10": {"contrast_power": 1.0},
    "B3-cp20": {"contrast_power": 2.0},

    # Group C: Contrast alpha
    "C1-ca1": {"contrast_alpha": 1.0},
    "C2-ca2": {"contrast_alpha": 2.0},
    "C3-ca8": {"contrast_alpha": 8.0},

    # Group D: Densification budget splits (fractions must sum to 1.0 with gradient_fraction)
    "D1-idw0": {"idw_fraction": 0.0, "contrast_fraction": 0.6},
    "D2-idw05": {"idw_fraction": 0.5, "contrast_fraction": 0.1},
    "D3-grad06": {"gradient_fraction": 0.6, "idw_fraction": 0.1},
    "D4-grad02": {"gradient_fraction": 0.2, "idw_fraction": 0.5},

    # Group E: Interpolation start timing
    "E1-is8k": {"interpolation_start": 8000},
    "E2-off": {"interpolation_start": -1},

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


def run_sweep(runs=None, summarize=False, worker=None, num_workers=None):
    """Run all (or selected) sweep 13 experiments."""
    all_names = list(SWEEP13_RUNS.keys())

    if runs:
        selected = set(runs)
        names = [n for n in all_names if n in selected]
        unknown = selected - set(all_names)
        for u in unknown:
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    if worker is not None and num_workers is not None:
        chunks = [names[i::num_workers] for i in range(num_workers)]
        names = chunks[worker - 1]
        print(f"Sweep 12: worker {worker}/{num_workers} — {len(names)} runs: {', '.join(names)}")
    else:
        print(f"Sweep 12: {len(names)}/{len(all_names)} runs selected")

    if not summarize:
        for name in names:
            cfg = build_config(SWEEP13_RUNS[name])
            run_experiment(name, cfg)

    return collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CT reconstruction hyperparameter sweep (sweep 13)",
        epilog="Examples:\n"
               "  python sweep.py                           # all runs\n"
               "  python sweep.py --runs A1-ss10 B2-cp10    # specific runs\n"
               "  python sweep.py --worker 1 --of 4         # run 1st quarter\n"
               "  python sweep.py --list                    # show run names\n"
               "  python sweep.py --summarize               # just collect results\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID",
                        help="Only run these specific run IDs")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip training, just collect existing results into summary CSV")
    parser.add_argument("--worker", type=int, metavar="W",
                        help="Worker index (1-indexed)")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers",
                        help="Total number of workers")
    parser.add_argument("--list", action="store_true",
                        help="Print all run names and exit")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\n{len(SWEEP13_RUNS)} sweep 13 runs:")
        for name in SWEEP13_RUNS:
            overrides = SWEEP13_RUNS[name]
            desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "(baseline)"
            print(f"  {name:16s}  {desc}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    run_sweep(runs=args.runs, summarize=args.summarize,
              worker=args.worker, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
