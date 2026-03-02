#!/usr/bin/env python3
"""Hyperparameter sweep for bilateral filter + entropy densification.

Sweep 15: Bilateral filter schedule × entropy bins/fraction.
Baseline: r2_bf.yaml (bf enabled, entropy densification).

Usage:
    python sweep_bf_entropy.py                           # all runs
    python sweep_bf_entropy.py --runs A1-bf-off B2-sv05  # specific runs
    python sweep_bf_entropy.py --worker 1 --of 4         # run 1st quarter
    python sweep_bf_entropy.py --list                    # print run names
    python sweep_bf_entropy.py --summarize               # collect results only
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Baseline (from r2_bf.yaml — bilateral filter enabled, entropy densification)
# ---------------------------------------------------------------------------

BASELINE = {
    # Pipeline
    "iterations": 10000,
    "densify_from": 1000,
    "densify_until": 6000,
    "densify_factor": 1.15,
    "gradient_fraction": 0.4,
    "idw_fraction": 0.3,
    "entropy_fraction": 0.3,
    "entropy_bins": 5,
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
    # Bilateral filter
    "bf_start": 0,
    "bf_until": 6000,
    "bf_period": 10,
    "bf_sigma_init": 2.0,
    "bf_sigma_final": 0.3,
    "bf_sigma_v_init": 10.0,
    "bf_sigma_v_final": 0.1,
    # Model
    "init_points": 64000,
    "final_points": 512000,
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

SWEEP_NAME = "sweep15"
SWEEP_DIR = f"output/{SWEEP_NAME}"

# ---------------------------------------------------------------------------
# Sweep 15 runs — bilateral filter × entropy densification
# ---------------------------------------------------------------------------

SWEEP15_RUNS = {
    # ---- Group A: BF on/off and duration ----
    "A1-bf-off":       {"bf_start": -1},                          # no bilateral filter
    "A2-bf-8k":        {"bf_until": 8000},                        # filter until 8k
    "A3-bf-9k":        {"bf_until": 9000},                        # filter until 9k
    "A4-bf-full":      {"bf_until": 10000},                       # filter for entire training
    "A5-bf-early":     {"bf_until": 3000},                        # filter only early phase

    # ---- Group B: BF final sigma_v (value selectivity at end) ----
    "B1-sv005":        {"bf_sigma_v_final": 0.05},                # very tight — strong edges
    "B2-sv02":         {"bf_sigma_v_final": 0.2},                 # looser — more smoothing
    "B3-sv05":         {"bf_sigma_v_final": 0.5},                 # very loose — near-Gaussian
    "B4-sv-flat":      {"bf_sigma_v_init": 0.1, "bf_sigma_v_final": 0.1},  # no anneal

    # ---- Group C: BF final sigma (spatial reach at end) ----
    "C1-ss01":         {"bf_sigma_final": 0.1},                   # tight spatial
    "C2-ss05":         {"bf_sigma_final": 0.5},                   # wide spatial
    "C3-ss10":         {"bf_sigma_final": 1.0},                   # very wide spatial

    # ---- Group D: BF period (frequency of smoothing) ----
    "D1-per5":         {"bf_period": 5},                          # more frequent
    "D2-per20":        {"bf_period": 20},                         # less frequent
    "D3-per50":        {"bf_period": 50},                         # infrequent

    # ---- Group E: Entropy bins ----
    "E1-bins3":        {"entropy_bins": 3},                       # coarse binning
    "E2-bins8":        {"entropy_bins": 8},                       # fine binning
    "E3-bins12":       {"entropy_bins": 12},                      # very fine binning

    # ---- Group F: Entropy fraction (budget split) ----
    "F1-ent0":         {"entropy_fraction": 0.0, "gradient_fraction": 0.6, "idw_fraction": 0.4},
    "F2-ent05":        {"entropy_fraction": 0.5, "gradient_fraction": 0.3, "idw_fraction": 0.2},
    "F3-ent-only":     {"entropy_fraction": 0.6, "gradient_fraction": 0.0, "idw_fraction": 0.4},
    "F4-grad-heavy":   {"entropy_fraction": 0.1, "gradient_fraction": 0.6, "idw_fraction": 0.3},

    # ---- Group G: BF + entropy interactions ----
    "G1-bf-off-ent0":  {"bf_start": -1, "entropy_fraction": 0.0,
                         "gradient_fraction": 0.6, "idw_fraction": 0.4},  # old-style baseline
    "G2-bf-full-bins8": {"bf_until": 10000, "entropy_bins": 8},           # full filter + fine bins
    "G3-bf-early-ent05": {"bf_until": 3000, "entropy_fraction": 0.5,
                          "gradient_fraction": 0.3, "idw_fraction": 0.2}, # short filter + heavy entropy

    # Reference
    "baseline":        {},
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
    """Run all (or selected) sweep 15 experiments."""
    all_names = list(SWEEP15_RUNS.keys())

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
        print(f"Sweep 15: worker {worker}/{num_workers} — {len(names)} runs: {', '.join(names)}")
    else:
        print(f"Sweep 15: {len(names)}/{len(all_names)} runs selected")

    if not summarize:
        for name in names:
            cfg = build_config(SWEEP15_RUNS[name])
            run_experiment(name, cfg)

    return collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CT reconstruction BF + entropy sweep (sweep 15)",
        epilog="Examples:\n"
               "  python sweep_bf_entropy.py                           # all runs\n"
               "  python sweep_bf_entropy.py --runs A1-bf-off B2-sv05  # specific runs\n"
               "  python sweep_bf_entropy.py --worker 1 --of 4         # run 1st quarter\n"
               "  python sweep_bf_entropy.py --list                    # show run names\n"
               "  python sweep_bf_entropy.py --summarize               # just collect results\n",
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
        print(f"\n{len(SWEEP15_RUNS)} sweep 15 runs:")
        for name in SWEEP15_RUNS:
            overrides = SWEEP15_RUNS[name]
            desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "(baseline)"
            print(f"  {name:20s}  {desc}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    run_sweep(runs=args.runs, summarize=args.summarize,
              worker=args.worker, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
