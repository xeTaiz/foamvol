#!/usr/bin/env python3
"""Cell count scaling sweep for CT reconstruction.

Sweep 17: Tests 2 rendering modes × 2 densification × 4 cell counts at
10k iterations (Part A), plus 20k iterations at selected cell counts (Part B).
Uses 500-projection dataset to isolate the representation ceiling.

Part A: 16 runs @ 10k iterations (512k, 1M, 2M, 4M cells)
Part B: 12 runs @ 20k iterations (512k, 1M, 4M cells)
Total: 28 runs

Usage:
    python sweep_scaling.py                    # all 28 runs
    python sweep_scaling.py --runs A01 B03     # specific runs
    python sweep_scaling.py --worker 1 --of 4  # distributed
    python sweep_scaling.py --list             # list all
    python sweep_scaling.py --summarize        # collect results
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SWEEP_DIR = "output/sweep17_scaling"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_500_angle_360/0_chest_cone"

# Scheduling ratios (proportion of total iterations)
SCHEDULE_RATIOS = {
    "densify_from": 0.10,
    "densify_until": 0.60,
    "interpolation_start": 0.90,
    "freeze_points": 0.95,
}


def make_baseline(iterations, final_points, interp=False, full_densify=False):
    """Build a config dict with proportional scheduling."""
    cfg = {
        # Training
        "iterations": iterations,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": final_points,
        "activation_scale": 1.0,
        "init_scale": 1.05,
        "init_type": "random",
        "init_density": 2.0,
        "loss_type": "l1",
        "debug": False,
        "viewer": False,
        "save_volume": False,
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
        # Densification schedule (proportional)
        "densify_from": int(iterations * SCHEDULE_RATIOS["densify_from"]),
        "densify_until": int(iterations * SCHEDULE_RATIOS["densify_until"]),
        "densify_factor": 1.15,
        "freeze_points": int(iterations * SCHEDULE_RATIOS["freeze_points"]),
        # Pruning OFF
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.0,
        # Targeted sampling OFF
        "targeted_fraction": 0.0,
        "targeted_start": -1,
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
        # Interpolation
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": True,
        "per_neighbor_sigma": True,
        # Contrast OFF
        "contrast_alpha": 0.0,
        "entropy_bins": 5,
        # Dataset
        "dataset": "r2_gaussian",
        "data_path": DATA_PATH,
    }

    # Densification strategy
    if full_densify:
        cfg["gradient_fraction"] = 0.4
        cfg["idw_fraction"] = 0.3
        cfg["entropy_fraction"] = 0.3
    else:
        cfg["gradient_fraction"] = 1.0
        cfg["idw_fraction"] = 0.0
        cfg["entropy_fraction"] = 0.0

    # Interpolation
    if interp:
        cfg["interpolation_start"] = int(iterations * SCHEDULE_RATIOS["interpolation_start"])
    else:
        cfg["interpolation_start"] = -1

    return cfg


# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------

CELL_COUNTS = {
    "512k": 512000,
    "1M": 1000000,
    "2M": 2000000,
    "4M": 4000000,
}

CONFIGS = [
    ("base-grad",    False, False),  # base rendering, gradient-only densify
    ("base-full",    False, True),   # base rendering, full densify
    ("interp-grad",  True,  False),  # interp rendering, gradient-only densify
    ("interp-full",  True,  True),   # interp rendering, full densify
]


def build_runs():
    """Build the full run dictionary."""
    runs = {}

    # Part A: 10k iterations, all cell counts
    for cfg_name, interp, full_densify in CONFIGS:
        for cells_name, cells in CELL_COUNTS.items():
            idx = len([r for r in runs if r.startswith("A")]) + 1
            name = f"A{idx:02d}-{cfg_name}-{cells_name}-10k"
            runs[name] = make_baseline(10000, cells, interp=interp,
                                       full_densify=full_densify)

    # Part B: 20k iterations, selected cell counts (skip 2M)
    selected = {"512k": 512000, "1M": 1000000, "4M": 4000000}
    for cfg_name, interp, full_densify in CONFIGS:
        for cells_name, cells in selected.items():
            idx = len([r for r in runs if r.startswith("B")]) + 1
            name = f"B{idx:02d}-{cfg_name}-{cells_name}-20k"
            runs[name] = make_baseline(20000, cells, interp=interp,
                                       full_densify=full_densify)

    return runs


ALL_RUNS = build_runs()


# ---------------------------------------------------------------------------
# Infrastructure (same pattern as sweep_ablation.py)
# ---------------------------------------------------------------------------


def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def parse_metrics(path):
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
    out_dir = os.path.join(SWEEP_DIR, name)
    mpath = metrics_path(name)

    if os.path.exists(mpath):
        print(f"[SKIP] {name} \u2014 metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)

    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable,
        "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep17_scaling/{name}",
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cell count scaling sweep (sweep 17)",
        epilog="Examples:\n"
               "  python sweep_scaling.py                    # all 28 runs\n"
               "  python sweep_scaling.py --runs A01 B03     # specific runs\n"
               "  python sweep_scaling.py --worker 1 --of 4  # distributed\n"
               "  python sweep_scaling.py --list             # show run names\n"
               "  python sweep_scaling.py --summarize        # just collect results\n",
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
        print(f"\n{len(ALL_RUNS)} sweep 17 scaling runs:")
        for name, cfg in ALL_RUNS.items():
            iters = cfg["iterations"]
            pts = cfg["final_points"]
            interp = "interp" if cfg["interpolation_start"] > 0 else "base"
            densify = "full" if cfg["entropy_fraction"] > 0 else "grad"
            print(f"  {name:40s}  {pts:>7,} cells  {iters:>5} iters  {interp:>6}  {densify}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)

    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        names = [n for n in all_names if n in selected]
        unknown = selected - set(all_names)
        for u in unknown:
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    if args.worker is not None and args.num_workers is not None:
        names = names[args.worker - 1::args.num_workers]
        print(f"Sweep 17: worker {args.worker}/{args.num_workers} \u2014 "
              f"{len(names)} runs: {', '.join(names)}")
    else:
        print(f"Sweep 17: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
