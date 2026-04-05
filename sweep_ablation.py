#!/usr/bin/env python3
"""Feature isolation ablation study for CT reconstruction.

Sweep 16: Strips to a naked constant-density Voronoi baseline, then adds
features one at a time (Part A: isolated) and in sensible combinations
(Part B: combos) to measure each feature's individual and joint effects.

Part A: 1 baseline + 44 isolated single-feature runs = 45 runs
Part B: 36 multi-feature combination runs
Grand total: 81 runs

Usage:
    python sweep_ablation.py                              # all 81 runs
    python sweep_ablation.py --runs baseline A01-tv-1e3   # specific runs
    python sweep_ablation.py --worker 1 --of 4            # distributed
    python sweep_ablation.py --list                       # list all
    python sweep_ablation.py --summarize                  # collect results
    python sweep_ablation.py --only isolated              # only Part A
    python sweep_ablation.py --only combos                # only Part B
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Naked baseline — constant density, all features disabled
# ---------------------------------------------------------------------------

BASELINE = {
    # Training
    "iterations": 10000,
    "rays_per_batch": 1000000,
    "init_points": 64000,
    "final_points": 512000,
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
    "freeze_points": 9500,
    "density_grad_clip": 10.0,
    # TV OFF
    "tv_weight": 0.0,
    "tv_start": 0,
    "tv_epsilon": 1e-4,
    "tv_area_weighted": False,
    "tv_border": False,
    "tv_anneal": False,
    "tv_on_raw": True,
    # Densification: gradient-only (simplest)
    "densify_from": 1000,
    "densify_until": 6000,
    "densify_factor": 1.15,
    "gradient_fraction": 1.0,
    "idw_fraction": 0.0,
    "entropy_fraction": 0.0,
    "entropy_bins": 5,
    "contrast_alpha": 0.0,
    # Pruning OFF
    "redundancy_threshold": 0.0,
    "redundancy_cap": 0.0,
    # Targeted sampling OFF
    "targeted_fraction": 0.0,
    "targeted_start": -1,
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
    # Linear Gradient OFF
    "gradient_start": -1,
    "gradient_lr_init": 1e-2,
    "gradient_lr_final": 1e-3,
    "gradient_warmup": 500,
    "gradient_max_slope": 5.0,
    "gradient_freeze_points": 500,
    # Dataset
    "dataset": "r2_gaussian",
    "data_path": "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone",
}

ISOLATED_DIR = "output/sweep16_isolated"
COMBO_DIR = "output/sweep16_combos"

# ---------------------------------------------------------------------------
# Part A: Isolated single-feature ablations (45 runs incl. baseline)
# ---------------------------------------------------------------------------

ISOLATED_RUNS = {
    "baseline": {},

    # --- A. TV Regularization (5 runs) ---
    "A01-tv-1e3":    {"tv_weight": 1e-3},
    "A02-tv-1e2":    {"tv_weight": 1e-2},
    "A03-tv-1e1":    {"tv_weight": 1e-1},
    "A04-tv-area":   {"tv_weight": 1e-2, "tv_area_weighted": True},
    "A05-tv-anneal": {"tv_weight": 1e-2, "tv_anneal": True},

    # --- B. Densification Strategy (8 runs) ---
    "B01-idw30":       {"gradient_fraction": 0.7, "idw_fraction": 0.3},
    "B02-entropy30":   {"gradient_fraction": 0.7, "entropy_fraction": 0.3},
    "B03-both-idw-ent": {"gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3},
    "B04-contrast2":   {"contrast_alpha": 2.0},
    "B05-contrast4":   {"contrast_alpha": 4.0},
    "B06-contrast8":   {"contrast_alpha": 8.0},
    "B07-ent-bins3":   {"gradient_fraction": 0.7, "entropy_fraction": 0.3, "entropy_bins": 3},
    "B08-ent-bins10":  {"gradient_fraction": 0.7, "entropy_fraction": 0.3, "entropy_bins": 10},

    # --- C. Redundancy Pruning (4 runs) ---
    "C01-prune-std":   {"redundancy_threshold": 0.01, "redundancy_cap": 0.05},
    "C02-prune-light": {"redundancy_threshold": 0.005, "redundancy_cap": 0.02},
    "C03-prune-heavy": {"redundancy_threshold": 0.02, "redundancy_cap": 0.10},
    "C04-prune-idw":   {"redundancy_threshold": 0.01, "redundancy_cap": 0.05,
                         "gradient_fraction": 0.7, "idw_fraction": 0.3},

    # --- D. Bilateral Filter (6 runs) ---
    "D01-bf-std":      {"bf_start": 5000, "bf_until": 9000},
    "D02-bf-early":    {"bf_start": 1000, "bf_until": 6000},
    "D03-bf-short":    {"bf_start": 5000, "bf_until": 7000},
    "D04-bf-gentle":   {"bf_start": 5000, "bf_until": 9000,
                         "bf_sigma_v_init": 1.0, "bf_sigma_v_final": 0.01},
    "D05-bf-period5":  {"bf_start": 5000, "bf_until": 9000, "bf_period": 5},
    "D06-bf-period50": {"bf_start": 5000, "bf_until": 9000, "bf_period": 50},

    # --- E. Interpolation Rendering (6 runs) ---
    "E01-interp-std":  {"interpolation_start": 9000},
    "E02-interp-early": {"interpolation_start": 7000},
    "E03-interp-ramp": {"interpolation_start": 9000, "interp_ramp": True},
    "E04-interp-ss10": {"interpolation_start": 9000, "interp_sigma_scale": 1.0},
    "E05-interp-sv05": {"interpolation_start": 9000, "interp_sigma_v": 0.5},
    "E06-interp-sv01": {"interpolation_start": 9000, "interp_sigma_v": 0.1},

    # --- F. Linear Gradient (4 runs) ---
    "F01-grad-std":     {"gradient_start": 5000},
    "F02-grad-early":   {"gradient_start": 2000},
    "F03-grad-slope2":  {"gradient_start": 5000, "gradient_max_slope": 2.0},
    "F04-grad-slope10": {"gradient_start": 5000, "gradient_max_slope": 10.0},

    # --- G. Gaussians (5 runs) ---
    "G01-gauss-std":       {"gaussian_start": 8000},
    "G02-gauss-freeze":    {"gaussian_start": 8000, "freeze_base_at_gaussian": True},
    "G03-gauss-joint":     {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                             "joint_finetune_start": 9500},
    "G04-gauss-early":     {"gaussian_start": 5000},
    "G05-gauss-veryearly": {"gaussian_start": 3000},

    # --- H. Targeted Sampling (3 runs) ---
    "H01-target20": {"targeted_fraction": 0.2},
    "H02-target40": {"targeted_fraction": 0.4},
    "H03-target10": {"targeted_fraction": 0.1},

    # --- I. Loss & LR (3 runs) ---
    "I01-l2-loss":    {"loss_type": "l2"},
    "I02-densLR-low": {"density_lr_init": 1e-2, "density_lr_final": 1e-3},
    "I03-gradclip-1": {"density_grad_clip": 1.0},
}

# ---------------------------------------------------------------------------
# Part B: Multi-feature combinations (36 runs)
# ---------------------------------------------------------------------------

COMBO_RUNS = {
    # --- TV + densification variants (6 runs) ---
    "TV01-tv-idw":          {"tv_weight": 1e-2,
                              "gradient_fraction": 0.7, "idw_fraction": 0.3},
    "TV02-tv-entropy":      {"tv_weight": 1e-2,
                              "gradient_fraction": 0.7, "entropy_fraction": 0.3},
    "TV03-tv-full-densify": {"tv_weight": 1e-2,
                              "gradient_fraction": 0.4, "idw_fraction": 0.3,
                              "entropy_fraction": 0.3, "contrast_alpha": 4.0},
    "TV04-tv-area-anneal":  {"tv_weight": 1e-2, "tv_area_weighted": True, "tv_anneal": True},
    "TV05-tv-prune":        {"tv_weight": 1e-2,
                              "redundancy_threshold": 0.01, "redundancy_cap": 0.05},
    "TV06-tv-border-grad":  {"tv_weight": 1e-2, "tv_border": True, "gradient_start": 5000},

    # --- BF + other features (5 runs) ---
    "BF01-bf-tv":     {"bf_start": 5000, "bf_until": 9000, "tv_weight": 1e-3},
    "BF02-bf-interp": {"bf_start": 5000, "bf_until": 9000, "interpolation_start": 9000},
    "BF03-bf-prune":  {"bf_start": 5000, "bf_until": 9000,
                        "redundancy_threshold": 0.01, "redundancy_cap": 0.05},
    "BF04-bf-grad":   {"bf_start": 5000, "bf_until": 9000, "gradient_start": 5000},
    "BF05-bf-gauss":  {"bf_start": 5000, "bf_until": 8000, "gaussian_start": 8000},

    # --- Interpolation combos (5 runs) ---
    "INT01-interp-tv":           {"interpolation_start": 9000, "tv_weight": 1e-2},
    "INT02-interp-grad":         {"interpolation_start": 9000, "gradient_start": 5000},
    "INT03-interp-prune":        {"interpolation_start": 9000,
                                   "redundancy_threshold": 0.01, "redundancy_cap": 0.05},
    "INT04-interp-full-densify": {"interpolation_start": 9000,
                                   "gradient_fraction": 0.4, "idw_fraction": 0.3,
                                   "entropy_fraction": 0.3, "contrast_alpha": 4.0},
    "INT05-interp-target":       {"interpolation_start": 9000, "targeted_fraction": 0.2},

    # --- Gaussian combos (6 runs) ---
    "GS01-gauss-tv":        {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                              "joint_finetune_start": 9500, "tv_weight": 1e-2},
    "GS02-gauss-bf":        {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                              "joint_finetune_start": 9500,
                              "bf_start": 5000, "bf_until": 8000},
    "GS03-gauss-interp":    {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                              "joint_finetune_start": 9500, "interpolation_start": 9000},
    "GS04-gauss-prune":     {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                              "joint_finetune_start": 9500,
                              "redundancy_threshold": 0.01, "redundancy_cap": 0.05},
    "GS05-gauss-target":    {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                              "joint_finetune_start": 9500, "targeted_fraction": 0.2},
    "GS06-gauss-border-tv": {"gaussian_start": 8000, "freeze_base_at_gaussian": True,
                              "joint_finetune_start": 9500,
                              "tv_weight": 1e-2, "tv_border": True},

    # --- Kitchen sink: replicate existing configs (6 runs) ---
    "KS01-r2fast": {
        "tv_weight": 1e-2,
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "contrast_alpha": 4.0,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
        "interpolation_start": 9000,
    },
    "KS02-r2fast-v2": {
        "final_points": 256000,
        "tv_weight": 1e-2,
        "gradient_fraction": 0.6, "idw_fraction": 0.1, "entropy_fraction": 0.3,
        "contrast_alpha": 2.0,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
        "interpolation_start": 9000,
    },
    "KS03-r2bf": {
        "final_points": 256000,
        "tv_weight": 1e-3,
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "contrast_alpha": 4.0,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
        "interpolation_start": 9000,
        "bf_start": 5000, "bf_until": 9000,
        "bf_sigma_final": 0.5, "bf_sigma_v_init": 3.0, "bf_sigma_v_final": 0.01,
    },
    "KS04-r2gauss": {
        "final_points": 256000,
        "tv_weight": 1e-3,
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "contrast_alpha": 4.0,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
        "bf_start": 5000, "bf_until": 8000,
        "bf_sigma_final": 0.5, "bf_sigma_v_init": 3.0, "bf_sigma_v_final": 0.01,
        "gaussian_start": 8000, "freeze_base_at_gaussian": True,
        "joint_finetune_start": 9500,
    },
    "KS05-all-rendering": {
        "interpolation_start": 9000, "gradient_start": 5000,
        "tv_weight": 1e-2,
        "bf_start": 5000, "bf_until": 9000,
    },
    "KS06-everything": {
        "tv_weight": 1e-2,
        "bf_start": 5000, "bf_until": 9000,
        "interpolation_start": 9000,
        "gradient_start": 5000,
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "contrast_alpha": 4.0,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
        "targeted_fraction": 0.2,
    },

    # --- Gradient combos (4 runs) ---
    "GR01-grad-tv":        {"gradient_start": 5000, "tv_weight": 1e-2},
    "GR02-grad-interp":    {"gradient_start": 5000, "interpolation_start": 9000},
    "GR03-grad-bf":        {"gradient_start": 5000, "bf_start": 5000, "bf_until": 9000},
    "GR04-grad-border-tv": {"gradient_start": 5000, "tv_weight": 1e-2, "tv_border": True},

    # --- Densification combos (4 runs) ---
    "DN01-all-densify-contrast": {
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "contrast_alpha": 4.0,
    },
    "DN02-densify-target": {
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "targeted_fraction": 0.2,
    },
    "DN03-densify-prune": {
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
    },
    "DN04-densify-prune-target": {
        "gradient_fraction": 0.4, "idw_fraction": 0.3, "entropy_fraction": 0.3,
        "redundancy_threshold": 0.01, "redundancy_cap": 0.05,
        "targeted_fraction": 0.2,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_config(overrides):
    """Merge baseline with overrides."""
    cfg = dict(BASELINE)
    cfg.update(overrides)
    return cfg


def metrics_path(sweep_dir, name):
    return os.path.join(sweep_dir, name, "metrics.txt")


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


def run_experiment(name, cfg, sweep_dir):
    """Run a single training experiment via subprocess."""
    out_dir = os.path.join(sweep_dir, name)
    mpath = metrics_path(sweep_dir, name)

    if os.path.exists(mpath):
        print(f"[SKIP] {name} \u2014 metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)

    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    sweep_name = os.path.basename(sweep_dir)
    cmd = [
        sys.executable,
        "train.py",
        "-c", config_file,
        "--experiment_name", f"{sweep_name}/{name}",
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


def collect_summary(names, sweep_dir, output_csv, sort_key="vol_idw_psnr"):
    """Read metrics.txt from each run and write a sorted summary CSV."""
    rows = []
    for name in names:
        mpath = metrics_path(sweep_dir, name)
        if not os.path.exists(mpath):
            continue
        metrics = parse_metrics(mpath)
        rows.append({"name": name, **metrics})

    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    if not rows:
        print(f"[WARN] No completed runs to summarize in {sweep_dir}")
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


def run_partition(runs_dict, sweep_dir, selected_names, summarize):
    """Run experiments for one partition (isolated or combos)."""
    all_names = list(runs_dict.keys())
    names = [n for n in all_names if n in selected_names] if selected_names else all_names

    if not summarize:
        for name in names:
            cfg = build_config(runs_dict[name])
            run_experiment(name, cfg, sweep_dir)

    return collect_summary(all_names, sweep_dir,
                           os.path.join(sweep_dir, "summary.csv"))


def run_sweep(only=None, runs=None, summarize=False, worker=None, num_workers=None):
    """Run all (or selected) sweep 16 experiments."""
    # Build combined name list for sharding and selection
    partitions = []
    if only != "combos":
        partitions.append(("isolated", ISOLATED_RUNS, ISOLATED_DIR))
    if only != "isolated":
        partitions.append(("combos", COMBO_RUNS, COMBO_DIR))

    # Flatten all names (preserving order and partition mapping)
    all_entries = []
    for label, runs_dict, sweep_dir in partitions:
        for name in runs_dict:
            all_entries.append((name, label, runs_dict, sweep_dir))

    # Filter by --runs if specified
    if runs:
        selected = set(runs)
        all_entries = [e for e in all_entries if e[0] in selected]
        unknown = selected - {e[0] for e in all_entries}
        for u in unknown:
            print(f"[WARN] Unknown run ID: {u}")

    # Shard by worker
    if worker is not None and num_workers is not None:
        all_entries = all_entries[worker - 1::num_workers]
        print(f"Sweep 16: worker {worker}/{num_workers} \u2014 "
              f"{len(all_entries)} runs: {', '.join(e[0] for e in all_entries)}")
    else:
        print(f"Sweep 16: {len(all_entries)} runs selected")

    # Run experiments
    if not summarize:
        for name, label, runs_dict, sweep_dir in all_entries:
            cfg = build_config(runs_dict[name])
            run_experiment(name, cfg, sweep_dir)

    # Collect summaries (always over all names in each active partition)
    for label, runs_dict, sweep_dir in partitions:
        csv_path = os.path.join(sweep_dir, "summary.csv")
        collect_summary(list(runs_dict.keys()), sweep_dir, csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Feature isolation ablation study (sweep 16)",
        epilog="Examples:\n"
               "  python sweep_ablation.py                              # all 81 runs\n"
               "  python sweep_ablation.py --runs baseline A01-tv-1e3   # specific runs\n"
               "  python sweep_ablation.py --worker 1 --of 4            # run 1st quarter\n"
               "  python sweep_ablation.py --list                       # show run names\n"
               "  python sweep_ablation.py --summarize                  # just collect results\n"
               "  python sweep_ablation.py --only isolated              # only Part A\n"
               "  python sweep_ablation.py --only combos                # only Part B\n",
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
    parser.add_argument("--only", choices=["isolated", "combos"],
                        help="Only run Part A (isolated) or Part B (combos)")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        if args.only != "combos":
            print(f"\nPart A — Isolated ({len(ISOLATED_RUNS)} runs) [{ISOLATED_DIR}]:")
            for name, overrides in ISOLATED_RUNS.items():
                desc = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "(naked baseline)"
                print(f"  {name:28s}  {desc}")
        if args.only != "isolated":
            print(f"\nPart B — Combinations ({len(COMBO_RUNS)} runs) [{COMBO_DIR}]:")
            for name, overrides in COMBO_RUNS.items():
                desc = ", ".join(f"{k}={v}" for k, v in overrides.items())
                print(f"  {name:28s}  {desc}")
        total = (len(ISOLATED_RUNS) if args.only != "combos" else 0) + \
                (len(COMBO_RUNS) if args.only != "isolated" else 0)
        print(f"\nTotal: {total} runs")
        return

    os.makedirs(ISOLATED_DIR, exist_ok=True)
    os.makedirs(COMBO_DIR, exist_ok=True)
    run_sweep(only=args.only, runs=args.runs, summarize=args.summarize,
              worker=args.worker, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
