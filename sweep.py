#!/usr/bin/env python3
"""Hyperparameter sweep for CT reconstruction.

Sweep 8: Hyperparameter Screening & Refinement.
Phase 1: One-at-a-time screening (25 runs).
Phase 2: Combine best per-axis settings + ablation (~4-8 runs).

Usage:
    python sweep.py --phase 1
    python sweep.py --phase 2
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Baseline (matches r2fast.yaml)
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
    "interp_sigma_scale": 0.55,
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
    "density_lr_init": 1e-1,
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

SWEEP_NAME = "sweep8"
SWEEP_DIR = f"output/{SWEEP_NAME}"

# ---------------------------------------------------------------------------
# Phase 1: One-at-a-time screening (25 runs)
# ---------------------------------------------------------------------------

PHASE1_RUNS = {
    "baseline": {},
    # Axis A: Learning rates
    "A1-dli5e2": {"density_lr_init": 5e-2},
    "A2-dli2e1": {"density_lr_init": 2e-1},
    "A3-dli5e1": {"density_lr_init": 5e-1},
    "A4-dlf5e3": {"density_lr_final": 5e-3},
    "A5-dlf2e2": {"density_lr_final": 2e-2},
    "A6-pli1e4": {"points_lr_init": 1e-4},
    "A7-pli5e4": {"points_lr_init": 5e-4},
    "A8-pli1e3": {"points_lr_init": 1e-3},
    # Axis B: Schedule timing
    "B1-frz8k": {"freeze_points": 8000},
    "B2-frz99": {"freeze_points": 9900},
    "B3-den45": {"densify_until": 4500},
    "B4-den75": {"densify_until": 7500},
    "B5-int75": {"interpolation_start": 7500},
    "B6-intOff": {"interpolation_start": -1},
    # Axis C: TV regularization
    "C1-tv0": {"tv_weight": 0.0},
    "C2-tv1e5": {"tv_weight": 1e-5},
    "C3-tv1e3": {"tv_weight": 1e-3},
    # Axis D: Interpolation sigmas
    "D1-ss03": {"interp_sigma_scale": 0.3},
    "D2-ss04": {"interp_sigma_scale": 0.4},
    "D3-ss07": {"interp_sigma_scale": 0.7},
    "D4-ss09": {"interp_sigma_scale": 0.9},
    "D5-sv005": {"interp_sigma_v": 0.05},
    "D6-sv01": {"interp_sigma_v": 0.1},
    "D7-sv04": {"interp_sigma_v": 0.4},
}

# Map each run to its axis for Phase 2 analysis
AXIS_MAP = {}
for run_id in PHASE1_RUNS:
    if run_id == "baseline":
        AXIS_MAP[run_id] = "baseline"
    else:
        AXIS_MAP[run_id] = run_id[0]  # 'A', 'B', 'C', 'D'

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


def load_phase1_summary():
    """Load Phase 1 summary CSV and return rows as list of dicts."""
    csv_path = os.path.join(SWEEP_DIR, "summary_phase1.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Phase 1 summary not found: {csv_path}")
        print("        Run --phase 1 first.")
        sys.exit(1)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert numeric fields
    for row in rows:
        for k, v in row.items():
            if k == "name":
                continue
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def filter_runs(names, runs=None, axes=None):
    """Filter run names by explicit IDs and/or axis letters."""
    if runs is None and axes is None:
        return names
    selected = set()
    if runs:
        for r in runs:
            if r in PHASE1_RUNS:
                selected.add(r)
            else:
                print(f"[WARN] Unknown run ID: {r}")
    if axes:
        for name in names:
            axis = AXIS_MAP.get(name, "")
            if axis in axes or name in axes:
                selected.add(name)
    # Preserve original order
    return [n for n in names if n in selected]


def phase1(runs=None, axes=None, summarize=False):
    """Run one-at-a-time screening experiments.

    Args:
        runs: Optional list of specific run IDs to execute.
        axes: Optional list of axis letters (A/B/C/D) or 'baseline'.
        summarize: If True, only collect summary from existing results.
    """
    all_names = list(PHASE1_RUNS.keys())
    names = filter_runs(all_names, runs, axes)
    print(f"Phase 1: {len(names)}/{len(all_names)} runs selected")

    if not summarize:
        for name in names:
            cfg = build_config(PHASE1_RUNS[name])
            run_experiment(name, cfg)

    # Summary always covers all available results
    return collect_summary(all_names, os.path.join(SWEEP_DIR, "summary_phase1.csv"))


# ---------------------------------------------------------------------------
# Phase 2: Combine best per-axis + ablations
# ---------------------------------------------------------------------------


def find_best_per_axis(rows):
    """For each axis, find the run that beats baseline on vol_idw_psnr.

    Returns dict: axis_letter -> (run_id, overrides_dict, psnr_delta).
    Only includes axes where the best variant beats baseline.
    """
    baseline_row = None
    for row in rows:
        if row["name"] == "baseline":
            baseline_row = row
            break

    if baseline_row is None:
        print("[ERROR] No baseline run found in Phase 1 results")
        sys.exit(1)

    baseline_psnr = baseline_row.get("vol_idw_psnr", 0)
    print(f"\nBaseline vol_idw_psnr: {baseline_psnr:.4f}")

    # Group runs by axis
    axes = {}  # axis_letter -> list of (run_id, psnr)
    for row in rows:
        run_id = row["name"]
        if run_id == "baseline":
            continue
        axis = AXIS_MAP.get(run_id, "?")
        psnr = row.get("vol_idw_psnr", 0)
        axes.setdefault(axis, []).append((run_id, psnr))

    winners = {}
    for axis in sorted(axes.keys()):
        best_id, best_psnr = max(axes[axis], key=lambda x: x[1])
        delta = best_psnr - baseline_psnr
        status = "BETTER" if delta > 0 else "worse"
        print(f"  Axis {axis}: best={best_id} psnr={best_psnr:.4f} (delta={delta:+.4f}) [{status}]")
        if delta > 0:
            winners[axis] = (best_id, PHASE1_RUNS[best_id], delta)

    return winners, baseline_psnr


def phase2():
    """Combine best per-axis winners and run ablations."""
    rows = load_phase1_summary()
    winners, baseline_psnr = find_best_per_axis(rows)

    if not winners:
        print("\n[INFO] No axis improved over baseline. Nothing to combine.")
        return

    print(f"\n--- Phase 2: Combining {len(winners)} winning axes ---")

    # Build combined config
    combined_overrides = {}
    for axis, (run_id, overrides, delta) in sorted(winners.items()):
        print(f"  Including {run_id}: {overrides} (delta={delta:+.4f})")
        combined_overrides.update(overrides)

    # Phase 2 runs: combined + ablation of each winner
    phase2_runs = {"P2-combined": combined_overrides}

    # Ablation: combined minus each axis
    if len(winners) > 1:
        for axis, (run_id, overrides, _) in sorted(winners.items()):
            ablation_overrides = {k: v for k, v in combined_overrides.items()
                                  if k not in overrides}
            phase2_runs[f"P2-no{axis}"] = ablation_overrides

    # If both sigma_scale and sigma_v won, add 2x2 mini-grid
    if "D" in winners:
        d_winner_id = winners["D"][0]
        d_overrides = winners["D"][1]
        # Check if there's a sigma_scale winner and a sigma_v winner
        # Find the second-best D variant of the other type
        d_scale_runs = [r for r in rows if r["name"].startswith("D") and "ss" in r["name"]]
        d_sv_runs = [r for r in rows if r["name"].startswith("D") and "sv" in r["name"]]

        if d_scale_runs and d_sv_runs:
            best_scale = max(d_scale_runs, key=lambda r: r.get("vol_idw_psnr", 0))
            best_sv = max(d_sv_runs, key=lambda r: r.get("vol_idw_psnr", 0))

            if (best_scale["name"] != "baseline" and best_sv["name"] != "baseline"
                    and best_scale.get("vol_idw_psnr", 0) > baseline_psnr
                    and best_sv.get("vol_idw_psnr", 0) > baseline_psnr):
                # Both sigma types beat baseline — add combined D variant
                scale_overrides = PHASE1_RUNS[best_scale["name"]]
                sv_overrides = PHASE1_RUNS[best_sv["name"]]
                combo_d = {**scale_overrides, **sv_overrides}
                # Only add if it differs from what's already in combined
                if combo_d != {k: v for k, v in combined_overrides.items()
                               if k in ("interp_sigma_scale", "interp_sigma_v")}:
                    phase2_runs["P2-Dcombo"] = {
                        **{k: v for k, v in combined_overrides.items()
                           if k not in ("interp_sigma_scale", "interp_sigma_v")},
                        **combo_d,
                    }

    print(f"\nPhase 2: {len(phase2_runs)} runs")
    names = list(phase2_runs.keys())

    for name in names:
        cfg = build_config(phase2_runs[name])
        run_experiment(name, cfg)

    all_names = ["baseline"] + names
    return collect_summary(all_names, os.path.join(SWEEP_DIR, "summary_phase2.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CT reconstruction hyperparameter sweep (sweep 8)",
        epilog="Examples:\n"
               "  python sweep.py --phase 1                    # all 25 runs\n"
               "  python sweep.py --phase 1 --axis A B         # axes A+B (9 runs)\n"
               "  python sweep.py --phase 1 --axis C D baseline  # axes C+D + baseline\n"
               "  python sweep.py --phase 1 --runs A1-dli5e2 A2-dli2e1  # specific runs\n"
               "  python sweep.py --phase 1 --summarize        # just collect results\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1 = one-at-a-time screening, 2 = combine winners")
    parser.add_argument("--axis", nargs="+", metavar="X",
                        help="Phase 1: only run these axes (A/B/C/D/baseline)")
    parser.add_argument("--runs", nargs="+", metavar="ID",
                        help="Phase 1: only run these specific run IDs")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip training, just collect existing results into summary CSV")
    args = parser.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)

    if args.phase == 1:
        phase1(runs=args.runs, axes=args.axis, summarize=args.summarize)
    elif args.phase == 2:
        phase2()


if __name__ == "__main__":
    main()
