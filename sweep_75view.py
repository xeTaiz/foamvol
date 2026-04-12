#!/usr/bin/env python3
"""Sweep 22: Anti-grittiness strategies on 75-view data.

Baseline: 1M cells, HE 30%, vvar w=1e-3 r=64 sv=0.2 (current best from sweeps 18-20)
Four strategies tested against the baseline on the 75-view chest cone:

  S1 — Fewer cells     : 512k and 256k (fewer DOF → less room for noise)
  S2 — Pruning         : remove redundant cells during densification
  S3 — Stronger vvar   : weight 1e-2 (10× baseline) and 5e-3 (5×)
  S4 — FDK init        : initialize densities from FDK volume

Combinations: 512k+prune, 512k+vvar-1e2, fdk+vvar to test interactions.

Usage:
    python sweep_75view.py                    # all runs
    python sweep_75view.py --runs S1 S2       # all runs with prefix S1 or S2
    python sweep_75view.py --worker 1 --of 4  # distributed
    python sweep_75view.py --list
    python sweep_75view.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep22_75view"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
FDK_PATH  = f"{DATA_PATH}/traditional/fdk/ct_pred.npy"


def base_config(
    final_points=1000000,
    vvar_weight=1e-3,
    redundancy_threshold=0.0,
    redundancy_cap=0.0,
    init_volume_path="",
):
    return {
        "iterations": 10000,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": final_points,
        "activation_scale": 1.0,
        "init_scale": 1.05,
        "init_type": "random",
        "init_density": 2.0,
        "init_volume_path": init_volume_path,
        "loss_type": "l1",
        "debug": False,
        "viewer": False,
        "save_volume": False,
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "density_grad_clip": 10.0,
        "freeze_points": 9500,
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "gradient_fraction": 0.4,
        "idw_fraction": 0.3,
        "entropy_fraction": 0.3,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        "redundancy_threshold": redundancy_threshold,
        "redundancy_cap": redundancy_cap,
        "targeted_fraction": 0.0,
        "targeted_start": -1,
        "high_error_fraction": 0.3,
        "high_error_power": 1.0,
        "high_error_start": -1,
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        "voxel_var_weight": vvar_weight,
        "voxel_var_resolution": 64,
        "voxel_var_sigma_v": 0.2,
        "voxel_var_start": 0,
        "interpolation_start": 9000,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": True,
        "per_neighbor_sigma": True,
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        "gaussian_start": -1,
        "freeze_base_at_gaussian": False,
        "joint_finetune_start": -1,
        "peak_lr_init": 1e-2,
        "peak_lr_final": 1e-3,
        "offset_lr_init": 1e-3,
        "offset_lr_final": 1e-4,
        "cov_lr_init": 1e-2,
        "cov_lr_final": 1e-3,
        "gradient_start": -1,
        "gradient_lr_init": 1e-2,
        "gradient_lr_final": 1e-3,
        "gradient_warmup": 500,
        "gradient_max_slope": 5.0,
        "gradient_freeze_points": 500,
        "dataset": "r2_gaussian",
        "data_path": DATA_PATH,
    }


def build_runs():
    runs = {}

    # Baseline
    runs["S00-baseline-1M"] = base_config()

    # ---- S1: Fewer cells ----
    runs["S01-512k"]  = base_config(final_points=512000)
    runs["S01-256k"]  = base_config(final_points=256000)

    # ---- S2: Pruning ----
    runs["S02-prune-1M"]   = base_config(redundancy_threshold=0.01, redundancy_cap=0.05)
    runs["S02-prune-512k"] = base_config(final_points=512000,
                                         redundancy_threshold=0.01, redundancy_cap=0.05)

    # ---- S3: Stronger vvar ----
    runs["S03-vvar5e3"]       = base_config(vvar_weight=5e-3)
    runs["S03-vvar1e2"]       = base_config(vvar_weight=1e-2)
    runs["S03-vvar1e2-512k"]  = base_config(final_points=512000, vvar_weight=1e-2)

    # ---- S4: FDK init ----
    runs["S04-fdk"]       = base_config(vvar_weight=0.0,  init_volume_path=FDK_PATH)
    runs["S04-fdk+vvar"]  = base_config(vvar_weight=1e-3, init_volume_path=FDK_PATH)

    return runs


ALL_RUNS = build_runs()


# ---------------------------------------------------------------------------
# Infrastructure
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
        print(f"[SKIP] {name} — metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep22_75view/{name}",
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
        rows.append({"name": name, **parse_metrics(mpath)})

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


def main():
    parser = argparse.ArgumentParser(
        description="Anti-grittiness sweep on 75-view data (sweep 22)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--worker", type=int, metavar="W")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 22 runs:")
        for name, cfg in ALL_RUNS.items():
            pts   = cfg["final_points"] // 1000
            vw    = cfg["voxel_var_weight"]
            rt    = cfg["redundancy_threshold"]
            fdk   = " FDK" if cfg["init_volume_path"] else ""
            prune = f" prune={rt}" if rt > 0 else ""
            print(f"  {name:25s}  {pts}k cells  vvar={vw}{prune}{fdk}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        names = [n for n in all_names
                 if n in selected or any(n.startswith(s) for s in selected)]
        unknown = [s for s in selected
                   if s not in all_names and not any(n.startswith(s) for n in all_names)]
        for u in unknown:
            print(f"[WARN] Unknown run ID or prefix: {u}")
    else:
        names = all_names

    if args.worker is not None:
        names = names[args.worker - 1::args.num_workers]
        print(f"Sweep 22: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 22: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
