#!/usr/bin/env python3
"""Sweep 19: Voxel variance regularization.

Tests bilateral variance loss at different weights, resolutions, and sigma_v
values. All runs use the r2_1m config (1M cells, interp, full densify, HE 30%)
on the 500-projection dataset.

Usage:
    python sweep_vvar.py                    # all runs
    python sweep_vvar.py --runs V01 V02     # specific runs
    python sweep_vvar.py --worker 1 --of 4  # distributed
    python sweep_vvar.py --list             # list all
    python sweep_vvar.py --summarize        # collect results
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep19_vvar"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_500_angle_360/0_chest_cone"


def make_config(vvar_weight=0.0, vvar_resolution=32, vvar_sigma_v=0.2,
                vvar_start=0, final_points=1000000):
    """Build config based on r2_1m with voxel variance overrides."""
    return {
        "iterations": 10000,
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
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "density_grad_clip": 10.0,
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "freeze_points": 9500,
        "gradient_fraction": 0.4,
        "idw_fraction": 0.3,
        "entropy_fraction": 0.3,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.0,
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
        "voxel_var_resolution": vvar_resolution,
        "voxel_var_sigma_v": vvar_sigma_v,
        "voxel_var_start": vvar_start,
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

    # Baseline (no vvar, but with HE 30%)
    runs["baseline"] = make_config()

    # Vary weight (res=32, sigma_v=0.2)
    for w, tag in [(1e-4, "1e4"), (5e-4, "5e4"), (1e-3, "1e3"),
                   (5e-3, "5e3"), (1e-2, "1e2"), (5e-2, "5e2")]:
        runs[f"V01-w{tag}-r32"] = make_config(vvar_weight=w, vvar_resolution=32)

    # Vary weight (res=64, sigma_v=0.2)
    for w, tag in [(1e-4, "1e4"), (5e-4, "5e4"), (1e-3, "1e3"),
                   (5e-3, "5e3"), (1e-2, "1e2")]:
        runs[f"V02-w{tag}-r64"] = make_config(vvar_weight=w, vvar_resolution=64)

    # Vary sigma_v (res=32, weight=1e-3)
    for sv, tag in [(0.05, "sv005"), (0.1, "sv01"), (0.5, "sv05"), (1.0, "sv10")]:
        runs[f"V03-{tag}-r32"] = make_config(vvar_weight=1e-3, vvar_sigma_v=sv)

    # Vary sigma_v (res=64, weight=1e-3)
    for sv, tag in [(0.05, "sv005"), (0.1, "sv01"), (0.5, "sv05")]:
        runs[f"V04-{tag}-r64"] = make_config(vvar_weight=1e-3, vvar_resolution=64, vvar_sigma_v=sv)

    # Late start (only after densification)
    runs["V05-w1e3-r32-late"] = make_config(vvar_weight=1e-3, vvar_start=6000)
    runs["V05-w1e3-r64-late"] = make_config(vvar_weight=1e-3, vvar_resolution=64, vvar_start=6000)

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
        print(f"[SKIP] {name} \u2014 metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep19_vvar/{name}",
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


def main():
    parser = argparse.ArgumentParser(
        description="Voxel variance sweep (sweep 19)",
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
        print(f"\n{len(ALL_RUNS)} sweep 19 runs:")
        for name, cfg in ALL_RUNS.items():
            w = cfg["voxel_var_weight"]
            r = cfg["voxel_var_resolution"]
            sv = cfg["voxel_var_sigma_v"]
            s = cfg["voxel_var_start"]
            if w > 0:
                desc = f"w={w} r={r} sv={sv}"
                if s > 0:
                    desc += f" start={s}"
            else:
                desc = "(baseline, no vvar)"
            print(f"  {name:30s}  {desc}")
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
        print(f"Sweep 19: worker {args.worker}/{args.num_workers} \u2014 "
              f"{len(names)} runs")
    else:
        print(f"Sweep 19: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
