#!/usr/bin/env python3
"""Sweep 21: FDK density initialization.

Compares three configurations to isolate the effect of initializing cell
densities from an FDK reconstruction volume instead of uniform init_density:

  F01-baseline  — r2_1m best (HE30%, vvar w=1e-3 r=32 sv=0.5), uniform init
  F02-fdk       — same, but densities initialized from FDK volume
  F03-fdk+vvar  — FDK init + vvar (w=1e-3, r=32, sv=0.5)

FDK volume: DATA_PATH/traditional/fdk/ct_pred.npy  (256³, float32, (X,Y,Z) order)

Usage:
    python sweep_fdk.py                    # all runs
    python sweep_fdk.py --runs F02 F03     # specific runs
    python sweep_fdk.py --list             # list all
    python sweep_fdk.py --summarize        # collect results
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep21_fdk"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_500_angle_360/0_chest_cone"
FDK_PATH = f"{DATA_PATH}/traditional/fdk/ct_pred.npy"

# Sweep 19 baseline for cross-sweep reference in summary
SWEEP19_DIR = "output/sweep19_vvar"


def base_config(vvar_weight=0.0, vvar_resolution=32, vvar_sigma_v=0.5,
                init_volume_path=""):
    """r2_1m + HE30% base."""
    return {
        "iterations": 10000,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": 1000000,
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


ALL_RUNS = {
    # Current best: uniform init, vvar on (same as sweep 19 V03-sv0.5-r32)
    "F01-baseline": base_config(vvar_weight=1e-3),
    # FDK init, no vvar — isolates the init effect
    "F02-fdk":      base_config(vvar_weight=0.0, init_volume_path=FDK_PATH),
    # FDK init + vvar — do they compound?
    "F03-fdk+vvar": base_config(vvar_weight=1e-3, init_volume_path=FDK_PATH),
}


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def metrics_path(name, sweep_dir=None):
    d = sweep_dir or SWEEP_DIR
    return os.path.join(d, name, "metrics.txt")


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
        "--experiment_name", f"sweep21_fdk/{name}",
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

    # Sweep 19 baseline for reference
    s19_refs = [
        ("s19/baseline",      "baseline"),
        ("s19/V03-sv0.5-r32", "V03-sv05-r32"),
    ]
    for display_name, s19_name in s19_refs:
        mpath = metrics_path(s19_name, SWEEP19_DIR)
        if os.path.exists(mpath):
            rows.append({"name": display_name, **parse_metrics(mpath)})

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
        description="FDK initialization sweep (sweep 21)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 21 runs:")
        for name, cfg in ALL_RUNS.items():
            fdk = "FDK init" if cfg["init_volume_path"] else "uniform init"
            vw  = cfg["voxel_var_weight"]
            vvar = f"vvar w={vw}" if vw > 0 else "no vvar"
            print(f"  {name:20s}  {fdk}  {vvar}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        names = [n for n in all_names if n in selected]
        for u in selected - set(all_names):
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    print(f"Sweep 21: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
