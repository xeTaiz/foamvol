#!/usr/bin/env python3
"""Sweep 28: Reference volume sources & hyperparameters on 75-view data.

Tests two ref_volume sources (FDK .npy vs IDW warmup .pt) across weight
magnitudes, decay schedules, edge mask settings, and blur sigmas.

IDW warmup: a 1k-iteration training with no densification is run once
automatically and stored at output/sweep28_refvol/_warmup/model.pt.
load_reference_volume() voxelizes it at 128³ via IDW before downsampling.

Usage:
    python sweep_28_refvol.py --list
    python sweep_28_refvol.py
    python sweep_28_refvol.py --runs V01 V09
    python sweep_28_refvol.py --worker 1 --of 4
    python sweep_28_refvol.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep28_refvol"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
FDK_PATH  = f"{DATA_PATH}/traditional/fdk/ct_pred.npy"
IDW_CKPT  = f"{SWEEP_DIR}/_warmup/model.pt"


def base_config(
    # ref_volume
    ref_volume_path="",
    ref_volume_weight=0.0,
    ref_volume_weight_final=-1.0,
    ref_volume_start=0,
    ref_volume_until=-1,
    ref_volume_resolution=64,
    ref_volume_blur_sigma=2.0,
    ref_volume_edge_mask=False,
    ref_volume_edge_alpha=10.0,
    # densification / pruning (shared with sweep24 best)
    final_points=512000,
    redundancy_cap=0.03,
    redundancy_cap_init=0.0,
    redundancy_cap_final=0.0,
    prune_variance_criterion=True,
    prune_hops=1,
    # regularization
    voxel_var_weight=1e-3,
    voxel_var_resolution=64,
    neighbor_var_weight=1e-3,
    neighbor_var_hops=1,
    var_sigma_v_init=50.0,
    var_sigma_v_final=0.2,
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
        "init_volume_path": "",
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
        "redundancy_cap": redundancy_cap,
        "redundancy_cap_init": redundancy_cap_init,
        "redundancy_cap_final": redundancy_cap_final,
        "prune_variance_criterion": prune_variance_criterion,
        "prune_hops": prune_hops,
        "targeted_fraction": 0.1,
        "targeted_start": -1,
        "high_error_fraction": 0.2,
        "high_error_power": 1.0,
        "high_error_start": -1,
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        "voxel_var_weight": voxel_var_weight,
        "voxel_var_resolution": voxel_var_resolution,
        "voxel_var_start": 0,
        "neighbor_var_weight": neighbor_var_weight,
        "neighbor_var_hops": neighbor_var_hops,
        "neighbor_var_start": 0,
        "var_sigma_v_init": var_sigma_v_init,
        "var_sigma_v_final": var_sigma_v_final,
        "ref_volume_path": ref_volume_path,
        "ref_volume_weight": ref_volume_weight,
        "ref_volume_weight_final": ref_volume_weight_final,
        "ref_volume_start": ref_volume_start,
        "ref_volume_until": ref_volume_until,
        "ref_volume_resolution": ref_volume_resolution,
        "ref_volume_blur_sigma": ref_volume_blur_sigma,
        "ref_volume_edge_mask": ref_volume_edge_mask,
        "ref_volume_edge_alpha": ref_volume_edge_alpha,
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


def fdk(**kwargs):
    kwargs.setdefault("ref_volume_edge_mask", True)
    return base_config(ref_volume_path=FDK_PATH, **kwargs)


def idw(**kwargs):
    kwargs.setdefault("ref_volume_blur_sigma", 0.5)
    kwargs.setdefault("ref_volume_edge_mask", True)
    return base_config(ref_volume_path=IDW_CKPT, **kwargs)


ALL_RUNS = {
    # Baseline: sweep24/27 best settings, no ref_volume
    "V00-baseline":       base_config(),

    # --- FDK source (blur_sigma=2.0 to suppress ringing artifacts) ---
    # Weight magnitude: constant (weight_final=-1 = hold)
    "V01-fdk-1e0":        fdk(ref_volume_weight=1.0),
    "V02-fdk-5e0":        fdk(ref_volume_weight=5.0),
    "V03-fdk-1e-1":       fdk(ref_volume_weight=0.1),

    # Weight decay schedule
    "V04-fdk-1e0-decay":  fdk(ref_volume_weight=1.0,  ref_volume_weight_final=0.1),
    "V05-fdk-5e0-decay":  fdk(ref_volume_weight=5.0,  ref_volume_weight_final=0.5),

    # Edge mask off / gentler alpha
    "V06-fdk-no-edge":    fdk(ref_volume_weight=1.0,  ref_volume_edge_mask=False),
    "V07-fdk-alpha1":     fdk(ref_volume_weight=1.0,  ref_volume_edge_alpha=1.0),

    # Heavier blur
    "V08-fdk-blur4":      fdk(ref_volume_weight=1.0,  ref_volume_blur_sigma=4.0),

    # --- IDW warmup source (blur_sigma=0.5, no FDK artifacts) ---
    # Weight magnitude: constant
    "V09-idw-1e0":        idw(ref_volume_weight=1.0),
    "V10-idw-5e0":        idw(ref_volume_weight=5.0),
    "V11-idw-1e-1":       idw(ref_volume_weight=0.1),

    # Weight decay schedule
    "V12-idw-1e0-decay":  idw(ref_volume_weight=1.0,  ref_volume_weight_final=0.1),

    # Edge mask off
    "V13-idw-no-edge":    idw(ref_volume_weight=1.0,  ref_volume_edge_mask=False),

    # No blur (IDW has no artifacts)
    "V14-idw-blur0":      idw(ref_volume_weight=1.0,  ref_volume_blur_sigma=0.0),
}


# ---------------------------------------------------------------------------
# IDW warmup
# ---------------------------------------------------------------------------

def prepare_idw_warmup():
    """Run 1k-iteration warmup to produce IDW_CKPT if it doesn't exist yet."""
    if os.path.exists(IDW_CKPT):
        print(f"[SKIP] IDW warmup — {IDW_CKPT} already exists")
        return

    warmup_cfg = base_config(
        final_points=64000,
        redundancy_cap=0.0,
        prune_variance_criterion=False,
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
    )
    warmup_cfg.update({
        "iterations": 1000,
        "densify_from": 10000,    # no densification
        "interpolation_start": -1,
        "freeze_points": 10000,
    })

    warmup_dir = os.path.join(SWEEP_DIR, "_warmup")
    os.makedirs(warmup_dir, exist_ok=True)
    config_file = os.path.join(warmup_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(warmup_cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", "sweep28_refvol/_warmup",
    ]
    print("[RUN]  IDW warmup (1k iters, 64k points, no densification)")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[FAIL] IDW warmup exited with code {result.returncode}")
        sys.exit(1)
    if not os.path.exists(IDW_CKPT):
        print(f"[FAIL] IDW warmup finished but {IDW_CKPT} not found")
        sys.exit(1)
    print(f"[DONE] IDW warmup → {IDW_CKPT}")


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

    if cfg.get("ref_volume_path") == IDW_CKPT:
        prepare_idw_warmup()

    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep28_refvol/{name}",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Summary written to {output_csv} ({len(rows)} runs)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Reference volume source & hyperparameter sweep on 75-view data (sweep 28)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", type=int, metavar="W")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 28 runs:")
        for name, cfg in ALL_RUNS.items():
            src   = ("FDK" if cfg["ref_volume_path"] == FDK_PATH else
                     "IDW" if cfg["ref_volume_path"] == IDW_CKPT else
                     "none")
            w     = cfg["ref_volume_weight"]
            wf    = cfg["ref_volume_weight_final"]
            edge  = " edge" if cfg["ref_volume_edge_mask"] else ""
            alpha = f" α={cfg['ref_volume_edge_alpha']}" if cfg["ref_volume_edge_mask"] else ""
            blur  = cfg["ref_volume_blur_sigma"]
            wsched = f"{w}→{wf}" if wf >= 0 else f"{w} (const)"
            wpart  = f" w={wsched}" if w > 0 else ""
            print(f"  {name:28s}  {src:4s}{wpart}  blur={blur}{edge}{alpha}")
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

    if args.worker is not None:
        names = names[args.worker - 1::args.num_workers]
        print(f"Sweep 28: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 28: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
