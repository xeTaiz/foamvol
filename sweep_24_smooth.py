#!/usr/bin/env python3
"""Sweep 24: Graph neighbor variance + adaptive pruning on 75-view data.

Baseline: 512k cells, vvar w=1e-3 r=64 sv=0.2, HE=0.2 tgt=0.1 (sweep22 best).

Tests four strategies:
  V00  baseline (sweep22 S01-512k config)
  V01  graph neighbor variance, 1-hop, no sigma schedule
  V02  graph neighbor variance, 1-hop, sigma schedule (init=100 → final=0.2)
  V03  graph neighbor variance, 2-hop, sigma schedule
  V04  existing voxel var + sigma schedule
  V05  variance-based pruning, fixed cap=0.03
  V06  adaptive pruning cap (IDW criterion): cap_init=0.08 → cap_final=0.01
  V07  best regularization + best pruning combo (V02 + V06)

Usage:
    python sweep_24_smooth.py
    python sweep_24_smooth.py --runs V01 V02
    python sweep_24_smooth.py --list
    python sweep_24_smooth.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep24_smooth"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"


def base_config(
    final_points=512000,
    voxel_var_weight=1e-3,
    voxel_var_resolution=64,
    var_sigma_v_init=0.2,
    var_sigma_v_final=0.2,
    neighbor_var_weight=0.0,
    neighbor_var_hops=1,
    redundancy_cap=0.0,
    redundancy_cap_init=0.0,
    redundancy_cap_final=0.0,
    prune_variance_criterion=False,
    prune_hops=1,
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
    # Baseline: sweep22 S01-512k with T01 sampling settings
    "V00-baseline-512k": base_config(),

    # Graph neighbor variance, no schedule (matches current vvar behavior but graph-based)
    "V01-nvar-1h":        base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                      neighbor_var_hops=1),

    # Graph neighbor variance + sigma annealing (strong smoothing early)
    "V02-nvar-1h-sched":  base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                      neighbor_var_hops=1,
                                      var_sigma_v_init=100.0, var_sigma_v_final=0.2),

    # 2-hop graph neighbor variance + sigma annealing (broader neighborhood)
    "V03-nvar-2h-sched":  base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                      neighbor_var_hops=2,
                                      var_sigma_v_init=100.0, var_sigma_v_final=0.2),

    # Existing voxel var with sigma annealing
    "V04-vvar-sched":     base_config(voxel_var_weight=1e-3,
                                      var_sigma_v_init=100.0, var_sigma_v_final=0.2),

    # Variance-based pruning, fixed cap
    "V05-prune-var":      base_config(redundancy_cap=0.03,
                                      prune_variance_criterion=True, prune_hops=1),

    # Adaptive cap schedule (IDW criterion)
    "V06-prune-adapt":    base_config(redundancy_cap_init=0.08, redundancy_cap_final=0.01),

    # Combo: graph neighbor var + sigma schedule + adaptive cap
    "V07-combo":          base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                      neighbor_var_hops=1,
                                      var_sigma_v_init=100.0, var_sigma_v_final=0.2,
                                      redundancy_cap_init=0.08, redundancy_cap_final=0.01),

    # ---- Extended: sigma_v_final sweep ----
    # Does smaller final sigma help? (tighter edge-preservation)
    "V08-nvar-sched-sv01": base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                       neighbor_var_hops=1,
                                       var_sigma_v_init=50.0, var_sigma_v_final=0.1),
    # Larger final sigma (gentler edge-preservation)
    "V09-nvar-sched-sv05": base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                       neighbor_var_hops=1,
                                       var_sigma_v_init=50.0, var_sigma_v_final=0.5),

    # ---- Extended: stronger regularization weight with schedule ----
    "V10-nvar-5e3-sched": base_config(voxel_var_weight=0.0, neighbor_var_weight=5e-3,
                                      neighbor_var_hops=1,
                                      var_sigma_v_init=50.0, var_sigma_v_final=0.2),
    "V11-nvar-1e2-sched": base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-2,
                                      neighbor_var_hops=1,
                                      var_sigma_v_init=50.0, var_sigma_v_final=0.2),

    # ---- Extended: 2-hop without schedule (isolate hop effect) ----
    "V12-nvar-2h":        base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                      neighbor_var_hops=2),

    # ---- Extended: pruning combos ----
    # Variance criterion + adaptive cap (closes the gap between V05 and V06)
    "V13-prune-var-adapt": base_config(redundancy_cap_init=0.08, redundancy_cap_final=0.01,
                                       prune_variance_criterion=True, prune_hops=1),
    # More aggressive adaptive cap
    "V14-prune-adapt-hi": base_config(redundancy_cap_init=0.12, redundancy_cap_final=0.02),

    # ---- Extended: full combos ----
    # Best reg + variance prune + adaptive cap
    "V15-combo-full":     base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                      neighbor_var_hops=1,
                                      var_sigma_v_init=50.0, var_sigma_v_final=0.2,
                                      redundancy_cap_init=0.08, redundancy_cap_final=0.01,
                                      prune_variance_criterion=True, prune_hops=1),
    # Stronger reg + pruning
    "V16-combo-strong":   base_config(voxel_var_weight=0.0, neighbor_var_weight=5e-3,
                                      neighbor_var_hops=1,
                                      var_sigma_v_init=50.0, var_sigma_v_final=0.2,
                                      redundancy_cap_init=0.08, redundancy_cap_final=0.01,
                                      prune_variance_criterion=True, prune_hops=1),
}


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
        "--experiment_name", f"sweep24_smooth/{name}",
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
        description="Graph variance + adaptive pruning sweep on 75-view data (sweep 24)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 24 runs:")
        for name, cfg in ALL_RUNS.items():
            pts   = cfg["final_points"] // 1000
            nv    = cfg["neighbor_var_weight"]
            vv    = cfg["voxel_var_weight"]
            sv_i  = cfg["var_sigma_v_init"]
            sv_f  = cfg["var_sigma_v_final"]
            hops  = cfg["neighbor_var_hops"]
            cap   = cfg["redundancy_cap"]
            cap_i = cfg["redundancy_cap_init"]
            cap_f = cfg["redundancy_cap_final"]
            pvar  = " var-prune" if cfg["prune_variance_criterion"] else ""
            reg   = (f"nvar_w={nv} hops={hops}" if nv > 0 else
                     f"vvar_w={vv}" if vv > 0 else "no reg")
            sched = f" sv={sv_i}→{sv_f}" if sv_i != sv_f else f" sv={sv_i}"
            pruning = (f" cap={cap_i}→{cap_f}" if cap_i > 0 or cap_f > 0 else
                       f" cap={cap}" if cap > 0 else "")
            print(f"  {name:25s}  {pts}k  {reg}{sched}{pruning}{pvar}")
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

    print(f"Sweep 24: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
