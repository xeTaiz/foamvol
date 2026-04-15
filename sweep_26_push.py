#!/usr/bin/env python3
"""Sweep 26: Push regularization weight past sweep 25's ceiling.

Sweep 25 showed weight=1e-2 (the max tested) dominates top PSNR. This sweep
pushes to 2e-2 ... 2e-1 to find where it breaks, and probes variables that
interact with weight (sigma schedule, combined vvar+nvar, pruning combos).

46 configs × 2 datasets (chest, pepper) = 92 runs on 8 GPUs.

Groups:
  W01-W15   weight breaker (nvar 1h/2h, vvar) at 2e-2 ... 2e-1
  S01-S12   sigma schedule × high weight (nvar 1h, w=2e-2/5e-2/1e-1)
  T01-T06   sv_init stretch (nvar 1h, w=5e-2/1e-1)
  C01-C07   combined vvar+nvar — wide range
  P01-P06   high reg + variance pruning

Usage:
    python sweep_26_push.py --list
    python sweep_26_push.py --worker 1 --of 8
    python sweep_26_push.py --summarize
    python sweep_26_push.py --runs W01_chest
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import OrderedDict

import yaml

SWEEP_DIR = "output/sweep26_push"

DATASETS = OrderedDict([
    ("chest",  "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"),
    ("pepper", "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"),
])


def base_config(**overrides):
    cfg = {
        "iterations": 10000,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": 512000,
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
        "redundancy_cap": 0.0,
        "redundancy_cap_init": 0.0,
        "redundancy_cap_final": 0.0,
        "prune_variance_criterion": False,
        "prune_hops": 1,
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
        "voxel_var_weight": 0.0,
        "voxel_var_resolution": 64,
        "voxel_var_start": 0,
        "neighbor_var_weight": 0.0,
        "neighbor_var_hops": 1,
        "neighbor_var_start": 0,
        "var_sigma_v_init": 0.2,
        "var_sigma_v_final": 0.2,
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
        "data_path": "",
    }
    cfg.update(overrides)
    return cfg


def nvar(w, hops=1, sv_init=0.2, sv_final=0.2):
    return base_config(neighbor_var_weight=w, neighbor_var_hops=hops,
                       var_sigma_v_init=sv_init, var_sigma_v_final=sv_final)


def vvar(w, res=64, sv_init=0.2, sv_final=0.2):
    return base_config(voxel_var_weight=w, voxel_var_resolution=res,
                       var_sigma_v_init=sv_init, var_sigma_v_final=sv_final)


def both(vw, nw, hops=1, res=64, sv_init=0.2, sv_final=0.2):
    return base_config(voxel_var_weight=vw, voxel_var_resolution=res,
                       neighbor_var_weight=nw, neighbor_var_hops=hops,
                       var_sigma_v_init=sv_init, var_sigma_v_final=sv_final)


def reg_prune(reg_cfg, criterion="variance", cap=0.0, cap_init=0.0, cap_final=0.0, hops=1):
    is_var = criterion == "variance"
    return {**reg_cfg,
            "redundancy_cap": cap,
            "redundancy_cap_init": cap_init,
            "redundancy_cap_final": cap_final,
            "prune_variance_criterion": is_var,
            "prune_hops": hops}


HIGH_WEIGHTS = [2e-2, 3e-2, 5e-2, 1e-1, 2e-1]
HW_LABELS    = ["2e2", "3e2", "5e2", "1e1", "2e1"]


def build_configs():
    c = OrderedDict()

    # ---- Group W: weight breaker, no schedule (sv=0.2) ----
    for i, (w, wl) in enumerate(zip(HIGH_WEIGHTS, HW_LABELS)):
        c[f"W{i+1:02d}-nvar1h-{wl}"] = nvar(w, hops=1)
    for i, (w, wl) in enumerate(zip(HIGH_WEIGHTS, HW_LABELS)):
        c[f"W{i+6:02d}-nvar2h-{wl}"] = nvar(w, hops=2)
    for i, (w, wl) in enumerate(zip(HIGH_WEIGHTS, HW_LABELS)):
        c[f"W{i+11:02d}-vvar-{wl}"] = vvar(w)

    # ---- Group S: schedule × high weight (nvar 1h, sv_init=50) ----
    s_weights = [(2e-2, "2e2"), (5e-2, "5e2"), (1e-1, "1e1")]
    s_finals  = [0.05, 0.1, 0.2, 0.3]
    idx = 1
    for w, wl in s_weights:
        for sf in s_finals:
            c[f"S{idx:02d}-n1h-{wl}-sf{sf}"] = nvar(w, hops=1, sv_init=50, sv_final=sf)
            idx += 1

    # ---- Group T: sv_init stretch (nvar 1h, sv_final=0.2) ----
    t_weights = [(5e-2, "5e2"), (1e-1, "1e1")]
    t_inits   = [20, 100, 200]
    idx = 1
    for w, wl in t_weights:
        for si in t_inits:
            c[f"T{idx:02d}-n1h-{wl}-si{si}"] = nvar(w, hops=1, sv_init=si, sv_final=0.2)
            idx += 1

    # ---- Group C: combined vvar+nvar, wide range ----
    c["C01-both-v1e2-n1e2"]        = both(1e-2, 1e-2, hops=1, sv_init=50, sv_final=0.2)
    c["C02-both-v2e2-n5e2"]        = both(2e-2, 5e-2, hops=1, sv_init=50, sv_final=0.2)
    c["C03-both-v5e2-n2e2"]        = both(5e-2, 2e-2, hops=1, sv_init=50, sv_final=0.2)
    c["C04-both-v5e2-n5e2"]        = both(5e-2, 5e-2, hops=1, sv_init=50, sv_final=0.2)
    c["C05-both-v1e1-n1e1"]        = both(1e-1, 1e-1, hops=1, sv_init=50, sv_final=0.2)
    c["C06-both-v2e1-n2e1"]        = both(2e-1, 2e-1, hops=1, sv_init=50, sv_final=0.2)
    c["C07-both-v5e2-n5e2-sf01"]   = both(5e-2, 5e-2, hops=1, sv_init=50, sv_final=0.1)

    # ---- Group P: best reg + variance pruning ----
    p_base_2e2 = nvar(2e-2, hops=1, sv_init=50, sv_final=0.2)
    p_base_5e2 = nvar(5e-2, hops=1, sv_init=50, sv_final=0.2)
    p_base_1e1 = nvar(1e-1, hops=1, sv_init=50, sv_final=0.2)
    p_base_5e2_2h = nvar(5e-2, hops=2, sv_init=50, sv_final=0.2)

    c["P01-n1h-2e2-pva08"]    = reg_prune(p_base_2e2, "variance", cap_init=0.08, cap_final=0.01)
    c["P02-n1h-5e2-pva08"]    = reg_prune(p_base_5e2, "variance", cap_init=0.08, cap_final=0.01)
    c["P03-n1h-1e1-pva08"]    = reg_prune(p_base_1e1, "variance", cap_init=0.08, cap_final=0.01)
    c["P04-n1h-5e2-pva12"]    = reg_prune(p_base_5e2, "variance", cap_init=0.12, cap_final=0.02)
    c["P05-n2h-5e2-pva08"]    = reg_prune(p_base_5e2_2h, "variance", cap_init=0.08, cap_final=0.01)
    c["P06-n1h-5e2-pv03"]     = reg_prune(p_base_5e2, "variance", cap=0.03)

    return c


ALL_CONFIGS = build_configs()

ALL_RUNS = OrderedDict()
for ds_name, ds_path in DATASETS.items():
    for cfg_name, cfg in ALL_CONFIGS.items():
        ALL_RUNS[f"{cfg_name}_{ds_name}"] = {**cfg, "data_path": ds_path}


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
        "--experiment_name", f"sweep26_push/{name}",
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

    if not rows:
        print("[WARN] No completed runs to summarize")
        return rows

    seen = set()
    all_keys = []
    for row in rows:
        for k in row:
            if k != "name" and k not in seen:
                all_keys.append(k)
                seen.add(k)

    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
    fieldnames = ["name"] + all_keys

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Summary written to {output_csv} ({len(rows)} runs)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Push regularization weight past sweep 25's ceiling (sweep 26)",
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
        print(f"\n{len(ALL_CONFIGS)} configs × {len(DATASETS)} datasets "
              f"= {len(ALL_RUNS)} runs:")
        for cfg_name, cfg in ALL_CONFIGS.items():
            nv = cfg["neighbor_var_weight"]
            vv = cfg["voxel_var_weight"]
            si = cfg["var_sigma_v_init"]
            sf = cfg["var_sigma_v_final"]
            hp = cfg["neighbor_var_hops"]
            cap = cfg["redundancy_cap"]
            ci = cfg["redundancy_cap_init"]
            cf = cfg["redundancy_cap_final"]
            pv = cfg["prune_variance_criterion"]
            res = cfg["voxel_var_resolution"]

            parts = []
            if nv > 0:
                parts.append(f"nvar={nv} h={hp}")
            if vv > 0:
                parts.append(f"vvar={vv} r={res}")
            if si != sf:
                parts.append(f"sv={si}→{sf}")
            else:
                parts.append(f"sv={sf}")
            if ci > 0 or cf > 0:
                parts.append(f"cap={ci}→{cf}")
            elif cap > 0:
                parts.append(f"cap={cap}")
            if pv:
                parts.append("var-prune")
            if not parts:
                parts.append("no reg/prune")
            print(f"  {cfg_name:30s}  {', '.join(parts)}")
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
        print(f"Sweep 26: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 26: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
