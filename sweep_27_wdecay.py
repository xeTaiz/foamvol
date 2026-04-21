#!/usr/bin/env python3
"""Sweep 27: Regularization weight decay schedule.

Hypothesis: hold high weight through densification (kills grittiness, enables
good cell allocation + variance-prune), then linearly decay to a low weight
over [densify_until, interpolation_start] = [6000, 9000] so borders can
recover before IDW mode + point freeze.

Weight schedule (new, layered on top of sigma schedule):
  - Iters 0..6000: w = w_init (constant)
  - Iters 6000..9000: w = linear(w_init -> w_final)
  - Iters 9000..10000: w = w_final  (~1000 iters at w_final, incl. freeze)

Variance pruning (adapt 0.08 -> 0.01) on for all runs to match manual best.

15 configs, chest only (75-view, 512k cells). User can extend to pepper.

Usage:
    python sweep_27_wdecay.py --list
    python sweep_27_wdecay.py --worker 1 --of 4
    python sweep_27_wdecay.py --summarize
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from collections import OrderedDict

import yaml

SWEEP_DIR = "output/sweep27_wdecay"
DATA_PATH = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"


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
        "redundancy_cap_init": 0.08,
        "redundancy_cap_final": 0.01,
        "prune_variance_criterion": True,
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
        "voxel_var_weight_final": -1.0,
        "voxel_var_resolution": 64,
        "voxel_var_start": 0,
        "neighbor_var_weight": 0.0,
        "neighbor_var_weight_final": -1.0,
        "neighbor_var_hops": 1,
        "neighbor_var_start": 0,
        "var_sigma_v_init": 50.0,
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
        "data_path": DATA_PATH,
    }
    cfg.update(overrides)
    return cfg


def nvar_decay(w_init, w_final, hops=1):
    return base_config(neighbor_var_weight=w_init,
                       neighbor_var_weight_final=w_final,
                       neighbor_var_hops=hops)


def vvar_decay(w_init, w_final):
    return base_config(voxel_var_weight=w_init,
                       voxel_var_weight_final=w_final)


def both_decay(vw_init, nw_init, vw_final, nw_final, hops=1):
    return base_config(voxel_var_weight=vw_init,
                       voxel_var_weight_final=vw_final,
                       neighbor_var_weight=nw_init,
                       neighbor_var_weight_final=nw_final,
                       neighbor_var_hops=hops)


def build_configs():
    c = OrderedDict()

    # A: nvar 1h, init=3e-2
    c["A01-n1h-3e2-f1e4"] = nvar_decay(3e-2, 1e-4)
    c["A02-n1h-3e2-f1e3"] = nvar_decay(3e-2, 1e-3)
    c["A03-n1h-3e2-f1e2"] = nvar_decay(3e-2, 1e-2)

    # B: nvar 1h, init=1e-1 (breaks when held — does decay save it?)
    c["B01-n1h-1e1-f1e4"] = nvar_decay(1e-1, 1e-4)
    c["B02-n1h-1e1-f1e3"] = nvar_decay(1e-1, 1e-3)
    c["B03-n1h-1e1-f1e2"] = nvar_decay(1e-1, 1e-2)

    # C: vvar r64, init=3e-2
    c["C01-vv-3e2-f1e4"] = vvar_decay(3e-2, 1e-4)
    c["C02-vv-3e2-f1e3"] = vvar_decay(3e-2, 1e-3)
    c["C03-vv-3e2-f1e2"] = vvar_decay(3e-2, 1e-2)

    # D: vvar r64, init=1e-1
    c["D01-vv-1e1-f1e4"] = vvar_decay(1e-1, 1e-4)
    c["D02-vv-1e1-f1e3"] = vvar_decay(1e-1, 1e-3)
    c["D03-vv-1e1-f1e2"] = vvar_decay(1e-1, 1e-2)

    # E: combined v1e-1 + n1e-1 (matches user's manual nw1e1_vw1e1_prun; decay both)
    c["E01-bothe1-f1e4"] = both_decay(1e-1, 1e-1, 1e-4, 1e-4)
    c["E02-bothe1-f1e3"] = both_decay(1e-1, 1e-1, 1e-3, 1e-3)
    c["E03-bothe1-f1e2"] = both_decay(1e-1, 1e-1, 1e-2, 1e-2)

    return c


ALL_CONFIGS = build_configs()
ALL_RUNS = OrderedDict((f"{n}_chest", cfg) for n, cfg in ALL_CONFIGS.items())


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
        "--experiment_name", f"sweep27_wdecay/{name}",
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
    parser = argparse.ArgumentParser()
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
        print(f"\n{len(ALL_RUNS)} runs (chest only):")
        for cfg_name, cfg in ALL_CONFIGS.items():
            nv = cfg["neighbor_var_weight"]
            nvf = cfg["neighbor_var_weight_final"]
            vv = cfg["voxel_var_weight"]
            vvf = cfg["voxel_var_weight_final"]
            parts = []
            if nv > 0:
                parts.append(f"nvar {nv}->{nvf} h={cfg['neighbor_var_hops']}")
            if vv > 0:
                parts.append(f"vvar {vv}->{vvf}")
            parts.append(f"sv={cfg['var_sigma_v_init']}->{cfg['var_sigma_v_final']}")
            parts.append(f"prune-v {cfg['redundancy_cap_init']}->{cfg['redundancy_cap_final']}")
            print(f"  {cfg_name:25s}  {', '.join(parts)}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        names = [n for n in all_names
                 if n in selected or any(n.startswith(s) for s in selected)]
        for s in selected:
            if s not in all_names and not any(n.startswith(s) for n in all_names):
                print(f"[WARN] Unknown run ID or prefix: {s}")
    else:
        names = all_names

    if args.worker is not None:
        names = names[args.worker - 1::args.num_workers]
        print(f"Sweep 27: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 27: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
