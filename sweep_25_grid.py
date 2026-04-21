#!/usr/bin/env python3
"""Sweep 25: Comprehensive regularization + pruning grid on 75-view data.

100 configs × 2 datasets (chest, pepper) = 200 runs.
Designed for 8-GPU parallel execution via --worker/--of.

Groups:
  B00       baseline (vvar w=1e-3 r=64)
  A01-A12   nvar weight sweep (1h/2h, no schedule)
  B01-B06   vvar weight sweep (r64, no schedule)
  C01-C18   sigma schedule sweep (nvar 1h/2h, vvar)
  D01-D08   higher weight + schedule
  E01-E10   combined vvar+nvar
  F01-F04   vvar resolution
  G01-G02   nvar 3-hop
  H01-H15   pruning standalone
  I01-I24   regularization + pruning combos

Usage:
    python sweep_25_grid.py --list
    python sweep_25_grid.py --worker 1 --of 8
    python sweep_25_grid.py --summarize
    python sweep_25_grid.py --runs A01_chest A01_pepper
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import OrderedDict

import yaml

SWEEP_DIR = "output/sweep25_grid"

DATASETS = OrderedDict([
    ("chest",  "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"),
    ("pepper", "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"),
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
        "data_path": "",  # filled per dataset
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


def prune(criterion="variance", cap=0.0, cap_init=0.0, cap_final=0.0, hops=1):
    is_var = criterion == "variance"
    return base_config(voxel_var_weight=1e-3,  # baseline reg
                       redundancy_cap=cap,
                       redundancy_cap_init=cap_init,
                       redundancy_cap_final=cap_final,
                       prune_variance_criterion=is_var,
                       prune_hops=hops)


def reg_prune(reg_cfg, criterion="variance", cap=0.0, cap_init=0.0, cap_final=0.0, hops=1):
    """Combine a reg config with pruning settings."""
    is_var = criterion == "variance"
    return {**reg_cfg,
            "redundancy_cap": cap,
            "redundancy_cap_init": cap_init,
            "redundancy_cap_final": cap_final,
            "prune_variance_criterion": is_var,
            "prune_hops": hops}


WEIGHTS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
WEIGHT_LABELS = ["1e4", "5e4", "1e3", "2e3", "5e3", "1e2"]


def build_configs():
    c = OrderedDict()

    # ---- Baseline ----
    c["B00"] = base_config(voxel_var_weight=1e-3)

    # ---- Group A: nvar weight × hops (no schedule) ----
    for w, wl in zip(WEIGHTS, WEIGHT_LABELS):
        c[f"A{WEIGHTS.index(w)+1:02d}-nvar1h-{wl}"] = nvar(w, hops=1)
    for w, wl in zip(WEIGHTS, WEIGHT_LABELS):
        c[f"A{WEIGHTS.index(w)+7:02d}-nvar2h-{wl}"] = nvar(w, hops=2)

    # ---- Group B: vvar weight (no schedule) ----
    for w, wl in zip(WEIGHTS, WEIGHT_LABELS):
        c[f"B{WEIGHTS.index(w)+1:02d}-vvar-{wl}"] = vvar(w)

    # ---- Group C: sigma schedule sweep (weight=1e-3) ----
    # nvar 1h: vary init
    for i, si in enumerate([5, 20, 50, 100]):
        c[f"C{i+1:02d}-n1h-si{si}"] = nvar(1e-3, hops=1, sv_init=si, sv_final=0.2)
    # nvar 1h: vary final (init=50)
    for i, sf in enumerate([0.05, 0.1, 0.5]):
        c[f"C{i+5:02d}-n1h-sf{sf}"] = nvar(1e-3, hops=1, sv_init=50, sv_final=sf)
    # nvar 2h: vary init
    for i, si in enumerate([5, 20, 50, 100]):
        c[f"C{i+8:02d}-n2h-si{si}"] = nvar(1e-3, hops=2, sv_init=si, sv_final=0.2)
    # nvar 2h: vary final (init=50)
    for i, sf in enumerate([0.05, 0.1, 0.5]):
        c[f"C{i+12:02d}-n2h-sf{sf}"] = nvar(1e-3, hops=2, sv_init=50, sv_final=sf)
    # vvar: vary init
    for i, si in enumerate([5, 20, 50, 100]):
        c[f"C{i+15:02d}-vv-si{si}"] = vvar(1e-3, sv_init=si, sv_final=0.2)

    # ---- Group D: higher weight + schedule ----
    c["D01-n1h-5e3-si20"] = nvar(5e-3, hops=1, sv_init=20, sv_final=0.2)
    c["D02-n1h-5e3-si50"] = nvar(5e-3, hops=1, sv_init=50, sv_final=0.2)
    c["D03-n1h-1e2-si20"] = nvar(1e-2, hops=1, sv_init=20, sv_final=0.2)
    c["D04-n1h-1e2-si50"] = nvar(1e-2, hops=1, sv_init=50, sv_final=0.2)
    c["D05-n2h-5e3-si20"] = nvar(5e-3, hops=2, sv_init=20, sv_final=0.2)
    c["D06-n2h-5e3-si50"] = nvar(5e-3, hops=2, sv_init=50, sv_final=0.2)
    c["D07-vv-5e3-si20"]  = vvar(5e-3, sv_init=20, sv_final=0.2)
    c["D08-vv-5e3-si50"]  = vvar(5e-3, sv_init=50, sv_final=0.2)

    # ---- Group E: combined vvar+nvar ----
    c["E01-both-1e3-nosched"]   = both(1e-3, 1e-3, hops=1)
    c["E02-both-1e3-si50"]      = both(1e-3, 1e-3, hops=1, sv_init=50, sv_final=0.2)
    c["E03-both-1e3-2h-si50"]   = both(1e-3, 1e-3, hops=2, sv_init=50, sv_final=0.2)
    c["E04-both-5e4-si50"]      = both(5e-4, 5e-4, hops=1, sv_init=50, sv_final=0.2)
    c["E05-both-5e3-si50"]      = both(5e-3, 5e-3, hops=1, sv_init=50, sv_final=0.2)
    c["E06-v1e3-n5e3-si50"]     = both(1e-3, 5e-3, hops=1, sv_init=50, sv_final=0.2)
    c["E07-v5e3-n1e3-si50"]     = both(5e-3, 1e-3, hops=1, sv_init=50, sv_final=0.2)
    c["E08-both-1e3-si50-sf01"] = both(1e-3, 1e-3, hops=1, sv_init=50, sv_final=0.1)
    c["E09-both-1e3-2h-sf01"]   = both(1e-3, 1e-3, hops=2, sv_init=50, sv_final=0.1)
    c["E10-both-5e3-si50-sf01"] = both(5e-3, 5e-3, hops=1, sv_init=50, sv_final=0.1)

    # ---- Group F: vvar resolution ----
    c["F01-vv-r32-si50"]  = vvar(1e-3, res=32,  sv_init=50, sv_final=0.2)
    c["F02-vv-r48-si50"]  = vvar(1e-3, res=48,  sv_init=50, sv_final=0.2)
    c["F03-vv-r96-si50"]  = vvar(1e-3, res=96,  sv_init=50, sv_final=0.2)
    c["F04-vv-r128-si50"] = vvar(1e-3, res=128, sv_init=50, sv_final=0.2)

    # ---- Group G: nvar 3-hop ----
    c["G01-n3h-nosched"] = nvar(1e-3, hops=3)
    c["G02-n3h-si50"]    = nvar(1e-3, hops=3, sv_init=50, sv_final=0.2)

    # ---- Group H: pruning standalone (baseline reg = vvar 1e-3) ----
    # Fixed cap, variance criterion
    c["H01-pvar-c002"] = prune("variance", cap=0.02)
    c["H02-pvar-c003"] = prune("variance", cap=0.03)
    c["H03-pvar-c005"] = prune("variance", cap=0.05)
    c["H04-pvar-c003-2h"] = prune("variance", cap=0.03, hops=2)
    # Fixed cap, IDW criterion
    c["H05-pidw-c002"] = prune("IDW", cap=0.02)
    c["H06-pidw-c003"] = prune("IDW", cap=0.03)
    c["H07-pidw-c005"] = prune("IDW", cap=0.05)
    # Adaptive cap, variance criterion
    c["H08-pvar-a05-01"]  = prune("variance", cap_init=0.05, cap_final=0.01)
    c["H09-pvar-a08-01"]  = prune("variance", cap_init=0.08, cap_final=0.01)
    c["H10-pvar-a12-02"]  = prune("variance", cap_init=0.12, cap_final=0.02)
    c["H11-pvar-a08-01-2h"] = prune("variance", cap_init=0.08, cap_final=0.01, hops=2)
    # Adaptive cap, IDW criterion
    c["H12-pidw-a05-01"] = prune("IDW", cap_init=0.05, cap_final=0.01)
    c["H13-pidw-a08-01"] = prune("IDW", cap_init=0.08, cap_final=0.01)
    c["H14-pidw-a12-02"] = prune("IDW", cap_init=0.12, cap_final=0.02)
    # Adaptive cap that stops pruning at end
    c["H15-pvar-a08-00"] = prune("variance", cap_init=0.08, cap_final=0.0)

    # ---- Group I: reg + pruning combos ----
    regs = {
        "n1h":  nvar(1e-3, hops=1, sv_init=50, sv_final=0.2),
        "n2h":  nvar(1e-3, hops=2, sv_init=50, sv_final=0.2),
        "n1h5": nvar(5e-3, hops=1, sv_init=50, sv_final=0.2),
        "bn":   both(1e-3, 1e-3, hops=1, sv_init=50, sv_final=0.2),
    }
    prunes = {
        "pv03":   dict(criterion="variance", cap=0.03),
        "pva08":  dict(criterion="variance", cap_init=0.08, cap_final=0.01),
        "pva12":  dict(criterion="variance", cap_init=0.12, cap_final=0.02),
        "pi03":   dict(criterion="IDW", cap=0.03),
        "pia08":  dict(criterion="IDW", cap_init=0.08, cap_final=0.01),
        "pia12":  dict(criterion="IDW", cap_init=0.12, cap_final=0.02),
    }
    idx = 1
    for rname, rcfg in regs.items():
        for pname, pkw in prunes.items():
            c[f"I{idx:02d}-{rname}-{pname}"] = reg_prune(rcfg, **pkw)
            idx += 1

    return c


ALL_CONFIGS = build_configs()

# Expand: each config × each dataset
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
        "--experiment_name", f"sweep25_grid/{name}",
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

    # Union of all keys
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
        description="Comprehensive reg + pruning grid on 75-view data (sweep 25)",
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
        print(f"Sweep 25: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 25: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
