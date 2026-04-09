#!/usr/bin/env python3
"""Sweep 20: Voxel variance regularization — resolution × sigma_v grid + BF combos.

Builds on sweep 19 best result (V03-sv0.5-r32: weight=1e-3, res=32, sigma_v=0.5).
New in this sweep:
  - Resolutions 48 and 128 (filling in and extending beyond 64)
  - sigma_v 0.1, 0.2, 0.3 at the new resolutions (0.5 already established)
  - sigma_v 0.2, 0.3 at res=64 (filling gaps from sweep 19)
  - Bilateral filter (BF) runs: BF alone, BF + vvar, different timing/sigma

All runs use r2_1m baseline: 1M cells, HE 30%, interp at 9k.

Usage:
    python sweep_vvar2.py                    # all runs
    python sweep_vvar2.py --runs W01 W02     # specific runs (prefix match)
    python sweep_vvar2.py --worker 1 --of 4  # distributed
    python sweep_vvar2.py --list             # list all
    python sweep_vvar2.py --summarize        # collect results
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep20_vvar2"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_500_angle_360/0_chest_cone"

# Sweep 19 output dir for cross-reference baselines in summary
SWEEP19_DIR = "output/sweep19_vvar"


def base_config(vvar_weight=1e-3, vvar_resolution=32, vvar_sigma_v=0.5,
                vvar_start=0, bf_start=-1, bf_until=6000, bf_period=10,
                bf_sigma_init=2.0, bf_sigma_final=0.3,
                bf_sigma_v_init=10.0, bf_sigma_v_final=0.1):
    """r2_1m + HE30% base, with vvar and bf overrides."""
    return {
        "iterations": 10000,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": 1000000,
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
        "voxel_var_start": vvar_start,
        "interpolation_start": 9000,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": True,
        "per_neighbor_sigma": True,
        "bf_start": bf_start,
        "bf_until": bf_until,
        "bf_period": bf_period,
        "bf_sigma_init": bf_sigma_init,
        "bf_sigma_final": bf_sigma_final,
        "bf_sigma_v_init": bf_sigma_v_init,
        "bf_sigma_v_final": bf_sigma_v_final,
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

    # ------------------------------------------------------------------
    # W01: New resolutions at sigma_v=0.5 (sweep 19 winner sigma_v)
    # ------------------------------------------------------------------
    runs["W01-r48-sv05"]  = base_config(vvar_resolution=48,  vvar_sigma_v=0.5)
    runs["W01-r128-sv05"] = base_config(vvar_resolution=128, vvar_sigma_v=0.5)

    # ------------------------------------------------------------------
    # W02: sigma_v sweep at res=48 (new resolution)
    # ------------------------------------------------------------------
    for sv, tag in [(0.1, "sv01"), (0.2, "sv02"), (0.3, "sv03")]:
        runs[f"W02-r48-{tag}"] = base_config(vvar_resolution=48, vvar_sigma_v=sv)

    # ------------------------------------------------------------------
    # W03: sigma_v gaps at res=64 (sweep 19 only tested 0.05, 0.1, 0.5)
    # ------------------------------------------------------------------
    for sv, tag in [(0.2, "sv02"), (0.3, "sv03")]:
        runs[f"W03-r64-{tag}"] = base_config(vvar_resolution=64, vvar_sigma_v=sv)

    # ------------------------------------------------------------------
    # W04: sigma_v sweep at res=128
    # ------------------------------------------------------------------
    for sv, tag in [(0.1, "sv01"), (0.2, "sv02"), (0.3, "sv03")]:
        runs[f"W04-r128-{tag}"] = base_config(vvar_resolution=128, vvar_sigma_v=sv)

    # ------------------------------------------------------------------
    # W05: Bilateral filter runs
    # ------------------------------------------------------------------

    # BF alone (no vvar) — is direct BF smoothing competitive?
    # Timing: during densification (3k→6k) so BF guides point placement
    runs["W05-bf-only-dur"] = base_config(
        vvar_weight=0.0,
        bf_start=3000, bf_until=6000,
    )
    # BF alone, post-densification polish (6k→9k)
    runs["W05-bf-only-post"] = base_config(
        vvar_weight=0.0,
        bf_start=6000, bf_until=9000,
    )
    # vvar (best settings) + BF during densification
    runs["W05-vvar+bf-dur"] = base_config(
        vvar_resolution=32, vvar_sigma_v=0.5,
        bf_start=3000, bf_until=6000,
    )
    # vvar (best settings) + BF post-densification
    runs["W05-vvar+bf-post"] = base_config(
        vvar_resolution=32, vvar_sigma_v=0.5,
        bf_start=6000, bf_until=9000,
    )
    # vvar + BF with tighter value sigma (more edge-preserving BF)
    runs["W05-vvar+bf-tight"] = base_config(
        vvar_resolution=32, vvar_sigma_v=0.5,
        bf_start=6000, bf_until=9000,
        bf_sigma_v_init=1.0, bf_sigma_v_final=0.05,
    )

    return runs


ALL_RUNS = build_runs()


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
        "--experiment_name", f"sweep20_vvar2/{name}",
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

    # Include sweep 19 reference runs in summary
    s19_refs = [
        ("s19/baseline",       "baseline"),
        ("s19/V03-sv0.5-r32",  "V03-sv05-r32"),
        ("s19/V04-sv0.05-r64", "V04-sv005-r64"),
        ("s19/V04-sv0.5-r64",  "V04-sv05-r64"),
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
        description="Voxel variance sweep 2 (sweep 20)",
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
        print(f"\n{len(ALL_RUNS)} sweep 20 runs:")
        for name, cfg in ALL_RUNS.items():
            w   = cfg["voxel_var_weight"]
            r   = cfg["voxel_var_resolution"]
            sv  = cfg["voxel_var_sigma_v"]
            bf  = cfg["bf_start"]
            if w > 0:
                desc = f"vvar: w={w} r={r} sv={sv}"
            else:
                desc = "vvar: OFF"
            if bf >= 0:
                desc += f"  BF: {bf}→{cfg['bf_until']} sv_v={cfg['bf_sigma_v_init']}→{cfg['bf_sigma_v_final']}"
            print(f"  {name:30s}  {desc}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        # Support prefix matching (e.g. "W05" matches all W05-* runs)
        names = [n for n in all_names
                 if n in selected or any(n.startswith(s) for s in selected)]
        unknown = [s for s in selected
                   if s not in all_names and not any(n.startswith(s) for n in all_names)]
        for u in unknown:
            print(f"[WARN] Unknown run ID or prefix: {u}")
    else:
        names = all_names

    if args.worker is not None and args.num_workers is not None:
        names = names[args.worker - 1::args.num_workers]
        print(f"Sweep 20: worker {args.worker}/{args.num_workers} — "
              f"{len(names)} runs")
    else:
        print(f"Sweep 20: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
