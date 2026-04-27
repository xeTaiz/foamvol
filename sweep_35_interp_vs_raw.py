#!/usr/bin/env python3
"""Sweep 35: Interp training vs. raw-only — do we need interpolation_start?

Motivation:
  Interp training (interpolation_start=9000) routes density gradients to all
  Delaunay neighbors via ct_interp_backward. Combined with the per-ray CT
  loss ambiguity, this injects noise into raw cell densities. The IDW metrics
  improve slightly, but raw densities are corrupted and locked to a specific σ.

  Hypothesis: train pure raw (interpolation_start=-1), evaluate with IDW
  post-hoc at chosen σ — matches or beats interp-trained IDW quality with
  cleaner raw densities and σ-free training.

  Two paired runs per dataset, identical except interpolation_start:
    interp: interpolation_start=9000  (CUDA IDW kernel active in training)
    raw:    interpolation_start=-1    (constant per-cell density throughout)

  All σ-dependent training features stripped (no vvar, no redundancy pruning,
  no TV, no BF). Grad smooth hops=1 kept (σ-free). rays_per_batch_late=4M.
  Follow-up: eval_sigma_sweep.py sweeps σ on the raw checkpoints.

Datasets: chest-D and pepper-D at 512k points.
Total: 2 datasets × 2 conditions = 4 runs.

Usage:
    python sweep_35_interp_vs_raw.py --list
    python sweep_35_interp_vs_raw.py
    python sweep_35_interp_vs_raw.py --runs pepper-raw-512k chest-interp-512k
    python sweep_35_interp_vs_raw.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR    = "output/sweep35_interp_vs_raw"
CHEST_DATA   = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
PEPPER_DATA  = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"


def base_config(
    data_path=CHEST_DATA,
    interpolation_start=9000,
):
    return {
        # Training
        "iterations": 10000,
        "rays_per_batch": 1000000,
        "rays_per_batch_late": 4000000,
        "rays_per_batch_late_start": 9500,
        "init_points": 64000,
        "final_points": 512000,
        "activation_scale": 1.0,
        "init_scale": 1.05,
        "init_type": "random",
        "init_density": 2.0,
        "init_volume_path": "",
        "device": "cuda",
        "debug": False,
        "viewer": False,
        "save_volume": False,
        "dataset": "r2_gaussian",
        "data_path": data_path,
        # Loss
        "loss_type": "l1",
        # Optimization
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "density_grad_clip": 10.0,
        "freeze_points": 9500,
        # Densification
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "gradient_fraction": 0.4,
        "idw_fraction": 0.3,
        "entropy_fraction": 0.3,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        # Pruning — basic prune only (no IDW redundancy pruning)
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.0,
        "redundancy_cap_init": 0.0,
        "redundancy_cap_final": 0.0,
        "prune_variance_criterion": False,
        "prune_hops": 1,
        "ref_guided_pruning": False,
        "ref_guided_densify": False,
        "ref_guided_eps": 0.01,
        # Ray sampling
        "targeted_fraction": 0.1,
        "targeted_start": -1,
        "high_error_fraction": 0.2,
        "high_error_power": 1.0,
        "high_error_start": -1,
        # TV (off)
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        # Voxel variance (off)
        "voxel_var_weight": 0.0,
        "voxel_var_weight_final": -1.0,
        "voxel_var_resolution": 64,
        "voxel_var_start": 0,
        "voxel_var_supersample": 4,
        # Neighbor variance (off)
        "neighbor_var_weight": 0.0,
        "neighbor_var_weight_final": -1.0,
        "neighbor_var_hops": 1,
        "neighbor_var_start": 0,
        "neighbor_reg_type": "bilateral_var",
        "neighbor_huber_delta": 0.1,
        # Sigma schedule (unused since vvar is off)
        "var_sigma_v_init": 50.0,
        "var_sigma_v_final": 0.2,
        # Reference volume (off)
        "ref_volume_path": "",
        "ref_volume_weight": 0.0,
        "ref_volume_weight_final": -1.0,
        "ref_volume_start": 0,
        "ref_volume_until": -1,
        "ref_volume_resolution": 64,
        "ref_volume_blur_sigma": 0.0,
        "ref_volume_edge_mask": True,
        "ref_volume_edge_alpha": 10.0,
        "ref_volume_supersample": 4,
        # Grad smoothing (σ-free; keep)
        "grad_smooth_hops": 1,
        # Interpolation
        "interpolation_start": interpolation_start,
        "interp_ramp": False,
        "interp_sigma_abs": 0.0,          # use scale × median path
        "interp_sigma_scale": 1.5,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": False,
        "per_neighbor_sigma": False,
        # BF (off)
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        # Gaussians / gradient field (off)
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
        # Logging
        "log_percent": 5,
        "diag_percent": 10,
    }


ALL_RUNS = {
    "pepper-interp-512k": base_config(data_path=PEPPER_DATA, interpolation_start=9000),
    "pepper-raw-512k":    base_config(data_path=PEPPER_DATA, interpolation_start=-1),
    "chest-interp-512k":  base_config(data_path=CHEST_DATA,  interpolation_start=9000),
    "chest-raw-512k":     base_config(data_path=CHEST_DATA,  interpolation_start=-1),
}


# ── Infrastructure ─────────────────────────────────────────────────────────────

def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def parse_metrics(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                metrics[key] = float(m.group(2))
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
        "--experiment_name", f"sweep35_interp_vs_raw/{name}",
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


def collect_summary(names, output_csv, sort_key="mesh_idw_f1_1v"):
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

    seen = set()
    all_keys = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    fieldnames = ["name"] + [k for k in all_keys if k != "name"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Summary written to {output_csv} ({len(rows)} runs)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Interp vs. raw training comparison (sweep 35)",
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
        print(f"\nSweep 35 — interp vs. raw — {len(ALL_RUNS)} runs total")
        print(f"  common: 512k pts, gs=1, no vvar/TV/BF/redundancy, rays_per_batch_late=4M")
        print(f"  interp: interpolation_start=9000, scale=1.5, sv=0.2, global sigma")
        print(f"  raw:    interpolation_start=-1")
        print()
        for name, cfg in ALL_RUNS.items():
            istart = cfg["interpolation_start"]
            cond = "interp" if istart >= 0 else "raw"
            print(f"  {name:<28}  {cond}  data={cfg['data_path'].split('/')[-1]}")
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
        if not names:
            print(f"[WARN] worker {args.worker}/{args.num_workers} has no runs — nothing to do")
            return
        print(f"Sweep 35: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 35: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
