#!/usr/bin/env python3
"""Sweep 29: Anti-grittiness — isolate each new mechanism.

Focuses on: IDW-supersampled voxel variance (group B), IDW-supersampled refvol
loss (group C), density-gradient smoothing (group D), and neighbor-reg variants
including new Huber and median targets (group E). Also covers a handful of
unexplored older params (group F).

All runs use the 0_chest_cone 75-view CT dataset. IDW ref-volume is loaded from
output/manual/64_init/model.pt (the current best base config already points here).
ss=4 means voxel_var_supersample=4 / ref_volume_supersample=4.

Usage:
    python sweep_29_antigrit.py --list
    python sweep_29_antigrit.py
    python sweep_29_antigrit.py --runs A00 B01 C01
    python sweep_29_antigrit.py --worker 1 --of 4
    python sweep_29_antigrit.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep29_antigrit"
DATA_PATH = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
REF_VOL   = "output/manual/64_init/model.pt"


def base_config(
    # --- VVAR ---
    voxel_var_weight=1e-3,
    voxel_var_weight_final=-1.0,   # -1 = hold constant
    voxel_var_resolution=64,
    voxel_var_supersample=4,
    # --- NVAR ---
    neighbor_var_weight=1e-3,
    neighbor_var_weight_final=-1.0,
    neighbor_var_hops=1,
    neighbor_reg_type="bilateral_var",
    neighbor_huber_delta=0.1,
    # --- REFVOL ---
    ref_volume_path=REF_VOL,
    ref_volume_weight=1.0,
    ref_volume_weight_final=-1.0,  # -1 = hold constant
    ref_volume_start=0,
    ref_volume_until=-1,
    ref_volume_resolution=64,
    ref_volume_blur_sigma=0.0,
    ref_volume_edge_mask=True,
    ref_volume_edge_alpha=10.0,
    ref_volume_supersample=4,
    # --- GRAD SMOOTHING ---
    grad_smooth_hops=0,
    # --- PRUNING / DENSIFY ---
    final_points=512000,
    prune_variance_criterion=False,
    redundancy_cap=0.03,
    ref_guided_pruning=True,
    ref_guided_densify=True,
    ref_guided_eps=0.01,
    targeted_fraction=0.1,
    # --- SIGMA SCHEDULE ---
    var_sigma_v_init=50.0,
    var_sigma_v_final=0.2,
):
    return {
        # Training
        "iterations": 10000,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": final_points,
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
        "data_path": DATA_PATH,
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
        # Pruning
        "redundancy_threshold": 0.0,
        "redundancy_cap": redundancy_cap,
        "redundancy_cap_init": 0.0,
        "redundancy_cap_final": 0.0,
        "prune_variance_criterion": prune_variance_criterion,
        "prune_hops": 1,
        "ref_guided_pruning": ref_guided_pruning,
        "ref_guided_densify": ref_guided_densify,
        "ref_guided_eps": ref_guided_eps,
        # Ray sampling
        "targeted_fraction": targeted_fraction,
        "targeted_start": -1,
        "high_error_fraction": 0.2,
        "high_error_power": 1.0,
        "high_error_start": -1,
        # TV
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        # Voxel variance
        "voxel_var_weight": voxel_var_weight,
        "voxel_var_weight_final": voxel_var_weight_final,
        "voxel_var_resolution": voxel_var_resolution,
        "voxel_var_start": 0,
        "voxel_var_supersample": voxel_var_supersample,
        # Neighbor variance
        "neighbor_var_weight": neighbor_var_weight,
        "neighbor_var_weight_final": neighbor_var_weight_final,
        "neighbor_var_hops": neighbor_var_hops,
        "neighbor_var_start": 0,
        "neighbor_reg_type": neighbor_reg_type,
        "neighbor_huber_delta": neighbor_huber_delta,
        # Sigma schedule (shared by vvar and nvar)
        "var_sigma_v_init": var_sigma_v_init,
        "var_sigma_v_final": var_sigma_v_final,
        # Reference volume
        "ref_volume_path": ref_volume_path,
        "ref_volume_weight": ref_volume_weight,
        "ref_volume_weight_final": ref_volume_weight_final,
        "ref_volume_start": ref_volume_start,
        "ref_volume_until": ref_volume_until,
        "ref_volume_resolution": ref_volume_resolution,
        "ref_volume_blur_sigma": ref_volume_blur_sigma,
        "ref_volume_edge_mask": ref_volume_edge_mask,
        "ref_volume_edge_alpha": ref_volume_edge_alpha,
        "ref_volume_supersample": ref_volume_supersample,
        # Grad smoothing
        "grad_smooth_hops": grad_smooth_hops,
        # Interpolation
        "interpolation_start": 9000,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": True,
        "per_neighbor_sigma": True,
        # BF off
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        # Gaussians / gradient field: off
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
    }


ALL_RUNS = {
    # ── A: Baselines ──────────────────────────────────────────────────────────
    # A00: current best config — refvol + vvar + nvar + ref-guided pruning, ss=4
    "A00-baseline": base_config(),

    # A01: refvol only (vvar/nvar off) — isolates refvol contribution
    "A01-refvol-only": base_config(
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
    ),

    # A02: no regularization at all — raw grittiness floor
    "A02-no-reg": base_config(
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
        ref_volume_weight=0.0,
        ref_volume_path="",
    ),

    # ── B: VVAR × IDW supersample ─────────────────────────────────────────────
    # Compare ss=1 (IDW center-only) vs ss=4 (stochastic k-sample IDW)
    "B00-vvar-ss1-res64": base_config(voxel_var_supersample=1),
    "B01-vvar-ss4-res64": base_config(),  # default ss=4, res=64, w=1e-3 const

    # Resolution sweep (all ss=4, w=1e-3 const)
    "B02-vvar-ss4-res32": base_config(voxel_var_resolution=32),
    "B03-vvar-ss4-res96": base_config(voxel_var_resolution=96),

    # Weight sweep (ss=4, res=64, constant)
    "B04-vvar-ss4-w5e-4": base_config(voxel_var_weight=5e-4),
    "B05-vvar-ss4-w5e-3": base_config(voxel_var_weight=5e-3),

    # Schedule: inverted (small→large = increase during training)
    "B06-vvar-ss4-w-inc": base_config(voxel_var_weight=1e-4, voxel_var_weight_final=1e-3),

    # Schedule: explicit decay (large→small, matches old behavior)
    "B07-vvar-ss4-w-dec": base_config(voxel_var_weight=1e-3, voxel_var_weight_final=1e-4),

    # ── C: REFVOL × IDW supersample ───────────────────────────────────────────
    # Compare ss=1 (IDW center) vs ss=4 (stochastic k-sample IDW)
    "C00-refvol-ss1": base_config(ref_volume_supersample=1, ref_volume_weight_final=-1.0),
    "C01-refvol-ss4-w1-const": base_config(),  # default ss=4, res=64, w=1 const

    # Resolution sweep (all ss=4, w=1 const)
    "C02-refvol-ss4-res48": base_config(ref_volume_resolution=48),
    "C03-refvol-ss4-res96": base_config(ref_volume_resolution=96),

    # Weight schedule: increasing (brainstorm §2/§7 — weight grows as noise emerges)
    "C04-refvol-ss4-w-inc": base_config(ref_volume_weight=0.1, ref_volume_weight_final=1.0),

    # Weight schedule: explicit decay (current default: 1→0.1)
    "C05-refvol-ss4-w-dec": base_config(ref_volume_weight=1.0, ref_volume_weight_final=0.1),

    # Higher constant weight
    "C06-refvol-ss4-w2-const": base_config(ref_volume_weight=2.0),

    # ── D: Gradient smoothing ─────────────────────────────────────────────────
    # (uniform neighbor averaging of density.grad before optimizer.step)
    # On top of full best config (A00)
    "D01-gs1": base_config(grad_smooth_hops=1),
    "D02-gs2": base_config(grad_smooth_hops=2),

    # Isolated with refvol only (no vvar/nvar)
    "D03-gs1-refvolonly": base_config(
        grad_smooth_hops=1,
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
    ),

    # Isolated — no regularization at all; pure gradient smoothing anti-noise
    "D04-gs1-nothing": base_config(
        grad_smooth_hops=1,
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
        ref_volume_weight=0.0,
        ref_volume_path="",
    ),

    # ── E: Neighbor reg variants ──────────────────────────────────────────────
    # Huber: aggressively kills sub-delta noise; caps penalty at edges
    "E01-huber-d0.1":    base_config(neighbor_reg_type="huber", neighbor_huber_delta=0.1),
    "E02-huber-d0.05":   base_config(neighbor_reg_type="huber", neighbor_huber_delta=0.05),
    "E03-bi-huber-d0.1": base_config(neighbor_reg_type="bilateral_huber", neighbor_huber_delta=0.1),

    # Median: robust to outlier neighbors (shot-noise cells don't corrupt neighbors' target)
    "E04-median":    base_config(neighbor_reg_type="median"),
    "E05-bi-median": base_config(neighbor_reg_type="bilateral_median"),

    # ── F: Unexplored older params ────────────────────────────────────────────
    # ref_guided_eps: floor on (1-ref_w) densify multiplier
    "F01-ref-eps-0.05":  base_config(ref_guided_eps=0.05),   # current is 0.01
    "F02-ref-eps-0.001": base_config(ref_guided_eps=0.001),  # tighter

    # Neighbor variance with deeper neighborhood
    "F03-nvar-hops-2": base_config(neighbor_var_hops=2),

    # Stronger targeted sampling
    "F04-targeted-0.2": base_config(targeted_fraction=0.2),
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
        "--experiment_name", f"sweep29_antigrit/{name}",
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
        description="Anti-grittiness mechanism isolation sweep (sweep 29)",
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
        group_labels = {
            "A": "Baselines",
            "B": "VVAR × IDW supersample",
            "C": "REFVOL × IDW supersample",
            "D": "Gradient smoothing",
            "E": "Neighbor reg variants",
            "F": "Unexplored older params",
        }
        current_group = None
        print(f"\nSweep 29 — {len(ALL_RUNS)} runs:\n")
        for name, cfg in ALL_RUNS.items():
            group = name[0]
            if group != current_group:
                current_group = group
                print(f"  ── {group}: {group_labels.get(group, '')} ──")

            vvar   = cfg["voxel_var_weight"]
            nvar   = cfg["neighbor_var_weight"]
            rv_w   = cfg["ref_volume_weight"]
            rv_wf  = cfg["ref_volume_weight_final"]
            ss_v   = cfg["voxel_var_supersample"]
            ss_r   = cfg["ref_volume_supersample"]
            gs     = cfg["grad_smooth_hops"]
            nrt    = cfg["neighbor_reg_type"]
            rv_on  = cfg.get("ref_volume_path", "") != ""
            rv_str = f"refvol w={rv_w}" + (f"→{rv_wf}" if rv_wf >= 0 else "(const)") if rv_on else "no refvol"
            gs_str = f" gs={gs}" if gs > 0 else ""
            nrt_str = f" nrt={nrt}" if nrt != "bilateral_var" else ""
            print(f"  {name:32s}  vvar={vvar}@ss{ss_v}  nvar={nvar}  {rv_str}@ss{ss_r}{gs_str}{nrt_str}")
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
        print(f"Sweep 29: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 29: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
