#!/usr/bin/env python3
"""Sweep 31: Refvol tuning, grad-smoothing depth, vvar/nvar variants, interp on light-reg baselines.

Motivation:
  - Sweep 29 surface-metric winners are A02-no-reg and D04-gs1-nothing; sweep 30's interp
    ablation was run on a heavier-reg pre-sweep-29 base. This sweep fills that gap.
  - Many refvol parameters were fixed throughout sweep 29 (edge_mask/alpha, blur_sigma,
    guided toggles, start/until) — this sweep varies them on an A01-style base to isolate
    refvol-parameter effects.
  - grad_smooth_hops was only tested up to 2 in combination with A00; this sweep tests
    deeper hops in isolation (D04-style, no other regularization).
  - vvar/nvar were always run together in sweep 29; this sweep isolates each.

Groups:
  V  interp × {A02-no-reg, D04-gs1-nothing}              (6 runs)
  G  gs-only depth: hops 2/3/4 on D04-base               (3 runs)
  R  refvol-only parameter gaps on A01-base               (7 runs)
  N  vvar/nvar isolation and lighter weights              (4 runs)

Usage:
    python sweep_31_refgs_interp.py --list
    python sweep_31_refgs_interp.py
    python sweep_31_refgs_interp.py --runs V00 G01 R01 N01
    python sweep_31_refgs_interp.py --worker 1 --of 4
    python sweep_31_refgs_interp.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep31_refgs_interp"
DATA_PATH = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
REF_VOL   = "output/manual/64_init/model.pt"


def base_config(
    # --- VVAR ---
    voxel_var_weight=1e-3,
    voxel_var_weight_final=-1.0,
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
    ref_volume_weight_final=-1.0,
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
    # --- INTERPOLATION ---
    interpolation_start=9000,
    per_cell_sigma=True,
    per_neighbor_sigma=True,
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
        # Sigma schedule
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
        "interpolation_start": interpolation_start,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": per_cell_sigma,
        "per_neighbor_sigma": per_neighbor_sigma,
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


# ── Shared bases ──────────────────────────────────────────────────────────────

def a02_base(**kw):
    """A02-no-reg: no vvar, no nvar, no refvol."""
    return base_config(
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
        ref_volume_weight=0.0,
        ref_volume_path="",
        ref_guided_pruning=False,
        ref_guided_densify=False,
        **kw,
    )


def d04_base(grad_smooth_hops=1, **kw):
    """D04-gs1-nothing: no vvar, no nvar, no refvol, gs=1."""
    return a02_base(grad_smooth_hops=grad_smooth_hops, **kw)


def a01_base(**kw):
    """A01-refvol-only: refvol on, vvar/nvar off."""
    return base_config(
        voxel_var_weight=0.0,
        neighbor_var_weight=0.0,
        **kw,
    )


ALL_RUNS = {
    # ── V: Interp × light-reg baselines ──────────────────────────────────────
    # Sweep 29's A02 and D04 already ran per_cell=True, per_neighbor=True (= sweep30-V03).
    # Here we fill the remaining three interp variants for each.

    # A02 (no reg) + interp variants
    "V00-a02-no-interp":  a02_base(interpolation_start=-1),
    "V01-a02-global":     a02_base(per_cell_sigma=False, per_neighbor_sigma=False),
    "V02-a02-percell":    a02_base(per_cell_sigma=True,  per_neighbor_sigma=False),

    # A02 per-neigh: already run in sweep 29 — metrics pre-seeded from A02-no-reg
    "V06-a02-perneigh":   a02_base(per_cell_sigma=True, per_neighbor_sigma=True),

    # D04 (gs1, no reg) + interp variants
    "V03-d04-no-interp":  d04_base(interpolation_start=-1),
    "V04-d04-global":     d04_base(per_cell_sigma=False, per_neighbor_sigma=False),
    "V05-d04-percell":    d04_base(per_cell_sigma=True,  per_neighbor_sigma=False),

    # D04 per-neigh: already run in sweep 29 — metrics pre-seeded from D04-gs1-nothing
    "V07-d04-perneigh":   d04_base(per_cell_sigma=True, per_neighbor_sigma=True),

    # ── G: Grad-smoothing depth (D04-base: no other regularization) ───────────
    # D04 covers hops=1; extend to test saturation / overshoot.
    "G01-gs2": d04_base(grad_smooth_hops=2),
    "G02-gs3": d04_base(grad_smooth_hops=3),
    "G03-gs4": d04_base(grad_smooth_hops=4),

    # ── R: Refvol parameter gaps (A01-base: refvol only, no vvar/nvar) ────────
    # Baseline for this group: A01 = base_config with vvar=nvar=0.
    # All R runs keep ref_volume_weight=1.0 const unless noted.

    # Never tested in sweep 29 — edge weighting off entirely
    "R01-no-edge-mask":      a01_base(ref_volume_edge_mask=False),

    # Softer edge mask (10 → 2) — less aggressive surface down-weighting
    "R02-edge-alpha-2":      a01_base(ref_volume_edge_alpha=2.0),

    # Non-zero blur (default was 0 throughout sweep 29; config default is 2.0)
    "R03-blur-1.0":          a01_base(ref_volume_blur_sigma=1.0),

    # Isolate ref-guided pruning vs densify
    "R04-no-guided-prune":   a01_base(ref_guided_pruning=False),
    "R05-no-guided-densify": a01_base(ref_guided_densify=False),

    # Higher weight ceiling on the increasing schedule (C04 winner: 0.1→1.0; try 0.1→2.0)
    "R06-w-inc-high":        a01_base(ref_volume_weight=0.1, ref_volume_weight_final=2.0),

    # Delayed activation — let geometry stabilize through early densification first
    "R07-delayed-3000":      a01_base(ref_volume_start=3000),

    # ── N: vvar/nvar isolation and lighter weights ─────────────────────────────
    # Sweep 29 always ran vvar and nvar together (both at 1e-3 baseline).
    # These runs isolate each and test a lighter joint weight.

    # vvar alone (nvar off)
    "N01-vvar-only":    base_config(voxel_var_weight=1e-3, neighbor_var_weight=0.0,
                                    ref_volume_weight=0.0, ref_volume_path="",
                                    ref_guided_pruning=False, ref_guided_densify=False),

    # nvar alone (vvar off)
    "N02-nvar-only":    base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                    ref_volume_weight=0.0, ref_volume_path="",
                                    ref_guided_pruning=False, ref_guided_densify=False),

    # Both, but lighter (3× lower than baseline) — minimal smoothing nudge
    "N03-lighter-both": base_config(voxel_var_weight=3e-4, neighbor_var_weight=3e-4,
                                    ref_volume_weight=0.0, ref_volume_path="",
                                    ref_guided_pruning=False, ref_guided_densify=False),

    # Deeper nvar neighborhood (sweep 29 F03 was hops=2; try 3)
    "N04-nvar-hops-3":  base_config(voxel_var_weight=0.0, neighbor_var_weight=1e-3,
                                    neighbor_var_hops=3,
                                    ref_volume_weight=0.0, ref_volume_path="",
                                    ref_guided_pruning=False, ref_guided_densify=False),
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
        "--experiment_name", f"sweep31_refgs_interp/{name}",
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


def collect_summary(names, output_csv, sort_key="vol_raw_dice"):
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
        description="Refvol/gs/vvar-nvar/interp sweep (sweep 31)",
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
            "V": "Interp × light-reg baselines (A02 / D04)",
            "G": "Grad-smoothing depth (D04-base, no other reg)",
            "R": "Refvol parameter gaps (A01-base, no vvar/nvar)",
            "N": "vvar/nvar isolation and lighter weights",
        }
        current_group = None
        print(f"\nSweep 31 — {len(ALL_RUNS)} runs:\n")
        for name, cfg in ALL_RUNS.items():
            group = name[0]
            if group != current_group:
                current_group = group
                print(f"  ── {group}: {group_labels.get(group, '')} ──")
            vvar  = cfg["voxel_var_weight"]
            nvar  = cfg["neighbor_var_weight"]
            rv    = cfg.get("ref_volume_path", "")
            rv_w  = cfg["ref_volume_weight"]
            rv_wf = cfg["ref_volume_weight_final"]
            gs    = cfg["grad_smooth_hops"]
            i_s   = cfg["interpolation_start"]
            pc    = cfg["per_cell_sigma"]
            pn    = cfg["per_neighbor_sigma"]
            rv_str = (f"refvol w={rv_w}" + (f"→{rv_wf}" if rv_wf >= 0 else "(const)")) if rv else "no-refvol"
            gs_str = f"  gs={gs}" if gs > 0 else ""
            interp_str = "no-interp" if i_s < 0 else f"interp(pc={pc},pn={pn})"
            print(f"  {name:28s}  vvar={vvar}  nvar={nvar}  {rv_str}{gs_str}  {interp_str}")
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
        print(f"Sweep 31: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 31: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
