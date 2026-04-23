#!/usr/bin/env python3
"""Sweep 32: Interp sigma grid on global-sigma winners.

Motivation:
  Sweep 31 established that global interp sigma (per_cell=False, per_neighbor=False)
  wins on surface metrics (V04-d04-global: CD=0.861, H95=4.54, F1@1v=0.844), but
  interp_sigma_scale=0.7 and interp_sigma_v=0.2 were frozen throughout sweeps 29-31.
  This sweep varies both on a 3x3 log-spaced grid across three bases:
    D  — gs=1, no refvol, no vvar/nvar  (sweep 31 d04_base)
    A  — refvol (unblurred, alpha=10), no gs, no vvar/nvar  (sweep 31 a01_base)
    DA — gs=1 + refvol  (new combination)
  and two point budgets (512k, 1M) across two datasets (chest, pepper).

Note: interp_sigma_* also affects redundancy pruning scoring (train.py:1125) and
  reference-volume baking (_idw_voxelize), so effects are not purely post-hoc.

Per_neighbor_sigma is NOT included — it is a confirmed no-op when per_cell_sigma=False
  (scene.py:82-88).

Datasets:
  chest  — 0_chest_cone (refvol available: output/manual/64_init/model.pt)
  pepper — 1_pepper_cone (no refvol yet; A/DA bases skipped until PEPPER_REFVOL is set)

Total runs: 72 (chest: 54, pepper: 18). Expands to 108 when PEPPER_REFVOL is set.

Usage:
    python sweep_32_interp_grid.py --list
    python sweep_32_interp_grid.py
    python sweep_32_interp_grid.py --runs chest-D-p512-s07-v020
    python sweep_32_interp_grid.py --worker 1 --of 4
    python sweep_32_interp_grid.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR  = "output/sweep32_interp_grid"
CHEST_DATA  = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
PEPPER_DATA = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"
CHEST_REFVOL  = "output/manual/64_init/model.pt"
PEPPER_REFVOL = "output/init/pepper/model.pt"


def base_config(
    data_path=CHEST_DATA,
    # --- VVAR ---
    voxel_var_weight=0.0,
    voxel_var_weight_final=-1.0,
    voxel_var_resolution=64,
    voxel_var_supersample=4,
    # --- NVAR ---
    neighbor_var_weight=0.0,
    neighbor_var_weight_final=-1.0,
    neighbor_var_hops=1,
    neighbor_reg_type="bilateral_var",
    neighbor_huber_delta=0.1,
    # --- REFVOL ---
    ref_volume_path="",
    ref_volume_weight=1.0,
    ref_volume_weight_final=-1.0,
    ref_volume_start=0,
    ref_volume_until=-1,
    ref_volume_resolution=64,
    ref_volume_blur_sigma=0.0,
    ref_volume_edge_mask=True,
    ref_volume_edge_alpha=10.0,
    ref_volume_supersample=4,
    ref_guided_pruning=False,
    ref_guided_densify=False,
    ref_guided_eps=0.01,
    # --- GRAD SMOOTHING ---
    grad_smooth_hops=0,
    # --- PRUNING / DENSIFY ---
    final_points=512000,
    prune_variance_criterion=False,
    redundancy_cap=0.03,
    targeted_fraction=0.1,
    # --- SIGMA SCHEDULE (vvar/nvar, unused here but required) ---
    var_sigma_v_init=50.0,
    var_sigma_v_final=0.2,
    # --- INTERPOLATION ---
    interpolation_start=9000,
    interp_sigma_scale=0.7,
    interp_sigma_v=0.2,
    per_cell_sigma=False,
    per_neighbor_sigma=False,
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
        # Sigma schedule (vvar/nvar)
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
        "interp_sigma_scale": interp_sigma_scale,
        "interp_sigma_v": interp_sigma_v,
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


# ── Base helpers ──────────────────────────────────────────────────────────────

def d_base(data_path, ref_vol, **kw):
    """D: gs=1, no refvol, no vvar/nvar."""
    return base_config(
        data_path=data_path,
        grad_smooth_hops=1,
        ref_volume_path="",
        ref_volume_weight=0.0,
        ref_guided_pruning=False,
        ref_guided_densify=False,
        **kw,
    )


def a_base(data_path, ref_vol, **kw):
    """A: refvol (unblurred, alpha=10), no gs, no vvar/nvar."""
    return base_config(
        data_path=data_path,
        grad_smooth_hops=0,
        ref_volume_path=ref_vol,
        ref_volume_weight=1.0,
        ref_volume_edge_mask=True,
        ref_volume_edge_alpha=10.0,
        ref_volume_blur_sigma=0.0,
        ref_guided_pruning=True,
        ref_guided_densify=True,
        **kw,
    )


def da_base(data_path, ref_vol, **kw):
    """DA: gs=1 + refvol (unblurred, alpha=10). New combination."""
    return base_config(
        data_path=data_path,
        grad_smooth_hops=1,
        ref_volume_path=ref_vol,
        ref_volume_weight=1.0,
        ref_volume_edge_mask=True,
        ref_volume_edge_alpha=10.0,
        ref_volume_blur_sigma=0.0,
        ref_guided_pruning=True,
        ref_guided_densify=True,
        **kw,
    )


# ── Build ALL_RUNS ─────────────────────────────────────────────────────────────

SIGMA_SCALES = [0.3, 0.7, 1.5]   # spatial: sharp / default / flat
SIGMA_VS     = [0.05, 0.2, 0.8]  # bilateral value: sharp / default / flat
BUDGETS      = [512000, 1000000]

# dataset_tag → (data_path, refvol_path, bases_to_include)
DATASETS = {
    "chest":  (CHEST_DATA,  CHEST_REFVOL,  ["D", "A", "DA"]),
    "pepper": (PEPPER_DATA, PEPPER_REFVOL, ["D"] + (["A", "DA"] if PEPPER_REFVOL else [])),
}

BASE_FNS = {"D": d_base, "A": a_base, "DA": da_base}


def _scale_tag(s):
    return f"s{int(s * 10):02d}"   # 0.3→s03, 0.7→s07, 1.5→s15


def _v_tag(v):
    return f"v{int(v * 100):03d}"  # 0.05→v005, 0.2→v020, 0.8→v080


def _budget_tag(n):
    return "p512k" if n == 512000 else "p1M"


ALL_RUNS = {}
for ds_tag, (data_path, ref_vol, bases) in DATASETS.items():
    for base_tag in bases:
        base_fn = BASE_FNS[base_tag]
        for budget in BUDGETS:
            for scale in SIGMA_SCALES:
                for sv in SIGMA_VS:
                    name = f"{ds_tag}-{base_tag}-{_budget_tag(budget)}-{_scale_tag(scale)}-{_v_tag(sv)}"
                    ALL_RUNS[name] = base_fn(
                        data_path, ref_vol,
                        final_points=budget,
                        interp_sigma_scale=scale,
                        interp_sigma_v=sv,
                    )


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
        "--experiment_name", f"sweep32_interp_grid/{name}",
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


def collect_summary(names, output_csv, sort_key="vol_idw_dice"):
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
        description="Interp sigma grid sweep (sweep 32)",
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
        print(f"\nSweep 32 — interp sigma grid — {len(ALL_RUNS)} runs total")
        print(f"  sigma_scale: {SIGMA_SCALES}")
        print(f"  sigma_v:     {SIGMA_VS}")
        print(f"  budgets:     {BUDGETS}")
        print()
        current_ds = None
        current_base = None
        for name, cfg in ALL_RUNS.items():
            ds = name.split("-")[0]
            base = name.split("-")[1]
            if ds != current_ds:
                current_ds = ds
                current_base = None
                print(f"  ── {ds} ({cfg['data_path']}) ──")
            if base != current_base:
                current_base = base
                rv = cfg["ref_volume_path"]
                gs = cfg["grad_smooth_hops"]
                rv_str = f"refvol={rv}" if rv else "no-refvol"
                print(f"    [{base}]  gs={gs}  {rv_str}")
            fp = cfg["final_points"]
            ss = cfg["interp_sigma_scale"]
            sv = cfg["interp_sigma_v"]
            print(f"      {name:<46}  pts={fp//1000}k  scale={ss}  sv={sv}")
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
        print(f"Sweep 32: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 32: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
