#!/usr/bin/env python3
"""Sweep 33: Vvar regularization × Interp sigma combos.

Motivation:
  Sweep 32 found two contradictory optima (vvar=0 throughout):
    Surface metrics (mesh-F1):  scale=0.7, v=0.05
    Volume PSNR/Dice:           scale=1.5, v=0.8
  This sweep adds vvar back in at 4 weights to see if it can shift the Pareto frontier.

  Key mechanism: vvar's inner IDW estimator reads interp_sigma_scale/_v directly from
  the model's cached _idw_sigma/_idw_sigma_v (scene.py:712-735), so sigma changes
  genuinely affect vvar behaviour — not just rendering.

  Logging cadence is reduced for speed:
    log_percent=50  → cheap logs at iter 4999 and 9999 only (10x fewer than default 5)
    diag_percent=200 → diag_interval=20000, so in-loop diag trigger never fires; the
                       unconditional post-loop log_diag at train.py:1250 runs once at end.

Bases:
  D — gs=1, no refvol (sweep 32 winner on chest)
  A — refvol (unblurred, alpha=10, ref_guided on), no gs (sweep 32 winner on pepper)

Sigma points: 6 (5 from sweep-32 winners + scale=2.5 extension at v=0.8)
Vvar weights: 4 (1e-3, 3e-3, 1e-2, 1e-1); vvar=0 covered by sweep 32.
Total: 6 × 4 × 2 bases × 2 datasets × 2 budgets = 192 runs.

Usage:
    python sweep_33_vvar_sigma.py --list
    python sweep_33_vvar_sigma.py
    python sweep_33_vvar_sigma.py --runs chest-D-p1M-s07-v020-w1e3
    python sweep_33_vvar_sigma.py --worker 1 --of 4
    python sweep_33_vvar_sigma.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR     = "output/sweep33_vvar_sigma"
CHEST_DATA    = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
PEPPER_DATA   = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"
CHEST_REFVOL  = "output/manual/64_init/model.pt"
PEPPER_REFVOL = "output/init/pepper/model.pt"


def base_config(
    data_path=CHEST_DATA,
    # --- VVAR ---
    voxel_var_weight=0.0,
    voxel_var_weight_final=-1.0,
    voxel_var_resolution=64,
    voxel_var_supersample=4,
    # --- NVAR (off) ---
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
    # --- SIGMA SCHEDULE (vvar outer bilateral) ---
    var_sigma_v_init=50.0,
    var_sigma_v_final=0.2,
    # --- INTERPOLATION ---
    interpolation_start=9000,
    interp_sigma_scale=0.7,
    interp_sigma_v=0.2,
    per_cell_sigma=False,
    per_neighbor_sigma=False,
    # --- LOGGING (reduced for speed) ---
    log_percent=50,
    diag_percent=200,
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
        # Neighbor variance (off)
        "neighbor_var_weight": neighbor_var_weight,
        "neighbor_var_weight_final": neighbor_var_weight_final,
        "neighbor_var_hops": neighbor_var_hops,
        "neighbor_var_start": 0,
        "neighbor_reg_type": neighbor_reg_type,
        "neighbor_huber_delta": neighbor_huber_delta,
        # Sigma schedule (vvar outer bilateral; unchanged from sweeps 31/32)
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
        # Logging cadence
        "log_percent": log_percent,
        "diag_percent": diag_percent,
    }


# ── Base helpers ──────────────────────────────────────────────────────────────

def d_base(data_path, ref_vol, voxel_var_weight, **kw):
    """D: gs=1, no refvol. vvar weight is a free parameter."""
    return base_config(
        data_path=data_path,
        grad_smooth_hops=1,
        ref_volume_path="",
        ref_volume_weight=0.0,
        ref_guided_pruning=False,
        ref_guided_densify=False,
        voxel_var_weight=voxel_var_weight,
        **kw,
    )


def a_base(data_path, ref_vol, voxel_var_weight, **kw):
    """A: refvol (unblurred, alpha=10), no gs. vvar weight is a free parameter."""
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
        voxel_var_weight=voxel_var_weight,
        **kw,
    )


# ── Sweep axes ────────────────────────────────────────────────────────────────

# 6 sigma points: 5 sweep-32 winners + (2.5, 0.8) wider extension
SIGMA_POINTS = [
    (0.7, 0.05),
    (0.7, 0.2),
    (0.7, 0.8),
    (1.5, 0.2),
    (1.5, 0.8),
    (2.5, 0.8),
]

# 4 vvar weights; vvar=0 corner already covered by sweep 32
VVAR_WEIGHTS = [1e-3, 3e-3, 1e-2, 1e-1]

BUDGETS = [512000, 1000000]

DATASETS = {
    "chest":  (CHEST_DATA,  CHEST_REFVOL,  ["D", "A"]),
    "pepper": (PEPPER_DATA, PEPPER_REFVOL, ["D", "A"]),
}

BASE_FNS = {"D": d_base, "A": a_base}


def _scale_tag(s):
    # 0.7→s07, 1.5→s15, 2.5→s25
    return f"s{int(round(s * 10)):02d}"


def _v_tag(v):
    # 0.05→v005, 0.2→v020, 0.8→v080
    return f"v{int(round(v * 100)):03d}"


def _budget_tag(n):
    return "p512k" if n == 512000 else "p1M"


def _weight_tag(w):
    # 1e-3→w1e3, 3e-3→w3e3, 1e-2→w1e2, 1e-1→w1e1
    if w == 3e-3:
        return "w3e3"
    exp = round(-1 * __import__('math').log10(w))
    return f"w1e{exp}"


ALL_RUNS = {}
for ds_tag, (data_path, ref_vol, bases) in DATASETS.items():
    for base_tag in bases:
        base_fn = BASE_FNS[base_tag]
        for budget in BUDGETS:
            for (scale, sv) in SIGMA_POINTS:
                for vw in VVAR_WEIGHTS:
                    name = (f"{ds_tag}-{base_tag}-{_budget_tag(budget)}"
                            f"-{_scale_tag(scale)}-{_v_tag(sv)}-{_weight_tag(vw)}")
                    ALL_RUNS[name] = base_fn(
                        data_path, ref_vol,
                        voxel_var_weight=vw,
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
        "--experiment_name", f"sweep33_vvar_sigma/{name}",
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
        description="Vvar × interp sigma sweep (sweep 33)",
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
        print(f"\nSweep 33 — vvar × sigma — {len(ALL_RUNS)} runs total")
        print(f"  sigma_points: {SIGMA_POINTS}")
        print(f"  vvar_weights: {VVAR_WEIGHTS}")
        print(f"  budgets:      {BUDGETS}")
        print(f"  log_percent=50  diag_percent=200 (diags deferred to post-loop only)")
        print()
        current_ds = None
        current_base = None
        current_budget = None
        for name, cfg in ALL_RUNS.items():
            parts = name.split("-")
            ds, base, budget_tag = parts[0], parts[1], parts[2]
            if ds != current_ds:
                current_ds = ds
                current_base = None
                print(f"  ── {ds} ({cfg['data_path']}) ──")
            if base != current_base or budget_tag != current_budget:
                current_base = base
                current_budget = budget_tag
                rv = cfg["ref_volume_path"]
                gs = cfg["grad_smooth_hops"]
                rv_str = f"refvol" if rv else "no-refvol"
                fp = cfg["final_points"]
                print(f"    [{base}]  gs={gs}  {rv_str}  pts={fp//1000}k")
            sc = cfg["interp_sigma_scale"]
            sv = cfg["interp_sigma_v"]
            vw = cfg["voxel_var_weight"]
            print(f"      {name:<56}  scale={sc}  sv={sv}  vvar={vw:.0e}")
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
        print(f"Sweep 33: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 33: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
