#!/usr/bin/env python3
"""Sweep 36: Re-validate regularization techniques with interpolation OFF.

Motivation:
  Sweep 35 showed that training with interpolation_start=9000 injects noise into
  raw cell densities. Previous regularization sweeps (nvar, vvar, TV, BF, pruning,
  FDK init, L1/L2) were all evaluated under this contaminated regime — their
  direction of effect may be wrong. We re-test each technique with interp OFF
  (interpolation_start=-1) and post-hoc IDW evaluation at dataset-specific σ.

Post-hoc evaluation σ (from eval_sigma_sweep.py on sweep 35 raw checkpoints):
  chest:  sigma_s=0.008, sigma_v=0.05  (scale_equiv ≈ 0.65)
  pepper: sigma_s=0.010, sigma_v=0.40  (scale_equiv ≈ 1.02)

Stage A — baselines:
  best427:  current best config (best427_nointerp) with interp off
  vanilla:  strip targeted/he/grad-smooth/redundancy for a clean reference

Stage B — single-factor ablations vs best427 baseline:
  B1 nvar:     neighbor variance weight × hops
  B2 vvar:     voxel variance weight × resolution
  B3 tv:       TV weight × tv_on_raw
  B4 bf:       bilateral filter σ schedules
  B5 pruning:  redundancy_cap variants + variance criterion
  B6 loss:     l2 vs l1
  B7 refvol:   reference volume regularization (output/init/{chest,pepper}/model.pt)

Usage:
    python sweep_36_validate_no_interp.py --list
    python sweep_36_validate_no_interp.py
    python sweep_36_validate_no_interp.py --runs chest-best427 pepper-nvar-w1e3-h1
    python sweep_36_validate_no_interp.py --worker 1 --of 4
    python sweep_36_validate_no_interp.py --summarize
    python sweep_36_validate_no_interp.py --eval-only
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR    = "output/sweep36_validate_no_interp"
CHEST_DATA      = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
PEPPER_DATA     = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"
CHEST_INIT_PT   = "output/init/chest/model.pt"
PEPPER_INIT_PT  = "output/init/pepper/model.pt"

# Post-hoc σ per dataset (from eval_sigma_sweep.py on sweep 35 raw checkpoints)
EVAL_SIGMA = {
    CHEST_DATA:  {"spatial": [0.008], "sigma_v": [0.05]},
    PEPPER_DATA: {"spatial": [0.010], "sigma_v": [0.40]},
}


def base_config(data_path=CHEST_DATA, **overrides):
    cfg = {
        # Training
        "iterations": 10000,
        "rays_per_batch": 1_000_000,
        "rays_per_batch_late": 8_000_000,
        "rays_per_batch_late_start": 9000,
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
        # Sigma schedule for variance regs
        "var_sigma_v_init": 50.0,
        "var_sigma_v_final": 0.2,
        # Grad smoothing
        "grad_smooth_hops": 1,
        # Densification
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "gradient_fraction": 0.4,
        "idw_fraction": 0.3,
        "entropy_fraction": 0.3,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        # Pruning — basic + light IDW redundancy
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.03,
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
        # Interpolation — OFF during training
        "interpolation_start": -1,
        "interp_ramp": False,
        "interp_sigma_abs": 0.0,
        "interp_sigma_scale": 1.0,
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
    cfg.update(overrides)
    return cfg


# ── Run definitions ─────────────────────────────────────────────────────────

def _both(name, **kw):
    """Return two entries (chest + pepper) for a named setting."""
    return {
        f"chest-{name}": base_config(data_path=CHEST_DATA, **kw),
        f"pepper-{name}": base_config(data_path=PEPPER_DATA, **kw),
    }


ALL_RUNS = {}

# ── Stage A: baselines ───────────────────────────────────────────────────────
ALL_RUNS.update(_both("best427"))  # current best, interp off

ALL_RUNS.update(_both(
    "vanilla",
    # strip targeted / high-error / grad-smooth / redundancy pruning
    targeted_fraction=0.0,
    high_error_fraction=0.0,
    grad_smooth_hops=0,
    redundancy_cap=0.0,
))

# ── Stage B1: neighbor variance (nvar) ──────────────────────────────────────
for weight, wtag in [(1e-4, "w1e4"), (1e-3, "w1e3"), (1e-2, "w1e2")]:
    for hops, htag in [(1, "h1"), (2, "h2")]:
        ALL_RUNS.update(_both(
            f"nvar-{wtag}-{htag}",
            neighbor_var_weight=weight,
            neighbor_var_hops=hops,
            neighbor_var_start=1000,  # densify_from
        ))

# ── Stage B2: voxel variance (vvar) ─────────────────────────────────────────
for weight, wtag in [(1e-4, "w1e4"), (1e-3, "w1e3"), (1e-2, "w1e2")]:
    for res, rtag in [(32, "r32"), (64, "r64")]:
        ALL_RUNS.update(_both(
            f"vvar-{wtag}-{rtag}",
            voxel_var_weight=weight,
            voxel_var_resolution=res,
            voxel_var_start=1000,  # densify_from
        ))

# ── Stage B3: TV ─────────────────────────────────────────────────────────────
for weight, wtag in [(1e-4, "w1e4"), (1e-3, "w1e3")]:
    for on_raw, rtag in [(True, "raw"), (False, "act")]:
        ALL_RUNS.update(_both(
            f"tv-{wtag}-{rtag}",
            tv_weight=weight,
            tv_start=5000,
            tv_on_raw=on_raw,
        ))

# ── Stage B4: bilateral filter ───────────────────────────────────────────────
# light: spatial 2.0→0.3, value 10.0→0.5
ALL_RUNS.update(_both(
    "bf-light",
    bf_start=2000, bf_until=6000, bf_period=10,
    bf_sigma_init=2.0, bf_sigma_final=0.3,
    bf_sigma_v_init=10.0, bf_sigma_v_final=0.5,
))
# medium: spatial 1.5→0.3, value 5.0→0.2
ALL_RUNS.update(_both(
    "bf-medium",
    bf_start=2000, bf_until=6000, bf_period=10,
    bf_sigma_init=1.5, bf_sigma_final=0.3,
    bf_sigma_v_init=5.0, bf_sigma_v_final=0.2,
))
# aggressive: spatial 1.0→0.2, value 2.0→0.1
ALL_RUNS.update(_both(
    "bf-aggressive",
    bf_start=2000, bf_until=6000, bf_period=10,
    bf_sigma_init=1.0, bf_sigma_final=0.2,
    bf_sigma_v_init=2.0, bf_sigma_v_final=0.1,
))

# ── Stage B5: redundancy pruning ─────────────────────────────────────────────
ALL_RUNS.update(_both("pruning-off",     redundancy_cap=0.0))
ALL_RUNS.update(_both("pruning-cap05",   redundancy_cap=0.05))
ALL_RUNS.update(_both(
    "pruning-var",
    prune_variance_criterion=True,
    redundancy_cap=0.03,
))

# ── Stage B6: loss function ───────────────────────────────────────────────────
ALL_RUNS.update(_both("loss-l2", loss_type="l2"))

# ── Stage B7: refvol regularization (scratch init, .pt reference) ─────────────
ALL_RUNS.update(_both(
    "refvol-w1e3",
    ref_volume_weight=1e-3,
    ref_volume_start=0,
))
ALL_RUNS["chest-refvol-w1e3"]["ref_volume_path"] = CHEST_INIT_PT
ALL_RUNS["pepper-refvol-w1e3"]["ref_volume_path"] = PEPPER_INIT_PT


# ── Infrastructure ─────────────────────────────────────────────────────────────

def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def sigma_sweep_csv_path(name):
    return os.path.join(SWEEP_DIR, name, "sigma_sweep.csv")


def parse_metrics(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                metrics[key] = float(m.group(2))
    return metrics


def parse_sigma_sweep_csv(path):
    """Return a flat dict of sigma-sweep metrics, prefixed with 'idw_'."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    # Only one row expected (1×1 σ grid)
    r = rows[0]
    flat = {}
    for k, v in r.items():
        try:
            flat[k] = float(v)
        except ValueError:
            flat[k] = v
    return flat


def run_experiment(name, cfg):
    out_dir = os.path.join(SWEEP_DIR, name)
    mpath = metrics_path(name)

    if os.path.exists(mpath):
        print(f"[SKIP] {name} — metrics.txt already exists")
    else:
        os.makedirs(out_dir, exist_ok=True)
        config_file = os.path.join(out_dir, "sweep_config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        cmd = [
            sys.executable, "train.py",
            "-c", config_file,
            "--experiment_name", f"sweep36_validate_no_interp/{name}",
        ]
        print(f"[RUN]  {name}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if result.returncode != 0:
            print(f"[FAIL] {name} exited with code {result.returncode}")
            return False
        if not os.path.exists(mpath):
            print(f"[WARN] {name} finished but metrics.txt not found")
            return False

    # Post-hoc σ evaluation
    run_eval(name, cfg)
    return True


def run_eval(name, cfg):
    """Run eval_sigma_sweep.py on an existing checkpoint with dataset-specific σ."""
    config_yaml = os.path.join(SWEEP_DIR, name, "config.yaml")
    if not os.path.exists(config_yaml):
        print(f"[WARN] {name}: config.yaml not found, skipping eval")
        return False

    sigma_csv = sigma_sweep_csv_path(name)
    if os.path.exists(sigma_csv):
        print(f"[SKIP] {name} — sigma_sweep.csv already exists")
        return True

    data_path = cfg["data_path"]
    if data_path not in EVAL_SIGMA:
        print(f"[WARN] {name}: no eval σ defined for data_path={data_path}, skipping eval")
        return False

    sigma_cfg = EVAL_SIGMA[data_path]
    cmd = [
        sys.executable, "eval_sigma_sweep.py",
        "--config", config_yaml,
        "--grid-spatial", *[str(s) for s in sigma_cfg["spatial"]],
        "--grid-sigma-v", *[str(v) for v in sigma_cfg["sigma_v"]],
    ]
    print(f"[EVAL] {name}  σ_s={sigma_cfg['spatial']}  σ_v={sigma_cfg['sigma_v']}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[WARN] {name} eval failed with code {result.returncode}")
        return False
    return True


def collect_summary(names, output_csv, sort_key="vol_idw_f1_1v"):
    rows = []
    for name in names:
        mpath = metrics_path(name)
        if not os.path.exists(mpath):
            continue
        row = {"name": name, **parse_metrics(mpath)}

        sigma_csv = sigma_sweep_csv_path(name)
        if os.path.exists(sigma_csv):
            sigma_data = parse_sigma_sweep_csv(sigma_csv)
            row.update(sigma_data)

        rows.append(row)

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

    print(f"[DONE] Summary → {output_csv}  ({len(rows)} runs, sorted by {sort_key})")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Sweep 36 — re-validate regularization with interp OFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; only run post-hoc σ eval on existing checkpoints")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", type=int, metavar="W")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        # Group by prefix for readable output
        groups = {}
        for name in ALL_RUNS:
            parts = name.split("-")
            dataset = parts[0]
            group = "-".join(parts[1:]) if len(parts) > 1 else "base"
            groups.setdefault(group, []).append(name)

        print(f"\nSweep 36 — regularization validation, interp OFF — {len(ALL_RUNS)} runs total")
        print(f"  base: best427_nointerp on chest + pepper")
        print(f"  eval σ: chest (0.008/0.05), pepper (0.010/0.40)\n")
        prev_group = None
        for name, cfg in ALL_RUNS.items():
            parts = name.split("-")
            group = "-".join(parts[1:]) if len(parts) > 1 else "base"
            prefix = parts[1] if len(parts) > 1 else "A"
            if group != prev_group:
                stage_tag = {
                    "best427": "A", "vanilla": "A",
                }.get(group, "B")
                print(f"  --- Stage {stage_tag}: {group} ---")
                prev_group = group
            dp = cfg["data_path"].split("/")[-1]
            print(f"  {name:<38}  {dp}")
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
            print(f"[WARN] worker {args.worker}/{args.num_workers} has no runs")
            return
        print(f"Sweep 36: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 36: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            cfg = ALL_RUNS[name]
            if args.eval_only:
                run_eval(name, cfg)
            else:
                run_experiment(name, cfg)

    collect_summary(
        all_names,
        os.path.join(SWEEP_DIR, "summary.csv"),
        sort_key="vol_idw_f1_1v",
    )


if __name__ == "__main__":
    main()
