#!/usr/bin/env python3
"""Sweep 37 — pruning aggressiveness, grad-smoothing ablation, combinations.

Context:
  Sweep 36 found pruning-var (prune_variance_criterion=True, redundancy_cap=0.03)
  as the only consistent winner across all metrics on chest and pepper. This sweep
  follows up with:

  B — Pruning-var aggressiveness: vary redundancy_cap ∈ {0.01, 0.05, 0.10} ×2 datasets
      + pruning-var-h2 (prune_hops=2) — run AFTER seeing B results (use --runs to select).
  C — Grad-smooth ablation: grad_smooth_hops ∈ {0, 2} vs sweep-36 baseline (hops=1).
  D — Combinations of confirmed winners (add to ALL_RUNS after B/C/E complete).

After each training run, automatically:
  1. Runs eval_sigma_sweep.py with EVAL_GRID (12 combos: 3 σ_s × 4 σ_v).
  2. Runs sigma_sweep_to_tb.py to log best-σ scalars + heatmaps to the run's TB dir.

Stage E (pending sweep-36 refvol pair) is run separately via:
  python sweep_36_validate_no_interp.py --runs chest-refvol-w1e3 pepper-refvol-w1e3

Usage:
    python sweep_37_followups.py --list
    python sweep_37_followups.py                          # all runs
    python sweep_37_followups.py --runs chest-gs-off pepper-pruning-var-cap05
    python sweep_37_followups.py --worker 1 --of 2        # half the runs on this GPU
    python sweep_37_followups.py --summarize
    python sweep_37_followups.py --eval-only              # re-run σ sweep on existing checkpoints
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR   = "output/sweep37_followups"
CHEST_DATA  = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
PEPPER_DATA = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/1_pepper_cone"

# σ grid for per-run post-training eval (shared across all runs and datasets)
EVAL_GRID = {
    "spatial": [0.008, 0.010, 0.012],
    "sigma_v": [0.05, 0.10, 0.20, 0.30],
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
        # Sigma schedule for variance regs (unused since vvar/nvar off)
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
        # Pruning baseline (basic + low LOO-IDW redundancy cap)
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
        # Interpolation — OFF during training; σ used for end-of-training TB eval only
        "interpolation_start": -1,
        "interp_ramp": False,
        "interp_sigma_abs": 0.010,
        "interp_sigma_scale": 1.0,
        "interp_sigma_v": 0.10,
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


def _both(name, **kw):
    return {
        f"chest-{name}":  base_config(data_path=CHEST_DATA,  **kw),
        f"pepper-{name}": base_config(data_path=PEPPER_DATA, **kw),
    }


ALL_RUNS = {}

# ── Stage B: pruning-var aggressiveness ──────────────────────────────────────
# Baseline from sweep 36: prune_variance_criterion=True, cap=0.03, hops=1.
# Test other caps; h2 variant should ideally be run at the best cap from this batch.
for cap, ctag in [(0.01, "cap01"), (0.05, "cap05"), (0.10, "cap10")]:
    ALL_RUNS.update(_both(
        f"pruning-var-{ctag}",
        prune_variance_criterion=True,
        redundancy_cap=cap,
        prune_hops=1,
    ))

# prune_hops=2 at cap=0.03 (sweep-36 winner cap); re-run at best cap after B results
ALL_RUNS.update(_both(
    "pruning-var-h2-cap03",
    prune_variance_criterion=True,
    redundancy_cap=0.03,
    prune_hops=2,
))

# ── Stage C: grad-smooth ablation ────────────────────────────────────────────
# Baseline from sweep 36: grad_smooth_hops=1. Test 0 and 2.
ALL_RUNS.update(_both("gs-off", grad_smooth_hops=0))
ALL_RUNS.update(_both("gs-h2",  grad_smooth_hops=2))

# ── Stage D: combinations (add after B/C/E complete) ────────────────────────
# Placeholder — uncomment and customise once B/C/E results are in:
#
#   BEST_CAP = 0.03  # update from B results
#   BEST_GS  = 1     # update from C results
#
#   ALL_RUNS.update(_both("combo-pv-best",
#       prune_variance_criterion=True, redundancy_cap=BEST_CAP,
#   ))
#   ALL_RUNS.update(_both("combo-pv-refvol",
#       prune_variance_criterion=True, redundancy_cap=BEST_CAP,
#       ref_volume_path="output/init/chest/model.pt",  # override per dataset in _both
#       ref_volume_weight=1e-3, ref_volume_weight_final=-1.0,
#       ref_volume_start=0, ref_volume_until=-1,
#   ))


# ── Infrastructure ────────────────────────────────────────────────────────────

def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def sigma_sweep_csv_path(name):
    return os.path.join(SWEEP_DIR, name, "sigma_sweep.csv")


def parse_metrics(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+\-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                metrics[key] = float(m.group(2))
    return metrics


def parse_sigma_sweep_csv(path):
    """Return the single best row (by vol_idw_f1_1v) from sigma_sweep.csv."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get("vol_idw_f1_1v", 0)))
    return {f"sigma_{k}": float(v) for k, v in best.items()}


def run_eval(name, cfg):
    """Run eval_sigma_sweep.py + sigma_sweep_to_tb.py on a finished checkpoint."""
    out_dir = os.path.join(SWEEP_DIR, name)
    config_yaml = os.path.join(out_dir, "config.yaml")
    if not os.path.exists(config_yaml):
        print(f"[WARN] {name}: config.yaml not found, skipping eval")
        return False

    sigma_csv = sigma_sweep_csv_path(name)
    if not os.path.exists(sigma_csv):
        cmd = [
            sys.executable, "eval_sigma_sweep.py",
            "--config", config_yaml,
            "--grid-spatial", *[str(s) for s in EVAL_GRID["spatial"]],
            "--grid-sigma-v", *[str(v) for v in EVAL_GRID["sigma_v"]],
        ]
        print(f"[EVAL] {name}  σ_s={EVAL_GRID['spatial']}  σ_v={EVAL_GRID['sigma_v']}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"[WARN] {name} sigma_sweep failed (code {result.returncode})")
            return False

    if os.path.exists(sigma_csv):
        cmd = [
            sys.executable, "sigma_sweep_to_tb.py",
            "--csv", sigma_csv,
            "--tb-dir", out_dir,
        ]
        print(f"[TB]   {name}  heatmaps + best/* scalars")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"[WARN] {name} sigma_sweep_to_tb failed (code {result.returncode})")
            return False

    return True


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
            "--experiment_name", f"sweep37_followups/{name}",
        ]
        print(f"[RUN]  {name}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if result.returncode != 0:
            print(f"[FAIL] {name} exited with code {result.returncode}")
            return False
        if not os.path.exists(mpath):
            print(f"[WARN] {name} finished but metrics.txt not found")
            return False

    run_eval(name, cfg)
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
            row.update(parse_sigma_sweep_csv(sigma_csv))
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
        description="Sweep 37 — pruning aggressiveness + grad-smoothing ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID",
                        help="Run specific experiments by name")
    parser.add_argument("--summarize", action="store_true",
                        help="Only write summary CSV from completed runs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; re-run σ sweep on existing checkpoints")
    parser.add_argument("--list", action="store_true",
                        help="Print all run names and exit")
    parser.add_argument("--worker", type=int, metavar="W")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\nSweep 37 — {len(ALL_RUNS)} runs total")
        print(f"  σ sweep grid: σ_s={EVAL_GRID['spatial']}  σ_v={EVAL_GRID['sigma_v']}")
        print()
        for name, cfg in ALL_RUNS.items():
            tags = []
            if cfg.get("prune_variance_criterion"):
                tags.append(f"var-prune cap={cfg['redundancy_cap']} hops={cfg['prune_hops']}")
            elif cfg.get("redundancy_cap", 0) > 0:
                tags.append(f"loo-prune cap={cfg['redundancy_cap']}")
            gs = cfg.get("grad_smooth_hops", 1)
            if gs != 1:
                tags.append(f"gs={gs}")
            print(f"  {name:<36}  {', '.join(tags) or 'baseline-like'}")
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
        print(f"Sweep 37: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 37: {len(names)}/{len(all_names)} runs selected")

    if args.summarize:
        collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))
        return

    if args.eval_only:
        for name in names:
            run_eval(name, ALL_RUNS[name])
    else:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
