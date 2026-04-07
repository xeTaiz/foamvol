#!/usr/bin/env python3
"""Sweep 18: High-error sampling + Voxel TV regularization.

Tests high-error ray sampling parameters and voxel-grid TV regularization
at 512k and 1M cells on the 500-projection dataset.

Part A: High-error sampling (HE) variants
Part B: Voxel TV variants
Part C: HE + Voxel TV combinations

Total: ~56 runs across 4 workers

Usage:
    python sweep_he_tv.py                    # all runs
    python sweep_he_tv.py --runs A01 B03     # specific runs
    python sweep_he_tv.py --worker 1 --of 4  # distributed
    python sweep_he_tv.py --list             # list all
    python sweep_he_tv.py --summarize        # collect results
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep18_he_tv"
DATA_PATH = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_500_angle_360/0_chest_cone"


def make_config(final_points=512000, iterations=10000,
                he_fraction=0.0, he_power=1.0, he_start=-1,
                voxel_tv_weight=0.0, voxel_tv_resolution=32, voxel_tv_start=0,
                interp=True, full_densify=True):
    """Build a config dict."""
    cfg = {
        "iterations": iterations,
        "rays_per_batch": 1000000,
        "init_points": 64000,
        "final_points": final_points,
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
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "freeze_points": 9500,
        # TV OFF (neighbor-based)
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        # Voxel TV
        "voxel_tv_weight": voxel_tv_weight,
        "voxel_tv_resolution": voxel_tv_resolution,
        "voxel_tv_start": voxel_tv_start,
        # Densification
        "gradient_fraction": 0.4 if full_densify else 1.0,
        "idw_fraction": 0.3 if full_densify else 0.0,
        "entropy_fraction": 0.3 if full_densify else 0.0,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.0,
        # Targeted sampling OFF
        "targeted_fraction": 0.0,
        "targeted_start": -1,
        # High-error sampling
        "high_error_fraction": he_fraction,
        "high_error_power": he_power,
        "high_error_start": he_start,
        # Interpolation
        "interpolation_start": 9000 if interp else -1,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": True,
        "per_neighbor_sigma": True,
        # BF OFF
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        # Gaussians OFF
        "gaussian_start": -1,
        "freeze_base_at_gaussian": False,
        "joint_finetune_start": -1,
        "peak_lr_init": 1e-2,
        "peak_lr_final": 1e-3,
        "offset_lr_init": 1e-3,
        "offset_lr_final": 1e-4,
        "cov_lr_init": 1e-2,
        "cov_lr_final": 1e-3,
        # Linear gradient OFF
        "gradient_start": -1,
        "gradient_lr_init": 1e-2,
        "gradient_lr_final": 1e-3,
        "gradient_warmup": 500,
        "gradient_max_slope": 5.0,
        "gradient_freeze_points": 500,
        # Dataset
        "dataset": "r2_gaussian",
        "data_path": DATA_PATH,
    }
    return cfg


def build_runs():
    runs = {}

    # =====================================================================
    # Baselines (no HE, no voxel TV) — reference for each cell count
    # =====================================================================
    for cells, tag in [(512000, "512k"), (1000000, "1M")]:
        runs[f"base-{tag}"] = make_config(final_points=cells)

    # =====================================================================
    # Part A: High-error sampling variants
    # =====================================================================
    for cells, tag in [(512000, "512k"), (1000000, "1M")]:
        # Vary fraction
        runs[f"A01-he10-{tag}"] = make_config(final_points=cells, he_fraction=0.1)
        runs[f"A02-he20-{tag}"] = make_config(final_points=cells, he_fraction=0.2)
        runs[f"A03-he30-{tag}"] = make_config(final_points=cells, he_fraction=0.3)
        runs[f"A04-he40-{tag}"] = make_config(final_points=cells, he_fraction=0.4)

        # Vary power (with fraction=0.2)
        runs[f"A05-he20-p15-{tag}"] = make_config(final_points=cells, he_fraction=0.2, he_power=1.5)
        runs[f"A06-he20-p20-{tag}"] = make_config(final_points=cells, he_fraction=0.2, he_power=2.0)
        runs[f"A07-he20-p30-{tag}"] = make_config(final_points=cells, he_fraction=0.2, he_power=3.0)

        # Vary start iteration (with fraction=0.2)
        runs[f"A08-he20-s0-{tag}"] = make_config(final_points=cells, he_fraction=0.2, he_start=0)
        runs[f"A09-he20-s3k-{tag}"] = make_config(final_points=cells, he_fraction=0.2, he_start=3000)

    # =====================================================================
    # Part B: Voxel TV variants
    # =====================================================================
    for cells, tag in [(512000, "512k"), (1000000, "1M")]:
        # Vary weight (res=32)
        runs[f"B01-vtv1e3-{tag}"] = make_config(final_points=cells, voxel_tv_weight=1e-3)
        runs[f"B02-vtv1e2-{tag}"] = make_config(final_points=cells, voxel_tv_weight=1e-2)
        runs[f"B03-vtv5e2-{tag}"] = make_config(final_points=cells, voxel_tv_weight=5e-2)
        runs[f"B04-vtv1e1-{tag}"] = make_config(final_points=cells, voxel_tv_weight=1e-1)

        # Resolution 64 (with weight=1e-2)
        runs[f"B05-vtv1e2-r64-{tag}"] = make_config(final_points=cells, voxel_tv_weight=1e-2,
                                                      voxel_tv_resolution=64)
        runs[f"B06-vtv5e2-r64-{tag}"] = make_config(final_points=cells, voxel_tv_weight=5e-2,
                                                      voxel_tv_resolution=64)

    # =====================================================================
    # Part C: HE + Voxel TV combinations (best guesses)
    # =====================================================================
    for cells, tag in [(512000, "512k"), (1000000, "1M")]:
        runs[f"C01-he20-vtv1e2-{tag}"] = make_config(
            final_points=cells, he_fraction=0.2, voxel_tv_weight=1e-2)
        runs[f"C02-he20-vtv5e2-{tag}"] = make_config(
            final_points=cells, he_fraction=0.2, voxel_tv_weight=5e-2)
        runs[f"C03-he20-p20-vtv1e2-{tag}"] = make_config(
            final_points=cells, he_fraction=0.2, he_power=2.0, voxel_tv_weight=1e-2)
        runs[f"C04-he30-vtv1e2-{tag}"] = make_config(
            final_points=cells, he_fraction=0.3, voxel_tv_weight=1e-2)
        runs[f"C05-he20-vtv1e2-r64-{tag}"] = make_config(
            final_points=cells, he_fraction=0.2, voxel_tv_weight=1e-2,
            voxel_tv_resolution=64)

    return runs


ALL_RUNS = build_runs()


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
        print(f"[SKIP] {name} \u2014 metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)

    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable,
        "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep18_he_tv/{name}",
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
        metrics = parse_metrics(mpath)
        rows.append({"name": name, **metrics})

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
        description="HE sampling + Voxel TV sweep (sweep 18)",
        epilog="Examples:\n"
               "  python sweep_he_tv.py                    # all runs\n"
               "  python sweep_he_tv.py --runs A01 B03     # specific runs\n"
               "  python sweep_he_tv.py --worker 1 --of 4  # distributed\n"
               "  python sweep_he_tv.py --list             # show run names\n"
               "  python sweep_he_tv.py --summarize        # just collect results\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID",
                        help="Only run these specific run IDs")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip training, just collect existing results into summary CSV")
    parser.add_argument("--worker", type=int, metavar="W",
                        help="Worker index (1-indexed)")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers",
                        help="Total number of workers")
    parser.add_argument("--list", action="store_true",
                        help="Print all run names and exit")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 18 runs:")
        for name, cfg in ALL_RUNS.items():
            pts = cfg["final_points"]
            he = cfg["high_error_fraction"]
            hp = cfg["high_error_power"]
            hs = cfg["high_error_start"]
            vw = cfg["voxel_tv_weight"]
            vr = cfg["voxel_tv_resolution"]
            parts = []
            if he > 0:
                parts.append(f"HE={he:.0%} p={hp}")
                if hs >= 0:
                    parts.append(f"s={hs}")
            if vw > 0:
                parts.append(f"VTV={vw} r={vr}")
            if not parts:
                parts.append("(baseline)")
            print(f"  {name:35s}  {pts:>7,} cells  {' '.join(parts)}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)

    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        names = [n for n in all_names if n in selected]
        unknown = selected - set(all_names)
        for u in unknown:
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    if args.worker is not None and args.num_workers is not None:
        names = names[args.worker - 1::args.num_workers]
        print(f"Sweep 18: worker {args.worker}/{args.num_workers} \u2014 "
              f"{len(names)} runs: {', '.join(names)}")
    else:
        print(f"Sweep 18: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
