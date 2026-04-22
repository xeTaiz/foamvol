#!/usr/bin/env python3
"""Sweep 30: IDW interpolation mode ablation on 75-view data.

Tests whether IDW rendering still helps given current regularization (neighbor
variance + adaptive pruning), and which per_*_sigma variant is optimal.

Motivation: at interpolation_start=9000, the forward model switches to IDW
(low-pass). Both RAW and IDW slice metrics drop at the switch because sub-sigma
detail enters the null space of the projection loss. The historical volume-PSNR
win was recorded before strong regularizers were in place — those now do similar
smoothing on the constant-cell field. This sweep answers: is IDW still beneficial?

Sigma mechanics when per_cell=False:
    sigma = interp_sigma_scale * cell_radius.median()  (world-space, auto-calibrated)
When per_cell=True:
    sigma = interp_sigma_scale * cell_radius_i  (per-cell adaptive)
With interp_sigma_scale=0.7 and ~512k cells, sigma ≈ 0.014 world units ≈ 1.8 voxels.

Runs:
  V00  no interpolation  (baseline — constant-cell rendering throughout)
  V01  interp on, global sigma  (per_cell=False, per_neighbor=False)
  V02  interp on, per-cell sigma  (per_cell=True, per_neighbor=False)
  V03  interp on, per-neighbor sigma  (per_cell=True, per_neighbor=True) ← current default

Usage:
    python sweep_30_interp.py
    python sweep_30_interp.py --runs V00 V01
    python sweep_30_interp.py --list
    python sweep_30_interp.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep30_interp"
BASE_CONFIG = "configs/r2_512k_best420_fdk.yaml"

with open(BASE_CONFIG) as _f:
    _BASE = yaml.safe_load(_f)


def base_config(**overrides):
    return {**_BASE, **overrides}


ALL_RUNS = {
    # No interpolation — constant-cell rendering throughout
    "V00-no-interp":        base_config(interpolation_start=-1),

    # Interp on, global sigma = interp_sigma_scale * median_cell_radius
    "V01-interp-global":    base_config(interpolation_start=9000,
                                        per_cell_sigma=False, per_neighbor_sigma=False),

    # Interp on, per-query-cell adaptive sigma (query uses its own radius)
    "V02-interp-percell":   base_config(interpolation_start=9000,
                                        per_cell_sigma=True, per_neighbor_sigma=False),

    # Interp on, per-neighbor sigma (each edge uses neighbor's radius) — current default
    "V03-interp-perneigh":  base_config(interpolation_start=9000,
                                        per_cell_sigma=True, per_neighbor_sigma=True),
}


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
        "--experiment_name", f"sweep30_interp/{name}",
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
        description="Interpolation mode ablation (sweep 30)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 30 runs:")
        for name, cfg in ALL_RUNS.items():
            i_start = cfg.get("interpolation_start", -1)
            pc = cfg.get("per_cell_sigma", False)
            pn = cfg.get("per_neighbor_sigma", False)
            scale = cfg.get("interp_sigma_scale", 0.7)
            sv = cfg.get("interp_sigma_v", 0.2)
            if i_start < 0:
                mode = "no interp"
            else:
                mode = f"interp@{i_start}  per_cell={pc}  per_neighbor={pn}  scale={scale}  sv={sv}"
            print(f"  {name:28s}  {mode}")
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

    print(f"Sweep 30: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
