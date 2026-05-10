#!/usr/bin/env python3
"""Sweep 25: Boundary-alignment regularizer weight sweep on 75-view data.

Base: configs/best428_nointerp.yaml (512k cells, variance pruning, HE+targeted sampling).
Sweeps boundary_align_weight logarithmically from 1e-4 to 1.0.
boundary_align_start/until default to densify_from/freeze_points.
var_sigma_v_init/final from base config (50.0 → 0.2) are reused as σ_v schedule.

Runs:
  V00  baseline (boundary_align_weight=0)
  V01  weight=1e-4
  V02  weight=1e-3
  V03  weight=1e-2
  V04  weight=1e-1
  V05  weight=1.0

Usage:
    python sweep_25_bdalign.py
    python sweep_25_bdalign.py --runs V00 V01
    python sweep_25_bdalign.py --list
    python sweep_25_bdalign.py --summarize
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR  = "output/boundary_align"
BASE_CFG   = "configs/best428_nointerp.yaml"


def load_base():
    with open(BASE_CFG) as f:
        return yaml.safe_load(f)


def make_config(boundary_align_weight=0.0):
    cfg = load_base()
    cfg["boundary_align_weight"] = boundary_align_weight
    cfg["boundary_align_start"]  = -1   # densify_from
    cfg["boundary_align_until"]  = -1   # freeze_points
    return cfg


ALL_RUNS = {
    "V00-baseline":  make_config(boundary_align_weight=0.0),
    "V01-ba-1e-4":   make_config(boundary_align_weight=1e-4),
    "V02-ba-1e-3":   make_config(boundary_align_weight=1e-3),
    "V03-ba-1e-2":   make_config(boundary_align_weight=1e-2),
    "V04-ba-1e-1":   make_config(boundary_align_weight=1e-1),
    "V05-ba-1e0":    make_config(boundary_align_weight=1.0),
}


# ---------------------------------------------------------------------------
# Infrastructure (mirrors other sweep scripts)
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
                metrics[key] = float(m.group(2))
    return metrics


def run_experiment(name, cfg):
    out_dir = os.path.join(SWEEP_DIR, name)
    mpath   = metrics_path(name)

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
        "--experiment_name", f"sweep25_bdalign/{name}",
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
        description="Boundary-alignment weight sweep on 75-view data (sweep 25)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 25 runs:")
        for name, cfg in ALL_RUNS.items():
            w = cfg["boundary_align_weight"]
            sv_i = cfg.get("var_sigma_v_init", "?")
            sv_f = cfg.get("var_sigma_v_final", "?")
            label = f"ba_weight={w}" if w > 0 else "baseline (no boundary align)"
            print(f"  {name:20s}  {label}  σ_v={sv_i}→{sv_f}")
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

    print(f"Sweep 25: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
