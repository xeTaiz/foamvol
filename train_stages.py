"""
Sequential multi-stage training coordinator.

Each stage is a separate train.py run. Stage N+1 automatically receives the
frozen checkpoint from Stage N via --frozen_points_file, and --final_points is
computed as N_frozen + cell_increment so the densification target is always
"add cell_increment new cells on top of whatever the previous stage produced".

Stage names use <name>/stage_N so TensorBoard groups them under a single run.

Usage:
    python train_stages.py --configs stage1.yaml stage2.yaml --name my_run
    python train_stages.py --configs stage1.yaml stage2.yaml stage2.yaml stage2.yaml --name my_run
    python train_stages.py --configs stage1.yaml stage2.yaml --name my_run --cell-increment 32000
    python train_stages.py --configs stage1.yaml stage2.yaml --name my_run --summarize
    python train_stages.py --configs stage1.yaml stage2.yaml --name my_run --list
"""

import argparse
import csv
import os
import subprocess
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-stage RadFoam training")
    parser.add_argument(
        "--configs", nargs="+", required=True, metavar="YAML",
        help="Stage config files in order; repeat a file to run the same stage multiple times",
    )
    parser.add_argument(
        "--name", required=True,
        help="Base experiment name; stages land in output/<name>/stage_N/",
    )
    parser.add_argument(
        "--cell-increment", type=int, default=64000, metavar="N",
        help="Cells added per stage (default: 64000). Used to compute final_points for each "
             "stage after stage 1 as N_frozen + cell_increment.",
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="Collect metrics.txt from all stage dirs into summary.csv and exit",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print resolved stage names and exit without running",
    )
    return parser.parse_args()


def stage_experiment_name(base_name, stage_idx):
    # Slash-separated so TensorBoard groups all stages under base_name
    return f"{base_name}/stage_{stage_idx + 1}"


def stage_output_dir(base_name, stage_idx):
    return os.path.join("output", base_name, f"stage_{stage_idx + 1}")


def read_num_points(model_pt_path):
    """Return the number of points in a saved model.pt."""
    data = torch.load(model_pt_path, map_location="cpu", weights_only=True)
    return data["xyz"].shape[0]


def run_stage(config_path, extra_overrides):
    cmd = [sys.executable, "train.py", "-c", config_path]
    for k, v in extra_overrides.items():
        cmd += [f"--{k}", str(v)]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)


def summarize(configs, base_name):
    rows = []
    for i in range(len(configs)):
        out_dir = stage_output_dir(base_name, i)
        metrics_path = os.path.join(out_dir, "metrics.txt")
        if not os.path.exists(metrics_path):
            print(f"[skip] {metrics_path} not found")
            continue
        with open(metrics_path) as f:
            content = f.read()
        row = {"stage": i + 1, "experiment": stage_experiment_name(base_name, i)}
        for line in content.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                row[k.strip()] = v.strip()
        rows.append(row)

    if not rows:
        print("No metrics found.")
        return

    os.makedirs(os.path.join("output", base_name), exist_ok=True)
    out_csv = os.path.join("output", base_name, "summary.csv")
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


def main():
    args = parse_args()

    if args.list:
        for i, cfg in enumerate(args.configs):
            exp_name = stage_experiment_name(args.name, i)
            out_dir = stage_output_dir(args.name, i)
            print(f"  Stage {i+1}: {cfg}  →  {out_dir}/  (experiment_name={exp_name})")
        return

    if args.summarize:
        summarize(args.configs, args.name)
        return

    prev_model_pt = None
    for i, cfg_path in enumerate(args.configs):
        if not os.path.exists(cfg_path):
            sys.exit(f"Config not found: {cfg_path}")

        exp_name = stage_experiment_name(args.name, i)
        overrides = {"experiment_name": exp_name}

        if prev_model_pt is not None:
            n_frozen = read_num_points(prev_model_pt)
            final_points = n_frozen + args.cell_increment
            overrides["frozen_points_file"] = prev_model_pt
            overrides["final_points"] = final_points
            print(f"[stage {i+1}] {n_frozen} frozen cells + {args.cell_increment} increment "
                  f"→ final_points={final_points}")

        run_stage(cfg_path, overrides)

        model_pt = os.path.join(stage_output_dir(args.name, i), "model.pt")
        if not os.path.exists(model_pt):
            sys.exit(f"Stage {i+1} did not produce {model_pt} — aborting")
        prev_model_pt = model_pt

    print(f"\nAll {len(args.configs)} stages complete.")


if __name__ == "__main__":
    main()
