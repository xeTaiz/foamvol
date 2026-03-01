#!/usr/bin/env python3
"""Train across all R2-Gaussian datasets.

Recursively discovers datasets (directories containing proj_train/) under
--data-root, so you can point it at any level of the hierarchy:

    --data-root .../cone_ntrain_75_angle_360   →  15 datasets
    --data-root .../synthetic_dataset          →  45 datasets (3 view counts)
    --data-root .../r2_data                    →  all synthetic + real datasets

Usage:
    python train_all.py -c configs/r2fast.yaml --name myrun
    python train_all.py -c configs/r2fast.yaml --name myrun --worker 1 --of 4
    python train_all.py --name myrun --summarize
    python train_all.py --list --data-root /mnt/hdd/r2_data
"""

import argparse
import csv
import os
import re
import subprocess
import sys

DATA_ROOT = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360"


def discover_datasets(data_root):
    """Recursively find dataset directories (those containing proj_train/).

    Returns sorted list of paths relative to data_root.
    """
    datasets = []
    for dirpath, dirnames, _filenames in os.walk(data_root):
        if "proj_train" in dirnames:
            rel = os.path.relpath(dirpath, data_root)
            datasets.append(rel)
            dirnames.clear()  # don't recurse into dataset subdirs
    return sorted(datasets)


def parse_metrics(path):
    """Parse a metrics.txt file into a dict of floats."""
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                val = float(m.group(2))
                metrics[key] = val
    return metrics


def run_dataset(name, config_file, run_name, data_root):
    """Run train.py on a single dataset. Returns True on success."""
    mpath = os.path.join("output", run_name, name, "metrics.txt")

    if os.path.exists(mpath):
        print(f"[SKIP] {name} — metrics.txt already exists")
        return True

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"{run_name}/{name}",
        "--data_path", os.path.join(data_root, name),
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


def collect_summary(datasets, run_name):
    """Read metrics.txt from each dataset and write summary CSV with MEAN/STD rows."""
    rows = []
    for name in datasets:
        mpath = os.path.join("output", run_name, name, "metrics.txt")
        if not os.path.exists(mpath):
            continue
        metrics = parse_metrics(mpath)
        rows.append({"name": name, **metrics})

    if not rows:
        print("[WARN] No completed datasets to summarize")
        return

    # Determine columns
    metric_keys = [k for k in rows[0] if k != "name"]
    fieldnames = ["name"] + metric_keys

    # Compute MEAN and STD
    n = len(rows)
    mean_row = {"name": "MEAN"}
    std_row = {"name": "STD"}
    for k in metric_keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            avg = sum(vals) / len(vals)
            mean_row[k] = avg
            std_row[k] = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5

    rows_out = rows + [mean_row, std_row]

    output_csv = os.path.join("output", run_name, "summary.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[DONE] Summary written to {output_csv} ({n} datasets + MEAN/STD)")
    return rows_out


def main():
    parser = argparse.ArgumentParser(
        description="Train across all R2-Gaussian datasets",
        epilog="Examples:\n"
               "  python train_all.py -c configs/r2fast.yaml --name myrun\n"
               "  python train_all.py -c configs/r2fast.yaml --name myrun --worker 1 --of 4\n"
               "  python train_all.py --name myrun --summarize\n"
               "  python train_all.py --list --data-root /mnt/hdd/r2_data\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", metavar="FILE",
                        help="Config YAML file (required unless --summarize or --list)")
    parser.add_argument("--name", metavar="NAME",
                        help="Run name / output directory (required unless --list)")
    parser.add_argument("--worker", type=int, metavar="W",
                        help="Worker index (1-indexed)")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers",
                        help="Total number of workers")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip training, just collect existing results into summary CSV")
    parser.add_argument("--list", action="store_true",
                        help="Print all dataset names and exit")
    parser.add_argument("--datasets", nargs="+", metavar="DS",
                        help="Run only specific datasets by name (relative path)")
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help=f"Override data root path (default: {DATA_ROOT})")
    args = parser.parse_args()

    # Validation
    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    all_datasets = discover_datasets(args.data_root)

    if args.list:
        print(f"\n{len(all_datasets)} datasets under {args.data_root}:")
        for ds in all_datasets:
            print(f"  {ds}")
        return

    if not args.name:
        parser.error("--name is required")
    if not args.summarize and not args.config:
        parser.error("-c/--config is required unless --summarize")

    # Filter datasets
    if args.datasets:
        selected = set(args.datasets)
        datasets = [d for d in all_datasets if d in selected]
        unknown = selected - set(all_datasets)
        for u in unknown:
            print(f"[WARN] Unknown dataset: {u}")
    else:
        datasets = all_datasets

    # Worker splitting (round-robin)
    if args.worker is not None:
        datasets = datasets[args.worker - 1::args.num_workers]
        print(f"Worker {args.worker}/{args.num_workers} — {len(datasets)} datasets: {', '.join(datasets)}")
    else:
        print(f"{len(datasets)}/{len(all_datasets)} datasets selected")

    # Train
    if not args.summarize:
        for ds in datasets:
            run_dataset(ds, args.config, args.name, args.data_root)

    # Summarize (always use all datasets, not just this worker's slice)
    collect_summary(all_datasets, args.name)


if __name__ == "__main__":
    main()
