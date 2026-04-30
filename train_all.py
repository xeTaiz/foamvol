#!/usr/bin/env python3
"""Train across all R2-Gaussian datasets, optionally with multiple configs.

Recursively discovers datasets (directories containing proj_train/) under
--data-root. Multiple configs produce a (configs × datasets) job pool that
workers split round-robin.

Output layout:
    output/<name>/<config_slug>/<dataset>/metrics.txt

Usage:
    # single config, all datasets
    python train_all.py -c configs/r2fast.yaml --name myrun

    # multiple configs — Cartesian product
    python train_all.py -c configs/thresh_100.yaml configs/thresh_500.yaml --name sweep25

    # worker splitting (round-robin over entire job pool)
    python train_all.py -c configs/thresh_*.yaml --name sweep25 --worker 1 --of 4

    # summarize (no -c needed; auto-discovers all config slugs under output/<name>/)
    python train_all.py --name sweep25 --summarize

    # list discovered datasets
    python train_all.py --list --data-root r2_data
"""

import argparse
import csv
import fnmatch
import os
import re
import subprocess
import sys

DATA_ROOT = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360"


def discover_datasets(data_roots):
    """Recursively find dataset directories (those containing proj_train/)."""
    datasets = []
    for data_root in data_roots:
        for dirpath, dirnames, _filenames in os.walk(data_root):
            if "proj_train" in dirnames:
                rel = os.path.relpath(dirpath, data_root)
                datasets.append((data_root, rel))
                dirnames.clear()
    datasets.sort(key=lambda x: x[1])
    return datasets


def filter_datasets(datasets, patterns):
    """Keep datasets where relative path matches any pattern (substring or glob)."""
    filtered = []
    for root, rel in datasets:
        for pat in patterns:
            if pat in rel or fnmatch.fnmatch(rel, pat):
                filtered.append((root, rel))
                break
    return filtered


def expand_configs(config_args):
    """Expand a mixed list of files and directories into a sorted list of .yaml paths."""
    configs = []
    for c in config_args:
        if os.path.isdir(c):
            configs.extend(sorted(
                os.path.join(c, f) for f in os.listdir(c) if f.endswith(".yaml")
            ))
        else:
            configs.append(c)
    return configs


def config_slug(config_file):
    """Derive a short directory name from a config file path."""
    return os.path.splitext(os.path.basename(config_file))[0]


def parse_metrics(path):
    """Parse a metrics.txt file into a dict of floats."""
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                try:
                    metrics[key] = float(m.group(2))
                except ValueError:
                    pass
    return metrics


def run_dataset(data_root, ds_name, config_file, run_name):
    """Run train.py on one (config, dataset) pair. Returns True on success."""
    mpath = os.path.join("output", run_name, ds_name, "metrics.txt")

    if os.path.exists(mpath):
        print(f"[SKIP] {run_name}/{ds_name} — metrics.txt already exists")
        return True

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"{run_name}/{ds_name}",
        "--data_path", os.path.join(data_root, ds_name),
    ]
    print(f"[RUN]  {run_name}/{ds_name}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"[FAIL] {run_name}/{ds_name} exited with code {result.returncode}")
        return False

    if not os.path.exists(mpath):
        print(f"[WARN] {run_name}/{ds_name} finished but metrics.txt not found")
        return False

    return True


def collect_summary(run_name, jobs=None):
    """Collect metrics into a summary CSV.

    If jobs is given, use that ordering; otherwise auto-discover from disk.
    Outputs output/<run_name>/summary.csv with a 'config' column.
    Appends MEAN/STD rows per config slug.
    """
    run_dir = os.path.join("output", run_name)

    if jobs is not None:
        # (config_file, slug, data_root, ds_name)
        pairs = [(slug, ds_name) for _, slug, _, ds_name in jobs]
    else:
        # Auto-discover: output/<run_name>/<slug>/<dataset>/metrics.txt
        pairs = []
        if not os.path.isdir(run_dir):
            print(f"[WARN] {run_dir} does not exist")
            return
        for slug in sorted(os.listdir(run_dir)):
            slug_dir = os.path.join(run_dir, slug)
            if not os.path.isdir(slug_dir):
                continue
            for ds in sorted(os.listdir(slug_dir)):
                if os.path.isfile(os.path.join(slug_dir, ds, "metrics.txt")):
                    pairs.append((slug, ds))

    rows = []
    for slug, ds_name in pairs:
        mpath = os.path.join(run_dir, slug, ds_name, "metrics.txt")
        if not os.path.exists(mpath):
            continue
        metrics = parse_metrics(mpath)
        rows.append({"config": slug, "name": ds_name, **metrics})

    if not rows:
        print("[WARN] No completed runs to summarize")
        return

    # Collect metric keys in first-seen order
    seen: set = set()
    metric_keys = []
    for row in rows:
        for k in row:
            if k not in ("config", "name") and k not in seen:
                metric_keys.append(k)
                seen.add(k)
    fieldnames = ["config", "name"] + metric_keys

    # Append MEAN/STD rows grouped by config slug
    slugs_seen = list(dict.fromkeys(slug for slug, _ in pairs))
    for slug in slugs_seen:
        slug_rows = [r for r in rows if r["config"] == slug]
        if not slug_rows:
            continue
        mean_row = {"config": slug, "name": "MEAN"}
        std_row = {"config": slug, "name": "STD"}
        for k in metric_keys:
            vals = [r[k] for r in slug_rows if k in r]
            if vals:
                avg = sum(vals) / len(vals)
                mean_row[k] = avg
                std_row[k] = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        rows += [mean_row, std_row]

    output_csv = os.path.join(run_dir, "summary.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    n_runs = len([r for r in rows if r["name"] not in ("MEAN", "STD")])
    print(f"[DONE] Summary written to {output_csv} ({n_runs} runs, {len(slugs_seen)} configs)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Train across all R2-Gaussian datasets",
        epilog=(
            "Examples:\n"
            "  python train_all.py -c configs/r2fast.yaml --name myrun\n"
            "  python train_all.py -c configs/thresh_100.yaml configs/thresh_500.yaml --name sweep25\n"
            "  python train_all.py -c configs/thresh_*.yaml --name sweep25 --worker 1 --of 4\n"
            "  python train_all.py --name sweep25 --summarize\n"
            "  python train_all.py --list --data-root r2_data\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", nargs="+", metavar="FILE",
                        help="One or more config YAML files (required unless --summarize or --list)")
    parser.add_argument("--name", metavar="NAME",
                        help="Run name / output root (required unless --list)")
    parser.add_argument("--worker", type=int, metavar="W",
                        help="Worker index (1-indexed)")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers",
                        help="Total number of workers")
    parser.add_argument("--summarize", action="store_true",
                        help="Skip training; collect existing results into summary CSV")
    parser.add_argument("--list", action="store_true",
                        help="Print all dataset names and exit")
    parser.add_argument("--datasets", nargs="+", metavar="DS",
                        help="Run only specific datasets by exact relative path")
    parser.add_argument("--filter", nargs="+", metavar="PAT",
                        help="Keep datasets matching any pattern (substring or glob)")
    parser.add_argument("--data-root", nargs="+", default=[DATA_ROOT], metavar="DIR",
                        help=f"Data root path(s) to scan (default: {DATA_ROOT})")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    # Discover and filter datasets
    all_datasets = discover_datasets(args.data_root)
    datasets = all_datasets
    if args.filter:
        datasets = filter_datasets(datasets, args.filter)
    if args.datasets:
        selected = set(args.datasets)
        datasets = [(r, d) for r, d in datasets if d in selected]
        found = {d for _, d in datasets}
        for u in selected - found:
            print(f"[WARN] Unknown dataset: {u}")

    if args.list:
        roots_str = ", ".join(args.data_root)
        print(f"\n{len(datasets)} datasets under {roots_str}:")
        for _root, rel in datasets:
            print(f"  {rel}")
        return

    if not args.name:
        parser.error("--name is required")

    # Summarize-only: no config needed
    if args.summarize and not args.config:
        collect_summary(args.name)
        return

    if not args.config:
        parser.error("-c/--config is required unless --summarize")

    # Expand any directory entries in --config to sorted lists of .yaml files
    configs = expand_configs(args.config)
    if not configs:
        parser.error("No .yaml files found in the provided --config paths")

    # Build job pool: (config_file, slug, data_root, ds_name)
    all_jobs = [
        (cfg, config_slug(cfg), root, ds)
        for cfg in configs
        for root, ds in datasets
    ]

    print(f"{len(configs)} config(s) × {len(datasets)} datasets = {len(all_jobs)} jobs total")

    # Worker splitting (round-robin over entire pool)
    if args.worker is not None:
        my_jobs = all_jobs[args.worker - 1 :: args.num_workers]
        print(f"Worker {args.worker}/{args.num_workers} — {len(my_jobs)} jobs")
    else:
        my_jobs = all_jobs

    if not args.summarize:
        for cfg, slug, root, ds in my_jobs:
            run_dataset(root, ds, cfg, f"{args.name}/{slug}")

    collect_summary(args.name, jobs=all_jobs)


if __name__ == "__main__":
    main()
