#!/usr/bin/env python3
"""Train across R2-Gaussian and 3D non-R2 datasets, optionally with multiple configs.

Recursively discovers R2 datasets (directories containing proj_train/) under
--data-root.  Additional 3D DICOM datasets (MORE, AAPM-Mayo) are discovered
via --more-root and --mayo-root, enumerating patients automatically.

Multiple configs produce a (configs × datasets) job pool that workers split
round-robin.

Output layout:
    output/<name>/<config_slug>/<dataset>/metrics.txt

Usage:
    # single config, all R2 datasets
    python train_all.py -c configs/r2fast.yaml --name myrun

    # multiple configs — Cartesian product
    python train_all.py -c configs/thresh_100.yaml configs/thresh_500.yaml --name sweep25

    # worker splitting (round-robin over entire job pool)
    python train_all.py -c configs/thresh_*.yaml --name sweep25 --worker 1 --of 4

    # include MORE and AAPM-Mayo subsets
    python train_all.py -c configs/fixed_final --name fixed_3d \\
        --more-root /mnt/hdd/more_subset --mayo-root /mnt/hdd/LDCT_Mayo_subset

    # summarize (no -c needed; auto-discovers all config slugs under output/<name>/)
    python train_all.py --name sweep25 --summarize

    # list discovered datasets (all sources)
    python train_all.py --list --data-root r2_data \\
        --more-root /mnt/hdd/more_subset --mayo-root /mnt/hdd/LDCT_Mayo_subset
"""

import argparse
import csv
import fnmatch
import glob
import os
import re
import subprocess
import sys

DATA_ROOT = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360"


def discover_datasets(data_roots):
    """Recursively find R2-Gaussian dataset directories (those containing proj_train/)."""
    datasets = []
    for data_root in data_roots:
        for dirpath, dirnames, _filenames in os.walk(data_root):
            if "proj_train" in dirnames:
                rel = os.path.relpath(dirpath, data_root)
                datasets.append((data_root, rel))
                dirnames.clear()
    datasets.sort(key=lambda x: x[1])
    return datasets


def discover_more(root):
    """Enumerate MORE DICOM studies, replicating MOREDataset._detect_layout sort order.

    Mirrors data_loader/more.py:65-77: sort by organ dir then patient dir,
    preferring full_1mm over full_3mm. Returns list of (ds_name, extra_args).
    sample_index values here match what the loader assigns at runtime.
    """
    entries = []
    idx = 0
    for organ_entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not organ_entry.is_dir():
            continue
        for patient_entry in sorted(os.scandir(organ_entry.path), key=lambda e: e.name):
            if not patient_entry.is_dir():
                continue
            series_1mm = os.path.join(patient_entry.path, "full_1mm")
            series_3mm = os.path.join(patient_entry.path, "full_3mm")
            if os.path.isdir(series_1mm) and glob.glob(os.path.join(series_1mm, "*.dcm")):
                thickness = "1mm"
            elif os.path.isdir(series_3mm) and glob.glob(os.path.join(series_3mm, "*.dcm")):
                thickness = "3mm"
            else:
                continue
            ds_name = f"more/{organ_entry.name}/{patient_entry.name}_{thickness}"
            extra_args = {"data_path": root, "dataset": "more", "sample_index": idx}
            entries.append((ds_name, extra_args))
            idx += 1
    return entries


def discover_aapm_mayo(root):
    """Enumerate AAPM-Mayo DICOM patients, replicating AAPMMayoDataset._detect_layout order.

    Mirrors data_loader/aapm_mayo.py:66-77: sorted top-level patient dirs that
    contain *.dcm or *.IMA files. Returns list of (ds_name, extra_args).
    """
    patient_dirs = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        dcm_files = (
            glob.glob(os.path.join(entry.path, "**", "*.dcm"), recursive=True)
            or glob.glob(os.path.join(entry.path, "**", "*.IMA"), recursive=True)
        )
        if dcm_files:
            patient_dirs.append(entry.name)

    entries = []
    for i, patient_id in enumerate(patient_dirs):
        ds_name = f"mayo/{patient_id}"
        extra_args = {"data_path": root, "dataset": "aapm_mayo", "sample_index": i}
        entries.append((ds_name, extra_args))
    return entries


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


def run_dataset(ds_name, config_file, run_name, extra_args, sigma_sweep=False):
    """Run train.py on one (config, dataset) pair. Returns True on success.

    extra_args: dict of CLI overrides forwarded to train.py, e.g.
        {"data_path": "/mnt/hdd/...", "dataset": "more", "sample_index": 3}
    sigma_sweep: if True, run eval_sigma_sweep.py immediately after training.
    """
    mpath = os.path.join("output", run_name, ds_name, "metrics.txt")

    if os.path.exists(mpath):
        print(f"[SKIP] {run_name}/{ds_name} — metrics.txt already exists")
    else:
        cmd = [
            sys.executable, "train.py",
            "-c", config_file,
            "--experiment_name", f"{run_name}/{ds_name}",
        ]
        for k, v in extra_args.items():
            cmd += [f"--{k}", str(v)]

        print(f"[RUN]  {run_name}/{ds_name}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if result.returncode != 0:
            print(f"[FAIL] {run_name}/{ds_name} exited with code {result.returncode}")
            return False

        if not os.path.exists(mpath):
            print(f"[WARN] {run_name}/{ds_name} finished but metrics.txt not found")
            return False

    if sigma_sweep:
        run_dir = os.path.join("output", run_name, ds_name)
        sweep_csv = os.path.join(run_dir, "sigma_sweep.csv")
        if os.path.exists(sweep_csv):
            print(f"[SKIP-σ] sigma_sweep.csv already exists for {run_name}/{ds_name}")
        else:
            config_path = os.path.join(run_dir, "config.yaml")
            print(f"[σ-SWEEP] {run_name}/{ds_name}")
            sr = subprocess.run(
                [sys.executable, "eval_sigma_sweep.py", "--config", config_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if sr.returncode != 0:
                print(f"[WARN] sigma-sweep failed for {run_name}/{ds_name} (code {sr.returncode})")

    return True


def collect_summary(run_name, jobs=None):
    """Collect metrics into a summary CSV.

    If jobs is given, use that ordering; otherwise auto-discover from disk.
    Outputs output/<run_name>/summary.csv with a 'config' column.
    Appends MEAN/STD rows per config slug.
    """
    run_dir = os.path.join("output", run_name)

    if jobs is not None:
        # (config_file, slug, ds_name, extra_args)
        pairs = [(slug, ds_name) for _, slug, ds_name, _ in jobs]
    else:
        # Auto-discover: recurse under output/<run_name>/<slug>/**/metrics.txt
        pairs = []
        if not os.path.isdir(run_dir):
            print(f"[WARN] {run_dir} does not exist")
            return
        for slug in sorted(os.listdir(run_dir)):
            slug_dir = os.path.join(run_dir, slug)
            if not os.path.isdir(slug_dir):
                continue
            for dirpath, _, filenames in os.walk(slug_dir):
                if "metrics.txt" in filenames:
                    ds_name = os.path.relpath(dirpath, slug_dir)
                    pairs.append((slug, ds_name))

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
        description="Train across R2-Gaussian and 3D non-R2 CT datasets",
        epilog=(
            "Examples:\n"
            "  python train_all.py -c configs/r2fast.yaml --name myrun\n"
            "  python train_all.py -c configs/thresh_100.yaml configs/thresh_500.yaml --name sweep25\n"
            "  python train_all.py -c configs/thresh_*.yaml --name sweep25 --worker 1 --of 4\n"
            "  python train_all.py -c configs/fixed_final --name fixed_3d \\\n"
            "      --more-root /mnt/hdd/more_subset --mayo-root /mnt/hdd/LDCT_Mayo_subset\n"
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
                        help="Run only specific R2 datasets by exact relative path")
    parser.add_argument("--filter", nargs="+", metavar="PAT",
                        help="Keep R2 datasets matching any pattern (substring or glob)")
    parser.add_argument("--data-root", nargs="+", default=[DATA_ROOT], metavar="DIR",
                        help=f"R2-Gaussian data root path(s) to scan (default: {DATA_ROOT})")
    parser.add_argument("--more-root", metavar="DIR",
                        help="Root of a MORE DICOM subset folder; enumerates all patients")
    parser.add_argument("--mayo-root", metavar="DIR",
                        help="Root of an AAPM-Mayo DICOM subset folder; enumerates all patients")
    parser.add_argument("--sigma-sweep", action="store_true",
                        help="Run eval_sigma_sweep.py on each run dir immediately after training")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    # --- R2-Gaussian jobs ---
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

    # (ds_name, extra_args) for non-R2 datasets
    non_r2_entries = []
    if args.more_root:
        entries = discover_more(args.more_root)
        print(f"Discovered {len(entries)} MORE patients under {args.more_root}")
        non_r2_entries.extend(entries)
    if args.mayo_root:
        entries = discover_aapm_mayo(args.mayo_root)
        print(f"Discovered {len(entries)} AAPM-Mayo patients under {args.mayo_root}")
        non_r2_entries.extend(entries)

    if args.list:
        total = len(datasets) + len(non_r2_entries)
        print(f"\n{len(datasets)} R2 datasets:")
        for _root, rel in datasets:
            print(f"  {rel}")
        if non_r2_entries:
            print(f"\n{len(non_r2_entries)} non-R2 datasets:")
            for ds_name, ea in non_r2_entries:
                print(f"  {ds_name}  (dataset={ea['dataset']} sample_index={ea['sample_index']})")
        print(f"\n{total} total")
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

    # Build job pool: (config_file, slug, ds_name, extra_args)
    # R2 jobs: extra_args contains only data_path (derived from root + ds)
    r2_jobs = [
        (cfg, config_slug(cfg), ds, {"data_path": os.path.join(root, ds)})
        for cfg in configs
        for root, ds in datasets
    ]
    # Non-R2 jobs: extra_args contains data_path, dataset, sample_index
    non_r2_jobs = [
        (cfg, config_slug(cfg), ds_name, extra_args)
        for cfg in configs
        for ds_name, extra_args in non_r2_entries
    ]
    all_jobs = r2_jobs + non_r2_jobs

    n_ds = len(datasets) + len(non_r2_entries)
    print(f"{len(configs)} config(s) × {n_ds} datasets = {len(all_jobs)} jobs total")

    # Worker splitting (round-robin over entire pool)
    if args.worker is not None:
        my_jobs = all_jobs[args.worker - 1 :: args.num_workers]
        print(f"Worker {args.worker}/{args.num_workers} — {len(my_jobs)} jobs")
    else:
        my_jobs = all_jobs

    if not args.summarize:
        for cfg, slug, ds_name, extra_args in my_jobs:
            run_dataset(ds_name, cfg, f"{args.name}/{slug}", extra_args,
                        sigma_sweep=args.sigma_sweep)

    collect_summary(args.name, jobs=all_jobs)


if __name__ == "__main__":
    main()
