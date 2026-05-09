#!/usr/bin/env python3
"""Run eval_sigma_sweep.py across a set of output subdirectories.

Discovers all run directories (those containing both config.yaml and model.pt)
under output/<subdir> for each requested subdir. Skips runs where
sigma_sweep.csv already exists (resume-friendly). Supports worker sharding.

Usage:
    python run_sigma_sweep.py --runs fixed_target gradmag_thresh/best428_thresh_100
    python run_sigma_sweep.py --runs fixed_target --worker 1 --of 4
    python run_sigma_sweep.py --runs fixed_target --force   # re-run all
    python run_sigma_sweep.py --list --runs fixed_target    # list discovered runs
"""

import argparse
import os
import subprocess
import sys

# 5 (sigma_s, sigma_v) pairs — post-2026-05 shortlist from 355-run sweep analysis.
# PSNR/SSIM axis: high sigma_v (1-2) wins. Surface axis: low sigma_v (0.05) wins.
# These five cover both axes and the most-balanced single point.
SIGMA_PAIRS = [
    (0.015, 2.00),  # PSNR/SSIM champ at 64k-128k; robust across families
    (0.008, 2.00),  # PSNR champ at 256k+; CD champ at 512k+
    (0.008, 0.05),  # mesh-CD / mesh-F1 champ (top-3 in 89% of runs)
    (0.015, 0.05),  # vol-F1 / HD95 champ at low-mid budgets
    (0.008, 1.00),  # most-balanced: PSNR top-3 + decent surface
]

OUTPUT_ROOT = "output"


def discover_runs(subdirs):
    """Walk output/<subdir> recursively, yield run dirs containing config.yaml + model.pt."""
    runs = []
    for sub in subdirs:
        root = os.path.join(OUTPUT_ROOT, sub)
        if not os.path.isdir(root):
            print(f"[WARN] {root} does not exist — skipping", file=sys.stderr)
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "config.yaml" in files and "model.pt" in files:
                runs.append(dirpath)
    runs.sort()
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run eval_sigma_sweep.py across trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs", nargs="+", required=True, metavar="SUBDIR",
                        help="Output subdirectories to scan (e.g. fixed_target "
                             "gradmag_thresh/best428_thresh_100)")
    parser.add_argument("--worker", type=int, metavar="W",
                        help="Worker index (1-indexed)")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers",
                        help="Total number of workers")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if sigma_sweep.csv already exists")
    parser.add_argument("--list", action="store_true",
                        help="Print discovered run directories and exit")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    all_runs = discover_runs(args.runs)
    print(f"Discovered {len(all_runs)} run(s) under: {', '.join(args.runs)}")

    if args.list:
        for r in all_runs:
            tag = " [done]" if os.path.exists(os.path.join(r, "sigma_sweep.csv")) else ""
            print(f"  {r}{tag}")
        return

    if args.worker is not None:
        my_runs = all_runs[args.worker - 1 :: args.num_workers]
        print(f"Worker {args.worker}/{args.num_workers} — {len(my_runs)} run(s)")
    else:
        my_runs = all_runs

    pairs_args = [f"{s}:{v}" for s, v in SIGMA_PAIRS]

    ok = fail = skip = 0
    for run_dir in my_runs:
        out_csv = os.path.join(run_dir, "sigma_sweep.csv")
        if os.path.exists(out_csv) and not args.force:
            print(f"[SKIP] {run_dir}")
            skip += 1
            continue

        config = os.path.join(run_dir, "config.yaml")
        cmd = [sys.executable, "eval_sigma_sweep.py", "--config", config,
               "--pairs"] + pairs_args
        print(f"[RUN]  {run_dir}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"[FAIL] {run_dir} exited with code {result.returncode}")
            fail += 1
        else:
            ok += 1

    print(f"\nDone — ok={ok}  fail={fail}  skip={skip}  total={len(my_runs)}")


if __name__ == "__main__":
    main()
