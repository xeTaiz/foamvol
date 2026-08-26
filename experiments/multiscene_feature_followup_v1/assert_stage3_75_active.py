#!/usr/bin/env python3
"""Enforce activation evidence for every Stage-3 75-view setting."""
from __future__ import annotations

import argparse
from pathlib import Path

from assert_stage1_active import PRUNE_RE, fail, load_config, tv_is_active, write_marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    log_path = args.run / "run.log"
    if not log_path.is_file():
        return fail(args.run, args.tag, "run.log is missing")
    events = [(label, int(removed), int(current)) for label, removed, current in PRUNE_RE.findall(log_path.read_text(errors="replace"))]

    if "CAP030" in args.tag:
        cap = float(config["redundancy_cap"])
        selected = [(removed, current) for label, removed, current in events if label == "variance"]
        if not selected or any(removed <= 0 or removed > int(cap * current) for removed, current in selected):
            return fail(args.run, args.tag, "variance pruning is absent or violates CAP030")
    elif "IDW020" in args.tag:
        if not any(label.startswith("IDW threshold=") and removed > 0 for label, removed, _ in events):
            return fail(args.run, args.tag, "no nonzero IDW pruning event")
    elif "NOPRUNE" in args.tag:
        if events:
            return fail(args.run, args.tag, "unexpected redundancy pruning event")
    elif "REFG_A20" in args.tag:
        if not any(label == "ref_weight" and removed > 0 for label, removed, _ in events):
            return fail(args.run, args.tag, "no nonzero reference-guided pruning event")

    if "TV1e3" in args.tag or "TV3e4" in args.tag:
        if not tv_is_active(args.run, int(config["tv_start"])):
            return fail(args.run, args.tag, "train/tv_loss lacks a nonzero value at or after tv_start")
    if "ENT60_B3" in args.tag:
        expected = {"gradient_fraction": 0.2, "idw_fraction": 0.2, "entropy_fraction": 0.6, "entropy_bins": 3}
        if any(config.get(key) != value for key, value in expected.items()):
            return fail(args.run, args.tag, "resolved entropy component is incorrect")

    write_marker(args.run, "ACTIVE")
    print(f"[assert_stage3_75_active] {args.tag}: intended evidence present -> ACTIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
