#!/usr/bin/env python3
"""Enforce every component assertion for a Stage-2A combination."""
from __future__ import annotations

import argparse
from pathlib import Path

from assert_stage1_active import PRUNE_RE, fail, load_config, tv_is_active, write_marker

TAGS = {
    "S2A_CAP030_TV3e4",
    "S2A_CAP030_TV1e3",
    "S2A_IDW020_TV3e4",
    "S2A_IDW020_TV1e3",
    "S2A_NOPRUNE_TV3e4",
    "S2A_NOPRUNE_TV1e3",
    "S2A_REFG_A20_TV3e4",
    "S2A_REFG_A20_TV1e3",
    "S2A_CAP030_ENT60_B3",
    "S2A_IDW020_ENT60_B3",
    "S2A_NOPRUNE_ENT60_B3",
    "S2A_REFG_A20_ENT60_B3",
    "S2A_TV3e4_ENT60_B3",
    "S2A_TV1e3_ENT60_B3",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    tag = args.tag
    if tag not in TAGS:
        return fail(args.run, tag, "unrecognized Stage-2A tag")
    config = load_config(args.config)
    log_path = args.run / "run.log"
    if not log_path.is_file():
        return fail(args.run, tag, "run.log is missing")
    prune_events = [
        (label, int(removed), int(current))
        for label, removed, current in PRUNE_RE.findall(
            log_path.read_text(errors="replace")
        )
    ]

    if "CAP030" in tag:
        cap = float(config["redundancy_cap"])
        events = [
            (removed, current)
            for label, removed, current in prune_events
            if label == "variance"
        ]
        if not events:
            return fail(args.run, tag, "no nonzero variance-pruning event")
        if any(removed <= 0 or removed > int(cap * current) for removed, current in events):
            return fail(args.run, tag, f"variance-pruning count violates cap {cap}")
    elif "IDW020" in tag:
        if not any(
            label.startswith("IDW threshold=") and removed > 0
            for label, removed, _ in prune_events
        ):
            return fail(args.run, tag, "no nonzero IDW-threshold pruning event")
    elif "NOPRUNE" in tag:
        if prune_events:
            return fail(args.run, tag, "unexpected redundancy-pruning event")
    elif "REFG_A20" in tag:
        if not any(
            label == "ref_weight" and removed > 0
            for label, removed, _ in prune_events
        ):
            return fail(args.run, tag, "no nonzero reference-weight pruning event")

    if "TV3e4" in tag or "TV1e3" in tag:
        if not tv_is_active(args.run, int(config["tv_start"])):
            return fail(
                args.run,
                tag,
                "train/tv_loss lacks a nonzero value at or after tv_start",
            )
    if "ENT60_B3" in tag:
        expected = {
            "gradient_fraction": 0.2,
            "idw_fraction": 0.2,
            "entropy_fraction": 0.6,
            "entropy_bins": 3,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            return fail(args.run, tag, "resolved entropy component is incorrect")

    write_marker(args.run, "ACTIVE")
    print(f"[assert_stage2a_active] {tag}: all component evidence present -> ACTIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
