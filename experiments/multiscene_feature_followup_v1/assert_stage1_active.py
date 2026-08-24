#!/usr/bin/env python3
"""Enforce the Stage-1 branch-specific activation assertions."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Any

import yaml

PASSIVE_TAGS = {"F_BASE", "F_BINS3", "F_ENT60_B3", "F_ENT80_B3"}
VARIANCE_TAGS = {"F_CAP015", "F_CAP020", "F_CAP025", "F_CAP030"}
IDW_TAGS = {"F_IDW005", "F_IDW010", "F_IDW020"}
REFG_TAGS = {
    "F_REFG_C02_A5",
    "F_REFG_C02_A10",
    "F_REFG_C02_A20",
    "F_REFG_C03_A10",
    "F_REFG_C02_A10_B1",
}
TV_TAGS = {"F_TV_3e4", "F_TV_1e3", "F_TV_3e3"}
RECOGNIZED_TAGS = PASSIVE_TAGS | VARIANCE_TAGS | IDW_TAGS | REFG_TAGS | TV_TAGS | {"F_NOPRUNE"}
PRUNE_RE = re.compile(r"Redundancy prune \(([^)]+)\):\s*(\d+)/(\d+) cells")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"expected mapping in {path}")
    return config


def write_marker(run: Path, state: str) -> None:
    for name in ("ACTIVE", "INACTIVE", "UNKNOWN"):
        (run / name).unlink(missing_ok=True)
    (run / state).touch()


def fail(run: Path, tag: str, reason: str) -> int:
    write_marker(run, "INACTIVE")
    print(f"[assert_stage1_active] {tag}: {reason} -> INACTIVE")
    return 1


def tv_is_active(run: Path, start: int) -> bool:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = glob.glob(str(run / "events.out.tfevents.*"))
    if not event_files:
        return False
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar = "train/tv_loss"
    if scalar not in accumulator.Tags().get("scalars", []):
        return False
    return any(event.step >= start and abs(event.value) > 0 for event in accumulator.Scalars(scalar))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    tag = args.tag
    if tag not in RECOGNIZED_TAGS:
        return fail(args.run, tag, "unrecognized Stage-1 tag")
    config = load_config(args.config)
    log_path = args.run / "run.log"
    if not log_path.is_file():
        return fail(args.run, tag, "run.log is missing")
    log = log_path.read_text(errors="replace")
    prune_events = [
        (label, int(removed), int(current))
        for label, removed, current in PRUNE_RE.findall(log)
    ]

    if tag in VARIANCE_TAGS:
        events = [(removed, current) for label, removed, current in prune_events if label == "variance"]
        cap = float(config["redundancy_cap"])
        if not events:
            return fail(args.run, tag, "no nonzero variance-pruning event")
        if any(removed <= 0 or removed > int(cap * current) for removed, current in events):
            return fail(args.run, tag, f"variance-pruning count violates cap {cap}")
    elif tag in IDW_TAGS:
        if not any(label.startswith("IDW threshold=") and removed > 0 for label, removed, _ in prune_events):
            return fail(args.run, tag, "no nonzero IDW-threshold pruning event")
    elif tag in REFG_TAGS:
        if not any(label == "ref_weight" and removed > 0 for label, removed, _ in prune_events):
            return fail(args.run, tag, "no nonzero reference-weight pruning event")
    elif tag == "F_NOPRUNE":
        if prune_events:
            return fail(args.run, tag, "unexpected redundancy-pruning event")
    elif tag in TV_TAGS:
        if not tv_is_active(args.run, int(config["tv_start"])):
            return fail(args.run, tag, "train/tv_loss lacks a nonzero value at or after tv_start")

    write_marker(args.run, "ACTIVE")
    print(f"[assert_stage1_active] {tag}: required evidence present -> ACTIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
