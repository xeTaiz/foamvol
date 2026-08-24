#!/usr/bin/env python3
"""Publish final volume and air metrics into each curated TensorBoard run."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def scalar_leaves(value: object, prefix: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from scalar_leaves(child, child_prefix)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield prefix, float(value)


def publish(run: Path) -> bool:
    sources = {
        "volume": run / "eval_vol.json",
        "air": run / "air_metrics.json",
    }
    if not all(path.is_file() for path in sources.values()):
        return False

    for event in run.glob("events.out.tfevents.*.final_metrics"):
        event.unlink()

    writer = SummaryWriter(log_dir=str(run), filename_suffix=".final_metrics")
    try:
        for group, path in sources.items():
            payload: Any = json.loads(path.read_text())
            for name, value in scalar_leaves(payload):
                writer.add_scalar(f"final/{group}/{name}", value, global_step=10_000)
    finally:
        writer.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "output" / "multiscene_feature_v1")
    args = parser.parse_args()

    published = 0
    for scene in sorted(args.root.iterdir()):
        if not scene.is_dir() or scene.name.startswith("."):
            continue
        for run in sorted(scene.iterdir()):
            if run.is_dir() and publish(run):
                published += 1
    print(f"published final metrics for {published} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
