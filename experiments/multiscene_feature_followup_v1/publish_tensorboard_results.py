#!/usr/bin/env python3
"""Publish post-training hard-volume metrics into each run's event stream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "output/multiscene_feature_followup_v1/stage2a"
MARKER = ".final_volume_tensorboard.json"


def load_mapping(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(run: Path) -> str:
    eval_path = run / "eval_vol.json"
    provenance_path = run / "provenance.json"
    if not (run / "DONE").is_file() or not eval_path.is_file() or not provenance_path.is_file():
        return "incomplete"

    eval_hash = sha256(eval_path)
    marker_path = run / MARKER
    if marker_path.is_file():
        marker = load_mapping(marker_path)
        if marker.get("eval_vol_sha256") == eval_hash:
            return "unchanged"

    metrics = load_mapping(eval_path)
    provenance = load_mapping(provenance_path)
    config_path = REPO / str(provenance["config_path"])
    config = load_mapping(config_path)
    step = int(config["iterations"])
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    achieved_cells = provenance.get("achieved_cell_count")
    if isinstance(achieved_cells, int):
        numeric_metrics["achieved_cell_count"] = float(achieved_cells)

    writer = SummaryWriter(log_dir=str(run), filename_suffix=".final-volume")
    try:
        for key, value in sorted(numeric_metrics.items()):
            writer.add_scalar(f"final_volume/{key}", value, step)
        writer.flush()
    finally:
        writer.close()

    marker_path.write_text(
        json.dumps(
            {
                "eval_vol_sha256": eval_hash,
                "step": step,
                "tags": [f"final_volume/{key}" for key in sorted(numeric_metrics)],
            },
            indent=2,
        )
        + "\n"
    )
    return "published"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    runs = sorted(path for path in args.root.glob("*/s*/*") if path.is_dir())
    counts = {"published": 0, "unchanged": 0, "incomplete": 0}
    for run in runs:
        counts[publish(run)] += 1
    print(json.dumps({"runs": len(runs), **counts}, sort_keys=True))
    if counts["incomplete"]:
        raise RuntimeError(f"{counts['incomplete']} incomplete runs under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
