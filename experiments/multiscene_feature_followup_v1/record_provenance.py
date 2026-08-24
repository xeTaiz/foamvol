#!/usr/bin/env python3
"""Record immutable runtime and result provenance for one follow-up run."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

torch = importlib.import_module("torch")

REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def parse_num_cells(metrics_path: Path) -> int | None:
    if not metrics_path.is_file():
        return None
    match = re.search(r"^Num Cells:\s*(\d+)\s*$", metrics_path.read_text(), re.MULTILINE)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--started-at", required=True)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    signature_config = {
        key: value for key, value in config.items() if key != "experiment_name"
    }
    ref_volume_value = str(config.get("ref_volume_path", ""))
    ref_volume_path = Path(ref_volume_value) if ref_volume_value else None
    if ref_volume_path is not None and not ref_volume_path.is_absolute():
        ref_volume_path = REPO / ref_volume_path
    ref_volume_sha256 = (
        sha256_file(ref_volume_path)
        if ref_volume_path is not None and ref_volume_path.is_file()
        else None
    )
    state = next(
        (name for name in ("FAILED", "DONE") if (args.run / name).is_file()),
        "NONTERMINAL",
    )
    eval_metrics = None
    eval_path = args.run / "eval_vol.json"
    if eval_path.is_file():
        eval_metrics = json.loads(eval_path.read_text())

    evaluator_files = ("split_voxelize.py", "eval_vol.py", "air_metrics.py")
    provenance = {
        "arm": args.arm,
        "experiment_name": config["experiment_name"],
        "state": state,
        "started_at": args.started_at,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "repeat_signature_sha256": canonical_sha(signature_config),
        "ref_volume_sha256": ref_volume_sha256,
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "hostname": socket.gethostname(),
        "gpu_index": args.gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_driver_version": command_output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ).splitlines()[0],
        "evaluator_sha256": {
            name: sha256_file(REPO / name) for name in evaluator_files
        },
        "achieved_cell_count": parse_num_cells(args.run / "metrics.txt"),
        "eval_vol": eval_metrics,
    }
    args.run.mkdir(parents=True, exist_ok=True)
    destination = args.run / "provenance.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary.replace(destination)
    print(
        f"[{args.arm}] provenance state={state} cells={provenance['achieved_cell_count']} "
        f"worker={provenance['hostname']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
