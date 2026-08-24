#!/usr/bin/env python3
"""Materialize the Stage-0 fixed-seed reproducibility gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.yaml"
OUT = HERE / "configs"
INDEX = HERE / "arms.json"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def canonical_sha(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    manifest = load_yaml(MANIFEST_PATH)
    base = load_yaml(REPO / manifest["base_config"])
    seed = int(manifest["seed"])
    repeats = int(manifest["repeats"])
    prefix = str(manifest["output_prefix"])
    OUT.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    expected_files: set[str] = set()
    signatures: dict[str, set[str]] = {}
    for scene, scene_data in manifest["scenes"].items():
        signatures[scene] = set()
        for repeat in range(1, repeats + 1):
            tag = f"S0_BASE_r{repeat}"
            file_tag = f"{scene}__{tag}"
            config = dict(base)
            config["data_path"] = scene_data["data_path"]
            config["seed"] = seed
            config["experiment_name"] = f"{prefix}/{scene}/{tag}"

            signature_config = {
                key: value for key, value in config.items() if key != "experiment_name"
            }
            signature = canonical_sha(signature_config)
            signatures[scene].add(signature)

            content = yaml.safe_dump(config, sort_keys=True)
            path = OUT / f"{file_tag}.yaml"
            path.write_text(content)
            expected_files.add(path.name)
            rows.append(
                {
                    "file_tag": file_tag,
                    "scene": scene,
                    "tag": tag,
                    "repeat": repeat,
                    "seed": seed,
                    "data_path": scene_data["data_path"],
                    "experiment_name": config["experiment_name"],
                    "config_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "repeat_signature_sha256": signature,
                }
            )

    expected_total = int(manifest["coverage"]["total_configs"])
    if len(rows) != expected_total:
        raise AssertionError(f"wrote {len(rows)} configs, expected {expected_total}")
    for scene, scene_signatures in signatures.items():
        if len(scene_signatures) != 1:
            raise AssertionError(f"{scene} repeats differ beyond experiment_name")
    actual_files = {path.name for path in OUT.glob("*.yaml")}
    if actual_files != expected_files:
        raise AssertionError(
            f"config directory mismatch: extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )

    INDEX.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} Stage-0 configs to {OUT}")
    for scene in manifest["scenes"]:
        signature = next(iter(signatures[scene]))
        print(f"{scene}: repeat_signature_sha256={signature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
