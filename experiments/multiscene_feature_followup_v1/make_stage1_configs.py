#!/usr/bin/env python3
"""Materialize the replicated Stage-1 main-effect matrix and block queues."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST_PATH = HERE / "stage1_manifest.yaml"
OUT = HERE / "stage1_configs"
INDEX = HERE / "stage1_arms.json"
BLOCKS = HERE / "stage1_blocks.json"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def config_sha(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    manifest = load_yaml(MANIFEST_PATH)
    base = load_yaml(REPO / manifest["base_config"])
    settings = manifest["settings"]
    seeds = [int(seed) for seed in manifest["seeds"]]
    prefix = str(manifest["output_prefix"])
    OUT.mkdir(exist_ok=True)

    expected_settings = int(manifest["coverage"]["settings"])
    if len(settings) != expected_settings:
        raise AssertionError(f"found {len(settings)} settings, expected {expected_settings}")
    tags = [str(setting["tag"]) for setting in settings]
    if len(tags) != len(set(tags)):
        raise AssertionError("Stage-1 setting tags are not unique")
    if tags.count("F_BASE") != 1:
        raise AssertionError("Stage-1 requires exactly one F_BASE setting")

    rows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    expected_files: set[str] = set()
    for scene, scene_data in manifest["scenes"].items():
        for seed in seeds:
            block_arms: list[str] = []
            for setting in settings:
                tag = str(setting["tag"])
                family = str(setting["family"])
                overrides = dict(setting.get("overrides", {}))
                for key, value in list(overrides.items()):
                    if value == "__SCENE_REF_NPY__":
                        overrides[key] = scene_data["ref_volume_path"]

                config = dict(base)
                config.update(overrides)
                config["data_path"] = scene_data["data_path"]
                config["seed"] = seed
                config["experiment_name"] = f"{prefix}/{scene}/s{seed}/{tag}"
                fractions = sum(
                    float(config[key])
                    for key in ("gradient_fraction", "idw_fraction", "entropy_fraction")
                )
                if abs(fractions - 1.0) > 1e-12:
                    raise AssertionError(f"{scene} s{seed} {tag}: fractions sum to {fractions}")
                if int(config["final_points"]) != 256000:
                    raise AssertionError(f"{scene} s{seed} {tag}: final_points changed")
                if family == "refg" and not config.get("ref_volume_path"):
                    raise AssertionError(f"{scene} s{seed} {tag}: missing reference volume")

                file_tag = f"{scene}__{tag}_s{seed}"
                content = yaml.safe_dump(config, sort_keys=True)
                path = OUT / f"{file_tag}.yaml"
                path.write_text(content)
                expected_files.add(path.name)
                block_arms.append(file_tag)
                rows.append(
                    {
                        "file_tag": file_tag,
                        "scene": scene,
                        "seed": seed,
                        "tag": tag,
                        "family": family,
                        "experiment_name": config["experiment_name"],
                        "config_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "resolved_signature_sha256": config_sha(config),
                    }
                )

            random.Random(f"stage1:{scene}:{seed}").shuffle(block_arms)
            blocks.append({"scene": scene, "seed": seed, "arms": block_arms})

    expected_total = int(manifest["coverage"]["total_configs"])
    if len(rows) != expected_total:
        raise AssertionError(f"wrote {len(rows)} configs, expected {expected_total}")
    actual_files = {path.name for path in OUT.glob("*.yaml")}
    if actual_files != expected_files:
        raise AssertionError(
            f"config directory mismatch: extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}"
        )

    INDEX.write_text(json.dumps(rows, indent=2) + "\n")
    BLOCKS.write_text(json.dumps(blocks, indent=2) + "\n")
    print(f"wrote {len(rows)} Stage-1 configs in {len(blocks)} randomized blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
