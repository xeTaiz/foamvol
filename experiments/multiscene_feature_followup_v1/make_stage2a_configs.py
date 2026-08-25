#!/usr/bin/env python3
"""Materialize the approved Stage-2A two-way combination matrix."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST_PATH = HERE / "stage2a_manifest.yaml"
OUT = HERE / "stage2a_configs"
INDEX = HERE / "stage2a_arms.json"
BLOCKS = HERE / "stage2a_blocks.json"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def main() -> int:
    manifest = load_yaml(MANIFEST_PATH)
    stage1 = load_yaml(REPO / manifest["stage1_manifest"])
    base = load_yaml(REPO / stage1["base_config"])
    combinations = manifest["combinations"]
    components: dict[str, dict[str, Any]] = {}
    for axis in ("pruning", "tv", "entropy"):
        for name, overrides in manifest[axis].items():
            if name in components:
                raise AssertionError(f"duplicate component {name}")
            components[name] = dict(overrides)

    expected_combinations = int(manifest["coverage"]["combinations"])
    if len(combinations) != expected_combinations:
        raise AssertionError(
            f"found {len(combinations)} combinations, expected {expected_combinations}"
        )
    tags = [str(combo["tag"]) for combo in combinations]
    if len(tags) != len(set(tags)):
        raise AssertionError("Stage-2A tags are not unique")

    OUT.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    expected_files: set[str] = set()
    for scene, scene_data in stage1["scenes"].items():
        for seed_value in stage1["seeds"]:
            seed = int(seed_value)
            block_arms: list[str] = []
            for combo in combinations:
                tag = str(combo["tag"])
                component_names = [str(name) for name in combo["components"]]
                if len(component_names) != 2 or len(set(component_names)) != 2:
                    raise AssertionError(f"{tag}: expected two distinct components")
                overrides: dict[str, Any] = {}
                for component_name in component_names:
                    if component_name not in components:
                        raise AssertionError(f"{tag}: unknown component {component_name}")
                    for key, value in components[component_name].items():
                        if key in overrides and overrides[key] != value:
                            raise AssertionError(f"{tag}: conflicting override {key}")
                        overrides[key] = value
                for key, value in list(overrides.items()):
                    if value == "__SCENE_REF_NPY__":
                        overrides[key] = scene_data["ref_volume_path"]

                config = dict(base)
                config.update(overrides)
                config["data_path"] = scene_data["data_path"]
                config["seed"] = seed
                config["experiment_name"] = (
                    f"{manifest['output_prefix']}/{scene}/s{seed}/{tag}"
                )
                fractions = sum(
                    float(config[key])
                    for key in ("gradient_fraction", "idw_fraction", "entropy_fraction")
                )
                if abs(fractions - 1.0) > 1e-12:
                    raise AssertionError(f"{scene} s{seed} {tag}: fractions sum to {fractions}")
                if int(config["final_points"]) != 256000:
                    raise AssertionError(f"{scene} s{seed} {tag}: final_points changed")
                if "REFG_A20" in component_names and not config.get("ref_volume_path"):
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
                        "parents": list(combo["parents"]),
                        "components": component_names,
                        "experiment_name": config["experiment_name"],
                        "config_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    }
                )

            random.Random(f"stage2a:{scene}:{seed}").shuffle(block_arms)
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
    print(f"wrote {len(rows)} Stage-2A configs in {len(blocks)} randomized blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
