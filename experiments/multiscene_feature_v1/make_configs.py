#!/usr/bin/env python3
"""Materialize the multi-scene feature screen from its source manifest.

The revalidation manifest owns the original Stage-B overrides. This generator
only cross-products those immutable arms with scene data paths and adds the
explicit local grids declared in this directory's manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.yaml"
OUT = HERE / "configs"
INDEX = HERE / "arms.json"
COMMON = {
    "checkpoint_steps": "",
    "diag": False,
    "corr_diag": False,
    "log_percent": 5,
    "save_volume": False,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def scene_ref(prefix: str, scene: str) -> str:
    return f"output/{prefix}/{scene}/BASE_s42/volume_hard_ss4.npy"


def materialize(
    *,
    base: dict[str, Any],
    scene: str,
    scene_data: dict[str, Any],
    tag: str,
    family: str,
    overrides: dict[str, Any],
    seed: int,
    prefix: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    config = dict(base)
    config.update(COMMON)
    resolved = {
        key: scene_ref(prefix, scene) if value == "__SCENE_REF_NPY__" else value
        for key, value in overrides.items()
    }
    # Source revalidation arms use its generic placeholder; it becomes a
    # scene-local seed-42 baseline here.
    resolved = {
        key: scene_ref(prefix, scene) if value == "__REF_NPY__" else value
        for key, value in resolved.items()
    }
    config.update(resolved)
    config["data_path"] = scene_data["data_path"]
    config["seed"] = seed
    file_tag = f"{scene}__{tag}"
    config["experiment_name"] = f"{prefix}/{scene}/{tag}"
    metadata = {
        "file_tag": file_tag,
        "scene": scene,
        "tag": tag,
        "family": family,
        "seed": seed,
        "data_path": scene_data["data_path"],
        "needs_reference": any(value == "__REF_NPY__" or value == "__SCENE_REF_NPY__" for value in overrides.values()),
    }
    return file_tag, config, metadata


def main() -> int:
    manifest = load_yaml(MANIFEST_PATH)
    source = load_yaml(REPO / manifest["source_manifest"])
    source_arms = {arm["tag"]: arm for arm in source["arms"]}
    requested = manifest["source_stage_b_tags"]
    smoke_tag = manifest["smoke_source_tag"]
    missing = sorted((set(requested) | {smoke_tag}) - source_arms.keys())
    if missing:
        raise ValueError(f"source tags absent from revalidation manifest: {missing}")

    screen_base = load_yaml(REPO / source["base_configs"]["screen"])
    OUT.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    baseline_signatures: dict[str, dict[str, Any]] = {}
    for scene, scene_data in manifest["scenes"].items():
        for seed in manifest["seeds"]["baseline"]:
            file_tag, config, row = materialize(
                base=screen_base, scene=scene, scene_data=scene_data, tag=f"BASE_s{seed}",
                family="BASE", overrides={}, seed=seed, prefix=manifest["output_prefix"],
            )
            if seed == manifest["seeds"]["screen"]:
                baseline_signatures[scene] = {
                    key: value for key, value in config.items() if key != "experiment_name"
                }
            with (OUT / f"{file_tag}.yaml").open("w") as f:
                yaml.safe_dump(config, f, sort_keys=True)
            rows.append(row)
        smoke_arm = source_arms[smoke_tag]
        smoke_base = load_yaml(REPO / source["base_configs"][smoke_arm["base"]])
        file_tag, config, row = materialize(
            base=smoke_base, scene=scene, scene_data=scene_data, tag=smoke_tag,
            family=smoke_arm["family"], overrides=smoke_arm.get("overrides") or {},
            seed=manifest["seeds"]["screen"], prefix=manifest["output_prefix"],
        )
        if {
            key: value for key, value in config.items() if key != "experiment_name"
        } == baseline_signatures[scene]:
            raise AssertionError(f"{file_tag} duplicates its scene baseline")
        with (OUT / f"{file_tag}.yaml").open("w") as f:
            yaml.safe_dump(config, f, sort_keys=True)
        rows.append(row)
        for source_tag in requested:
            arm = source_arms[source_tag]
            source_base = load_yaml(REPO / source["base_configs"][arm["base"]])
            file_tag, config, row = materialize(
                base=source_base, scene=scene, scene_data=scene_data, tag=source_tag,
                family=arm["family"], overrides=arm.get("overrides") or {},
                seed=manifest["seeds"]["screen"], prefix=manifest["output_prefix"],
            )
            if {
                key: value for key, value in config.items() if key != "experiment_name"
            } == baseline_signatures[scene]:
                raise AssertionError(f"{file_tag} duplicates its scene baseline")
            with (OUT / f"{file_tag}.yaml").open("w") as f:
                yaml.safe_dump(config, f, sort_keys=True)
            rows.append(row)
        for arm in manifest["extra_arms"]:
            file_tag, config, row = materialize(
                base=screen_base, scene=scene, scene_data=scene_data, tag=arm["tag"],
                family=arm["family"], overrides=arm["overrides"],
                seed=manifest["seeds"]["screen"], prefix=manifest["output_prefix"],
            )
            if {
                key: value for key, value in config.items() if key != "experiment_name"
            } == baseline_signatures[scene]:
                raise AssertionError(f"{file_tag} duplicates its scene baseline")
            with (OUT / f"{file_tag}.yaml").open("w") as f:
                yaml.safe_dump(config, f, sort_keys=True)
            rows.append(row)

    if len(rows) != manifest["coverage"]["total_configs"]:
        raise AssertionError(f"wrote {len(rows)} configs, expected {manifest['coverage']['total_configs']}")
    INDEX.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} configs to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
