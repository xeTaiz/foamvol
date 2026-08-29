#!/usr/bin/env python3
"""Materialize the seed-45, 75-view, 15-volume feature panel."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "stage3_25_manifest.yaml"
OUT = HERE / "stage3_25_configs"
INDEX = HERE / "stage3_25_arms.json"
QUEUE = HERE / "stage3_25_queue.txt"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected mapping in {path}")
    return value


def main() -> int:
    manifest = load_yaml(MANIFEST)
    base = load_yaml(REPO / str(manifest["base_config"]))
    volumes = manifest["volumes"]
    settings = manifest["settings"]
    components = manifest["components"]
    if len(volumes) != int(manifest["coverage"]["volumes"]):
        raise AssertionError("volume count mismatch")
    if len(settings) != int(manifest["coverage"]["settings"]):
        raise AssertionError("setting count mismatch")
    tags = [str(setting["tag"]) for setting in settings]
    if len(tags) != len(set(tags)):
        raise AssertionError("setting tags are not unique")

    OUT.mkdir(exist_ok=True)
    expected: set[str] = set()
    rows: list[dict[str, Any]] = []
    baselines: list[str] = []
    remainder: list[str] = []
    seed = int(manifest["seed"])
    prefix = str(manifest["output_prefix"])
    for volume, directory in volumes.items():
        reference = f"output/{prefix}/{volume}/s{seed}/BASE/volume_hard_ss4.npy"
        for setting in settings:
            tag = str(setting["tag"])
            names = [str(name) for name in setting["components"]]
            config = dict(base)
            overrides: dict[str, Any] = {}
            for name in names:
                if name not in components:
                    raise AssertionError(f"{tag}: unknown component {name}")
                for key, value in components[name].items():
                    if key in overrides and overrides[key] != value:
                        raise AssertionError(f"{tag}: conflicting override {key}")
                    overrides[key] = value
            config.update(overrides)
            if "REFG_A20" in names:
                config["ref_volume_path"] = reference
            else:
                config.pop("ref_volume_path", None)
                config["ref_guided_pruning"] = False
            config["data_path"] = f"{manifest['data_root']}/{directory}"
            config["seed"] = seed
            config["experiment_name"] = f"{prefix}/{volume}/s{seed}/{tag}"
            fractions = sum(float(config[key]) for key in ("gradient_fraction", "idw_fraction", "entropy_fraction"))
            if abs(fractions - 1.0) > 1e-12:
                raise AssertionError(f"{volume} {tag}: fractions sum to {fractions}")
            if int(config["final_points"]) != 256000:
                raise AssertionError(f"{volume} {tag}: final_points changed")
            arm = f"{volume}__{tag}"
            content = yaml.safe_dump(config, sort_keys=True)
            path = OUT / f"{arm}.yaml"
            path.write_text(content)
            expected.add(path.name)
            row = {"arm": arm, "volume": volume, "tag": tag, "components": names, "experiment_name": config["experiment_name"], "config_sha256": hashlib.sha256(content.encode()).hexdigest(), "requires_reference": "REFG_A20" in names}
            rows.append(row)
            (baselines if tag == "BASE" else remainder).append(arm)

    if len(rows) != int(manifest["coverage"]["total_configs"]):
        raise AssertionError("total config count mismatch")
    actual = {path.name for path in OUT.glob("*.yaml")}
    if actual != expected:
        raise AssertionError(f"config directory mismatch: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")
    random.Random("stage3-25:seed45").shuffle(remainder)
    INDEX.write_text(json.dumps(rows, indent=2) + "\n")
    QUEUE.write_text("\n".join(baselines + remainder) + "\n")
    print(f"wrote {len(rows)} configs: {len(baselines)} prioritized baselines + {len(remainder)} randomized arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
