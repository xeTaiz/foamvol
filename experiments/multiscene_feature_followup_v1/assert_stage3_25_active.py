#!/usr/bin/env python3
"""Enforce 25-view provenance and activation evidence for Stage-3 settings."""
from __future__ import annotations

import argparse
from pathlib import Path

from assert_stage1_active import fail, load_config
from assert_stage3_75_active import main as assert_activation


EXPECTED_CONFIG_DIR = "stage3_25_configs"
EXPECTED_DATASET_DIR = "cone_ntrain_25_angle_360"
EXPECTED_EXPERIMENT_PREFIX = "multiscene_feature_followup_v1/stage3_25/"


def assert_25_view_contract(run: Path, config_path: Path, tag: str) -> int:
    if config_path.parent.name != EXPECTED_CONFIG_DIR:
        return fail(run, tag, f"config is not under {EXPECTED_CONFIG_DIR}")

    config = load_config(config_path)
    data_path = Path(str(config.get("data_path", "")))
    if EXPECTED_DATASET_DIR not in data_path.parts:
        return fail(run, tag, f"data_path is not the 25-view dataset: {data_path}")

    experiment_name = str(config.get("experiment_name", ""))
    if not experiment_name.startswith(EXPECTED_EXPERIMENT_PREFIX):
        return fail(run, tag, f"experiment_name is not Stage-3 25-view: {experiment_name}")
    if Path("output") / experiment_name != run:
        return fail(run, tag, "run path does not match config experiment_name")

    reference = str(config.get("ref_volume_path", ""))
    if reference and "/stage3_25/" not in f"/{reference}":
        return fail(run, tag, f"reference is not from the 25-view panel: {reference}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    contract_result = assert_25_view_contract(args.run, args.config, args.tag)
    if contract_result:
        return contract_result
    return assert_activation()


if __name__ == "__main__":
    raise SystemExit(main())
