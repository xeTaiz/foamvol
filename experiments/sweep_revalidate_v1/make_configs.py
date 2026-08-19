#!/usr/bin/env python3
"""Generate the sweep_revalidate_v1 config matrix.

Reads every arm from manifest.yaml (the single source of truth for the arm
list; nothing here duplicates tags or overrides) and writes one YAML config
per arm into experiments/sweep_revalidate_v1/configs/<tag>.yaml.

Three arms (REF_1e3, REF_1e2, REF_1e3_noedge, REFG_dens, REFG_prune,
INIT_ref) reference the literal placeholder "__REF_NPY__" for
ref_volume_path / init_volume_path. That path only exists once Stage A's
BASE_s42 arm has finished training and been voxelized, so those arms are
skipped (with a warning) until --ref-npy is passed.

Usage:
    python experiments/sweep_revalidate_v1/make_configs.py [--ref-npy PATH]
"""

import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "configs")
MANIFEST = os.path.join(HERE, "manifest.yaml")

REF_PLACEHOLDER = "__REF_NPY__"

COMMON = {
    "seed": 42,            # overridden per replicate via the arm's own override
    "checkpoint_steps": "",
    "diag": False,          # screen runs need metrics, not panels
    "corr_diag": False,
    "log_percent": 5,
    "save_volume": False,
}


def load_manifest():
    with open(MANIFEST) as f:
        return yaml.safe_load(f)


def load_base(manifest, base_key):
    rel = manifest["base_configs"][base_key]
    with open(os.path.join(REPO, rel)) as f:
        return yaml.safe_load(f)


def _resolve_placeholders(overrides, ref_npy):
    resolved = {}
    needs_ref = False
    for k, v in overrides.items():
        if v == REF_PLACEHOLDER:
            needs_ref = True
            if ref_npy is None:
                continue
            v = ref_npy
        resolved[k] = v
    return resolved, needs_ref


def build(manifest, arm, ref_npy):
    cfg = load_base(manifest, arm["base"])
    cfg.update(COMMON)
    overrides, needs_ref = _resolve_placeholders(arm.get("overrides") or {}, ref_npy)
    if needs_ref and ref_npy is None:
        return None
    cfg.update(overrides)
    tag = arm["tag"]
    cfg["experiment_name"] = f"sweep_revalidate/{tag}"
    return tag, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-npy", default=None,
                         help="Path to the Stage-A BASE_s42 volume_hard_ss4.npy, "
                              "required to materialize REF_*/REFG_*/INIT_ref arms.")
    args = parser.parse_args()

    manifest = load_manifest()
    os.makedirs(OUT, exist_ok=True)

    written, skipped = [], []
    for arm in manifest["arms"]:
        result = build(manifest, arm, args.ref_npy)
        if result is None:
            skipped.append(arm["tag"])
            continue
        tag, cfg = result
        path = os.path.join(OUT, f"{tag}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        written.append(tag)

    print(f"wrote {len(written)} configs to {OUT}:")
    for n in written:
        print("  ", n)
    if skipped:
        print(f"\nskipped {len(skipped)} arms needing --ref-npy "
              f"(run Stage A first, then re-run with --ref-npy <path>):")
        for n in skipped:
            print("  ", n)


if __name__ == "__main__":
    main()
