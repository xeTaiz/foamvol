#!/usr/bin/env python3
"""Generate Stage-C replicate configs for arms that passed Stage-B screening.

For each winning tag, writes <tag>_s43.yaml, <tag>_s44.yaml, <tag>_s45.yaml by
copying the already-generated Stage-B config and overriding only `seed` and
`experiment_name`. Kept separate from make_configs.py because the winner list
is a Stage-B *result*, not part of the static manifest.

Usage:
    python experiments/sweep_revalidate_v1/make_stagec_configs.py TAG [TAG ...]
"""
import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "configs")
SEEDS = (43, 44, 45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="Stage-B tags that passed screening")
    args = parser.parse_args()

    written = []
    for tag in args.tags:
        base_path = os.path.join(CFG_DIR, f"{tag}.yaml")
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        for seed in SEEDS:
            cfg = dict(base_cfg)
            cfg["seed"] = seed
            rep_tag = f"{tag}_s{seed}"
            cfg["experiment_name"] = f"sweep_revalidate/{rep_tag}"
            out_path = os.path.join(CFG_DIR, f"{rep_tag}.yaml")
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
            written.append(rep_tag)

    print(f"wrote {len(written)} Stage-C configs:")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
