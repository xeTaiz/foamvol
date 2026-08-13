#!/usr/bin/env python3
"""Materialize the fixed-cell ray-batch scaling matrix."""
from __future__ import annotations

import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "lc64_air_base.yaml"
OUT = Path(__file__).resolve().parent / "configs"

SCHEDULES = {
    # Matches the historical 1M*9000 + 4M*1000 = 13B sampled rays.
    "ref": dict(iterations=10000, rays_per_batch=1_000_000,
                rays_per_batch_late=4_000_000,
                rays_per_batch_late_start=9000, lr_mult=1.0,
                hard_freeze=1500, split_start=1500),
    "4m_lr1": dict(iterations=3250, rays_per_batch=4_000_000,
                   rays_per_batch_late=4_000_000,
                   rays_per_batch_late_start=-1, lr_mult=1.0,
                   hard_freeze=375, split_start=375),
    "4m_lr2": dict(iterations=3250, rays_per_batch=4_000_000,
                   rays_per_batch_late=4_000_000,
                   rays_per_batch_late_start=-1, lr_mult=2.0,
                   hard_freeze=375, split_start=375),
    "8m_lr3": dict(iterations=1625, rays_per_batch=8_000_000,
                   rays_per_batch_late=8_000_000,
                   rays_per_batch_late_start=-1, lr_mult=3.0,
                   hard_freeze=188, split_start=188),
}


def main():
    base = yaml.safe_load(BASE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cells in (64_000, 128_000):
        for schedule_name, schedule in SCHEDULES.items():
            for mode in ("scalar", "split"):
                cfg = copy.deepcopy(base)
                cfg.update({k: v for k, v in schedule.items()
                            if k not in ("lr_mult", "hard_freeze", "split_start")})
                cfg["init_points"] = cells
                cfg["final_points"] = cells
                cfg["points_hard_freeze_at"] = schedule["hard_freeze"]
                cfg["freeze_points"] = schedule["iterations"]
                # `densify_from` is also optimizer warmup length even when
                # densification is disabled. Scale the legacy 1B-ray warmup.
                cfg["densify_from"] = max(
                    1, round(1_000_000_000 / schedule["rays_per_batch"]))
                cfg["checkpoint_steps"] = str(schedule["iterations"])
                m = schedule["lr_mult"]
                cfg["points_lr_init"] = base["points_lr_init"] * m
                cfg["points_lr_final"] = base["points_lr_final"] * m
                cfg["density_lr_init"] = base["density_lr_init"] * m
                cfg["density_lr_final"] = base["density_lr_final"] * m
                # Diagnostics are collected once after training. Mid-run diagnostics
                # would add unequal overhead across schedules.
                cfg["diag"] = False
                cfg["corr_diag"] = False
                cfg["top_eig_align_weight"] = 0.0
                if mode == "scalar":
                    cfg["thin_surface_start"] = -1
                    cfg["thin_surface_density_mode"] = "scalar"
                    cfg["thin_surface_relative_delta"] = False
                else:
                    cfg["thin_surface_start"] = schedule["split_start"]
                    cfg["thin_surface_density_mode"] = "relative"
                    cfg["thin_surface_relative_delta"] = True
                    cfg["thin_surface_delta_max_frac"] = 0.5
                    cfg["thin_surface_delta_weight"] = 0.0
                    cfg["thin_surface_height_weight"] = 0.0
                    # Absolute initial LRs at multiplier 1: delta=5e-4,
                    # quaternion=height=2e-4; sites remain frozen.
                    cfg["thin_surface_delta_lr_scale"] = 0.1
                    cfg["thin_surface_quat_lr_scale"] = 0.04
                    cfg["thin_surface_sites_lr_scale"] = 0.0
                    cfg["thin_surface_heights_lr_scale"] = 0.04
                tag = f"RB{cells // 1000}K_{mode}_{schedule_name}"
                path = OUT / f"{tag}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                manifest.append({
                    "tag": tag, "cells": cells, "mode": mode,
                    "schedule": schedule_name,
                    "iterations": schedule["iterations"],
                    "rays_per_batch": schedule["rays_per_batch"],
                    "lr_mult": m, "config": str(path.relative_to(ROOT)),
                })
    (Path(__file__).resolve().parent / "manifest.yaml").write_text(
        yaml.safe_dump({"runs": manifest}, sort_keys=False))
    print(f"Wrote {len(manifest)} configs to {OUT}")


if __name__ == "__main__":
    main()
