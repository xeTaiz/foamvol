#!/usr/bin/env python3
"""Print the sweep_splitcell_v1 results table from the on-disk metric JSONs.

These numbers do NOT exist in TensorBoard. They are written after training by
split_voxelize.py (256^3, 4x supersampled, hard side selection) plus the surface
and continuity evaluators, all of which write JSON and never call add_scalar.
TensorBoard's test/vol_raw_psnr is the same quantity at 1 sample/voxel (~3.8 dB
lower) and test/vol_r2_psnr is a different normalization entirely, so neither
matches this table. Compare TB against TB and JSON against JSON.

Usage:
    python experiments/sweep_splitcell_v1/summarize.py [run_root]

Defaults to output/sweep_splitcell relative to the repository root.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELL_GROUPS = {
    "128k": ["SC128_ctrl", "SC128_w1e-5", "SC128_w1e-5d", "SC128_w3e-5d"],
    "256k": ["SC256_ctrl", "SC256_w1e-5", "SC256_w1e-5d", "SC256_w3e-5d"],
    "512k": ["SC512_ctrl", "SC512_w1e-5", "SC512_w1e-5d", "SC512_w3e-5d"],
}
SCALAR_ARM = "SC256_scalar"
ORDER = (CELL_GROUPS["128k"] + CELL_GROUPS["256k"] + [SCALAR_ARM]
         + CELL_GROUPS["512k"])


@dataclass(frozen=True)
class ArmMetrics:
    """One arm's headline metrics, all read from *_metrics.json on disk."""

    volume_psnr: float
    volume_ssim_3d: float
    sobel_psnr: float
    dice: float
    air_mae: float
    air_fpr: float
    chamfer: float
    hausdorff_95: float
    f1_1v: float
    # None means "no thin-surface state, never evaluated" (the scalar control),
    # which is different information from a measured zero candidate pool.
    candidates: float | None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _row(root: Path, arm: str) -> ArmMetrics | None:
    volume = _load(root / arm / "volume_hard_ss4_metrics.json")
    surface = _load(root / arm / "surface_hard_ss4_metrics.json")
    continuity = _load(root / arm / "face_continuity_eval.json")
    if volume is None or surface is None:
        return None
    air = volume["air"]
    candidates = None
    if continuity is not None:
        raw = continuity["mean"].get("candidate_faces")
        candidates = None if raw is None else float(raw)
    return ArmMetrics(
        volume_psnr=float(volume["volume_psnr"]),
        volume_ssim_3d=float(volume["volume_ssim_3d"]),
        sobel_psnr=float(volume["sobel_psnr"]),
        dice=float(volume["dice"]),
        air_mae=float(air["mae"]["strict_air"]),
        air_fpr=float(air["strict_air_fpr"]),
        chamfer=float(surface["chamfer"]),
        hausdorff_95=float(surface["hausdorff_95"]),
        f1_1v=float(surface["f1_1v"]),
        candidates=candidates,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="?", default="output/sweep_splitcell")
    args = parser.parse_args()
    root = Path(str(args.run_root))

    rows: dict[str, ArmMetrics | None] = {
        arm: _row(root, arm) for arm in ORDER}

    header = ("arm", "volPSNR", "SSIM3D", "sobPSNR", "dice", "airMAE",
              "airFPR", "chamfer", "hd95", "f1_1v", "cand")
    print("%-13s %8s %7s %8s %7s %9s %9s %8s %7s %7s %5s" % header)
    for arm in ORDER:
        row = rows[arm]
        if row is None:
            print("%-13s %s" % (arm, "MISSING (not finished, or eval failed)"))
            continue
        cand = "-" if row.candidates is None else "%g" % row.candidates
        print("%-13s %8.3f %7.4f %8.3f %7.4f %9.2e %9.2e %8.4f %7.3f %7.4f %5s"
              % (arm, row.volume_psnr, row.volume_ssim_3d, row.sobel_psnr,
                 row.dice, row.air_mae, row.air_fpr, row.chamfer,
                 row.hausdorff_95, row.f1_1v, cand))

    # The continuity loss was identically zero in every split arm (the candidate
    # pool stayed empty for the whole run), so the four arms of a group are
    # repeats of one effective configuration: this spread is the noise floor,
    # not an effect of the regularization weight.
    print("\nsame-effective-config spread (noise floor):")
    for group, arms in CELL_GROUPS.items():
        present = [row for row in (rows[arm] for arm in arms)
                   if row is not None]
        if len(present) < 2:
            print("  %s: need >=2 finished arms" % group)
            continue
        psnr = [row.volume_psnr for row in present]
        chamfer = [row.chamfer for row in present]
        hd95 = [row.hausdorff_95 for row in present]
        print("  %s n=%d volPSNR mean=%.3f sd=%.3f range=%.3f | chamfer "
              "mean=%.4f sd=%.4f | hd95 mean=%.2f sd=%.2f"
              % (group, len(present), st.mean(psnr), st.stdev(psnr),
                 max(psnr) - min(psnr), st.mean(chamfer), st.stdev(chamfer),
                 st.mean(hd95), st.stdev(hd95)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
