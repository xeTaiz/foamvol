#!/usr/bin/env python3
"""Print the sweep_revalidate_v1 results table from the on-disk metric JSONs.

Reads, per finished arm: eval_vol.json (written by eval_vol.py --json;
PSNR/SSIM/Sobel/Dice/Chamfer/Hausdorff/F1, all on the corrected centre-
registered SS4 volume), air_metrics.json (written by air_metrics.py; strict
-air MAE/FPR), the ACTIVE/INACTIVE/UNKNOWN marker written by
assert_active.py, and the "Num Cells:" line of metrics.txt.

Usage:
    python experiments/sweep_revalidate_v1/summarize.py [run_root]

Defaults to output/sweep_revalidate relative to the repository root.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.yaml"

# Reference rows, already measured with this exact evaluator on this exact GT.
REFERENCE_ROWS = [
    dict(tag="R2-Gaussian", family="REFERENCE", psnr=35.8512, ssim_3d=0.943398,
         dice=0.889397, chamfer=0.7283, hausdorff_95=4.1286, f1_1v=0.8838,
         f1_2v=0.9481),
    dict(tag="SC256_ctrl", family="REFERENCE", psnr=34.8299, ssim_3d=0.924534,
         dice=0.849052, chamfer=1.4388, hausdorff_95=12.7060, f1_1v=0.7880,
         f1_2v=0.8777),
]

BASE_FAMILY = "BASE"
BASE_ARMS = [f"BASE_s{s}" for s in (42, 43, 44, 45, 46)]

NUM_CELLS_RE = re.compile(r"Num Cells:\s*(\d+)")


@dataclass(frozen=True)
class ArmMetrics:
    tag: str
    family: str
    active: str
    num_cells: int | None
    psnr: float
    ssim_3d: float
    dice: float
    chamfer: float
    hausdorff_95: float
    f1_1v: float
    f1_2v: float
    sobel_psnr: float
    air_mae_strict: float | None
    air_fpr_strict: float | None


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _active_state(run_dir: Path) -> str:
    for state in ("ACTIVE", "INACTIVE", "UNKNOWN"):
        if (run_dir / state).exists():
            return state
    return "MISSING"


def _num_cells(run_dir: Path) -> int | None:
    path = run_dir / "metrics.txt"
    try:
        text = path.read_text()
    except OSError:
        return None
    m = NUM_CELLS_RE.search(text)
    return int(m.group(1)) if m else None


def _row(root: Path, tag: str, family: str) -> ArmMetrics | None:
    run_dir = root / tag
    vol = _load(run_dir / "eval_vol.json")
    if vol is None:
        return None
    air = _load(run_dir / "air_metrics.json")
    air_mae = air["mae"]["strict_air"] if air is not None else None
    air_fpr = air.get("strict_air_fpr") if air is not None else None
    return ArmMetrics(
        tag=tag,
        family=family,
        active=_active_state(run_dir),
        num_cells=_num_cells(run_dir),
        psnr=float(vol["psnr"]),
        ssim_3d=float(vol["ssim_3d"]),
        dice=float(vol["dice"]),
        chamfer=float(vol["chamfer"]),
        hausdorff_95=float(vol["hausdorff_95"]),
        f1_1v=float(vol["f1_1v"]),
        f1_2v=float(vol["f1_2v"]),
        sobel_psnr=float(vol["sobel_psnr"]),
        air_mae_strict=None if air_mae is None else float(air_mae),
        air_fpr_strict=None if air_fpr is None else float(air_fpr),
    )


def load_arms() -> list[dict[str, Any]]:
    with open(MANIFEST) as f:
        return yaml.safe_load(f)["arms"]


def print_csv(rows: list[ArmMetrics]) -> None:
    header = ("tag", "family", "active", "num_cells", "psnr", "ssim_3d", "dice",
               "chamfer", "hausdorff_95", "f1_1v", "f1_2v", "sobel_psnr",
               "air_mae_strict", "air_fpr_strict")
    print(",".join(header))
    for r in rows:
        print(",".join(str(v) for v in (
            r.tag, r.family, r.active, r.num_cells, r.psnr, r.ssim_3d, r.dice,
            r.chamfer, r.hausdorff_95, r.f1_1v, r.f1_2v, r.sobel_psnr,
            r.air_mae_strict, r.air_fpr_strict)))
    for ref in REFERENCE_ROWS:
        print(",".join(str(ref.get(k, "")) for k in header))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="?", default="output/sweep_revalidate")
    parser.add_argument("--sigma", action="store_true",
                         help="Print the Stage-A noise floor (sigma_psnr, sigma_chamfer) and exit")
    args = parser.parse_args()
    root = Path(str(args.run_root))

    arms = load_arms()
    tag_to_family = {a["tag"]: a["family"] for a in arms}

    rows = []
    for tag, family in tag_to_family.items():
        row = _row(root, tag, family)
        if row is not None:
            rows.append(row)

    base_rows = [r for r in rows if r.family == BASE_FAMILY]
    if len(base_rows) >= 2:
        psnrs = [r.psnr for r in base_rows]
        chamfers = [r.chamfer for r in base_rows]
        mean_psnr, sigma_psnr = st.mean(psnrs), st.stdev(psnrs)
        mean_chamfer, sigma_chamfer = st.mean(chamfers), st.stdev(chamfers)
        print(f"# Stage-A noise floor (n={len(base_rows)}): "
              f"psnr mean={mean_psnr:.4f} sigma={sigma_psnr:.4f} | "
              f"chamfer mean={mean_chamfer:.4f} sigma={sigma_chamfer:.4f}")
        if sigma_psnr > 0.30:
            print(f"# WARNING: sigma_psnr={sigma_psnr:.4f} dB exceeds the 0.30 dB "
                  f"stop-and-report bound; a single-replicate screen is uninterpretable.")
    else:
        mean_psnr = mean_chamfer = sigma_psnr = sigma_chamfer = None
        print(f"# Stage-A noise floor: need >=2 finished BASE_* arms, have {len(base_rows)}")

    if args.sigma:
        return 0

    print_csv(sorted(rows, key=lambda r: (r.family, r.tag)))

    if sigma_psnr is not None and mean_psnr is not None and mean_chamfer is not None and sigma_chamfer is not None:
        base_mean_psnr: float = mean_psnr
        base_mean_chamfer: float = mean_chamfer
        base_sigma_psnr: float = sigma_psnr
        base_sigma_chamfer: float = sigma_chamfer
        print("\n# Per-family delta vs Stage-A baseline mean, decision-rule verdict "
              "(favourable: higher psnr, lower chamfer; threshold 2*sigma):")
        by_family: dict[str, list[ArmMetrics]] = {}
        for r in rows:
            if r.family in (BASE_FAMILY,):
                continue
            by_family.setdefault(r.family, []).append(r)
        for family in sorted(by_family):
            for r in sorted(by_family[family], key=lambda r: r.tag):
                if r.active == "INACTIVE":
                    verdict = "untested-inactive"
                elif r.active == "UNKNOWN":
                    verdict = "untested-unknown"
                else:
                    dpsnr = r.psnr - base_mean_psnr
                    dchamfer = base_mean_chamfer - r.chamfer  # favourable = lower chamfer
                    hit = (dpsnr > 2 * base_sigma_psnr) or (dchamfer > 2 * base_sigma_chamfer)
                    verdict = "SIGNIFICANT" if hit else "noise"
                dpsnr = r.psnr - base_mean_psnr
                dchamfer = r.chamfer - base_mean_chamfer
                print(f"  {r.tag:<18} family={family:<6} active={r.active:<8} "
                      f"dpsnr={dpsnr:+.4f} dchamfer={dchamfer:+.4f} -> {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
