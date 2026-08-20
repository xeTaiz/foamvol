#!/usr/bin/env python3
"""Summarize completed multiscene_feature_v1 arms against per-scene baselines."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import TypedDict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class Arm(TypedDict):
    file_tag: str
    scene: str
    tag: str
    family: str
    seed: int
    data_path: str
    needs_reference: bool


def metric(path: Path) -> dict[str, float]:
    with path.open() as f:
        raw = json.load(f)
    return {"psnr": float(raw["psnr"]), "chamfer": float(raw["chamfer"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "output" / "multiscene_feature_v1")
    args = parser.parse_args()
    arms: list[Arm] = json.loads((HERE / "arms.json").read_text())
    rows: list[dict[str, object]] = []
    baselines: dict[str, list[dict[str, float]]] = {}
    for arm in arms:
        run = args.root / arm["scene"] / arm["tag"]
        if not (run / "DONE").is_file() or not (run / "eval_vol.json").is_file():
            continue
        values = metric(run / "eval_vol.json")
        if arm["tag"].startswith("BASE_s"):
            baselines.setdefault(arm["scene"], []).append(values)
        rows.append({**arm, **values, "state": "INACTIVE" if (run / "INACTIVE").is_file() else "ACTIVE"})

    print("scene,tag,family,state,psnr,delta_psnr,chamfer,delta_chamfer,verdict")
    for row in rows:
        scene = str(row["scene"])
        bases = baselines.get(scene, [])
        if len(bases) < 2:
            verdict = "PENDING_BASELINE"
            d_psnr = d_chamfer = float("nan")
        else:
            mean_psnr = statistics.mean(x["psnr"] for x in bases)
            mean_chamfer = statistics.mean(x["chamfer"] for x in bases)
            sigma_psnr = statistics.stdev(x["psnr"] for x in bases)
            sigma_chamfer = statistics.stdev(x["chamfer"] for x in bases)
            d_psnr = float(row["psnr"]) - mean_psnr
            d_chamfer = float(row["chamfer"]) - mean_chamfer
            if row["state"] == "INACTIVE":
                verdict = "INACTIVE"
            elif d_psnr > 2 * sigma_psnr or d_chamfer < -2 * sigma_chamfer:
                verdict = "SCREEN_PASS"
            else:
                verdict = "NOISE"
        print(
            f"{scene},{row['tag']},{row['family']},{row['state']},"
            f"{float(row['psnr']):.4f},{d_psnr:.4f},{float(row['chamfer']):.6f},"
            f"{d_chamfer:.6f},{verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
