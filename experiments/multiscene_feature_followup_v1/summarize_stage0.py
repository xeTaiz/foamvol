#!/usr/bin/env python3
"""Validate and summarize the completed Stage-0 reproducibility gate."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import TypedDict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INDEX = HERE / "arms.json"
ROOT = REPO / "output/multiscene_feature_followup_v1/stage0"
SUMMARY_JSON = ROOT / "summary.json"
SUMMARY_CSV = ROOT / "summary.csv"

class ResultRow(TypedDict):
    scene: str
    repeat: int
    arm: str
    worker: str
    psnr: float
    chamfer: float
    achieved_cell_count: int
    config_sha256: str
    repeat_signature_sha256: str
    git_commit: str
    torch_version: str
    torch_cuda_version: str
    gpu_name: str


def main() -> int:
    arms = json.loads(INDEX.read_text())
    if len(arms) != 15:
        raise AssertionError(f"expected 15 arms, found {len(arms)}")

    rows: list[ResultRow] = []
    incomplete: list[str] = []
    for arm in arms:
        run = REPO / "output" / arm["experiment_name"]
        provenance_path = run / "provenance.json"
        if not (run / "DONE").is_file() or not provenance_path.is_file():
            incomplete.append(arm["file_tag"])
            continue
        provenance = json.loads(provenance_path.read_text())
        if provenance["state"] != "DONE":
            raise AssertionError(f"{arm['file_tag']} provenance state={provenance['state']}")
        if provenance["config_sha256"] != arm["config_sha256"]:
            raise AssertionError(f"{arm['file_tag']} config hash mismatch")
        if provenance["repeat_signature_sha256"] != arm["repeat_signature_sha256"]:
            raise AssertionError(f"{arm['file_tag']} repeat signature mismatch")
        metrics = provenance["eval_vol"]
        if not isinstance(metrics, dict):
            raise AssertionError(f"{arm['file_tag']} missing eval_vol metrics")
        cells = provenance["achieved_cell_count"]
        if not isinstance(cells, int):
            raise AssertionError(f"{arm['file_tag']} missing achieved cell count")
        rows.append(
            {
                "scene": arm["scene"],
                "repeat": arm["repeat"],
                "arm": arm["file_tag"],
                "worker": provenance["hostname"],
                "psnr": float(metrics["psnr"]),
                "chamfer": float(metrics["chamfer"]),
                "achieved_cell_count": cells,
                "config_sha256": provenance["config_sha256"],
                "repeat_signature_sha256": provenance["repeat_signature_sha256"],
                "git_commit": provenance["git_commit"],
                "torch_version": provenance["torch_version"],
                "torch_cuda_version": provenance["torch_cuda_version"],
                "gpu_name": provenance["gpu_name"],
            }
        )

    if incomplete:
        raise RuntimeError(f"Stage 0 incomplete: {', '.join(incomplete)}")

    scenes: dict[str, dict[str, object]] = {}
    for scene in ("chest", "pepper", "engine"):
        scene_rows = [row for row in rows if row["scene"] == scene]
        if len(scene_rows) != 5:
            raise AssertionError(f"{scene}: expected 5 rows, found {len(scene_rows)}")
        workers = {str(row["worker"]) for row in scene_rows}
        signatures = {str(row["repeat_signature_sha256"]) for row in scene_rows}
        if len(workers) != 5:
            raise AssertionError(f"{scene}: repeats used only {len(workers)} workers")
        if len(signatures) != 1:
            raise AssertionError(f"{scene}: repeat configs differ beyond experiment_name")
        psnr = [row["psnr"] for row in scene_rows]
        chamfer = [row["chamfer"] for row in scene_rows]
        cells = [row["achieved_cell_count"] for row in scene_rows]
        scenes[scene] = {
            "workers": sorted(workers),
            "repeat_signature_sha256": next(iter(signatures)),
            "psnr_mean": statistics.mean(psnr),
            "psnr_sd": statistics.stdev(psnr),
            "psnr_range": max(psnr) - min(psnr),
            "stage1_psnr_promotion_floor": max(0.15, 2 * statistics.stdev(psnr)),
            "chamfer_mean": statistics.mean(chamfer),
            "chamfer_sd": statistics.stdev(chamfer),
            "chamfer_range": max(chamfer) - min(chamfer),
            "stage1_chamfer_promotion_floor": 2 * statistics.stdev(chamfer),
            "achieved_cell_count_mean": statistics.mean(cells),
            "achieved_cell_count_min": min(cells),
            "achieved_cell_count_max": max(cells),
        }

    commits = sorted({str(row["git_commit"]) for row in rows})
    runtimes = sorted(
        {
            f"{row['gpu_name']}|torch={row['torch_version']}|cuda={row['torch_cuda_version']}"
            for row in rows
        }
    )
    summary = {
        "state": "COMPLETE",
        "run_count": len(rows),
        "git_commits": commits,
        "runtimes": runtimes,
        "scenes": scenes,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["scene"], row["repeat"])))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
