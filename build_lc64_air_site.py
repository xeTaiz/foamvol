#!/usr/bin/env python3
"""Build a static status/results page for the LC64 air diagnosis."""
import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path

import yaml


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/LC64-AIR-SPLIT-DIAGNOSIS-MANIFEST-v1.yaml")
    ap.add_argument("--output-root", default="output/lc64_air_v1")
    ap.add_argument("--web-root", default="output/web-results/LC64-air-v1")
    args = ap.parse_args()
    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text())
    output_root = Path(args.output_root)
    web_root = Path(args.web_root)
    web_root.mkdir(parents=True, exist_ok=True)
    (web_root / "configs").mkdir(exist_ok=True)
    shutil.copy2(manifest_path, web_root / "manifest.yaml")

    rows = []
    status = {"name": manifest["name"], "commit": _git_commit(), "arms": {}}
    for stage_name, stage in manifest.get("stages", {}).items():
        for arm in stage.get("arms", []):
            tag = arm.get("tag")
            if not tag or "reuse" in arm:
                continue
            arm_dir = output_root / tag
            config_path = Path("experiments/lc64_air/configs") / f"{tag}.yaml"
            if config_path.exists():
                shutil.copy2(config_path, web_root / "configs" / config_path.name)
            model = arm_dir / "model.pt"
            metrics_candidates = sorted(arm_dir.glob("*_metrics.json")) + sorted(arm_dir.glob("metrics.json"))
            marker_path = arm_dir / "remote_job.json"
            marker = None
            if marker_path.exists():
                try:
                    marker = json.loads(marker_path.read_text())
                except Exception:
                    marker = None
            if model.exists() and metrics_candidates:
                state = "complete"
            elif model.exists():
                state = "training-complete / evaluation-pending"
            elif marker and marker.get("state") == "running":
                state = f"running on {marker.get('worker', 'remote worker')}"
            elif arm_dir.exists():
                state = "interrupted / awaiting recovery"
            else:
                state = "queued (gated)"
            metrics = None
            if metrics_candidates:
                try:
                    metrics = json.loads(metrics_candidates[-1].read_text())
                except Exception:
                    state = "invalid-metrics"
            cfg_hash = sha256(config_path) if config_path.exists() else None
            status["arms"][tag] = {
                "stage": stage_name, "state": state, "config_sha256": cfg_hash,
                "model": str(model) if model.exists() else None,
                "metrics": str(metrics_candidates[-1]) if metrics_candidates else None,
            }
            metric_text = "—"
            if metrics:
                vp = metrics.get("volume_psnr")
                sp = metrics.get("sobel_psnr")
                air = metrics.get("air", {})
                am = air.get("mae", {}).get("strict_air") if isinstance(air, dict) else None
                fpr = air.get("strict_air_fpr") if isinstance(air, dict) else None
                metric_text = f"volume PSNR={vp}; Sobel PSNR={sp}; air MAE={am}; FPR={fpr}"
            rows.append((stage_name, tag, arm.get("mode", ""), state, cfg_hash, metric_text))

    (web_root / "status.json").write_text(json.dumps(status, indent=2, allow_nan=True))
    row_html = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>LC64 air-artifact diagnosis</title><style>
body{{font:15px system-ui;margin:2rem;background:#111;color:#eee}}a{{color:#8cf}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #555;padding:.5rem;text-align:left}}
.complete{{color:#8f8}}code{{background:#222;padding:.15rem}}</style></head><body>
<h1>LC64 air-artifact split-cell diagnosis</h1>
<p>Commit <code>{html.escape(status['commit'])}</code>. This page includes queued, running, completed, failed, and pruned arms.</p>
<p><strong>Queued (gated)</strong> means planned but intentionally not launched until the preceding stage passes; it does not mean completed. Results accumulate here as each staged gate advances.</p>
<p><a href='manifest.yaml'>Manifest</a> · <a href='status.json'>Machine-readable status</a> · <a href='configs/'>Resolved configs</a></p>
<table><thead><tr><th>Stage</th><th>Arm</th><th>Mode</th><th>Status</th><th>Config SHA-256</th><th>Metrics</th></tr></thead>
<tbody>{row_html}</tbody></table></body></html>"""
    (web_root / "index.html").write_text(page)
    print(web_root / "index.html")


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
