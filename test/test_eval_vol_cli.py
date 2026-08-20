"""CLI failures that keep sweep markers and metric tables trustworthy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest

pytest.importorskip("torch")

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_shape_mismatch_exits_nonzero_without_json(tmp_path: Path):
    """A sweep must fail, not mark DONE without metrics, on unequal grids."""
    pred = tmp_path / "pred.npy"
    gt = tmp_path / "gt.npy"
    metrics = tmp_path / "metrics.json"
    np.save(pred, np.zeros((8, 8, 8), dtype=np.float32))
    np.save(gt, np.zeros((9, 9, 9), dtype=np.float32))

    result = subprocess.run(
        [sys.executable, "eval_vol.py", str(pred), str(gt), "--cpu", "--json", str(metrics)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "shape mismatch" in result.stderr
    assert not metrics.exists()
