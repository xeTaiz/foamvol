#!/usr/bin/env python3
"""GT-air diagnostic for the approved LC64 air-artifact diagnosis plan.

This module computes a fixed set of GT-derived air-ROI metrics on aligned 3D
prediction and ground-truth volumes. The metric set follows the approved
LC64 air-artifact diagnosis plan recorded in the project vault under
``radfoam-split-cell-fixed-64k-gate-2026-07-20`` ("Approved LC64 air-artifact
diagnosis plan (2026-07-23)").

Air / object partitioning is derived from the GT volume only, so the metric
set is reproducible without a paired training run:

    p99_gt       = np.percentile(gt, 99)
    raw_air      = gt <= 0.01 * p99_gt
    object       = ~raw_air
    halo         = (raw_air voxels adjacent to object under 26-connectivity,
                    via one iteration of binary_dilation of object_mask
                    intersected with raw_air)
    strict_air   = raw_air & ~halo
    fp_threshold = 0.05 * p99_gt

All metrics are JSON-safe (plain Python types, ``None`` for empty masks,
``float("inf")`` for zero-MSE PSNR). The CLI atomically writes the JSON
output and optionally saves the four boolean masks as a compressed ``.npz``.

Dependencies: numpy, scipy.ndimage (only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation


# --- Constants (fixed by the approved LC64 air-diagnosis plan) ---------------

AIR_FRACTION_OF_P99: float = 0.01   # raw_air = gt <= AIR_FRACTION_OF_P99 * p99_gt
FP_FRACTION_OF_P99: float = 0.05    # fp_threshold = FP_FRACTION_OF_P99 * p99_gt
HALO_ITERATIONS: int = 1            # one-voxel boundary halo
HALO_STRUCTURE: np.ndarray = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
DATA_RANGE_FLOOR: float = 1e-12     # guard against log10(0) in PSNR


class AirMetricsError(ValueError):
    """Raised on invalid inputs (shape mismatch, non-finite arrays, etc.)."""


# --- Public API ---------------------------------------------------------------

def load_pair(prediction_path: Any, gt_path: Any) -> tuple[np.ndarray, np.ndarray]:
    """Load prediction and GT arrays from ``.npy`` files and validate them.

    Both arrays must be 3D, have the same shape, and contain only finite
    values. Returns ``(pred, gt)`` as ``numpy.ndarray`` views.
    """
    pred = np.asarray(np.load(str(prediction_path)))
    gt = np.asarray(np.load(str(gt_path)))
    _validate_pair(pred, gt)
    return pred, gt


def _validate_pair(pred: np.ndarray, gt: np.ndarray) -> None:
    """Raise :class:`AirMetricsError` if the arrays are not usable."""
    if pred.ndim != 3 or gt.ndim != 3:
        raise AirMetricsError(
            f"Expected 3D arrays, got pred.ndim={pred.ndim}, gt.ndim={gt.ndim}"
        )
    if pred.shape != gt.shape:
        raise AirMetricsError(
            f"Shape mismatch: pred.shape={pred.shape} vs gt.shape={gt.shape}"
        )
    if not np.all(np.isfinite(pred)):
        raise AirMetricsError("Prediction contains non-finite values")
    if not np.all(np.isfinite(gt)):
        raise AirMetricsError("Ground truth contains non-finite values")


def compute_air_masks(gt: np.ndarray) -> dict:
    """Compute the raw_air / object / halo / strict_air boolean masks.

    Returns a dict containing:

        p99_gt       (float)        - 99th percentile of gt
        raw_air      (ndarray bool) - voxels classified as air
        object       (ndarray bool) - voxels classified as object
        halo         (ndarray bool) - boundary air voxels (subset of raw_air)
        strict_air   (ndarray bool) - raw_air voxels that are not halo
    """
    if gt.ndim != 3:
        raise AirMetricsError(f"compute_air_masks expects 3D gt, got ndim={gt.ndim}")
    p99_gt = float(np.percentile(gt, 99))
    raw_air = gt <= AIR_FRACTION_OF_P99 * p99_gt
    object_mask = ~raw_air
    # raw_air voxels adjacent to object under 26-connectivity:
    # binary_dilation(object) extends object by one; intersect with raw_air
    # keeps only those dilated voxels that are also raw_air.
    halo = binary_dilation(
        object_mask, structure=HALO_STRUCTURE, iterations=HALO_ITERATIONS
    ) & raw_air
    strict_air = raw_air & ~halo
    return {
        "p99_gt": p99_gt,
        "raw_air": raw_air,
        "object": object_mask,
        "halo": halo,
        "strict_air": strict_air,
    }


def compute_air_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    masks: dict | None = None,
) -> dict:
    """Compute the full JSON-safe air metric dict.

    If ``masks`` is ``None`` the masks are computed from ``gt`` first.
    All entries are plain Python types (``float``, ``int``, ``bool``, or
    ``None``); no numpy scalars leak into the returned dict.
    """
    _validate_pair(pred, gt)
    if masks is None:
        masks = compute_air_masks(gt)

    p99_gt = float(masks["p99_gt"])
    data_range = float(max(p99_gt, DATA_RANGE_FLOOR))
    fp_threshold = float(FP_FRACTION_OF_P99 * p99_gt)

    raw_air = masks["raw_air"]
    object_mask = masks["object"]
    halo = masks["halo"]
    strict_air = masks["strict_air"]

    n_total = int(raw_air.size)
    n_raw_air = int(raw_air.sum())
    n_object = int(object_mask.sum())
    n_halo = int(halo.sum())
    n_strict_air = int(strict_air.sum())

    full_mask = np.ones_like(raw_air)

    abs_pred = np.abs(pred)

    metrics: dict = {
        "p99_gt": p99_gt,
        "fp_threshold": fp_threshold,
        "data_range": data_range,
        "voxel_counts": {
            "n_total": n_total,
            "n_raw_air": n_raw_air,
            "n_object": n_object,
            "n_halo": n_halo,
            "n_strict_air": n_strict_air,
        },
        "voxel_fractions": {
            "raw_air": (float(n_raw_air) / float(n_total)) if n_total else None,
            "object": (float(n_object) / float(n_total)) if n_total else None,
            "halo": (float(n_halo) / float(n_total)) if n_total else None,
            "strict_air": (float(n_strict_air) / float(n_total)) if n_total else None,
        },
        "mae": {
            "full": _safe_mae(pred, gt, full_mask),
            "object": _safe_mae(pred, gt, object_mask),
            "halo": _safe_mae(pred, gt, halo),
            "strict_air": _safe_mae(pred, gt, strict_air),
        },
        "psnr": {
            "full": _safe_psnr(pred, gt, full_mask, data_range),
            "object": _safe_psnr(pred, gt, object_mask, data_range),
            "halo": _safe_psnr(pred, gt, halo, data_range),
            "strict_air": _safe_psnr(pred, gt, strict_air, data_range),
        },
        "strict_air_abs_pred": {
            "p95": _safe_percentile(abs_pred[strict_air], 95),
            "p99": _safe_percentile(abs_pred[strict_air], 99),
        },
        "strict_air_fpr": (
            float(np.mean(abs_pred[strict_air] > fp_threshold))
            if strict_air.any()
            else None
        ),
        "pred": {
            "finite": bool(np.all(np.isfinite(pred))) if pred.size else None,
            "min": float(pred.min()) if pred.size else None,
            "max": float(pred.max()) if pred.size else None,
            "mean": float(pred.mean()) if pred.size else None,
        },
    }
    return metrics


# --- Internal helpers ---------------------------------------------------------

def _safe_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Mean absolute error on ``mask``; ``None`` for empty masks."""
    if not mask.any():
        return None
    return float(np.mean(np.abs(pred[mask] - gt[mask])))


def _safe_mse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Mean squared error on ``mask``; ``None`` for empty masks."""
    if not mask.any():
        return None
    return float(np.mean((pred[mask] - gt[mask]) ** 2))


def _safe_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, data_range: float):
    """PSNR with ``inf`` for zero MSE and ``None`` for empty masks."""
    mse = _safe_mse(pred, gt, mask)
    if mse is None:
        return None
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def _safe_percentile(values: np.ndarray, q: float):
    """``np.percentile`` with ``None`` on empty input."""
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


# --- Output -------------------------------------------------------------------

def write_metrics_json(metrics: dict, path: Any) -> None:
    """Atomically write ``metrics`` to ``path`` as JSON.

    Uses a same-directory temp file + ``os.replace`` so a crash or
    concurrent reader never sees a half-written JSON file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=out_path.name + ".", suffix=".tmp", dir=str(out_path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True, allow_nan=True)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def save_masks_npz(masks: dict, path: Any) -> None:
    """Save the air masks dict as a compressed ``.npz`` file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **masks)


# --- CLI ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute GT-air metrics on a paired prediction/GT volume and "
            "write the result as JSON (and optional compressed masks npz)."
        ),
    )
    p.add_argument("--prediction", required=True,
                   help="Path to prediction .npy volume (3D float).")
    p.add_argument("--gt", required=True,
                   help="Path to ground-truth .npy volume (3D float).")
    p.add_argument("--output", required=True,
                   help="Path to metrics JSON output (written atomically).")
    p.add_argument("--mask-output", default=None,
                   help="Optional path to a compressed masks .npz output.")
    return p


def main(argv: list | None = None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    pred, gt = load_pair(args.prediction, args.gt)
    masks = compute_air_masks(gt)
    metrics = compute_air_metrics(pred, gt, masks=masks)
    write_metrics_json(metrics, args.output)
    if args.mask_output:
        save_masks_npz(masks, args.mask_output)
    return metrics


if __name__ == "__main__":
    main()
