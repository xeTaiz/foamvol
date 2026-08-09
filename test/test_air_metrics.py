#!/usr/bin/env python3
"""Tests for :mod:`air_metrics` — pure NumPy/SciPy GT-air metric utility.

Verifies thresholds, halo exclusion, strict-air FPR / percentiles, PSNR
(zero-MSE -> inf, known value), shape / non-finite rejection, the data
range floor, empty-mask null behavior, and the CLI's atomic JSON +
optional masks.npz output.

Run with:  python3 test/test_air_metrics.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

# Repo root on path so we can import air_metrics next to it.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import air_metrics  # noqa: E402


# --- Helpers ------------------------------------------------------------------

def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)


def _gt_with_object() -> np.ndarray:
    """7x7x7 GT: a 3x3x3 object cube centered at (3,3,3), else air (=0).

    This shape keeps ``p99_gt == 1.0`` (27 ones out of 343) and gives
    clean non-trivial strict-air / halo partitions:

      n_total      = 343
      n_object     = 27   (the 3x3x3 object)
      n_halo       = 98   (raw_air within 26-neighborhood of object)
      n_strict_air = 218  (raw_air outside that neighborhood)
    """
    gt = np.zeros((7, 7, 7), dtype=np.float32)
    gt[2:5, 2:5, 2:5] = 1.0
    return gt


# --- Tests --------------------------------------------------------------------

def test_compute_air_masks_thresholds() -> None:
    print("\n[test] compute_air_masks thresholds")
    gt = _gt_with_object()
    masks = air_metrics.compute_air_masks(gt)

    check(isinstance(masks["p99_gt"], float), "p99_gt is a float")
    check(masks["p99_gt"] == 1.0, f"p99_gt == 1.0 (got {masks['p99_gt']})")
    check(masks["raw_air"].dtype == bool, "raw_air is bool")
    check(masks["object"].dtype == bool, "object is bool")
    check(masks["halo"].dtype == bool, "halo is bool")
    check(masks["strict_air"].dtype == bool, "strict_air is bool")

    check(masks["raw_air"].sum() == 316, f"n_raw_air == 316 (got {masks['raw_air'].sum()})")
    check(masks["object"].sum() == 27, f"n_object == 27 (got {masks['object'].sum()})")
    check(masks["halo"].sum() == 98, f"n_halo == 98 (got {masks['halo'].sum()})")
    check(masks["strict_air"].sum() == 218, f"n_strict_air == 218 (got {masks['strict_air'].sum()})")

    # raw_air and object partition the volume.
    check(
        (masks["raw_air"] | masks["object"]).all(),
        "raw_air | object covers every voxel",
    )
    check(
        not (masks["raw_air"] & masks["object"]).any(),
        "raw_air & object is empty",
    )


def test_halo_exclusion() -> None:
    print("\n[test] halo exclusion (26-connectivity, one-iteration dilation)")
    gt = _gt_with_object()
    masks = air_metrics.compute_air_masks(gt)
    halo = masks["halo"]
    strict_air = masks["strict_air"]
    object_mask = masks["object"]

    # Halo and strict_air are disjoint and together cover raw_air.
    check(not (halo & strict_air).any(), "halo & strict_air is empty")
    check(
        (halo | strict_air).sum() == masks["raw_air"].sum(),
        "halo | strict_air == raw_air",
    )

    # halo is exactly the raw_air voxels in the 26-neighborhood of object.
    expected_halo = (
        binary_dilation(object_mask, structure=air_metrics.HALO_STRUCTURE,
                        iterations=air_metrics.HALO_ITERATIONS)
        & masks["raw_air"]
    )
    check(np.array_equal(halo, expected_halo), "halo matches binary_dilation(object) & raw_air")

    # Spot-check specific voxels.
    check(not halo[2, 2, 2].item(), "object voxel itself is not halo")
    check(halo[1, 2, 2].item(), "voxel (1,2,2) is halo (adjacent to object)")
    check(halo[5, 4, 4].item(), "voxel (5,4,4) is halo (adjacent to object)")
    check(not strict_air[1, 2, 2].item(), "voxel (1,2,2) is not strict_air")
    check(strict_air[0, 0, 0].item(), "outer corner (0,0,0) is strict_air")
    check(strict_air[6, 6, 6].item(), "outer corner (6,6,6) is strict_air")


def test_strict_air_fpr() -> None:
    print("\n[test] strict_air FPR = mean(|pred| > fp_threshold)")
    gt = _gt_with_object()
    masks = air_metrics.compute_air_masks(gt)
    pred = np.zeros_like(gt)

    fp_threshold = 0.05 * masks["p99_gt"]
    check(fp_threshold == 0.05, f"fp_threshold == 0.05 (got {fp_threshold})")

    # Half of strict_air voxels above threshold (0.1), half below (0.01).
    n = int(masks["strict_air"].sum())
    vals = np.where(np.arange(n) < n // 2, 0.1, 0.01).astype(np.float32)
    pred[masks["strict_air"]] = vals

    metrics = air_metrics.compute_air_metrics(pred, gt, masks=masks)
    expected = 0.5
    check(
        metrics["strict_air_fpr"] is not None
        and abs(metrics["strict_air_fpr"] - expected) < 1e-9,
        f"strict_air_fpr ~ 0.5 (got {metrics['strict_air_fpr']})",
    )

    # All-zero pred -> FPR = 0 (nothing above threshold).
    pred2 = np.zeros_like(gt)
    metrics2 = air_metrics.compute_air_metrics(pred2, gt, masks=masks)
    check(metrics2["strict_air_fpr"] == 0.0, "all-zero pred -> FPR = 0")

    # All-large pred -> FPR = 1.
    pred3 = np.full_like(gt, 1.0)
    metrics3 = air_metrics.compute_air_metrics(pred3, gt, masks=masks)
    check(metrics3["strict_air_fpr"] == 1.0, "all-large pred -> FPR = 1")


def test_strict_air_percentiles() -> None:
    print("\n[test] strict-air abs(pred) P95 / P99")
    gt = _gt_with_object()
    masks = air_metrics.compute_air_masks(gt)
    pred = np.zeros_like(gt)

    n = int(masks["strict_air"].sum())
    pred[masks["strict_air"]] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    # Plant a known large value just inside strict_air so percentiles are exact.
    pred[0, 0, 0] = 0.0  # leave corner at 0 so the linear ramp is monotone
    # Re-spread with the planted corner = 0 unchanged.
    pred[masks["strict_air"]] = np.linspace(0.0, 1.0, n, dtype=np.float32)

    abs_pred_strict = np.abs(pred[masks["strict_air"]])
    expected_p95 = float(np.percentile(abs_pred_strict, 95))
    expected_p99 = float(np.percentile(abs_pred_strict, 99))

    metrics = air_metrics.compute_air_metrics(pred, gt, masks=masks)
    p95 = metrics["strict_air_abs_pred"]["p95"]
    p99 = metrics["strict_air_abs_pred"]["p99"]
    check(p95 is not None and abs(p95 - expected_p95) < 1e-5,
          f"p95 matches np.percentile (got {p95}, expected {expected_p95})")
    check(p99 is not None and abs(p99 - expected_p99) < 1e-5,
          f"p99 matches np.percentile (got {p99}, expected {expected_p99})")


def test_psnr_zero_mse_is_inf() -> None:
    print("\n[test] PSNR = inf for zero MSE")
    gt = _gt_with_object()
    pred = gt.copy()  # identical -> MSE = 0 everywhere
    metrics = air_metrics.compute_air_metrics(pred, gt)
    # Map PSNR region -> which count key to consult.
    region_to_count = {
        "full": "n_total",
        "object": "n_object",
        "halo": "n_halo",
        "strict_air": "n_strict_air",
    }
    for region, psnr in metrics["psnr"].items():
        n = metrics["voxel_counts"][region_to_count[region]]
        if n > 0:
            check(psnr == float("inf"), f"psnr[{region}] == inf (got {psnr})")
        else:
            check(psnr is None, f"psnr[{region}] is None for empty mask (got {psnr})")


def test_psnr_known_value() -> None:
    print("\n[test] PSNR closed-form value")
    # All-zero gt, pred = c on every voxel -> MSE = c^2.
    gt = np.zeros((5, 5, 5), dtype=np.float32)
    c = 0.1
    pred = np.full_like(gt, c)
    metrics = air_metrics.compute_air_metrics(pred, gt)

    # p99_gt on all-zeros is 0 -> data_range = 1e-12.
    expected = 20.0 * np.log10(metrics["data_range"]) - 10.0 * np.log10(c * c)
    check(
        metrics["psnr"]["full"] is not None
        and abs(metrics["psnr"]["full"] - expected) < 1e-6,
        f"psnr[full] matches closed-form (got {metrics['psnr']['full']}, expected {expected})",
    )
    check(metrics["data_range"] == 1e-12, "data_range == 1e-12 when p99_gt == 0")


def test_data_range_floor() -> None:
    print("\n[test] data_range = max(p99_gt, 1e-12)")
    gt = np.zeros((5, 5, 5), dtype=np.float32)
    pred = np.full_like(gt, 1e-15)
    metrics = air_metrics.compute_air_metrics(pred, gt)
    check(metrics["data_range"] >= 1e-12,
          f"data_range >= 1e-12 (got {metrics['data_range']})")
    check(np.isfinite(metrics["psnr"]["full"]),
          "psnr[full] is finite thanks to data_range floor")


def test_empty_strict_air_returns_null() -> None:
    print("\n[test] empty strict_air returns None for strict-only metrics")
    # All nonzero gt -> no raw_air, no halo, no strict_air.
    gt = np.ones((5, 5, 5), dtype=np.float32)
    pred = np.full_like(gt, 0.5)
    metrics = air_metrics.compute_air_metrics(pred, gt)
    check(metrics["voxel_counts"]["n_strict_air"] == 0, "n_strict_air == 0")
    check(metrics["voxel_counts"]["n_halo"] == 0, "n_halo == 0")
    check(metrics["voxel_counts"]["n_raw_air"] == 0, "n_raw_air == 0")
    check(metrics["strict_air_abs_pred"]["p95"] is None, "strict p95 is None")
    check(metrics["strict_air_abs_pred"]["p99"] is None, "strict p99 is None")
    check(metrics["strict_air_fpr"] is None, "strict_air_fpr is None")
    check(metrics["mae"]["strict_air"] is None, "mae[strict_air] is None")
    check(metrics["mae"]["halo"] is None, "mae[halo] is None")
    check(metrics["psnr"]["strict_air"] is None, "psnr[strict_air] is None")
    check(metrics["psnr"]["halo"] is None, "psnr[halo] is None")


def test_pred_stats() -> None:
    print("\n[test] pred stats (finite / min / max / mean)")
    gt = np.zeros((5, 5, 5), dtype=np.float32)
    pred = np.full_like(gt, 0.3)
    metrics = air_metrics.compute_air_metrics(pred, gt)
    check(metrics["pred"]["finite"] is True, "pred.finite is True")
    # 0.3 is not exactly representable in float32 -> compare approximately.
    for key in ("min", "max", "mean"):
        v = metrics["pred"][key]
        check(v is not None and abs(v - 0.3) < 1e-6,
              f"pred.{key} ~ 0.3 (got {v})")


def test_load_pair_validates_shape_and_finite() -> None:
    print("\n[test] load_pair rejects bad shapes and non-finite values")
    tmp = Path(tempfile.mkdtemp())

    # Shape mismatch.
    p1, g1 = tmp / "p1.npy", tmp / "g1.npy"
    np.save(p1, np.zeros((5, 5, 5), dtype=np.float32))
    np.save(g1, np.zeros((5, 5, 6), dtype=np.float32))
    raised = False
    try:
        air_metrics.load_pair(str(p1), str(g1))
    except air_metrics.AirMetricsError:
        raised = True
    check(raised, "shape mismatch raises AirMetricsError")

    # NaN in prediction.
    p2, g2 = tmp / "p2.npy", tmp / "g2.npy"
    bad = np.zeros((5, 5, 5), dtype=np.float32); bad[0, 0, 0] = np.nan
    np.save(p2, bad)
    np.save(g2, np.zeros((5, 5, 5), dtype=np.float32))
    raised = False
    try:
        air_metrics.load_pair(str(p2), str(g2))
    except air_metrics.AirMetricsError:
        raised = True
    check(raised, "non-finite pred raises AirMetricsError")

    # Inf in GT.
    p3, g3 = tmp / "p3.npy", tmp / "g3.npy"
    np.save(p3, np.zeros((5, 5, 5), dtype=np.float32))
    bad3 = np.zeros((5, 5, 5), dtype=np.float32); bad3[0, 0, 0] = np.inf
    np.save(g3, bad3)
    raised = False
    try:
        air_metrics.load_pair(str(p3), str(g3))
    except air_metrics.AirMetricsError:
        raised = True
    check(raised, "non-finite gt raises AirMetricsError")

    # 2D array rejection.
    p4, g4 = tmp / "p4.npy", tmp / "g4.npy"
    np.save(p4, np.zeros((5, 5), dtype=np.float32))
    np.save(g4, np.zeros((5, 5), dtype=np.float32))
    raised = False
    try:
        air_metrics.load_pair(str(p4), str(g4))
    except air_metrics.AirMetricsError:
        raised = True
    check(raised, "2D array raises AirMetricsError")


def test_json_safety() -> None:
    print("\n[test] JSON-safe metrics dict (no numpy scalars, inf allowed)")
    gt = _gt_with_object()
    pred = gt.copy()  # zero MSE -> psnr == inf
    metrics = air_metrics.compute_air_metrics(pred, gt)
    # Must serialize to JSON without TypeError.
    s = json.dumps(metrics, allow_nan=True)
    check(isinstance(s, str) and len(s) > 0, "metrics dict serializes to JSON")
    check("Infinity" in s or "inf" in s.lower(),
          "JSON output contains the inf PSNR (allowed via allow_nan=True)")

    # Ensure no numpy scalars leaked into the dict.
    def _all_native(obj):
        if isinstance(obj, dict):
            return all(_all_native(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_all_native(v) for v in obj)
        return isinstance(obj, (int, float, str, bool, type(None)))

    check(_all_native(metrics), "all values are JSON-native Python types")


def test_atomic_write_no_partial_file() -> None:
    print("\n[test] write_metrics_json is atomic")
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "metrics.json"
    metrics = air_metrics.compute_air_metrics(np.zeros((3, 3, 3), np.float32),
                                              np.zeros((3, 3, 3), np.float32))
    air_metrics.write_metrics_json(metrics, out)
    check(out.exists(), "output file exists")
    # Verify no leftover .tmp file in the directory.
    leftovers = [p for p in tmp.iterdir() if p.name.endswith(".tmp")]
    check(not leftovers, f"no leftover .tmp files (got {leftovers})")


def test_cli_output() -> None:
    print("\n[test] CLI writes metrics JSON")
    tmp = Path(tempfile.mkdtemp())
    p_path, g_path = tmp / "pred.npy", tmp / "gt.npy"
    out_path = tmp / "metrics.json"

    gt = _gt_with_object()
    pred = np.zeros_like(gt)
    np.save(p_path, pred)
    np.save(g_path, gt)

    script = REPO_ROOT / "air_metrics.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--prediction", str(p_path),
         "--gt", str(g_path),
         "--output", str(out_path)],
        capture_output=True, text=True,
    )
    check(res.returncode == 0,
          f"CLI exit code 0 (stderr: {res.stderr.strip() or 'none'})")
    check(out_path.exists(), "metrics.json exists")

    with open(out_path) as f:
        metrics = json.load(f)

    for key in ("p99_gt", "fp_threshold", "data_range",
                "voxel_counts", "voxel_fractions",
                "mae", "psnr",
                "strict_air_abs_pred", "strict_air_fpr", "pred"):
        check(key in metrics, f"metrics.json has '{key}'")

    check(metrics["voxel_counts"]["n_object"] == 27,
          f"voxel_counts.n_object == 27 (got {metrics['voxel_counts']['n_object']})")
    check(metrics["voxel_counts"]["n_halo"] == 98,
          f"voxel_counts.n_halo == 98 (got {metrics['voxel_counts']['n_halo']})")
    check(metrics["voxel_counts"]["n_strict_air"] == 218,
          f"voxel_counts.n_strict_air == 218 (got {metrics['voxel_counts']['n_strict_air']})")
    # Atomic: no leftover .tmp files.
    leftovers = [p for p in tmp.iterdir() if p.name.endswith(".tmp")]
    check(not leftovers, f"no leftover .tmp files (got {leftovers})")


def test_cli_mask_output() -> None:
    print("\n[test] CLI writes masks.npz when --mask-output is given")
    tmp = Path(tempfile.mkdtemp())
    p_path, g_path = tmp / "pred.npy", tmp / "gt.npy"
    out_path = tmp / "metrics.json"
    mask_path = tmp / "masks.npz"

    gt = _gt_with_object()
    np.save(p_path, np.zeros_like(gt))
    np.save(g_path, gt)

    script = REPO_ROOT / "air_metrics.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--prediction", str(p_path),
         "--gt", str(g_path),
         "--output", str(out_path),
         "--mask-output", str(mask_path)],
        capture_output=True, text=True,
    )
    check(res.returncode == 0,
          f"CLI exit code 0 (stderr: {res.stderr.strip() or 'none'})")
    check(mask_path.exists(), "masks.npz exists")

    data = np.load(mask_path)
    for key in ("raw_air", "object", "halo", "strict_air", "p99_gt"):
        check(key in data.files, f"masks.npz has '{key}'")
    check(int(data["halo"].sum()) == 98, "masks.npz halo sum == 98")
    check(int(data["strict_air"].sum()) == 218, "masks.npz strict_air sum == 218")
    check(float(data["p99_gt"]) == 1.0, "masks.npz p99_gt == 1.0")


def test_cli_rejects_mismatched_shapes() -> None:
    print("\n[test] CLI exits non-zero on shape mismatch")
    tmp = Path(tempfile.mkdtemp())
    p_path, g_path = tmp / "pred.npy", tmp / "gt.npy"
    out_path = tmp / "metrics.json"
    np.save(p_path, np.zeros((5, 5, 5), dtype=np.float32))
    np.save(g_path, np.zeros((5, 5, 6), dtype=np.float32))

    script = REPO_ROOT / "air_metrics.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--prediction", str(p_path),
         "--gt", str(g_path),
         "--output", str(out_path)],
        capture_output=True, text=True,
    )
    check(res.returncode != 0, f"CLI exit code != 0 on shape mismatch (got {res.returncode})")
    check(not out_path.exists(), "metrics.json NOT created on failed CLI run")


# --- Runner -------------------------------------------------------------------

def main() -> None:
    tests = [
        test_compute_air_masks_thresholds,
        test_halo_exclusion,
        test_strict_air_fpr,
        test_strict_air_percentiles,
        test_psnr_zero_mse_is_inf,
        test_psnr_known_value,
        test_data_range_floor,
        test_empty_strict_air_returns_null,
        test_pred_stats,
        test_load_pair_validates_shape_and_finite,
        test_json_safety,
        test_atomic_write_no_partial_file,
        test_cli_output,
        test_cli_mask_output,
        test_cli_rejects_mismatched_shapes,
    ]
    print(f"Running {len(tests)} air_metrics tests...")
    for t in tests:
        t()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
