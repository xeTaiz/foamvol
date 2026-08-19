"""Contract tests for the canonical R2-Gaussian volume PSNR helper."""

import math

import numpy as np
import torch

from radfoam_model.utils import compute_volume_psnr


def _expected_r2_psnr(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mse = np.mean((pred - gt) ** 2)
    return float(10 * np.log10(np.max(gt) ** 2 / mse))


def test_numpy_and_torch_paths_match_r2_formula():
    gt = np.linspace(0.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
    pred = gt * np.float32(0.8) + np.float32(0.03)
    expected = _expected_r2_psnr(pred, gt)

    numpy_result = compute_volume_psnr(pred, gt)
    torch_result = compute_volume_psnr(torch.from_numpy(pred), torch.from_numpy(gt))

    assert math.isclose(numpy_result, expected, rel_tol=1e-6)
    assert math.isclose(torch_result, expected, rel_tol=1e-6)


def test_nonzero_gt_floor_uses_gt_max_not_peak_to_peak_range():
    gt = np.linspace(0.25, 1.0, 8, dtype=np.float32).reshape(2, 2, 2)
    pred = gt + np.float32(0.1)

    result = compute_volume_psnr(pred, gt)
    expected = _expected_r2_psnr(pred, gt)
    range_based = float(
        10 * np.log10((float(gt.max()) - float(gt.min())) ** 2 / 0.01)
    )

    assert math.isclose(result, expected, rel_tol=1e-6)
    assert not math.isclose(result, range_based, rel_tol=1e-3)


def test_identical_volumes_return_positive_infinity():
    gt = np.ones((2, 2, 2), dtype=np.float32)
    assert compute_volume_psnr(gt, gt) == float("inf")
    assert compute_volume_psnr(torch.from_numpy(gt), torch.from_numpy(gt)) == float("inf")


def test_zero_gt_peak_with_nonzero_error_returns_negative_infinity():
    gt = np.zeros((2, 2, 2), dtype=np.float32)
    pred = np.ones_like(gt)
    assert compute_volume_psnr(pred, gt) == float("-inf")
    assert compute_volume_psnr(torch.from_numpy(pred), torch.from_numpy(gt)) == float("-inf")
