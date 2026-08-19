"""Pin the one voxel-grid convention: centres, and align_corners=False.

These tests exist because mixing the centre and endpoint conventions silently
cost 1.7-1.9 dB of volume PSNR and reordered experiment arms whose true
separation was inside the noise floor.  They are pure torch/numpy on CPU -- no
CUDA, no checkpoints.

The decisive test is :func:`test_grid_sample_round_trips_voxel_centers`: it
asserts that world coordinates produced by ``voxel_center_grid`` read back the
*exact* stored voxel values through ``grid_sample`` when (and only when)
``align_corners=False``.  If someone flips that flag back, this fails.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from voxel_grid import (
    ALIGN_CORNERS,
    subvoxel_offsets,
    voxel_center_coords,
    voxel_center_coords_np,
    voxel_center_grid,
    voxel_center_grid_np,
    voxel_pitch,
)


def test_align_corners_constant_is_false():
    """The whole point of the module."""
    assert ALIGN_CORNERS is False


def test_center_coords_geometry():
    R, extent = 8, 1.0
    c = voxel_center_coords(R, extent, dtype=torch.float64)

    assert c.shape == (R,)
    # First/last centres sit half a voxel inside the box, never on the face.
    assert torch.isclose(c[0], torch.tensor(-extent + extent / R, dtype=torch.float64))
    assert torch.isclose(c[-1], torch.tensor(extent - extent / R, dtype=torch.float64))
    # Uniform pitch 2*extent/R -- NOT the endpoint pitch 2*extent/(R-1).
    diffs = c[1:] - c[:-1]
    assert torch.allclose(diffs, torch.full_like(diffs, voxel_pitch(R, extent)))
    # Strictly interior, and symmetric about 0.
    assert bool((c.abs() < extent).all())
    assert torch.allclose(c, -c.flip(0))


def test_center_coords_differ_from_linspace():
    """Guard against a silent regression back to the endpoint grid."""
    R = 256
    centers = voxel_center_coords(R, 1.0, dtype=torch.float64)
    endpoints = torch.linspace(-1.0, 1.0, R, dtype=torch.float64)

    # Half a voxel apart at the edges, and the endpoint grid touches the faces.
    assert torch.isclose(endpoints[0], torch.tensor(-1.0, dtype=torch.float64))
    assert not torch.allclose(centers, endpoints)
    assert float((centers - endpoints).abs().max()) > 0.5 * voxel_pitch(R, 1.0) * 0.99


def test_numpy_and_torch_agree():
    R = 32
    t = voxel_center_coords(R, 1.7, dtype=torch.float64).numpy()
    n = voxel_center_coords_np(R, 1.7, dtype=np.float64)
    assert np.allclose(t, n)

    tg = voxel_center_grid(R, 1.7, dtype=torch.float64).numpy()
    ng = voxel_center_grid_np(R, 1.7, dtype=np.float64)
    assert np.allclose(tg, ng)


def test_center_grid_reshapes_to_volume():
    R = 5
    g = voxel_center_grid(R, 1.0, dtype=torch.float64).reshape(R, R, R, 3)
    c = voxel_center_coords(R, 1.0, dtype=torch.float64)
    # 'ij' ordering: axis 0 varies x, axis 1 y, axis 2 z.
    assert torch.allclose(g[:, 0, 0, 0], c)
    assert torch.allclose(g[0, :, 0, 1], c)
    assert torch.allclose(g[0, 0, :, 2], c)


def test_grid_sample_round_trips_voxel_centers():
    """World centres must read back stored voxel values EXACTLY.

    This is the contract that makes prediction grids and GT volumes comparable.
    """
    R, extent = 6, 1.0
    torch.manual_seed(0)
    vol = torch.randn(R, R, R, dtype=torch.float64)

    centers = voxel_center_grid(R, extent, dtype=torch.float64)  # (R^3, 3)
    # grid_sample expects normalized coords in (x, y, z) = (W, H, D) order,
    # while our volume is indexed [x, y, z]; flip to address it as (D=x, H=y, W=z).
    norm = (centers / extent).flip(-1).reshape(1, 1, 1, -1, 3)
    vol_5d = vol.reshape(1, 1, R, R, R)

    got = F.grid_sample(
        vol_5d, norm, mode="bilinear", padding_mode="border",
        align_corners=ALIGN_CORNERS,
    ).reshape(-1)

    assert torch.allclose(got, vol.reshape(-1), atol=1e-12), (
        "voxel centres must sample stored values exactly under "
        f"align_corners={ALIGN_CORNERS}"
    )


def test_align_corners_true_is_wrong_for_center_grid():
    """The bug, asserted: align_corners=True misregisters the same query."""
    R, extent = 6, 1.0
    torch.manual_seed(0)
    vol = torch.randn(R, R, R, dtype=torch.float64)

    centers = voxel_center_grid(R, extent, dtype=torch.float64)
    norm = (centers / extent).flip(-1).reshape(1, 1, 1, -1, 3)
    vol_5d = vol.reshape(1, 1, R, R, R)

    wrong = F.grid_sample(
        vol_5d, norm, mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(-1)

    # Not merely different -- it is a systematic half-voxel-scale error.
    assert not torch.allclose(wrong, vol.reshape(-1), atol=1e-6)
    assert float((wrong - vol.reshape(-1)).abs().max()) > 1e-3


def test_endpoint_grid_is_what_align_corners_true_matches():
    """Confirms the equivalence claim: align_corners=True == endpoint grid."""
    R = 6
    torch.manual_seed(1)
    vol = torch.randn(R, R, R, dtype=torch.float64)
    vol_5d = vol.reshape(1, 1, R, R, R)

    lin = torch.linspace(-1.0, 1.0, R, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    endpoints = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    norm = endpoints.flip(-1).reshape(1, 1, 1, -1, 3)

    got = F.grid_sample(
        vol_5d, norm, mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(-1)
    assert torch.allclose(got, vol.reshape(-1), atol=1e-12)


def test_subvoxel_offsets_tile_the_voxel():
    R, extent, k = 16, 1.0, 4
    off = subvoxel_offsets(k, R, extent, dtype=torch.float64)
    pitch = voxel_pitch(R, extent)

    assert off.shape == (k ** 3, 3)
    # Every sub-sample stays strictly inside its voxel.
    assert float(off.abs().max()) < 0.5 * pitch
    # Sub-centres are symmetric, so they average to the voxel centre.
    assert torch.allclose(off.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-15)
    # k=1 degenerates to the voxel centre itself.
    assert torch.allclose(
        subvoxel_offsets(1, R, extent, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
    )
    # Recursive centres: sub-pitch is pitch/k.
    uniq = torch.unique(off[:, 0])
    assert torch.allclose(
        uniq[1:] - uniq[:-1],
        torch.full((k - 1,), pitch / k, dtype=torch.float64),
    )


def test_supersampling_preserves_the_grid_mean_for_linear_fields():
    """A linear field must be unbiased by supersampling -- catches offset bugs."""
    R, extent, k = 8, 1.0, 3
    centers = voxel_center_grid(R, extent, dtype=torch.float64)
    off = subvoxel_offsets(k, R, extent, dtype=torch.float64)
    q = (centers.unsqueeze(1) + off.unsqueeze(0)).reshape(-1, 3)

    def field(p):  # linear in x, y, z
        return 0.3 * p[:, 0] - 1.1 * p[:, 1] + 2.0 * p[:, 2] + 0.7

    supersampled = field(q).reshape(centers.shape[0], k ** 3).mean(1)
    assert torch.allclose(supersampled, field(centers), atol=1e-12)
