"""Canonical voxel-grid convention for every volume in this repo.

There is exactly one convention, and it is **voxel centers**:

    a volume of resolution R covering [-extent, extent] has voxel pitch
    2*extent/R, and voxel i is represented by its centre

        c_i = -extent + (i + 0.5) * (2*extent/R)

so c_0 = -extent + extent/R and c_{R-1} = extent - extent/R.  No sample ever
lands exactly on +/-extent, and the R voxels tile the box without
double-counting the boundary.

The rejected alternative is the *endpoint* convention,
``linspace(-extent, extent, R)``, which has pitch ``2*extent/(R-1)`` and places
samples on the box faces.  Mixing the two was a real, measured bug: sampling a
centre-defined GT on the endpoint grid cost **1.7-1.9 dB** of volume PSNR and
**2.7-3.1 dB** of Sobel PSNR, and because the penalty depends on how sharp the
reconstruction is, it varied per run (spread 0.21 dB) and reordered arms whose
true difference was under ~0.25 dB.  See ``specs/VOXEL-GRID-CONVENTION-v1.md``.

Two things must agree for a volume comparison to be meaningful:

1.  **Where predictions are sampled** -- use :func:`voxel_center_coords` /
    :func:`voxel_center_grid` (or their numpy twins) instead of ``linspace``.
2.  **How GT volumes are read** -- ``F.grid_sample`` and ``F.interpolate`` must
    use ``align_corners=`` :data:`ALIGN_CORNERS`, i.e. ``False``.
    ``align_corners=True`` *is* the endpoint convention expressed as a boolean:
    it maps normalized -1/+1 onto the first/last voxel **centres**, giving pitch
    2/(R-1).  With ``align_corners=False`` normalized -1/+1 land on the outer
    **faces** of the first/last voxels, so a world coordinate in
    [-extent, extent] divided by ``extent`` addresses a centre-defined volume
    exactly.  ``test/test_voxel_grid_convention.py`` pins this round-trip.

Dependencies: torch and numpy only; safe to import from anywhere.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "ALIGN_CORNERS",
    "voxel_center_coords",
    "voxel_center_coords_np",
    "voxel_center_grid",
    "voxel_center_grid_np",
    "voxel_pitch",
    "subvoxel_offsets",
]


#: ``align_corners`` for every ``grid_sample``/``interpolate`` on a volume that
#: follows the voxel-centre convention.  Never pass ``True``.
ALIGN_CORNERS = False


def voxel_pitch(resolution: int, extent: float = 1.0) -> float:
    """Edge length of one voxel: ``2*extent/resolution``."""
    return 2.0 * float(extent) / int(resolution)


def voxel_center_coords(
    resolution: int,
    extent: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """1-D voxel-centre coordinates: ``(R,)`` spanning ``(-extent, extent)``.

    Replaces ``torch.linspace(-extent, extent, resolution)``, which is the
    endpoint convention and is off by half a voxel with the wrong pitch.
    """
    idx = torch.arange(int(resolution), device=device, dtype=dtype)
    return -float(extent) + (idx + 0.5) * voxel_pitch(resolution, extent)


def voxel_center_coords_np(
    resolution: int, extent: float = 1.0, dtype=np.float32
) -> np.ndarray:
    """Numpy twin of :func:`voxel_center_coords`."""
    idx = np.arange(int(resolution), dtype=dtype)
    return (-float(extent) + (idx + 0.5) * voxel_pitch(resolution, extent)).astype(dtype)


def voxel_center_grid(
    resolution: int,
    extent: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Flattened ``(R**3, 3)`` voxel centres in ``ij`` (x, y, z) order.

    Row order matches ``reshape(R, R, R)`` indexing, so the result can be
    reshaped straight back into a volume.
    """
    c = voxel_center_coords(resolution, extent, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(c, c, c, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def voxel_center_grid_np(
    resolution: int, extent: float = 1.0, dtype=np.float32
) -> np.ndarray:
    """Numpy twin of :func:`voxel_center_grid`."""
    c = voxel_center_coords_np(resolution, extent, dtype=dtype)
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(dtype)


def subvoxel_offsets(
    k: int,
    resolution: int,
    extent: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(k**3, 3)`` supersampling offsets tiling one voxel by centres.

    Each voxel is split into ``k**3`` sub-voxels and sampled at *their* centres,
    so the same centre convention holds recursively and ``k=1`` reduces to a
    single zero offset.  Add to rows of :func:`voxel_center_grid`.
    """
    k = int(k)
    if k <= 1:
        return torch.zeros(1, 3, device=device, dtype=dtype)
    pitch = voxel_pitch(resolution, extent)
    sub = torch.linspace(-0.5 + 0.5 / k, 0.5 - 0.5 / k, k, device=device, dtype=dtype)
    ox, oy, oz = torch.meshgrid(sub, sub, sub, indexing="ij")
    return torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3) * pitch
