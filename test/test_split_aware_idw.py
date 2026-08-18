"""Contract tests for split-aware IDW value resolution.

Flat IDW gathers one density per neighbor cell. That is wrong for split cells:
a split cell holds two densities and which one applies depends on where the
query point sits relative to *that cell's* internal surface. ``vis_foam``
supplies ``idw_query`` with a ``value_fn`` that resolves every
(query, neighbor) pair against the neighbor's own surface.

These tests pin the two properties that make that safe to enable by default:
  1. per-neighbor resolution - column k really is neighbor k's own surface
     evaluated at the query point, not a flat per-cell density;
  2. exact reduction - with no split (delta = 0) it returns the flat gather
     bit-for-bit, so scalar-cell runs cannot change.

CPU-only: the value_fn and split_cell_query are plain torch. The surrounding
idw_query plumbing needs the CUDA extension and is covered separately.
"""

import math

import torch
import torch.nn.functional as F

from vis_foam import (
    _make_split_value_fn,
    _split_eval,
    _thin_surface_query_config,
)


def _z_normal_quaternion(dtype=torch.float32):
    # -90 degrees about y maps local +x to world +z.
    return torch.tensor([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0], dtype=dtype)


def _split_field(n_cells=4, k_texels=4, delta=0.5, dtype=torch.float64):
    """Minimal field dict that passes _thin_surface_query_config validation.

    Cells sit along x; each carries a flat surface whose normal is world +z, so
    a query's side is decided by its z coordinate relative to the cell center.
    """
    points = torch.zeros(n_cells, 3, dtype=dtype)
    points[:, 0] = torch.linspace(-0.6, 0.6, n_cells, dtype=dtype)
    return {
        "points": points,
        "density_flat": torch.full((n_cells,), 1.0, dtype=dtype),
        "density_delta": torch.full((n_cells,), delta, dtype=dtype),
        "quaternions": _z_normal_quaternion(dtype).repeat(n_cells, 1),
        "texel_sites_2d": torch.zeros(n_cells, k_texels, 2, dtype=dtype),
        "texel_heights": torch.zeros(n_cells, k_texels, dtype=dtype),
        "cell_radius": torch.full((n_cells,), 0.5, dtype=dtype),
        "activation_scale": 1.0,
        "thin_surface_active": True,
        "thin_surface_density_mode": "relative",
        "thin_surface_relative_delta": True,
        "thin_surface_delta_max_frac": 0.5,
    }


def test_value_fn_resolves_each_neighbor_against_its_own_surface():
    field = _split_field()
    thin_config = _thin_surface_query_config(field)
    assert thin_config is not None

    value_fn = _make_split_value_fn(field, thin_config)

    # Queries straddling z=0 so both sides of the split are exercised.
    query = torch.tensor([
        [0.0, 0.0, 0.30],
        [0.0, 0.0, -0.30],
        [0.2, 0.0, 0.05],
        [-0.2, 0.0, -0.05],
    ], dtype=field["points"].dtype)
    # Slot 0 must be the containing cell, mirroring idw_query's pad_idx layout.
    pad_idx = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ])
    valid = torch.ones_like(pad_idx, dtype=torch.bool)

    vals, ref_val = value_fn(query, pad_idx, valid)
    assert vals.shape == pad_idx.shape

    # Every column must equal that column's cell evaluated at the query point.
    for k in range(pad_idx.shape[1]):
        expected = _split_eval(field, thin_config, query, pad_idx[:, k])
        assert torch.allclose(vals[:, k], expected)

    # The bilateral reference is the containing cell's own split value.
    assert torch.allclose(ref_val, vals[:, 0])

    # A real split must actually separate the two sides, otherwise this test
    # would pass trivially against a flat gather.
    assert not torch.allclose(vals[0, 0], vals[1, 1])


def test_value_fn_reduces_to_flat_gather_without_split():
    """delta = 0 => both sides equal the base density => flat IDW exactly."""
    field = _split_field(delta=0.0)
    thin_config = _thin_surface_query_config(field)
    assert thin_config is not None

    query = torch.tensor([
        [0.0, 0.0, 0.4],
        [0.1, 0.0, -0.4],
        [-0.3, 0.0, 0.02],
    ], dtype=field["points"].dtype)
    pad_idx = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0]])
    valid = torch.ones_like(pad_idx, dtype=torch.bool)

    vals, _ = _make_split_value_fn(field, thin_config)(query, pad_idx, valid)

    activated = F.softplus(field["density_flat"], beta=10)
    assert torch.allclose(vals, activated[pad_idx])


def test_value_fn_chunking_is_transparent():
    """Row chunking bounds memory; it must not change the result."""
    field = _split_field(n_cells=6)
    thin_config = _thin_surface_query_config(field)

    torch.manual_seed(0)
    query = (torch.rand(64, 3, dtype=field["points"].dtype) * 2 - 1) * 0.5
    pad_idx = torch.randint(0, 6, (64, 5))
    valid = torch.ones_like(pad_idx, dtype=torch.bool)

    whole, _ = _make_split_value_fn(
        field, thin_config, max_pairs=10_000)(query, pad_idx, valid)
    # max_pairs=5 forces one row per chunk (5 neighbor slots per row).
    chunked, _ = _make_split_value_fn(
        field, thin_config, max_pairs=5)(query, pad_idx, valid)
    assert torch.equal(whole, chunked)


def test_scalar_field_yields_no_split_config():
    """Scalar cells must fall back, leaving idw_query's flat path untouched."""
    field = _split_field()
    field["thin_surface_active"] = False
    assert _thin_surface_query_config(field) is None
