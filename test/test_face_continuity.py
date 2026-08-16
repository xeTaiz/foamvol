import math
from types import SimpleNamespace

import torch

from radfoam_model.face_continuity import (
    VoronoiFaceCache,
    build_voronoi_face_cache,
    evaluate_surface_field,
    face_continuity_loss,
)


def _z_normal_quaternion(dtype=torch.float32):
    # -90 degrees about y maps local +x to world +z.
    return torch.tensor([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0], dtype=dtype)


def test_gpu_style_cache_recovers_bounded_dual_face_on_cpu():
    points = torch.tensor([
        [-0.3, 0.0, 0.0], [0.3, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -0.5, 0.8660254],
        [0.0, -0.5, -0.8660254],
    ])
    # The central edge (0,1) has a closed three-tetrahedron fan; its dual face
    # is bounded. Every other edge touches the convex hull.
    tets = torch.tensor([
        [0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2],
    ])
    cache = build_voronoi_face_cache(
        points, tets, torch.arange(5), num_samples=12, domain_extent=10.0)
    assert cache.num_faces == 1
    assert torch.equal(cache.pairs, torch.tensor([[0, 1]]))
    assert cache.samples.shape == (1, 12, 3)
    assert cache.area.item() > 0
    # The dual face lies on the x=0 bisector.
    assert cache.samples[..., 0].abs().max().item() < 1e-5


def test_cache_permutation_maps_internal_tets_to_external_rows_once():
    points = torch.tensor([
        [-0.3, 0.0, 0.0], [0.3, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -0.5, 0.8660254],
        [0.0, -0.5, -0.8660254],
    ])
    external_tets = torch.tensor([
        [0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2],
    ])
    permutation = torch.tensor([2, 0, 4, 1, 3])  # internal -> external
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(5)
    internal_tets = inverse[external_tets]
    cache = build_voronoi_face_cache(
        points, internal_tets, permutation, num_samples=8, domain_extent=10.0)
    assert torch.equal(cache.pairs, torch.tensor([[0, 1]]))


def test_surface_field_matches_flat_plane_and_has_height_gradient():
    dtype = torch.float64
    points = torch.tensor([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=dtype)
    quaternions = _z_normal_quaternion(dtype).repeat(2, 1).requires_grad_()
    sites = torch.tensor([
        [[0.4, 0.0], [0.0, 0.4], [-0.4, 0.0], [0.0, -0.4]],
    ], dtype=dtype).repeat(2, 1, 1)
    heights = torch.zeros(2, 4, dtype=dtype, requires_grad=True)
    radius = torch.ones(2, dtype=dtype)
    query = torch.tensor([
        [[0.0, 0.0, -0.3], [0.0, 0.0, 0.25]],
        [[0.0, 0.0, -0.3], [0.0, 0.0, 0.25]],
    ], dtype=dtype)
    signed, normal = evaluate_surface_field(
        torch.tensor([0, 1]), query, points, quaternions, sites, heights, radius)
    assert torch.allclose(signed, query[..., 2], atol=1e-10)
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    assert torch.allclose(normal, expected.expand_as(normal), atol=1e-10)
    signed.square().sum().backward()
    assert torch.isfinite(quaternions.grad).all()
    assert torch.isfinite(heights.grad).all()
    assert heights.grad.abs().sum().item() > 0


def _fake_model(second_height=0.0, second_delta=0.3):
    dtype = torch.float64
    model = SimpleNamespace()
    model.primal_points = torch.tensor(
        [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=dtype)
    model.quaternions = torch.nn.Parameter(
        _z_normal_quaternion(dtype).repeat(2, 1))
    model.texel_sites_2d = torch.nn.Parameter(torch.tensor([
        [[0.4, 0.0], [0.0, 0.4], [-0.4, 0.0], [0.0, -0.4]],
    ], dtype=dtype).repeat(2, 1, 1), requires_grad=False)
    h = torch.zeros(2, 4, dtype=dtype)
    h[1] = second_height
    model.texel_heights = torch.nn.Parameter(h)
    model.density_delta = torch.nn.Parameter(
        torch.tensor([[0.3], [second_delta]], dtype=dtype))
    # beta=10 softplus inverse of 0.5.
    raw_density = math.log(math.expm1(5.0)) / 10.0
    model.density = torch.nn.Parameter(
        torch.full((2, 1), raw_density, dtype=dtype))
    model.activation_scale = 1.0
    model._cached_cell_radius = torch.ones(2, dtype=dtype)
    model._thin_surface_density_mode = "absolute"
    return model


def _manual_face_cache(dtype=torch.float64):
    z = torch.linspace(-0.5, 0.5, 12, dtype=dtype)
    samples = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), dim=-1)
    return VoronoiFaceCache(
        pairs=torch.tensor([[0, 1]]), samples=samples[None],
        vertices=samples, vertex_offsets=torch.tensor([0, 12]),
        area=torch.ones(1, dtype=dtype), scale=torch.ones(1, dtype=dtype),
        build_seconds=0.0, num_input_tets=0, num_finite_tets=0,
        num_faces_before_domain_filter=1, max_vertices=12)


def test_identical_oriented_surface_and_densities_have_zero_loss():
    model = _fake_model()
    loss, diag = face_continuity_loss(
        model, _manual_face_cache(), step=0, batch_size=1,
        density_scale=1.0, crossing_margin_fraction=0.0)
    assert diag["eligible_pairs"] == 1
    assert loss.item() < 1e-10


def test_high_density_orientation_resolves_parameter_sign_flip():
    model = _fake_model(second_delta=-0.3)
    # +90 degrees about y maps local +x to -z. Combined with negative delta,
    # the second cell's high-density-oriented normal/field still points +z.
    model.quaternions.data[1] = torch.tensor(
        [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=torch.float64)
    loss, diag = face_continuity_loss(
        model, _manual_face_cache(), step=0, batch_size=1,
        density_scale=1.0, crossing_margin_fraction=0.0)
    assert diag["eligible_pairs"] == 1
    assert loss.item() < 1e-10


def test_offset_and_density_disagreement_produce_gradients():
    model = _fake_model(second_height=0.15, second_delta=0.2)
    loss, diag = face_continuity_loss(
        model, _manual_face_cache(), step=0, batch_size=1,
        density_scale=1.0, crossing_margin_fraction=0.0,
        normal_weight=0.25, density_weight=0.1)
    assert diag["eligible_pairs"] == 1
    assert diag["zero_loss"] > 0
    assert diag["density_loss"] > 0
    loss.backward()
    assert model.texel_heights.grad[1].abs().sum().item() > 0
    assert model.density_delta.grad.abs().sum().item() > 0
    assert torch.isfinite(model.quaternions.grad).all()
