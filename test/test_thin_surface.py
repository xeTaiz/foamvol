"""CPU-runnable tests for the thin-surface split-cell feature.

These do NOT require a CUDA device:
  - test_split_cell_query_side_selection / height / inertness: pure-torch
    checks of split_cell_query against hand-built single-cell scenes.
  - test_checkpoint_roundtrip_thin_surface: save_pt -> load_pt preserves the
    four tensors + flags, with radfoam.build_aabb_tree stubbed out.
  - test_K_guard: assert_supported_thin_K rejects K not in {4} and K>8.

Run:  micromamba run -n radfoam python -m pytest test/test_thin_surface.py -q
(or:  micromamba run -n radfoam python test/test_thin_surface.py)
"""

import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make repo root importable when running from test/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Stub `radfoam` so importing scene/split_voxelize does not require a GPU.
# (scene.py calls radfoam.build_aabb_tree only inside load_pt; split_voxelize
#  imports radfoam at top level but we never call its CUDA functions here.)
# ---------------------------------------------------------------------------

def _install_radfoam_stub():
    if "radfoam" in sys.modules:
        return
    mod = types.ModuleType("radfoam")
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda *a, **k: (None, None)
    mod.nn = lambda *a, **k: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()

from radfoam_model.scene import (  # noqa: E402
    CTScene,
    assert_supported_thin_K,
)
from split_voxelize import split_cell_query, quat_to_frame  # noqa: E402


# ---------------------------------------------------------------------------
# split_cell_query
# ---------------------------------------------------------------------------

def _single_cell_scene(delta=0.5, mu_raw=2.0, K=4, normal=(1, 0, 0),
                       heights=None, cell_radius=1.0, activation_scale=1.0):
    """One cell at origin with a flat (heights=0) or set height field."""
    N = 1
    points = torch.zeros(N, 3)
    density = torch.tensor([mu_raw])
    density_delta = torch.tensor([[delta]])
    # quaternion rotating [1,0,0] onto `normal`
    ref = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor(normal, dtype=torch.float32)
    v = v / v.norm().clamp_min(1e-12)
    cross = torch.cross(ref, v, dim=-1)
    dot = (ref * v).sum()
    w = torch.sqrt(((dot + 1.0) * 0.5).clamp_min(0.0))
    xyz = cross / (2.0 * w.clamp_min(1e-12))
    q = torch.cat([w.unsqueeze(0), xyz]).unsqueeze(0)  # (1,4)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    angles = torch.linspace(0, 2 * 3.14159265, K + 1)[:-1]
    sites = torch.stack([torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    sites = sites.unsqueeze(0).expand(N, -1, -1).clone()
    if heights is None:
        heights = torch.zeros(N, K)
    cr = torch.tensor([cell_radius])
    return dict(points=points, density=density, density_delta=density_delta,
                quaternions=q, texel_sites_2d=sites, texel_heights=heights,
                cell_radius=cr, activation_scale=activation_scale)


def test_split_cell_query_side_selection():
    s = _single_cell_scene(delta=0.5, mu_raw=2.0)
    # mu_bar = softplus(2) ~ 2.0; mu_p ~ 2.5, mu_n ~ 1.5
    nn_idx = torch.zeros(3, dtype=torch.long)
    query = torch.tensor([[0.5, 0.0, 0.0],   # +n side -> mu_p
                          [-0.5, 0.0, 0.0],  # -n side -> mu_n
                          [0.0, 0.0, 0.0]])  # on surface (heights=0) -> blend
    val, side, sd = split_cell_query(
        query, s["points"], nn_idx, s["density"], s["density_delta"],
        s["quaternions"], s["texel_sites_2d"], s["texel_heights"],
        s["cell_radius"], activation_scale=s["activation_scale"], blend_eps=1e-6)
    mu_bar = F.softplus(torch.tensor(2.0), beta=10.0)
    assert torch.isclose(val[0], mu_bar + 0.5, atol=1e-4), val[0]
    assert torch.isclose(val[1], mu_bar - 0.5, atol=1e-4), val[1]
    assert side[0] > 0 and side[1] < 0, side
    assert abs(sd[0].item() - 0.5) < 1e-4 and abs(sd[1].item() + 0.5) < 1e-4, sd
    print("OK test_split_cell_query_side_selection")


def test_split_cell_query_inertness():
    """delta=0 and heights=0 -> mu_plus == mu_minus == softplus(density)."""
    s = _single_cell_scene(delta=0.0, mu_raw=1.5)
    nn_idx = torch.zeros(4, dtype=torch.long)
    query = torch.tensor([[0.7, 0.1, -0.2], [-0.3, 0.4, 0.0],
                          [0.0, 0.0, 0.0], [0.2, -0.5, 0.3]])
    val, side, sd = split_cell_query(
        query, s["points"], nn_idx, s["density"], s["density_delta"],
        s["quaternions"], s["texel_sites_2d"], s["texel_heights"],
        s["cell_radius"], blend_eps=1e-6)
    mu_bar = F.softplus(torch.tensor(1.5), beta=10.0)
    assert torch.allclose(val, mu_bar.expand_as(val), atol=1e-5), val
    print("OK test_split_cell_query_inertness")


def test_split_cell_query_height_field_shifts_boundary():
    """Uniform positive height h shifts the surface by +h along n for all x."""
    K = 4
    h = 0.3
    s = _single_cell_scene(delta=0.5, mu_raw=2.0, K=K,
                           heights=torch.full((1, K), h))
    # The soft-Voronoi field with all heights equal to h and sites on a ring of
    # radius 0.4*r=0.4 evaluates to ~h at the center (sites are far enough that
    # weights are near-uniform and small, so h_eval ~ h). The boundary should
    # be near s = +0.3.
    nn_idx = torch.zeros(2, dtype=torch.long)
    query = torch.tensor([[0.31, 0.0, 0.0],   # just past +h -> mu_p
                          [0.29, 0.0, 0.0]])  # just before +h -> mu_n
    val, side, sd = split_cell_query(
        query, s["points"], nn_idx, s["density"], s["density_delta"],
        s["quaternions"], s["texel_sites_2d"], s["texel_heights"],
        s["cell_radius"], blend_eps=1e-4)
    # signed_dist should be ~ (x - h_eval); h_eval ~ h (within the softness).
    assert sd[1] < 0 < sd[0], (sd, "boundary did not shift by ~h")
    assert side[0] > 0 and side[1] < 0, side
    print(f"OK test_split_cell_query_height_field_shifts_boundary (sd={sd.tolist()})")


def test_split_cell_query_hard_side_no_blend():
    """blend_eps=0 (default) -> hard side, no division-by-zero, no smoothing."""
    s = _single_cell_scene(delta=0.5, mu_raw=2.0)
    nn_idx = torch.zeros(3, dtype=torch.long)
    query = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    val, side, sd = split_cell_query(
        query, s["points"], nn_idx, s["density"], s["density_delta"],
        s["quaternions"], s["texel_sites_2d"], s["texel_heights"],
        s["cell_radius"], blend_eps=0.0)
    mu_bar = F.softplus(torch.tensor(2.0), beta=10.0)
    # Hard side: +n -> mu_plus exactly, -n -> mu_minus exactly (no blend).
    assert torch.isclose(val[0], mu_bar + 0.5, atol=1e-6), val[0]
    assert torch.isclose(val[1], mu_bar - 0.5, atol=1e-6), val[1]
    assert side[0] > 0 and side[1] < 0, side
    # On-surface point (s=0): hard selection picks mu_minus (s>0 is False).
    assert torch.isclose(val[2], mu_bar - 0.5, atol=1e-6), val[2]
    print("OK test_split_cell_query_hard_side_no_blend")


def test_split_cell_query_matches_kernel_side_convention():
    """+n side must be mu_plus (kernel: dp>0 -> near=mu_n, far=mu_p)."""
    s = _single_cell_scene(delta=0.4, mu_raw=2.0)
    nn_idx = torch.zeros(1, dtype=torch.long)
    query = torch.tensor([[0.6, 0.0, 0.0]])
    val, side, _ = split_cell_query(
        query, s["points"], nn_idx, s["density"], s["density_delta"],
        s["quaternions"], s["texel_sites_2d"], s["texel_heights"],
        s["cell_radius"], blend_eps=1e-6)
    mu_bar = F.softplus(torch.tensor(2.0), beta=10.0)
    assert torch.isclose(val[0], mu_bar + 0.4, atol=1e-4), val
    print("OK test_split_cell_query_matches_kernel_side_convention")


# ---------------------------------------------------------------------------
# K guard
# ---------------------------------------------------------------------------

def test_K_guard():
    assert_supported_thin_K(4)
    try:
        assert_supported_thin_K(8)
    except ValueError as e:
        print(f"OK test_K_guard rejects K=8: {e}")
        return
    raise AssertionError("K=8 should be rejected until gradcheck is extended")


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def _make_fake_scene(N=5, K=4, device="cpu"):
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device(device)
    scene.primal_points = nn.Parameter(torch.randn(N, 3))
    scene.density = nn.Parameter(torch.randn(N, 1))
    scene.point_adjacency = torch.zeros(2 * N, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = torch.arange(0, 2 * (N + 1), 2).to(torch.uint32)
    scene.density_delta = nn.Parameter(torch.randn(N, 1))
    scene.quaternions = nn.Parameter(torch.nn.functional.normalize(
        torch.randn(N, 4), dim=-1))
    angles = torch.linspace(0, 2 * 3.14159265, K + 1)[:-1]
    sites = torch.stack([torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    scene.texel_sites_2d = nn.Parameter(sites.unsqueeze(0).expand(N, -1, -1).clone())
    scene.texel_heights = nn.Parameter(torch.randn(N, K))
    scene._thin_surface_active = True
    scene._thin_K = K
    scene._thin_surface_start = 6000
    scene._thin_surface_scheduler_cfg = {"lr_init": 5e-3, "lr_final": 5e-4,
                                          "max_steps": 4000}
    return scene


def test_checkpoint_roundtrip_thin_surface(tmp_path=None):
    import tempfile
    tmp = tmp_path or tempfile.mkdtemp()
    path = os.path.join(tmp, "model.pt")

    scene = _make_fake_scene(N=5, K=4)
    # Remember expected values
    exp = {k: getattr(scene, k).detach().clone() for k in
           ["density_delta", "quaternions", "texel_sites_2d", "texel_heights"]}

    scene.save_pt(path)

    # Load into a fresh scene. load_pt calls radfoam.build_aabb_tree at the end
    # which is stubbed to return None.
    loaded = object.__new__(CTScene)
    nn.Module.__init__(loaded)
    loaded.activation_scale = 1.0
    loaded.device = torch.device("cpu")
    loaded.load_pt(path)

    assert loaded._thin_surface_active is True, "thin_surface_active flag lost"
    assert loaded._thin_K == 4, "K lost"
    assert loaded._thin_surface_start == 6000, "start lost"
    assert loaded._thin_surface_scheduler_cfg is not None, "scheduler cfg lost"
    for k, v in exp.items():
        t = getattr(loaded, k)
        assert t is not None, f"{k} not restored"
        assert torch.equal(t.detach(), v), f"{k} mismatch after round-trip"
    # forward() keys surface mode off _thin_surface_active; get_trace_data must
    # surface the tensors.
    td = loaded.get_trace_data()
    dd, q, ts, th = td[9], td[10], td[11], td[12]
    assert torch.equal(dd, exp["density_delta"]) and torch.equal(q, exp["quaternions"]) \
        and torch.equal(ts, exp["texel_sites_2d"]) and torch.equal(th, exp["texel_heights"])
    print("OK test_checkpoint_roundtrip_thin_surface")


def test_checkpoint_baseline_unchanged():
    """A baseline checkpoint (no thin-surface) must still load without error."""
    import tempfile
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "model.pt")
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device("cpu")
    N = 3
    scene.primal_points = nn.Parameter(torch.randn(N, 3))
    scene.density = nn.Parameter(torch.randn(N, 1))
    scene.point_adjacency = torch.zeros(2 * N, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = torch.arange(0, 2 * (N + 1), 2).to(torch.uint32)
    scene.save_pt(path)

    loaded = object.__new__(CTScene)
    nn.Module.__init__(loaded)
    loaded.activation_scale = 1.0
    loaded.device = torch.device("cpu")
    loaded.load_pt(path)
    assert not getattr(loaded, "_thin_surface_active", False), \
        "baseline checkpoint should not activate thin surface"
    assert loaded.get_trace_data()[9] is None, "baseline should expose no density_delta"
    print("OK test_checkpoint_baseline_unchanged")


def main():
    test_split_cell_query_side_selection()
    test_split_cell_query_inertness()
    test_split_cell_query_height_field_shifts_boundary()
    test_split_cell_query_hard_side_no_blend()
    test_split_cell_query_matches_kernel_side_convention()
    test_K_guard()
    test_checkpoint_roundtrip_thin_surface()
    test_checkpoint_baseline_unchanged()
    print("\nAll thin-surface tests passed.")


if __name__ == "__main__":
    main()
