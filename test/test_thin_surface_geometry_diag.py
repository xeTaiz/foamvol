"""Focused CPU unit test for the geometry-health thin-surface diagnostics added
to ``CTScene.thin_surface_diagnostics`` (and its helper
``_normal_neighbor_coherence`` / module-level ``quaternion_to_normals``).

These diagnostics were added to support the active LC64 oriented-height run
diagnosis.  They are read-only numeric summaries and must NOT alter rendering,
initialization, or optimization.  This test pins their numeric behaviour on a
fully controlled CPU scene so a future refactor that changes the math (or
accidentally wires a diagnostic into a loss/optimizer) is caught here.

Covered diagnostics:
  (a) per-group pre-step gradient norms   -> grad_norm_<group>
  (b) quaternion normal neighbour coherence via CSR adjacency
        -> quat_normal_coherence_sq, quat_normal_flip_frac
  (c) height mean/std and uniform-vs-curved measure
        -> height_mean, height_std, height_curvedness
  (d) height extent (texel_heights are dimensionless; forward applies r*h_k):
        (a dimensionless) height_l1_norm_mean, height_l1_norm_p95  (h_l1/p95)
        (b world units) height_extent_mean, height_extent_p95       (r*h_l1)

The test is CPU-only and uses a radfoam stub (mirrors the other thin-surface
test files) so it runs without a CUDA device or a compiled extension.

Run:  micromamba run -n radfoam python test/test_thin_surface_geometry_diag.py
"""

import math
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# CPU radfoam stub (mirrors test_thin_surface_diag_logging.py).
# ---------------------------------------------------------------------------
def _install_radfoam_stub():
    if "radfoam" in sys.modules:
        return
    mod = types.ModuleType("radfoam")
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda *a, **k: (None, None)
    mod.nn = lambda *a, **k: None
    mod.TriangulationFailedError = type(
        "TriangulationFailedError", (Exception,), {})
    mod.Triangulation = None
    mod.BatchFetcher = lambda *a, **k: None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()

from radfoam_model.scene import (  # noqa: E402
    CTScene,
    quaternion_to_normals,
)

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Scene / args helpers (kept self-contained; mirror diag-logging fixture).
# ---------------------------------------------------------------------------
def _args():
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-2
    a.freeze_points = 9500
    a.thin_surface_start = 0
    a.thin_surface_K = 4
    a.thin_surface_delta_weight = 1e-3
    a.thin_surface_height_weight = 5e-4
    a.thin_surface_gate_tau = 0.01
    a.thin_surface_lr_scale = 1.0
    a.thin_surface_delta_lr_scale = 1.0
    a.thin_surface_quat_lr_scale = 1.0
    a.thin_surface_sites_lr_scale = 1.0
    a.thin_surface_heights_lr_scale = 1.0
    a.thin_surface_delta_clip = 2.0
    a.thin_surface_grad_clip = 1.0
    a.thin_surface_relative_delta = False
    a.thin_surface_delta_max_frac = 0.5
    return a


def _full_graph_adjacency(n):
    """CSR adjacency where every cell neighbours every other cell (no self)."""
    adj = []
    offsets = [0]
    for i in range(n):
        adj.extend(j for j in range(n) if j != i)
        offsets.append(len(adj))
    return (
        torch.tensor(adj, dtype=torch.int32).to(torch.uint32),
        torch.tensor(offsets, dtype=torch.int32).to(torch.uint32),
    )


def _make_scene(n_points=6, device="cpu"):
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device(device)
    scene.num_init_points = n_points
    scene.num_final_points = n_points
    scene._thin_surface_active = False
    scene._thin_surface_K = 4
    scene._thin_surface_gate_tau = 0.01
    scene.thin_surface_scheduler_args = None

    pts = (torch.rand(n_points, 3) - 0.5)
    scene.primal_points = nn.Parameter(pts)
    scene.density = nn.Parameter(
        torch.linspace(0.2, 2.0, n_points).unsqueeze(-1))

    adj, off = _full_graph_adjacency(n_points)
    scene.point_adjacency = adj
    scene.point_adjacency_offsets = off
    scene._cached_cell_radius = torch.ones(n_points)
    return scene


def _activate(scene):
    args = _args()
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)
    return scene


# ---------------------------------------------------------------------------
# quaternion_to_normals sanity (module-level helper).
# ---------------------------------------------------------------------------
def test_quaternion_to_normals_reference_axes():
    print("\n--- quaternion_to_normals: reference rotations ---")
    # Identity -> +x
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    n = quaternion_to_normals(q_id)
    assert torch.allclose(n, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6), \
        f"identity -> +x, got {n}"
    # 90 deg about +z takes +x -> +y: q = (cos45, 0, 0, sin45)
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    q_z = torch.tensor([[c, 0.0, 0.0, s]])
    n = quaternion_to_normals(q_z)
    assert torch.allclose(n, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6), \
        f"rot_z(90) +x->+y, got {n}"
    # 180 deg about +y takes +x -> -x: q = (0, 0, 1, 0)
    q_flip = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    n = quaternion_to_normals(q_flip)
    assert torch.allclose(n, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-6), \
        f"rot_y(180) +x->-x, got {n}"
    # Output is unit length.
    q = torch.randn(5, 4)
    n = quaternion_to_normals(q)
    norms = n.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5), \
        f"unit normals, got {norms}"
    print("  [PASS] identity/rot_z/rot_y/unit-length all correct")


# ---------------------------------------------------------------------------
# (b) coherence: aligned normals -> coh_sq == 1, flip_frac == 0.
# ---------------------------------------------------------------------------
def test_coherence_aligned_normals():
    print("\n--- (b) coherence: aligned normals ---")
    n = 6
    scene = _activate(_make_scene(n_points=n))
    # initialize_thin_surface sets identity quaternions -> all normals +x.
    d = scene.thin_surface_diagnostics()
    assert d is not None
    assert math.isclose(d["quat_normal_coherence_sq"], 1.0, abs_tol=1e-5), \
        f"aligned -> coh_sq==1, got {d['quat_normal_coherence_sq']}"
    assert math.isclose(d["quat_normal_flip_frac"], 0.0, abs_tol=1e-6), \
        f"aligned -> flip_frac==0, got {d['quat_normal_flip_frac']}"
    print("  [PASS] identity quats: coh_sq=1.000, flip_frac=0.000")


# ---------------------------------------------------------------------------
# (b) coherence: mixed (+x / +y) -> coh_sq in (0,1), flip_frac == 0;
#     one flipped (-x) cell -> flip_frac > 0 (coh_sq stays sign-insensitive).
# ---------------------------------------------------------------------------
def test_coherence_mixed_and_flipped():
    print("\n--- (b) coherence: mixed + flipped ---")
    n = 8
    scene = _activate(_make_scene(n_points=n))
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    q = torch.zeros(n, 4)
    q[: n // 2, 0] = 1.0                       # +x
    q[n // 2:, 0] = c                          # +y (rot z 90)
    q[n // 2:, 3] = s
    scene.quaternions.data.copy_(q)
    d = scene.thin_surface_diagnostics()
    coh = d["quat_normal_coherence_sq"]
    flip = d["quat_normal_flip_frac"]
    assert 0.0 < coh < 1.0, f"mixed -> 0<coh_sq<1, got {coh}"
    assert math.isclose(flip, 0.0, abs_tol=1e-6), \
        f"+x/+y -> no flips, got {flip}"
    print(f"  [PASS] +x/+y mix: coh_sq={coh:.4f}, flip_frac=0.000")

    # Now flip one cell to -x; within-group edges flip sign (squared unchanged)
    # so coh_sq is sign-insensitive and UNCHANGED, while flip_frac rises.
    q2 = q.clone()
    q2[0, :] = 0.0
    q2[0, 2] = 1.0                              # 180 deg about +y -> -x
    scene.quaternions.data.copy_(q2)
    d = scene.thin_surface_diagnostics()
    assert d["quat_normal_flip_frac"] > 0.0, \
        f"flipped cell -> flip_frac>0, got {d['quat_normal_flip_frac']}"
    assert math.isclose(d["quat_normal_coherence_sq"], coh, abs_tol=1e-5), \
        f"sign-insensitive coh_sq unchanged ({coh}), " \
        f"got {d['quat_normal_coherence_sq']}"
    print(f"  [PASS] one flipped cell: flip_frac="
          f"{d['quat_normal_flip_frac']:.4f}, coh_sq unchanged "
          f"(sign-insensitive)")


# ---------------------------------------------------------------------------
# (c) heights: uniform translation -> curvedness == 0; spread -> curvedness>0.
# ---------------------------------------------------------------------------
def test_height_uniform_vs_curved():
    print("\n--- (c) height uniform vs curved ---")
    n = 5
    scene = _activate(_make_scene(n_points=n))
    K = scene.texel_heights.shape[1]

    # Uniform: every texel in every cell shares the same height -> no curvature.
    scene.texel_heights.data.fill_(0.5)
    d = scene.thin_surface_diagnostics()
    assert math.isclose(d["height_mean"], 0.5, abs_tol=1e-6), \
        f"uniform height_mean=0.5, got {d['height_mean']}"
    assert math.isclose(d["height_std"], 0.0, abs_tol=1e-6), \
        f"uniform height_std=0, got {d['height_std']}"
    assert math.isclose(d["height_curvedness"], 0.0, abs_tol=1e-6), \
        f"uniform -> curvedness=0, got {d['height_curvedness']}"
    print("  [PASS] uniform heights: mean=0.500, std=0.000, curvedness=0.000")

    # Curved: per-cell heights vary across texels -> curvedness > 0, std > 0.
    ramp = torch.linspace(0.1, 0.4, K).unsqueeze(0).expand(n, K).clone()
    scene.texel_heights.data.copy_(ramp)
    d = scene.thin_surface_diagnostics()
    assert d["height_std"] > 0.0, f"curved height_std>0, got {d['height_std']}"
    assert d["height_curvedness"] > 0.0, \
        f"curved -> curvedness>0, got {d['height_curvedness']}"
    assert d["height_curvedness"] < 1.0 + 1e-6, \
        f"curvedness capped <1, got {d['height_curvedness']}"
    print(f"  [PASS] ramped heights: std={d['height_std']:.4f}, "
          f"curvedness={d['height_curvedness']:.4f}")


# ---------------------------------------------------------------------------
# (d) height/radius ratio scales with height and inverse with radius.
# ---------------------------------------------------------------------------
def test_height_extent_and_normalized():
    print("\n--- (d) height extent (world r*h_l1) + dimensionless normalized ---")
    n = 8
    scene = _activate(_make_scene(n_points=n))
    K = scene.texel_heights.shape[1]
    h_l1_lo = K * 0.1                       # half the cells at h=0.1
    h_l1_hi = K * 0.2                       # half at h=0.2
    h = torch.zeros(n, K)
    h[: n // 2] = 0.1
    h[n // 2:] = 0.2
    scene.texel_heights.data.copy_(h)
    scene._cached_cell_radius = torch.ones(n)
    d = scene.thin_surface_diagnostics()

    # (a) dimensionless normalized height L1 = h_l1 / p95(h_l1).
    # p95(h_l1)=h_l1_hi; lo cells -> 0.5, hi cells -> 1.0; mean -> 0.75.
    assert math.isclose(d["height_l1_norm_p95"], 1.0, abs_tol=1e-6), \
        f"normalized p95 cell ==1, got {d['height_l1_norm_p95']}"
    assert math.isclose(d["height_l1_norm_mean"], 0.75, abs_tol=1e-6), \
        f"normalized mean = (0.5+1.0)/2 = 0.75, got {d['height_l1_norm_mean']}"
    print(f"  [PASS] normalized: mean={d['height_l1_norm_mean']:.4f}, "
          f"p95={d['height_l1_norm_p95']:.4f}")

    # (b) world height extent = cell_radius * h_l1 (r=1 here).
    assert math.isclose(d["height_extent_mean"], (h_l1_lo + h_l1_hi) / 2,
                        abs_tol=1e-6), \
        f"extent mean = r*(h_l1 mean) = {(h_l1_lo+h_l1_hi)/2}, " \
        f"got {d['height_extent_mean']}"
    assert math.isclose(d["height_extent_p95"], h_l1_hi, abs_tol=1e-6), \
        f"extent p95 = r*h_l1_hi = {h_l1_hi}, got {d['height_extent_p95']}"
    print(f"  [PASS] extent r=1: mean={d['height_extent_mean']:.4f}, "
          f"p95={d['height_extent_p95']:.4f}")

    # Radius scaling: world extent scales linearly with r; dimensionless
    # normalized measure is INVARIANT (it must not confound scene scale / cell
    # count -- the defect the old h_l1/r ratio had).
    scene._cached_cell_radius = torch.full((n,), 0.5)
    d = scene.thin_surface_diagnostics()
    assert math.isclose(d["height_l1_norm_mean"], 0.75, abs_tol=1e-6), \
        f"normalized invariant to radius, got {d['height_l1_norm_mean']}"
    assert math.isclose(d["height_l1_norm_p95"], 1.0, abs_tol=1e-6), \
        f"normalized p95 invariant, got {d['height_l1_norm_p95']}"
    assert math.isclose(d["height_extent_mean"], 0.5 * (h_l1_lo + h_l1_hi) / 2,
                        abs_tol=1e-6), \
        f"extent halves with r, got {d['height_extent_mean']}"
    assert math.isclose(d["height_extent_p95"], 0.5 * h_l1_hi, abs_tol=1e-6), \
        f"extent p95 halves with r, got {d['height_extent_p95']}"
    print(f"  [PASS] r=0.5: normalized unchanged; extent halves "
          f"(mean={d['height_extent_mean']:.4f}, p95={d['height_extent_p95']:.4f})")

    # Uniform heights + zero heights edge case: p95 normaliser must not blow up
    # and a degenerate (all-zero) field reports NaN rather than /0.
    scene.texel_heights.data.fill_(0.0)
    d = scene.thin_surface_diagnostics()
    assert math.isnan(d["height_l1_norm_mean"]), \
        f"all-zero heights -> NaN normalized, got {d['height_l1_norm_mean']}"
    assert math.isclose(d["height_extent_mean"], 0.0, abs_tol=1e-6), \
        f"all-zero heights -> extent 0, got {d['height_extent_mean']}"
    print(f"  [PASS] all-zero heights: normalized=NaN, extent=0.0000")


# ---------------------------------------------------------------------------
# (a) per-group gradient norms: only a populated .grad is reported; absent
#     grads report NaN (the "if feasible" caveat).
# ---------------------------------------------------------------------------
def test_grad_norms():
    print("\n--- (a) per-group gradient norms ---")
    n = 4
    scene = _activate(_make_scene(n_points=n))
    # No backward yet -> all grad norms NaN.
    d = scene.thin_surface_diagnostics()
    for g in ("density_delta", "quaternions",
              "texel_sites_2d", "texel_heights"):
        key = f"grad_norm_{g}"
        assert key in d, f"missing {key}"
        assert math.isnan(d[key]), \
            f"no backward -> {key} NaN, got {d[key]}"
    print("  [PASS] no backward: all four grad_norm_* == NaN")

    # Populate density_delta.grad with a known constant -> norm == |c|*sqrt(N).
    c = 3.0
    scene.density_delta.grad = torch.full_like(scene.density_delta, c)
    expected = abs(c) * math.sqrt(scene.density_delta.numel())
    d = scene.thin_surface_diagnostics()
    assert math.isclose(d["grad_norm_density_delta"], expected, abs_tol=1e-5), \
        f"grad norm = |c|*sqrt(numel) = {expected}, " \
        f"got {d['grad_norm_density_delta']}"
    # The other three remain NaN.
    for g in ("quaternions", "texel_sites_2d", "texel_heights"):
        assert math.isnan(d[f"grad_norm_{g}"]), \
            f"{g} still NaN, got {d[f'grad_norm_{g}']}"
    print(f"  [PASS] density_delta.grad set: grad_norm={expected:.4f}, "
          f"others NaN")


# ---------------------------------------------------------------------------
# All new keys are numeric (TensorBoard-safe) and the diagnostic dict round
# trips through the real train.py `_log_diag_kv` helper without raising.
# ---------------------------------------------------------------------------
def test_all_new_keys_numeric_and_loggable():
    print("\n--- all new keys numeric + loggable end-to-end ---")
    import train as _train_mod
    from torch.utils.tensorboard import SummaryWriter
    import tempfile
    scene = _activate(_make_scene(n_points=5))
    d = scene.thin_surface_diagnostics()
    new_keys = [
        "grad_norm_density_delta", "grad_norm_quaternions",
        "grad_norm_texel_sites_2d", "grad_norm_texel_heights",
        "quat_normal_coherence_sq", "quat_normal_flip_frac",
        "height_mean", "height_std", "height_curvedness",
        "height_l1_norm_mean", "height_l1_norm_p95",
        "height_extent_mean", "height_extent_p95",
    ]
    for k in new_keys:
        assert k in d, f"missing new key {k}"
        assert isinstance(d[k], (int, float)), \
            f"{k} must be numeric for add_scalar, got {type(d[k]).__name__}"
    with tempfile.TemporaryDirectory() as td:
        w = SummaryWriter(td, purge_step=0)
        try:
            for k, v in d.items():
                _train_mod._log_diag_kv(w, k, v, step=0)
        finally:
            w.close()
    print(f"  [PASS] {len(new_keys)} new keys numeric; full dict "
          f"loggable via _log_diag_kv")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


def _run(fn, name):
    global _any_failed
    try:
        fn()
    except AssertionError as e:
        check(False, f"{name}: {e}")
    except Exception as e:
        check(False, f"{name}: unexpected {type(e).__name__}: {e}")


def main():
    print("=" * 60)
    print("Thin-Surface Geometry-Health Diagnostics (CPU unit test)")
    print("=" * 60)
    _run(test_quaternion_to_normals_reference_axes,
         "quaternion_to_normals_reference_axes")
    _run(test_coherence_aligned_normals, "coherence_aligned_normals")
    _run(test_coherence_mixed_and_flipped, "coherence_mixed_and_flipped")
    _run(test_height_uniform_vs_curved, "height_uniform_vs_curved")
    _run(test_height_extent_and_normalized, "height_extent_and_normalized")
    _run(test_grad_norms, "grad_norms")
    _run(test_all_new_keys_numeric_and_loggable,
         "all_new_keys_numeric_and_loggable")
    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED.")
        sys.exit(1)
    print("SUMMARY: ALL GEOMETRY-HEALTH DIAGNOSTIC TESTS PASSED.")


if __name__ == "__main__":
    main()