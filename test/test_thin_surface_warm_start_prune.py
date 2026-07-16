"""Regression test for the CH4 warm-start prune bug.

Bug: CH4_clean deterministically crashed at thin activation (iter 6000) when a
standalone prune had just shrunk the scene.  `initialize_thin_surface` read
`self._last_top_eigvec` (a per-cell (N, 3) tensor cached by the boundary
eigenvector losses) and used it directly inside
`torch.cross(ref, v, dim=-1)`.  After the prune the cache still held N_pre
entries (352977) while `primal_points` had only N_post (352041), so the
broadcasting op crashed with::

    RuntimeError: The size of tensor a (352041) must match the size of
    tensor b (352977) at non-singleton dimension 0

This test simulates that exact mismatch with a tiny CPU-only scene and
verifies:

  Test A (the bug): when `_last_top_eigvec` is set with a stale shape that
    disagrees with the current primal_points count,
    `initialize_thin_surface` must NOT crash, must discard the stale cache,
    must warn, and must fall back to identity quaternions.

  Test B (the proper fix): after a real `prune_points`, the
    `_last_top_eigvec` cache (and the three other per-cell boundary caches)
    must be permuted to match the surviving point set, so the next
    `initialize_thin_surface` warm-start uses the *right* rows.

  Test C (no-warm path): without any `_last_top_eigvec` at all,
    `initialize_thin_surface` must still produce identity quaternions and
    the four registered param tensors with the correct shapes.

  Test D (cache permute is identity when no prune happens): a no-op prune
    keeps the cache row-aligned and warm-start still works.

Run with:  micromamba run -n radfoam python test/test_thin_surface_warm_start_prune.py
"""
import os
import sys
import math
import types
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np

# Repo root on path (match other test files).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Stubs so we can import scene.py on CPU.
# ---------------------------------------------------------------------------
def _install_radfoam_stub():
    if "radfoam" in sys.modules:
        return
    import types
    mod = types.ModuleType("radfoam")
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda pts, adj, off, **kw: (
        torch.zeros(pts.shape[0], dtype=torch.long),
        torch.ones(pts.shape[0], device=pts.device),
    )
    mod.nn = lambda points, tree, query, **kw: torch.zeros(
        query.shape[0], dtype=torch.long, device=query.device)
    mod.BatchFetcher = lambda *a, **k: None
    mod.TriangulationFailedError = type("TriangulationFailedError", (Exception,), {})
    mod.Triangulation = None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()

from radfoam_model.scene import CTScene  # noqa: E402


torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers (modeled on test_thin_surface_lr_scale.py).
# ---------------------------------------------------------------------------
def _args():
    """Namespace with every field declare_optimizer + initialize_thin_surface read."""
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-2
    a.freeze_points = 9500
    a.thin_surface_start = 6000          # mimic CH4 schedule
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
    return a


def _make_minimal_scene(n_points, device="cpu"):
    """CPU-friendly scene with a fake all-pairs adjacency (no triangulation)."""
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device(device)
    scene.num_init_points = n_points
    scene.num_final_points = n_points
    scene._thin_surface_active = False
    scene._thin_K = 4
    scene._thin_surface_gate_tau = 0.01
    scene.thin_surface_scheduler_args = None

    pts = (torch.rand(n_points, 3, device=device) - 0.5) * 1.0
    scene.primal_points = nn.Parameter(pts)
    scene.density = nn.Parameter(0.5 * torch.ones(n_points, 1, device=device))

    # All-pairs fake adjacency (CSR).
    adj = []
    offsets = [0]
    for i in range(n_points):
        nbrs = [j for j in range(n_points) if j != i]
        adj.extend(nbrs)
        offsets.append(len(adj))
    scene.point_adjacency = torch.tensor(adj, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = torch.tensor(offsets, dtype=torch.int32).to(torch.uint32)
    scene._cached_cell_radius = torch.ones(n_points, device=device)
    return scene


THIN_GROUP_NAMES = ("density_delta", "quaternions", "texel_sites_2d", "texel_heights")


_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# ---------------------------------------------------------------------------
# Test A: stale _last_top_eigvec (the CH4 reproducer).
# ---------------------------------------------------------------------------
def test_stale_warm_start_cache_falls_back_safely():
    """Simulate the CH4 race: prune-only just ran and shrank the scene, but
    `_last_top_eigvec` still holds the pre-prune count.  Calling
    `initialize_thin_surface` must NOT crash, must drop the stale cache,
    and must produce identity quaternions (no warm-start)."""
    print("\n--- Test A: stale _last_top_eigvec is discarded, no crash ---")
    n_pre = 16          # pre-prune cell count (CH4 was 352977)
    n_post = 12         # post-prune cell count (CH4 was 352041)
    scene = _make_minimal_scene(n_post, device="cpu")
    args = _args()
    scene.declare_optimizer(args, warmup=0, max_iterations=10000)

    # Inject a stale cache as if a boundary-loss call had populated it before
    # a prune.  We attach it directly so we don't need the CUDA boundary path.
    stale_v = torch.randn(n_pre, 3, device="cpu")
    stale_v = stale_v / stale_v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scene._last_top_eigvec = stale_v
    scene._last_M_trace = torch.rand(n_pre, device="cpu")
    scene._last_M_valid = torch.ones(n_pre, dtype=torch.bool, device="cpu")
    scene._last_normal_lap_residual = torch.rand(n_pre, device="cpu")

    # Capture the warning output so the assertion is robust to printout.
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            scene.initialize_thin_surface(args, K=4)
    except RuntimeError as e:
        check(False,
              f"initialize_thin_surface crashed with stale cache: {e}")
        return
    captured = buf.getvalue()

    # Stale cache must have been discarded; surviving warm-start signals gone.
    check(not hasattr(scene, "_last_top_eigvec"),
          "_last_top_eigvec removed after stale-shape fallback")
    check("_last_top_eigvec" in captured and "WARNING" in captured,
          "WARNING printed explaining the stale-cache fallback "
          f"(captured head: {captured.splitlines()[0] if captured else 'EMPTY'!r})")

    # Activation still happened; four param tensors are registered.
    check(getattr(scene, "_thin_surface_active", False),
          "_thin_surface_active True after fallback")
    check(scene.quaternions.shape == (n_post, 4),
          f"quaternions shape == (N_post={n_post}, 4) (got {tuple(scene.quaternions.shape)})")
    check(scene.density_delta.shape == (n_post, 1),
          f"density_delta shape == (N_post={n_post}, 1) "
          f"(got {tuple(scene.density_delta.shape)})")
    check(scene.texel_sites_2d.shape == (n_post, 4, 2),
          f"texel_sites_2d shape == (N_post, K=4, 2) "
          f"(got {tuple(scene.texel_sites_2d.shape)})")
    check(scene.texel_heights.shape == (n_post, 4),
          f"texel_heights shape == (N_post, K=4) "
          f"(got {tuple(scene.texel_heights.shape)})")

    # No-warm fallback -> quaternions must equal the identity quaternion
    # [w=1, x=0, y=0, z=0] for every cell.
    q_id = torch.zeros(n_post, 4)
    q_id[:, 0] = 1.0
    check(torch.allclose(scene.quaternions.detach(), q_id, atol=1e-6),
          "quaternions == identity for every cell after stale-cache fallback")

    # Diagnostics must reflect warm_start=0 (cache discarded).
    d = scene.thin_surface_diagnostics()
    check(d is not None, "diagnostics returns a dict")
    check(d["warm_start"] == 0.0,
          f"diagnostics warm_start == 0.0 (got {d['warm_start']})")


# ---------------------------------------------------------------------------
# Test B: real prune permutes the cache so warm-start works post-prune.
# ---------------------------------------------------------------------------
def test_prune_permutes_warm_start_cache():
    """After a real `prune_points`, the four per-cell boundary caches must
    be permuted by the surviving-point mask so that warm-start at the next
    `initialize_thin_surface` call uses the right rows."""
    print("\n--- Test B: prune_points permutes _last_top_eigvec and friends ---")
    n_pre = 20
    n_prune = 6
    n_post = n_pre - n_prune
    scene = _make_minimal_scene(n_pre, device="cpu")
    args = _args()
    scene.declare_optimizer(args, warmup=0, max_iterations=10000)

    # Populate all four caches with values we can verify after the prune.
    torch.manual_seed(123)
    pre_v = torch.randn(n_pre, 3, device="cpu")
    pre_v = pre_v / pre_v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pre_trace = torch.arange(n_pre, dtype=torch.float32, device="cpu")
    pre_valid = (pre_trace % 2 == 0)
    pre_resid = pre_trace.clone() * 0.1
    scene._last_top_eigvec = pre_v.clone()
    scene._last_M_trace = pre_trace.clone()
    scene._last_M_valid = pre_valid.clone()
    scene._last_normal_lap_residual = pre_resid.clone()

    # Build a mask: drop the first n_prune cells (any mask shape works; we
    # only need to verify the surviving rows are kept in order).
    prune_mask = torch.zeros(n_pre, dtype=torch.bool, device="cpu")
    prune_mask[:n_prune] = True
    scene.prune_points(prune_mask)

    # After prune: scene has n_post points; caches must also be n_post.
    check(scene.primal_points.shape[0] == n_post,
          f"primal_points shrunk to N_post={n_post} "
          f"(got {scene.primal_points.shape[0]})")
    for name in ("_last_top_eigvec", "_last_M_trace", "_last_M_valid",
                 "_last_normal_lap_residual"):
        t = getattr(scene, name)
        check(t is not None and t.shape[0] == n_post,
              f"{name} shape[0] == N_post={n_post} "
              f"(got {t.shape[0] if t is not None else None})")

    # The surviving rows must match the original pre_v pre_trace etc. for
    # indices n_prune..n_pre (since prune_mask[:n_prune] = True).
    check(torch.allclose(scene._last_top_eigvec, pre_v[n_prune:], atol=1e-7),
          "_last_top_eigvec permuted to surviving rows")
    check(torch.allclose(scene._last_M_trace, pre_trace[n_prune:], atol=1e-7),
          "_last_M_trace permuted to surviving rows")
    check(torch.equal(scene._last_M_valid, pre_valid[n_prune:]),
          "_last_M_valid permuted to surviving rows")
    check(torch.allclose(scene._last_normal_lap_residual, pre_resid[n_prune:], atol=1e-7),
          "_last_normal_lap_residual permuted to surviving rows")

    # Now activate thin surface -- the warm-start must succeed without a
    # crash and must use the permuted cache.
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            scene.initialize_thin_surface(args, K=4)
    except RuntimeError as e:
        check(False,
              f"initialize_thin_surface crashed after correct prune: {e}")
        return
    captured = buf.getvalue()

    check("WARNING" not in captured or "_last_top_eigvec shape" not in captured,
          "no stale-cache WARNING when the cache was permuted by prune")

    # Verify warm-start actually applied: the surviving cache rows are unit
    # vectors pointing in random directions, so q[:,0] should NOT be all 1
    # (which it would be for identity quaternions).  At least one row must
    # have a non-trivial w-component (or non-trivial xyz).
    q = scene.quaternions.detach()
    q_norm = q.norm(dim=-1)
    check(torch.allclose(q_norm, torch.ones_like(q_norm), atol=1e-4),
          "quaternions are unit-norm after warm-start")
    # Heuristic: at least one cell whose xyz components are non-negligible.
    xyz_norm = q[:, 1:].norm(dim=-1)
    check((xyz_norm > 1e-3).any().item(),
          "at least one quaternion has non-identity xyz (warm-start applied)")

    # Diagnostics must reflect warm_start=1.
    d = scene.thin_surface_diagnostics()
    check(d is not None and d["warm_start"] == 1.0,
          f"diagnostics warm_start == 1.0 (got {d['warm_start'] if d else None})")


# ---------------------------------------------------------------------------
# Test C: no-warm path (cache absent).  Should be unchanged.
# ---------------------------------------------------------------------------
def test_no_warm_path_unaffected():
    """Without any `_last_top_eigvec`, `initialize_thin_surface` must produce
    identity quaternions and a clean activation, exactly as before the fix."""
    print("\n--- Test C: no-warm path (no _last_top_eigvec) ---")
    n = 8
    scene = _make_minimal_scene(n, device="cpu")
    args = _args()
    scene.declare_optimizer(args, warmup=0, max_iterations=10000)

    # No cache present.
    check(not hasattr(scene, "_last_top_eigvec"),
          "scene has no _last_top_eigvec before activation")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            scene.initialize_thin_surface(args, K=4)
    except RuntimeError as e:
        check(False,
              f"initialize_thin_surface crashed on no-warm path: {e}")
        return
    captured = buf.getvalue()
    check("WARNING" not in captured,
          "no WARNING printed on the no-warm path")

    # Identity quaternions.
    q = scene.quaternions.detach()
    q_id = torch.zeros(n, 4)
    q_id[:, 0] = 1.0
    check(torch.allclose(q, q_id, atol=1e-6),
          "quaternions == identity on no-warm path")

    d = scene.thin_surface_diagnostics()
    check(d["warm_start"] == 0.0,
          f"diagnostics warm_start == 0.0 on no-warm path "
          f"(got {d['warm_start']})")


# ---------------------------------------------------------------------------
# Test D: no-op prune keeps the cache row-aligned and warm-start works.
# ---------------------------------------------------------------------------
def test_no_op_prune_preserves_warm_start():
    """A prune that drops zero points must keep the cache intact (shape and
    rows) so that warm-start at the next `initialize_thin_surface` call still
    succeeds with no warning."""
    print("\n--- Test D: no-op prune preserves warm-start ---")
    n = 10
    scene = _make_minimal_scene(n, device="cpu")
    args = _args()
    scene.declare_optimizer(args, warmup=0, max_iterations=10000)

    # Populate the cache with a deterministic non-identity rotation pattern
    # so warm-start is observable.
    torch.manual_seed(7)
    pre_v = torch.randn(n, 3, device="cpu")
    pre_v = pre_v / pre_v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scene._last_top_eigvec = pre_v.clone()
    scene._last_M_trace = torch.zeros(n, device="cpu")
    scene._last_M_valid = torch.ones(n, dtype=torch.bool, device="cpu")
    scene._last_normal_lap_residual = torch.zeros(n, device="cpu")

    # No-op prune: every cell survives.
    prune_mask = torch.zeros(n, dtype=torch.bool, device="cpu")
    scene.prune_points(prune_mask)

    check(scene.primal_points.shape[0] == n,
          f"primal_points still N={n} after no-op prune")
    check(scene._last_top_eigvec.shape[0] == n,
          f"_last_top_eigvec still N={n} after no-op prune")
    check(torch.allclose(scene._last_top_eigvec, pre_v, atol=1e-7),
          "_last_top_eigvec rows unchanged after no-op prune")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        scene.initialize_thin_surface(args, K=4)
    captured = buf.getvalue()
    check("WARNING" not in captured,
          "no WARNING printed after no-op prune")

    d = scene.thin_surface_diagnostics()
    check(d["warm_start"] == 1.0,
          f"diagnostics warm_start == 1.0 after no-op prune (got {d['warm_start']})")


def main():
    print("=" * 60)
    print("Thin-Surface Warm-Start Prune-Alignment Regression Test")
    print("=" * 60)
    if not _HAS_CUDA:
        print("NOTE: CPU-only test.  The activation path is device-agnostic")

    test_stale_warm_start_cache_falls_back_safely()
    test_prune_permutes_warm_start_cache()
    test_no_warm_path_unaffected()
    test_no_op_prune_preserves_warm_start()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL WARM-START PRUNE-ALIGNMENT TESTS PASSED.")


if __name__ == "__main__":
    main()