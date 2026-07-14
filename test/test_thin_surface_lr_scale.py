"""Lightweight test: `thin_surface_lr_scale` config field and its
optimum-zero R1 behaviour.

Verifies the R1 isolation contract:
  1. Without the field, default = 1.0, all four thin param-group LRs are
     the previous (failed-recipe) value `density_lr_init * 0.1`.
  2. With `thin_surface_lr_scale = 0.0`, all four thin param-group LRs
     become exactly 0.0 (scaled initial and post-update_learning_rate),
     and `_thin_surface_active` remains True (so the two-sided forward
     kernel still runs in the training loop).
  3. The scheduler-config block also reflects the scaled value: for
     scale=0 the cached `lr_init` and `lr_final` are both 0.0.

This is the R1 gate: thin params are frozen at init values, the optimizer
can still step the base density and primal points, and the surface remains
inert (so we can observe the network's stage-0-equivalent behaviour
while the two-sided forward kernel still emits its contribution).

Run with:  micromamba run -n radfoam python test/test_thin_surface_lr_scale.py
"""
import sys
import os
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np

# Repo root on path (match other test files).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


_HAS_CUDA = torch.cuda.is_available()
if not _HAS_CUDA:
    # We don't actually need CUDA for this test -- initialize_thin_surface
    # only touches ParameterList/optimizer on the configured device, but
    # `radfoam` is imported at the top of scene.py, so the stub must be
    # installed to load the module on CPU.
    pass


# ---------------------------------------------------------------------------
# Stub radfoam so we can import scene.py on CPU (the model never actually
# calls into the CUDA pipeline in this test).
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
# Helpers
# ---------------------------------------------------------------------------
def _args(thin_lr_scale=1.0):
    """Build a minimal Namespace for declare_optimizer + initialize_thin_surface."""
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-2
    a.freeze_points = 9500
    a.thin_surface_start = 0          # activate immediately for the test
    a.thin_surface_K = 4
    a.thin_surface_delta_weight = 1e-3
    a.thin_surface_height_weight = 5e-4
    a.thin_surface_gate_tau = 0.01
    a.thin_surface_lr_scale = thin_lr_scale
    # Per-group scales default to 1.0 (the field is the global knob).
    a.thin_surface_delta_lr_scale = 1.0
    a.thin_surface_quat_lr_scale = 1.0
    a.thin_surface_sites_lr_scale = 1.0
    a.thin_surface_heights_lr_scale = 1.0
    # Required by initialize_thin_surface for the post-step clip/grad-clip:
    a.thin_surface_delta_clip = 2.0
    a.thin_surface_grad_clip = 1.0
    return a


def _make_minimal_scene(n_points=8, device="cpu"):
    """CPU-friendly scene with a fake all-pairs adjacency.

    No Delaunay triangulation needed -- initialize_thin_surface does not
    touch the triangulation.
    """
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


def _lr_for(scene, name):
    for g in scene.optimizer.param_groups:
        if g["name"] == name:
            return g["lr"]
    return None


_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# ---------------------------------------------------------------------------
# Test 1: default scale = 1.0 produces the failed-recipe LRs (back-compat).
# ---------------------------------------------------------------------------
def test_default_scale_preserves_failed_recipe():
    """thin_surface_lr_scale=1.0 (default) must produce exactly the original
    LRs: density_lr_init * 0.1 (= 5e-3 when density_lr_init=5e-2)."""
    print("\n--- Test 1: default scale = 1.0 preserves failed-recipe LRs ---")
    scene = _make_minimal_scene(n_points=8, device="cpu")
    args = _args(thin_lr_scale=1.0)        # explicit default
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)

    expected_lr = args.density_lr_init * 0.1   # 5e-3
    check(scene._thin_surface_active,
          "_thin_surface_active is True after init")
    check(scene._thin_surface_lr_scale == 1.0,
          f"_thin_surface_lr_scale == 1.0 (got {scene._thin_surface_lr_scale})")
    for name in THIN_GROUP_NAMES:
        lr = _lr_for(scene, name)
        check(lr == expected_lr,
              f"{name} LR == density_lr_init * 0.1 = {expected_lr} (got {lr})")

    # Verify the per-group scheduler also returns the expected init LR.
    check(scene._thin_surface_group_lr_init["density_delta"] == expected_lr,
          f"group scheduler lr_init for density_delta == {expected_lr}")


# ---------------------------------------------------------------------------
# Test 2: scale = 0.0 freezes all four thin param-group LRs while leaving
# _thin_surface_active = True.  This is the R1 isolation contract.
# ---------------------------------------------------------------------------
def test_zero_scale_freezes_thin_lrs():
    """thin_surface_lr_scale=0.0 -> every thin param-group LR is exactly 0.0
    at activation iter; _thin_surface_active remains True; the schedulers
    are zero; update_learning_rate does not bring them back to nonzero."""
    print("\n--- Test 2: thin_surface_lr_scale=0.0 freezes thin LRs (R1) ---")
    scene = _make_minimal_scene(n_points=8, device="cpu")
    args = _args(thin_lr_scale=0.0)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)

    check(scene._thin_surface_active,
          "_thin_surface_active is True (kernel still active in R1)")
    check(scene._thin_surface_lr_scale == 0.0,
          f"_thin_surface_lr_scale == 0.0 (got {scene._thin_surface_lr_scale})")

    # All four thin param-group LRs must be exactly 0.0 at activation.
    for name in THIN_GROUP_NAMES:
        lr = _lr_for(scene, name)
        check(lr == 0.0,
              f"{name} LR == 0.0 at activation (got {lr})")

    # Per-group scheduler init and final must both be 0.0.
    for name in THIN_GROUP_NAMES:
        lr0 = scene._thin_surface_group_lr_init[name]
        s = scene._thin_surface_group_scheduler[name]
        s_at_0 = s(0)
        s_at_max = s(args.thin_surface_start + 100)
        check(lr0 == 0.0,
              f"group scheduler lr_init[{name}] == 0.0 (got {lr0})")
        check(s_at_0 == 0.0,
              f"group scheduler[{name}](0) == 0.0 (got {s_at_0})")
        check(s_at_max == 0.0,
              f"group scheduler[{name}](max_iter) == 0.0 (got {s_at_max})")

    # After update_learning_rate at the activation iter the LRs must remain
    # zero (the frozen-surface contract).
    scene.update_learning_rate(args.thin_surface_start)
    for name in THIN_GROUP_NAMES:
        lr = _lr_for(scene, name)
        check(lr == 0.0,
              f"{name} LR still 0.0 after update_learning_rate (got {lr})")

    # update_learning_rate at a much later iter also leaves them at zero
    # (the cosine scheduler's LR_final is 0.1 * lr_init = 0).
    scene.update_learning_rate(args.thin_surface_start + 5000)
    for name in THIN_GROUP_NAMES:
        lr = _lr_for(scene, name)
        check(lr == 0.0,
              f"{name} LR still 0.0 at later iter (got {lr})")


# ---------------------------------------------------------------------------
# Test 3: non-zero scale (e.g. 0.5) scales proportionally without disabling.
# ---------------------------------------------------------------------------
def test_partial_scale_proportional():
    """thin_surface_lr_scale=0.5 halves every thin param-group LR; the
    base density and primal_points param groups are unaffected."""
    print("\n--- Test 3: thin_surface_lr_scale=0.5 halves thin LRs ---")
    scene = _make_minimal_scene(n_points=8, device="cpu")
    args = _args(thin_lr_scale=0.5)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)

    base_lr = args.density_lr_init * 0.1   # 5e-3
    expected_lr = base_lr * 0.5          # 2.5e-3
    for name in THIN_GROUP_NAMES:
        lr = _lr_for(scene, name)
        check(abs(lr - expected_lr) < 1e-15,
              f"{name} LR == 0.5 * base = {expected_lr} (got {lr})")

    # Sanity: base param groups (primal_points, density) are unaffected.
    base_density_lr = _lr_for(scene, "density")
    base_points_lr = _lr_for(scene, "primal_points")
    check(base_density_lr == args.density_lr_init,
          f"density (base) LR unchanged: {base_density_lr}")
    check(base_points_lr == args.points_lr_init,
          f"primal_points (base) LR unchanged: {base_points_lr}")


# ---------------------------------------------------------------------------
# Test 4: scale=0 path on a "resumed" scene (initialize_thin_surface is
# idempotent and respects the scale on the resume path too).
# ---------------------------------------------------------------------------
def test_zero_scale_idempotent_on_resume():
    """A second call to initialize_thin_surface on a resumed scene must
    not re-add a param group and must keep all thin LRs at 0.0."""
    print("\n--- Test 4: thin_surface_lr_scale=0.0 idempotent on resume ---")
    scene = _make_minimal_scene(n_points=8, device="cpu")
    args = _args(thin_lr_scale=0.0)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)
    pre_count = len(scene.optimizer.param_groups)
    pre_lrs = {n: _lr_for(scene, n) for n in THIN_GROUP_NAMES}

    # Re-activate (idempotency path).  initialize_thin_surface detects the
    # existing tensors and re-attaches the param groups.
    scene.initialize_thin_surface(args, K=4)
    post_count = len(scene.optimizer.param_groups)
    check(post_count == pre_count,
          f"Re-activation does not duplicate param groups "
          f"({pre_count} -> {post_count})")
    for n in THIN_GROUP_NAMES:
        lr = _lr_for(scene, n)
        check(lr == pre_lrs[n],
              f"{n} LR preserved on resume ({pre_lrs[n]} -> {lr})")
        check(lr == 0.0,
              f"{n} LR still 0.0 on resume (got {lr})")


def main():
    print("=" * 60)
    print("Thin-Surface LR-Scale Test (R1 isolation gate)")
    print("=" * 60)

    if not _HAS_CUDA:
        print("NOTE: running on CPU.  The optimizer + scheduler logic is")
        print("device-agnostic; this test exercises the config plumbing")

    test_default_scale_preserves_failed_recipe()
    test_zero_scale_freezes_thin_lrs()
    test_partial_scale_proportional()
    test_zero_scale_idempotent_on_resume()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL LR-SCALE TESTS PASSED.")


if __name__ == "__main__":
    main()
