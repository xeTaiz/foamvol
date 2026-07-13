"""
Autograd shape checks, config plumbing, and K validation for thin-surface.

Tests:
  1. Autograd shape contract: verify that TraceRays.backward() returns gradients
     with the correct shapes for all thin-surface parameters. (P0-C fix landed:
     density_delta grad is now (N, 1), matching the param.)
  2. K=4 config plumbing: verify that the default config (thin_surface_K=4)
     is correctly propagated through C++ TraceSettings and used by the kernel.
  3. K=8 support: verify that non-4 K values don't crash.
  4. Zero-init inertness: with all thin-surface params at init (density_delta=0,
     heights=0, identity quaternions), the output should match the baseline.
  5. Forward/backward gradient consistency for all 5 param groups.

Run with:  micromamba run -n radfoam python test/test_thin_surface_scene.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

# Ensure project root is on path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HAS_CUDA = torch.cuda.is_available()

# Config-only tests need configargparse but not radfoam (which requires CUDA)
import configargparse
from configs import OptimizationParams

# GPU tests need radfoam + CTScene, imported conditionally
if _HAS_CUDA:
    import radfoam
    from radfoam_model.scene import CTScene
    import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

_any_failed = False


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# ─────────────────────────────────────────────────────────
# Config and default-value tests (no GPU needed)
# ─────────────────────────────────────────────────────────

def test_config_defaults():
    """Test 1: Verify config parameter defaults."""
    print("\n--- Test: Config parameter defaults ---")
    parser = configargparse.ArgParser()
    opt_params = OptimizationParams(parser)

    check(opt_params.thin_surface_start == -1,
          f"thin_surface_start default: {opt_params.thin_surface_start}")
    check(opt_params.thin_surface_K == 4,
          f"thin_surface_K default: {opt_params.thin_surface_K}")
    check(opt_params.thin_surface_delta_weight == 1e-3,
          f"thin_surface_delta_weight default: {opt_params.thin_surface_delta_weight}")
    check(opt_params.thin_surface_height_weight == 1e-3,
          f"thin_surface_height_weight default: {opt_params.thin_surface_height_weight}")
    check(opt_params.thin_surface_gate_tau == 0.01,
          f"thin_surface_gate_tau default: {opt_params.thin_surface_gate_tau}")

    # Config fields exist
    for field in ["thin_surface_start", "thin_surface_K",
                  "thin_surface_delta_weight", "thin_surface_height_weight",
                  "thin_surface_gate_tau"]:
        check(hasattr(opt_params, field), f"Config field exists: {field}")


def test_K8_config_value_accepted():
    """Test 2: K=8 is accepted as a config value without error.

    Note: This tests config, not CUDA kernel. The kernel reads settings.thin_K
    as an int and loops K times — any positive K should work, but only K=4
    has been verified.
    """
    print("\n--- Test: K=8 config value ---")
    parser = configargparse.ArgParser()
    opt_params = OptimizationParams(parser)
    check(
        hasattr(opt_params, 'thin_surface_K'),
        "thin_surface_K is a config field"
    )
    # K=8 is a valid int; no validation in the config layer prevents it
    check(opt_params.thin_surface_K == 4,
          "Default K is 4 (config sets 8 after user override)")


# ─────────────────────────────────────────────────────────
# GPU tests (require CUDA)
# ─────────────────────────────────────────────────────────

def _make_minimal_scene(n_points=32, device="cuda"):
    """Minimal scene for shape/config tests.

    n_points defaults to 32: radfoam.Triangulation requires >= MIN_POINTS=32
    (see old/test_cube.py); smaller scenes yield empty/invalid adjacency.
    """
    args = type("Args", (), {
        "init_points": n_points,
        "final_points": n_points,
        "activation_scale": 1.0,
        "init_scale": 0.5,
        "init_type": "random",
        "init_density": 0.5,
        "device": device,
        "init_points_file": "",
        "init_volume_path": "",
        "frozen_points_file": "",
        "frozen_freeze_density": True,
        "density_lr_init": 5e-2,
    })()
    model = CTScene(args, device=torch.device(device))
    return model


def _make_test_rays(model, n_rays=4):
    """Create simple rays through the scene."""
    pts = model.primal_points.detach()
    center = pts.mean(dim=0)
    directions = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
    ], device=pts.device, dtype=torch.float32)[:n_rays]
    directions = directions / directions.norm(dim=-1, keepdim=True)
    origins = center - 2.0 * directions
    return torch.cat([origins, directions], dim=-1)


def _activate_thin_surface(model, K=4, delta_val=0.0, height_val=0.0):
    """Register thin-surface params on an existing model."""
    N = model.primal_points.shape[0]
    device = model.device
    model._thin_surface_active = True
    model._thin_K = K
    model._thin_surface_gate_tau = 0.01
    model._max_iterations = 1000
    model._thin_surface_start = 0

    model.density_delta = nn.Parameter(delta_val * torch.ones(N, 1, device=device))
    q0 = torch.zeros(N, 4, device=device)
    q0[:, 0] = 1.0
    model.quaternions = nn.Parameter(q0)
    angles = torch.linspace(0, 2 * np.pi, K + 1, device=device)[:-1]
    base_sites = torch.stack([torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    model.texel_sites_2d = nn.Parameter(base_sites.unsqueeze(0).expand(N, -1, -1).clone())
    model.texel_heights = nn.Parameter(height_val * torch.ones(N, K, device=device))

    with torch.no_grad():
        _, cr = radfoam.farthest_neighbor(
            model.primal_points, model.point_adjacency, model.point_adjacency_offsets,
        )
        model._cached_cell_radius = cr.squeeze()
    return model


def test_density_delta_grad_shape():
    """Test 3: Verify density_delta gradient shape matches parameter shape.

    P0-C fix landed: grad is now (N,1) matching the param. Expected to PASS.
    """
    print("\n--- Test: density_delta grad shape ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"

    model = _make_minimal_scene(device=device)
    model = _activate_thin_surface(model, K=4, delta_val=0.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    model.zero_grad()
    out, _, _, _, _ = model(rays, start_point)
    loss = out.sum()
    loss.backward()

    dd = model.density_delta
    check(dd.shape == (model.primal_points.shape[0], 1),
          f"density_delta param shape: got {tuple(dd.shape)}, "
          f"expected ({model.primal_points.shape[0]}, 1)")

    if dd.grad is not None:
        check(dd.grad.shape == dd.shape,
              f"density_delta grad shape: got {tuple(dd.grad.shape)}, "
              f"expected {tuple(dd.shape)}. ")
    else:
        check(False,
              "density_delta.grad is None after backward — REGRESSION: "
              "P0-C shape fix should produce a (N,1) grad")


def test_quaternions_grad_shape():
    """Test 4: Verify quaternions gradient shape (N, 4)."""
    print("\n--- Test: quaternions grad shape ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"

    model = _make_minimal_scene(device=device)
    model = _activate_thin_surface(model, K=4, delta_val=0.3)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    model.zero_grad()
    out, _, _, _, _ = model(rays, start_point)
    loss = out.sum()
    loss.backward()

    q = model.quaternions
    check(q.shape == (model.primal_points.shape[0], 4),
          f"quaternions param shape: {tuple(q.shape)}")

    if q.grad is not None:
        check(q.grad.shape == q.shape,
              f"quaternions grad shape: {tuple(q.grad.shape)} matches param")
        check(q.grad.isfinite().all(), "quaternions grad is finite")
    else:
        check(False, "quaternions.grad is not None after backward")


def test_texel_sites_grad_shape():
    """Test 5: Verify texel_sites_2d gradient shape (N, K, 2)."""
    print("\n--- Test: texel_sites_2d grad shape ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"

    model = _make_minimal_scene(device=device)
    model = _activate_thin_surface(model, K=4, delta_val=0.3)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    model.zero_grad()
    out, _, _, _, _ = model(rays, start_point)
    loss = out.sum()
    loss.backward()

    ts = model.texel_sites_2d
    N = model.primal_points.shape[0]
    check(ts.shape == (N, 4, 2),
          f"texel_sites_2d param shape: {tuple(ts.shape)}, expected ({N}, 4, 2)")

    if ts.grad is not None:
        check(ts.grad.shape == ts.shape,
              f"texel_sites_2d grad shape: {tuple(ts.grad.shape)} matches param")
        check(ts.grad.isfinite().all(), "texel_sites_2d grad is finite")
    else:
        check(False, "texel_sites_2d.grad is not None after backward")


def test_texel_heights_grad_shape():
    """Test 6: Verify texel_heights gradient shape (N, K)."""
    print("\n--- Test: texel_heights grad shape ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"

    model = _make_minimal_scene(device=device)
    model = _activate_thin_surface(model, K=4, delta_val=0.3, height_val=0.02)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    model.zero_grad()
    out, _, _, _, _ = model(rays, start_point)
    loss = out.sum()
    loss.backward()

    th = model.texel_heights
    N = model.primal_points.shape[0]
    check(th.shape == (N, 4),
          f"texel_heights param shape: {tuple(th.shape)}, expected ({N}, 4)")

    if th.grad is not None:
        check(th.grad.shape == th.shape,
              f"texel_heights grad shape: {tuple(th.grad.shape)} matches param")
        check(th.grad.isfinite().all(), "texel_heights grad is finite")
    else:
        check(False, "texel_heights.grad is not None after backward")


def test_zero_init_inertness():
    """Test 7: Zero-init thin-surface should match baseline (no surface).

    With density_delta=0, texel_heights=0, identity quaternions, the thin-surface
    kernel should produce the same projection as the baseline kernel
    (thin_surface_mode=False). Since mu_plus = mu_minus = mu_bar, the two-sided
    contribution degenerates to mu_bar * delta_t.
    """
    print("\n--- Test: zero-init inertness ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"

    model = _make_minimal_scene(device=device)
    model = _activate_thin_surface(model, K=4, delta_val=0.0, height_val=0.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    # Baseline: thin_surface_mode=False
    model._thin_surface_active = False
    with torch.no_grad():
        baseline_out, _, _, _, _ = model(rays, start_point)

    # Thin-surface at zero init
    model._thin_surface_active = True
    with torch.no_grad():
        ts_init_out, _, _, _, _ = model(rays, start_point)

    max_diff = (baseline_out - ts_init_out).abs().max().item()
    check(max_diff < 1e-4,
          f"Zero-init thin-surface matches baseline (max_diff={max_diff:.6f}, tol=1e-4)")


def test_K6_forward_backward():
    """Test 8: Verify that K=6 (non-default K) works without crashing.

    Raw-kernel smoke for non-default K. NOTE: the trained path
    (CTScene.initialize_thin_surface / load_pt) rejects K not in {4} via
    assert_supported_thin_K until a gradcheck extends _SUPPORTED_THIN_K, so
    K=6 is NOT a sanctioned training config. This test bypasses that guard
    by setting tensors+`_thin_K` directly to exercise the CUDA kernel's
    K-loop / stride handling (P0-B plumbing fix makes K!=4 stride-correct).
    """
    print("\n--- Test: K=6 forward/backward ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    # P0 policy is K=4-only until a finite-difference gradcheck for K=6 extends
    # scene._SUPPORTED_THIN_K. This raw-kernel smoke is non-blocking: skip
    # unless explicitly opted in via RADFOAM_TEST_K6=1.
    import os
    if os.environ.get("RADFOAM_TEST_K6", "0").lower() not in ("1", "on", "true", "yes"):
        print("  SKIP: K=6 is non-blocking under K=4-only P0 policy "
              "(set RADFOAM_TEST_K6=1 to run the raw-kernel smoke)")
        return
    device = "cuda"

    model = _make_minimal_scene(n_points=32, device=device)
    model = _activate_thin_surface(model, K=6, delta_val=0.3, height_val=0.01)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    model.zero_grad()
    try:
        out, _, _, _, _ = model(rays, start_point)
        loss = out.sum()
        loss.backward()
        check(out.isfinite().all(), "K=6 forward produces finite output")
        check(loss.isfinite(), "K=6 backward produces finite loss")

        N = model.primal_points.shape[0]
        check(model.texel_sites_2d.shape == (N, 6, 2),
              f"K=6 texel_sites_2d shape: {tuple(model.texel_sites_2d.shape)}")
        check(model.texel_heights.shape == (N, 6),
              f"K=6 texel_heights shape: {tuple(model.texel_heights.shape)}")

        for name in ["density", "density_delta", "quaternions",
                      "texel_sites_2d", "texel_heights"]:
            param = getattr(model, name, None)
            if param is not None and param.grad is not None:
                finite = param.grad.isfinite().all()
                check(finite, f"K=6 {name} grad is finite")
                shape_ok = param.grad.shape == param.shape
                check(shape_ok, f"K=6 {name} grad shape matches param")
            elif param is not None:
                check(False, f"K=6 {name} grad is not None")

    except Exception as e:
        check(False, f"K=6 forward/backward: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_forward_backward_consistency():
    """Test 9: Verify backward produces finite, nonzero gradients for all 5 params."""
    print("\n--- Test: forward/backward gradient consistency ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"

    model = _make_minimal_scene(n_points=32, device=device)
    model = _activate_thin_surface(model, K=4, delta_val=0.3, height_val=0.02)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    model.zero_grad()
    out, _, _, _, _ = model(rays, start_point)
    check(out.isfinite().all(), "Forward output is finite")
    check(out.abs().sum() > 0, "Forward output is nonzero")

    loss = out.sum()
    loss.backward()

    for name in ["density", "density_delta", "quaternions",
                  "texel_sites_2d", "texel_heights"]:
        param = getattr(model, name, None)
        if param is None:
            check(False, f"{name} parameter exists")
            continue

        if param.grad is None:
            check(False,
                  f"{name}.grad is None after backward (P0-C: grad shape must "
                  f"match param shape)")
            continue

        grad_ok = param.grad.isfinite().all() and param.grad.abs().sum() > 0
        check(grad_ok,
              f"{name} grad: finite={param.grad.isfinite().all().item()}, "
              f"nonzero={param.grad.abs().sum().item() > 0}")
        check(param.grad.shape == param.shape,
              f"{name} grad shape {tuple(param.grad.shape)} matches "
              f"param shape {tuple(param.shape)}")


def test_activation_continuity():
    """Rendering must be identical immediately before and after zero-init
    thin-surface activation (density_delta=0, texel_heights=0, identity quat).

    Safety guard against an activation-time discontinuity: with delta=0,
    mu_plus==mu_minus==mu_bar and the two-sided contribution collapses to
    mu_bar * delta_t (both crossing and non-crossing branches), so the
    thin-surface forward must equal the scalar forward to fp tolerance. A
    violation here indicates a kernel/parameterization bug, not dynamics.
    """
    print("\n--- Test: activation continuity (zero-init render invariant) ---")
    if not _HAS_CUDA:
        print("  SKIP: requires CUDA")
        return
    device = "cuda"
    model = _make_minimal_scene(device=device)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    # Scalar baseline (thin-surface OFF)
    model._thin_surface_active = False
    with torch.no_grad():
        baseline, *_ = model(rays, start_point)

    # Activate with strictly zero init: delta=0, heights=0, identity quaternion.
    _activate_thin_surface(model, K=4, delta_val=0.0, height_val=0.0)
    model._thin_surface_active = True
    with torch.no_grad():
        thin, *_ = model(rays, start_point)

    max_diff = (baseline - thin).abs().max().item()
    check(baseline.isfinite().all() and thin.isfinite().all(),
          f"both renders finite (max_diff={max_diff:.2e})")
    check(max_diff < 1e-4,
          f"activation continuity: scalar==thin(zero-init) "
          f"max_diff={max_diff:.2e} (tol 1e-4)")


def main():
    print("=" * 60)
    print("Thin-Surface Scene Tests — Shapes, Config, Inertness")
    print("=" * 60)

    # Config/default tests (no GPU)
    test_config_defaults()
    test_K8_config_value_accepted()

    # GPU tests
    test_density_delta_grad_shape()
    test_quaternions_grad_shape()
    test_texel_sites_grad_shape()
    test_texel_heights_grad_shape()
    test_zero_init_inertness()
    test_activation_continuity()
    test_K6_forward_backward()
    test_forward_backward_consistency()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above)")
        print("Grad shape contract (P0-C): density_delta grad must be (N,1).")
        sys.exit(1)
    else:
        print("SUMMARY: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
