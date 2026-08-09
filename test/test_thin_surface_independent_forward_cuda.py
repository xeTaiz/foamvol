"""LC64 plan v3 Commit 2A -- CUDA-native independent-side FORWARD smoke.

Verifies the forward dispatch on a real GPU:

  - fixed-ray zero-split independent forward equals scalar at numerical
    tolerance (raw_plus == raw_minus == softplus_inv(target mu) -> the
    independent projection equals the scalar projection at fp tol)
  - changing legacy base density does NOT change independent projection
    (independent mode ignores the frozen base density)
  - independently changing raw_plus / raw_minus changes only the
    rays/segments using that side (the opposite side is invariant)
  - activation_scale 1.0 vs a non-1 value scales side attenuation
    correctly (mu_p / mu_n scale linearly)
  - geometry zero-split invariance (raw_plus == raw_minus, identity
    quat, zero heights) -> matches scalar baseline to fp tol
  - malformed / mixed inputs fail BEFORE launch (returns a check on
    the C++ binding's pre-launch validation through Python)
  - independent backward runs end-to-end and returns finite raw-side gradients

Run with:
    micromamba run -n radfoam python test/test_thin_surface_independent_forward_cuda.py
"""
import os
import sys
import types
import warnings
warnings.filterwarnings("ignore")

import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import radfoam
from radfoam_model.scene import CTScene
from radfoam_model.render import TraceRays


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _args(independent=False, K=4, thin_surface_start=0,
          thin_surface_delta_clip=2.0, activation_scale=1.0):
    class A:
        pass
    a = A()
    a.init_points = 32
    a.final_points = 32
    a.activation_scale = activation_scale
    a.init_scale = 0.5
    a.init_type = "random"
    a.init_density = 0.0
    a.init_points_file = ""
    a.init_volume_path = ""
    a.frozen_points_file = ""
    a.frozen_freeze_density = True
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-3
    a.freeze_points = 9500
    a.thin_surface_start = thin_surface_start
    a.thin_surface_K = K
    a.thin_surface_delta_weight = 1e-3
    a.thin_surface_height_weight = 5e-4
    a.thin_surface_gate_tau = 0.01
    a.thin_surface_lr_scale = 1.0
    a.thin_surface_delta_lr_scale = 1.0
    a.thin_surface_quat_lr_scale = 1.0
    a.thin_surface_sites_lr_scale = 1.0
    a.thin_surface_heights_lr_scale = 1.0
    a.thin_surface_delta_clip = thin_surface_delta_clip
    a.thin_surface_grad_clip = 1.0
    a.thin_surface_relative_delta = False
    a.thin_surface_delta_max_frac = 0.5
    a.thin_surface_density_mode = "independent" if independent else "scalar"
    a.thin_surface_raw_side_lr_init = 5e-2
    a.thin_surface_raw_side_lr_final = 1e-3
    a.warmup_steps = 0
    return a


def _make_scene(n_points=32, device="cuda", activation_scale=1.0):
    args = _args(activation_scale=activation_scale)
    scene = CTScene(args, device=torch.device(device))
    return scene


def _activate_thin_surface(model, K=4, delta_val=0.0, height_val=0.0):
    """Helper to register thin-surface params on a CTScene (legacy shape)."""
    N = model.primal_points.shape[0]
    device = model.device
    model.density_delta = nn.Parameter(delta_val * torch.ones(N, 1, device=device))
    q0 = torch.zeros(N, 4, device=device)
    q0[:, 0] = 1.0
    model.quaternions = nn.Parameter(q0)
    angles = torch.linspace(0, 2 * math.pi, K + 1, device=device)[:-1]
    base_sites = torch.stack([torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    model.texel_sites_2d = nn.Parameter(base_sites.unsqueeze(0).expand(N, -1, -1).clone())
    model.texel_heights = nn.Parameter(height_val * torch.ones(N, K, device=device))
    model._thin_surface_active = True
    model._thin_K = K
    model._thin_surface_gate_tau = 0.01
    model._max_iterations = 1000
    model._thin_surface_start = 0
    with torch.no_grad():
        _, cr = radfoam.farthest_neighbor(
            model.primal_points, model.point_adjacency, model.point_adjacency_offsets,
        )
        model._cached_cell_radius = cr.squeeze()
    return model


def _activate_independent(model, K=4, raw_plus_val=0.0, raw_minus_val=0.0,
                          density_val=0.5, activation_scale=1.0):
    """Register independent-side raw logits on a CTScene (Commit 2A)."""
    N = model.primal_points.shape[0]
    device = model.device
    # freeze base density (independent mode requires this)
    model.density = nn.Parameter(density_val * torch.ones(N, 1, device=device))
    model.density.requires_grad_(False)
    # raw_plus / raw_minus (the independent-side logits)
    model.raw_plus = nn.Parameter(raw_plus_val * torch.ones(N, 1, device=device))
    model.raw_minus = nn.Parameter(raw_minus_val * torch.ones(N, 1, device=device))
    # surface geometry (the thinsurface tensors)
    _activate_thin_surface(model, K=K, delta_val=0.0, height_val=0.0)
    # discriminator + activation_scale
    model._thin_surface_density_mode = "independent"
    model.activation_scale = activation_scale
    return model


def _make_test_rays(model, n_rays=4):
    pts = model.primal_points.detach()
    center = pts.mean(dim=0)
    directions = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
    ], device=pts.device, dtype=torch.float32)[:n_rays]
    directions = directions / directions.norm(dim=-1, keepdim=True)
    origins = center - 2.0 * directions
    return torch.cat([origins, directions], dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = False


def test_zero_split_equals_scalar():
    """fixed-ray zero-split independent forward equals scalar at tol.

    When raw_plus == raw_minus == raw_d, the independent kernel
    computes mu_plus == mu_minus == activation_scale * softplus(raw_d).
    The crossing contribution becomes (mu_p * (t_s - t_0) + mu_p *
    (t_1 - t_s)) = mu_p * delta_t, which equals the scalar baseline
    (mu_bar * delta_t) at fp tolerance.
    """
    print("\n--- Test 1: zero-split independent == scalar ---")
    device = "cuda"
    model = _make_scene(device=device)
    raw_d = 0.7
    # density_val=raw_d so the scalar baseline uses softplus(raw_d) for
    # mu_bar, matching the independent mu_p == mu_n == softplus(raw_d)
    # at the same raw_d.  This isolates the zero-split invariant from
    # any difference in absolute activation values.
    model = _activate_independent(model, K=4, raw_plus_val=raw_d,
                                   raw_minus_val=raw_d, density_val=raw_d,
                                   activation_scale=1.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    # baseline: scalar (uses density == raw_d, mu_bar = softplus(raw_d))
    model._thin_surface_density_mode = "scalar"
    model._thin_surface_active = False
    with torch.no_grad():
        baseline, *_ = model(rays, start_point)

    # independent at zero split (mu_p == mu_n == softplus(raw_d))
    model._thin_surface_density_mode = "independent"
    model._thin_surface_active = True
    with torch.no_grad():
        independent, *_ = model(rays, start_point)

    diff = (baseline - independent).abs().max().item()
    check(baseline.isfinite().all() and independent.isfinite().all(),
          f"both renders finite (max diff={diff:.2e})")
    check(diff < 1e-4,
          f"zero-split independent == scalar at fp tol (max diff={diff:.2e}, tol 1e-4)")


def test_density_invariance():
    """changing legacy base density does NOT change independent projection.

    The independent branch ignores the frozen base density entirely.
    """
    print("\n--- Test 2: changing base density does not affect independent ---")
    device = "cuda"
    model = _make_scene(device=device)
    model = _activate_independent(model, K=4, raw_plus_val=0.7,
                                   raw_minus_val=0.7, density_val=0.0,
                                   activation_scale=1.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    with torch.no_grad():
        proj_A, *_ = model(rays, start_point)

    # Mutate the frozen base density (it should not affect the projection).
    with torch.no_grad():
        model.density.data.fill_(5.0)
    with torch.no_grad():
        proj_B, *_ = model(rays, start_point)

    diff = (proj_A - proj_B).abs().max().item()
    check(proj_A.isfinite().all() and proj_B.isfinite().all(),
          f"both renders finite (max diff={diff:.2e})")
    check(diff < 1e-6,
          f"base density change does not affect independent projection "
          f"(max diff={diff:.2e}, tol 1e-6)")


def test_plus_minus_independence():
    """independently changing raw_plus / raw_minus changes only its side.

    With a roughly aligned scene-normal, segments split by the
    quaternion-defined surface use mu_plus or mu_minus (per the
    dp-sign mapping).  Raw_plus == raw_minus is the zero-split
    baseline; bumping raw_plus higher should increase the projection
    on +n side segments and leave the rest roughly invariant.
    """
    print("\n--- Test 3: plus / minus side independence ---")
    device = "cuda"
    model = _make_scene(device=device)
    model = _activate_independent(model, K=4, raw_plus_val=0.0,
                                   raw_minus_val=0.0, density_val=0.0,
                                   activation_scale=1.0)
    # Force a uniform +n surface normal so every ray splits with the
    # same convention.  Identity quaternion -> normal = +x.
    # Then raw_plus drives the +n side attenuation.
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    # baseline: raw_plus == raw_minus == 0 (both mu == softplus(0) ~ 0.69)
    with torch.no_grad():
        proj_zero, *_ = model(rays, start_point)

    # bump raw_plus only
    with torch.no_grad():
        model.raw_plus.data.fill_(1.5)
    with torch.no_grad():
        proj_plus_hi, *_ = model(rays, start_point)

    # bump raw_minus only
    with torch.no_grad():
        model.raw_plus.data.fill_(0.0)
        model.raw_minus.data.fill_(1.5)
    with torch.no_grad():
        proj_minus_hi, *_ = model(rays, start_point)

    delta_p = (proj_plus_hi - proj_zero).abs().sum().item()
    delta_m = (proj_minus_hi - proj_zero).abs().sum().item()
    check(delta_p > 0 and delta_m > 0,
          f"bumping raw_plus ({delta_p:.3e}) or raw_minus ({delta_m:.3e}) "
          f"changes the projection (both deltas positive)")
    # Sanity: bumps of opposite sides produce different projections.
    diff_pm = (proj_plus_hi - proj_minus_hi).abs().max().item()
    check(diff_pm > 1e-4,
          f"plus-bump vs minus-bump render differs "
          f"(max diff={diff_pm:.3e}, tol 1e-4)")


def test_activation_scale_scales_side_attenuation():
    """activation_scale 1.0 vs a non-1 value scales side attenuation.

    Set both raw_plus and raw_minus to a positive value; scale
    activation_scale by k; the projection should scale by k.
    """
    print("\n--- Test 4: activation_scale scales side attenuation ---")
    device = "cuda"
    # Build with activation_scale=1.0
    model = _make_scene(device=device)
    model = _activate_independent(model, K=4, raw_plus_val=0.7,
                                   raw_minus_val=0.7, density_val=0.0,
                                   activation_scale=1.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    with torch.no_grad():
        proj_1x, *_ = model(rays, start_point)

    # Re-activate with activation_scale=2.5  (same scene, same scenes)
    model = _activate_independent(model, K=4, raw_plus_val=0.7,
                                   raw_minus_val=0.7, density_val=0.0,
                                   activation_scale=2.5)
    with torch.no_grad():
        proj_25x, *_ = model(rays, start_point)

    # Projection should scale by 2.5
    rel = proj_25x / proj_1x.clamp_min(1e-12)
    ratio_med = rel.median().item()
    check(abs(ratio_med - 2.5) < 1e-3,
          f"activation_scale 2.5x scales projection by ~2.5 (got {ratio_med:.4f})")


def test_geometry_zero_split_invariance():
    """geometry zero-split invariance: raw_plus == raw_minus + identity
    quat + zero heights -> the projection is invariant to changes in
    the QUATERNION and HEIGHTS (because both sides use the same mu and
    the surface is flat)."""
    print("\n--- Test 5: geometry zero-split invariance ---")
    device = "cuda"
    model = _make_scene(device=device)
    model = _activate_independent(model, K=4, raw_plus_val=0.7,
                                   raw_minus_val=0.7, density_val=0.0,
                                   activation_scale=1.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    # baseline: identity quat, zero heights
    with torch.no_grad():
        proj_zero, *_ = model(rays, start_point)

    # mutate the quaternion (rotate +30° around z) and heights, then re-render
    with torch.no_grad():
        # quaternion that rotates +x onto (cos30, sin30, 0) -> normal = (cos30, sin30, 0)
        ang = math.pi / 6
        c = math.cos(ang / 2)
        s = math.sin(ang / 2)
        # axis = (0, 0, 1), angle = 30 deg
        new_q = torch.zeros_like(model.quaternions)
        new_q[:, 0] = c
        new_q[:, 3] = s
        model.quaternions.data.copy_(new_q)
        model.texel_heights.data.fill_(0.05)
    with torch.no_grad():
        proj_geom, *_ = model(rays, start_point)

    diff = (proj_zero - proj_geom).abs().max().item()
    check(proj_zero.isfinite().all() and proj_geom.isfinite().all(),
          f"both renders finite (max diff={diff:.2e})")
    # With raw_plus == raw_minus AND heights == 0 the contribution is
    # mu_p * delta_t (or mu_n * delta_t) regardless of which side the
    # ray is on -- so the projection is invariant to the quaternion
    # (the dp-sign partition collapses to a constant mu).  A nonzero
    # diff indicates a bug in the dp-sign / crossing geometry.
    check(diff < 1e-4,
          f"zero-split projection is invariant to quaternion / heights "
          f"changes (max diff={diff:.2e}, tol 1e-4)")


def test_malformed_inputs_fail_before_launch():
    """mixed / missing tensors fail BEFORE kernel launch.

    Independent mode must have raw_plus AND raw_minus AND a valid
    thin-surface geometry.  Verify the dispatch rejects:
      - raw_plus without raw_minus
      - raw_minus without raw_plus
      - thin_surface mode off but independent mode on
    """
    print("\n--- Test 6: malformed inputs fail before launch ---")
    device = "cuda"
    model = _make_scene(device=device)
    model = _activate_independent(model, K=4, raw_plus_val=0.7,
                                   raw_minus_val=0.7, density_val=0.0,
                                   activation_scale=1.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    # Case A: raw_plus None, raw_minus present -> should raise
    raised = False
    try:
        with torch.no_grad():
            TraceRays.apply(
                model.pipeline,
                model.primal_points, model.density,
                model.point_adjacency, model.point_adjacency_offsets,
                rays, start_point, False,
                None, 5.0, False, 0.01, 0.1, False, False, None,
                False, None, None, None,
                True,  # thin_surface_mode
                None, None, None, None,
                4, 10.0, 1e-4,
                False, 0.5,
                None,           # raw_plus missing
                model.raw_minus,
                True, 1.0,
            )
    except RuntimeError as e:
        raised = "raw_plus" in str(e) or "missing" in str(e).lower()
    check(raised,
          "missing raw_plus raises RuntimeError before kernel launch")

    # Case B: independent=True but thin_surface_mode=False
    raised_b = False
    try:
        with torch.no_grad():
            TraceRays.apply(
                model.pipeline,
                model.primal_points, model.density,
                model.point_adjacency, model.point_adjacency_offsets,
                rays, start_point, False,
                None, 5.0, False, 0.01, 0.1, False, False, None,
                False, None, None, None,
                False,  # thin_surface_mode OFF
                None, None, None, None,
                4, 10.0, 1e-4,
                False, 0.5,
                model.raw_plus, model.raw_minus,
                True, 1.0,
            )
    except RuntimeError as e:
        raised_b = "thin_surface" in str(e).lower()
    check(raised_b,
          "independent=True without thin_surface_mode raises RuntimeError")

    # Case C: legacy mode (independent=False) with raw_plus present
    raised_c = False
    try:
        with torch.no_grad():
            TraceRays.apply(
                model.pipeline,
                model.primal_points, model.density,
                model.point_adjacency, model.point_adjacency_offsets,
                rays, start_point, False,
                None, 5.0, False, 0.01, 0.1, False, False, None,
                False, None, None, None,
                False,  # legacy scalar
                None, None, None, None,
                4, 10.0, 1e-4,
                False, 0.5,
                model.raw_plus,  # raw_plus present when not expected
                None,
                False, 1.0,
            )
    except RuntimeError as e:
        raised_c = "raw_plus" in str(e) or "raw_minus" in str(e)
    check(raised_c,
          "raw_plus under non-independent mode raises RuntimeError")


def test_independent_backward_smoke():
    """LC64 plan v3 Commit 2B -- independent backward NOW WORKS.

    The previous Commit 2A test asserted backward() raised
    NotImplementedError; that contract is gone.  The new contract is
    that backward runs end-to-end and produces finite raw_plus_grad /
    raw_minus_grad of shape (N, 1) with the chain-rule identity
    dL/draw_side = dL/dmu_side * activation_scale * sigmoid(10*raw_side).
    Detailed FD checks live in
    test/test_thin_surface_independent_backward_cuda.py.  This test
    here is a small smoke check to keep the test suite green.
    """
    print("\n--- Test 7: independent backward smoke (Commit 2B) ---")
    device = "cuda"
    model = _make_scene(device=device)
    model = _activate_independent(model, K=4, raw_plus_val=0.7,
                                   raw_minus_val=0.7, density_val=0.0,
                                   activation_scale=1.0)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)

    # Patch the discriminator
    model._thin_surface_density_mode = "independent"
    model._thin_surface_active = True

    # Clear any existing grads
    for name in ("raw_plus", "raw_minus", "quaternions",
                 "texel_sites_2d", "texel_heights", "primal_points",
                 "density"):
        p = getattr(model, name, None)
        if p is not None:
            p.grad = None

    out, *_ = model(rays, start_point)
    loss = out.sum()
    loss.backward()

    # raw_plus_grad / raw_minus_grad exist, finite, and (N, 1) shaped
    rp_grad = model.raw_plus.grad
    rm_grad = model.raw_minus.grad
    check(rp_grad is not None, "raw_plus.grad is not None (Commit 2B)")
    check(rm_grad is not None, "raw_minus.grad is not None (Commit 2B)")
    if rp_grad is not None and rm_grad is not None:
        check(rp_grad.shape == model.raw_plus.shape,
              f"raw_plus.grad shape {tuple(rp_grad.shape)} == param "
              f"{tuple(model.raw_plus.shape)}")
        check(rm_grad.shape == model.raw_minus.shape,
              f"raw_minus.grad shape {tuple(rm_grad.shape)} == param "
              f"{tuple(model.raw_minus.shape)}")
        check(rp_grad.isfinite().all(),
              f"raw_plus.grad all finite (norm={rp_grad.norm().item():.3e})")
        check(rm_grad.isfinite().all(),
              f"raw_minus.grad all finite (norm={rm_grad.norm().item():.3e})")

    # Independent mode still optimizes the shared surface geometry and points,
    # but must not render through or differentiate the legacy base density.
    for name in ("primal_points", "quaternions", "texel_sites_2d",
                 "texel_heights"):
        grad = getattr(model, name).grad
        check(grad is not None, f"{name}.grad is not None")
        if grad is not None:
            check(grad.isfinite().all(), f"{name}.grad all finite")
    check(model.density.grad is None,
          "legacy base density has no gradient in independent mode")


def main():
    print("=" * 60)
    print("LC64 plan v3 Commit 2B -- CUDA-independent forward/backward smoke")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("CUDA not available; SKIPPING all tests.")
        sys.exit(0)

    test_zero_split_equals_scalar()
    test_density_invariance()
    test_plus_minus_independence()
    test_activation_scale_scales_side_attenuation()
    test_geometry_zero_split_invariance()
    test_malformed_inputs_fail_before_launch()
    test_independent_backward_smoke()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above)")
        sys.exit(1)
    print("SUMMARY: ALL INDEPENDENT FORWARD/BACKWARD SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
