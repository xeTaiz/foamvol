"""GPU finite-difference gradient checks for the M5 relative-delta parameterization.

Scope (focused per SPLIT-CELL-EXECUTION-LOG.md M5 -> kw995 pre-CH8 gate):

  - Base-density gradient (dL/d(density)) analytic vs central FD
  - Raw density_delta gradient (dL/d(density_delta)) analytic vs central FD

Both checked in TWO configurations:
  - crossing:    surface intersects the ray chord (delta_val != 0, nonzero
                 t_surf between t_0 and t_1)  -> exercises the d(delta)/d(mu_bar)
                 and sech^2(raw_delta) chains through mu_p / mu_n.
  - non-crossing: surface plane pushed outside the chord (large |heights|) so
                 the kernel falls back to contrib = mu_eff * delta_t.  In
                 relative mode mu_eff = mu_p or mu_n still depends on raw_delta
                 through delta_val = rho * mu_bar * tanh(raw_delta), so both
                 density and raw_delta grads are nonzero -- this is a strict
                 check that the relative branch is being executed (not just
                 the absolute one with rho applied later).

Plus activation forward continuity at init: at raw_delta = 0 the relative
formula collapses to delta_val = 0 (tanh(0)=0), so the forward projection
must equal the absolute-mode projection bit-for-bit.  This is the safety
property that lets training start at the same PSNR as the scalar baseline.

Configuration: rho = 0.5 (matches best428_thinsurface_relative.yaml staging
config and the chest-rescue execution plan).

Setup: reuses the >=32-point padded-scene layout from
test_thin_surface_gradcheck.py so the relative-mode FD results are directly
comparable to the absolute-mode gradchecks that already gate CH4-CH7.

Run with:  micromamba run -n radfoam python test/test_thin_surface_relative_delta_gradcheck.py
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np


# Graceful skip if CUDA is unavailable -- must come before radfoam import.
_HAS_CUDA = torch.cuda.is_available()
if not _HAS_CUDA:
    print("SKIP: No CUDA device. Relative-delta gradcheck tests require GPU.")
    print("Run on: kw995 or kw996 (RTX 6000 Ada)")
    sys.exit(0)


# Repo root on path (matches sibling test files).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import radfoam
from radfoam_model.scene import CTScene


torch.manual_seed(42)
np.random.seed(42)


# FD settings (matches test_thin_surface_gradcheck.py)
EPS = 1e-3               # central-difference step size
TOL_FD = 5e-2             # relative tolerance for FD check (loose, float32 CUDA)
RHO = 0.5                 # M5 chest-rescue rho (per task / staging config)


# 32-point minimum for valid Delaunay triangulation (mirrors
# test_thin_surface_gradcheck.py so results are directly comparable).
MIN_POINTS = 32
_FILLER_SCALE = 1.05


def _make_1cell_scene(device="cuda"):
    """Build a minimal valid CTScene (>= MIN_POINTS) with thin-surface params
    in RELATIVE mode at rho=RHO.

    Reuses the layout from test_thin_surface_gradcheck._make_1cell_scene so
    the FD results here are directly comparable to the absolute-mode
    gradchecks.  Only the relative flag and rho are added.
    """
    args = type("Args", (), {
        "init_points": MIN_POINTS,
        "final_points": MIN_POINTS,
        "activation_scale": 1.0,
        "init_scale": _FILLER_SCALE,
        "init_type": "random",
        "init_density": 1.0,     # nonzero starting density for the test cell
        "device": device,
        "init_points_file": "",
        "init_volume_path": "",
        "frozen_points_file": "",
        "frozen_freeze_density": True,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-3,
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "freeze_points": 9500,
    })()

    model = CTScene(args, device=torch.device(device))

    # Build the point set: 1 real cell at the origin + (MIN_POINTS-1) fillers
    # on a shell.  All cells keep the active init density (1.0) so that the
    # cells the ray visits produce nonzero base-density / density_delta grads,
    # making the FD check nontrivial and independent of the triangulation
    # permutation (see test_thin_surface_gradcheck.py for the rationale).
    with torch.no_grad():
        real = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        n_pad = MIN_POINTS - real.shape[0]
        filler = torch.randn(n_pad, 3, device=device) * 0.1
        filler = filler / filler.norm(dim=-1, keepdim=True).clamp_min(1e-6) * _FILLER_SCALE
        filler += torch.randn_like(filler) * 1e-4
        pts = torch.cat([real, filler], dim=0).clamp(-0.999, 0.999)
        assert pts.shape[0] == MIN_POINTS
        model.primal_points.data.copy_(pts)

    # declare_optimizer must run before update_triangulation(rebuild=True):
    # rebuild may trigger a triangulation permutation, and permute_points()
    # walks self.optimizer.param_groups -- without an optimizer the scene
    # build raises AttributeError.  Mirrors train.py's order.
    model.declare_optimizer(args, warmup=0, max_iterations=1000)
    model.update_triangulation(rebuild=True, incremental=False)

    # Manually register thin-surface params (simulating initialize_thin_surface),
    # with RELATIVE mode enabled at rho=RHO.
    N = model.primal_points.shape[0]  # == MIN_POINTS
    model._thin_surface_active = True
    model._thin_K = 4
    model._thin_surface_gate_tau = 0.01
    model._max_iterations = 1000
    model._thin_surface_start = 0
    model._thin_surface_relative_delta = True
    model._thin_surface_delta_max_frac = RHO

    model.density_delta = nn.Parameter(torch.zeros(N, 1, device=device))
    q0 = torch.zeros(N, 4, device=device)
    q0[:, 0] = 1.0
    model.quaternions = nn.Parameter(q0)
    K = 4
    angles = torch.linspace(0, 2 * np.pi, K + 1, device=device)[:-1]
    base_sites = torch.stack([torch.cos(angles) * 0.4,
                              torch.sin(angles) * 0.4], dim=-1)
    model.texel_sites_2d = nn.Parameter(
        base_sites.unsqueeze(0).expand(N, -1, -1).clone())
    model.texel_heights = nn.Parameter(torch.zeros(N, K, device=device))

    with torch.no_grad():
        _, cr = radfoam.farthest_neighbor(
            model.primal_points, model.point_adjacency,
            model.point_adjacency_offsets,
        )
        model._cached_cell_radius = cr.squeeze()

    return model


def _make_single_ray(device="cuda", origin=None, direction=None):
    """Single ray.  Default: along +X through the origin from (-2, 0, 0)."""
    if origin is None:
        origin = torch.tensor([[-2.0, 0.0, 0.0]], device=device)
    if direction is None:
        direction = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    return torch.cat([origin, direction], dim=-1)  # (1, 6)


def _clone_scene(base_model):
    """Clone a CTScene for use with a different thin-surface parameterization.

    The cloned scene has its own nn.Module state (so zero_grad() etc. track
    grads on its own parameters) but shares the C++ Triangulation, AABB tree,
    and CSR adjacency buffers with the source -- those depend only on
    primal_points and are immutable for forward.  All learnable parameters
    are detached-cloned so the clone is autograd-independent.

    This is the only way to make an absolute-vs-relative forward comparison
    bit-identical on scene inputs: two independent calls to _make_1cell_scene
    produce two different randomized scenes (different RNG draws during
    random_initialize, different filler shells, possibly different Delaunay
    vertex orderings in the C++ binding).  Even if the *kernel* is bit-
    identical at init, two different scenes trivially produce two different
    projections -- a test artifact, not a kernel bug.
    """
    new_model = object.__new__(CTScene)
    nn.Module.__init__(new_model)

    # Scalar / config attrs (all copied verbatim; the clone starts in the
    # same mode as the base, callers can override the parameterization flags).
    new_model.activation_scale = base_model.activation_scale
    new_model.device = base_model.device
    new_model.num_init_points = base_model.num_init_points
    new_model.num_final_points = base_model.num_final_points
    new_model._thin_surface_active = base_model._thin_surface_active
    new_model._thin_K = base_model._thin_K
    new_model._thin_surface_gate_tau = base_model._thin_surface_gate_tau
    new_model._thin_surface_start = base_model._thin_surface_start
    new_model._max_iterations = base_model._max_iterations
    new_model._thin_surface_relative_delta = base_model._thin_surface_relative_delta
    new_model._thin_surface_delta_max_frac = base_model._thin_surface_delta_max_frac
    new_model._thin_temp = getattr(base_model, "_thin_temp", 10.0)
    new_model._thin_height_eps = getattr(base_model, "_thin_height_eps", 1e-4)

    # Learnable parameters: detached-cloned so each scene has its own
    # nn.Parameter (independent autograd graph + independent .grad slot).
    new_model.primal_points = nn.Parameter(base_model.primal_points.detach().clone())
    new_model.density = nn.Parameter(base_model.density.detach().clone())
    new_model.density_delta = nn.Parameter(base_model.density_delta.detach().clone())
    new_model.quaternions = nn.Parameter(base_model.quaternions.detach().clone())
    new_model.texel_sites_2d = nn.Parameter(base_model.texel_sites_2d.detach().clone())
    new_model.texel_heights = nn.Parameter(base_model.texel_heights.detach().clone())

    # Shared immutable C++/CSR buffers (depend only on primal_points, which
    # is the same tensor values on both scenes -- we just clone it).
    new_model.triangulation = base_model.triangulation
    new_model.aabb_tree = base_model.aabb_tree
    new_model.point_adjacency = base_model.point_adjacency
    new_model.point_adjacency_offsets = base_model.point_adjacency_offsets
    new_model._cached_cell_radius = base_model._cached_cell_radius
    new_model.pipeline = base_model.pipeline

    # Forward-only scenes don't need an optimizer; leave it None so any
    # accidental .zero_grad() / .backward() on the clone can't mutate the
    # base scene's optimizer state.
    new_model.optimizer = None

    return new_model


def _render_with_grad(model, rays, mode="relative", rho=RHO):
    """Run a forward pass with the requested parameterization and return
    the scalar loss = sum(proj) used by the FD check.

    `mode='relative'` engages the kernel branch
      delta_val = rho * mu_bar * tanh(raw_delta)
    while `mode='absolute'` engages the legacy branch
      delta_val = raw_delta
    so the same scene can be exercised in either mode.  The flag plumbing
    goes via CTScene.forward() -> TraceRays.apply -> pipeline.trace_forward,
    so the on-device kernel sees the same settings the FD estimate uses.
    """
    model._thin_surface_active = True
    model._thin_surface_relative_delta = (mode == "relative")
    model._thin_surface_delta_max_frac = float(rho)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree
    )
    out, _, _, _, _ = model(rays, start_point)
    return out.sum()


def _fd_grad(model, param_name, loss_fn, eps=EPS):
    """Central finite-difference gradient estimator.  Returns dL/dp with
    the same shape as the parameter; loss_fn() should return a scalar loss
    (no grad tracking needed, since this is the analytic-side estimate)."""
    param = getattr(model, param_name)
    flat = param.data.flatten()
    grad_flat = torch.zeros_like(flat)

    for i in range(flat.shape[0]):
        orig = flat[i].item()

        flat[i] = orig + eps
        loss_plus = loss_fn().detach()

        flat[i] = orig - eps
        loss_minus = loss_fn().detach()

        grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)
        flat[i] = orig  # restore

    return grad_flat.reshape(param.shape)


def _check_fd(param_name, model, loss_fn, desc="", tol=TOL_FD):
    """Compare analytic gradient vs FD for a single parameter.  Returns
    True on pass.  Mirrors test_thin_surface_gradcheck._check_fd."""
    param = getattr(model, param_name)
    if param.grad is None:
        print(f"  [{param_name}] FAIL: analytic grad is None (shape mismatch?)")
        return False

    analytic = param.grad.detach().clone()
    if not analytic.isfinite().all():
        print(f"  [{param_name}] FAIL: analytic grad has NaN/Inf values")
        return False

    fd = _fd_grad(model, param_name, loss_fn, eps=EPS)

    # Shape reconciliation: FD produces param.shape; analytic might come from
    # the C++ binding with a slightly different rank (rare; defensive).
    if analytic.shape != fd.shape:
        if analytic.dim() < fd.dim() and analytic.shape == fd.shape[:-1]:
            analytic = analytic.unsqueeze(-1)
        elif fd.dim() < analytic.dim() and fd.shape == analytic.shape[:-1]:
            fd = fd.unsqueeze(-1)
        else:
            print(f"  [{param_name}] SHAPE MISMATCH: analytic={tuple(analytic.shape)}, "
                  f"fd={tuple(fd.shape)}")
            return False

    # Inert / no-effect case: if BOTH analytic and FD are near-zero, accept
    # without a relative-error comparison (a near-zero denominator makes the
    # relative error meaningless from numerical noise alone).
    fd_norm = fd.norm()
    an_norm = analytic.norm()
    ZERO_TOL = 1e-6
    if fd_norm < ZERO_TOL and an_norm < ZERO_TOL:
        print(f"  [{param_name}] PASS (inert): both analytic and FD near-zero "
              f"(norms an={an_norm:.2e}, fd={fd_norm:.2e}) [{desc}]")
        return True
    if fd_norm < ZERO_TOL:
        print(f"  [{param_name}] FAIL: FD near-zero (norm={fd_norm:.2e}) but "
              f"analytic nonzero (norm={an_norm:.2e}) [{desc}]")
        return False
    if an_norm < ZERO_TOL:
        print(f"  [{param_name}] FAIL: analytic near-zero (norm={an_norm:.2e}) "
              f"but FD nonzero (norm={fd_norm:.2e}) [{desc}]")
        return False

    diff = (analytic - fd).norm() / fd_norm
    max_abs_diff = (analytic - fd).abs().max().item()
    match = diff < tol
    if match:
        print(f"  [{param_name}] PASS: rel_err={diff:.4f} (tol={tol}), "
              f"max_abs_diff={max_abs_diff:.6f} [{desc}]")
    else:
        print(f"  [{param_name}] FAIL: rel_err={diff:.4f} (tol={tol}), "
              f"max_abs_diff={max_abs_diff:.6f} [{desc}]")
    return match


_any_failed = False


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# ---------------------------------------------------------------------------
# Test 1: activation forward continuity at init
# ---------------------------------------------------------------------------
def test_forward_continuity_at_init():
    """At raw_delta = 0 (the init value), tanh(0)=0 collapses the relative
    formula to delta_val = rho * mu_bar * 0 = 0, identical to the absolute
    branch with raw_delta = 0.  The kernel's two branches therefore produce
    bit-identical forward projections at init, so M5 training can start at
    the scalar baseline's PSNR with no discontinuity.

    Math:  abs mode: delta_val = raw_delta            -> at init: 0
           rel mode: delta_val = rho * mu_bar * tanh(raw) -> at init: 0
           both yield mu_p = mu_n = mu_bar -> contrib identical.

    Implementation: build ONE valid scene (the "base") then clone it twice
    via _clone_scene() so the absolute and relative forward passes see
    identical points/density/topology/params.  Without this, two independent
    _make_1cell_scene calls produce two randomized scenes that differ by
    enough floating-point noise in their bounds to break the 1e-7 continuity
    check purely from scene mismatch -- a test artifact, not a kernel bug.
    """
    print("\n--- Test 1: activation forward continuity at init (rho=.5) ---")
    device = "cuda"
    rays = _make_single_ray(device)

    # Single source of truth: build a valid scene once, clone twice.
    base_model = _make_1cell_scene(device)

    # Absolute reference (delta_val = raw_delta at init).  The clone starts
    # in relative mode (matching the base); flip it to absolute here.
    model_abs = _clone_scene(base_model)
    model_abs._thin_surface_relative_delta = False
    model_abs._thin_surface_delta_max_frac = 0.5
    out_abs = _render_with_grad(model_abs, rays, mode="absolute")

    # Relative at rho=.5 (delta_val = rho * mu_bar * tanh(raw)).
    model_rel = _clone_scene(base_model)
    model_rel._thin_surface_relative_delta = True
    model_rel._thin_surface_delta_max_frac = RHO
    out_rel = _render_with_grad(model_rel, rays, mode="relative", rho=RHO)

    # Sanity: confirm the two scenes really do see identical scene state.
    # If the base model mutates during forward (it shouldn't -- forward is
    # read-only), this guard catches it before we trust the equality.
    pts_match = torch.equal(
        model_abs.primal_points.detach(), model_rel.primal_points.detach()
    )
    dens_match = torch.equal(
        model_abs.density.detach(), model_rel.density.detach()
    )
    dd_match = torch.equal(
        model_abs.density_delta.detach(), model_rel.density_delta.detach()
    )
    quat_match = torch.equal(
        model_abs.quaternions.detach(), model_rel.quaternions.detach()
    )
    sites_match = torch.equal(
        model_abs.texel_sites_2d.detach(), model_rel.texel_sites_2d.detach()
    )
    heights_match = torch.equal(
        model_abs.texel_heights.detach(), model_rel.texel_heights.detach()
    )
    check(pts_match and dens_match and dd_match and quat_match and sites_match and heights_match,
          "clone parity: primal_points/density/density_delta/quaternions/sites/heights "
          "are byte-identical between absolute and relative scenes")

    # Per-element comparison: at init both paths compute delta_val = 0 exactly
    # (rho * mu_bar * tanh(0) = rho * mu_bar * 0 = 0 in IEEE-754 float32), so
    # downstream contrib computation is bit-identical.
    diff_per_elem = (out_abs - out_rel).abs()
    max_diff = diff_per_elem.max().item()
    sum_diff = diff_per_elem.sum().item()
    check(max_diff < 1e-7,
          f"per-element projection diff at init < 1e-7 "
          f"(max |P_abs - P_rel|={max_diff:.2e}, sum={sum_diff:.2e})")

    # Also confirm both branches produce a nontrivial projection (sanity:
    # the comparison is not 0-vs-0 from a degenerate scene).
    p_abs_n = out_abs.abs().sum().item()
    p_rel_n = out_rel.abs().sum().item()
    check(p_abs_n > 1e-3 and p_rel_n > 1e-3,
          f"nontrivial projection at init: |P_abs|={p_abs_n:.3e}, "
          f"|P_rel|={p_rel_n:.3e}")


# ---------------------------------------------------------------------------
# Test 2: density (base) grad in crossing configuration, rho=.5
# ---------------------------------------------------------------------------
def test_density_grad_crossing_rel():
    """Crossing case, relative mode, rho=.5.

    In crossing, the kernel computes contrib = mu_near*(t_s - t_0) + mu_far*(t_1 - t_s)
    and the relative-mode backward adds a chain through d(delta)/d(mu_bar) =
    rho * tanh(raw_delta) into the mu_bar adjoint.  So the analytic base-
    density grad is the sum of two contributions that must match FD.
    """
    print("\n--- Test 2: density grad (crossing, rho=.5) ---")
    device = "cuda"
    model = _make_1cell_scene(device)
    rays = _make_single_ray(device)
    with torch.no_grad():
        model.density_delta.data[:, 0] = 0.3    # nonzero raw -> nonzero delta_val
        model.texel_heights.data[:, 0] = 0.05   # surface intersects the chord

    model.zero_grad()
    loss = _render_with_grad(model, rays, mode="relative", rho=RHO)
    loss.backward()

    fd_ok = _check_fd(
        "density", model,
        lambda: _render_with_grad(model, rays, mode="relative", rho=RHO),
        desc="crossing rho=.5",
    )
    check(fd_ok, "density grad analytic matches FD (crossing rho=.5)")


# ---------------------------------------------------------------------------
# Test 3: density_delta (raw) grad in crossing configuration, rho=.5
# ---------------------------------------------------------------------------
def test_density_delta_grad_crossing_rel():
    """Crossing case, relative mode, rho=.5.

    Kernel: dL/d(raw_delta) = dL/d(delta) * rho * mu_bar * sech^2(raw_delta).
    At raw_delta = 0.3, sech^2(0.3) ~ 0.91 (well-conditioned multiplier),
    so this is a strict FD check of the sech^2 chain.
    """
    print("\n--- Test 3: density_delta grad (crossing, rho=.5) ---")
    device = "cuda"
    model = _make_1cell_scene(device)
    rays = _make_single_ray(device)
    with torch.no_grad():
        model.density_delta.data[:, 0] = 0.3
        model.texel_heights.data[:, 0] = 0.05

    model.zero_grad()
    loss = _render_with_grad(model, rays, mode="relative", rho=RHO)
    loss.backward()

    fd_ok = _check_fd(
        "density_delta", model,
        lambda: _render_with_grad(model, rays, mode="relative", rho=RHO),
        desc="crossing rho=.5",
    )
    check(fd_ok, "density_delta grad analytic matches FD (crossing rho=.5)")


# ---------------------------------------------------------------------------
# Test 4: density (base) grad in non-crossing configuration, rho=.5
# ---------------------------------------------------------------------------
def test_density_grad_noncrossing_rel():
    """Non-crossing case, relative mode, rho=.5.

    Surface plane pushed outside the chord by setting all texel heights to
    a large negative value (h_eval ~ -5 * r -> t_surf well before t_0).
    The kernel falls back to contrib = mu_eff * delta_t where mu_eff is one
    of {mu_p, mu_n}.  In relative mode mu_p/mu_n still depend on raw_delta
    (via delta_val = rho * mu_bar * tanh(raw_delta) != 0 for raw != 0), so
    BOTH density and density_delta grads are nonzero.  This is the strict
    check that the relative branch is being executed.
    """
    print("\n--- Test 4: density grad (non-crossing, rho=.5) ---")
    device = "cuda"
    model = _make_1cell_scene(device)
    rays = _make_single_ray(device)
    with torch.no_grad():
        model.density_delta.data[:, 0] = 0.3   # keep raw_delta nonzero
        model.texel_heights.data[:] = -5.0     # push surface outside chord

    model.zero_grad()
    loss = _render_with_grad(model, rays, mode="relative", rho=RHO)
    loss.backward()

    # Sanity: both grads must be nonzero (mu_p, mu_n differ from mu_bar for
    # raw=0.3 in relative mode, so density_delta grad can't collapse to zero).
    an_density = model.density.grad
    an_dd = model.density_delta.grad
    if an_density is not None and an_dd is not None:
        check(an_density.isfinite().all() and an_density.abs().sum().item() > 1e-6,
              f"density grad nonzero (non-crossing rho=.5; "
              f"gnorm={an_density.abs().sum().item():.3e})")
        check(an_dd.isfinite().all() and an_dd.abs().sum().item() > 1e-6,
              f"density_delta grad nonzero (non-crossing rho=.5; "
              f"gnorm={an_dd.abs().sum().item():.3e})")

    fd_ok = _check_fd(
        "density", model,
        lambda: _render_with_grad(model, rays, mode="relative", rho=RHO),
        desc="non-crossing rho=.5",
    )
    check(fd_ok, "density grad analytic matches FD (non-crossing rho=.5)")


# ---------------------------------------------------------------------------
# Test 5: density_delta (raw) grad in non-crossing configuration, rho=.5
# ---------------------------------------------------------------------------
def test_density_delta_grad_noncrossing_rel():
    """Non-crossing case, relative mode, rho=.5.  Companion to test 4 for
    the density_delta grad: must match FD as well, confirming the
    d/d(raw) = rho * mu_bar * sech^2(raw) chain is correctly applied in the
    non-crossing branch (where dL/dt_s = 0 but mu_eff still depends on raw)."""
    print("\n--- Test 5: density_delta grad (non-crossing, rho=.5) ---")
    device = "cuda"
    model = _make_1cell_scene(device)
    rays = _make_single_ray(device)
    with torch.no_grad():
        model.density_delta.data[:, 0] = 0.3
        model.texel_heights.data[:] = -5.0

    model.zero_grad()
    loss = _render_with_grad(model, rays, mode="relative", rho=RHO)
    loss.backward()

    fd_ok = _check_fd(
        "density_delta", model,
        lambda: _render_with_grad(model, rays, mode="relative", rho=RHO),
        desc="non-crossing rho=.5",
    )
    check(fd_ok, "density_delta grad analytic matches FD (non-crossing rho=.5)")


def main():
    print("=" * 60)
    print("Thin-Surface Relative-Delta GPU Gradcheck (rho=.5)")
    print("=" * 60)

    test_forward_continuity_at_init()
    test_density_grad_crossing_rel()
    test_density_delta_grad_crossing_rel()
    test_density_grad_noncrossing_rel()
    test_density_delta_grad_noncrossing_rel()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above)")
        sys.exit(1)
    else:
        print("SUMMARY: ALL RELATIVE-DELTA GRADCHECKS PASSED")


if __name__ == "__main__":
    main()
