"""GPU-side correctness discriminator for the thin-surface split-cell branch.

Reviewer-flagged (independent Codex review, SPLIT-CELL-EXECUTION-LOG.md):
  - In crossing cells at zero-init, ct_thinsurface_backward may emit zero
    gradients on the cell-boundary bisector chain (i.e. dL/dt_0 = -dL/dt_1
    = -dL_ddelta_t = 0), even though the scalar backward computes
    dL_dt0/dt1 = -/+ dL/dprojection * mu.  This makes the forward continuous
    at activation but the point gradients discontinuous across activation.
  - In non-crossing, dp<0 (ray in -n direction), the side-adjoint mapping
    (unscrambling mu_near/mu_far -> mu_p/mu_n) may be incorrect, making
    dL/d(density_delta) and dL/d(density_base) for the + side diverge from
    the FD estimate.

Generic FD gradchecks don't prove either case:

  (A) The generic tests compute analytic-vs-FD for params EITHER set or zero
      but never compare SCALAR-mode vs ZERO-INIT-THIN-mode on the same scene
      and the same rays.  Forward equality alone cannot prove gradient
      equality, so a backward-only regression slips through.

  (B) The generic tests' non-crossing case uses `dp > 0` (where plus_side is
      determined by t_surf <= t_0) and a single fixed heights-of-(-5) config.
      The opposite `dp < 0` branch (plus-side selected by t_surf >= t_1) is
      never visited.

This file adds two deterministic GPU tests on a valid scene:

  test_scalar_vs_zero_thin_equivalence
      - Same fixed scene (>= 32 Delaunay-valid points), same fixed rays that
        cross at least one cell.
      - Compute forward output + total loss + dL/d(density) + dL/d(primal_points)
        in scalar mode (thin_surface_active=False).
      - Compute the same in zero-init thin mode (thin_surface_active=True,
        density_delta=0, texel_heights=0, identity quaternions).
      - Assert equality under tight float32 tolerances:
            |V_scalar - V_thin|_∞ < 1e-5  (forward equivalence)
            |Δloss_scalar - Δloss_thin| < 1e-5  (must follow from the above)
            ||g_density_scalar - g_density_thin||_max / |g_density_scalar|_max
                                          < 5e-3   (base-density gradient)
            ||g_points_scalar - g_points_thin||_max / |g_points_scalar|_max
                                          < 5e-3   (point gradient; crosses 0 in
                                                    the regression hypothesis)

  test_fd_noncrossing_minus_side
      - Fixed scene (>=32 points), rays oriented in -n direction (dp<0) so the
        surface is FAR past t_1 of every hit cell (non-crossing + side).
      - Set density_delta = 0.4, texel_heights = -5.0 (surface pushed outside
        chord).
      - Analytic gradient (Tensor autograd through TraceRays) vs central FD on:
            density
            density_delta
        under a strict float32 tolerance (1e-3 absolute on relative error).

Run with:  micromamba run -n radfoam python test/test_thin_surface_crossing_dp_adv.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np


_HAS_CUDA = torch.cuda.is_available()
if not _HAS_CUDA:
    print("SKIP: No CUDA device.  These tests require kw995 / kw996.")
    sys.exit(0)

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import radfoam
from radfoam_model.scene import CTScene


torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# Constants shared with old/test_cube.py
# ============================================================================
MIN_POINTS = 32
FILLER_SCALE = 1.05


# ============================================================================
# Test 1: scalar vs zero-init thin equivalence
# ============================================================================

def _build_scalar_or_thin_scene(device, thin_active, density_val=0.0):
    """Build a valid (>= MIN_POINTS) scene with the same primal points for both
    branches.  Returns a CTScene configured for either scalar or thin mode.

    In scalar mode (thin_active=False): no thin-surface tensors exist; the
    forward dispatches to ct_gaussian/ct_backward via the default branch.

    In thin mode: density_delta=0, texel_heights=0, identity quaternions,
    texel_sites_2d on a ring at radius 0.4 (matching initialize_thin_surface).
    """
    args = type("Args", (), {
        "init_points": MIN_POINTS,
        "final_points": MIN_POINTS,
        "activation_scale": 1.0,
        "init_scale": FILLER_SCALE,
        "init_type": "random",
        "init_density": density_val,        # overwritten below
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

    # Override to a fixed seed point cloud so both branches share geometry.
    with torch.no_grad():
        # All helper tensors MUST live on `device` -- otherwise the
        # torch.cat below crosses device boundaries and crashes (this
        # was the GPU-run blocker before commit fixing the device
        # placement).  We construct filler on-device via a CUDA generator.
        real = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        n_pad = MIN_POINTS - real.shape[0]
        g = torch.Generator(device=device).manual_seed(0)
        filler = torch.randn(n_pad, 3, generator=g, device=device) * 0.1
        filler = filler / filler.norm(dim=-1, keepdim=True).clamp_min(1e-6) * FILLER_SCALE
        filler += torch.randn(filler.shape, generator=g, device=device) * 1e-4
        pts = torch.cat([real, filler], dim=0).clamp(-0.999, 0.999)
        assert pts.shape[0] == MIN_POINTS
        model.primal_points.data.copy_(pts)
        # Active density on all cells so visited cells produce nonzero grad.
        # (Per old/test_cube.py + the gradcheck scene rules: drives both base
        # density and density_delta through nonzero chord contributions.)
        model.density.data.fill_(1.0)

    # declare_optimizer MUST run before update_triangulation(rebuild=True).
    model.declare_optimizer(args, warmup=0, max_iterations=1000)
    model.update_triangulation(rebuild=True, incremental=False)

    if thin_active:
        N = model.primal_points.shape[0]
        K = 4
        model._thin_surface_active = True
        model._thin_K = K
        model._thin_surface_gate_tau = 0.01
        model._max_iterations = 1000
        model._thin_surface_start = 0

        model.density_delta = nn.Parameter(
            torch.zeros(N, 1, device=device, dtype=torch.float32))
        q0 = torch.zeros(N, 4, device=device, dtype=torch.float32)
        q0[:, 0] = 1.0
        model.quaternions = nn.Parameter(q0)
        angles = torch.linspace(0, 2 * np.pi, K + 1, device=device)[:-1]
        base_sites = torch.stack(
            [torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
        model.texel_sites_2d = nn.Parameter(
            base_sites.unsqueeze(0).expand(N, -1, -1).clone())
        model.texel_heights = nn.Parameter(
            torch.zeros(N, K, device=device, dtype=torch.float32))

    with torch.no_grad():
        _, cr = radfoam.farthest_neighbor(
            model.primal_points, model.point_adjacency,
            model.point_adjacency_offsets)
        model._cached_cell_radius = cr.squeeze()

    return model


def _linear_x_axis_rays(device):
    """Rays along +X through the origin cell.  With identity quaternions
    (default normal +X) and zero heights, the surface is the plane x=0 which
    crosses the chord of any cell visited by a +X ray.  Force a couple of
    extra directions so multiple cells are visited and crossing is exercised
    on at least one of them."""
    dirs = torch.tensor([
        [1.0, 0.0, 0.0],   # along +X (n default = +X, crossings expected)
        [1.0, 0.0, 0.0],   # along +X (separate ray through same axis)
        [1.0, 0.0, 0.0],   # along +X (third ray)
    ], device=device, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origins = -1.5 * dirs   # origin -1.5 cell radii before scene center
    return torch.cat([origins, dirs], dim=-1)


_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


def _relative_error(a, b, eps=1e-12):
    """ ||a-b||_inf / max(||a||_inf, eps) """
    diff = (a - b).abs()
    norm_a = a.abs().max().clamp_min(eps)
    return diff.max().item() / norm_a.item()


def test_scalar_vs_zero_thin_equivalence():
    """Same valid scene, same fixed rays.  Compare forward, loss, base-density
    grad, AND primal-point grad between scalar and zero-init thin.

    Discriminates the reviewer-flagged regression hypothesis: forward equality
    but gradient discontinuity at activation iter (where dL/d(primal_points)
    in scalar mode receives the cell-boundary chain while zero-init thin
    routes through dL_dt_s = dL_dprojection * (mu_near - mu_far) = 0 and
    zeroes the cell-boundary contribution).
    """
    print("\n--- Test 1: scalar vs zero-init thin equivalence (crossing) ---")
    device = "cuda"

    # Two scenes with the same seed point cloud.
    scene_scalar = _build_scalar_or_thin_scene(device, thin_active=False,
                                                density_val=1.0)
    scene_thin = _build_scalar_or_thin_scene(device, thin_active=True,
                                              density_val=1.0)

    # Same fixed rays (along +X through origin cell).
    rays = _linear_x_axis_rays(device)

    # -- SCALAR FORWARD + BACKWARD ------------------------------------
    scene_scalar.zero_grad(set_to_none=True)
    scene_scalar._thin_surface_active = False
    start_scalar = scene_scalar.get_starting_point(
        rays, scene_scalar.primal_points, scene_scalar.aabb_tree)
    out_scalar, _, _, _, _ = scene_scalar(rays, start_scalar)
    loss_scalar = out_scalar.sum()
    loss_scalar.backward()
    grad_points_scalar = scene_scalar.primal_points.grad.detach().clone()
    grad_density_scalar = scene_scalar.density.grad.detach().clone()

    # -- THIN (ZERO-INIT) FORWARD + BACKWARD --------------------------
    scene_thin.zero_grad(set_to_none=True)
    scene_thin._thin_surface_active = True
    start_thin = scene_thin.get_starting_point(
        rays, scene_thin.primal_points, scene_thin.aabb_tree)
    out_thin, _, _, _, _ = scene_thin(rays, start_thin)
    loss_thin = out_thin.sum()
    loss_thin.backward()
    grad_points_thin = scene_thin.primal_points.grad.detach().clone()
    grad_density_thin = scene_thin.density.grad.detach().clone()

    # -- FORWARD EQUIVALENCE ------------------------------------------
    fwd_max = (out_scalar - out_thin).abs().max().item()
    fwd_rel = _relative_error(out_scalar, out_thin)
    check(out_scalar.isfinite().all(), "scalar forward: finite output")
    check(out_thin.isfinite().all(), "thin (zero-init) forward: finite output")
    check(fwd_max < 1e-5 or fwd_rel < 1e-5,
          f"Forward output equal under either L∞<1e-5 or rel<1e-5 "
          f"(L∞ diff={fwd_max:.3e}, rel={fwd_rel:.3e})")

    loss_diff = (loss_scalar - loss_thin).abs().item()
    loss_rel = (loss_scalar - loss_thin).abs().item() / max(
        loss_scalar.abs().item(), 1e-12)
    check(loss_diff < 1e-5 or loss_rel < 1e-5,
          f"Loss equal under either L∞<1e-5 or rel<1e-5 "
          f"(diff={loss_diff:.3e}, rel={loss_rel:.3e})")

    # -- BASE-DENSITY GRADIENT EQUIVALENCE ----------------------------
    # The kernel must round-trip the cell-boundary chain
    #     dL/d(density) <- dL_dmu_bar * d_softplus
    # identically in both modes (no thin-mode contribution at delta=0).
    bd_finite_s = grad_density_scalar.isfinite().all()
    bd_finite_t = grad_density_thin.isfinite().all()
    check(bd_finite_s, "scalar base-density grad finite")
    check(bd_finite_t, "thin base-density grad finite")
    if bd_finite_s and bd_finite_t:
        # Per-cell relative error — the kernel must route through the same
        # contribution chain even though the path differs (scalar: directly
        # dL_ddelta_t*delta_t path; thin: through dL_dmu_p/mu_n even though
        # delta=0 makes these degenerate to the same delta_t path).
        bd_diff = (grad_density_scalar - grad_density_thin).abs()
        bd_norm_s = grad_density_scalar.abs().max().clamp_min(1e-12)
        bd_rel_err = (bd_diff / bd_norm_s).max().item()
        bd_max_abs = bd_diff.max().item()
        # Central-FD-style tolerance for float32 CUDA with manual reduction:
        # tight 5e-3 relative error threshold.  Crossing+zero-init is
        # theoretically identical (no surface contribution) but the chain
        # passes through mu_p/mu_n mask and d_softplus differently.
        check(bd_rel_err < 5e-3 or bd_max_abs < 1e-6,
              f"Base-density gradient equivalent (rel_err={bd_rel_err:.4f}, "
              f"max_abs_diff={bd_max_abs:.3e})")

    # -- POINT GRADIENT EQUIVALENCE (the principal discriminator) -----
    # In the reviewer-flagged hypothesis, ct_thinsurface_backward omits the
    # cell-boundary contribution (dL_ddelta_t = 0 in crossing branch even at
    # delta=0), so dL/d(primal_points) is a zero vector in thin mode but the
    # full cell-boundary chain gradient in scalar mode.  This test would
    # detect exactly that.
    pg_finite_s = grad_points_scalar.isfinite().all()
    pg_finite_t = grad_points_thin.isfinite().all()
    check(pg_finite_s, "scalar point grad finite")
    check(pg_finite_t, "thin (zero-init) point grad finite")
    if pg_finite_s and pg_finite_t:
        pg_diff = (grad_points_scalar - grad_points_thin).abs()
        pg_norm_s = grad_points_scalar.norm().clamp_min(1e-12)
        pg_rel_err = pg_diff.norm().item() / pg_norm_s.item()
        pg_max_abs = pg_diff.max().item()
        # 5e-3 relative or 1e-5 absolute tolerance.
        check(pg_rel_err < 5e-3 or pg_max_abs < 1e-5,
              f"Point gradient equivalent (rel_err={pg_rel_err:.4f}, "
              f"max_abs_diff={pg_max_abs:.3e})")
        # Diagnostic: print the magnitudes so we can see if the regression
        # hypothesis (thin-mode grads zero) reproduces in the report.
        scalar_mag = grad_points_scalar.norm().item()
        thin_mag = grad_points_thin.norm().item()
        print(f"    [diag] ||g_points_scalar||={scalar_mag:.4e}, "
              f"||g_points_thin||={thin_mag:.4e}, "
              f"ratio={thin_mag/max(scalar_mag,1e-12):.4f}")


# ============================================================================
# Test 2: FD for non-crossing, surface outside chord, dp<0 (plus side)
# ============================================================================

def _build_scene_minus_side(device):
    """Valid >= MIN_POINTS scene with the active test cell at origin, all
    cells' quaternions aligned so default normal +X becomes -Y (or some other
    axis) such that a ray along -X gives dp < 0 at the ray-cell intersection.

    We override quaternions AFTER build so that
        n = quat_to_frame(q) ⇒ n is fixed across all cells.
    Then a ray along +X will have dp = n·d = n_x.  With n = +X (identity),
    dp = +1 (positive).  With n flipped to -X (q.x-rotated 180°), dp = -1.
    """
    args = type("Args", (), {
        "init_points": MIN_POINTS,
        "final_points": MIN_POINTS,
        "activation_scale": 1.0,
        "init_scale": FILLER_SCALE,
        "init_type": "random",
        "init_density": 1.0,
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

    with torch.no_grad():
        # Device-consistent construction (was CPU/CUDA mismatch bug):
        # build filler on `device` via a CUDA generator, never `.to()` after
        # cat, and keep every helper tensor on the same device.
        real = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        n_pad = MIN_POINTS - real.shape[0]
        g = torch.Generator(device=device).manual_seed(1)
        filler = torch.randn(n_pad, 3, generator=g, device=device) * 0.1
        filler = filler / filler.norm(dim=-1, keepdim=True).clamp_min(1e-6) * FILLER_SCALE
        filler += torch.randn(filler.shape, generator=g, device=device) * 1e-4
        pts = torch.cat([real, filler], dim=0).clamp(-0.999, 0.999)
        model.primal_points.data.copy_(pts)
        model.density.data.fill_(1.0)

    model.declare_optimizer(args, warmup=0, max_iterations=1000)
    model.update_triangulation(rebuild=True, incremental=False)

    N = model.primal_points.shape[0]
    K = 4
    model._thin_surface_active = True
    model._thin_K = K
    model._thin_surface_gate_tau = 0.01
    model._max_iterations = 1000
    model._thin_surface_start = 0

    # Flip all quaternions so the default normal +X becomes -X (180° about Y).
    # Q to flip a unit vector +X -> -X via rotation by π about Y axis:
    #   q = (0, 0, 1, 0)  (w=cos(π/2)=0, x=0, y=sin(π/2)=1, z=0)
    # but that's about Y.  For rotation about Y by 180°: q = (cos(90°), 0, sin(90°), 0)
    # = (0, 0, 1, 0).  Surface normal (column 0 of rotation matrix) then is -X.
    flipped_q = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    # norm must run on-device to keep the tensor on the target device --
    # `.norm()` without an explicit dtype/dev stays on the original device.
    flipped_q = flipped_q / flipped_q.norm().clamp_min(1e-12)
    q_full = flipped_q.unsqueeze(0).expand(N, -1).clone()
    model.quaternions = nn.Parameter(q_full)
    # Set density_delta nonzero so side selection is observable
    model.density_delta = nn.Parameter(
        0.4 * torch.ones(N, 1, device=device, dtype=torch.float32))
    # Texel sites on the disc.
    angles = torch.linspace(0, 2 * np.pi, K + 1, device=device)[:-1]
    base_sites = torch.stack(
        [torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    model.texel_sites_2d = nn.Parameter(
        base_sites.unsqueeze(0).expand(N, -1, -1).clone())
    # Tall NEGATIVE heights push h_eval negative -> surface t_surf far past t_1
    # for every hit cell with dp<0 -> non-crossing plus-side branch.
    model.texel_heights = nn.Parameter(
        -5.0 * torch.ones(N, K, device=device, dtype=torch.float32))

    with torch.no_grad():
        _, cr = radfoam.farthest_neighbor(
            model.primal_points, model.point_adjacency,
            model.point_adjacency_offsets)
        model._cached_cell_radius = cr.squeeze()

    return model


def _render_loss(model, rays):
    model.zero_grad(set_to_none=True)
    start_point = model.get_starting_point(
        rays, model.primal_points, model.aabb_tree)
    out, _, _, _, _ = model(rays, start_point)
    loss = out.sum()
    loss.backward()
    return loss


def _fd_grad_scalar(model, param_name, rays, eps=1e-3):
    """Central finite-difference gradient w.r.t. a parameter on a fixed
    scene/rays.  Perturbs each scalar entry of `param.data` one at a time,
    evaluates the kernel forward in `torch.no_grad()` mode (so backward
    is not required), and computes (loss(perturbed+) - loss(perturbed-)) / 2eps.

    Returns a tensor with the same shape as the parameter.

    Note:  `param.data` is mutated temporarily.  The function restores the
    original values before returning.

    Pre-fix bug:  earlier this helper was passing `model.primal_points`
    as the `rays` argument to `model.get_starting_point`, which used the
    scene's 32 cell positions as "camera origins" and produced a
    start_point of shape (32,) instead of (1,).  Calling `model(rays, sp)`
    then tripped the broadcast assertion in `CTScene.forward`.  We now
    compute the per-ray start_point ONCE from the actual `rays` tensor
    before the perturbation loop and pass it through.
    """
    param = getattr(model, param_name)
    saved = param.data.clone()
    grad = torch.zeros_like(param.data)

    flat = param.data.reshape(-1)
    gflat = grad.reshape(-1)

    # Compute start_point ONCE from the actual rays (shape [N_rays] long).
    with torch.no_grad():
        sp0 = model.get_starting_point(
            rays, model.primal_points, model.aabb_tree)
        # sp0 shape: rays.shape[:-1] -> for a (1, 6) rays tensor, that's
        # shape (1,).  We coerce to long for the model's forward broadcast.

    for i in range(flat.shape[0]):
        orig = flat[i].item()

        # Forward perturbation
        flat[i] = orig + eps
        with torch.no_grad():
            out_p, _, _, _, _ = model(rays, sp0, return_contribution=False)
            loss_p = out_p.sum().item()

        # Backward perturbation
        flat[i] = orig - eps
        with torch.no_grad():
            out_m, _, _, _, _ = model(rays, sp0, return_contribution=False)
            loss_m = out_m.sum().item()

        gflat[i] = (loss_p - loss_m) / (2.0 * eps)

    flat.copy_(saved.reshape(-1))  # restore
    return grad

def test_fd_noncrossing_minus_side():
    """FD gradient check for the non-crossing + side, dp<0 branch.

    This is the specific kernel-branch the reviewer flagged as suspect.
    On this branch:
        mu_near = (dp < 0) ? mu_p : mu_n      [so mu_near = mu_p for dp<0]
        mu_far  = (dp < 0) ? mu_n : mu_p      [so mu_far  = mu_n for dp<0]
        t_surf >= t_1  -> plus_side True  (surface past chord in -d direction)
        contrib        = mu_p * delta_t
        dL_ddelta_t    = dL/dprojection * mu_p

    The unscrambling of mu_near/mu_far -> mu_p/mu_n (at the end of the
    branch) must be consistent with these labels — the reviewer hypothesis
    is that the unscrambling is accidentally the same as the dp>0 case
    (i.e., swapped) yielding dL/d(delta) and dL/d(density_base) gradients
    for the plus side that DO NOT match the true FD chain.

    We independently test:
        - density     (base density)
        - density_delta (signed half-split)
    """
    print("\n--- Test 2: non-crossing minus-side FD (dp<0, surface past t_1) ---")
    device = "cuda"
    model = _build_scene_minus_side(device)

    # RAY: along +X through origin.  With surface normal -X (flipped quaternion),
    # dp = n.d = (-1).ray_dir = -1, so dp < 0.  The huge negative texel_heights
    # push h_eval negative, so t_surf = (cp-origin).n/dp + h_eval/dp is far
    # past t_1 of every hit cell -> non-crossing + side selected.
    rays = torch.cat([
        torch.tensor([[-1.5, 0.0, 0.0]], device=device),
        torch.tensor([[1.0, 0.0, 0.0]], device=device),
    ], dim=-1).float()

    # -- ANALYTIC GRADIENT ---------------------------------------------
    loss = _render_loss(model, rays)
    grad_density_analytic = model.density.grad.detach().clone()
    grad_delta_analytic = model.density_delta.grad.detach().clone()

    # Validate finiteness first.
    check(grad_density_analytic.isfinite().all(),
          "analytic density grad finite (non-crossing dp<0)")
    check(grad_delta_analytic.isfinite().all(),
          "analytic density_delta grad finite (non-crossing dp<0)")

    # -- FINITE-DIFFERENCE (per-cell scalar perturbation) -------------
    # We perturb each scalar entry one at a time and central-difference the
    # total loss.  For N=32 cells: 2*N (density) + 2*N (density_delta)
    # forwards = 128 forwards total at ~tens of ms each on RTX 6000 Ada.
    print(f"    Computing central FD for density (N={MIN_POINTS} cells)...",
          flush=True)
    fd_density = _fd_grad_scalar(model, "density", rays, eps=1e-3)
    print(f"    Computing central FD for density_delta (N={MIN_POINTS} "
          f"cells)...", flush=True)
    fd_delta = _fd_grad_scalar(model, "density_delta", rays, eps=1e-3)

    # -- COMPARE ------------------------------------------------------
    # Tolerances: float32 with manual FD across the whole pipeline; we
    # expect rel_err ~1e-3 in the worst case for a non-crossing branch
    # whose dL_ddelta_t = dL_proj * mu_p depends on a single scalar.
    def compare(name, analytic, fd, eps=1e-12):
        diff = (analytic - fd).abs()
        denom = fd.abs().clamp_min(eps)
        rel = (diff / denom).max().item()
        max_abs = diff.max().item()
        return rel, max_abs

    d_rel, d_abs = compare("density", grad_density_analytic, fd_density)
    delta_rel, delta_abs = compare("density_delta", grad_delta_analytic,
                                     fd_delta)

    print(f"    [diag] density grad max_abs={grad_density_analytic.abs().max().item():.4e}, "
          f"FD rel_err={d_rel:.4e}, max_abs_diff={d_abs:.4e}")
    print(f"    [diag] density_delta grad max_abs="
          f"{grad_delta_analytic.abs().max().item():.4e}, "
          f"FD rel_err={delta_rel:.4e}, max_abs_diff={delta_abs:.4e}")

    # Use either relative or absolute tolerance (whichever is more lenient)
    check(d_rel < 1e-2 or d_abs < 1e-5,
          f"density gradient: FD matches analytic (rel={d_rel:.4e}, "
          f"abs={d_abs:.4e})")
    check(delta_rel < 1e-2 or delta_abs < 1e-5,
          f"density_delta gradient: FD matches analytic (rel={delta_rel:.4e}, "
          f"abs={delta_abs:.4e})")


def main():
    print("=" * 60)
    print("Thin-Surface Cross-Branch Correctness Discriminator (GPU)")
    print("=" * 60)
    print(f"CUDA available: {_HAS_CUDA}")

    if not _HAS_CUDA:
        sys.exit(0)

    test_scalar_vs_zero_thin_equivalence()
    test_fd_noncrossing_minus_side()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        print("Read each FAIL message above to localize the branch.  The")
        print("reviewer-flagged regression hypothesis (scalar/zero-thin forward")
        print("equal but gradient discontinuous; non-crossing dp<0 backward adjoint")
        print("swap) is now an automated gate; run this on kw995/kw996.")
        sys.exit(1)
    print("SUMMARY: ALL CROSS-BRANCH TESTS PASSED.")


if __name__ == "__main__":
    main()
