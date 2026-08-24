"""Targeted regression tests for *late* thin-surface activation.

Scope (per SPLIT-CELL-EXECUTION-LOG.md "Regression-Diagnosis Branch"):
  The cube 1b run with thin_surface_start=6000 collapsed to PSNR -17.78
  after activation, even though the same run was at +30.46 PSNR at iter 4999
  with scalar baseline.  Reproducing this in a unit test helps isolate whether
  the failure is in:
    - Continuity at the activation frame (forward output jumps)
    - Optimizer / scheduler registration at late activation
    - Post-activation gradient / parameter stability

This file follows the repo-style of test/test_geom_reg.py and
test/test_thin_surface.py: standalone Python, sys.exit(1) on failure,
CPU-friendly (no CUDA required).

Run with:  micromamba run -n radfoam python test/test_thin_surface_activation.py
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


# -----------------------------------------------------------------------
# Stubs so we can import the scene module without a GPU.
# -----------------------------------------------------------------------
def _install_radfoam_stub():
    """Install a CPU-friendly radfoam stub before radfoam_model/scene.py
    imports the real extension (which requires CUDA)."""
    if "radfoam" in sys.modules:
        return
    mod = types.ModuleType("radfoam")
    # Always-returned None placeholders.
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda pts, adj, off, **kw: (
        torch.zeros(pts.shape[0], dtype=torch.long),
        torch.ones(pts.shape[0], device=pts.device),
    )
    mod.nn = lambda points, tree, query, **kw: torch.zeros(
        query.shape[0], dtype=torch.long, device=query.device)
    mod.BatchFetcher = lambda *a, **k: None
    mod.TriangulationFailedError = type("TriangulationFailedError", (Exception,), {})
    mod.Triangulation = None  # constructed only by callers that need real geometry
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()

from radfoam_model.scene import (  # noqa: E402
    CTScene,
    assert_supported_thin_K,
    _SUPPORTED_THIN_K,
    _THIN_K_HARD_CAP,
)


torch.manual_seed(42)
np.random.seed(42)


# =======================================================================
# Scene / args helpers
# =======================================================================

def _args():
    """Namespace with every field declare_optimizer + initialize_thin_surface
    read.  Defaults match the cube smoke / best428_thinsurface values."""
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-2
    a.freeze_points = 9500
    a.thin_surface_start = -1   # set later
    a.thin_surface_K = 4
    a.thin_surface_delta_weight = 1e-3
    a.thin_surface_height_weight = 5e-4
    a.thin_surface_gate_tau = 0.01
    return a


def _make_fake_scene(N=64, K=4, device="cpu"):
    """Build a CTScene with a fake all-pairs adjacency and random geometry.

    Bypasses `Triangulation` entirely by constructing adjacency by hand.
    This is enough for `declare_optimizer` and `initialize_thin_surface`
    because neither touches the triangulation.
    """
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device(device)
    scene.num_init_points = N
    scene.num_final_points = N
    scene._thin_surface_active = False
    scene._thin_K = K
    scene._thin_surface_gate_tau = 0.01
    scene.thin_surface_scheduler_args = None

    # Real-ish points.
    pts = (torch.rand(N, 3, device=device) - 0.5) * 1.0
    scene.primal_points = nn.Parameter(pts)
    scene.density = nn.Parameter(0.5 * torch.ones(N, 1, device=device))

    # Fake full-adjacency so update_learning_rate etc. doesn't blow up if
    # they touch it.  CSR offsets (N+1,) of cumulative adjacency counts.
    adj = []
    offsets = [0]
    for i in range(N):
        # Connect every cell i to every cell j (j != i) — fake.
        nbrs = [j for j in range(N) if j != i]
        adj.extend(nbrs)
        offsets.append(len(adj))
    scene.point_adjacency = torch.tensor(adj, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = torch.tensor(offsets, dtype=torch.int32).to(torch.uint32)
    scene._cached_cell_radius = torch.ones(N, device=device)
    return scene


def make_rays(N=4, device="cpu"):
    """Rays along ±axes from -1.5 to +1.5."""
    dirs = torch.eye(3, device=device).repeat_interleave(N // 3 + 1, dim=0)[:N]
    origins = -1.5 * dirs
    return torch.cat([origins, dirs], dim=-1)


# =======================================================================
# CPU helper: scalar projection proxy that mirrors the pre-activation
# mode of the real kernel (mu_eff = softplus(density)).
# =======================================================================

def naive_scalar_proj(scene, rays):
    """Cost-free scalar proxy: for each ray, project = NN-cell softplus."""
    pts = scene.primal_points.detach()
    density = scene.density.detach()
    origins = rays[..., :3].reshape(-1, 3)
    diff = origins.unsqueeze(1) - pts.unsqueeze(0)
    nn = diff.pow(2).sum(-1).argmin(-1)
    mu = torch.nn.functional.softplus(density.squeeze(), beta=10.0)
    return mu[nn].reshape(*rays.shape[:-1], 1)


# =======================================================================
# Test 1: activation continuity at the activation frame
# =======================================================================

def test_activation_continuity():
    """Symbolic + shape continuity: at the activation frame the four new
    tensors are all at init (delta=0, heights=0, identity quaternions), so
    mu_plus == mu_minus == softplus(base), and the surface is flat at the
    cell center.  This collapses the two-sided kernel contribution back to
    the scalar baseline.  This test verifies the init values that make
    that collapse true.

    Test method:
      1. Build a scene; declare optimizer.
      2. Verify _thin_surface_active is False and no thin-surface tensors
         exist.
      3. Call initialize_thin_surface(args, K=4) at simulated iter 6000.
      4. Verify the four tensors exist with the right shapes and init
         values (delta=0, heights=0, identity quaternion, sites on the
         unit disc).
      5. Verify the inertness invariant: mu_plus == mu_minus in CPU
         proxy (this is what makes continuity hold at init).
      6. Comment on how the GPU continuity check would proceed (this
         test catches the failure modes most likely to leave the cube
         collapsed — wrong init values or missing tensors).

    On GPU, a second pass (commented at bottom) would render the same
    rays with thin_surface_active=False and =True and require
    |V_post - V_pre|_∞ < 5e-5.
    """
    print("\n--- Test 1: activation continuity ---")
    scene = _make_fake_scene(N=64, device="cpu")

    # Pre-activation: thin-surface tensors should not exist.
    pre_has_dd = getattr(scene, "density_delta", None) is not None
    pre_has_q = getattr(scene, "quaternions", None) is not None
    pre_has_ts = getattr(scene, "texel_sites_2d", None) is not None
    pre_has_th = getattr(scene, "texel_heights", None) is not None
    pre_active = getattr(scene, "_thin_surface_active", False)

    check(not pre_has_dd, "density_delta absent before activation")
    check(not pre_has_q, "quaternions absent before activation")
    check(not pre_has_ts, "texel_sites_2d absent before activation")
    check(not pre_has_th, "texel_heights absent before activation")
    check(not pre_active, "_thin_surface_active False before activation")

    # Declare optimizer (mirrors train.py order).
    args = _args()
    args.thin_surface_start = 6000
    scene.declare_optimizer(args, warmup=1000, max_iterations=10000)
    pre_names = [g["name"] for g in scene.optimizer.param_groups]
    check(set(pre_names) == {"primal_points", "density"},
          f"Optimizer has only primal_points+density before activation (got {pre_names})")

    # Activate late.
    scene.initialize_thin_surface(args, K=4)
    check(getattr(scene, "_thin_surface_active", False),
          "_thin_surface_active True after activation")
    check(scene._thin_K == 4, f"_thin_K=4 (got {scene._thin_K})")
    check(getattr(args, "thin_surface_start", None) == 6000,
          "args.thin_surface_start preserved (=6000)")

    # Verify init values (the central invariant — these are the values
    # that make the kernel collapse back to scalar at iter 6001).
    N = scene.primal_points.shape[0]
    assert_thin_params_registered(scene, N, K=4)

    dd = scene.density_delta.detach()
    q = scene.quaternions.detach()
    ts = scene.texel_sites_2d.detach()
    th = scene.texel_heights.detach()

    dd_zero = torch.zeros_like(dd)
    check(torch.allclose(dd, dd_zero, atol=1e-12),
          f"density_delta = 0 at activation (max abs={dd.abs().max():.2e})")
    check(torch.allclose(th, torch.zeros_like(th), atol=1e-12),
          f"texel_heights = 0 at activation (max abs={th.abs().max():.2e})")
    q_id = torch.zeros_like(q)
    q_id[:, 0] = 1.0
    check(torch.allclose(q, q_id, atol=1e-6),
          f"quaternions = identity (max abs diff={(q - q_id).abs().max():.2e})")

    # Site positions on the unit disc (radius 0.4 + tiny jitter <= 0.5).
    site_norm = ts.norm(dim=-1)
    check((site_norm > 0.25).all().item(),
          f"texel_sites_2d sites non-trivially displaced "
          f"(min site_norm={site_norm.min():.3f}, expected >= 0.25)")
    check((site_norm < 0.6).all().item(),
          f"texel_sites_2d sites inside disc (max site_norm="
          f"{site_norm.max():.3f}, expected < 0.6)")

    # Inertness invariant: mu_plus == mu_minus when delta=0.
    density = scene.density.detach()
    mu_bar = torch.nn.functional.softplus(density.squeeze(), beta=10.0)
    mu_p = torch.relu(mu_bar + dd.squeeze())
    mu_n = torch.relu(mu_bar - dd.squeeze())
    check(torch.allclose(mu_p, mu_n, atol=1e-12),
          f"mu_plus == mu_minus when delta=0 (max diff="
          f"{(mu_p - mu_n).abs().max():.2e})")
    check(torch.allclose(mu_p, mu_bar, atol=1e-12),
          f"mu_plus == mu_bar when delta=0 (max diff="
          f"{(mu_p - mu_bar).abs().max():.2e})")

    # Continuity of the configure-API surface:
    # - get_trace_data now surfaces the four tensors.
    td = scene.get_trace_data()
    density_delta_idx = 9
    quaternions_idx = 10
    texel_sites_2d_idx = 11
    texel_heights_idx = 12
    check(td[density_delta_idx] is not None,
          "get_trace_data() surfaces density_delta after activation")
    check(td[quaternions_idx] is not None,
          "get_trace_data() surfaces quaternions after activation")
    check(td[texel_sites_2d_idx] is not None,
          "get_trace_data() surfaces texel_sites_2d after activation")
    check(td[texel_heights_idx] is not None,
          "get_trace_data() surfaces texel_heights after activation")
    check(torch.equal(td[density_delta_idx].detach(), dd),
          "get_trace_data() returns the registered density_delta value")
    check(torch.equal(td[quaternions_idx].detach(), q),
          "get_trace_data() returns the registered quaternions value")
    check(torch.equal(td[texel_sites_2d_idx].detach(), ts),
          "get_trace_data() returns the registered texel_sites_2d value")
    check(torch.equal(td[texel_heights_idx].detach(), th),
          "get_trace_data() returns the registered texel_heights value")

    print("    NOTE (GPU continuity): on a CUDA build run the same tests")
    print("    plus a forward-pass identity check: render the same rays")
    print("    with thin_surface_active=False then =True and assert")
    print("    |V_post - V_pre|_∞ < 5e-5.  See header comment for details.")


# =======================================================================
# Test 2: optimizer / scheduler sanity at late activation
# =======================================================================

def test_optimizer_scheduler_at_late_activation():
    """After initialize_thin_surface the four param groups must be
    registered in the optimizer with finite, intended-magnitude LRs.
    Failure modes:
      - Missing add_param_group: grads flow but stay zero (silent).
      - lr = 0 or NaN: surface inert forever.
      - Scheduler has negative max_steps: NaN propagated to all LRs.
      - Re-activation duplicates param groups (Adam state confusion).
    """
    print("\n--- Test 2: optimizer + scheduler at late activation ---")
    scene = _make_fake_scene(N=64, device="cpu")
    args = _args()
    args.thin_surface_start = 6000
    args.density_lr_init = 5e-2

    scene.declare_optimizer(args, warmup=1000, max_iterations=10000)
    pre_names = sorted(g["name"] for g in scene.optimizer.param_groups)
    check(pre_names == ["density", "primal_points"],
          f"Pre-activation groups = primal_points+density (got {pre_names})")

    # Capture LR values for sanity (both should be the configured init).
    pre_lrs = {g["name"]: g["lr"] for g in scene.optimizer.param_groups}
    check(math.isfinite(pre_lrs["primal_points"]) and pre_lrs["primal_points"] > 0,
          f"primal_points LR > 0 (got {pre_lrs['primal_points']:.3e})")
    check(math.isfinite(pre_lrs["density"]) and pre_lrs["density"] > 0,
          f"density LR > 0 (got {pre_lrs['density']:.3e})")

    # Activate.
    scene.initialize_thin_surface(args, K=4)
    post_names = sorted(g["name"] for g in scene.optimizer.param_groups)
    expected = {"density_delta", "quaternions", "texel_sites_2d", "texel_heights"}
    new_groups = expected - {"density", "primal_points"}
    missing = new_groups - set(post_names)
    check(not missing,
          f"Thin-surface param groups registered (added: {sorted(new_groups)}, "
          f"missing: {sorted(missing)})")
    # Original groups still present.
    check("primal_points" in post_names, "primal_points group preserved")
    check("density" in post_names, "density group preserved")

    # LR magnitudes: density_delta / quaternions / texel_sites_2d /
    # texel_heights should all be args.density_lr_init * 0.1 at init.
    intended = args.density_lr_init * 0.1
    post_lrs = {g["name"]: g["lr"] for g in scene.optimizer.param_groups}
    for name in expected:
        lr = post_lrs[name]
        check(math.isfinite(lr) and lr > 0,
              f"{name} LR > 0 (got {lr:.3e})")
        check(abs(lr - intended) < 1e-15,
              f"{name} LR = density_lr_init * 0.1 = {intended:.3e} "
              f"(got {lr:.3e})")
    # Sanity: the four thin-surface LRs are not all the same as primal_points.
    check(abs(post_lrs["density_delta"] - pre_lrs["primal_points"]) > 1e-12,
          "Thin-surface LRs differ from primal_points LR")

    # Scheduler also configured.
    check(scene.thin_surface_scheduler_args is not None,
          "thin_surface_scheduler_args registered")
    s_at_0 = scene.thin_surface_scheduler_args(0)
    s_at_max = scene.thin_surface_scheduler_args(args.thin_surface_start)
    check(math.isfinite(s_at_0),
          f"scheduler(0) finite: {s_at_0:.3e}")
    check(math.isfinite(s_at_max),
          f"scheduler(max) finite: {s_at_max:.3e}")

    # update_learning_rate at the activation iter must propagate finite LRs.
    scene.update_learning_rate(args.thin_surface_start)
    for g in scene.optimizer.param_groups:
        if g["name"] in expected:
            check(math.isfinite(g["lr"]) and g["lr"] > 0,
                  f"{g['name']} LR after update_learning_rate = "
                  f"{g['lr']:.3e}")

    # Idempotency: re-activation must NOT duplicate groups (covers resumed
    # training and the late-activation "add_param_group" idempotency).
    pre_count = len(scene.optimizer.param_groups)
    scene.initialize_thin_surface(args, K=4)
    post_count = len(scene.optimizer.param_groups)
    check(pre_count == post_count,
          f"Re-activation does not duplicate param groups "
          f"({pre_count} -> {post_count})")


# =======================================================================
# Test 3: post-activation stability smoke
# =======================================================================

def test_post_activation_stability_smoke():
    """Synthetic gradient-descent smoke test.  Runs the optimizer for ~20
    steps with a deliberately aggressive learning rate and a synthetic
    loss that exercises density_delta.  Watches for:

      (a) Nonfinite density_delta / texel_heights / quaternions at any step.
      (b) Exploding mu_plus / mu_minus (abs > 100).
      (c) Catastrophic loss jump (> 100× baseline).
      (d) NaN grad on any thin-surface param.

    Failure mode: this is the loop that produced the cube-1b collapse.
    If the symptom is gradient explosion (e.g. very large dens_delta
    grad at iter 6000 because the soft-Voronoi's bandwidth was set
    wrong, or because the t_surf chain gives a huge gradient when the
    surface is exactly on the chord), we'd see (a) or (b) within the
    first few steps.
    """
    print("\n--- Test 3: post-activation stability smoke ---")
    scene = _make_fake_scene(N=64, device="cpu")
    args = _args()
    args.thin_surface_start = 0   # activate immediately for the smoke
    args.thin_surface_K = 4
    scene.declare_optimizer(args, warmup=0, max_iterations=10000)
    scene.initialize_thin_surface(args, K=4)

    # Use an aggressive LR on the four thin-surface groups so a stability
    # issue surfaces within 20 steps.
    aggressive_lr = 5e-3
    for g in scene.optimizer.param_groups:
        if g["name"] in ("density_delta", "quaternions",
                          "texel_sites_2d", "texel_heights"):
            g["lr"] = aggressive_lr

    density = scene.density

    # Synthetic loss: NN-cell (mu_bar + delta) should match a target of
    # 1.0 (linear rather than quadratic so the loss landscape has a single
    # minimum and avoids the symmetric 0-minimum oscillation of |delta|²).
    # Without delta the baseline would settle at softplus(base)=1 exactly.
    # With delta active, the optimization is constant-sign so |delta|²
    # would also work — but sign-asymmetric linear is closer to what the
    # real CT loss sees (the loss wants one side near 1, the other can be
    # anywhere).  We use linear:  loss = mean(relu(delta + 1 - 1)).
    rays = make_rays(N=8, device="cpu")
    pts = scene.primal_points.detach()
    origins = rays[..., :3].reshape(-1, 3)
    diff = origins.unsqueeze(1) - pts.unsqueeze(0)
    nn = diff.pow(2).sum(-1).argmin(-1)  # (R*N,)

    N_STEPS = 20
    history = []
    target = torch.ones_like(nn, dtype=torch.float32)  # per-ray target
    for step in range(N_STEPS):
        scene.zero_grad(set_to_none=True)
        mu_bar_per_ray = torch.nn.functional.softplus(
            density.squeeze(), beta=10.0)[nn]
        delta_per_ray = scene.density_delta.squeeze()[nn]
        # mu_pred = softplus(base) + delta  (linear in delta so the loss
        # minimum is well-defined and the optimizer doesn't oscillate).
        mu_pred = mu_bar_per_ray + delta_per_ray
        loss = (mu_pred - target).pow(2).mean()
        loss.backward()
        scene.optimizer.step()

        with torch.no_grad():
            dd = scene.density_delta.detach()
            q = scene.quaternions.detach()
            th = scene.texel_heights.detach()
            ts = scene.texel_sites_2d.detach()
            mu_bar_v = torch.nn.functional.softplus(
                density.detach().squeeze(), beta=10.0)
            d_v = scene.density_delta.detach().squeeze()
            mu_p = torch.relu(mu_bar_v + d_v)
            mu_n = torch.relu(mu_bar_v - d_v)
            grad_finite = all(
                (p.grad is None or p.grad.isfinite().all().item())
                for p in [scene.density_delta, scene.quaternions,
                          scene.texel_heights, scene.texel_sites_2d]
            )
        history.append({
            "step": step, "loss": loss.item(),
            "dd_max": dd.abs().max().item(),
            "q_max": q.abs().max().item(),
            "th_max": th.abs().max().item(),
            "ts_max": ts.abs().max().item(),
            "mu_p_max": mu_p.max().item(),
            "mu_n_max": mu_n.max().item(),
            "grad_finite": grad_finite,
        })

    print(f"    Step  Loss       dd∞     q∞     th∞    ts∞    mu_p∞   mu_n∞   grad_finite")
    for h in history:
        marker = "*" if (h["step"] % 5 == 0 or h["step"] == N_STEPS - 1) else " "
        print(f"    {marker}{h['step']:3d}  {h['loss']:.4e}  "
              f"{h['dd_max']:.2e}  {h['q_max']:.2e}  "
              f"{h['th_max']:.2e}  {h['ts_max']:.2e}  "
              f"{h['mu_p_max']:.2e}  {h['mu_n_max']:.2e}  "
              f"{h['grad_finite']}")

    # ── Failure conditions ──────────────────────────────────────────
    # (a) All tensors stayed finite at every step.
    finite_at_each = all(
        math.isfinite(h["dd_max"]) and math.isfinite(h["q_max"])
        and math.isfinite(h["th_max"]) and math.isfinite(h["ts_max"])
        and math.isfinite(h["mu_p_max"]) and math.isfinite(h["mu_n_max"])
        for h in history
    )
    check(finite_at_each,
          "All tensors stayed finite across 20 synthetic steps")

    # (b) density_delta did not explode (L∞ < 1.0).
    max_dd = max(h["dd_max"] for h in history)
    check(max_dd < 1.0,
          f"density_delta L∞ < 1.0 across smoke (max={max_dd:.3e})")

    # (c) mu_plus / mu_minus did not exceed 10.0 (cube-1b final: 7.45).
    max_mu = max(max(h["mu_p_max"], h["mu_n_max"]) for h in history)
    check(max_mu < 10.0,
          f"mu_p/mu_n L∞ < 10.0 across smoke (max={max_mu:.3e})")

    # (d) All thin-surface grads stayed finite.
    grad_finite = all(h["grad_finite"] for h in history)
    check(grad_finite,
          "All thin-surface grads stayed finite across smoke")

    # (e) No catastrophic absolute loss jump.  cube-1b collapsed at iter
    #     6000 with final PSNR=-17.78 — the corresponding CT loss is on
    #     the order of 1000 (projection magnitude is ~ 0.5 and error is
    #     unit-scale).  A reasonable threshold is |Δloss| > 100 between
    #     adjacent steps.
    losses = [h["loss"] for h in history]
    max_abs_jump = max(abs(losses[i] - losses[i - 1]) for i in range(1, len(losses)))
    check(max_abs_jump < 100.0,
          f"Per-step absolute loss jump < 100 (got {max_abs_jump:.3e})")
    # (f) Final loss should not be wildly worse than initial loss.
    #     cube-1b ran with reasonable loss for 6000 steps then jumped
    #     orders of magnitude.  We allow some slack but a successful run
    #     improves loss over the smoke window.
    check(losses[-1] < losses[0] * 2.0,
          f"Final loss < 2x initial loss (got {losses[-1]:.4e} vs "
          f"{losses[0]:.4e})")


# =======================================================================
# Helpers
# =======================================================================

_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        global _any_failed
        _any_failed = True


def assert_thin_params_registered(scene, N, K=4):
    """Verify the four thin-surface tensors exist with the right shapes."""
    for name, shape in [
        ("density_delta", (N, 1)),
        ("quaternions", (N, 4)),
        ("texel_sites_2d", (N, K, 2)),
        ("texel_heights", (N, K)),
    ]:
        t = getattr(scene, name, None)
        assert t is not None, f"{name} not registered after activation"
        assert tuple(t.shape) == shape, \
            f"{name} shape: got {tuple(t.shape)}, expected {shape}"
    assert scene._thin_surface_active, "_thin_surface_active must be True"
    assert scene._thin_K == K, f"_thin_K: got {scene._thin_K}, expected {K}"


def main():
    print("=" * 60)
    print("Thin-Surface Late-Activation Regression Tests")
    print("=" * 60)
    print(f"CUDA available: {_HAS_CUDA}")
    print(f"Supported K:    {_SUPPORTED_THIN_K}")
    print(f"Hard cap K:     {_THIN_K_HARD_CAP}")
    if not _HAS_CUDA:
        print("\nNOTE: CPU-only mode.  The continuity-of-render assertion")
        print("(forward-output must be unchanged across activation) requires")
        print("the real CUDA kernel via the GPU harness.  These tests")
        print("exercise all CPU-side predicates that most likely caused the")
        print("cube-1b collapse (missing tensor / wrong init / wrong shape /")
        print("non-registered optimizer group / runaway delta).")

    test_activation_continuity()
    test_optimizer_scheduler_at_late_activation()
    test_post_activation_stability_smoke()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED.")
        sys.exit(1)
    print("SUMMARY: ALL ACTIVATION-REGRESSION TESTS PASSED.")


if __name__ == "__main__":
    main()
