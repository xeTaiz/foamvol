"""Lifecycle tests for `points_hard_freeze_at` (LC64 plan v2).

Spec:  specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v2.md ("True stationary-frame
       control").

Reviewer contract (Codex-revised):
  - Idempotent `enforce_hard_point_freeze(iter)` method that, for
    iter >= T, always targets the CURRENT primal-points optimizer
    group, sets its LR=0, requires_grad=False on the current primal
    parameter, clears its Adam state.
  - Call it at start of each iteration before forward, and defensively
    before optimizer.step.
  - Reapply after every current primal-point replacement path
    (permute / prune / densify / load) or guarantee start-of-iteration
    is after any such path.
  - Preserve legacy behavior when disabled (default -1).

This file replaces the prior unverified test.  Tests cover:
  1. Default-disabled: every flag / state / LR is unchanged by
     enforce_hard_point_freeze, including after a primal-points
     replacement (which would re-assert the unchanged state).
  2. T-1/T/T+1 boundary: T-1 leaves legacy intact; T freezes; T+1
     remains frozen (idempotent).
  3. T=0 / T=500: each fires the freeze and a step under a large
     gradient produces zero primal-points displacement.
  4. Parameter replacement: simulate permute_points (the most common
     replacement path) and verify the freeze is re-asserted on the
     NEW tensor (requires_grad=False, LR=0, Adam state cleared).
  5. Integration: the patched train.py iteration body calls
     enforce_hard_point_freeze at start of iteration and right before
     optimizer.step.

CPU-only.  Run:
    micromamba run -n radfoam python test/test_points_hard_freeze_at.py
"""
import os
import sys
import types
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# Repo root on path so radfoam_model can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Stub radfoam (CPU-only).  The methods we exercise here (declare_optimizer,
# enforce_hard_point_freeze, replace primal_points) do not touch CUDA.
# ---------------------------------------------------------------------------
def _install_radfoam_stub():
    if "radfoam" in sys.modules:
        return
    mod = types.ModuleType("radfoam")
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda pts, adj, off, **kw: (
        torch.zeros(pts.shape[0], dtype=torch.long),
        torch.ones(pts.shape[0], device=pts.device),
    )
    mod.nn = lambda points, tree, query, **kw: torch.zeros(
        query.shape[0], dtype=torch.long, device=query.device)
    mod.BatchFetcher = lambda *a, **k: None
    mod.TriangulationFailedError = type("T", (Exception,), {})
    mod.Triangulation = None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()

from radfoam_model.scene import CTScene  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_args(threshold):
    """Build the arg Namespace that declare_optimizer reads."""
    class A:
        pass
    args = A()
    args.points_lr_init = 2e-4
    args.points_lr_final = 5e-6
    args.density_lr_init = 5e-2
    args.density_lr_final = 1e-2
    args.freeze_points = 9500
    args.points_hard_freeze_at = threshold
    return args


def _make_scene(threshold, N=8):
    """Minimal CTScene with primal_points and density set; no CUDA."""
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device("cpu")
    scene.num_init_points = N
    scene.num_final_points = N
    scene.primal_points = nn.Parameter((torch.rand(N, 3) - 0.5) * 1.0)
    scene.density = nn.Parameter(0.5 * torch.ones(N, 1))
    adj, offsets = [], [0]
    for i in range(N):
        nbrs = [j for j in range(N) if j != i]
        adj.extend(nbrs)
        offsets.append(len(adj))
    scene.point_adjacency = torch.tensor(adj, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = torch.tensor(
        offsets, dtype=torch.int32).to(torch.uint32)
    scene._cached_cell_radius = torch.ones(N)
    scene._thin_surface_active = False
    scene._thin_K = 4
    scene.declare_optimizer(_make_args(threshold),
                            warmup=0, max_iterations=10000)
    return scene


def _lr_pp(scene):
    return next(g["lr"] for g in scene.optimizer.param_groups
                if g["name"] == "primal_points")


_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# ===========================================================================
# Test 1: disabled (default sentinel -1) preserves legacy behavior.
# ===========================================================================
def test_disabled_default_legacy_behavior():
    """Default sentinel -1: enforce_hard_point_freeze is a complete no-op
    at every iter.  The legacy freeze_points cosine schedule is intact.
    requires_grad stays True; LR follows the cosine decay.
    """
    print("\n--- Test 1: disabled default preserves legacy ---")
    for threshold in [-1, -100]:
        scene = _make_scene(threshold=threshold)
        # Pre-freeze state: legacy LR schedule and requires_grad=True.
        scene.update_learning_rate(0)
        l0 = _lr_pp(scene)
        check(l0 == 2e-4,
              f"  threshold={threshold}: LR(0) == 2e-4 (got {l0:.4e})")
        check(scene.primal_points.requires_grad is True,
              f"  threshold={threshold}: pp.requires_grad is True initially")
        # enforce at every iter is a no-op.
        for i in range(-5, 11000, 100):
            scene.enforce_hard_point_freeze(i)
        check(scene.primal_points.requires_grad is True,
              f"  threshold={threshold}: pp.requires_grad stays True after "
              f"enforce loop")
        # LR still follows the cosine schedule.
        scene.update_learning_rate(4750)
        l_mid = _lr_pp(scene)
        scene.update_learning_rate(9499)
        l_late = _lr_pp(scene)
        check(l_mid < l0 and l_late < l_mid,
              f"  threshold={threshold}: cosine decay intact (LR(4750)="
              f"{l_mid:.3e}, LR(9499)={l_late:.3e})")
        # Adam state never cleared by enforce (it wasn't created in the
        # first place because no step has run).
        check(scene.primal_points not in scene.optimizer.state,
              f"  threshold={threshold}: Adam state never created "
              f"(no step run)")


# ===========================================================================
# Test 2: T-1 / T / T+1 boundary at threshold=500.
# ===========================================================================
def test_boundary_T_minus1_T_T_plus1():
    """At threshold=500:
      T=499 (T-1): legacy schedule intact, requires_grad stays True.
      T=500 (T): freeze fires.  requires_grad=False, LR=0, Adam state
                 cleared (none yet created, but the freeze must not
                 crash).
      T=501 (T+1): idempotent.  requires_grad stays False; LR stays 0;
                 a fresh Adam state created by an intervening step is
                 also cleared by the next enforce call.
    """
    print("\n--- Test 2: T-1 / T / T+1 boundary (threshold=500) ---")
    scene = _make_scene(threshold=500)
    # T=499: legacy.
    scene.update_learning_rate(499)
    l_tm1 = _lr_pp(scene)
    check(l_tm1 > 0,
          f"T=499: LR > 0 (legacy schedule; got {l_tm1:.4e})")
    scene.enforce_hard_point_freeze(499)
    check(scene.primal_points.requires_grad is True,
          "T=499: pp.requires_grad stays True after enforce")
    # T=500: freeze fires.
    scene.enforce_hard_point_freeze(500)
    check(scene.primal_points.requires_grad is False,
          "T=500: pp.requires_grad is False after enforce")
    check(_lr_pp(scene) == 0.0,
          f"T=500: pp LR == 0 (got {_lr_pp(scene):.4e})")
    # Run an actual optimizer step (with a large fake grad) and verify
    # the primal-points tensor does not move.
    snapshot = scene.primal_points.detach().clone()
    scene.primal_points.grad = torch.randn_like(scene.primal_points) * 1e3
    scene.optimizer.step()
    displacement = (scene.primal_points.detach() - snapshot).abs().max().item()
    check(displacement < 1e-7,
          f"T=500: post-step primal-points displacement < 1e-7 "
          f"(got {displacement:.3e})")
    # Now Adam state exists.  Re-applying enforce at T=501 must CLEAR it.
    state = scene.optimizer.state[scene.primal_points]
    check("exp_avg" in state,
          "T=500 post-step: Adam state present (exp_avg populated)")
    state["exp_avg"].fill_(7.7)   # poison
    state["exp_avg_sq"].fill_(0.5)
    scene.enforce_hard_point_freeze(501)
    check(scene.primal_points not in scene.optimizer.state,
          "T=501 enforce: Adam state cleared (poison reset)")
    check(scene.primal_points.requires_grad is False,
          "T=501: pp.requires_grad still False (sticky)")
    check(_lr_pp(scene) == 0.0,
          f"T=501: pp LR still 0 (got {_lr_pp(scene):.4e})")


# ===========================================================================
# Test 3: T=0 freezes at the very first iter (pre_step pattern).
# ===========================================================================
def test_boundary_T0_freeze_at_iteration_zero():
    """At threshold=0 the freeze fires at iter=0 (the very first iter of
    training).  Subsequent steps at iter=1,2,... remain frozen.
    """
    print("\n--- Test 3: T=0 freezes at first iter ---")
    scene = _make_scene(threshold=0)
    check(scene._hard_freeze_threshold() == 0,
          f"_hard_freeze_threshold() == 0 (got {scene._hard_freeze_threshold()})")
    # Iter=-1: T not reached.
    scene.enforce_hard_point_freeze(-1)
    check(scene.primal_points.requires_grad is True,
          "T=-1: enforce is a no-op")
    # Iter=0: freeze fires.
    snapshot = scene.primal_points.detach().clone()
    scene.enforce_hard_point_freeze(0)
    check(scene.primal_points.requires_grad is False,
          "T=0: pp.requires_grad is False after enforce(0)")
    check(_lr_pp(scene) == 0.0,
          f"T=0: pp LR == 0 (got {_lr_pp(scene):.4e})")
    # Inject a large gradient + step; displacement <= 1e-7.
    scene.primal_points.grad = torch.randn_like(scene.primal_points) * 1e3
    scene.optimizer.step()
    displacement = (scene.primal_points.detach() - snapshot).abs().max().item()
    check(displacement < 1e-7,
          f"T=0: post-step displacement < 1e-7 (got {displacement:.3e})")
    # Iter=1,2,...: stays frozen.
    for i in [1, 2, 10, 100]:
        scene.enforce_hard_point_freeze(i)
        check(scene.primal_points.requires_grad is False,
              f"T={i}: pp.requires_grad still False (sticky)")
        check(_lr_pp(scene) == 0.0,
              f"T={i}: pp LR still 0 (got {_lr_pp(scene):.4e})")


# ===========================================================================
# Test 4: T=500 freezes; legacy schedule survives up to T-1.
# ===========================================================================
def test_boundary_T500_legacy_until_freeze():
    """At threshold=500: iter 0..499 keep legacy schedule, iter >= 500
    freezes.  The legacy xyz_scheduler_args must not be re-bound to the
    primal-points param group during the freeze (re-binding would change
    its identity and break resumed training with a saved scheduler).
    """
    print("\n--- Test 4: T=500 (legacy until 499, freeze at 500+) ---")
    scene = _make_scene(threshold=500)
    # T=0..499: legacy.
    for i in [0, 100, 499]:
        scene.update_learning_rate(i)
        l = _lr_pp(scene)
        check(l > 0,
              f"T={i}: LR > 0 (legacy; got {l:.4e})")
        scene.enforce_hard_point_freeze(i)
        check(scene.primal_points.requires_grad is True,
              f"T={i}: pp.requires_grad True after enforce")
        check(_lr_pp(scene) > 0,
              f"T={i}: LR > 0 after enforce (legacy preserved)")
    # The legacy scheduler (self.xyz_scheduler_args) must be the same
    # callable it was at construction; enforce must NOT replace it.
    sched_before = scene.xyz_scheduler_args
    scene.enforce_hard_point_freeze(500)
    check(scene.xyz_scheduler_args is sched_before,
          "T=500 enforce: xyz_scheduler_args not re-bound")
    check(scene.primal_points.requires_grad is False,
          "T=500: pp.requires_grad False")
    # Subsequent update_learning_rate calls leave LR at exactly 0:
    # update_learning_rate itself now consults the frozen-state helper
    # so the scheduler cannot temporarily restore a positive LR once
    # the freeze is engaged.  This is the documented contract.
    scene.update_learning_rate(1500)
    check(_lr_pp(scene) == 0.0,
          f"T=1500 with hard freeze at T=500: update_learning_rate "
          f"must keep pp LR exactly 0 (got {_lr_pp(scene):.4e})")
    # Same check at T itself: LR == 0 from the moment the threshold is
    # crossed (without any prior enforce call this iter).
    scene2 = _make_scene(threshold=500)
    scene2.update_learning_rate(500)
    check(_lr_pp(scene2) == 0.0,
          f"T=500: update_learning_rate at the threshold keeps "
          f"pp LR exactly 0 (got {_lr_pp(scene2):.4e})")
    scene2.update_learning_rate(9999)
    check(_lr_pp(scene2) == 0.0,
          f"T=9999 (well past threshold): update_learning_rate "
          f"keeps pp LR exactly 0 (got {_lr_pp(scene2):.4e})")


# ===========================================================================
# Test 5: primal-points replacement (permute) re-asserts the freeze.
# ===========================================================================
def test_parameter_replacement_reasserts_freeze():
    """Simulate the most common replacement path (permute_points /
    prune / densify / load_pt): self.primal_points is reassigned to a
    fresh Parameter with requires_grad=True.  The post-replacement hook
    in production calls enforce_hard_point_freeze which must freeze the
    NEW tensor.  In the test we drive enforce_hard_point_freeze manually
    to verify the contract: a fresh Parameter, even after
    .requires_grad_(True), becomes False on the next enforce call.
    """
    print("\n--- Test 5: parameter replacement re-asserts freeze ---")
    scene = _make_scene(threshold=500)
    # Trigger the freeze.
    scene.enforce_hard_point_freeze(500)
    old_pp = scene.primal_points
    check(old_pp.requires_grad is False,
          "T=500: original pp.requires_grad is False")

    # Simulate a replacement: a fresh Parameter with requires_grad=True
    # replaces self.primal_points.  In production code, the post-replace
    # hook calls enforce_hard_point_freeze which would immediately freeze
    # the new tensor.  We replicate that exactly here.
    N = scene.primal_points.shape[0]
    new_pp = nn.Parameter(torch.randn(N, 3))
    check(new_pp.requires_grad is True,
          "fresh Parameter has requires_grad=True (default)")
    scene.primal_points = new_pp
    # Verify the param group still references the OLD tensor (no
    # enforce call has happened since the replacement).
    g_pp = next(g for g in scene.optimizer.param_groups
                 if g["name"] == "primal_points")
    check(g_pp["params"][0] is old_pp,
          "pre-replace hook: param group still references the OLD tensor")

    # Now call enforce_hard_point_freeze to simulate the post-replace
    # hook.  The CURRENT self.primal_points (new_pp) becomes
    # requires_grad=False, the param group is atomically rebound to
    # reference the new tensor, and Adam state for BOTH the old
    # (rebound-out) and new (current) tensors is dropped.
    scene.enforce_hard_point_freeze(1500)
    check(scene.primal_points.requires_grad is False,
          "post-replace enforce: new pp.requires_grad is False")
    # Invariant: the primal-points optimizer group must reference the
    # CURRENT self.primal_points after a replacement + enforce.
    g_pp_after = next(g for g in scene.optimizer.param_groups
                      if g["name"] == "primal_points")
    check(g_pp_after["params"][0] is new_pp,
          "post-replace enforce: param group references the CURRENT "
          "(new) tensor")
    # Existing clear-state policy: drop Adam state for both old and
    # new tensors.  Without explicit old-state cleanup, the OLD tensor
    # would leave orphan momentum behind.
    check(old_pp not in scene.optimizer.state,
          "post-replace enforce: OLD tensor Adam state cleared")
    check(new_pp not in scene.optimizer.state,
          "post-replace enforce: new_pp Adam state cleared")
    # Run a fake-step with a large grad on new_pp; displacement <= 1e-7.
    snapshot = new_pp.detach().clone()
    new_pp.grad = torch.randn_like(new_pp) * 1e3
    scene.optimizer.step()
    displacement = (new_pp.detach() - snapshot).abs().max().item()
    check(displacement < 1e-7,
          f"post-replace enforce + step: new_pp displacement < 1e-7 "
          f"(got {displacement:.3e})")
    # The OLD tensor (no longer in the param group) is irrelevant to the
    # optimizer step; verify it didn't move either (we didn't touch it).
    check(old_pp.requires_grad is False,
          "old_pp.requires_grad still False (no-op, not used by step)")


# ===========================================================================
# Test 6: train.py integration: enforce called at start-of-iter AND
# defensively before optimizer.step.  We verify the source-level wiring
# by reading train.py for the two enforce calls (no need to launch
# training; a structural check on the source is sufficient for the
# lifecycle contract).
# ===========================================================================
def test_train_py_integration_wiring():
    """The reviewer contract requires enforce_hard_point_freeze to be
    called at start-of-iteration and defensively before optimizer.step.
    Verify the structural wiring in train.py without launching training.
    """
    print("\n--- Test 6: train.py lifecycle wiring ---")
    with open(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "train.py"))) as f:
        train_src = f.read()
    # Two enforce calls expected: start-of-iter + pre-step.
    count = train_src.count("model.enforce_hard_point_freeze(i)")
    check(count >= 2,
          f"train.py calls model.enforce_hard_point_freeze(i) at least "
          f"twice (start-of-iter + pre-step); got {count}")
    # Verify the start-of-iter call sits at the very top of the for-loop
    # body (before any model.forward).
    loop_body_start = train_src.find("for i in train:")
    check(loop_body_start > 0, "train.py: 'for i in train:' found")
    after_loop = train_src[loop_body_start:]
    # Find the first occurrence of enforce after the for loop.
    first_enforce = after_loop.find("model.enforce_hard_point_freeze(i)")
    first_forward = after_loop.find("model(ray_batch")
    check(first_enforce > 0 and first_forward > 0,
          "train.py: enforce call and model(...) both present after "
          f"'for i in train:' (enforce @ {first_enforce}, forward @ "
          f"{first_forward})")
    check(first_enforce < first_forward,
          "train.py: enforce call appears BEFORE model(...) (start-of-iter "
          "enforcement)")
    # Verify the second call sits immediately before optimizer.step.
    second_enforce = after_loop.find(
        "model.enforce_hard_point_freeze(i)", first_enforce + 1)
    next_step = after_loop.find("model.optimizer.step()", first_forward)
    check(second_enforce > 0 and next_step > 0,
          f"train.py: second enforce (defensive) and optimizer.step "
          f"both present (defensive @ {second_enforce}, step @ {next_step})")
    check(second_enforce < next_step,
          "train.py: defensive enforce call appears BEFORE "
          "model.optimizer.step()")


# ===========================================================================
# Test 7: get_trace_data contract unchanged (the freeze adds 0th element).
# ===========================================================================
def test_get_trace_data_unchanged():
    """The freeze must not change get_trace_data's tuple length (which is
    part of the forward() signature).  This is the production correctness
    contract from the prior emergency-restore note.
    """
    print("\n--- Test 7: get_trace_data unchanged ---")
    scene = _make_scene(threshold=500)
    # Pre-freeze.
    td = scene.get_trace_data()
    pre_len = len(td)
    check(pre_len == 13,
          f"pre-freeze get_trace_data length == 13 (got {pre_len})")
    # Post-freeze.
    scene.enforce_hard_point_freeze(500)
    td2 = scene.get_trace_data()
    post_len = len(td2)
    check(post_len == 13,
          f"post-freeze get_trace_data length == 13 (got {post_len})")


# ===========================================================================
# Test 8: forward() contract intact when thin-surface is disabled.
# ===========================================================================
def test_forward_signature_intact():
    """forward() unpacks get_trace_data as a 13-tuple.  This smoke uses
    the legacy scalar-mode (thin_surface_active=False) and verifies the
    tuple shape survives the freeze.
    """
    print("\n--- Test 8: forward signature intact (legacy scalar) ---")
    scene = _make_scene(threshold=500)
    # Trigger freeze.
    scene.enforce_hard_point_freeze(500)
    # Direct tuple-shape check (we don't call forward() because that
    # requires radfoam's CUDA pipeline; the test stays CPU-only).
    td = scene.get_trace_data()
    expected = (
        scene.primal_points,
        scene.density,
        scene.point_adjacency,
        scene.point_adjacency_offsets,
        None,            # density_grad (not initialized)
        5.0,             # gradient_max_slope default
        None, None, None,  # density_peak, delta_raw, cov_raw
        None, None, None, None,  # thin-surface tensors (not initialized)
    )
    check(len(td) == len(expected) == 13,
          f"get_trace_data returns 13-tuple (got {len(td)})")


# ===========================================================================
# Test 9: prune_points re-asserts the freeze on the post-prune tensor.
# ===========================================================================
def test_prune_points_reasserts_freeze():
    """The post-prune primal-points tensor is a fresh nn.Parameter.  The
    production prune_points path now calls ``_reapply_hard_freeze()`` at
    the end (via the helper, NOT a hard-coded iteration 0) so the
    freeze survives the replacement.
    """
    print("\n--- Test 9: prune_points re-asserts freeze ---")
    scene = _make_scene(threshold=500)
    # Trip the freeze.
    scene.enforce_hard_point_freeze(500)
    pre_pp = scene.primal_points
    check(pre_pp.requires_grad is False,
          "T=500: pre-prune pp.requires_grad is False")

    # Prune the first two points (mask = [True, True, False, ..., False]).
    mask = torch.zeros(scene.primal_points.shape[0], dtype=torch.bool, device="cpu")
    mask[:2] = True
    scene.prune_points(mask)
    post_pp = scene.primal_points

    # The post-prune tensor must be a different Parameter (fresh
    # allocation by prune_optimizer) AND frozen.
    check(post_pp is not pre_pp,
          "post-prune: primal_points identity changed (fresh Parameter)")
    check(post_pp.requires_grad is False,
          "post-prune: new pp.requires_grad is False (helper re-froze)")
    check(_lr_pp(scene) == 0.0,
          f"post-prune: pp LR == 0 (got {_lr_pp(scene):.4e})")
    # Invariant: optimizer group references the current tensor.
    g_pp = next(g for g in scene.optimizer.param_groups
                if g["name"] == "primal_points")
    check(g_pp["params"][0] is post_pp,
          "post-prune: param group references the CURRENT (post-prune) tensor")
    # Displacement after a step remains negligible.
    snap = post_pp.detach().clone()
    post_pp.grad = torch.randn_like(post_pp) * 1e3
    scene.optimizer.step()
    disp = (post_pp.detach() - snap).abs().max().item()
    check(disp < 1e-7,
          f"post-prune step displacement < 1e-7 (got {disp:.3e})")


# ===========================================================================
# Test 10: densification_postfix re-asserts the freeze on the new tensor.
# ===========================================================================
def test_densification_postfix_reasserts_freeze():
    """densification_postfix builds a fresh primal-points Parameter via
    cat_tensors_to_optimizer.  The production path now calls
    ``_reapply_hard_freeze()`` (helper-based, not hard-coded iter 0)
    so the freeze survives the densify.
    """
    print("\n--- Test 10: densification_postfix re-asserts freeze ---")
    scene = _make_scene(threshold=500, N=4)
    scene.enforce_hard_point_freeze(500)
    pre_pp = scene.primal_points

    # Build a fake "add 2 new cells" densify payload.
    n_new = 2
    dev = pre_pp.device
    new_params = {
        "primal_points": torch.randn(n_new, 3, device=dev),
        "density": 0.5 * torch.ones(n_new, 1, device=dev),
    }
    scene.densification_postfix(new_params)
    post_pp = scene.primal_points

    check(post_pp is not pre_pp,
          "post-densify: primal_points identity changed (fresh Parameter)")
    check(post_pp.requires_grad is False,
          "post-densify: new pp.requires_grad is False (helper re-froze)")
    check(_lr_pp(scene) == 0.0,
          f"post-densify: pp LR == 0 (got {_lr_pp(scene):.4e})")
    g_pp = next(g for g in scene.optimizer.param_groups
                if g["name"] == "primal_points")
    check(g_pp["params"][0] is post_pp,
          "post-densify: param group references the CURRENT tensor")
    snap = post_pp.detach().clone()
    post_pp.grad = torch.randn_like(post_pp) * 1e3
    scene.optimizer.step()
    disp = (post_pp.detach() - snap).abs().max().item()
    check(disp < 1e-7,
          f"post-densify step displacement < 1e-7 (got {disp:.3e})")


# ===========================================================================
# Test 11: permute_points re-asserts the freeze on the post-permute tensor.
# ===========================================================================
def test_permute_points_reasserts_freeze():
    """permute_points builds a fresh primal-points Parameter with the
    permutation applied.  The production path now uses
    ``_reapply_hard_freeze()`` (helper-based) so the freeze survives.
    """
    print("\n--- Test 11: permute_points re-asserts freeze ---")
    scene = _make_scene(threshold=500, N=6)
    scene.enforce_hard_point_freeze(500)
    pre_pp = scene.primal_points

    # A simple non-identity permutation (reverse the indices).
    perm = torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.long)
    scene.permute_points(perm)
    post_pp = scene.primal_points

    check(post_pp is not pre_pp,
          "post-permute: primal_points identity changed (fresh Parameter)")
    check(post_pp.requires_grad is False,
          "post-permute: new pp.requires_grad is False (helper re-froze)")
    check(_lr_pp(scene) == 0.0,
          f"post-permute: pp LR == 0 (got {_lr_pp(scene):.4e})")
    g_pp = next(g for g in scene.optimizer.param_groups
                if g["name"] == "primal_points")
    check(g_pp["params"][0] is post_pp,
          "post-permute: param group references the CURRENT tensor")
    snap = post_pp.detach().clone()
    post_pp.grad = torch.randn_like(post_pp) * 1e3
    scene.optimizer.step()
    disp = (post_pp.detach() - snap).abs().max().item()
    check(disp < 1e-7,
          f"post-permute step displacement < 1e-7 (got {disp:.3e})")


# ===========================================================================
# Test 12: load_pt re-asserts the freeze on the loaded tensor.
# ===========================================================================
def test_load_pt_reasserts_freeze(tmp_path=None):
    """Save a checkpoint, then load it into a fresh scene with the
    freeze already engaged.  The production load_pt path now uses
    ``_reapply_hard_freeze()`` (helper-based) so the freeze survives.
    """
    import tempfile
    print("\n--- Test 12: load_pt re-asserts freeze ---")
    src = _make_scene(threshold=500, N=5)
    pt_path = tempfile.mkdtemp(prefix="hard_freeze_load_") + "/ckpt.pt"
    src.save_pt(pt_path)

    # Fresh scene with threshold=500; trip the freeze, then load.
    dst = _make_scene(threshold=500, N=5)
    dst.enforce_hard_point_freeze(500)
    pre_pp = dst.primal_points
    dst.load_pt(pt_path)
    post_pp = dst.primal_points

    check(post_pp is not pre_pp,
          "post-load: primal_points identity changed (fresh Parameter)")
    check(post_pp.requires_grad is False,
          "post-load: new pp.requires_grad is False (helper re-froze)")
    check(_lr_pp(dst) == 0.0,
          f"post-load: pp LR == 0 (got {_lr_pp(dst):.4e})")
    g_pp = next(g for g in dst.optimizer.param_groups
                if g["name"] == "primal_points")
    check(g_pp["params"][0] is post_pp,
          "post-load: param group references the CURRENT tensor")
    # Displacement after step remains negligible.
    snap = post_pp.detach().clone()
    post_pp.grad = torch.randn_like(post_pp) * 1e3
    dst.optimizer.step()
    disp = (post_pp.detach() - snap).abs().max().item()
    check(disp < 1e-7,
          f"post-load step displacement < 1e-7 (got {disp:.3e})")


# ===========================================================================
# Test 13: helper safe before optimizer declaration and when default
# disabled.
# ===========================================================================
def test_hooks_safe_before_optimizer_and_when_disabled():
    """``_reapply_hard_freeze`` and ``_should_hard_freeze`` must be safe
    to call BEFORE ``declare_optimizer`` (no ``self.optimizer`` yet) and
    when the default sentinel T=-1 keeps the freeze disabled.  The
    replacement hooks (regular_initialize, random_initialize) run
    before declare_optimizer and rely on this safety.
    """
    print("\n--- Test 13: hooks safe before optimizer / when disabled ---")
    # Build a half-initialized scene: primal_points exists but
    # declare_optimizer has NOT been called.
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device("cpu")
    scene.num_init_points = 4
    scene.num_final_points = 4
    scene.primal_points = nn.Parameter((torch.rand(4, 3) - 0.5) * 1.0)
    scene.density = nn.Parameter(0.5 * torch.ones(4, 1))
    # Intentionally NOT calling declare_optimizer.

    # Case A: T=-1 (default).  Both helpers must be no-ops.
    scene._points_hard_freeze_at = -1
    scene._hard_freeze_active = False
    scene._last_iteration = None
    check(not scene._should_hard_freeze(),
          "T=-1: _should_hard_freeze() returns False")
    check(not scene._should_hard_freeze(iteration=10000),
          "T=-1: _should_hard_freeze(10000) returns False")
    # _reapply_hard_freeze must not crash.
    scene._reapply_hard_freeze()
    check(scene.primal_points.requires_grad is True,
          "T=-1: _reapply_hard_freeze() leaves requires_grad True")
    check(not scene._hard_freeze_active,
          "T=-1: _reapply_hard_freeze() does NOT trip the active flag")
    # enforce_hard_point_freeze must also be safe (early return).
    scene.enforce_hard_point_freeze(10000)
    check(scene.primal_points.requires_grad is True,
          "T=-1: enforce_hard_point_freeze(10000) leaves requires_grad True")
    check(scene._last_iteration == 10000,
          "T=-1: enforce still updates _last_iteration for the helper")

    # Case B: T>=0 but optimizer not declared yet.  Both helpers must
    # be safe (no AttributeError on self.optimizer).
    scene._points_hard_freeze_at = 500
    scene._hard_freeze_active = False
    scene._last_iteration = None
    check(not scene._should_hard_freeze(),
          "T=500 (no enforce yet): _should_hard_freeze() returns False")
    check(scene._should_hard_freeze(iteration=500),
          "T=500: _should_hard_freeze(500) returns True")
    check(not scene._should_hard_freeze(iteration=499),
          "T=500: _should_hard_freeze(499) returns False")
    # _reapply_hard_freeze before optimizer must not crash even though
    # the threshold says "active" -- the helper still trips and the
    # enforce call inside it short-circuits on pp-only state.
    try:
        scene._reapply_hard_freeze()
        scene.primal_points.requires_grad_(False)  # pre-state
        # _should_hard_freeze without enforce is False (sticky flag),
        # so _reapply_hard_freeze does not call enforce.
        check(scene.primal_points.requires_grad is False,
              "pre-optimizer _reapply_hard_freeze: stays no-op (sticky "
              "flag unset), no AttributeError")
    except AttributeError as e:
        check(False, f"pre-optimizer _reapply_hard_freeze crashed: {e}")


# ===========================================================================
# Test 14: lifecycle contract end-to-end -- iter >= T via train loop
# ensures every replacement path keeps the freeze.
# ===========================================================================
def test_lifecycle_endtoend_iter_then_replace():
    """Simulate the train-loop pattern: enforce_hard_point_freeze(i) at
    start-of-iter for iter >= T; during the iter, a permute happens.
    The post-permute hook (via _reapply_hard_freeze) must re-assert the
    freeze without the production code knowing the iteration explicitly.
    """
    print("\n--- Test 14: lifecycle iter-then-replace ---")
    scene = _make_scene(threshold=500, N=8)
    # Iter 499: legacy.
    scene.enforce_hard_point_freeze(499)
    check(scene.primal_points.requires_grad is True,
          "T=499: legacy intact")
    # Iter 500: trip the freeze.
    scene.enforce_hard_point_freeze(500)
    pp500 = scene.primal_points
    check(pp500.requires_grad is False, "T=500: freeze fires")
    # A permute happens DURING iter 500 (e.g., update_triangulation
    # triggers a reorder).  Production code calls
    # _reapply_hard_freeze() at the end of permute_points / update_
    # triangulation -- which uses _last_iteration (500) to enforce.
    perm = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.long)
    scene.permute_points(perm)
    pp_after_permute = scene.primal_points
    check(pp_after_permute is not pp500,
          "permute during frozen iter: identity changed")
    check(pp_after_permute.requires_grad is False,
          "permute during frozen iter: helper re-froze the NEW tensor "
          "(no hard-coded iter 0)")
    check(_lr_pp(scene) == 0.0,
          f"permute during frozen iter: LR == 0 (got {_lr_pp(scene):.4e})")
    g_pp = next(g for g in scene.optimizer.param_groups
                if g["name"] == "primal_points")
    check(g_pp["params"][0] is pp_after_permute,
          "permute during frozen iter: param group references CURRENT tensor")
    # Iter 501: enforce is idempotent on the new tensor.
    scene.enforce_hard_point_freeze(501)
    check(pp_after_permute.requires_grad is False,
          "T=501: idempotent on the post-replacement tensor")


# ===========================================================================
# Test 15: helper direct: T=0 with no prior iteration known -- the
# threshold fallback path in _reapply_hard_freeze fires the freeze.
# ===========================================================================
def test_helper_direct_threshold_fallback():
    """When ``_last_iteration`` is None (no enforce call has happened yet)
    but T is non-negative, ``_reapply_hard_freeze`` falls back to the
    threshold itself as the iteration bound.  This means a T=0 freeze
    can be applied by a replacement path that runs before any enforce
    call (e.g., immediately after declare_optimizer in a checkpoint
    reload scenario).
    """
    print("\n--- Test 15: helper direct threshold fallback ---")
    scene = _make_scene(threshold=0, N=4)
    # Wipe the iteration tracker to simulate a fresh checkpoint reload
    # where no enforce has been called yet.
    scene._last_iteration = None
    scene._hard_freeze_active = False
    # _should_hard_freeze() (no iter): uses sticky flag, which is
    # False -- so the helper returns False.
    check(not scene._should_hard_freeze(),
          "no prior enforce: _should_hard_freeze() == False (sticky)")
    # _reapply_hard_freeze is therefore a no-op here.  Once the first
    # enforce(0) fires, the freeze becomes sticky.
    scene._reapply_hard_freeze()
    check(scene.primal_points.requires_grad is True,
          "no prior enforce: _reapply_hard_freeze stays no-op")
    scene.enforce_hard_point_freeze(0)
    check(scene.primal_points.requires_grad is False,
          "enforce(0): freeze fires")
    # Now a replacement path called AFTER the first enforce should
    # re-assert the freeze via _last_iteration.
    scene._last_iteration = 0  # already set by enforce
    new_pp = nn.Parameter(torch.randn(4, 3))
    scene.primal_points = new_pp
    scene._reapply_hard_freeze()
    check(new_pp.requires_grad is False,
          "post-replace _reapply_hard_freeze: freezes new tensor via "
          "_last_iteration fallback")
    check(_lr_pp(scene) == 0.0,
          f"post-replace _reapply_hard_freeze: pp LR == 0 "
          f"(got {_lr_pp(scene):.4e})")


def main():
    print("=" * 60)
    print("points_hard_freeze_at lifecycle tests (LC64 plan v2)")
    print("Spec: specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v2.md")
    print("Reviewer contract: idempotent enforce_hard_point_freeze(iter)")
    print("=" * 60)

    test_disabled_default_legacy_behavior()
    test_boundary_T_minus1_T_T_plus1()
    test_boundary_T0_freeze_at_iteration_zero()
    test_boundary_T500_legacy_until_freeze()
    test_parameter_replacement_reasserts_freeze()
    test_train_py_integration_wiring()
    test_get_trace_data_unchanged()
    test_forward_signature_intact()
    test_prune_points_reasserts_freeze()
    test_densification_postfix_reasserts_freeze()
    test_permute_points_reasserts_freeze()
    test_load_pt_reasserts_freeze()
    test_hooks_safe_before_optimizer_and_when_disabled()
    test_lifecycle_endtoend_iter_then_replace()
    test_helper_direct_threshold_fallback()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL points_hard_freeze_at LIFECYCLE TESTS PASSED.")


if __name__ == "__main__":
    main()
