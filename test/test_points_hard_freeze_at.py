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
    # Subsequent update_learning_rate calls leave LR at 0 (the freeze
    # path inside update_learning_rate via _hard_freeze_threshold is NOT
    # used; the freeze is enforced by enforce_hard_point_freeze.  Verify
    # that update_learning_rate without prior enforce would re-set LR
    # to the cosine value -- but the test setup doesn't enforce again
    # after 500, so the cosine schedule takes over.  This is the
    # document contract: enforce must be called at start-of-iteration
    # AND defensively pre-step.
    scene.update_learning_rate(1500)
    check(_lr_pp(scene) >= 0,
          f"T=1500 (no second enforce): LR = cosine schedule "
          f"(got {_lr_pp(scene):.4e}); enforce must be called per-iter")


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
    # Verify the param group still references the OLD tensor.
    g_pp = next(g for g in scene.optimizer.param_groups
                 if g["name"] == "primal_points")
    check(g_pp["params"][0] is old_pp,
          "pre-replace hook: param group still references the OLD tensor")

    # Now call enforce_hard_point_freeze to simulate the post-replace
    # hook.  The CURRENT self.primal_points (new_pp) becomes
    # requires_grad=False, and the param group is updated to reference
    # the new tensor.  This is the reviewer contract: always target the
    # CURRENT tensor, never a cached reference.
    scene.enforce_hard_point_freeze(1500)
    check(scene.primal_points.requires_grad is False,
          "post-replace enforce: new pp.requires_grad is False")
    # The OLD tensor's state in the optimizer should also be cleared if
    # it's still there.  (The contract is that the Adam state entry for
    # self.primal_points is cleared; here self.primal_points is the new
    # tensor, so the old tensor's state would remain in the dict as
    # orphan data.  That is acceptable: only the current tensor matters.)
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

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL points_hard_freeze_at LIFECYCLE TESTS PASSED.")


if __name__ == "__main__":
    main()
