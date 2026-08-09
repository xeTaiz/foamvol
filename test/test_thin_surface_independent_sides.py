"""Commit 1 schema tests for the LC64 plan v3 independent-side mode.

Spec:  specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v2.md ("Resolved E0
       comparison contract" — native raw-side Adam, not coordinate-
       matched), v3 E0 amendment.

Scope of THIS commit (Python schema only — no CUDA):
  - `raw_plus`, `raw_minus` (each (N,1)) cloned from `density`.
  - Two ordinary Adam groups (`raw_plus`, `raw_minus`) sharing a
    single cosine LR scheduler (identical schedules by construction).
  - Base `density` frozen (requires_grad=False, removed from
    optimizer) as a third density degree.
  - `get_trace_data()` extends to 15 elements; legacy scalar /
    absolute / relative paths unaffected.
  - `forward()` raises NotImplementedError under independent mode
    (fail-fast, no silent scalar fallback).
  - `save_pt` / `load_pt` round-trip raw tensors + discriminator,
    reject mixed/malformed state, preserve legacy checkpoint
    inference.

This test file is CPU-only; it does NOT depend on CUDA.  It is
the Commit 1 reviewer contract.

Run:
  micromamba run -n radfoam python test/test_thin_surface_independent_sides.py
"""

import os
import sys
import math
import types
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# CPU radfoam stub so scene.py can be imported without CUDA.  Mirrors the
# stub used by test_thin_surface_lr_scale.py / test_thin_surface_activation.py.
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
    mod.TriangulationFailedError = type(
        "TriangulationFailedError", (Exception,), {})
    mod.Triangulation = None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()
from radfoam_model.scene import CTScene  # noqa: E402


torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _args(thin_surface_density_mode="scalar",
          thin_surface_relative_delta=False,
          thin_surface_raw_side_lr_init=5e-2,
          thin_surface_raw_side_lr_final=1e-3,
          thin_surface_start=-1):
    """Namespace with every field declare_optimizer + the independent-
    side path reads."""
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-3
    a.freeze_points = 9500
    a.thin_surface_start = thin_surface_start
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
    a.thin_surface_relative_delta = thin_surface_relative_delta
    a.thin_surface_delta_max_frac = 0.5
    a.thin_surface_density_mode = thin_surface_density_mode
    a.thin_surface_raw_side_lr_init = thin_surface_raw_side_lr_init
    a.thin_surface_raw_side_lr_final = thin_surface_raw_side_lr_final
    return a


def _make_scene(n_points=8, device="cpu"):
    """CPU-friendly CTScene with fake all-pairs adjacency (mirrors
    test_thin_surface_lr_scale.py)."""
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device(device)
    scene.num_init_points = n_points
    scene.num_final_points = n_points
    scene._thin_surface_active = False
    scene._thin_K = 4
    scene._thin_surface_gate_tau = 0.01
    scene._thin_surface_density_mode = "scalar"
    scene.thin_surface_scheduler_args = None

    pts = (torch.rand(n_points, 3, device=device) - 0.5) * 1.0
    scene.primal_points = nn.Parameter(pts)
    # Vary density values to make init-equality an interesting check.
    scene.density = nn.Parameter(
        torch.linspace(0.2, 2.0, n_points, device=device).unsqueeze(-1))

    # All-pairs fake adjacency (CSR).
    adj = []
    offsets = [0]
    for i in range(n_points):
        nbrs = [j for j in range(n_points) if j != i]
        adj.extend(nbrs)
        offsets.append(len(adj))
    scene.point_adjacency = torch.tensor(adj, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = (
        torch.tensor(offsets, dtype=torch.int32).to(torch.uint32))
    scene._cached_cell_radius = torch.ones(n_points, device=device)
    return scene


def _group_names(scene):
    return [g["name"] for g in scene.optimizer.param_groups]


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
# Test 1: config exposes the new mode + raw-side LR init/final flags.
# ---------------------------------------------------------------------------
def test_config_independent_mode_default_safe():
    """Config defaults are legacy-safe: thin_surface_density_mode='scalar'
    (never 'independent' unless opted in); raw-side LR fields exist with
    sensible defaults; the existing thin_surface_relative_delta bool is
    preserved unchanged."""
    print("\n--- Test 1: config defaults are legacy-safe ---")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import importlib
    import configargparse
    import configs  # noqa: F401  -- ensure package is importable
    from configs import OptimizationParams
    parser = configargparse.ArgumentParser()
    opt = OptimizationParams(parser)
    check(hasattr(opt, "thin_surface_density_mode"),
          "OptimizationParams exposes thin_surface_density_mode")
    check(opt.thin_surface_density_mode == "scalar",
          f"default thin_surface_density_mode == 'scalar' "
          f"(got {opt.thin_surface_density_mode!r})")
    check(opt.thin_surface_density_mode in ("scalar", "absolute",
                                              "relative", "independent"),
          f"thin_surface_density_mode is one of the documented labels "
          f"(got {opt.thin_surface_density_mode!r})")
    check(hasattr(opt, "thin_surface_raw_side_lr_init"),
          "OptimizationParams exposes thin_surface_raw_side_lr_init")
    check(hasattr(opt, "thin_surface_raw_side_lr_final"),
          "OptimizationParams exposes thin_surface_raw_side_lr_final")
    check(opt.thin_surface_raw_side_lr_init > 0,
          f"raw-side LR init is positive (got "
          f"{opt.thin_surface_raw_side_lr_init:.3e})")
    check(opt.thin_surface_raw_side_lr_final > 0,
          f"raw-side LR final is positive (got "
          f"{opt.thin_surface_raw_side_lr_final:.3e})")
    check(opt.thin_surface_relative_delta is False,
          "legacy thin_surface_relative_delta bool still defaults to False "
          "(not regressed by the new field)")


# ---------------------------------------------------------------------------
# Test 2: independent init clones density into raw_plus and raw_minus.
# ---------------------------------------------------------------------------
def test_independent_init_clones_density():
    """initialize_independent_sides: raw_plus == raw_minus == density at
    init (exact clone).  Shapes are (N, 1).  Both finite."""
    print("\n--- Test 2: init clones density into raw_plus / raw_minus ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)

    N = scene.density.shape[0]
    check(hasattr(scene, "raw_plus") and scene.raw_plus is not None,
          "raw_plus registered after init")
    check(hasattr(scene, "raw_minus") and scene.raw_minus is not None,
          "raw_minus registered after init")
    check(tuple(scene.raw_plus.shape) == (N, 1),
          f"raw_plus shape == (N, 1) (got {tuple(scene.raw_plus.shape)})")
    check(tuple(scene.raw_minus.shape) == (N, 1),
          f"raw_minus shape == (N, 1) (got {tuple(scene.raw_minus.shape)})")
    check(torch.equal(scene.raw_plus.detach(), scene.density.detach()),
          "raw_plus == density at init (exact clone)")
    check(torch.equal(scene.raw_minus.detach(), scene.density.detach()),
          "raw_minus == density at init (exact clone)")
    check(torch.equal(scene.raw_plus.detach(), scene.raw_minus.detach()),
          "raw_plus == raw_minus at init (exact clone of density)")
    check(torch.isfinite(scene.raw_plus).all()
          and torch.isfinite(scene.raw_minus).all(),
          "raw_plus and raw_minus are finite at init")


# ---------------------------------------------------------------------------
# Test 3: base density is frozen in independent mode.
# ---------------------------------------------------------------------------
def test_independent_freezes_base_density():
    """Base density is frozen: requires_grad=False, removed from the
    optimizer (no Adam group with name='density' remains)."""
    print("\n--- Test 3: base density frozen in independent mode ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)

    check(scene.density.requires_grad is False,
          "density.requires_grad is False after init")
    group_names = _group_names(scene)
    check("density" not in group_names,
          f"no 'density' optimizer group after init (got {group_names})")
    check(getattr(scene, "_density_frozen", False) is True,
          "_density_frozen sentinel set on scene")
    # Adam state for the base density must have been dropped.
    check(scene.density not in scene.optimizer.state,
          "density not in optimizer.state (Adam state cleared)")
    # primal_points is still in the optimizer (independent mode is
    # about replacing density, not points).
    check("primal_points" in group_names,
          f"primal_points still in optimizer (got {group_names})")


# ---------------------------------------------------------------------------
# Test 4: raw_plus and raw_minus are two separate ordinary Adam groups
# with an identical native raw-side LR schedule.
# ---------------------------------------------------------------------------
def test_raw_side_optimizer_groups_and_equal_schedule():
    """Two ordinary Adam groups (raw_plus, raw_minus) with identical
    native raw-side LR schedule at every iteration.  No mean_raw is
    introduced; the equal-schedule claim is the only Adam-isolation
    claim."""
    print("\n--- Test 4: separate Adam groups + equal LR schedule ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent",
                 thin_surface_raw_side_lr_init=5e-2,
                 thin_surface_raw_side_lr_final=1e-3)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)

    group_names = _group_names(scene)
    check("raw_plus" in group_names and "raw_minus" in group_names,
          f"both raw_plus and raw_minus in optimizer "
          f"(got {group_names})")
    # mean_raw must NOT exist (explicitly forbidden by the spec).
    check(not hasattr(scene, "mean_raw"),
          "mean_raw is NOT introduced (spec: do NOT introduce mean_raw)")
    # Equal initial LR.
    lr_p = _lr_for(scene, "raw_plus")
    lr_m = _lr_for(scene, "raw_minus")
    check(abs(lr_p - lr_m) < 1e-15,
          f"raw_plus and raw_minus share initial LR ({lr_p:.3e} == "
          f"{lr_m:.3e})")
    check(abs(lr_p - 5e-2) < 1e-15,
          f"raw-side LR init == 5e-2 (got {lr_p:.3e})")
    # Equal schedule at every iteration (verified by re-applying
    # update_learning_rate at a range of iterations).
    for i in [0, 1, 100, 500, 999, 1000]:
        scene.update_learning_rate(i)
        lp = _lr_for(scene, "raw_plus")
        lm = _lr_for(scene, "raw_minus")
        check(abs(lp - lm) < 1e-18,
              f"iter={i}: raw_plus LR ({lp:.6e}) == raw_minus LR "
              f"({lm:.6e})")
    # Schedulers are configured.
    check(getattr(scene, "raw_side_scheduler_args", None) is not None,
          "raw_side_scheduler_args registered on scene")


# ---------------------------------------------------------------------------
# Test 5: get_trace_data() extends to 15 elements; raw_plus/raw_minus are
# appended after the existing entries.  Scalar / relative paths still see
# None for the new fields.
# ---------------------------------------------------------------------------
def test_trace_data_appends_raw_sides():
    """get_trace_data() returns a 15-tuple: 13 legacy entries + raw_plus
    + raw_minus (appended at the end).  Scalar / relative / absolute
    paths leave the new fields as None."""
    print("\n--- Test 5: get_trace_data 15-tuple shape ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="scalar")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    # Pre-thin-surface: scalar mode.
    td = scene.get_trace_data()
    check(len(td) == 15,
          f"pre-thin-surface: get_trace_data length == 15 (got {len(td)})")
    check(td[13] is None,
          "pre-thin-surface: raw_plus slot is None (scalar mode)")
    check(td[14] is None,
          "pre-thin-surface: raw_minus slot is None (scalar mode)")

    # Independent mode.
    args2 = _args(thin_surface_density_mode="independent")
    scene2 = _make_scene(n_points=8, device="cpu")
    scene2.declare_optimizer(args2, warmup=0, max_iterations=1000)
    scene2.initialize_independent_sides(args2)
    td2 = scene2.get_trace_data()
    check(len(td2) == 15,
          f"independent: get_trace_data length == 15 (got {len(td2)})")
    check(td2[13] is not None and tuple(td2[13].shape) == (8, 1),
          f"independent: raw_plus slot is (N,1) tensor "
          f"(got type={type(td2[13]).__name__}, "
          f"shape={tuple(td2[13].shape) if td2[13] is not None else None})")
    check(td2[14] is not None and tuple(td2[14].shape) == (8, 1),
          f"independent: raw_minus slot is (N,1) tensor")


# ---------------------------------------------------------------------------
# Test 6: forward() raises NotImplementedError under independent mode.
# ---------------------------------------------------------------------------
def test_forward_fail_fast_under_independent_mode():
    """forward() must raise NotImplementedError when the discriminator
    `_thin_surface_density_mode == "independent"`.  A silent fallback to
    the scalar baseline is explicitly forbidden by the spec."""
    print("\n--- Test 6: forward() fail-fast under independent mode ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)
    rays = torch.zeros(1, 6)
    raised = False
    try:
        scene(rays)
    except NotImplementedError as e:
        raised = True
        check("Independent-side rendering" in str(e)
              or "independent" in str(e).lower(),
              f"NotImplementedError mentions independent rendering "
              f"(got {str(e)[:80]}...)")
    check(raised,
          "forward() raised NotImplementedError under independent mode "
          "(no silent scalar fallback)")

    # Sanity: scalar / absolute modes still go through the forward
    # path without raising.  We can't actually invoke the CUDA
    # kernel from CPU, but we can verify the fail-fast gate does
    # NOT fire (i.e., the error path is gated on the discriminator).
    scene2 = _make_scene(n_points=8, device="cpu")
    args2 = _args(thin_surface_density_mode="scalar")
    scene2.declare_optimizer(args2, warmup=0, max_iterations=1000)
    check(getattr(scene2, "_thin_surface_density_mode", "scalar") == "scalar",
          "scalar mode discriminator preserved (no regression)")
    raised_unwanted = False
    try:
        # We expect a non-NotImplementedError failure (radfoam stub
        # raises on .nn(...) etc.); the discriminator-specific
        # NotImplementedError must NOT fire.
        scene2(rays)
    except NotImplementedError as e:
        if "Independent-side" in str(e):
            raised_unwanted = True
    except Exception:
        # Any other exception type is fine (CUDA stub-related).
        raised_unwanted = False
    check(not raised_unwanted,
          "scalar mode does NOT trigger the independent-mode NotImplementedError")


# ---------------------------------------------------------------------------
# Test 7: save_pt / load_pt round-trip preserves raw tensors + discriminator.
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip_independent_sides():
    """save_pt writes raw_plus/raw_minus + density_mode='independent' to
    the `thin_surface` metadata block.  load_pt restores the tensors and
    flips `_thin_surface_density_mode` back to 'independent' without
    re-registering the optimizer groups."""
    print("\n--- Test 7: checkpoint round-trip ---")
    import tempfile
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)
    # Disturb the values so the round-trip can be checked exactly.
    with torch.no_grad():
        scene.raw_plus.data.fill_(0.7)
        scene.raw_minus.data.fill_(-0.3)
    orig_rp = scene.raw_plus.detach().clone().cpu()
    orig_rm = scene.raw_minus.detach().clone().cpu()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        scene.save_pt(path)
        # Inspect the on-disk record (no GraphicsMagick / pickle deps).
        sd = torch.load(path, map_location="cpu", weights_only=False)
        meta = sd.get("thin_surface", None)
        check(meta is not None and meta.get("density_mode") == "independent",
              f"saved thin_surface.density_mode == 'independent' "
              f"(got {meta.get('density_mode') if meta else None})")
        check(meta.get("relative_delta") is False,
              "saved relative_delta == False (mutual exclusion enforced)")
        check("raw_plus" in sd and "raw_minus" in sd,
              "saved raw_plus and raw_minus tensors present")
        check(torch.allclose(sd["raw_plus"], orig_rp, atol=1e-7),
              "saved raw_plus matches in-memory value")
        check(torch.allclose(sd["raw_minus"], orig_rm, atol=1e-7),
              "saved raw_minus matches in-memory value")

        # Reload into a fresh scene.
        loaded = _make_scene(n_points=8, device="cpu")
        loaded.load_pt(path)
        check(loaded._thin_surface_density_mode == "independent",
              f"loaded _thin_surface_density_mode == 'independent' "
              f"(got {loaded._thin_surface_density_mode!r})")
        check(hasattr(loaded, "raw_plus") and loaded.raw_plus is not None,
              "raw_plus restored by load_pt")
        check(hasattr(loaded, "raw_minus") and loaded.raw_minus is not None,
              "raw_minus restored by load_pt")
        check(torch.allclose(loaded.raw_plus.detach().cpu(), orig_rp,
                              atol=1e-7),
              "raw_plus matches after round-trip")
        check(torch.allclose(loaded.raw_minus.detach().cpu(), orig_rm,
                              atol=1e-7),
              "raw_minus matches after round-trip")


# ---------------------------------------------------------------------------
# Test 8: malformed / mixed-state rejection at save_pt and load_pt.
# ---------------------------------------------------------------------------
def test_checkpoint_mixed_state_rejection():
    """save_pt refuses to write a checkpoint when both relative_delta
    AND density_mode='independent' are active.  load_pt refuses to
    load a checkpoint that carries both.  Both errors must mention
    the conflict."""
    print("\n--- Test 8: malformed / mixed-state rejection ---")
    import tempfile
    # Save-time mixed state: flip the flag the same way initialize_
    # thin_surface would, then attempt save_pt.
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="scalar",
                 thin_surface_relative_delta=False)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(_args(thin_surface_density_mode=
                                              "independent"))
    # Now flip the relative flag as if initialize_thin_surface had run
    # with relative_delta=True.  save_pt must refuse.
    scene._thin_surface_relative_delta = True
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        raised = False
        try:
            scene.save_pt(path)
        except RuntimeError as e:
            raised = True
            check("mutually" in str(e).lower() or "mixed" in str(e).lower(),
                  f"save_pt RuntimeError mentions mutually-exclusive / "
                  f"mixed state (got {str(e)[:80]}...)")
        check(raised,
              "save_pt refused to write a mixed-state checkpoint")

    # Load-time malformed: fabricate a checkpoint dict on disk that
    # has both density_mode='independent' AND relative_delta=True.
    # Note: torch.save cannot serialize uint32 tensors directly, so
    # the fixture uses int32 (load_pt converts to uint32 internally).
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        N = 5
        sd = {
            "xyz": torch.zeros(N, 3),
            "density": torch.zeros(N, 1),
            "adjacency": torch.zeros(N, dtype=torch.int32),
            "adjacency_offsets": torch.arange(0, N + 1, dtype=torch.int32),
            "thin_surface": {
                "active": False,
                "density_mode": "independent",
                "relative_delta": True,
                "K": 4,
                "start": -1,
            },
            "raw_plus": torch.zeros(N, 1),
            "raw_minus": torch.zeros(N, 1),
        }
        torch.save(sd, path)
        loaded = _make_scene(n_points=N, device="cpu")
        raised = False
        try:
            loaded.load_pt(path)
        except RuntimeError as e:
            raised = True
            check("mutually" in str(e).lower() or "mixed" in str(e).lower()
                  or "malformed" in str(e).lower(),
                  f"load_pt RuntimeError mentions mutually-exclusive / "
                  f"mixed state (got {str(e)[:80]}...)")
        check(raised,
              "load_pt refused to load a mixed-state checkpoint")

    # Load-time malformed: density_mode='independent' but raw_plus/raw_minus
    # tensors are missing.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        N = 5
        sd = {
            "xyz": torch.zeros(N, 3),
            "density": torch.zeros(N, 1),
            "adjacency": torch.zeros(N, dtype=torch.int32),
            "adjacency_offsets": torch.arange(0, N + 1, dtype=torch.int32),
            "thin_surface": {
                "active": False,
                "density_mode": "independent",
                "relative_delta": False,
                "K": 4,
                "start": -1,
            },
        }
        torch.save(sd, path)
        loaded = _make_scene(n_points=N, device="cpu")
        raised = False
        try:
            loaded.load_pt(path)
        except RuntimeError as e:
            raised = True
            check("malformed" in str(e).lower() or "raw_plus" in str(e),
                  f"load_pt RuntimeError mentions raw tensor missing "
                  f"(got {str(e)[:80]}...)")
        check(raised,
              "load_pt refused an independent-mode checkpoint missing "
              "raw tensors")


# ---------------------------------------------------------------------------
# Test 9: legacy checkpoint inference is unchanged.
# ---------------------------------------------------------------------------
def test_legacy_checkpoint_inference_unchanged():
    """A legacy checkpoint (no `density_mode` in the metadata) is loaded
    with the discriminator inferred from the existing flags, NOT as
    'independent'.  Pre-existing thin-surface blocks keep their
    'absolute' / 'relative' semantics."""
    print("\n--- Test 9: legacy checkpoint inference ---")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        N = 5
        # Legacy baseline: no thin_surface block at all.
        sd = {
            "xyz": torch.zeros(N, 3),
            "density": torch.zeros(N, 1),
            "adjacency": torch.zeros(N, dtype=torch.int32),
            "adjacency_offsets": torch.arange(0, N + 1,
                                              dtype=torch.int32),
        }
        torch.save(sd, path)
        loaded = _make_scene(n_points=N, device="cpu")
        loaded.load_pt(path)
        check(loaded._thin_surface_density_mode == "scalar",
              f"baseline checkpoint -> discriminator='scalar' "
              f"(got {loaded._thin_surface_density_mode!r})")
        check(getattr(loaded, "_thin_surface_active", False) is False,
              "baseline checkpoint -> _thin_surface_active is False")

    # Legacy bounded-delta: active=True, no density_mode -> absolute.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        N = 5
        K = 4
        angles = torch.linspace(0, 2 * math.pi, K + 1)[:-1]
        sites = torch.stack([torch.cos(angles) * 0.4,
                              torch.sin(angles) * 0.4], dim=-1)
        sd = {
            "xyz": torch.zeros(N, 3),
            "density": torch.zeros(N, 1),
            "adjacency": torch.zeros(N, dtype=torch.int32),
            "adjacency_offsets": torch.arange(0, N + 1,
                                              dtype=torch.int32),
            "density_delta": torch.zeros(N, 1),
            "quaternions": torch.cat([
                torch.ones(N, 1),
                torch.zeros(N, 3),
            ], dim=-1),
            "texel_sites_2d": sites.unsqueeze(0).expand(N, -1, -1).clone(),
            "texel_heights": torch.zeros(N, K),
            "thin_surface": {
                "active": True, "K": K, "start": -1,
                "scheduler_cfg": None,
                "relative_delta": False, "delta_max_frac": 0.5,
            },
        }
        torch.save(sd, path)
        loaded = _make_scene(n_points=N, device="cpu")
        loaded.load_pt(path)
        check(loaded._thin_surface_density_mode == "absolute",
              f"legacy absolute -> 'absolute' "
              f"(got {loaded._thin_surface_density_mode!r})")

    # Legacy relative: active=True, relative_delta=True -> 'relative'.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pt")
        N = 5
        K = 4
        angles = torch.linspace(0, 2 * math.pi, K + 1)[:-1]
        sites = torch.stack([torch.cos(angles) * 0.4,
                              torch.sin(angles) * 0.4], dim=-1)
        sd = {
            "xyz": torch.zeros(N, 3),
            "density": torch.zeros(N, 1),
            "adjacency": torch.zeros(N, dtype=torch.int32),
            "adjacency_offsets": torch.arange(0, N + 1,
                                              dtype=torch.int32),
            "density_delta": torch.zeros(N, 1),
            "quaternions": torch.cat([
                torch.ones(N, 1),
                torch.zeros(N, 3),
            ], dim=-1),
            "texel_sites_2d": sites.unsqueeze(0).expand(N, -1, -1).clone(),
            "texel_heights": torch.zeros(N, K),
            "thin_surface": {
                "active": True, "K": K, "start": -1,
                "scheduler_cfg": None,
                "relative_delta": True, "delta_max_frac": 0.5,
            },
        }
        torch.save(sd, path)
        loaded = _make_scene(n_points=N, device="cpu")
        loaded.load_pt(path)
        check(loaded._thin_surface_density_mode == "relative",
              f"legacy relative -> 'relative' "
              f"(got {loaded._thin_surface_density_mode!r})")


# ---------------------------------------------------------------------------
# Test 10: permute / prune alignment for raw_plus / raw_minus.
# ---------------------------------------------------------------------------
def test_permute_prune_aligns_raw_sides():
    """After permute_points / prune_points, raw_plus and raw_minus are
    permuted/pruned in lock-step with primal_points so they stay
    aligned with the surviving cell rows."""
    print("\n--- Test 10: permute / prune tensor alignment ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)
    # Distinguish values row-by-row so permute/prune can be checked.
    with torch.no_grad():
        for i in range(8):
            scene.raw_plus.data[i, 0] = float(i)
            scene.raw_minus.data[i, 0] = float(i) + 0.5
    orig_pp = scene.primal_points.detach().clone()
    orig_rp = scene.raw_plus.detach().clone()
    orig_rm = scene.raw_minus.detach().clone()

    # Reverse permutation.
    perm = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.long)
    scene.permute_points(perm)
    expected_pp = orig_pp[perm]
    expected_rp = orig_rp[perm]
    expected_rm = orig_rm[perm]
    check(torch.allclose(scene.primal_points.detach(), expected_pp, atol=1e-7),
          "permute: primal_points aligned")
    check(torch.allclose(scene.raw_plus.detach(), expected_rp, atol=1e-7),
          "permute: raw_plus aligned in lock-step with primal_points")
    check(torch.allclose(scene.raw_minus.detach(), expected_rm, atol=1e-7),
          "permute: raw_minus aligned in lock-step with primal_points")

    # Prune the first 3 cells.
    mask = torch.zeros(8, dtype=torch.bool, device="cpu")
    mask[:3] = True
    pre_prune_rp = scene.raw_plus.detach().clone()
    pre_prune_rm = scene.raw_minus.detach().clone()
    scene.prune_points(mask)
    check(scene.raw_plus.shape[0] == 5,
          f"prune: raw_plus shrunk to 5 rows (got {scene.raw_plus.shape[0]})")
    check(scene.raw_minus.shape[0] == 5,
          f"prune: raw_minus shrunk to 5 rows "
          f"(got {scene.raw_minus.shape[0]})")
    check(torch.allclose(scene.raw_plus.detach(),
                          pre_prune_rp[3:], atol=1e-7),
          "prune: raw_plus kept the surviving rows")
    check(torch.allclose(scene.raw_minus.detach(),
                          pre_prune_rm[3:], atol=1e-7),
          "prune: raw_minus kept the surviving rows")
    # Optimizer groups still reference raw_plus / raw_minus.
    check(_lr_for(scene, "raw_plus") is not None,
          "prune: raw_plus optimizer group still registered")
    check(_lr_for(scene, "raw_minus") is not None,
          "prune: raw_minus optimizer group still registered")


# ---------------------------------------------------------------------------
# Test 11: independent-side diagnostics reports physical side mean/contrast.
# ---------------------------------------------------------------------------
def test_independent_side_diagnostics_keys():
    """independent_side_diagnostics() returns a dict with the physical
    side mean / contrast keys (read-only).  Does not affect losses or
    rendering -- a pure observability hook."""
    print("\n--- Test 11: independent-side diagnostics ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_independent_sides(args)
    d = scene.independent_side_diagnostics()
    check(d is not None, "diagnostics returns a dict (not None)")
    for k in ("side_physical_plus_mean", "side_physical_minus_mean",
              "side_physical_contrast_mean",
              "side_physical_contrast_p95", "side_physical_contrast_max",
              "side_raw_diff_mean", "base_density_frozen"):
        check(k in d, f"diagnostics has {k!r}")
    check(d["base_density_frozen"] == 1.0,
          f"base_density_frozen == 1.0 under independent mode "
          f"(got {d['base_density_frozen']})")
    # Inactive -> None.
    scene2 = _make_scene(n_points=4, device="cpu")
    args2 = _args(thin_surface_density_mode="scalar")
    scene2.declare_optimizer(args2, warmup=0, max_iterations=1000)
    check(scene2.independent_side_diagnostics() is None,
          "diagnostics returns None when independent mode is inactive")


# ---------------------------------------------------------------------------
# Test 12: initialize_thin_surface dispatches on density_mode.
# ---------------------------------------------------------------------------
def test_initialize_thin_surface_dispatches_on_mode():
    """initialize_thin_surface(args, K) routes to
    initialize_independent_sides when args.thin_surface_density_mode ==
    'independent'.  Scalar / absolute / relative paths take the legacy
    path; the discriminator is set on the scene for round-trip."""
    print("\n--- Test 12: initialize_thin_surface mode dispatch ---")
    # Independent dispatch.
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="independent")
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)
    check(scene._thin_surface_density_mode == "independent",
          "initialize_thin_surface dispatches to independent init when "
          "density_mode == 'independent'")
    check(hasattr(scene, "raw_plus") and hasattr(scene, "raw_minus"),
          "independent dispatch registers raw_plus / raw_minus")
    # Scalar (no thin-surface_active).
    scene2 = _make_scene(n_points=8, device="cpu")
    args2 = _args(thin_surface_density_mode="scalar")
    scene2.declare_optimizer(args2, warmup=0, max_iterations=1000)
    scene2.initialize_thin_surface(args2, K=4)
    # Note: initialize_thin_surface is the activation function.  When
    # called it transitions the discriminator to 'absolute' (or
    # 'relative' if the relative-delta flag is set) regardless of
    # args.thin_surface_density_mode, since calling the activation
    # function means the user has explicitly chosen the bounded-delta
    # path.  Only the explicit mode labels (absolute/relative) are
    # respected; 'scalar' is the default-only sentinel for callers
    # that never activate thin-surface.
    check(scene2._thin_surface_density_mode == "absolute",
          f"initialize_thin_surface with scalar flag activates "
          f"'absolute' (legacy safe); got "
          f"{scene2._thin_surface_density_mode!r}")
    check(not hasattr(scene2, "raw_plus"),
          "scalar mode: raw_plus NOT registered")


# ---------------------------------------------------------------------------
# Test 13: mutually-exclusive validation when constructing the optimizer
# path: relative+independent combination raises.
# ---------------------------------------------------------------------------
def test_mutually_exclusive_relative_independent():
    """A config that activates both thin_surface_relative_delta=True
    AND thin_surface_density_mode='independent' is rejected at
    initialize_independent_sides with a clear ValueError."""
    print("\n--- Test 13: mutually exclusive relative + independent ---")
    scene = _make_scene(n_points=8, device="cpu")
    args = _args(thin_surface_density_mode="scalar",
                 thin_surface_relative_delta=False)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    # Switch to a relative+independent conflict.
    args2 = _args(thin_surface_density_mode="independent",
                  thin_surface_relative_delta=True)
    raised = False
    try:
        scene.initialize_independent_sides(args2)
    except ValueError as e:
        raised = True
        check("mutually exclusive" in str(e).lower()
              or "relative" in str(e).lower(),
              f"ValueError mentions mutually exclusive / relative "
              f"(got {str(e)[:80]}...)")
    check(raised,
          "initialize_independent_sides refused a relative+independent "
          "config (no silent mode coercion)")


def main():
    print("=" * 60)
    print("LC64 plan v3 — Independent-side Commit 1 schema tests")
    print("Spec: specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v2.md (E0 amendment)")
    print("=" * 60)

    test_config_independent_mode_default_safe()
    test_independent_init_clones_density()
    test_independent_freezes_base_density()
    test_raw_side_optimizer_groups_and_equal_schedule()
    test_trace_data_appends_raw_sides()
    test_forward_fail_fast_under_independent_mode()
    test_checkpoint_roundtrip_independent_sides()
    test_checkpoint_mixed_state_rejection()
    test_legacy_checkpoint_inference_unchanged()
    test_permute_prune_aligns_raw_sides()
    test_independent_side_diagnostics_keys()
    test_initialize_thin_surface_dispatches_on_mode()
    test_mutually_exclusive_relative_independent()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL INDEPENDENT-SIDE COMMIT-1 TESTS PASSED.")


if __name__ == "__main__":
    main()
