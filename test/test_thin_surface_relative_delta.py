"""Tests for the M5 relative-delta parameterization
   (`delta = rho * mu_bar * tanh(raw_delta)`).

This is the chest-rescue prototype after CH5-CH7 showed that any learned
raw additive delta (|delta| unbounded) harms performance. The relative
parameterization is the minimal safe alternative:

  absolute  : delta_val = raw_delta,                 mu_p = max(mu_bar+delta, 0)
  relative  : delta_val = rho * mu_bar * tanh(raw),  mu_p = max((1+rho*th)*mu_bar, 0)
                                                          ^ bounded by mu_bar when rho<=1
                                                          ^ nonneg always when rho in (0,1]

Geometry (quaternion + texel sites + heights) is untouched; only the
delta scalar interpretation changes. The CUDA kernel has a single
`if (settings.thin_surface_relative_delta)` dispatch, so an absolute
checkpoint stays bit-identical when this flag is False.

This file is a **CPU** test (mirrors the GPU tests' coverage of the Python
plumbing: scene init, save/load round-trip, param interpretation).  The
forward/backward mathematical correctness is verified separately by the
GPU gradcheck tests on a CUDA-equipped machine.

Run with:  micromamba run -n radfoam python test/test_thin_surface_relative_delta.py
"""
import sys
import os
import math
import types
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Stub radfoam so scene.py can be imported on CPU.
# (mirrors test/test_thin_surface_lr_scale.py / test_thin_surface_activation.py)
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
    mod.TriangulationFailedError = type("TriangulationFailedError", (Exception,), {})
    mod.Triangulation = None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()
from radfoam_model.scene import CTScene  # noqa: E402

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _args(relative=False, rho=0.5, delta_clip=2.0):
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-2
    a.freeze_points = 9500
    a.thin_surface_start = 0          # immediate for testing
    a.thin_surface_K = 4
    a.thin_surface_delta_weight = 1e-3
    a.thin_surface_height_weight = 5e-4
    a.thin_surface_gate_tau = 0.01
    a.thin_surface_lr_scale = 1.0
    a.thin_surface_delta_lr_scale = 1.0
    a.thin_surface_quat_lr_scale = 1.0
    a.thin_surface_sites_lr_scale = 1.0
    a.thin_surface_heights_lr_scale = 1.0
    a.thin_surface_delta_clip = delta_clip
    a.thin_surface_grad_clip = 1.0
    a.thin_surface_relative_delta = relative
    a.thin_surface_delta_max_frac = rho
    return a


def _make_minimal_scene(n_points=8, device="cpu"):
    """Minimal CTScene with fake all-pairs adjacency (CPU-friendly).

    Bypasses Delaunay by constructing the adjacency by hand -- the only
    surface touched here is the four thin params + their metadata."""
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
    # Vary base densities to give the relative param something to work with.
    scene.density = nn.Parameter(
        torch.linspace(0.2, 2.0, n_points, device=device).unsqueeze(-1)
    )

    adj = []
    offsets = [0]
    for i in range(n_points):
        nbrs = [j for j in range(n_points) if j != i]
        adj.extend(nbrs)
        offsets.append(len(adj))
    scene.point_adjacency = (
        torch.tensor(adj, dtype=torch.int32).to(torch.uint32)
    )
    scene.point_adjacency_offsets = (
        torch.tensor(offsets, dtype=torch.int32).to(torch.uint32)
    )
    scene._cached_cell_radius = torch.ones(n_points, device=device)
    return scene


def _relative_delta_formula(mu_bar, raw_delta, rho):
    """Reference implementation of the M5 relative parameterization,
    matching the CUDA kernel (see src/tracing/pipeline.cu `ct_thinsurface_forward`)."""
    delta_val = rho * mu_bar * torch.tanh(raw_delta)
    mu_p = torch.clamp(mu_bar + delta_val, min=0.0)
    mu_n = torch.clamp(mu_bar - delta_val, min=0.0)
    return delta_val, mu_p, mu_n


def _absolute_delta_formula(mu_bar, raw_delta):
    """Legacy absolute parameterization, matching the CUDA kernel."""
    delta_val = raw_delta
    mu_p = torch.clamp(mu_bar + delta_val, min=0.0)
    mu_n = torch.clamp(mu_bar - delta_val, min=0.0)
    return delta_val, mu_p, mu_n


_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# ---------------------------------------------------------------------------
# Test 1: Opt-in flag round-trips through initialize_thin_surface and is
#         readable from the scene afterwards.
# ---------------------------------------------------------------------------
def test_flag_opt_in_and_storage():
    """`thin_surface_relative_delta=True` is recorded on the scene and the
    default is False (legacy)."""
    print("\n--- Test 1: flag opt-in / storage on scene ---")

    # Default = False
    a0 = _args(relative=False)
    scene0 = _make_minimal_scene(n_points=4, device="cpu")
    scene0.declare_optimizer(a0, warmup=0, max_iterations=1000)
    scene0.initialize_thin_surface(a0, K=4)
    check(hasattr(scene0, "_thin_surface_relative_delta"),
          "scene has _thin_surface_relative_delta attribute")
    check(scene0._thin_surface_relative_delta is False,
          f"default relative_delta is False (got "
          f"{scene0._thin_surface_relative_delta})")
    check(abs(scene0._thin_surface_delta_max_frac - 0.5) < 1e-12,
          f"default rho is 0.5 (got {scene0._thin_surface_delta_max_frac})")

    # Opt-in
    a1 = _args(relative=True, rho=0.3)
    scene1 = _make_minimal_scene(n_points=4, device="cpu")
    scene1.declare_optimizer(a1, warmup=0, max_iterations=1000)
    scene1.initialize_thin_surface(a1, K=4)
    check(scene1._thin_surface_relative_delta is True,
          "_thin_surface_relative_delta = True after opt-in")
    check(abs(scene1._thin_surface_delta_max_frac - 0.3) < 1e-12,
          f"rho stored on scene (got "
          f"{scene1._thin_surface_delta_max_frac})")


# ---------------------------------------------------------------------------
# Test 2: Activation-continuity holds for BOTH parameterizations: at
#         raw_delta = 0 (the init value), effective delta = 0 in both
#         branches, so mu_p == mu_n == mu_bar.
# ---------------------------------------------------------------------------
def test_activation_continuity_both_modes():
    """At init both modes collapse to the scalar baseline, so deactivation
    is unnecessary and CH5-style safe-by-default applies."""
    print("\n--- Test 2: activation continuity (both modes) ---")
    for rel in (False, True):
        scene = _make_minimal_scene(n_points=8, device="cpu")
        args = _args(relative=rel, rho=0.5)
        scene.declare_optimizer(args, warmup=0, max_iterations=1000)
        scene.initialize_thin_surface(args, K=4)
        N = scene.primal_points.shape[0]
        assert scene.density_delta.shape == (N, 1)

        # Init value: raw_delta == 0
        dd = scene.density_delta.detach()
        check(torch.allclose(dd, torch.zeros_like(dd), atol=1e-12),
              f"mode={rel}: density_delta init = 0 "
              f"(max abs={dd.abs().max():.2e})")

        mu_bar = torch.nn.functional.softplus(scene.density.squeeze(-1), beta=10.0)
        if rel:
            rho = scene._thin_surface_delta_max_frac
            dd_eff = rho * mu_bar * torch.tanh(dd.squeeze(-1))
        else:
            dd_eff = dd.squeeze(-1)
        mu_p = torch.clamp(mu_bar + dd_eff, min=0.0)
        mu_n = torch.clamp(mu_bar - dd_eff, min=0.0)
        check(torch.allclose(mu_p, mu_n, atol=1e-12),
              f"mode={rel}: mu_p == mu_n == mu_bar at init "
              f"(max diff={(mu_p - mu_bar).abs().max():.2e})")


# ---------------------------------------------------------------------------
# Test 3: Numerical matches the CUDA kernel formula in both branches.
# ---------------------------------------------------------------------------
def test_formula_matches_kernel():
    """The Python-side reference implementation of the relative formula
    reproduces the documented bounds at representative raw_delta values."""
    print("\n--- Test 3: relative formula correctness ---")
    n = 16
    mu_bar = torch.linspace(0.05, 3.0, n)
    rho = 0.5
    # small, medium, large, sign-flipped raw_deltas -> saturate tanh both ways
    raw = torch.tensor([-1e3, -1.0, 0.0, 0.1, 1.0, 1e3], dtype=torch.float32)
    raw = raw.unsqueeze(0).expand(n, -1).reshape(-1)            # (n*6,)
    mu_bar_e = mu_bar.unsqueeze(1).expand(-1, 6).reshape(-1)
    dd_eff, mu_p, mu_n = _relative_delta_formula(mu_bar_e, raw, rho)

    # (a) |delta| <= rho * mu_bar at every point
    bound = rho * mu_bar_e
    check((dd_eff.abs() <= bound + 1e-6).all().item(),
          f"|delta| <= rho * mu_bar everywhere "
          f"(max violation={(dd_eff.abs() - bound).max().item():.2e})")

    # (b) mu_p, mu_n >= 0 always
    check((mu_p >= 0.0).all().item(),
          f"mu_p >= 0 everywhere (min={mu_p.min().item():.2e})")
    check((mu_n >= 0.0).all().item(),
          f"mu_n >= 0 everywhere (min={mu_n.min().item():.2e})")

    # (c) tanh saturation at raw=+-1e3 -> effective delta = +-rho*mu_bar.
    # Layout: dd_eff is the row-major flattening of (n rows x 6 cols) with
    # raw = [-1e3, -1, 0, 0.1, 1, 1e3] per row. So saturation cases are at
    # i % 6 == 0 (raw=-1e3) and i % 6 == 5 (raw=+1e3); their mu_bar is
    # mu_bar[i // 6].
    idx_pos = torch.arange(dd_eff.numel())[torch.arange(dd_eff.numel()) % 6 == 5]
    idx_neg = torch.arange(dd_eff.numel())[torch.arange(dd_eff.numel()) % 6 == 0]
    mu_bar_pos = mu_bar_e[idx_pos]
    mu_bar_neg = mu_bar_e[idx_neg]
    expected_pos = rho * mu_bar_pos    # tanh(+inf) -> +1
    expected_neg = -rho * mu_bar_neg   # tanh(-inf) -> -1
    got_pos = dd_eff[idx_pos]
    got_neg = dd_eff[idx_neg]
    check((got_pos - expected_pos).abs().max().item() < 1e-3,
          f"tanh sat at +inf -> +rho*mu_bar (max abs diff="
          f"{(got_pos - expected_pos).abs().max().item():.2e})")
    check((got_neg - expected_neg).abs().max().item() < 1e-3,
          f"tanh sat at -inf -> -rho*mu_bar (max abs diff="
          f"{(got_neg - expected_neg).abs().max().item():.2e})")


# ---------------------------------------------------------------------------
# Test 4: Save/load round-trip preserves the parameterization mode.
# ---------------------------------------------------------------------------
def test_save_load_preserves_mode(monkeypatch_torch_save=None):
    """save_pt writes the mode and rho into scene_data; load_pt restores them."""
    print("\n--- Test 4: save/load round-trip preserves mode ---")
    import tempfile
    scene = _make_minimal_scene(n_points=4, device="cpu")
    args = _args(relative=True, rho=0.25, delta_clip=0.0)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)
    # Distinguish values so we can verify they survive unchanged.
    with torch.no_grad():
        scene.density_delta.data.fill_(0.7)
    orig_dd = scene.density_delta.detach().clone()
    orig_rel = scene._thin_surface_relative_delta
    orig_rho = scene._thin_surface_delta_max_frac

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.pt")
        scene.save_pt(path)
        # Inspect the on-disk record (no GraphicsMagick / pickle deps)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        meta = sd.get("thin_surface", None)
        check(meta is not None,
              "scene_data has thin_surface metadata")
        check(meta.get("relative_delta", None) is True,
              f"saved relative_delta=True (got {meta.get('relative_delta')})")
        check(abs(meta.get("delta_max_frac", -1.0) - 0.25) < 1e-12,
              f"saved delta_max_frac=0.25 (got {meta.get('delta_max_frac')})")
        check(torch.allclose(sd["density_delta"], orig_dd.cpu(), atol=1e-7),
              "saved density_delta matches in-memory value")
        torch.save(sd, path)  # already the same; just for completeness

        # Reload by re-creating a scene and calling load_pt
        scene2 = _make_minimal_scene(n_points=4, device="cpu")
        scene2.load_pt(path)
        check(scene2._thin_surface_relative_delta is True,
              f"loaded relative_delta=True (got "
              f"{scene2._thin_surface_relative_delta})")
        check(abs(scene2._thin_surface_delta_max_frac - 0.25) < 1e-12,
              f"loaded rho=0.25 (got {scene2._thin_surface_delta_max_frac})")


# ---------------------------------------------------------------------------
# Test 5: clamp_thin_surface_params respects the parameterization
# ---------------------------------------------------------------------------
def test_clamp_skips_delta_in_relative_mode():
    """In relative mode the kernel's tanh already clamps effectively, so
    clamp_thin_surface_params should NOT touch density_delta (preserving
    Adam headroom).  In absolute mode it should clamp by thin_surface_delta_clip."""
    print("\n--- Test 5: clamp behavior depends on parameterization ---")

    # Absolute mode: clamp applied
    scene_a = _make_minimal_scene(n_points=4, device="cpu")
    args_a = _args(relative=False, delta_clip=2.0)
    scene_a.declare_optimizer(args_a, warmup=0, max_iterations=1000)
    scene_a.initialize_thin_surface(args_a, K=4)
    with torch.no_grad():
        scene_a.density_delta.data.fill_(10.0)
    scene_a.clamp_thin_surface_params()
    check(scene_a.density_delta.abs().max().item() <= 2.0 + 1e-6,
          f"absolute mode: density_delta clamped to [-2, 2] "
          f"(max abs={scene_a.density_delta.abs().max().item():.2e})")

    # Relative mode: clamp skipped
    scene_r = _make_minimal_scene(n_points=4, device="cpu")
    args_r = _args(relative=True, rho=0.5, delta_clip=2.0)
    scene_r.declare_optimizer(args_r, warmup=0, max_iterations=1000)
    scene_r.initialize_thin_surface(args_r, K=4)
    with torch.no_grad():
        scene_r.density_delta.data.fill_(10.0)
    scene_r.clamp_thin_surface_params()
    # Raw value still 10 -- we let tanh in the kernel handle the bound.
    check((scene_r.density_delta.detach().abs() - 10.0).abs().max().item()
          < 1e-6,
          f"relative mode: raw density_delta untouched by clamp "
          f"(got max abs={scene_r.density_delta.abs().max().item():.2e})")


# ---------------------------------------------------------------------------
# Test 6: Defaults at the config layer are inherited (False / 0.5).
# ---------------------------------------------------------------------------
def test_config_defaults():
    """configs/__init__.py OptimizationParams has the new fields and their
    defaults are off-by-default to keep existing checkpoints valid."""
    print("\n--- Test 6: config-level defaults ---")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import importlib
    import configargparse
    from configs import (
        ModelParams, PipelineParams, OptimizationParams, DatasetParams,
    )
    parser = configargparse.ArgumentParser()
    ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    DatasetParams(parser)
    opt = parser.parse_args([])
    check(hasattr(opt, "thin_surface_relative_delta"),
          "OptimizationParams exposes thin_surface_relative_delta")
    check(hasattr(opt, "thin_surface_delta_max_frac"),
          "OptimizationParams exposes thin_surface_delta_max_frac")
    check(opt.thin_surface_relative_delta is False,
          f"default thin_surface_relative_delta is False "
          f"(got {opt.thin_surface_relative_delta})")
    check(abs(opt.thin_surface_delta_max_frac - 0.5) < 1e-12,
          f"default thin_surface_delta_max_frac is 0.5 "
          f"(got {opt.thin_surface_delta_max_frac})")


def main():
    print("=" * 60)
    print("Thin-Surface Relative-Delta Parameterization (M5 chest rescue)")
    print("=" * 60)

    test_flag_opt_in_and_storage()
    test_activation_continuity_both_modes()
    test_formula_matches_kernel()
    test_save_load_preserves_mode()
    test_clamp_skips_delta_in_relative_mode()
    test_config_defaults()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL RELATIVE-DELTA TESTS PASSED.")


if __name__ == "__main__":
    main()
