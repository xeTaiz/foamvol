"""Parse + config-emission tests for the per-group rescue driver flags.

R2 / R3 operational support (added in commit landing the --thin-delta-lr-scale
and --thin-geometry-lr-scale CLI flags on tracked old/test_cube.py):

  --thin-delta-lr-scale     : scalar multiplier on density_delta param-group LR
                             (R2: 0.01 = train delta at 1% of base)
  --thin-geometry-lr-scale  : scalar multiplier fanned out to quaternions +
                             texel_sites_2d + texel_heights param-group LRs
                             (R3: 0 = freeze the geometry)
  --thin-lr-scale           : existing global multiplier (R1: 0 = freeze all)

All three default 1.0 (preserves the failed recipe).  They are applied in
order: per-group (delta/geometry) first, then global, in initialize_thin_surface:

  lr_init[name] = base * thin_surface_lr_scale * thin_surface_{name}_lr_scale

This test verifies:
  1. CLI: --help exposes both new flags.
  2. CLI defaults are 1.0 (no behavioural change without explicit flags).
  3. Driver emits the matching YAML keys when --thin-surface is enabled.
  4. Driver omits the per-group keys when --thin-surface is NOT enabled
     (preserves historical cfg shape).
  5. Geometry flag fans out to the three geometry keys (delta key is NOT
     affected by the geometry flag).
  6. R2 invocation emits:
       thin_surface_lr_scale            = 1.0  (preserves the global)
       thin_surface_delta_lr_scale       = 0.01
       thin_surface_quat_lr_scale        = 0.0
       thin_surface_sites_lr_scale       = 0.0
       thin_surface_heights_lr_scale     = 0.0
  7. End-to-end with stubbed radfoam: the four param-group LRs are
     computed as base * global * per_group, e.g. R2 gives
     density_delta = 5e-3 * 1.0 * 0.01 = 5e-5
     quaternions   = 5e-3 * 1.0 * 0.0  = 0.0
     sites        = 5e-3 * 1.0 * 0.0  = 0.0
     heights      = 5e-3 * 1.0 * 0.0  = 0.0

Run with:  micromamba run -n radfoam python test/test_thin_surface_per_group_drivers.py
"""
import sys
import os
import math
import subprocess
import warnings
warnings.filterwarnings("ignore")

import torch

# Repo root on path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Driver on path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                   "old")))


_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _capture_config(monkey_argv):
    """Invoke the driver with `monkey_argv` (replaces sys.argv).  The driver
    will try to subprocess.run train.py; we patch that to a no-op so we
    just capture the emitted cfg path."""
    import test_cube
    captured = {}

    def fake_run(cmd, cwd=None, **kw):
        captured["cfg_path"] = cmd[cmd.index("-c") + 1]
        captured["cmd"] = cmd
        class _R:
            returncode = 0
        return _R()

    real_run = subprocess.run
    subprocess.run = fake_run
    try:
        import sys as _s
        old = _s.argv
        _s.argv = ["test_cube.py"] + monkey_argv
        try:
            test_cube.main()
        except SystemExit:
            pass
        finally:
            _s.argv = old
    finally:
        subprocess.run = real_run
    return captured


def _read_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


# Test 1: --help exposes the new flags.
def test_help_exposes_new_flags():
    """Run the driver as a subprocess with --help, capture stdout, assert
    both new flag names appear in the usage line."""
    import os
    import sys as _s
    import subprocess as _sp
    driver = os.path.join(os.path.dirname(__file__), "..", "old", "test_cube.py")
    result = _sp.run(
        [_s.executable, driver, "--help"],
        capture_output=True, text=True, timeout=15,
    )
    out = (result.stdout or "") + (result.stderr or "")
    check("--thin-delta-lr-scale" in out,
          "--thin-delta-lr-scale appears in --help")
    check("--thin-geometry-lr-scale" in out,
          "--thin-geometry-lr-scale appears in --help")
    check("R2 operational support" in out,
          "help text references R2")
    check("R3 operational support" in out,
          "help text references R3")
    # And the existing flags are still there.
    check("--run-tag" in out, "--run-tag still present")
    check("--thin-lr-scale" in out, "--thin-lr-scale still present")


# Test 2: defaults are 1.0 (no flag) -> 1.0 in cfg, behaviour preserved.
def test_defaults_preserved_when_no_flag():
    print("\n--- Test 2: defaults preserved when no flag ---")
    captured = _capture_config(["--test", "1b", "--thin-surface"])
    cfg = _read_cfg(captured["cfg_path"])
    check(cfg.get("thin_surface_lr_scale") == 1.0,
          f"default thin_surface_lr_scale in cfg: {cfg.get('thin_surface_lr_scale')}")
    check(cfg.get("thin_surface_delta_lr_scale") == 1.0,
          f"default thin_surface_delta_lr_scale: {cfg.get('thin_surface_delta_lr_scale')}")
    check(cfg.get("thin_surface_quat_lr_scale") == 1.0,
          f"default thin_surface_quat_lr_scale: {cfg.get('thin_surface_quat_lr_scale')}")
    check(cfg.get("thin_surface_sites_lr_scale") == 1.0,
          f"default thin_surface_sites_lr_scale: {cfg.get('thin_surface_sites_lr_scale')}")
    check(cfg.get("thin_surface_heights_lr_scale") == 1.0,
          f"default thin_surface_heights_lr_scale: {cfg.get('thin_surface_heights_lr_scale')}")


# Test 3: per-group keys are emitted only when --thin-surface is on.
def test_keys_absent_when_thin_surface_off():
    print("\n--- Test 3: per-group keys absent when --thin-surface is off ---")
    captured = _capture_config(["--test", "1b"])
    cfg = _read_cfg(captured["cfg_path"])
    check("thin_surface_lr_scale" not in cfg,
          "thin_surface_lr_scale absent (no thin-surface)")
    check("thin_surface_delta_lr_scale" not in cfg,
          "thin_surface_delta_lr_scale absent (no thin-surface)")
    check("thin_surface_quat_lr_scale" not in cfg,
          "thin_surface_quat_lr_scale absent (no thin-surface)")


# Test 4: --thin-geometry-lr-scale fans out to the three geometry keys,
# but NOT to density_delta.
def test_geometry_flag_fans_out_to_three_geometry_keys():
    print("\n--- Test 4: --thin-geometry-lr-scale fans out to 3 keys ---")
    captured = _capture_config([
        "--test", "1b", "--thin-surface",
        "--run-tag", "R3",
        "--thin-geometry-lr-scale", "0",
    ])
    cfg = _read_cfg(captured["cfg_path"])
    check(cfg["thin_surface_lr_scale"] == 1.0,
          "global scale default (1.0) preserved")
    check(cfg["thin_surface_delta_lr_scale"] == 1.0,
          f"delta UNCHANGED by --thin-geometry-lr-scale "
          f"(got {cfg['thin_surface_delta_lr_scale']})")
    check(cfg["thin_surface_quat_lr_scale"] == 0.0,
          f"quat scales to 0 (got {cfg['thin_surface_quat_lr_scale']})")
    check(cfg["thin_surface_sites_lr_scale"] == 0.0,
          f"sites scales to 0 (got {cfg['thin_surface_sites_lr_scale']})")
    check(cfg["thin_surface_heights_lr_scale"] == 0.0,
          f"heights scales to 0 (got {cfg['thin_surface_heights_lr_scale']})")


# Test 5: --thin-delta-lr-scale applies only to density_delta.
def test_delta_flag_applies_only_to_delta():
    print("\n--- Test 5: --thin-delta-lr-scale applies only to delta ---")
    captured = _capture_config([
        "--test", "1b", "--thin-surface",
        "--run-tag", "R2_partial",
        "--thin-delta-lr-scale", "0.5",
    ])
    cfg = _read_cfg(captured["cfg_path"])
    check(cfg["thin_surface_delta_lr_scale"] == 0.5,
          f"delta scales to 0.5 (got {cfg['thin_surface_delta_lr_scale']})")
    check(cfg["thin_surface_quat_lr_scale"] == 1.0,
          "quat UNCHANGED (default 1.0)")
    check(cfg["thin_surface_sites_lr_scale"] == 1.0,
          "sites UNCHANGED (default 1.0)")
    check(cfg["thin_surface_heights_lr_scale"] == 1.0,
          "heights UNCHANGED (default 1.0)")


# Test 6: full R2 invocation emits the expected R2 keys exactly.
def test_full_r2_invocation():
    print("\n--- Test 6: full R2 invocation ---")
    captured = _capture_config([
        "--test", "1b", "--thin-surface",
        "--run-tag", "R2",
        "--thin-lr-scale", "1.0",
        "--thin-delta-lr-scale", "0.01",
        "--thin-geometry-lr-scale", "0",
    ])
    cfg = _read_cfg(captured["cfg_path"])
    expected = {
        "thin_surface_lr_scale": 1.0,
        "thin_surface_delta_lr_scale": 0.01,
        "thin_surface_quat_lr_scale": 0.0,
        "thin_surface_sites_lr_scale": 0.0,
        "thin_surface_heights_lr_scale": 0.0,
    }
    for k, v in expected.items():
        check(cfg.get(k) == v,
              f"R2 cfg: {k} = {cfg.get(k)} (expected {v})")
    # The output dir / experiment name were tagged R2.
    cfg_path = captured["cfg_path"]
    check("R2" in cfg_path,
          f"R2 cfg path includes R2 tag: {cfg_path}")


# Test 7: end-to-end runtime proof -- the four thin param-group LRs are
# computed correctly in CTScene.initialize_thin_surface.  This is the
# R2 invariant: density_delta trains at base * global * 0.01 = 5e-5;
# the three geometry params train at 0.
def test_r2_runtime_lr_proof():
    print("\n--- Test 7: R2 runtime LR proof (stubbed radfoam) ---")
    import types
    mod = types.ModuleType("radfoam")
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda pts, adj, off, **kw: (
        __import__("torch").zeros(pts.shape[0], dtype=__import__("torch").long),
        __import__("torch").ones(pts.shape[0], device=pts.device),
    )
    mod.nn = lambda points, tree, query, **kw: __import__("torch").zeros(
        query.shape[0], dtype=__import__("torch").long, device=query.device)
    mod.BatchFetcher = lambda *a, **k: None
    mod.TriangulationFailedError = type("T", (Exception,), {})
    mod.Triangulation = None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod

    import torch
    import torch.nn as nn
    from radfoam_model.scene import CTScene

    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device("cpu")
    N = 8
    scene.num_init_points = N
    scene.num_final_points = N
    scene._thin_surface_active = False
    scene._thin_K = 4
    scene._thin_surface_gate_tau = 0.01
    scene.thin_surface_scheduler_args = None
    scene.primal_points = nn.Parameter(torch.zeros(N, 3))
    scene.density = nn.Parameter(0.5 * torch.ones(N, 1))
    adj = list(range(1, N)) + list(range(0, N-1))
    adj = adj * N
    offsets = [i * (N-1) for i in range(N+1)]
    scene.point_adjacency = torch.tensor(adj, dtype=torch.int32).to(torch.uint32)
    scene.point_adjacency_offsets = torch.tensor(offsets, dtype=torch.int32).to(torch.uint32)
    scene._cached_cell_radius = torch.ones(N)

    A = type("A", (), {})()
    A.density_lr_init = 5e-2
    A.density_lr_final = 1e-2
    A.points_lr_init = 2e-4
    A.points_lr_final = 5e-6
    A.freeze_points = 9500
    A.thin_surface_start = 0
    A.thin_surface_K = 4
    A.thin_surface_delta_weight = 1e-3
    A.thin_surface_height_weight = 5e-4
    A.thin_surface_gate_tau = 0.01
    A.thin_surface_delta_clip = 2.0
    A.thin_surface_grad_clip = 1.0
    # R2:
    A.thin_surface_lr_scale = 1.0
    A.thin_surface_delta_lr_scale = 0.01
    A.thin_surface_quat_lr_scale = 0.0
    A.thin_surface_sites_lr_scale = 0.0
    A.thin_surface_heights_lr_scale = 0.0

    scene.declare_optimizer(A, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(A, K=4)

    base = 5e-2 * 0.1     # 5e-3 = density_lr_init * 0.1
    expected = {
        "density_delta": base * 1.0 * 0.01,    # 5e-5
        "quaternions":   base * 1.0 * 0.0,     # 0.0
        "texel_sites_2d": base * 1.0 * 0.0,    # 0.0
        "texel_heights":  base * 1.0 * 0.0,     # 0.0
    }
    for name, exp in expected.items():
        lr = next(g["lr"] for g in scene.optimizer.param_groups
                  if g["name"] == name)
        check(abs(lr - exp) < 1e-15,
              f"{name} LR = {lr:.3e} (expected {exp:.3e})")
    # Sanity: the surface is still active.
    check(scene._thin_surface_active,
          "_thin_surface_active = True (R2 keeps kernel running, delta trains "
          "with delta weight too)")
    # Sanity: base param groups are unaffected.
    base_density_lr = next(g["lr"] for g in scene.optimizer.param_groups
                           if g["name"] == "density")
    check(base_density_lr == A.density_lr_init,
          f"density (base) LR unaffected: {base_density_lr}")


# Test 8: --run-tag coexists with the new per-group flags (preserves
# existing tag/suffix behaviour -- no double-suffix).
def test_run_tag_coexists_with_per_group_flags():
    print("\n--- Test 8: --run-tag coexists with per-group flags ---")
    captured = _capture_config([
        "--test", "1b", "--thin-surface",
        "--run-tag", "R2",
        "--thin-delta-lr-scale", "0.01",
        "--thin-geometry-lr-scale", "0",
    ])
    cfg_path = captured["cfg_path"]
    check("R2" in cfg_path,
          f"run-tag R2 applied to cfg path: {cfg_path}")
    cfg = _read_cfg(cfg_path)
    check(cfg["thin_surface_delta_lr_scale"] == 0.01,
          f"per-group delta=0.01 alongside R2 tag: "
          f"{cfg['thin_surface_delta_lr_scale']}")


def main():
    print("=" * 60)
    print("Per-Group Rescue Driver Tests (R2/R3)")
    print("=" * 60)

    test_help_exposes_new_flags()
    test_defaults_preserved_when_no_flag()
    test_keys_absent_when_thin_surface_off()
    test_geometry_flag_fans_out_to_three_geometry_keys()
    test_delta_flag_applies_only_to_delta()
    test_full_r2_invocation()
    test_r2_runtime_lr_proof()
    test_run_tag_coexists_with_per_group_flags()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above).")
        sys.exit(1)
    print("SUMMARY: ALL PER-GROUP DRIVER TESTS PASSED.")


if __name__ == "__main__":
    main()
