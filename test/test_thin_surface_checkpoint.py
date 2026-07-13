"""
Checkpoint round-trip for thin-surface parameters.

Verifies:
  1. save_pt / load_pt round-trip preserves all thin-surface parameters exactly.
  2. Render output (projection) is identical before save and after load for the same rays.
  3. save_pt / load_pt round-trip preserves nonzero density_delta, quaternions,
     texel_sites_2d, texel_heights (by setting them to nonzero values before save).

Status (2026-07-06 P0-A fix landed):
  - save_pt()/load_pt() NOW persist the four thin-surface tensors plus
    metadata (_thin_surface_active, _thin_K, _thin_surface_start, scheduler
    cfg). These tests are expected to PASS on a CUDA box. If any thin-surface
    param is missing after load_pt, that is a REGRESSION, not an expected
    failure.

Run with:  micromamba run -n radfoam python test/test_thin_surface_checkpoint.py
"""
import sys
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np

# Graceful skip if CUDA is unavailable — must come before radfoam import
_HAS_CUDA = torch.cuda.is_available()
if not _HAS_CUDA:
    print("SKIP: No CUDA device. All thin-surface checkpoint tests require GPU.")
    print("Run on: kw995 or kw996 (RTX 6000 Ada)")
    sys.exit(0)

# Ensure project root is on path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import radfoam
from radfoam_model.scene import CTScene


torch.manual_seed(42)
np.random.seed(42)


def _make_minimal_scene(device="cuda", n_points=32):
    """Create a CTScene with thin-surface activated and nonzero params.

    n_points defaults to 32: radfoam.Triangulation requires >= MIN_POINTS=32
    (see old/test_cube.py); smaller scenes yield invalid adjacency and the
    forward/render path misbehaves. Uses a regular grid for a well-behaved
    Delaunay mesh. Sets the four thin-surface tensors to known nonzero values
    so the save/load round-trip can be verified exactly.
    """
    # Minimal args to create the scene
    args = type("Args", (), {
        "init_points": n_points,
        "final_points": n_points,
        "activation_scale": 1.0,
        "init_scale": 0.5,
        "init_type": "random",
        "init_density": 0.0,
        "device": device,
        "init_points_file": "",
        "init_volume_path": "",
        "frozen_points_file": "",
        "frozen_freeze_density": True,
        "density_lr_init": 5e-2,
    })()

    model = CTScene(args, device=torch.device(device))

    # Manually activate thin-surface mode so params are registered
    # Simulates what train.py does at thin_surface_start
    model._thin_surface_active = True
    model._thin_K = 4
    model._thin_surface_gate_tau = 0.01
    model._max_iterations = 1000
    model._thin_surface_start = 0

    # Register thin-surface parameters manually (like initialize_thin_surface does)
    N = model.primal_points.shape[0]
    model.density_delta = nn.Parameter(0.1 * torch.randn(N, 1, device=device))
    # Quaternions: nonzero, normalized
    q = torch.randn(N, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True)
    model.quaternions = nn.Parameter(q)
    # Texel sites: known values (not just base ring)
    angles = torch.linspace(0, 2 * np.pi, 5, device=device)[:-1]
    base = torch.stack([torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    jitter = (torch.rand(N, 4, 2, device=device) - 0.5) * 0.05
    model.texel_sites_2d = nn.Parameter(base.unsqueeze(0).expand(N, -1, -1) + jitter)
    # Heights: nonzero
    model.texel_heights = nn.Parameter(0.02 * torch.randn(N, 4, device=device))

    model.update_triangulation(rebuild=False)
    return model


def _make_test_rays(model, n_rays=4):
    """Create simple rays that pass through the scene volume."""
    pts = model.primal_points.detach()
    center = pts.mean(dim=0)
    # Rays from multiple directions through the scene center
    directions = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], device=pts.device, dtype=torch.float32)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    origins = center - 2.0 * directions  # start before the scene
    rays = torch.cat([origins, directions], dim=-1)  # (N, 6)
    return rays


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        global _any_failed
        _any_failed = True


_any_failed = False


def test_checkpoint_round_trip():
    """Test 1: save_pt / load_pt preserves thin-surface params and render output."""
    print("\n--- Test: Checkpoint Round-Trip ---")

    device = "cuda"

    model = _make_minimal_scene(device)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    # Helper to render with thin-surface mode enabled
    def render(model):
        model._thin_surface_active = True
        out, _, _, _, _ = model(rays, start_point)
        return out.detach()

    # Render before save
    proj_before = render(model)
    check(proj_before.isfinite().all(), "Forward pass produces finite output")

    # Save params we expect to see in checkpoint
    saved_dd = model.density_delta.detach().clone().cpu()
    saved_q = model.quaternions.detach().clone().cpu()
    saved_ts = model.texel_sites_2d.detach().clone().cpu()
    saved_th = model.texel_heights.detach().clone().cpu()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        model.save_pt(ckpt_path)

        # Verify file exists and load it back
        check(os.path.exists(ckpt_path), f"Checkpoint saved to {ckpt_path}")

        # Create a fresh model and load
        loaded = _make_minimal_scene(device)
        loaded._thin_surface_active = True  # needed for forward() to dispatch thin-surface kernel
        loaded._thin_K = 4

        load_ok = True
        try:
            loaded.load_pt(ckpt_path)
        except Exception as e:
            check(False, f"load_pt() succeeded: FAILED with {type(e).__name__}: {e}")
            load_ok = False

        if load_ok:
            # Check which thin-surface params were preserved
            dd_ok = (hasattr(loaded, "density_delta") and
                     loaded.density_delta is not None and
                     loaded.density_delta.shape == (saved_dd.shape[0], 1))
            check(dd_ok, f"density_delta loaded: shape={getattr(loaded, 'density_delta', None)}")

            q_ok = (hasattr(loaded, "quaternions") and
                    loaded.quaternions is not None and
                    loaded.quaternions.shape == saved_q.shape)
            check(q_ok, f"quaternions loaded: shape={getattr(loaded, 'quaternions', None)}")

            ts_ok = (hasattr(loaded, "texel_sites_2d") and
                     loaded.texel_sites_2d is not None and
                     loaded.texel_sites_2d.shape == saved_ts.shape)
            check(ts_ok, f"texel_sites_2d loaded: shape={getattr(loaded, 'texel_sites_2d', None)}")

            th_ok = (hasattr(loaded, "texel_heights") and
                     loaded.texel_heights is not None and
                     loaded.texel_heights.shape == saved_th.shape)
            check(th_ok, f"texel_heights loaded: shape={getattr(loaded, 'texel_heights', None)}")

            # If all four loaded, verify values
            if dd_ok and q_ok and ts_ok and th_ok:
                dd_match = torch.allclose(loaded.density_delta.cpu(), saved_dd, atol=1e-6)
                q_match = torch.allclose(loaded.quaternions.cpu(), saved_q, atol=1e-6)
                ts_match = torch.allclose(loaded.texel_sites_2d.cpu(), saved_ts, atol=1e-6)
                th_match = torch.allclose(loaded.texel_heights.cpu(), saved_th, atol=1e-6)
                all_match = dd_match and q_match and ts_match and th_match
                check(all_match, "All thin-surface params match exactly after round-trip")

                # Render equivalence
                loaded = loaded.to(device)
                loaded._thin_surface_active = True
                try:
                    proj_after = render(loaded)
                    render_match = torch.allclose(proj_before.cpu(), proj_after.cpu(), atol=1e-5)
                    check(render_match, "Render output matches before/after checkpoint")
                except Exception as e:
                    check(False, f"Render after load: FAILED with {type(e).__name__}: {e}")
            else:
                # P0-A fix makes this a hard regression, not an expected failure.
                check(False,
                      "Thin-surface parameters missing from checkpoint after "
                      "load_pt — REGRESSION: save_pt()/load_pt() must persist "
                      "density_delta, quaternions, texel_sites_2d, texel_heights")


def test_render_with_nonzero_delta():
    """Test 2: Nonzero density_delta produces different output than zero delta.

    This is a forward-pass sanity check independent of checkpoint.
    """
    print("\n--- Test: Nonzero delta changes render output ---")

    device = "cuda"

    model = _make_minimal_scene(device)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    # Render with zero delta (all zeros)
    with torch.no_grad():
        model.density_delta.data.zero_()
        model._thin_surface_active = True
        proj_zero_delta, _, _, _, _ = model(rays, start_point)

    # Render with nonzero delta
    with torch.no_grad():
        model.density_delta.data.copy_(0.5 * torch.ones_like(model.density_delta))
        model._thin_surface_active = True
        proj_nonzero_delta, _, _, _, _ = model(rays, start_point)

    diff = (proj_zero_delta - proj_nonzero_delta).abs().max().item()
    # With delta=0.5 and chord lengths ~O(1), the projection difference should be substantial
    check(diff > 1e-4,
          f"Nonzero delta changes projection (max diff={diff:.6f}, "
          f"expected >> 1e-4)")
    check(proj_nonzero_delta.isfinite().all(),
          "Forward pass with nonzero delta produces finite output")


def test_quaternion_variation():
    """Test 3: Different quaternions (surface orientations) change render output.

    Confirms that the quaternion → frame rotation → surface normal pipeline
    has a measurable effect on projection.
    """
    print("\n--- Test: Quaternion orientation changes render output ---")

    device = "cuda"

    model = _make_minimal_scene(device)
    rays = _make_test_rays(model)
    start_point = model.get_starting_point(rays, model.primal_points, model.aabb_tree)

    N = model.primal_points.shape[0]

    # Identity quaternion: normal points along +X
    projs = []
    for label, q_init in [
        ("+X (identity)", torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, -1).clone()),
        ("+Y (90° about Z)", torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], device=device).expand(N, -1).clone()),
        ("+Z (90° about -Y)", torch.tensor([[0.7071, 0.0, -0.7071, 0.0]], device=device).expand(N, -1).clone()),
        ("-X (180° about Y)", torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device).expand(N, -1).clone()),
    ]:
        with torch.no_grad():
            model.quaternions.data.copy_(q_init / q_init.norm(dim=-1, keepdim=True))
            model._thin_surface_active = True
            out, _, _, _, _ = model(rays, start_point)
            projs.append(out.detach())

    # Check that at least one orientation pair produces different results
    diffs = [torch.abs(projs[0] - p).max().item() for p in projs[1:]]
    max_diff = max(diffs)
    check(max_diff > 1e-4,
          f"Different quaternions change projection (max diff={max_diff:.6f})")
    for i, p in enumerate(projs):
        check(p.isfinite().all(), f"Forward pass {i} with quaternion variation is finite")


def main():
    print("=" * 60)
    print("Thin-Surface Checkpoint and Render Sanity Tests")
    print("=" * 60)

    test_checkpoint_round_trip()
    test_render_with_nonzero_delta()
    test_quaternion_variation()

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED (see above)")
        print("Note: checkpoint round-trip is a hard gate (P0-A). "
              "save_pt()/load_pt() must persist all thin-surface params.")
        sys.exit(1)
    else:
        print("SUMMARY: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
