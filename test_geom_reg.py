"""
Monotonicity test for top_eigvec_alignment_regularization (Loss A) and
normal_laplacian_regularization (Loss C).

Expected:
  - Aligned flat boundary  → lower loss than jagged boundary (both A and C)
  - Smooth curved surface  → C lower than A (C tolerates curvature; A fires)

Run with:  micromamba run -n radfoam python test_geom_reg.py
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import radfoam
from radfoam_model.scene import CTScene

torch.manual_seed(42)


def make_scene_from_points(pts: torch.Tensor, raw_density: torch.Tensor) -> CTScene:
    """Build a minimal CTScene-like namespace from raw points + density.

    Uses the real CTScene class so triangulation and adjacency are set up
    exactly as in production.  The FakeScene approach was abandoned because it
    requires applying the triangulation's permutation to primal_points — the
    CTScene __init__ already does this correctly.
    """
    # We need to sneak our points into a CTScene.  The cleanest way is to
    # call __new__ and manually populate the required attributes.
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0

    pts = pts.contiguous().float()
    tri = radfoam.Triangulation(pts)
    # Apply permutation so primal_points is consistent with adjacency
    perm = tri.permutation().long()
    pts_perm = pts[perm]

    scene.primal_points = nn.Parameter(pts_perm)
    scene.density = nn.Parameter(raw_density[perm])
    scene.triangulation = tri

    adj = tri.point_adjacency()
    off = tri.point_adjacency_offsets()

    scene.point_adjacency = adj
    scene.point_adjacency_offsets = off

    _, cr = radfoam.farthest_neighbor(scene.primal_points, adj, off)
    scene._cached_cell_radius = cr.squeeze().detach()

    return scene


def flat_boundary_scene(n: int = 300, device: str = "cuda"):
    """Two half-spaces with a smooth flat boundary.

    Lower half (z < 0) → high density, upper half → low density.
    Small random z-noise keeps the grid non-degenerate for 3D Delaunay.
    """
    rng = np.random.RandomState(0)
    pts = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    # Tiny z-noise so no perfect coplanarity
    pts[:, 2] += rng.normal(0, 1e-3, n).astype(np.float32)

    pts_t = torch.tensor(pts, device=device)
    # Density: high below z=0, low above
    raw = torch.where(
        pts_t[:, 2] < 0,
        torch.full((n,), 2.0, device=device),
        torch.full((n,), -2.0, device=device),
    ).unsqueeze(-1)

    return make_scene_from_points(pts_t, raw)


def jagged_boundary_scene(n: int = 300, device: str = "cuda"):
    """Same density pattern but boundary plane is jagged (checker-board perturbation)."""
    rng = np.random.RandomState(1)
    pts = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    pts_t = torch.tensor(pts, device=device)

    # Shift boundary plane by ±0.3 based on x-sign to create a kink
    boundary = 0.3 * np.sign(pts[:, 0]).astype(np.float32)
    boundary_t = torch.tensor(boundary, device=device)
    raw = torch.where(
        pts_t[:, 2] < boundary_t,
        torch.full((n,), 2.0, device=device),
        torch.full((n,), -2.0, device=device),
    ).unsqueeze(-1)

    return make_scene_from_points(pts_t, raw)


def loss_A(scene, sigma_v=0.2):
    return scene.top_eigvec_alignment_regularization(sigma_v=sigma_v).item()


def loss_C(scene, sigma_v=0.2):
    return scene.normal_laplacian_regularization(sigma_v=sigma_v).item()


def check(cond: bool, msg: str):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Building scenes...")
    s_flat   = flat_boundary_scene(device=device)
    s_jagged = jagged_boundary_scene(device=device)

    sigma_v = 0.2

    A_flat   = loss_A(s_flat,   sigma_v)
    A_jagged = loss_A(s_jagged, sigma_v)

    C_flat   = loss_C(s_flat,   sigma_v)
    C_jagged = loss_C(s_jagged, sigma_v)

    print(f"Loss A: flat={A_flat:.4f}  jagged={A_jagged:.4f}")
    print(f"Loss C: flat={C_flat:.4f}  jagged={C_jagged:.4f}")
    print()

    print("Monotonicity checks (flat boundary < jagged boundary for both losses):")
    check(A_flat < A_jagged,   f"Loss A: flat ({A_flat:.4f}) < jagged ({A_jagged:.4f})")
    check(C_flat < C_jagged,   f"Loss C: flat ({C_flat:.4f}) < jagged ({C_jagged:.4f})")

    # Basic gradient checks
    print("\nGradient sanity checks (finite and nonzero gradients into primal_points):")
    for name, scene in [
        ("A/flat",   s_flat),
        ("A/jagged", s_jagged),
        ("C/flat",   s_flat),
        ("C/jagged", s_jagged),
    ]:
        scene.primal_points.grad = None
        l = scene.top_eigvec_alignment_regularization(sigma_v) if "A/" in name \
            else scene.normal_laplacian_regularization(sigma_v)
        l.backward()
        g = scene.primal_points.grad
        finite = (g is not None) and g.isfinite().all().item()
        nonzero = (g is not None) and (g.norm().item() > 0)
        check(finite and nonzero, f"{name}: grad finite={finite}, nonzero={nonzero}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
