"""Extract an iso-surface mesh from a trained CTScene checkpoint.

Runs marching tetrahedra directly on the Delaunay tet complex — no voxelization.
Optionally applies IDW density interpolation and/or Taubin mesh smoothing.

Usage (CLI):
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5 --out mesh.ply
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5 --idw
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5 --smooth_iters 20
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5 --idw --smooth_iters 20

Usage (Python):
    from extract_mesh import extract_mesh
    verts, faces = extract_mesh("output/<run>/model.pt", threshold=0.4)
    verts, faces = extract_mesh("output/<run>/model.pt", threshold=0.4,
                                activation_scale=1.0, output_path="mesh.ply")
    verts, faces = extract_mesh("output/<run>/model.pt", threshold=0.4,
                                use_idw=True, idw_sigma=0.03, idw_sigma_v=0.2,
                                smooth_iters=20)
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import radfoam

from radfoam_model.mesh import marching_tets, taubin_smooth, write_ply


def _load_checkpoint(model_path, device):
    scene_data = torch.load(model_path, map_location="cpu",
                            weights_only=False)
    points = scene_data["xyz"].to(device)
    density_raw = scene_data["density"].to(device).squeeze(-1)
    return points, density_raw


def _build_triangulation(points, density_raw):
    """Rebuild Delaunay triangulation and reorder points/density to match its internal index."""
    tri = radfoam.Triangulation(points.contiguous())
    perm = tri.permutation().long()
    return points[perm], density_raw[perm], tri.tets()


def extract_mesh(model_path, threshold, activation_scale=1.0, output_path=None,
                 device="cuda", verbose=True,
                 use_idw=False, idw_sigma=0.03, idw_sigma_v=0.2,
                 smooth_iters=0, smooth_lambda=0.5, smooth_mu=-0.53):
    """Extract Voronoi iso-surface from a model.pt checkpoint.

    Args:
        model_path:       path to model.pt
        threshold:        iso-value in activated density space (post-softplus × activation_scale)
        activation_scale: multiplier applied after softplus (default 1.0, read from config
                          when called via CLI)
        output_path:      if given, write a binary PLY to this path
        device:           torch device string
        verbose:          print progress
        use_idw:          refine edge crossings with bilateral IDW interpolation
        idw_sigma:        IDW spatial sigma (absolute, not scale-relative)
        idw_sigma_v:      IDW bilateral value sigma
        smooth_iters:     Taubin smoothing passes (0 = disabled)
        smooth_lambda:    Taubin positive shrink weight
        smooth_mu:        Taubin negative inflate weight

    Returns:
        vertices: (V, 3) float32 numpy array
        faces:    (F, 3) int64  numpy array
    """
    device = torch.device(device)
    t0 = time.time()

    if use_idw:
        from radfoam_model.scene import load_model_for_mesh, idw_query

        if verbose:
            print(f"Loading {model_path} (full model for IDW)")
        model = load_model_for_mesh(model_path, activation_scale=activation_scale,
                                    device=str(device))
        points = model.primal_points.detach()
        density = model.get_primal_density().detach().squeeze(-1)
        adj = model.point_adjacency
        adj_off = model.point_adjacency_offsets
        tree = model.aabb_tree
        tets = model.triangulation.tets()
        N = points.shape[0]

        if verbose:
            print(f"  {N:,} cells")

        _sigma, _sigma_v = idw_sigma, idw_sigma_v

        @torch.no_grad()
        def _density_fn(q):
            return idw_query(
                q, points, adj, adj_off, tree, density,
                sigma=_sigma, sigma_v=_sigma_v,
                per_cell_sigma=False, per_neighbor_sigma=False,
                cell_radius=None,
            ).idw_result

    else:
        if verbose:
            print(f"Loading {model_path}")
        points, density_raw = _load_checkpoint(model_path, device)
        N = points.shape[0]

        if verbose:
            print(f"  {N:,} cells — building triangulation...")
        t1 = time.time()
        points, density_raw, tets = _build_triangulation(points, density_raw)
        T = tets.shape[0]
        if verbose:
            print(f"  {T:,} tets in {time.time()-t1:.1f}s")

        density = activation_scale * F.softplus(density_raw, beta=10)
        _density_fn = None

    if verbose:
        d_min, d_max = density.min().item(), density.max().item()
        print(f"  Density [{d_min:.4f}, {d_max:.4f}], threshold={threshold}")
        if threshold <= d_min or threshold >= d_max:
            print("  WARNING: threshold outside density range — mesh may be empty")

    if verbose:
        print("  Running marching tetrahedra...")
    t2 = time.time()
    vertices, faces = marching_tets(points, density, tets, threshold,
                                    density_fn=_density_fn)
    if verbose:
        print(f"  {len(vertices):,} verts, {len(faces):,} faces in {time.time()-t2:.1f}s")

    if len(faces) == 0 and verbose:
        print("  No surface found at this threshold.")

    if smooth_iters > 0 and len(faces) > 0:
        if verbose:
            print(f"  Taubin smoothing ({smooth_iters} iters, λ={smooth_lambda}, μ={smooth_mu})...")
        t3 = time.time()
        vertices, faces = taubin_smooth(vertices, faces, n_iters=smooth_iters,
                                        lambda_=smooth_lambda, mu=smooth_mu)
        if verbose:
            print(f"  Done in {time.time()-t3:.1f}s")

    if output_path is not None:
        write_ply(output_path, vertices, faces)
        if verbose:
            print(f"  Saved {output_path}")

    if verbose:
        print(f"Done in {time.time()-t0:.1f}s")

    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description="Extract Voronoi iso-surface mesh")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to run config.yaml")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Iso-value in activated density space (post-softplus)")
    parser.add_argument("--activation_scale", type=float, default=None,
                        help="Override activation_scale (default: read from config)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .ply path (default: <run_dir>/mesh_<threshold>[_idw][_taubin<N>].ply)")

    # IDW options
    parser.add_argument("--idw", action="store_true",
                        help="Refine edge crossings with bilateral IDW interpolation")
    parser.add_argument("--idw_sigma", type=float, default=0.03,
                        help="IDW spatial sigma (default: 0.03)")
    parser.add_argument("--idw_sigma_v", type=float, default=0.2,
                        help="IDW bilateral value sigma (default: 0.2)")

    # Taubin smoothing options
    parser.add_argument("--smooth_iters", type=int, default=0,
                        help="Taubin smoothing iterations (default: 0 = off)")
    parser.add_argument("--smooth_lambda", type=float, default=0.5,
                        help="Taubin positive shrink weight (default: 0.5)")
    parser.add_argument("--smooth_mu", type=float, default=-0.53,
                        help="Taubin negative inflate weight (default: -0.53)")

    args = parser.parse_args()

    run_dir = os.path.dirname(args.config)
    model_path = os.path.join(run_dir, "model.pt")

    activation_scale = args.activation_scale
    if activation_scale is None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        activation_scale = float(cfg.get("activation_scale", 1.0))

    output_path = args.out
    if output_path is None:
        thresh_str = f"{args.threshold:.4f}".rstrip("0").rstrip(".")
        suffix = ""
        if args.idw:
            suffix += "_idw"
        if args.smooth_iters > 0:
            suffix += f"_taubin{args.smooth_iters}"
        output_path = os.path.join(run_dir, f"mesh_{thresh_str}{suffix}.ply")

    extract_mesh(model_path, args.threshold,
                 activation_scale=activation_scale,
                 output_path=output_path,
                 use_idw=args.idw,
                 idw_sigma=args.idw_sigma,
                 idw_sigma_v=args.idw_sigma_v,
                 smooth_iters=args.smooth_iters,
                 smooth_lambda=args.smooth_lambda,
                 smooth_mu=args.smooth_mu)


if __name__ == "__main__":
    main()
