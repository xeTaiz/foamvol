"""Extract an iso-surface mesh from a trained CTScene checkpoint.

Runs marching tetrahedra directly on the Delaunay tet complex — no voxelization.

Usage (CLI):
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5
    python extract_mesh.py -c output/<run>/config.yaml --threshold 0.5 --out mesh.ply

Usage (Python):
    from extract_mesh import extract_mesh
    verts, faces = extract_mesh("output/<run>/model.pt", threshold=0.4)
    verts, faces = extract_mesh("output/<run>/model.pt", threshold=0.4,
                                activation_scale=1.0, output_path="mesh.ply")
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import radfoam

from radfoam_model.mesh import marching_tets, write_ply


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
                 device="cuda", verbose=True):
    """Extract Voronoi iso-surface from a model.pt checkpoint.

    Args:
        model_path:       path to model.pt
        threshold:        iso-value in activated density space (post-softplus × activation_scale)
        activation_scale: multiplier applied after softplus (default 1.0, read from config
                          when called via CLI)
        output_path:      if given, write a binary PLY to this path
        device:           torch device string
        verbose:          print progress

    Returns:
        vertices: (V, 3) float32 numpy array
        faces:    (F, 3) int64  numpy array
    """
    device = torch.device(device)
    t0 = time.time()

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

    if verbose:
        d_min, d_max = density.min().item(), density.max().item()
        print(f"  Density [{d_min:.4f}, {d_max:.4f}], threshold={threshold}")
        if threshold <= d_min or threshold >= d_max:
            print("  WARNING: threshold outside density range — mesh may be empty")

    if verbose:
        print("  Running marching tetrahedra...")
    t2 = time.time()
    vertices, faces = marching_tets(points, density, tets, threshold)
    if verbose:
        print(f"  {len(vertices):,} verts, {len(faces):,} faces in {time.time()-t2:.1f}s")

    if len(faces) == 0 and verbose:
        print("  No surface found at this threshold.")

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
                        help="Output .ply path (default: <run_dir>/mesh_<threshold>.ply)")
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
        output_path = os.path.join(run_dir, f"mesh_{thresh_str}.ply")

    extract_mesh(model_path, args.threshold,
                 activation_scale=activation_scale,
                 output_path=output_path)


if __name__ == "__main__":
    main()
