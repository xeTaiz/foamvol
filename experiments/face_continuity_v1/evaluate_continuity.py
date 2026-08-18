#!/usr/bin/env python3
"""Evaluate final shared-face continuity statistics from a split checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import radfoam

from radfoam_model.face_continuity import (
    build_voronoi_face_cache, face_continuity_loss)


def _robust_triangulation(points_input, max_failures=10):
    """Triangulate, retrying with adaptive jitter like CTScene does.

    ``radfoam.Triangulation`` can raise TriangulationFailedError("divergent
    growth iterations") on a point cloud the live *incremental* triangulation
    renders and trains on happily -- observed at 512k on sweep_splitcell_v1's
    SC512_w1e-5, whose 10000-step run finished at test PSNR 48.91 and was then
    unevaluable. ``CTScene.update_triangulation`` already recovers from this via
    extent-relative perturbation, so reuse that policy verbatim (same 1e-5 and
    3**failures constants) rather than losing a completed run. The applied
    jitter is returned so it lands in the JSON and the number stays auditable.
    """
    points = points_input
    extent = points_input.abs().max().item()
    failures = 0
    perturbation = 0.0
    while True:
        try:
            return radfoam.Triangulation(points), points, failures, perturbation
        except radfoam.TriangulationFailedError as error:
            failures += 1
            if failures > max_failures:
                raise RuntimeError(
                    f"aborted triangulation after {max_failures} attempts") from error
            perturbation = extent * 1e-5 * (3.0 ** failures)
            print(f"[continuity-eval] caught {error}; retry {failures} with "
                  f"perturbation {perturbation:.3e}")
            points = points_input + perturbation * torch.randn_like(points_input)
            points = points.contiguous()


def main(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    points_input = checkpoint["xyz"].to(device).contiguous()
    triangulation, points_used, tri_retries, tri_perturbation = (
        _robust_triangulation(points_input))
    permutation = triangulation.permutation().long()
    # Geometry must come from the point set the tets actually describe.
    points = points_used[permutation].contiguous()
    identity = torch.arange(points.shape[0], device=device)
    cache = build_voronoi_face_cache(
        points, triangulation.tets(), identity,
        num_samples=args.samples, domain_extent=1.0, max_vertices=32)
    adjacency = triangulation.point_adjacency()
    offsets = triangulation.point_adjacency_offsets()
    _, radius = radfoam.farthest_neighbor(points, adjacency, offsets)

    model = SimpleNamespace(primal_points=points)
    for key in ("density", "density_delta", "quaternions",
                "texel_sites_2d", "texel_heights"):
        setattr(model, key, torch.nn.Parameter(
            checkpoint[key].to(device)[permutation].contiguous()))
    thin = checkpoint["thin_surface"]
    model.activation_scale = args.activation_scale
    model._cached_cell_radius = radius.reshape(-1)
    model._thin_surface_density_mode = thin.get("density_mode", "relative")
    model._thin_surface_delta_max_frac = float(thin.get("delta_max_frac", 0.5))

    gt = np.load(args.gt, mmap_mode="r")
    density_scale = float(np.percentile(gt, 99))
    collected = []
    with torch.no_grad():
        for index in range(args.batches):
            loss, diagnostics = face_continuity_loss(
                model, cache, step=100000 + index,
                batch_size=args.batch_size, density_scale=density_scale,
                candidate_refresh=1000000)
            collected.append({"raw_total": float(loss), **{
                key: float(value) for key, value in diagnostics.items()}})
    keys = collected[0].keys()
    summary = {
        "model": str(Path(args.model).resolve()),
        "cache_faces": cache.num_faces,
        "cache_build_seconds": cache.build_seconds,
        # Nonzero means the from-scratch rebuild diverged and the geometry was
        # jittered by this much to recover it; compare such an arm with care.
        "triangulation_retries": tri_retries,
        "triangulation_perturbation": tri_perturbation,
        "density_scale": density_scale,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "mean": {key: float(np.mean([row[key] for row in collected])) for key in keys},
        "std": {key: float(np.std([row[key] for row in collected])) for key in keys},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    main(parser.parse_args())
