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


def main(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    points_input = checkpoint["xyz"].to(device).contiguous()
    triangulation = radfoam.Triangulation(points_input)
    permutation = triangulation.permutation().long()
    points = points_input[permutation].contiguous()
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
