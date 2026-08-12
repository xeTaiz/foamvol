#!/usr/bin/env python3
"""Direct rays-per-cell supervision test for split-plane orientation.

For a fixed set of informative oracle cells, generate exact continuous cone-beam
rays through points verified to lie in each target Voronoi cell.  Sweep the
number of distinct rays available per cell while keeping optimizer steps and
sampled rays per step fixed (sampling with replacement for small pools).
Everything except the selected cells' quaternions remains frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import radfoam  # noqa: E402
from data_loader.r2_gaussian import R2GaussianDataset  # noqa: E402
from data_loader.utils import bilinear_proj_lookup  # noqa: E402
from experiments.orientation_recovery import (  # noqa: E402
    _angle_stats,
    _gradient_alignment,
    _load_scene,
    _normal_to_quaternion,
    _perturb_normals,
    _quaternion_to_normal,
    _render,
    _render_chunks,
)


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _select_cells(csv_path: str, meta_selected: torch.Tensor, count: int):
    rows = list(csv.DictReader(open(csv_path)))
    cells = []
    for row in rows:
        cell = int(row["cell"])
        if bool(meta_selected[cell]):
            cells.append(cell)
        if len(cells) == count:
            break
    if len(cells) != count:
        raise RuntimeError(f"requested {count} cells, found {len(cells)}")
    return torch.tensor(cells, dtype=torch.long)


def _sample_points_in_cells(scene, cells: torch.Tensor, max_rays: int,
                            radius_scale: float, seed: int):
    """Rejection-sample points whose exact nearest site is the requested cell."""
    device = scene.device
    cells_gpu = cells.to(device)
    centers = scene.primal_points.detach()[cells_gpu]
    radii = scene._get_cell_radius().detach()[cells_gpu]
    gen = torch.Generator(device=device).manual_seed(seed)
    accepted = []
    attempts = []
    for local, cell in enumerate(cells_gpu):
        chunks = []
        tried = 0
        while sum(x.shape[0] for x in chunks) < max_rays:
            need = max_rays - sum(x.shape[0] for x in chunks)
            n = max(need * 3, 256)
            # Uniform ball sampling, then exact Voronoi-owner verification.
            direction = torch.randn((n, 3), generator=gen, device=device)
            direction = F.normalize(direction, dim=-1)
            radial = torch.rand((n, 1), generator=gen, device=device).pow(1.0 / 3.0)
            query = centers[local] + direction * radial * radii[local] * radius_scale
            owner = radfoam.nn(scene.primal_points, scene.aabb_tree, query).long()
            valid = (owner == cell) & (query.abs() <= 1.0).all(dim=-1)
            chunks.append(query[valid][:need])
            tried += n
            if tried > max_rays * 200:
                raise RuntimeError(f"could not sample {max_rays} interior points for cell {int(cell)}")
        points = torch.cat(chunks, dim=0)[:max_rays]
        accepted.append(points.cpu())
        attempts.append(tried)
    return torch.stack(accepted), attempts


def _view_sequence(num_views: int, max_rays: int, cell_rank: int):
    """Nested, angularly spread deterministic view sequence."""
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    i = np.arange(max_rays, dtype=np.float64)
    phase = (cell_rank * 0.3819660112501051) % 1.0
    return np.floor(((i * golden + phase) % 1.0) * num_views).astype(np.int64)


def _build_ray_pool(scene, dataset, cells: torch.Tensor, points: torch.Tensor,
                    budget: int, device: str):
    """Build exact rays source->verified-cell-point and matched measured values."""
    num_cells, max_rays, _ = points.shape
    num_views, det_h, det_w = dataset.all_rays.shape[:3]
    c2ws = dataset.c2ws.to(device)
    proj = dataset.all_projections.to(device)
    all_rays, all_measured, all_owner, view_counts = [], [], [], []

    for local in range(num_cells):
        view = torch.from_numpy(_view_sequence(num_views, max_rays, local)[:budget]).to(device)
        target = points[local, :budget].to(device)
        pose = c2ws[view]
        origin = pose[:, :3, 3]
        rotation = pose[:, :3, :3]
        d_world = target - origin
        d_cam = torch.bmm(rotation.transpose(1, 2), d_world.unsqueeze(-1)).squeeze(-1)
        z = d_cam[:, 2]
        # Pixel array coordinate: dataset ray index iu uses iu+0.5-W/2.
        px = d_cam[:, 0] / z * dataset.fx + det_w / 2.0 - 0.5
        py = d_cam[:, 1] / z * dataset.fy + det_h / 2.0 - 0.5
        valid = ((z > 0) & (px >= 0) & (px <= det_w - 1)
                 & (py >= 0) & (py <= det_h - 1))
        if not bool(valid.all()):
            raise RuntimeError(
                f"cell {int(cells[local])}: only {int(valid.sum())}/{budget} rays project inside detector")
        direction = F.normalize(d_world, dim=-1)
        all_rays.append(torch.cat([origin, direction], dim=-1))
        all_measured.append(bilinear_proj_lookup(proj, view, px, py).reshape(-1))
        all_owner.append(torch.full((budget,), int(cells[local]), device=device, dtype=torch.long))
        view_counts.append(int(torch.unique(view).numel()))

    return (torch.cat(all_rays), torch.cat(all_measured), torch.cat(all_owner),
            view_counts)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene = _load_scene(args.oracle_model, args.device)
    meta = torch.load(args.oracle_meta, map_location=args.device, weights_only=True)
    oracle_n = F.normalize(meta["oracle_normals"].to(args.device), dim=-1)
    oracle_q = meta["oracle_quaternions"].to(args.device)
    selected_meta = meta["selected"].bool().cpu()
    cells = _select_cells(args.oracle_cells, selected_meta, args.num_cells)
    train_mask = torch.zeros(scene.primal_points.shape[0], dtype=torch.bool, device=args.device)
    train_mask[cells.to(args.device)] = True

    points, attempts = _sample_points_in_cells(
        scene, cells, args.max_rays_per_cell, args.radius_scale, args.pool_seed)
    dataset = R2GaussianDataset(args.data_path, split="train")
    rays, measured, ray_owner, view_counts = _build_ray_pool(
        scene, dataset, cells, points, args.rays_per_cell, args.device)
    # Exact geometric guarantee: every constructed ray passes through a point
    # whose nearest Voronoi site was verified to be its requested target cell.
    target_points = points[:, :args.rays_per_cell].reshape(-1, 3).to(args.device)
    owner_check = radfoam.nn(scene.primal_points, scene.aabb_tree, target_points).long()
    exact_owner_fraction = float((owner_check == ray_owner).float().mean().item())
    if exact_owner_fraction != 1.0:
        raise RuntimeError(f"owner verification failed: {exact_owner_fraction}")

    with torch.no_grad():
        scene.quaternions.copy_(oracle_q)
    if args.target == "teacher":
        target = _render_chunks(scene, rays, args.render_chunk).to(args.device)
    else:
        target = measured

    perturbed = _perturb_normals(oracle_n, train_mask, args.angle, args.seed)
    with torch.no_grad():
        scene.quaternions.copy_(_normal_to_quaternion(perturbed))
    initial_q = scene.quaternions.detach().clone()
    for name in ("primal_points", "density", "density_delta", "raw_plus",
                 "raw_minus", "texel_sites_2d", "texel_heights"):
        value = getattr(scene, name, None)
        if value is not None:
            value.requires_grad_(False)
    scene.quaternions.requires_grad_(True)
    optimizer = torch.optim.Adam([scene.quaternions], lr=args.lr, eps=1e-8)

    # Initial full-pool gradient direction.
    optimizer.zero_grad(set_to_none=True)
    initial_loss = F.mse_loss(_render(scene, rays), target)
    initial_loss.backward()
    initial_grad = scene.quaternions.grad.detach().clone()
    alignment = _gradient_alignment(scene.quaternions, initial_grad, oracle_n, train_mask)
    initial_grad_norm = float(initial_grad[train_mask].norm(dim=-1).median().item())
    initial_stats = _angle_stats(scene.quaternions, oracle_n, train_mask)
    initial_mse = float(initial_loss.item())

    generator = torch.Generator(device=args.device).manual_seed(args.seed + 12345)
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Constant optimization work across budgets. Small pools are sampled
        # with replacement; larger pools provide more distinct constraints.
        idx = torch.randint(
            0, rays.shape[0], (args.batch_rays,), generator=generator,
            device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(_render(scene, rays[idx]), target[idx])
        loss.backward()
        with torch.no_grad():
            scene.quaternions.grad[~train_mask] = 0
        optimizer.step()
        with torch.no_grad():
            scene.quaternions.copy_(F.normalize(scene.quaternions, dim=-1))
            scene.quaternions[~train_mask] = oracle_q[~train_mask]
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                mse = float(F.mse_loss(_render(scene, rays), target).item())
            row = {"step": step, "mse": mse,
                   **_angle_stats(scene.quaternions, oracle_n, train_mask)}
            history.append(row)
            print(json.dumps(row), flush=True)

    final = history[-1]
    final_n = _quaternion_to_normal(scene.quaternions.detach())
    per_cell_dot = (final_n[train_mask] * oracle_n[train_mask]).sum(-1).clamp(-1, 1)
    per_cell_angle = torch.rad2deg(torch.acos(per_cell_dot)).cpu().numpy()
    result = {
        "target": args.target,
        "angle_deg": args.angle,
        "rays_per_cell": args.rays_per_cell,
        "num_cells": args.num_cells,
        "total_distinct_rays": int(rays.shape[0]),
        "batch_rays_per_step": args.batch_rays,
        "steps": args.steps,
        "lr": args.lr,
        "unique_views_per_cell_min": int(min(view_counts)),
        "unique_views_per_cell_median": float(np.median(view_counts)),
        "unique_views_per_cell_max": int(max(view_counts)),
        "exact_owner_fraction": exact_owner_fraction,
        "sampling_attempts_median": float(np.median(attempts)),
        "initial": {"mse": initial_mse, "median_grad_norm": initial_grad_norm,
                    **initial_stats},
        "gradient_alignment": alignment,
        "final": final,
        "mse_ratio_final_to_initial": final["mse"] / max(initial_mse, 1e-30),
        "median_angle_improvement_deg": initial_stats["median_deg"] - final["median_deg"],
        "per_cell_angle_deg": per_cell_angle.tolist(),
        "cell_ids": cells.tolist(),
        "runtime_seconds": time.time() - t0,
    }
    stem = (f"targeted_{args.target}_angle{args.angle:g}_cells{args.num_cells}"
            f"_rpc{args.rays_per_cell}_seed{args.seed}")
    _json_dump(output / f"{stem}.json", result)
    _json_dump(output / f"{stem}_history.json", history)
    torch.save({
        "cells": cells, "initial_quaternions": initial_q.cpu(),
        "final_quaternions": scene.quaternions.detach().cpu(),
    }, output / f"{stem}.pt")
    print("SUMMARY", json.dumps(result, indent=2), flush=True)


def summarize(args):
    rows = []
    for path in sorted(Path(args.output).glob("targeted_*_rpc*_seed*.json")):
        if path.name.endswith("_history.json"):
            continue
        d = json.loads(path.read_text())
        rows.append({
            "target": d["target"], "angle_deg": d["angle_deg"],
            "rays_per_cell": d["rays_per_cell"], "num_cells": d["num_cells"],
            "unique_views_median": d["unique_views_per_cell_median"],
            "initial_median_deg": d["initial"]["median_deg"],
            "final_median_deg": d["final"]["median_deg"],
            "final_p90_deg": d["final"]["p90_deg"],
            "final_frac_le_5deg": d["final"]["frac_le_5deg"],
            "mse_ratio": d["mse_ratio_final_to_initial"],
            "initial_grad_norm": d["initial"]["median_grad_norm"],
            "gradient_alignment": d["gradient_alignment"]["mean_cosine"],
            "gradient_fraction_aligned": d["gradient_alignment"]["fraction_aligned"],
            "exact_owner_fraction": d["exact_owner_fraction"],
        })
    rows.sort(key=lambda r: (r["target"], r["rays_per_cell"]))
    if not rows:
        raise RuntimeError("no targeted result JSON files found")
    with (Path(args.output) / "targeted_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--oracle-model", required=True)
    run_p.add_argument("--oracle-meta", required=True)
    run_p.add_argument("--oracle-cells", required=True)
    run_p.add_argument("--data-path", required=True)
    run_p.add_argument("--output", required=True)
    run_p.add_argument("--target", choices=("teacher", "measured"), required=True)
    run_p.add_argument("--rays-per-cell", type=int, required=True)
    run_p.add_argument("--max-rays-per-cell", type=int, default=2048)
    run_p.add_argument("--num-cells", type=int, default=64)
    run_p.add_argument("--radius-scale", type=float, default=0.25)
    run_p.add_argument("--angle", type=float, default=15.0)
    run_p.add_argument("--steps", type=int, default=1000)
    run_p.add_argument("--batch-rays", type=int, default=32768)
    run_p.add_argument("--eval-every", type=int, default=100)
    run_p.add_argument("--render-chunk", type=int, default=32768)
    run_p.add_argument("--lr", type=float, default=2e-3)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--pool-seed", type=int, default=20260813)
    run_p.add_argument("--device", default="cuda:0")
    run_p.set_defaults(func=run)
    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--output", required=True)
    sum_p.set_defaults(func=summarize)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
