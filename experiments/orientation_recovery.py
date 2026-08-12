#!/usr/bin/env python3
"""Controlled thin-surface plane-orientation recoverability experiment.

This is a best-case diagnostic, not a reconstruction-quality experiment:
  1. Fit one flat two-density plane per informative Voronoi cell from GT.
  2. Freeze points, side densities, offsets, and texel sites.
  3. Perturb only the plane normals by a known angle.
  4. Optimize only quaternions against either oracle-rendered projections
     (teacher mode) or the measured synthetic CT projections (measured mode).

Subcommands:
  prepare   Build and save the GT-derived oracle checkpoint.
  recover   Perturb and optimize orientations for one angle/target.
  summarize Aggregate completed recovery JSON files.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import radfoam  # noqa: E402
from data_loader.r2_gaussian import R2GaussianDataset  # noqa: E402
from radfoam_model.scene import CTScene  # noqa: E402


def _model_args(device: str, activation_scale: float = 1.0):
    return SimpleNamespace(
        init_points=32,
        final_points=32,
        activation_scale=activation_scale,
        init_scale=1.05,
        init_type="random",
        init_density=0.0,
        device=device,
        init_points_file="",
        init_volume_path="",
        frozen_points_file="",
        frozen_freeze_density=True,
    )


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _normal_to_quaternion(n: torch.Tensor) -> torch.Tensor:
    """Shortest rotation taking +X to unit normal n; output [w,x,y,z]."""
    n = F.normalize(n, dim=-1)
    nx = n[..., 0]
    w = torch.sqrt(((1.0 + nx) * 0.5).clamp_min(0.0))
    # cross(+X,n) = (0,-nz,ny)
    denom = (2.0 * w).clamp_min(1e-8)
    xyz = torch.stack([torch.zeros_like(nx), -n[..., 2], n[..., 1]], dim=-1) / denom.unsqueeze(-1)
    q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    near_flip = nx < -0.999999
    if near_flip.any():
        replacement = torch.zeros_like(q[near_flip])
        replacement[:, 2] = 1.0  # 180 degrees around +Y maps +X to -X
        q[near_flip] = replacement
    return F.normalize(q, dim=-1)


def _quaternion_to_normal(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)


def _inverse_softplus_beta10(mu: torch.Tensor) -> torch.Tensor:
    mu = mu.clamp_min(1e-8)
    x = 10.0 * mu
    return torch.where(x > 20.0, mu, torch.log(torch.expm1(x).clamp_min(1e-30)) / 10.0)


def _load_scene(checkpoint: str, device: str) -> CTScene:
    scene = CTScene(_model_args(device), device=torch.device(device))
    scene.load_pt(checkpoint)
    scene.activation_scale = 1.0
    return scene


def _configure_independent_scene(scene: CTScene, raw_plus: torch.Tensor,
                                 raw_minus: torch.Tensor, quaternions: torch.Tensor,
                                 heights: torch.Tensor):
    n_cells = scene.primal_points.shape[0]
    k = heights.shape[1]
    angles = torch.linspace(0, 2 * math.pi, k + 1, device=scene.device)[:-1]
    sites = torch.stack([torch.cos(angles) * 0.4, torch.sin(angles) * 0.4], dim=-1)
    sites = sites.unsqueeze(0).expand(n_cells, -1, -1).clone()

    scene.raw_plus = nn.Parameter(raw_plus.to(scene.device), requires_grad=False)
    scene.raw_minus = nn.Parameter(raw_minus.to(scene.device), requires_grad=False)
    scene.quaternions = nn.Parameter(quaternions.to(scene.device), requires_grad=True)
    scene.texel_sites_2d = nn.Parameter(sites, requires_grad=False)
    scene.texel_heights = nn.Parameter(heights.to(scene.device), requires_grad=False)
    scene.density.requires_grad_(False)
    scene.primal_points.requires_grad_(False)
    scene._thin_surface_density_mode = "independent"
    scene._thin_surface_active = True
    scene._thin_surface_relative_delta = False
    scene._thin_surface_start = 0
    scene._thin_K = k
    scene._thin_temp = 10.0
    scene._thin_height_eps = 1e-4
    scene._cached_cell_radius = scene._get_cell_radius().detach()


def _render(scene: CTScene, rays: torch.Tensor) -> torch.Tensor:
    start = scene.get_starting_point(rays, scene.primal_points, scene.aabb_tree)
    return scene(rays, start)[0].reshape(-1)


def _render_chunks(scene: CTScene, rays: torch.Tensor, chunk: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for begin in range(0, rays.shape[0], chunk):
            out.append(_render(scene, rays[begin:begin + chunk]).detach().cpu())
    return torch.cat(out)


def _assign_samples(points: torch.Tensor, aabb_tree, gt: np.ndarray,
                    stride: int, chunk: int, device: str):
    """Assign a strided endpoint GT grid to cells; return CPU arrays."""
    ix = np.arange(0, gt.shape[0], stride, dtype=np.int32)
    iy = np.arange(0, gt.shape[1], stride, dtype=np.int32)
    iz = np.arange(0, gt.shape[2], stride, dtype=np.int32)
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
    indices = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)
    shape_m1 = np.asarray(gt.shape, dtype=np.float32) - 1.0
    xyz = -1.0 + 2.0 * indices.astype(np.float32) / shape_m1
    values = gt[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.float32)

    spacing = tuple(2.0 / (s - 1) for s in gt.shape)
    gradients = np.gradient(gt.astype(np.float32), *spacing, edge_order=1)
    grad = np.stack([
        gradients[0][indices[:, 0], indices[:, 1], indices[:, 2]],
        gradients[1][indices[:, 0], indices[:, 1], indices[:, 2]],
        gradients[2][indices[:, 0], indices[:, 1], indices[:, 2]],
    ], axis=-1).astype(np.float32)

    owners = np.empty(values.shape[0], dtype=np.int32)
    for begin in range(0, values.shape[0], chunk):
        q = torch.from_numpy(xyz[begin:begin + chunk]).to(device)
        owners[begin:begin + chunk] = radfoam.nn(points, aabb_tree, q).int().cpu().numpy()
    return xyz, values, grad, owners


def _fit_oracle(points: np.ndarray, radius: np.ndarray, xyz: np.ndarray,
                values: np.ndarray, gradients: np.ndarray, owners: np.ndarray,
                max_cells: int, min_samples: int, min_contrast: float,
                min_improvement: float):
    """Fit gradient-seeded, thresholded two-constant planes per cell."""
    order = np.argsort(owners, kind="stable")
    sorted_owner = owners[order]
    unique, starts, counts = np.unique(sorted_owner, return_index=True, return_counts=True)
    n_cells = points.shape[0]
    scalar_mu = np.zeros(n_cells, dtype=np.float32)
    global_mean = float(values.mean())
    scalar_mu.fill(global_mean)
    candidates = []

    for cell, start, count in zip(unique.tolist(), starts.tolist(), counts.tolist()):
        ids = order[start:start + count]
        y = values[ids]
        scalar_mu[cell] = float(y.mean())
        if count < min_samples or float(y.max() - y.min()) < min_contrast:
            continue
        gsum = gradients[ids].sum(axis=0)
        gnorm = float(np.linalg.norm(gsum))
        if not np.isfinite(gnorm) or gnorm < 1e-6:
            continue
        normal = gsum / gnorm  # directed toward increasing GT density
        u = (xyz[ids] - points[cell]) @ normal
        # Search interior quantiles; duplicated thresholds are harmless.
        thresholds = np.unique(np.quantile(u, np.linspace(0.15, 0.85, 15)))
        base_sse = float(np.square(y - y.mean()).sum()) + 1e-20
        best = None
        for threshold in thresholds:
            plus = u > threshold
            np_side = int(plus.sum())
            nm_side = count - np_side
            if min(np_side, nm_side) < max(3, int(0.10 * count)):
                continue
            mu_p = float(y[plus].mean())
            mu_m = float(y[~plus].mean())
            sse = float(np.square(y[plus] - mu_p).sum() + np.square(y[~plus] - mu_m).sum())
            improvement = 1.0 - sse / base_sse
            contrast = abs(mu_p - mu_m)
            score = improvement * contrast * math.sqrt(count)
            if best is None or score > best[0]:
                best = (score, improvement, contrast, float(threshold), mu_p, mu_m, sse)
        if best is None or best[1] < min_improvement or best[2] < min_contrast:
            continue
        score, improvement, contrast, threshold, mu_p, mu_m, _ = best
        # Canonical sign: +normal side always has the higher density.
        if mu_p < mu_m:
            normal = -normal
            threshold = -threshold
            mu_p, mu_m = mu_m, mu_p
        if not np.isfinite(threshold) or abs(threshold) > max(2.0 * radius[cell], 1e-5):
            continue
        candidates.append((score, cell, normal.astype(np.float32), threshold,
                           mu_p, mu_m, improvement, contrast, count))

    candidates.sort(key=lambda row: row[0], reverse=True)
    candidates = candidates[:max_cells]
    selected = np.zeros(n_cells, dtype=bool)
    normals = np.zeros((n_cells, 3), dtype=np.float32)
    normals[:, 0] = 1.0
    offsets = np.zeros(n_cells, dtype=np.float32)
    mu_plus = scalar_mu.copy()
    mu_minus = scalar_mu.copy()
    rows = []
    for score, cell, normal, threshold, mp, mm, improvement, contrast, count in candidates:
        selected[cell] = True
        normals[cell] = normal
        offsets[cell] = threshold
        mu_plus[cell] = max(mp, 1e-8)
        mu_minus[cell] = max(mm, 1e-8)
        rows.append({
            "cell": int(cell), "score": score, "improvement": improvement,
            "contrast": contrast, "samples": int(count), "offset": threshold,
            "mu_plus": mp, "mu_minus": mm,
        })
    return scalar_mu, selected, normals, offsets, mu_plus, mu_minus, rows


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    gt = np.load(args.gt).astype(np.float32)
    scene = _load_scene(args.base_model, device)
    points = scene.primal_points.detach()
    with torch.no_grad():
        radius = scene._get_cell_radius().detach().cpu().numpy().reshape(-1)
    xyz, values, gradients, owners = _assign_samples(
        points, scene.aabb_tree, gt, args.grid_stride, args.nn_chunk, device)
    fit = _fit_oracle(
        points.detach().cpu().numpy(), radius, xyz, values, gradients, owners,
        args.max_cells, args.min_samples,
        args.min_contrast_frac * float(np.percentile(gt, 99)), args.min_improvement)
    scalar_mu, selected, normals, offsets, mu_plus, mu_minus, rows = fit
    if int(selected.sum()) < args.min_selected:
        raise RuntimeError(f"only {selected.sum()} informative cells selected; need {args.min_selected}")

    normals_t = torch.from_numpy(normals)
    quats = _normal_to_quaternion(normals_t)
    heights = torch.zeros(points.shape[0], 4)
    heights[selected] = torch.from_numpy(
        offsets[selected] / np.maximum(radius[selected], 1e-8)).unsqueeze(-1).expand(-1, 4)
    raw_plus = _inverse_softplus_beta10(torch.from_numpy(mu_plus)).unsqueeze(-1)
    raw_minus = _inverse_softplus_beta10(torch.from_numpy(mu_minus)).unsqueeze(-1)
    _configure_independent_scene(scene, raw_plus, raw_minus, quats, heights)
    oracle_path = out / "oracle_model.pt"
    scene.save_pt(str(oracle_path))
    meta = {
        "selected": torch.from_numpy(selected),
        "oracle_normals": normals_t,
        "oracle_quaternions": quats,
        "offsets": torch.from_numpy(offsets),
        "cell_radius": torch.from_numpy(radius),
        "scalar_mu_gt": torch.from_numpy(scalar_mu),
        "mu_plus_gt": torch.from_numpy(mu_plus),
        "mu_minus_gt": torch.from_numpy(mu_minus),
    }
    torch.save(meta, out / "oracle_meta.pt")
    with (out / "oracle_cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "base_model": os.path.abspath(args.base_model),
        "gt": os.path.abspath(args.gt),
        "grid_stride": args.grid_stride,
        "grid_samples": int(values.size),
        "selected_cells": int(selected.sum()),
        "max_cells": args.max_cells,
        "p99_gt": float(np.percentile(gt, 99)),
        "median_contrast": float(np.median([r["contrast"] for r in rows])),
        "median_fit_improvement": float(np.median([r["improvement"] for r in rows])),
        "median_samples": float(np.median([r["samples"] for r in rows])),
        "oracle_model": str(oracle_path),
    }
    _json_dump(out / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _perturb_normals(normals: torch.Tensor, selected: torch.Tensor,
                     angle_deg: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=normals.device).manual_seed(seed)
    random = torch.randn(normals.shape, generator=g, device=normals.device)
    tangent = random - (random * normals).sum(-1, keepdim=True) * normals
    tangent = F.normalize(tangent, dim=-1)
    theta = math.radians(angle_deg)
    perturbed = math.cos(theta) * normals + math.sin(theta) * tangent
    return torch.where(selected.unsqueeze(-1), F.normalize(perturbed, dim=-1), normals)


def _angle_stats(q: torch.Tensor, oracle_n: torch.Tensor, selected: torch.Tensor):
    n = _quaternion_to_normal(q.detach())
    dot = (n[selected] * oracle_n[selected]).sum(-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(dot)).cpu().numpy()
    return {
        "median_deg": float(np.median(angle)),
        "p90_deg": float(np.percentile(angle, 90)),
        "mean_deg": float(np.mean(angle)),
        "frac_le_5deg": float(np.mean(angle <= 5.0)),
        "frac_le_10deg": float(np.mean(angle <= 10.0)),
    }


def _gradient_alignment(q: torch.Tensor, loss_grad: torch.Tensor,
                        oracle_n: torch.Tensor, selected: torch.Tensor):
    probe = q.detach().clone().requires_grad_(True)
    proxy = 1.0 - (_quaternion_to_normal(probe)[selected] * oracle_n[selected]).sum(-1).mean()
    angle_grad = torch.autograd.grad(proxy, probe)[0]
    lg = loss_grad[selected]
    ag = angle_grad[selected]
    # Remove radial directions because q is normalized after every step.
    qs = F.normalize(q.detach()[selected], dim=-1)
    lg = lg - (lg * qs).sum(-1, keepdim=True) * qs
    ag = ag - (ag * qs).sum(-1, keepdim=True) * qs
    grad_norm = lg.norm(dim=-1)
    active = torch.isfinite(grad_norm) & (grad_norm > 1e-12)
    cosine = F.cosine_similarity(lg[active], ag[active], dim=-1)
    finite = torch.isfinite(cosine)
    cosine = cosine[finite]
    if cosine.numel() == 0:
        return {
            "mean_cosine": float("nan"), "median_cosine": float("nan"),
            "fraction_aligned": float("nan"), "active_cells": 0,
            "selected_cells": int(selected.sum().item()), "active_fraction": 0.0,
        }
    return {
        "mean_cosine": float(cosine.mean().item()),
        "median_cosine": float(cosine.median().item()),
        "fraction_aligned": float((cosine > 0).float().mean().item()),
        "active_cells": int(cosine.numel()),
        "selected_cells": int(selected.sum().item()),
        "active_fraction": float(cosine.numel() / selected.sum().item()),
    }


def recover(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = _load_scene(args.oracle_model, args.device)
    meta = torch.load(args.oracle_meta, map_location=args.device, weights_only=True)
    selected = meta["selected"].to(args.device).bool()
    oracle_n = F.normalize(meta["oracle_normals"].to(args.device), dim=-1)
    oracle_q = meta["oracle_quaternions"].to(args.device)
    if not selected.any():
        raise RuntimeError("oracle metadata has no selected cells")

    dataset = R2GaussianDataset(args.data_path, split="train")
    all_rays = dataset.all_rays.reshape(-1, 6)
    all_measured = dataset.all_projections.reshape(-1)
    rng = np.random.default_rng(args.ray_seed)
    ray_idx = rng.choice(all_rays.shape[0], size=min(args.num_rays, all_rays.shape[0]), replace=False)
    rays_cpu = all_rays[ray_idx].contiguous()
    measured = all_measured[ray_idx].float().contiguous()

    # Oracle target uses the exact same renderer and fixed acquisition rays.
    with torch.no_grad():
        scene.quaternions.copy_(oracle_q)
    if args.target == "teacher":
        target = _render_chunks(scene, rays_cpu.to(args.device), args.render_chunk)
    else:
        target = measured

    perturbed_n = _perturb_normals(oracle_n, selected, args.angle, args.seed)
    with torch.no_grad():
        scene.quaternions.copy_(_normal_to_quaternion(perturbed_n))
    initial_q = scene.quaternions.detach().clone()
    for name in ("primal_points", "density", "raw_plus", "raw_minus",
                 "texel_sites_2d", "texel_heights"):
        getattr(scene, name).requires_grad_(False)
    scene.quaternions.requires_grad_(True)
    optimizer = torch.optim.Adam([scene.quaternions], lr=args.lr, eps=1e-8)

    eval_rays = rays_cpu.to(args.device)
    eval_target = target.to(args.device)
    mini_rng = np.random.default_rng(args.seed + 1000)
    history = []

    def evaluate(step: int):
        pred = _render_chunks(scene, eval_rays, args.render_chunk)
        diff = pred - target
        row = {
            "step": step,
            "mse": float(torch.mean(diff * diff).item()),
            "mae": float(torch.mean(diff.abs()).item()),
            **_angle_stats(scene.quaternions, oracle_n, selected),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    evaluate(0)

    # Initial MSE gradient alignment toward the oracle orientation.
    fd_count = min(args.fd_rays, eval_rays.shape[0])
    fd_rays = eval_rays[:fd_count]
    fd_target = eval_target[:fd_count]
    optimizer.zero_grad(set_to_none=True)
    alignment_loss = F.mse_loss(_render(scene, fd_rays), fd_target)
    alignment_loss.backward()
    loss_grad = scene.quaternions.grad.detach().clone()
    alignment = _gradient_alignment(scene.quaternions, loss_grad, oracle_n, selected)

    # Realistic directional FD check of the independent-mode quaternion
    # rendering Jacobian. Use a signed projection objective rather than MSE:
    # near the teacher target MSE derivatives are tiny enough that subtracting
    # two float32 losses obscures the central difference.
    optimizer.zero_grad(set_to_none=True)
    fd_pred = _render(scene, fd_rays)
    weight_gen = torch.Generator(device=args.device).manual_seed(args.seed + 991)
    fd_weights = torch.randn(fd_pred.shape, generator=weight_gen, device=args.device)
    fd_weights = fd_weights / math.sqrt(fd_pred.numel())
    # Center the objective at the unperturbed prediction for FD evaluation.
    # Without centering, summing O(1) projections in float32 makes the tiny
    # O(eps) difference fall below the accumulator ULP.
    fd_base_pred = fd_pred.detach().clone()
    fd_objective = ((fd_pred - fd_base_pred) * fd_weights).sum()
    fd_objective.backward()
    jacobian_grad = scene.quaternions.grad.detach().clone()
    direction = torch.zeros_like(scene.quaternions)
    sel_ids = selected.nonzero(as_tuple=False).flatten()
    probe_ids = sel_ids[:min(args.fd_cells, sel_ids.numel())]
    direction[probe_ids] = torch.randn_like(direction[probe_ids])
    qn = F.normalize(scene.quaternions.detach(), dim=-1)
    direction = direction - (direction * qn).sum(-1, keepdim=True) * qn
    direction = direction / direction.norm().clamp_min(1e-12)
    analytic_dir = float((jacobian_grad * direction).sum().item())
    q_base = scene.quaternions.detach().clone()
    fd_losses = []
    with torch.no_grad():
        for sign in (1.0, -1.0):
            scene.quaternions.copy_(F.normalize(q_base + sign * args.fd_eps * direction, dim=-1))
            fd_losses.append(float(((_render(scene, fd_rays) - fd_base_pred) * fd_weights).sum().item()))
        scene.quaternions.copy_(q_base)
    finite_dir = (fd_losses[0] - fd_losses[1]) / (2.0 * args.fd_eps)
    fd_rel = abs(analytic_dir - finite_dir) / max(abs(finite_dir), abs(analytic_dir), 1e-12)
    fd_summary = {
        "analytic_directional_derivative": analytic_dir,
        "finite_difference_directional_derivative": finite_dir,
        "relative_error": fd_rel,
        "eps": args.fd_eps,
        "rays": fd_count,
        "probe_cells": int(probe_ids.numel()),
    }
    print("FD", json.dumps(fd_summary), flush=True)
    print("ALIGNMENT", json.dumps(alignment), flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        ids = mini_rng.choice(eval_rays.shape[0], size=min(args.batch_rays, eval_rays.shape[0]), replace=False)
        ids = torch.from_numpy(ids).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        pred = _render(scene, eval_rays[ids])
        loss = F.mse_loss(pred, eval_target[ids])
        loss.backward()
        with torch.no_grad():
            scene.quaternions.grad[~selected] = 0
        optimizer.step()
        with torch.no_grad():
            scene.quaternions.copy_(F.normalize(scene.quaternions, dim=-1))
            scene.quaternions[~selected] = oracle_q[~selected]
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)

    final = history[-1]
    summary = {
        "target": args.target,
        "angle_deg": args.angle,
        "seed": args.seed,
        "selected_cells": int(selected.sum().item()),
        "num_rays": int(eval_rays.shape[0]),
        "steps": args.steps,
        "lr": args.lr,
        "runtime_seconds": time.time() - t0,
        "initial": history[0],
        "final": final,
        "angle_median_improvement_deg": history[0]["median_deg"] - final["median_deg"],
        "mse_ratio_final_to_initial": final["mse"] / max(history[0]["mse"], 1e-30),
        "gradient_alignment": alignment,
        "directional_fd": fd_summary,
    }
    stem = f"recover_{args.target}_angle{args.angle:g}_seed{args.seed}"
    _json_dump(out / f"{stem}.json", summary)
    _json_dump(out / f"{stem}_history.json", history)
    torch.save({
        "initial_quaternions": initial_q.cpu(),
        "final_quaternions": scene.quaternions.detach().cpu(),
        "selected": selected.cpu(),
    }, out / f"{stem}.pt")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


def summarize(args):
    rows = []
    for path in sorted(glob.glob(str(Path(args.output) / "recover_*_seed*.json"))):
        if path.endswith("_history.json"):
            continue
        d = json.loads(Path(path).read_text())
        rows.append({
            "target": d["target"], "angle_deg": d["angle_deg"], "seed": d["seed"],
            "initial_median_deg": d["initial"]["median_deg"],
            "final_median_deg": d["final"]["median_deg"],
            "final_p90_deg": d["final"]["p90_deg"],
            "final_frac_le_5deg": d["final"]["frac_le_5deg"],
            "final_frac_le_10deg": d["final"]["frac_le_10deg"],
            "mse_ratio": d["mse_ratio_final_to_initial"],
            "gradient_alignment_mean": d["gradient_alignment"]["mean_cosine"],
            "gradient_fraction_aligned": d["gradient_alignment"]["fraction_aligned"],
            "fd_relative_error": d["directional_fd"]["relative_error"],
        })
    if not rows:
        raise RuntimeError(f"no recovery JSON files found under {args.output}")
    with (Path(args.output) / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base-model", required=True)
    prep.add_argument("--gt", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--device", default="cuda:0")
    prep.add_argument("--grid-stride", type=int, default=2)
    prep.add_argument("--nn-chunk", type=int, default=500_000)
    prep.add_argument("--max-cells", type=int, default=4096)
    prep.add_argument("--min-selected", type=int, default=256)
    prep.add_argument("--min-samples", type=int, default=16)
    prep.add_argument("--min-contrast-frac", type=float, default=0.05)
    prep.add_argument("--min-improvement", type=float, default=0.10)
    prep.set_defaults(func=prepare)

    rec = sub.add_parser("recover")
    rec.add_argument("--oracle-model", required=True)
    rec.add_argument("--oracle-meta", required=True)
    rec.add_argument("--data-path", required=True)
    rec.add_argument("--output", required=True)
    rec.add_argument("--target", choices=("teacher", "measured"), default="teacher")
    rec.add_argument("--angle", type=float, required=True)
    rec.add_argument("--device", default="cuda:0")
    rec.add_argument("--seed", type=int, default=42)
    rec.add_argument("--ray-seed", type=int, default=1234)
    rec.add_argument("--num-rays", type=int, default=131072)
    rec.add_argument("--batch-rays", type=int, default=32768)
    rec.add_argument("--render-chunk", type=int, default=32768)
    rec.add_argument("--steps", type=int, default=300)
    rec.add_argument("--eval-every", type=int, default=25)
    rec.add_argument("--lr", type=float, default=5e-4)
    rec.add_argument("--fd-rays", type=int, default=8192)
    rec.add_argument("--fd-cells", type=int, default=64)
    rec.add_argument("--fd-eps", type=float, default=1e-3)
    rec.set_defaults(func=recover)

    summ = sub.add_parser("summarize")
    summ.add_argument("--output", required=True)
    summ.set_defaults(func=summarize)
    return p


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
