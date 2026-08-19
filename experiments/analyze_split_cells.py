#!/usr/bin/env python3
"""Quantify genuinely active cells in a bounded-relative split checkpoint.

The analysis assigns a uniform grid of in-volume voxel centers to their
Voronoi owners, checks whether each sampled cell region contains both signs of
its learned implicit surface, and combines that geometric test with absolute
and relative density-contrast tests.  It writes ``summary.json``, ``cells.csv``,
an inspectable ``examples.png``, and per-cell PNGs under ``web_panels`` to
``--output``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import radfoam  # noqa: E402
from split_voxelize import (  # noqa: E402
    assert_supported_thin_K, quat_to_frame, split_cell_query,
)
from voxel_grid import ALIGN_CORNERS, voxel_center_coords_np  # noqa: E402

REL_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.5)
ABS_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def _metadata_float(scene, thin_meta, key, default=None):
    """Read scalar run metadata without adding a CLI override."""
    containers = (thin_meta, scene.get("metadata"), scene.get("args"), scene)
    for container in containers:
        if isinstance(container, dict) and key in container:
            value = container[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            return float(value), f"checkpoint:{key}"
    if default is None:
        raise RuntimeError(f"checkpoint metadata is missing required {key!r}")
    return float(default), f"format default:{default}"


def _finite_quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"q{int(q * 100):02d}": None for q in QUANTILES}
    result = np.quantile(values, QUANTILES)
    return {f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, result)}


def _group_quantiles(mask, sample_count, min_s, max_s, mu_bar, abs_diff, rel_diff):
    span = max_s - min_s
    return {
        "count": int(mask.sum()),
        "sample_count": _finite_quantiles(sample_count[mask]),
        "min_s": _finite_quantiles(min_s[mask]),
        "max_s": _finite_quantiles(max_s[mask]),
        "s_span": _finite_quantiles(span[mask]),
        "mu_bar": _finite_quantiles(mu_bar[mask]),
        "abs_side_difference": _finite_quantiles(abs_diff[mask]),
        "relative_difference": _finite_quantiles(rel_diff[mask]),
    }


def _count_report(mask, total, crossing_count):
    count = int(mask.sum())
    return {
        "count": count,
        "fraction_of_all_cells": float(count / total) if total else 0.0,
        "fraction_of_geometry_crossing": (
            float(count / crossing_count) if crossing_count else 0.0),
    }


def _load(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    scene = torch.load(args.model, map_location="cpu", weights_only=True)
    required_base = ("xyz", "density", "adjacency", "adjacency_offsets")
    missing = [key for key in required_base if key not in scene]
    if missing:
        raise RuntimeError(f"checkpoint is missing required tensors: {missing}")

    meta = scene.get("thin_surface")
    if not isinstance(meta, dict) or not meta.get("active", False):
        raise RuntimeError("checkpoint must have active thin_surface metadata")
    mode = meta.get("density_mode")
    if mode is None and meta.get("relative_delta", False):
        mode = "relative"
    if mode != "relative" or not meta.get("relative_delta", mode == "relative"):
        raise RuntimeError(
            "checkpoint must use active bounded-relative thin_surface density_mode")
    required_split = ("density_delta", "quaternions", "texel_sites_2d", "texel_heights")
    missing = [key for key in required_split if key not in scene]
    if missing:
        raise RuntimeError(f"relative split checkpoint is missing tensors: {missing}")

    points = scene["xyz"].to(device)
    density = scene["density"].to(device).reshape(-1)
    delta = scene["density_delta"].to(device)
    quat = scene["quaternions"].to(device)
    sites = scene["texel_sites_2d"].to(device)
    heights = scene["texel_heights"].to(device)
    n_cells = points.shape[0]
    expected = {
        "density": n_cells, "density_delta": n_cells, "quaternions": n_cells,
        "texel_sites_2d": n_cells, "texel_heights": n_cells,
    }
    tensors = {"density": density, "density_delta": delta, "quaternions": quat,
               "texel_sites_2d": sites, "texel_heights": heights}
    bad = [f"{name}={tuple(tensors[name].shape)}" for name, n in expected.items()
           if tensors[name].shape[0] != n]
    if bad:
        raise RuntimeError("split tensor leading dimensions disagree: " + ", ".join(bad))
    if quat.shape != (n_cells, 4) or sites.ndim != 3 or sites.shape[-1] != 2:
        raise RuntimeError("expected quaternions (N,4) and texel_sites_2d (N,K,2)")
    if heights.shape != sites.shape[:2]:
        raise RuntimeError("texel_heights (N,K) must match texel_sites_2d (N,K,2)")
    if int(meta.get("K", sites.shape[1])) != sites.shape[1]:
        raise RuntimeError("thin_surface K metadata disagrees with texel tensors")
    assert_supported_thin_K(sites.shape[1])

    # Match split_voxelize.voxelize_split exactly for CUDA acceleration state.
    adjacency = scene["adjacency"].to(device).to(torch.uint32)
    offsets = scene["adjacency_offsets"].to(device).to(torch.uint32)
    tree = radfoam.build_aabb_tree(points)
    _, radius = radfoam.farthest_neighbor(points, adjacency, offsets)
    radius = radius.reshape(-1)

    rho = float(meta.get("delta_max_frac", 0.5))
    rho_source = ("checkpoint:delta_max_frac" if "delta_max_frac" in meta
                  else "format default:0.5")
    if args.activation_scale is None:
        activation_scale, scale_source = _metadata_float(
            scene, meta, "activation_scale", default=1.0)
    else:
        activation_scale, scale_source = float(args.activation_scale), "CLI"
    if args.thin_temp is None:
        thin_temp, temp_source = _metadata_float(
            scene, meta, "thin_temp", default=10.0)
    else:
        thin_temp, temp_source = float(args.thin_temp), "CLI"
    return (scene, meta, points, density, delta, quat, sites, heights, tree,
            radius, rho, activation_scale, thin_temp,
            {"rho": rho_source, "activation_scale": scale_source,
             "thin_temp": temp_source})


def _sample_grid(args, points, density, delta, quat, sites, heights, tree,
                 radius, rho, activation_scale, thin_temp):
    n_cells = points.shape[0]
    total = args.resolution ** 3
    device = points.device
    counts = torch.zeros(n_cells, dtype=torch.int64, device=device)
    min_s = torch.full((n_cells,), float("inf"), device=device)
    max_s = torch.full((n_cells,), -float("inf"), device=device)

    with torch.inference_mode():
        for begin in range(0, total, args.chunk):
            end = min(begin + args.chunk, total)
            flat = torch.arange(begin, end, dtype=torch.int64, device=device)
            plane = args.resolution * args.resolution
            ix = torch.div(flat, plane, rounding_mode="floor")
            rem = flat - ix * plane
            iy = torch.div(rem, args.resolution, rounding_mode="floor")
            iz = rem - iy * args.resolution
            q = torch.stack((ix, iy, iz), dim=-1).to(points.dtype)
            q = -1.0 + 2.0 * (q + 0.5) / args.resolution
            owner = radfoam.nn(points, tree, q).long()
            _, _, signed = split_cell_query(
                q, points, owner, density, delta, quat, sites, heights, radius,
                thin_temp=thin_temp, activation_scale=activation_scale,
                blend_eps=0.0, density_mode="relative", delta_max_frac=rho)
            finite = torch.isfinite(signed)
            owner_finite = owner[finite]
            signed_finite = signed[finite]
            counts += torch.bincount(owner_finite, minlength=n_cells)
            min_s.scatter_reduce_(
                0, owner_finite, signed_finite, reduce="amin", include_self=True)
            max_s.scatter_reduce_(
                0, owner_finite, signed_finite, reduce="amax", include_self=True)
            print(f"sampled {end:,}/{total:,} voxel centers", flush=True)
    return counts.cpu().numpy(), min_s.cpu().numpy(), max_s.cpu().numpy()


def _write_csv(path, points, radius, normals, density, raw_delta, sample_count,
               min_s, max_s, mu_bar, effective_delta, mu_plus, mu_minus,
               abs_diff, rel_diff, crossing, primary, rel_flags, abs_flags):
    fields = [
        "cell_id", "x", "y", "z", "normal_x", "normal_y", "normal_z",
        "cell_radius", "raw_density", "raw_density_delta", "sample_count",
        "min_s", "max_s", "s_span", "mu_bar", "effective_delta", "mu_plus",
        "mu_minus", "abs_side_difference", "relative_difference",
        "geometry_crossing", "meaningful_active",
    ]
    fields += [f"crossing_rel_ge_{t:g}" for t in REL_THRESHOLDS]
    fields += [f"crossing_mu_and_abs_ge_{t:g}_p99" for t in ABS_THRESHOLDS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(sample_count)):
            row = {
                "cell_id": i, "x": points[i, 0], "y": points[i, 1], "z": points[i, 2],
                "normal_x": normals[i, 0], "normal_y": normals[i, 1], "normal_z": normals[i, 2],
                "cell_radius": radius[i], "raw_density": density[i],
                "raw_density_delta": raw_delta[i], "sample_count": int(sample_count[i]),
                "min_s": min_s[i], "max_s": max_s[i], "s_span": max_s[i] - min_s[i],
                "mu_bar": mu_bar[i], "effective_delta": effective_delta[i],
                "mu_plus": mu_plus[i], "mu_minus": mu_minus[i],
                "abs_side_difference": abs_diff[i], "relative_difference": rel_diff[i],
                "geometry_crossing": int(crossing[i]), "meaningful_active": int(primary[i]),
            }
            row.update({f"crossing_rel_ge_{t:g}": int(rel_flags[t][i])
                        for t in REL_THRESHOLDS})
            row.update({f"crossing_mu_and_abs_ge_{t:g}_p99": int(abs_flags[t][i])
                        for t in ABS_THRESHOLDS})
            writer.writerow(row)


def _pixel_boundaries(owner):
    boundary = np.zeros(owner.shape, dtype=bool)
    boundary[1:, :] |= owner[1:, :] != owner[:-1, :]
    boundary[:-1, :] |= owner[1:, :] != owner[:-1, :]
    boundary[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    boundary[:, :-1] |= owner[:, 1:] != owner[:, :-1]
    return boundary


def _make_figure(path, web_dir, gt, selected, primary, points, density, delta,
                 quat, sites, heights, tree, radius, rho, activation_scale,
                 thin_temp, p99, mu_plus, mu_minus, abs_diff, rel_diff,
                 sample_count, min_s, max_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    web_dir.mkdir(parents=True, exist_ok=True)
    if len(selected) == 0:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "No geometry-crossing cells were found.\n"
                "No split-cell examples are available at this sampling resolution.",
                ha="center", va="center", fontsize=14)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    cols = min(3, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows), squeeze=False)
    # Do not clip the selected sides at GT p99.  That was misleading for e.g.
    # mu-/mu+=0.795/0.472 when p99=0.473: both sides appeared white despite a
    # 0.323 physical density difference.  This remains one shared physical
    # scale across the figure and web panels, but includes every selected side
    # value.  GT is intentionally allowed to clip above this learned-data cap.
    display_max = max(
        float(p99),
        1.02 * float(np.max(np.concatenate((mu_plus[selected], mu_minus[selected])))),
    )
    display_max = max(display_max, np.finfo(float).eps)
    device = points.device
    uv = torch.linspace(-1.0, 1.0, 320, device=device, dtype=points.dtype)
    uu, vv = torch.meshgrid(uv, uv, indexing="xy")

    # NumPy volumes are (x,y,z), while 5-D grid_sample sources are (z,y,x).
    # Copying also handles read-only memmaps and non-native NumPy storage safely.
    gt_xyz = np.array(gt, dtype=np.float32, order="C", copy=True)
    gt_zyx = torch.from_numpy(gt_xyz).permute(2, 1, 0).contiguous()
    gt_source = gt_zyx[None, None].to(device=device)

    def draw_oblique(ax, image, owner_np, signed_np, cell):
        ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap="gray",
                  vmin=0.0, vmax=display_max, interpolation="nearest")
        all_border = _pixel_boundaries(owner_np)
        ax.imshow(np.ma.masked_where(~all_border, all_border), origin="lower",
                  extent=(-1, 1, -1, 1), cmap=ListedColormap(["0.85"]),
                  interpolation="nearest", alpha=0.75)
        target_mask = owner_np == cell
        target_border = _pixel_boundaries(target_mask.astype(np.int8))
        ax.imshow(np.ma.masked_where(~target_border, target_border), origin="lower",
                  extent=(-1, 1, -1, 1), cmap=ListedColormap(["yellow"]),
                  interpolation="nearest")
        # `signed_np` is evaluated with each pixel's actual Voronoi owner.
        # Contour each owner's field only inside its own region: contouring the
        # whole piecewise field would create false zero-lines across owner seams.
        coords = np.linspace(-1, 1, 320)
        for owner_id in np.unique(owner_np):
            owner_mask = owner_np == owner_id
            owner_signed = signed_np[owner_mask]
            if owner_signed.size < 4 or not np.isfinite(owner_signed).all():
                continue
            if owner_signed.min() < 0 < owner_signed.max():
                masked_s = np.ma.masked_where(~owner_mask, signed_np)
                ax.contour(
                    coords, coords, masked_s, levels=[0.0], colors=["magenta"],
                    linewidths=(0.55 if owner_id == cell else 0.25), alpha=0.9,
                )
        ax.plot(0, 0, marker="+", color="cyan", markersize=7,
                markeredgewidth=0.8)

    def save_square_panel(panel_path, image, owner_np, signed_np, cell):
        panel_fig = plt.figure(figsize=(3.2, 3.2), dpi=100)
        panel_ax = panel_fig.add_axes((0, 0, 1, 1))
        draw_oblique(panel_ax, image, owner_np, signed_np, cell)
        panel_ax.set_axis_off()
        panel_fig.savefig(panel_path, dpi=100)
        plt.close(panel_fig)

    for ax, cell in zip(axes.flat, selected):
        with torch.inference_mode():
            center = points[cell]
            normal = quat_to_frame(quat[cell:cell + 1])[0][0]
            # Pick a deterministic tangent not nearly parallel to the normal.
            ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=points.dtype)
            if torch.abs(torch.dot(normal, ref)) > 0.9:
                ref = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=points.dtype)
            tangent = F.normalize(torch.linalg.cross(ref, normal), dim=0)
            extent = 2.2 * radius[cell]
            q = (center + extent * (uu.reshape(-1, 1) * tangent
                                    + vv.reshape(-1, 1) * normal))
            owner = radfoam.nn(points, tree, q).long()
            value, _, signed = split_cell_query(
                q, points, owner, density, delta, quat, sites, heights, radius,
                thin_temp=thin_temp, activation_scale=activation_scale,
                blend_eps=0.0, density_mode="relative", delta_max_frac=rho)
            # q is reused verbatim as the grid_sample grid. The GT volume follows
            # the voxel-centre convention, so align_corners must be False.
            gt_grid = q.to(dtype=gt_source.dtype).reshape(1, 1, 320, 320, 3)
            gt_value = F.grid_sample(
                gt_source, gt_grid, mode="bilinear", padding_mode="zeros",
                align_corners=ALIGN_CORNERS).reshape(320, 320)
            owner_np = owner.reshape(320, 320).detach().cpu().numpy()
            value_np = torch.nan_to_num(value).reshape(320, 320).detach().cpu().numpy()
            gt_value_np = torch.nan_to_num(gt_value).detach().cpu().numpy()
            signed_np = signed.reshape(320, 320).detach().cpu().numpy()
            center_np = center.detach().cpu().numpy()
            corners = torch.stack([
                center + extent * (-tangent - normal),
                center + extent * (tangent - normal),
                center + extent * (tangent + normal),
                center + extent * (-tangent + normal),
            ]).detach().cpu().numpy()

        draw_oblique(ax, value_np, owner_np, signed_np, cell)
        save_square_panel(web_dir / f"cell_{cell}_learned.png", value_np,
                          owner_np, signed_np, cell)
        save_square_panel(web_dir / f"cell_{cell}_gt.png", gt_value_np,
                          owner_np, signed_np, cell)

        z_axis = voxel_center_coords_np(gt_xyz.shape[2], 1.0, dtype=np.float64)
        z_index = int(np.argmin(np.abs(z_axis - center_np[2])))
        axial_z = float(z_axis[z_index])
        axial_xy = gt_xyz[:, :, z_index].T
        locator_fig, locator_ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
        locator_ax.imshow(axial_xy, origin="lower", extent=(-1, 1, -1, 1),
                          cmap="gray", vmin=0.0, vmax=display_max,
                          interpolation="nearest")
        closed = np.concatenate((corners[:, :2], corners[:1, :2]), axis=0)
        locator_ax.plot(closed[:, 0], closed[:, 1], color="yellow",
                        linewidth=1.0, label="Oblique q plane")
        locator_ax.plot(center_np[0], center_np[1], marker="+", color="cyan",
                        markersize=7, markeredgewidth=1.0, linestyle="none",
                        label="Cell center")
        locator_ax.set_xlim(-1, 1)
        locator_ax.set_ylim(-1, 1)
        locator_ax.set_aspect("equal")
        locator_ax.set_title(f"GT axial slice: z={axial_z:.4g} (index {z_index})",
                             fontsize=8)
        locator_ax.legend(loc="lower right", fontsize=6, framealpha=0.7)
        locator_ax.set_xticks([])
        locator_ax.set_yticks([])
        locator_fig.tight_layout(pad=0.25)
        locator_fig.savefig(web_dir / f"cell_{cell}_locator.png", dpi=100)
        plt.close(locator_fig)

        label = "PRIMARY" if primary[cell] else "FILL / NON-PRIMARY"
        ax.set_title(
            f"cell {cell} — {label}\n"
            f"mu-/mu+ {mu_minus[cell]:.4g}/{mu_plus[cell]:.4g}  "
            f"abs {abs_diff[cell]:.4g}  rel {rel_diff[cell]:.3g}\n"
            f"samples {sample_count[cell]}  s [{min_s[cell]:.3g}, {max_s[cell]:.3g}]",
            fontsize=9, color=("black" if primary[cell] else "darkred"))
        ax.set_xlabel("stable tangent / cell slice extent")
        ax.set_ylabel("learned normal / cell slice extent")
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle(
        f"Hard split density (shared non-clipping range [0, {display_max:.4g}]; "
        f"GT p99={p99:.4g})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(args):
    if args.resolution <= 0 or args.chunk <= 0 or args.num_examples <= 0:
        raise ValueError("resolution, chunk, and num-examples must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gt = np.load(args.gt, mmap_mode="r")
    if (not np.issubdtype(gt.dtype, np.number) or np.iscomplexobj(gt)
            or gt.size == 0 or gt.ndim != 3):
        raise RuntimeError("GT must be a nonempty real numeric 3-D NumPy array")
    p99 = float(np.percentile(np.asarray(gt), 99))
    if not np.isfinite(p99) or p99 <= 0:
        raise RuntimeError(f"GT p99 must be finite and positive, got {p99}")

    (scene, meta, points, density, delta, quat, sites, heights, tree, radius,
     rho, activation_scale, thin_temp, metadata_sources) = _load(args)
    sample_count, min_s, max_s = _sample_grid(
        args, points, density, delta, quat, sites, heights, tree, radius,
        rho, activation_scale, thin_temp)

    with torch.inference_mode():
        mu_t = activation_scale * F.softplus(density, beta=10.0)
        delta_t = rho * mu_t * torch.tanh(delta.reshape(-1))
        plus_t = torch.clamp(mu_t + delta_t, min=0.0)
        minus_t = torch.clamp(mu_t - delta_t, min=0.0)
        normal_t = quat_to_frame(quat)[0]
    points_np = points.cpu().numpy()
    radius_np = radius.cpu().numpy()
    normals = normal_t.cpu().numpy()
    raw_density = density.cpu().numpy()
    raw_delta = delta.reshape(-1).cpu().numpy()
    mu_bar = mu_t.cpu().numpy()
    effective_delta = delta_t.cpu().numpy()
    mu_plus = plus_t.cpu().numpy()
    mu_minus = minus_t.cpu().numpy()
    abs_diff = 2.0 * np.abs(effective_delta)
    rel_diff = abs_diff / np.maximum(mu_bar, np.finfo(np.float32).eps)

    crossing = ((sample_count >= 8) & np.isfinite(min_s) & np.isfinite(max_s)
                & (min_s < 0) & (max_s > 0))
    base_ok = mu_bar >= 0.05 * p99
    rel_flags = {t: crossing & (rel_diff >= t) for t in REL_THRESHOLDS}
    abs_flags = {t: crossing & base_ok & (abs_diff >= t * p99)
                 for t in ABS_THRESHOLDS}
    primary = crossing & base_ok & (abs_diff >= 0.01 * p99) & (rel_diff >= 0.1)
    n_cells = len(sample_count)
    crossing_count = int(crossing.sum())

    summary = {
        "inputs": {"model": str(Path(args.model).resolve()),
                   "gt": str(Path(args.gt).resolve()),
                   "resolution": args.resolution, "chunk": args.chunk,
                   "device": str(points.device)},
        "checkpoint": {"num_cells": n_cells, "density_mode": "relative",
                       "thin_surface_K": int(sites.shape[1]), "rho": rho,
                       "activation_scale": activation_scale, "thin_temp": thin_temp,
                       "metadata_sources": metadata_sources},
        "gt_p99": p99,
        "definitions": {
            "geometry_crossing": "sample_count >= 8 and min_s < 0 and max_s > 0",
            "meaningful_active": (
                "geometry_crossing and mu_bar >= 0.05*p99GT and "
                "abs_side_difference >= 0.01*p99GT and relative_difference >= 0.1"),
        },
        "geometry_crossing": _count_report(crossing, n_cells, crossing_count),
        "crossing_relative_thresholds": {
            str(t): _count_report(rel_flags[t], n_cells, crossing_count)
            for t in REL_THRESHOLDS},
        "crossing_base_mu_and_absolute_thresholds": {
            str(t): _count_report(abs_flags[t], n_cells, crossing_count)
            for t in ABS_THRESHOLDS},
        "meaningful_active": _count_report(primary, n_cells, crossing_count),
        "quantiles": {
            "geometry_crossing": _group_quantiles(
                crossing, sample_count, min_s, max_s, mu_bar, abs_diff, rel_diff),
            "meaningful_active": _group_quantiles(
                primary, sample_count, min_s, max_s, mu_bar, abs_diff, rel_diff),
        },
    }

    _write_csv(output / "cells.csv", points_np, radius_np, normals, raw_density,
               raw_delta, sample_count, min_s, max_s, mu_bar, effective_delta,
               mu_plus, mu_minus, abs_diff, rel_diff, crossing, primary,
               rel_flags, abs_flags)

    score = abs_diff * np.sqrt(sample_count.astype(np.float64))
    primary_ids = np.flatnonzero(primary)
    primary_ids = primary_ids[np.argsort(score[primary_ids])[::-1]]
    selected = list(primary_ids[:args.num_examples])
    if len(selected) < args.num_examples:
        fill = np.flatnonzero(crossing & ~primary)
        fill = fill[np.argsort(score[fill])[::-1]]
        selected.extend(fill[:args.num_examples - len(selected)].tolist())
    summary["visualized_cell_ids"] = [int(i) for i in selected]
    summary["web_panels"] = [
        {
            "id": int(i),
            "learned": f"web_panels/cell_{int(i)}_learned.png",
            "gt": f"web_panels/cell_{int(i)}_gt.png",
            "locator": f"web_panels/cell_{int(i)}_locator.png",
            "mu_bar": float(mu_bar[i]),
            "mu_plus": float(mu_plus[i]),
            "mu_minus": float(mu_minus[i]),
            "absolute_contrast": float(abs_diff[i]),
            "relative_contrast": float(rel_diff[i]),
            "center": [float(v) for v in points_np[i]],
        }
        for i in selected
    ]
    _make_figure(output / "examples.png", output / "web_panels", gt,
                 selected, primary, points, density, delta, quat, sites, heights,
                 tree, radius, rho, activation_scale, thin_temp, p99, mu_plus,
                 mu_minus, abs_diff, rel_diff, sample_count, min_s, max_s)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser():
    p = argparse.ArgumentParser(
        description=("Measure geometrically crossing, density-meaningful split "
                     "cells in an active bounded-relative thin-surface checkpoint."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to the trained model.pt checkpoint.")
    p.add_argument("--gt", required=True,
                   help="Path to the ground-truth NumPy volume used to obtain p99GT.")
    p.add_argument("--output", required=True,
                   help=("Output directory for summary.json, cells.csv, examples.png, "
                         "and web_panels/."))
    p.add_argument("--resolution", type=int, default=192,
                   help="Uniform voxel-center samples per axis over [-1,1]^3.")
    p.add_argument("--chunk", type=int, default=1_000_000,
                   help="Maximum number of grid or query samples evaluated at once.")
    p.add_argument("--num-examples", type=int, default=12,
                   help="Maximum ranked split-cell slice panels to render.")
    p.add_argument("--device", default="cuda",
                   help="Torch device for RadFoam nearest-neighbor queries.")
    p.add_argument("--activation-scale", type=float, default=None,
                   help="Override checkpoint activation scale; defaults to metadata or 1.0.")
    p.add_argument("--thin-temp", type=float, default=None,
                   help="Override checkpoint thin-surface temperature; defaults to metadata or 10.0.")
    return p


if __name__ == "__main__":
    analyze(parser().parse_args())
