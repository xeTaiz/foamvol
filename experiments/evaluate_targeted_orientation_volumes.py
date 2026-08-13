#!/usr/bin/env python3
"""Volume and tested-cell ROI PSNR for targeted orientation experiments."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import radfoam  # noqa: E402
from split_voxelize import split_cell_query  # noqa: E402


def _psnr(mse, data_range):
    return float("inf") if mse == 0 else float(10 * np.log10(data_range * data_range / mse))


def _load_variants(oracle_path, targeted_dir, device):
    scene = torch.load(oracle_path, map_location=device, weights_only=True)
    meta = scene.get("thin_surface")
    if not isinstance(meta, dict) or not meta.get("active"):
        raise RuntimeError("oracle checkpoint has no active thin surface")
    if meta.get("density_mode", "absolute") not in ("absolute", None):
        raise RuntimeError("evaluator expects stable absolute oracle checkpoint")
    paths = sorted(glob.glob(str(Path(targeted_dir) / "targeted_*_angle15_cells64_rpc*_seed42.pt")))
    if len(paths) != 8:
        raise RuntimeError(f"expected 8 targeted checkpoints, found {len(paths)}")
    cells0 = initial0 = None
    variants = {"oracle": scene["quaternions"].to(device)}
    for path in paths:
        d = torch.load(path, map_location=device, weights_only=True)
        cells, initial, final = d["cells"].long(), d["initial_quaternions"], d["final_quaternions"]
        if cells0 is None:
            cells0, initial0 = cells, initial
            variants["perturbed"] = initial.to(device)
        elif not torch.equal(cells0, cells) or not torch.equal(initial0, initial):
            raise RuntimeError(f"inconsistent targeted checkpoint {path}")
        name = Path(path).stem.replace("targeted_", "").replace("_angle15_cells64", "").replace("_seed42", "")
        variants[name] = final.to(device)
    return scene, cells0.to(device), variants


def evaluate(args):
    device = torch.device(args.device)
    out = Path(args.output)
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    scene, cells, variants = _load_variants(args.oracle_model, args.targeted_dir, device)
    gt = np.load(args.gt).astype(np.float32)
    if gt.shape != (args.resolution,) * 3:
        raise RuntimeError(f"GT {gt.shape} != requested resolution {args.resolution}")
    data_range = float(gt.max() - gt.min())

    points = scene["xyz"].to(device)
    density = scene["density"].to(device).squeeze(-1)
    delta = scene["density_delta"].to(device)
    sites = scene["texel_sites_2d"].to(device)
    heights = scene["texel_heights"].to(device)
    adjacency = scene["adjacency"].to(device).to(torch.uint32)
    offsets_csr = scene["adjacency_offsets"].to(device).to(torch.uint32)
    tree = radfoam.build_aabb_tree(points)
    _, radius = radfoam.farthest_neighbor(points, adjacency, offsets_csr)
    radius = radius.squeeze()
    activation_scale = float(args.activation_scale)
    thin_temp = float(args.thin_temp)

    r = args.resolution
    coords = (torch.arange(r, device=device) + 0.5) / r
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    centers = -1.0 + 2.0 * torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    owner_centers = torch.empty(centers.shape[0], dtype=torch.long, device="cpu")
    values = {name: torch.empty(centers.shape[0], dtype=torch.float32) for name in variants}

    k = args.supersample
    sub = torch.linspace(-0.5 + 0.5 / k, 0.5 - 0.5 / k, k, device=device)
    ox, oy, oz = torch.meshgrid(sub, sub, sub, indexing="ij")
    sub_offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3) * (2.0 / r)
    spv = k ** 3
    batch = max(1, args.max_samples // spv)
    cell_lookup = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
    cell_lookup[cells] = True

    for start in range(0, centers.shape[0], batch):
        end = min(start + batch, centers.shape[0])
        c = centers[start:end]
        owner_c = radfoam.nn(points, tree, c).long()
        owner_centers[start:end] = owner_c.cpu()
        q = (c[:, None, :] + sub_offsets[None, :, :]).reshape(-1, 3)
        owner = radfoam.nn(points, tree, q).long()
        for name, quat in variants.items():
            val, _, _ = split_cell_query(
                q, points, owner, density, delta, quat, sites, heights, radius,
                thin_temp=thin_temp, activation_scale=activation_scale,
                blend_eps=0.0, density_mode="absolute")
            values[name][start:end] = torch.nan_to_num(val).reshape(-1, spv).mean(1).cpu()
        if start == 0 or end == centers.shape[0] or start // batch % 20 == 0:
            print(f"{end}/{centers.shape[0]}", flush=True)

    owner_np = owner_centers.numpy()
    cells_np = cells.cpu().numpy()
    roi = np.isin(owner_np, cells_np).reshape(gt.shape)
    np.save(out / "tested_cell_roi.npy", roi)
    gt_flat = gt.reshape(-1)
    roi_flat = roi.reshape(-1)
    perturbed_metrics = None
    rows = []

    def metrics(pred):
        err2 = np.square(pred - gt_flat, dtype=np.float64)
        mse_global = float(err2.mean())
        mse_roi = float(err2[roi_flat].mean())
        mse_out = float(err2[~roi_flat].mean())
        per = []
        for cell in cells_np:
            mask = owner_np == cell
            if mask.any():
                per.append(_psnr(float(err2[mask].mean()), data_range))
        return {
            "global_mse": mse_global, "global_psnr": _psnr(mse_global, data_range),
            "roi_mse": mse_roi, "roi_psnr": _psnr(mse_roi, data_range),
            "outside_mse": mse_out, "outside_psnr": _psnr(mse_out, data_range),
            "roi_voxels": int(roi_flat.sum()), "roi_fraction": float(roi_flat.mean()),
            "per_cell_psnr_median": float(np.median(per)),
            "per_cell_psnr_p10": float(np.percentile(per, 10)),
            "per_cell_psnr_p90": float(np.percentile(per, 90)),
            "per_cell_count": len(per),
        }

    computed = {}
    for name, tensor in values.items():
        pred = tensor.numpy()
        np.save(out / "volumes" / f"{name}.npy", pred.reshape(gt.shape))
        computed[name] = metrics(pred)
    perturbed_metrics = computed["perturbed"]
    for name in variants:
        row = {"variant": name, **computed[name]}
        row["global_psnr_improvement_vs_perturbed"] = row["global_psnr"] - perturbed_metrics["global_psnr"]
        row["roi_psnr_improvement_vs_perturbed"] = row["roi_psnr"] - perturbed_metrics["roi_psnr"]
        row["per_cell_median_improvement_vs_perturbed"] = row["per_cell_psnr_median"] - perturbed_metrics["per_cell_psnr_median"]
        rows.append(row)
    rows.sort(key=lambda x: (0 if x["variant"] == "oracle" else 1 if x["variant"] == "perturbed" else 2, x["variant"]))
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-model", required=True)
    p.add_argument("--targeted-dir", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--supersample", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int, default=2_000_000)
    p.add_argument("--activation-scale", type=float, default=1.0)
    p.add_argument("--thin-temp", type=float, default=10.0)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
