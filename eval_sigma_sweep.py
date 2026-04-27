#!/usr/bin/env python3
"""Post-hoc σ sweep evaluator for raw-only checkpoints.

Loads a saved checkpoint and evaluates volume + mesh metrics at many
(σ_spatial × σ_v) combinations without retraining.  Use this to find
whether any post-hoc σ choice for a raw-trained model matches the IDW
metrics of an interp-trained model (see sweep_35_interp_vs_raw.py).

For each (σ_s, σ_v) combo:
  - vol_idw_*:  voxelized IDW volume vs. GT
  - mesh_idw_*: marching-tets IDW surface vs. GT

Raw metrics (σ-independent) are computed once and replicated into every row:
  - vol_raw_*:  voxelized nearest-cell volume vs. GT
  - mesh_raw_*: marching-tets raw surface vs. GT

Each row also reports scale_equiv = σ_s / median(cell_radius), so results
can be read in scale-multiple terms without committing to abs-σ in training.

TensorBoard output:
  <run_dir_parent>/<run_name>_ALLSIGMAS/
    raw/          — σ-independent metrics, raw volume slices, raw mesh
    s{S}_v{V}/    — per-combo: IDW metrics, slices_interleaved, IDW mesh
  (one sub-run per combo; compare them side-by-side in TB)

Usage:
    python eval_sigma_sweep.py \\
        --config output/sweep35_interp_vs_raw/pepper-raw-512k/config.yaml

    python eval_sigma_sweep.py \\
        --config output/sweep35_interp_vs_raw/pepper-raw-512k/config.yaml \\
        --grid-spatial 0.013 0.035 --grid-sigma-v 0.2 0.8
"""

import argparse
import csv
import os
import sys
import types
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import radfoam
from torch.utils.tensorboard import SummaryWriter
from radfoam_model.scene import CTScene, idw_query
from radfoam_model.mesh import surface_metrics_vs_gt_volume, marching_tets
from vis_foam import (
    load_density_field, load_gt_volume, voxelize_volumes,
    sample_gt_slice, compute_cell_density_slice, make_slice_coords,
    query_density, sample_idw, visualize_slices,
)
from train import (
    compute_volume_psnr,
    compute_volume_ssim,
    compute_volume_ssim_3d,
    compute_dice,
    compute_surface_metrics,
    sobel_filter_3d,
)

DEFAULT_GRID_SPATIAL = [0.010, 0.020, 0.030, 0.045, 0.060, 0.080]
DEFAULT_GRID_SIGMA_V = [0.05, 0.10, 0.20, 0.40, 0.80]
SLICE_AXES   = [0, 1, 2]
SLICE_COORDS = [-0.2, 0.0, 0.2]
MESH_THRESHOLD = 0.5   # single representative iso-value for TB mesh vis


def _load_model_for_mesh(model_path, activation_scale=1.0, device="cuda"):
    """Load a checkpoint into a live CTScene with a fresh triangulation.

    load_pt restores points/density/adjacency but not the triangulation object.
    We rebuild it by calling radfoam.Triangulation directly, apply any resulting
    permutation to the parameters in-place (no optimizer needed), then call
    update_triangulation(rebuild=False) to refresh adjacency/aabb_tree from the
    new triangulation without hitting the permute_points → optimizer path.
    """
    args = types.SimpleNamespace(
        init_points=64000,
        final_points=512000,
        activation_scale=activation_scale,
        init_scale=1.05,
        init_type="random",
        init_density=0.0,
    )
    model = CTScene(args, device=torch.device(device))
    model.load_pt(model_path)

    with torch.no_grad():
        pts = model.primal_points.detach().contiguous()
        model.triangulation = radfoam.Triangulation(pts)
        perm = model.triangulation.permutation().to(torch.long)
        model.primal_points = torch.nn.Parameter(pts[perm])
        model.density = torch.nn.Parameter(model.density.detach()[perm])

    model.update_triangulation(rebuild=False)
    model.eval()
    return model


def _extract_slices(volume_np, gt_volume_np, res=256):
    """Extract 9 (axis × coord) 2D slices from a volume array and GT."""
    vol_slices, gt_slices = [], []
    for a in SLICE_AXES:
        for c in SLICE_COORDS:
            vol_slices.append(sample_gt_slice(volume_np, a, c, res, 1.0))
            gt_slices.append(sample_gt_slice(gt_volume_np, a, c, res, 1.0))
    return vol_slices, gt_slices


def _log_mesh_to_tb(writer, tag, verts_np, faces_np):
    """Log a triangle mesh to TensorBoard (add_mesh expects batched tensors)."""
    if verts_np is None or len(verts_np) == 0:
        return
    v = torch.from_numpy(verts_np).float().unsqueeze(0)   # (1, V, 3)
    f = torch.from_numpy(faces_np).int().unsqueeze(0)     # (1, F, 3)
    writer.add_mesh(tag, vertices=v, faces=f, global_step=0)


def _log_scalars(writer, vol_metrics, mesh_metrics):
    """Log vol + mesh metric dicts to TB with consistent tags."""
    for k, v in vol_metrics.items():
        writer.add_scalar(f"vol/{k}", v, global_step=0)
    for k, v in mesh_metrics.items():
        writer.add_scalar(f"mesh/{k}", v, global_step=0)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc σ sweep evaluator for raw checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True,
                        help="Path to config.yaml in the run directory")
    parser.add_argument("--out",
                        help="Output CSV path (default: sigma_sweep.csv next to config.yaml)")
    parser.add_argument("--grid-spatial", nargs="+", type=float,
                        default=DEFAULT_GRID_SPATIAL, metavar="S")
    parser.add_argument("--grid-sigma-v", nargs="+", type=float,
                        default=DEFAULT_GRID_SIGMA_V, metavar="V")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(config_path)
    run_name = os.path.basename(run_dir)
    model_path = os.path.join(run_dir, "model.pt")
    out_csv = args.out or os.path.join(run_dir, "sigma_sweep.csv")

    # TB base: sibling of run_dir so all σ sub-runs are in one TB experiment
    tb_base = os.path.join(os.path.dirname(run_dir), run_name + "_ALLSIGMAS")

    if not os.path.exists(model_path):
        print(f"[ABORT] model.pt not found: {model_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["data_path"]
    dataset_type = cfg.get("dataset", "r2_gaussian")
    activation_scale = float(cfg.get("activation_scale", 1.0))

    print(f"Config:  {config_path}")
    print(f"Model:   {model_path}")
    print(f"Output:  {out_csv}")
    print(f"TB:      {tb_base}/{{raw, s*_v*/}}")
    print(f"Dataset: {dataset_type}  path={data_path}")

    gt_volume = load_gt_volume(data_path, dataset_type)
    if gt_volume is None:
        print("[ABORT] No GT volume found; cannot evaluate.")
        sys.exit(1)
    vol_res = gt_volume.shape[0]
    vol_gt_t = torch.from_numpy(gt_volume).float().cuda()
    gt_sobel = sobel_filter_3d(vol_gt_t)
    vmax = float(gt_volume.max())

    print(f"GT volume: {gt_volume.shape}  vmax={vmax:.3f}")
    n_combos = len(args.grid_spatial) * len(args.grid_sigma_v)
    print(f"Combos: {n_combos}  ({len(args.grid_spatial)} σ_s × {len(args.grid_sigma_v)} σ_v)")

    # ── Field + cell radius ─────────────────────────────────────────────────
    print("\nLoading field...")
    field = load_density_field(model_path)

    with torch.no_grad():
        _, cr = radfoam.farthest_neighbor(
            field["points"],
            field["adjacency"].to(torch.int32),
            field["adjacency_offsets"].to(torch.int32),
        )
        median_cr = cr.median().item()
    print(f"Median cell radius: {median_cr:.6f}")

    # ── Cell density slices (σ-independent, for visualize_slices panel) ────
    print("Computing cell density slices...")
    cd_slices = [
        compute_cell_density_slice(field["points"], a, c, 64, 1.0)
        for a in SLICE_AXES for c in SLICE_COORDS
    ]

    # ── Raw volume ──────────────────────────────────────────────────────────
    print(f"\nVoxelizing raw ({vol_res}³)...")
    raw_vol, _ = voxelize_volumes(field, vol_res, 1.0, sigma=0.03, sigma_v=0.2)
    raw_vol_t = torch.from_numpy(raw_vol).float().cuda()

    raw_psnr  = compute_volume_psnr(raw_vol_t, vol_gt_t)
    raw_ssim, _ = compute_volume_ssim(raw_vol_t, vol_gt_t)
    raw_ssim3d  = compute_volume_ssim_3d(raw_vol_t, vol_gt_t)
    raw_dice, _ = compute_dice(raw_vol_t, vol_gt_t)
    raw_surf    = compute_surface_metrics(raw_vol_t, vol_gt_t)
    raw_sobel_psnr = compute_volume_psnr(sobel_filter_3d(raw_vol_t), gt_sobel)
    raw_sobel_ssim, _ = compute_volume_ssim(sobel_filter_3d(raw_vol_t), gt_sobel)

    raw_vol_metrics = {
        "psnr": raw_psnr, "ssim": raw_ssim, "ssim3d": raw_ssim3d,
        "dice": raw_dice,
        "chamfer": raw_surf["chamfer"], "hausdorff_95": raw_surf["hausdorff_95"],
        "f1_1v": raw_surf["f1_1v"], "f1_2v": raw_surf["f1_2v"],
        "sobel_psnr": raw_sobel_psnr, "sobel_ssim": raw_sobel_ssim,
    }
    raw_vol_row = {
        "vol_raw_psnr": raw_psnr, "vol_raw_ssim": raw_ssim,
        "vol_raw_ssim3d": raw_ssim3d, "vol_raw_dice": raw_dice,
        "vol_raw_chamfer": raw_surf["chamfer"],
        "vol_raw_hausdorff_95": raw_surf["hausdorff_95"],
        "vol_raw_f1_1v": raw_surf["f1_1v"], "vol_raw_f1_2v": raw_surf["f1_2v"],
        "vol_raw_sobel_psnr": raw_sobel_psnr, "vol_raw_sobel_ssim": raw_sobel_ssim,
    }
    print(f"Raw vol:  PSNR={raw_psnr:.4f}  SSIM={raw_ssim:.6f}  Dice={raw_dice:.6f}")
    print(f"          CD={raw_surf['chamfer']:.4f}v  HD95={raw_surf['hausdorff_95']:.4f}v  "
          f"F1@1v={raw_surf['f1_1v']:.4f}")

    # Raw slices (extracted from voxelized volume, reuse for every IDW combo)
    d_slices, gt_slices_list = _extract_slices(raw_vol, gt_volume, res=vol_res)

    # ── Model for mesh metrics ──────────────────────────────────────────────
    print("\nLoading model for mesh metrics...")
    model = _load_model_for_mesh(model_path, activation_scale=activation_scale)

    _pts    = model.primal_points.detach()
    _mu     = model.get_primal_density().detach().squeeze(-1)
    _adj    = model.point_adjacency
    _adj_off = model.point_adjacency_offsets
    _tree   = model.aabb_tree
    _tets   = model.triangulation.tets()

    # ── Raw mesh metrics + TB mesh ──────────────────────────────────────────
    print("Computing raw mesh metrics...")
    mesh_raw_surf = surface_metrics_vs_gt_volume(_pts, _mu, _tets, gt_volume)
    mesh_raw_metrics = {
        "chamfer": mesh_raw_surf["chamfer"],
        "hausdorff_95": mesh_raw_surf["hausdorff_95"],
        "f1_1v": mesh_raw_surf["f1_1v"], "f1_2v": mesh_raw_surf["f1_2v"],
    }
    mesh_raw_row = {
        "mesh_raw_chamfer": mesh_raw_surf["chamfer"],
        "mesh_raw_hausdorff_95": mesh_raw_surf["hausdorff_95"],
        "mesh_raw_f1_1v": mesh_raw_surf["f1_1v"],
        "mesh_raw_f1_2v": mesh_raw_surf["f1_2v"],
    }
    print(f"Mesh raw: CD={mesh_raw_surf['chamfer']:.4f}v  "
          f"HD95={mesh_raw_surf['hausdorff_95']:.4f}v  "
          f"F1@1v={mesh_raw_surf['f1_1v']:.4f}")

    print("Extracting raw mesh for TB...")
    raw_verts, raw_faces = marching_tets(_pts, _mu, _tets, threshold=MESH_THRESHOLD)

    # GT mesh (marching cubes on gt_volume at same threshold)
    from skimage.measure import marching_cubes as _mc
    try:
        _gt_verts_vox, _gt_faces, _, _ = _mc(gt_volume, level=MESH_THRESHOLD)
        gt_verts_world = (_gt_verts_vox * (2.0 / (vol_res - 1)) - 1.0).astype(np.float32)
        gt_faces_np = _gt_faces.astype(np.int32)
    except Exception:
        gt_verts_world, gt_faces_np = None, None

    # ── TB: raw sub-run ─────────────────────────────────────────────────────
    print(f"\nWriting TB raw sub-run → {tb_base}/raw/")
    raw_writer = SummaryWriter(os.path.join(tb_base, "raw"))

    _log_scalars(raw_writer, raw_vol_metrics, mesh_raw_metrics)

    visualize_slices(
        d_slices, d_slices, cd_slices,
        gt_slices=gt_slices_list,
        vmax=vmax,
        writer_fn_interleaved=partial(
            raw_writer.add_figure, "slices_interleaved", global_step=0
        ),
        writer_fn_sobel=partial(
            raw_writer.add_figure, "slices_sobel", global_step=0
        ),
    )

    _log_mesh_to_tb(raw_writer, "mesh/raw", raw_verts, raw_faces)
    if gt_verts_world is not None:
        _log_mesh_to_tb(raw_writer, "mesh/gt", gt_verts_world, gt_faces_np)

    raw_writer.close()

    # ── σ sweep ─────────────────────────────────────────────────────────────
    rows = []
    print(f"\nSweeping {n_combos} σ combos...")

    for sigma_s in args.grid_spatial:
        for sigma_v in args.grid_sigma_v:
            scale_equiv = sigma_s / median_cr if median_cr > 0 else float("nan")
            combo_name = f"s{sigma_s:.3f}_v{sigma_v:.2f}"
            print(f"  {combo_name}  (scale_equiv={scale_equiv:.2f})", flush=True)

            # IDW volume
            _, idw_vol = voxelize_volumes(field, vol_res, 1.0,
                                          sigma=sigma_s, sigma_v=sigma_v)
            idw_vol_t = torch.from_numpy(idw_vol).float().cuda()

            idw_psnr = compute_volume_psnr(idw_vol_t, vol_gt_t)
            idw_ssim, _ = compute_volume_ssim(idw_vol_t, vol_gt_t)
            idw_ssim3d = compute_volume_ssim_3d(idw_vol_t, vol_gt_t)
            idw_dice, _ = compute_dice(idw_vol_t, vol_gt_t)
            idw_surf = compute_surface_metrics(idw_vol_t, vol_gt_t)
            idw_sobel_psnr = compute_volume_psnr(sobel_filter_3d(idw_vol_t), gt_sobel)
            idw_sobel_ssim, _ = compute_volume_ssim(sobel_filter_3d(idw_vol_t), gt_sobel)

            # IDW mesh
            _ss, _sv = sigma_s, sigma_v

            @torch.no_grad()
            def _mesh_idw_fn(q,
                             pts=_pts, mu=_mu, adj=_adj, adj_off=_adj_off,
                             tree=_tree, ss=_ss, sv=_sv):
                return idw_query(
                    q, pts, adj, adj_off, tree, mu,
                    sigma=ss, sigma_v=sv,
                    per_cell_sigma=False, per_neighbor_sigma=False,
                    cell_radius=None,
                ).idw_result

            mesh_idw_surf = surface_metrics_vs_gt_volume(
                _pts, _mu, _tets, gt_volume, density_fn=_mesh_idw_fn
            )

            print(f"    IDW PSNR={idw_psnr:.4f}  Dice={idw_dice:.6f}  "
                  f"MeshF1={mesh_idw_surf['f1_1v']:.4f}")

            # ── TB: per-combo sub-run ───────────────────────────────────────
            combo_writer = SummaryWriter(os.path.join(tb_base, combo_name))

            idw_vol_metrics = {
                "psnr": idw_psnr, "ssim": idw_ssim, "ssim3d": idw_ssim3d,
                "dice": idw_dice,
                "chamfer": idw_surf["chamfer"],
                "hausdorff_95": idw_surf["hausdorff_95"],
                "f1_1v": idw_surf["f1_1v"], "f1_2v": idw_surf["f1_2v"],
                "sobel_psnr": idw_sobel_psnr, "sobel_ssim": idw_sobel_ssim,
            }
            idw_mesh_metrics = {
                "chamfer": mesh_idw_surf["chamfer"],
                "hausdorff_95": mesh_idw_surf["hausdorff_95"],
                "f1_1v": mesh_idw_surf["f1_1v"], "f1_2v": mesh_idw_surf["f1_2v"],
            }
            _log_scalars(combo_writer, idw_vol_metrics, idw_mesh_metrics)
            combo_writer.add_scalar("sigma/spatial",    sigma_s,     global_step=0)
            combo_writer.add_scalar("sigma/value",      sigma_v,     global_step=0)
            combo_writer.add_scalar("sigma/scale_equiv", scale_equiv, global_step=0)

            # IDW slices from voxelized volume
            idw_slices, _ = _extract_slices(idw_vol, gt_volume, res=vol_res)

            visualize_slices(
                d_slices, idw_slices, cd_slices,
                gt_slices=gt_slices_list,
                vmax=vmax,
                writer_fn_interleaved=partial(
                    combo_writer.add_figure, "slices_interleaved", global_step=0
                ),
                writer_fn_sobel=partial(
                    combo_writer.add_figure, "slices_sobel", global_step=0
                ),
            )

            # IDW mesh at representative threshold
            idw_verts, idw_faces = marching_tets(
                _pts, _mu, _tets, threshold=MESH_THRESHOLD, density_fn=_mesh_idw_fn
            )
            _log_mesh_to_tb(combo_writer, "mesh/idw", idw_verts, idw_faces)

            combo_writer.close()

            # ── CSV row ─────────────────────────────────────────────────────
            row = {
                "sigma_s": sigma_s,
                "sigma_v": sigma_v,
                "scale_equiv": round(scale_equiv, 3),
                **raw_vol_row,
                **mesh_raw_row,
                "vol_idw_psnr":         idw_psnr,
                "vol_idw_ssim":         idw_ssim,
                "vol_idw_ssim3d":       idw_ssim3d,
                "vol_idw_dice":         idw_dice,
                "vol_idw_chamfer":      idw_surf["chamfer"],
                "vol_idw_hausdorff_95": idw_surf["hausdorff_95"],
                "vol_idw_f1_1v":        idw_surf["f1_1v"],
                "vol_idw_f1_2v":        idw_surf["f1_2v"],
                "vol_idw_sobel_psnr":   idw_sobel_psnr,
                "vol_idw_sobel_ssim":   idw_sobel_ssim,
                "mesh_idw_chamfer":      mesh_idw_surf["chamfer"],
                "mesh_idw_hausdorff_95": mesh_idw_surf["hausdorff_95"],
                "mesh_idw_f1_1v":        mesh_idw_surf["f1_1v"],
                "mesh_idw_f1_2v":        mesh_idw_surf["f1_2v"],
            }
            rows.append(row)

    # ── CSV ─────────────────────────────────────────────────────────────────
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows → {out_csv}")
        print(f"TB logs → {tb_base}/")
    else:
        print("[WARN] No rows produced — check σ grid args.")


if __name__ == "__main__":
    main()
