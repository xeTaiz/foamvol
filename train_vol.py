"""Volume-supervised Voronoi foam training.

Trains the adaptive Voronoi grid directly against a ground-truth 3D volume
(vol_gt.npy) using a per-sample L1/L2 loss over randomly sampled query points.
No ray tracing or projection geometry required.

Density gradient: trivial (weighted sum of neighbor densities).
Position gradient: from IDW spatial weights — cells where neighbor densities
differ from GT are pushed toward consistent neighborhoods.
"""

import os
import time
import uuid
import yaml
import gc
import math
from functools import partial
import numpy as np
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.measure import marching_cubes
from scipy.spatial import KDTree

from configs import *
from radfoam_model.scene import CTScene
from radfoam_model.mesh import surface_metrics_vs_gt_volume
from radfoam_model.scene import idw_query
from radfoam_model.utils import (
    compute_volume_psnr,
    gauss_conv3d_separable as _gauss_conv3d_separable,
)
from visualize_volume import visualize
from vis_foam import (load_density_field, field_from_model, query_density,
                      sample_idw, sample_idw_diagnostic,
                      visualize_idw_diagnostics,
                      make_slice_coords, compute_cell_density_slice,
                      compute_voronoi_edges, visualize_cell_heatmap,
                      visualize_grad_weights,
                      visualize_slices, load_gt_volume, load_r2_volume,
                      sample_gt_slice, render_volume_drr,
                      voxelize_volumes, log_density_histogram,
                      log_volume_slices, visualize_cells_vs_gradient)
import radfoam
from voxel_grid import ALIGN_CORNERS


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)




@torch.no_grad()
def compute_volume_ssim(pred, gt, window_size=11):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    pred, gt = pred.float(), gt.float()

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
    if pred.is_cuda:
        kernel = kernel.cuda()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    axis_ssims = []
    for axis in range(3):
        n_slices = pred.shape[axis]
        ssim_sum = 0.0
        count = 0
        for i in range(n_slices):
            if axis == 0:
                s_pred, s_gt = pred[i, :, :], gt[i, :, :]
            elif axis == 1:
                s_pred, s_gt = pred[:, i, :], gt[:, i, :]
            else:
                s_pred, s_gt = pred[:, :, i], gt[:, :, i]
            if s_gt.max() <= 0:
                continue
            img1 = s_pred.unsqueeze(0).unsqueeze(0)
            img2 = s_gt.unsqueeze(0).unsqueeze(0)
            mu1 = F.conv2d(img1, kernel, padding=pad)
            mu2 = F.conv2d(img2, kernel, padding=pad)
            sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=pad) - mu1 ** 2
            sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=pad) - mu2 ** 2
            sigma12 = F.conv2d(img1 * img2, kernel, padding=pad) - mu1 * mu2
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
            )
            ssim_sum += ssim_map.mean().item()
            count += 1
        axis_ssims.append(ssim_sum / count if count > 0 else 0.0)
    return float(np.mean(axis_ssims)), axis_ssims


@torch.no_grad()
def compute_volume_ssim_3d(pred, gt):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    pred, gt = pred.float(), gt.float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    v = pred.unsqueeze(0).unsqueeze(0)
    w = gt.unsqueeze(0).unsqueeze(0)
    k3 = torch.ones(1, 1, 3, 3, 3, device=pred.device) / 27
    mu1 = F.conv3d(v, k3, padding=1)
    mu2 = F.conv3d(w, k3, padding=1)
    s1 = F.conv3d(v * v, k3, padding=1) - mu1 * mu1
    s2 = F.conv3d(w * w, k3, padding=1) - mu2 * mu2
    s12 = F.conv3d(v * w, k3, padding=1) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    )
    return ssim.mean().item()


def sobel_filter_3d(vol):
    v = vol.unsqueeze(0).unsqueeze(0)
    v = F.pad(v, (1, 1, 1, 1, 1, 1), mode="replicate")
    smooth = torch.tensor([1, 2, 1], dtype=torch.float32, device=vol.device)
    diff = torch.tensor([-1, 0, 1], dtype=torch.float32, device=vol.device)
    def outer3(a, b, c):
        return (a[:, None, None] * b[None, :, None] * c[None, None, :]).reshape(1, 1, 3, 3, 3)
    kx = outer3(diff, smooth, smooth)
    ky = outer3(smooth, diff, smooth)
    kz = outer3(smooth, smooth, diff)
    gx = F.conv3d(v, kx)
    gy = F.conv3d(v, ky)
    gz = F.conv3d(v, kz)
    return (gx ** 2 + gy ** 2 + gz ** 2).sqrt().squeeze()


@torch.no_grad()
def compute_dice(pred, gt, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    dices = {}
    for t in thresholds:
        p, g = pred > t, gt > t
        inter = (p & g).sum()
        dices[t] = 2 * inter / (p.sum() + g.sum() + 1e-8)
    return float(np.mean(list(dices.values()))), dices


@torch.no_grad()
def _surface_metrics_at_level(pred, gt, level, f_thresholds=(1.0, 2.0)):
    try:
        verts_p, _, _, _ = marching_cubes(pred, level=level)
        verts_g, _, _, _ = marching_cubes(gt, level=level)
    except (ValueError, RuntimeError):
        return None
    if len(verts_p) < 3 or len(verts_g) < 3:
        return None
    tree_p = KDTree(verts_p)
    tree_g = KDTree(verts_g)
    d_pred_to_gt, _ = tree_g.query(verts_p)
    d_gt_to_pred, _ = tree_p.query(verts_g)
    chamfer = 0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean())
    hausdorff_max = float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))
    hausdorff_95 = float(max(np.percentile(d_pred_to_gt, 95),
                             np.percentile(d_gt_to_pred, 95)))
    result = {"chamfer": float(chamfer), "hausdorff": hausdorff_max,
              "hausdorff_95": hausdorff_95}
    for d in f_thresholds:
        prec = (d_pred_to_gt <= d).mean()
        rec = (d_gt_to_pred <= d).mean()
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        result[f"f1_{d:.0f}v"] = float(f1)
    return result


@torch.no_grad()
def compute_surface_metrics(pred, gt,
                            thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            f_thresholds=(1.0, 2.0)):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    per_level = []
    for t in thresholds:
        m = _surface_metrics_at_level(pred, gt, t, f_thresholds)
        if m is not None:
            per_level.append(m)
    if not per_level:
        return {"chamfer": float("inf"), "hausdorff": float("inf"),
                "hausdorff_95": float("inf"),
                **{f"f1_{d:.0f}v": 0.0 for d in f_thresholds}}
    keys = per_level[0].keys()
    return {k: float(np.mean([m[k] for m in per_level])) for k in keys}


def log_diagnostics(model, writer, step):
    with torch.no_grad():
        _, cell_radius = radfoam.farthest_neighbor(
            model.primal_points,
            model.point_adjacency,
            model.point_adjacency_offsets,
        )
        writer.add_histogram("diagnostics/cell_radius", cell_radius, step)


def _resolve_global_sigma(model, pipeline_args):
    if pipeline_args.interp_sigma_abs > 0:
        return pipeline_args.interp_sigma_abs
    _, cr = radfoam.farthest_neighbor(
        model.primal_points, model.point_adjacency, model.point_adjacency_offsets,
    )
    return pipeline_args.interp_sigma_scale * cr.median().item()


@torch.no_grad()
def _log_grad_distribution(writer, step, point_error):
    if point_error is None:
        return
    pe = point_error.squeeze().float()
    pe_np = pe.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(pe_np, bins=50, log=True)
    ax.set_xlabel("||position gradient||")
    ax.set_ylabel("count")
    writer.add_figure("densify/grad_distribution", fig, step)
    plt.close(fig)


def train_vol(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)

    if not pipeline_args.debug:
        if len(pipeline_args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{dataset_args.scene}@{unique_str}"
        else:
            experiment_name = pipeline_args.experiment_name
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
        yaml.add_representer(list, represent_list_inline)
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Load ground-truth volume
    gt_volume_path_override = getattr(pipeline_args, 'gt_volume_path', '')
    if gt_volume_path_override:
        gt_volume = np.load(gt_volume_path_override).astype(np.float32)
    else:
        gt_volume = load_gt_volume(dataset_args.data_path, dataset_args.dataset,
                                   dataset_args=dataset_args)
    if gt_volume is None:
        raise RuntimeError(
            f"Could not load vol_gt from {dataset_args.data_path}. "
            "Provide --gt_volume_path or ensure vol_gt.npy exists in the dataset directory."
        )
    print(f"Loaded GT volume: shape={gt_volume.shape}, "
          f"min={gt_volume.min():.4f}, max={gt_volume.max():.4f}, "
          f"mean={gt_volume.mean():.6f}")

    # vol_gt_5d: [1, 1, X, Y, Z] — used with F.grid_sample (expects D,H,W = X,Y,Z here)
    vol_gt_5d = torch.from_numpy(gt_volume).float().to(device).unsqueeze(0).unsqueeze(0)

    r2_volume = load_r2_volume(dataset_args.data_path)
    if r2_volume is not None:
        print(f"Loaded R2 volume: shape={r2_volume.shape}")

    extent = pipeline_args.volume_sampling_extent
    n_query = pipeline_args.n_query_per_step

    # Auto warm-start: set init_density so softplus output ≈ gt_volume mean
    if model_args.init_density == 0.0:
        gt_mean = float(gt_volume.mean())
        if gt_mean > 1e-6:
            raw_warmstart = math.log(max(1e-8, 10.0 * gt_mean / model_args.activation_scale)) / 10.0
            model_args.init_density = raw_warmstart
            print(f"Auto-warm-start init_density={raw_warmstart:.4f} "
                  f"(target mean={gt_mean:.6f})")

    # Create model
    model = CTScene(args=model_args, device=device)

    # Optional FDK init
    init_volume_path = getattr(model_args, 'init_volume_path', '')
    if init_volume_path:
        model.initialize_from_volume(
            init_volume_path,
            ref_resolution=getattr(optimizer_args, 'ref_volume_resolution', 64),
            ref_blur_sigma=getattr(optimizer_args, 'ref_volume_blur_sigma', 2.0),
        )

    # Optional ref-volume regularization loss
    ref_volume_path = getattr(optimizer_args, 'ref_volume_path', '')
    if ref_volume_path:
        model.load_reference_volume(
            ref_volume_path,
            resolution=getattr(optimizer_args, 'ref_volume_resolution', 64),
            blur_sigma=getattr(optimizer_args, 'ref_volume_blur_sigma', 2.0),
            edge_mask=getattr(optimizer_args, 'ref_volume_edge_mask', False),
            edge_alpha=getattr(optimizer_args, 'ref_volume_edge_alpha', 10.0),
        )
        model._ref_vol_supersample = getattr(optimizer_args, 'ref_volume_supersample', 1)

    # Optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    # IDW params — set on model so _idw_query_at and _idw_voxelize both use them.
    # For volume training we default to per_cell_sigma=True and disable bilateral (large sigma_v).
    _use_adaptive = pipeline_args.per_cell_sigma or pipeline_args.per_neighbor_sigma
    _sigma_init = (pipeline_args.interp_sigma_scale if _use_adaptive
                   else _resolve_global_sigma(model, pipeline_args))
    model.set_interpolation_mode(
        False,  # ray-tracing interpolation mode off (unused in volume training)
        sigma=_sigma_init,
        sigma_v=pipeline_args.interp_sigma_v,
        per_cell_sigma=pipeline_args.per_cell_sigma,
        per_neighbor_sigma=pipeline_args.per_neighbor_sigma,
    )

    def _gt_sample(query):
        """Trilinear GT density at query points [B, 3] in scene coordinates."""
        grid = (query / extent).flip(-1)[None, None, None]  # (z,y,x) normalized for grid_sample
        return F.grid_sample(
            vol_gt_5d, grid, mode='bilinear',
            align_corners=ALIGN_CORNERS, padding_mode='zeros',
        ).reshape(-1)

    def _voxel_step(query):
        """One forward step: IDW predict + GT lookup → scalar loss."""
        mu_pred = model._idw_query_at(query)
        mu_gt = _gt_sample(query).detach()
        if pipeline_args.volume_loss_type == 'l2':
            return (mu_pred - mu_gt).pow(2).mean(), mu_pred, mu_gt
        return (mu_pred - mu_gt).abs().mean(), mu_pred, mu_gt

    def log_basic(step, loss_val=None, tv_loss_val=None, tv_scale_val=None):
        num_points = model.primal_points.shape[0]
        if not pipeline_args.debug:
            writer.add_scalar("train/num_points", num_points, step)
            if loss_val is not None:
                writer.add_scalar("train/loss", loss_val, step)
            if tv_loss_val is not None and optimizer_args.tv_weight > 0:
                writer.add_scalar("train/tv_loss", tv_loss_val, step)
            if hasattr(model, '_triangulation_retries'):
                writer.add_scalar("diagnostics/triangulation_retries",
                                  model._triangulation_retries, step)
            if (pipeline_args.bf_start >= 0
                    and pipeline_args.bf_start <= step < pipeline_args.bf_until):
                t = (step - pipeline_args.bf_start) / max(
                    1, pipeline_args.bf_until - pipeline_args.bf_start - 1)
                writer.add_scalar("train/bf_sigma",
                                  pipeline_args.bf_sigma_init + t * (
                                      pipeline_args.bf_sigma_final - pipeline_args.bf_sigma_init), step)
                writer.add_scalar("train/bf_sigma_v",
                                  pipeline_args.bf_sigma_v_final + 0.5 * (
                                      pipeline_args.bf_sigma_v_init - pipeline_args.bf_sigma_v_final) * (
                                      1 + math.cos(math.pi * t)), step)
            writer.add_scalar("lr/points_lr", model.xyz_scheduler_args(step), step)
            writer.add_scalar("lr/density_lr", model.den_scheduler_args(step), step)
        print(f"Step {step}: points={num_points}"
              + (f", loss={loss_val:.5f}" if loss_val is not None else ""))

    def log_diag(step):
        if pipeline_args.debug:
            return None
        log_density_histogram(model, writer, step)
        log_diagnostics(model, writer, step)
        cell_entropy = model.compute_neighbor_entropy(n_bins=pipeline_args.entropy_bins)
        writer.add_histogram("diagnostics/cell_entropy", cell_entropy, step)
        writer.add_scalar("diagnostics/entropy_mean", cell_entropy.mean().item(), step)

        with torch.no_grad():
            field = field_from_model(model)
            _, cell_radius = radfoam.farthest_neighbor(
                model.primal_points,
                model.point_adjacency,
                model.point_adjacency_offsets,
            )
            sigma = (pipeline_args.interp_sigma_abs if pipeline_args.interp_sigma_abs > 0
                     else pipeline_args.interp_sigma_scale * cell_radius.median().item())
            sigma_v = pipeline_args.interp_sigma_v

            axes = [0, 1, 2]
            slice_coords = [-0.2, 0.0, 0.2]
            d_slices, idw_slices, cd_slices = [], [], []
            gt_slices, ve_slices = [], []
            for a in axes:
                for c in slice_coords:
                    coords_2d = make_slice_coords(a, c, 256, 1.0)
                    d_slices.append(query_density(field, coords_2d))
                    idw_slices.append(sample_idw(field, coords_2d, sigma=sigma, sigma_v=sigma_v))
                    cd_slices.append(compute_cell_density_slice(field["points"], a, c, 64, 1.0))
                    gt_slices.append(sample_gt_slice(gt_volume, a, c, 256, 1.0))
                    ve_slices.append(compute_voronoi_edges(field, a, c, 256, 1.0))

            log_fig_il = partial(writer.add_figure, f"slices_interleaved/{experiment_name}",
                                 global_step=step)
            log_fig_sobel = partial(writer.add_figure, f"slices_sobel/{experiment_name}",
                                    global_step=step)
            metrics = visualize_slices(
                d_slices, idw_slices, cd_slices,
                gt_slices=gt_slices,
                writer_fn_interleaved=log_fig_il,
                writer_fn_sobel=log_fig_sobel,
                voronoi_edges=ve_slices,
            )
            if metrics is not None:
                for key, val in metrics.items():
                    parts = key.split('_')
                    tag = (f"slice_{parts[1]}/{parts[0]}" if len(parts) == 2
                           else f"slice_{parts[-1]}/{'_'.join(parts[:-1])}")
                    writer.add_scalar(tag, val, step)

            log_fig_hm = partial(writer.add_figure, f"cell_heatmap/{experiment_name}",
                                 global_step=step)
            visualize_cell_heatmap(cd_slices, writer_fn=log_fig_hm)

            if gt_volume is not None:
                stats = visualize_cells_vs_gradient(
                    model.primal_points.detach(), gt_volume,
                    writer_fn=partial(writer.add_figure,
                                      f"cells_vs_gt_grad/{experiment_name}",
                                      global_step=step),
                )
                writer.add_scalar("diagnostics/cells_vs_grad_spearman_rho",
                                  stats["spearman_rho"], step)
        return metrics

    def train_loop():
        print("Training (volume-supervised)")

        log_interval = max(1, pipeline_args.iterations * pipeline_args.log_percent // 100)
        diag_interval = max(1, pipeline_args.iterations * pipeline_args.diag_percent // 100)

        torch.cuda.synchronize()

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        if pipeline_args.densify_grad_thresh > 0:
            _K = math.ceil(
                math.log(model_args.final_points / model_args.init_points)
                / math.log(pipeline_args.densify_factor)
            )
            _fixed_interval = max(
                1, (pipeline_args.densify_until - pipeline_args.densify_from) // _K,
            )

        if not pipeline_args.debug:
            log_basic(0)
            if pipeline_args.diag:
                log_diag(0)
            log_volume_slices(model, writer, gt_volume, 0, experiment_name)

        _loss_cpu = None
        tv_loss = torch.tensor(0.0, device=device)
        with tqdm.trange(pipeline_args.iterations) as progress:
            for i in progress:
                # Variance sigma schedule
                _densify_range = max(1, pipeline_args.densify_until - pipeline_args.densify_from)
                _var_sched_t = max(0.0, min(1.0,
                    (i - pipeline_args.densify_from) / _densify_range))
                var_sigma_v = (optimizer_args.var_sigma_v_init * (1.0 - _var_sched_t)
                               + optimizer_args.var_sigma_v_final * _var_sched_t)

                # Sample random query points
                query = (torch.rand(n_query, 3, device=device) * 2 - 1) * extent

                # Forward: IDW predict + GT lookup
                loss, _, _ = _voxel_step(query)

                # TV regularization
                if optimizer_args.tv_weight > 0 and i >= optimizer_args.tv_start:
                    if optimizer_args.tv_border:
                        tv_loss = model.tv_border_regularization(
                            epsilon=optimizer_args.tv_epsilon,
                            area_weighted=optimizer_args.tv_area_weighted,
                            on_raw=optimizer_args.tv_on_raw,
                        )
                    else:
                        tv_loss = model.tv_regularization(
                            epsilon=optimizer_args.tv_epsilon,
                            area_weighted=optimizer_args.tv_area_weighted,
                            on_raw=optimizer_args.tv_on_raw,
                        )
                    tv_scale = 1.0
                    if optimizer_args.tv_anneal:
                        anneal_range = optimizer_args.freeze_points - optimizer_args.tv_start
                        if anneal_range > 0:
                            tv_scale = max(0.0, 1.0 - (i - optimizer_args.tv_start) / anneal_range)
                    loss = loss + optimizer_args.tv_weight * tv_scale * tv_loss

                # Variance weight schedule
                _decay_range = max(1, pipeline_args.interpolation_start - pipeline_args.densify_until)
                _w_t = max(0.0, min(1.0, (i - pipeline_args.densify_until) / _decay_range))
                vvar_w_final = (optimizer_args.voxel_var_weight
                                if optimizer_args.voxel_var_weight_final < 0
                                else optimizer_args.voxel_var_weight_final)
                nvar_w_final = (optimizer_args.neighbor_var_weight
                                if optimizer_args.neighbor_var_weight_final < 0
                                else optimizer_args.neighbor_var_weight_final)
                vvar_w = optimizer_args.voxel_var_weight * (1.0 - _w_t) + vvar_w_final * _w_t
                nvar_w = optimizer_args.neighbor_var_weight * (1.0 - _w_t) + nvar_w_final * _w_t

                if vvar_w > 0 and i >= optimizer_args.voxel_var_start:
                    voxel_var_loss = model.voxel_variance_regularization(
                        resolution=optimizer_args.voxel_var_resolution,
                        sigma_v=var_sigma_v,
                        supersample=getattr(optimizer_args, 'voxel_var_supersample', 1),
                    )
                    loss = loss + vvar_w * voxel_var_loss

                if nvar_w > 0 and i >= optimizer_args.neighbor_var_start:
                    neighbor_var_loss = model.neighbor_variance_regularization(
                        sigma_v=var_sigma_v,
                        hops=optimizer_args.neighbor_var_hops,
                        reg_type=getattr(optimizer_args, 'neighbor_reg_type', 'bilateral_var'),
                        huber_delta=getattr(optimizer_args, 'neighbor_huber_delta', 0.1),
                    )
                    loss = loss + nvar_w * neighbor_var_loss

                ba_w = getattr(optimizer_args, 'boundary_align_weight', 0.0)
                boundary_align_loss = None
                if ba_w > 0:
                    ba_start = getattr(optimizer_args, 'boundary_align_start', -1)
                    ba_until = getattr(optimizer_args, 'boundary_align_until', -1)
                    if ba_start < 0:
                        ba_start = pipeline_args.densify_from
                    if ba_until < 0:
                        ba_until = optimizer_args.freeze_points
                    if ba_start <= i < ba_until:
                        boundary_align_loss = model.boundary_alignment_regularization(
                            sigma_v=var_sigma_v,
                        )
                        loss = loss + ba_w * boundary_align_loss

                rv_w_cfg = getattr(optimizer_args, 'ref_volume_weight', 0.0)
                rv_start = getattr(optimizer_args, 'ref_volume_start', 0)
                rv_until = getattr(optimizer_args, 'ref_volume_until', -1)
                if (rv_w_cfg > 0 and hasattr(model, '_ref_volume')
                        and i >= rv_start and (rv_until < 0 or i < rv_until)):
                    rv_w_final_cfg = getattr(optimizer_args, 'ref_volume_weight_final', -1.0)
                    rv_w = (rv_w_cfg * (1.0 - _w_t) + rv_w_final_cfg * _w_t
                            if rv_w_final_cfg >= 0 else rv_w_cfg)
                    ref_vol_loss = model.reference_volume_loss(
                        resolution=getattr(optimizer_args, 'ref_volume_resolution', 64),
                    )
                    loss = loss + rv_w * ref_vol_loss

                model.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if getattr(optimizer_args, 'grad_smooth_hops', 0) > 0:
                    model.smooth_density_grad(hops=optimizer_args.grad_smooth_hops)

                if optimizer_args.density_grad_clip > 0 and model.density.grad is not None:
                    model.density.grad.clamp_(-optimizer_args.density_grad_clip,
                                              optimizer_args.density_grad_clip)

                model.apply_frozen_mask()
                model.optimizer.step()
                model.update_starvation_count()

                if i < pipeline_args.densify_until:
                    model.density.data.clamp_(min=-1.0)

                # Bilateral filter
                if (pipeline_args.bf_start >= 0
                        and pipeline_args.bf_start <= i < pipeline_args.bf_until
                        and i % pipeline_args.bf_period == 0):
                    t = (i - pipeline_args.bf_start) / max(
                        1, pipeline_args.bf_until - pipeline_args.bf_start - 1)
                    bf_sigma = (pipeline_args.bf_sigma_init
                                + t * (pipeline_args.bf_sigma_final - pipeline_args.bf_sigma_init))
                    bf_sigma_v = (pipeline_args.bf_sigma_v_final
                                  + 0.5 * (pipeline_args.bf_sigma_v_init - pipeline_args.bf_sigma_v_final)
                                  * (1 + math.cos(math.pi * t)))
                    model.apply_bilateral_filter(bf_sigma, bf_sigma_v)

                model.update_learning_rate(i)

                if _loss_cpu is not None:
                    progress.set_postfix(loss=f"{_loss_cpu.item():.5f}")
                _loss_cpu = loss.detach().to("cpu", non_blocking=True)

                if i % log_interval == log_interval - 1 and not pipeline_args.debug:
                    tv_loss_val = (tv_loss.item()
                                   if optimizer_args.tv_weight > 0 and i >= optimizer_args.tv_start
                                   else None)
                    if optimizer_args.voxel_var_weight > 0 and i >= optimizer_args.voxel_var_start:
                        writer.add_scalar("train/voxel_var_loss", voxel_var_loss.item(), i)
                        writer.add_scalar("train/voxel_var_weight", vvar_w, i)
                    if optimizer_args.neighbor_var_weight > 0 and i >= optimizer_args.neighbor_var_start:
                        writer.add_scalar("train/neighbor_var_loss", neighbor_var_loss.item(), i)
                        writer.add_scalar("train/neighbor_var_weight", nvar_w, i)
                    log_basic(i, loss_val=loss.item(), tv_loss_val=tv_loss_val)

                if pipeline_args.diag and i % diag_interval == diag_interval - 1 and not pipeline_args.debug:
                    if i % log_interval != log_interval - 1:
                        log_basic(i)
                    log_diag(i)
                    log_volume_slices(model, writer, gt_volume, i, experiment_name)

                # Triangulation update
                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0
                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                # Densification
                if (iters_since_densification == next_densification_after
                        and model.primal_points.shape[0] < 0.9 * model.num_final_points):
                    point_error, point_contribution = model.collect_error_map_volume(
                        vol_gt_5d,
                        n_query=pipeline_args.n_query_error_map,
                        batch_size=pipeline_args.error_map_batch_size,
                        extent=extent,
                    )

                    if pipeline_args.redundancy_cap_init > 0 or pipeline_args.redundancy_cap_final > 0:
                        _redundancy_cap = (
                            pipeline_args.redundancy_cap_init * (1.0 - _var_sched_t)
                            + pipeline_args.redundancy_cap_final * _var_sched_t
                        )
                    else:
                        _redundancy_cap = pipeline_args.redundancy_cap

                    densify_stats = model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                        gradient_fraction=pipeline_args.gradient_fraction,
                        idw_fraction=pipeline_args.idw_fraction,
                        entropy_fraction=pipeline_args.entropy_fraction,
                        entropy_bins=pipeline_args.entropy_bins,
                        redundancy_threshold=pipeline_args.redundancy_threshold,
                        redundancy_cap=_redundancy_cap,
                        sigma_scale=pipeline_args.interp_sigma_scale,
                        sigma_v=pipeline_args.interp_sigma_v,
                        variance_pruning=pipeline_args.prune_variance_criterion,
                        prune_hops=pipeline_args.prune_hops,
                        ref_guided_pruning=getattr(pipeline_args, 'ref_guided_pruning', False),
                        ref_guided_densify=getattr(pipeline_args, 'ref_guided_densify', False),
                        ref_guided_eps=getattr(pipeline_args, 'ref_guided_eps', 0.05),
                        grad_thresh=pipeline_args.densify_grad_thresh,
                        var_thresh=pipeline_args.densify_var_thresh,
                        var_power=pipeline_args.densify_var_power,
                        var_hops=pipeline_args.densify_var_hops,
                    )

                    if not pipeline_args.debug and densify_stats is not None:
                        for key, val in densify_stats.items():
                            writer.add_scalar(f"densify/{key}", val, i)
                        _log_grad_distribution(writer, i, point_error)
                        writer.add_histogram("densify/contribution",
                                             point_contribution.squeeze().cpu(), i)

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

                    iters_since_densification = 0
                    if pipeline_args.densify_grad_thresh > 0:
                        next_densification_after = _fixed_interval
                    else:
                        next_densification_after = int(
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (pipeline_args.densify_until - pipeline_args.densify_from)
                            / (model.num_final_points - model.num_init_points)
                        )
                        next_densification_after = max(next_densification_after, 100)

                # End-of-densify hook
                if i == pipeline_args.densify_until:
                    model.update_triangulation(incremental=False)
                    n_standalone_pruned = model.prune_only_volume(
                        vol_gt_5d,
                        n_query=pipeline_args.n_query_error_map,
                        batch_size=pipeline_args.error_map_batch_size,
                        extent=extent,
                    )
                    if not pipeline_args.debug:
                        writer.add_scalar("densify/standalone_pruned", n_standalone_pruned, i)
                        writer.add_scalar("densify/points_after",
                                          model.primal_points.shape[0], i)

                    # Refresh IDW sigma after densification stabilizes
                    use_adaptive = pipeline_args.per_cell_sigma or pipeline_args.per_neighbor_sigma
                    sigma = (pipeline_args.interp_sigma_scale if use_adaptive
                             else _resolve_global_sigma(model, pipeline_args))
                    model.set_interpolation_mode(
                        False, sigma=sigma, sigma_v=pipeline_args.interp_sigma_v,
                        per_cell_sigma=pipeline_args.per_cell_sigma,
                        per_neighbor_sigma=pipeline_args.per_neighbor_sigma,
                    )

                if (optimizer_args.gradient_start >= 0
                        and i == optimizer_args.gradient_start):
                    model.initialize_gradients(optimizer_args)

                if (optimizer_args.gaussian_start >= 0
                        and i == optimizer_args.gaussian_start):
                    model.initialize_gaussian(optimizer_args)
                    if optimizer_args.freeze_base_at_gaussian:
                        model.density.requires_grad_(False)

                if (optimizer_args.joint_finetune_start >= 0
                        and i == optimizer_args.joint_finetune_start
                        and getattr(model, '_gaussian_active', False)):
                    model.density.requires_grad_(True)

                frozen_unfreeze = getattr(optimizer_args, 'frozen_unfreeze_step', -1)
                if frozen_unfreeze >= 0 and i == frozen_unfreeze:
                    model.unfreeze_all()

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

        if not pipeline_args.debug:
            model.save_ply(f"{out_dir}/scene.ply")
            model.save_pt(f"{out_dir}/model.pt")

    _train_start_time = time.time()
    train_loop()
    iters = pipeline_args.iterations
    _train_duration = time.time() - _train_start_time
    print(f"Training time: {_train_duration:.1f}s ({_train_duration / 60:.2f}min)")

    if not pipeline_args.debug:
        writer.add_scalar("train/training_time_seconds", _train_duration, iters)
        log_basic(iters)
        log_diag(iters)
        log_volume_slices(model, writer, gt_volume, iters, experiment_name)

        point_error_final, _ = model.collect_error_map_volume(
            vol_gt_5d,
            n_query=pipeline_args.n_query_error_map,
            batch_size=pipeline_args.error_map_batch_size,
            extent=extent,
        )
        _log_grad_distribution(writer, iters, point_error_final)

        with open(f"{out_dir}/metrics.txt", "w") as f:
            f.write(f"Train Time: {_train_duration:.1f}s\n")
            f.write(f"Num Cells: {model.primal_points.shape[0]}\n")

        model_path = f"{out_dir}/model.pt"

        with torch.no_grad():
            _, cell_radius = radfoam.farthest_neighbor(
                model.primal_points,
                model.point_adjacency,
                model.point_adjacency_offsets,
            )
            interp_sigma = (pipeline_args.interp_sigma_abs if pipeline_args.interp_sigma_abs > 0
                            else pipeline_args.interp_sigma_scale * cell_radius.median().item())
            interp_sigma_v = pipeline_args.interp_sigma_v

        field = load_density_field(model_path)

        # 3D volume metrics
        vol_res = gt_volume.shape[0]
        print(f"Voxelizing at {vol_res}³ for 3D volume metrics...")
        raw_vol, idw_vol = voxelize_volumes(
            field, resolution=vol_res, extent=1.0,
            sigma=interp_sigma, sigma_v=interp_sigma_v,
        )

        vol_gt_t = torch.from_numpy(gt_volume).float().cuda()
        raw_vol_t = torch.from_numpy(raw_vol).float().cuda()
        idw_vol_t = torch.from_numpy(idw_vol).float().cuda()

        raw_psnr_3d = compute_volume_psnr(raw_vol_t, vol_gt_t)
        raw_ssim_3d, raw_ssim_ax = compute_volume_ssim(raw_vol_t, vol_gt_t)
        idw_psnr_3d = compute_volume_psnr(idw_vol_t, vol_gt_t)
        idw_ssim_3d, idw_ssim_ax = compute_volume_ssim(idw_vol_t, vol_gt_t)

        print(f"Vol Raw  PSNR: {raw_psnr_3d:.4f}, SSIM: {raw_ssim_3d:.6f}")
        print(f"Vol IDW  PSNR: {idw_psnr_3d:.4f}, SSIM: {idw_ssim_3d:.6f}")

        writer.add_scalar("test/vol_raw_psnr", raw_psnr_3d, iters)
        writer.add_scalar("test/vol_raw_ssim", raw_ssim_3d, iters)
        writer.add_scalar("test/vol_idw_psnr", idw_psnr_3d, iters)
        writer.add_scalar("test/vol_idw_ssim", idw_ssim_3d, iters)
        for ax_i, ax_name in enumerate(["x", "y", "z"]):
            writer.add_scalar(f"test/vol_raw_ssim_{ax_name}", raw_ssim_ax[ax_i], iters)
            writer.add_scalar(f"test/vol_idw_ssim_{ax_name}", idw_ssim_ax[ax_i], iters)

        raw_ssim3d = compute_volume_ssim_3d(raw_vol_t, vol_gt_t)
        idw_ssim3d = compute_volume_ssim_3d(idw_vol_t, vol_gt_t)
        writer.add_scalar("test/vol_raw_ssim3d", raw_ssim3d, iters)
        writer.add_scalar("test/vol_idw_ssim3d", idw_ssim3d, iters)

        gt_sobel_vol = sobel_filter_3d(vol_gt_t)
        raw_sobel_vol = sobel_filter_3d(raw_vol_t)
        idw_sobel_vol = sobel_filter_3d(idw_vol_t)
        sobel_raw_psnr_3d = compute_volume_psnr(raw_sobel_vol, gt_sobel_vol)
        sobel_idw_psnr_3d = compute_volume_psnr(idw_sobel_vol, gt_sobel_vol)
        sobel_raw_ssim_3d, _ = compute_volume_ssim(raw_sobel_vol, gt_sobel_vol)
        sobel_idw_ssim_3d, _ = compute_volume_ssim(idw_sobel_vol, gt_sobel_vol)

        writer.add_scalar("test/vol_raw_sobel_psnr", sobel_raw_psnr_3d, iters)
        writer.add_scalar("test/vol_idw_sobel_psnr", sobel_idw_psnr_3d, iters)

        raw_dice, _ = compute_dice(raw_vol_t, vol_gt_t)
        idw_dice, _ = compute_dice(idw_vol_t, vol_gt_t)
        raw_surf = compute_surface_metrics(raw_vol_t, vol_gt_t)
        idw_surf = compute_surface_metrics(idw_vol_t, vol_gt_t)

        print(f"Vol Raw  Dice: {raw_dice:.6f}, CD: {raw_surf['chamfer']:.4f}v")
        print(f"Vol IDW  Dice: {idw_dice:.6f}, CD: {idw_surf['chamfer']:.4f}v")

        writer.add_scalar("test/vol_raw_dice", raw_dice, iters)
        writer.add_scalar("test/vol_idw_dice", idw_dice, iters)
        writer.add_scalar("test/vol_raw_chamfer", raw_surf["chamfer"], iters)
        writer.add_scalar("test/vol_idw_chamfer", idw_surf["chamfer"], iters)
        writer.add_scalar("test/vol_raw_hausdorff_95", raw_surf["hausdorff_95"], iters)
        writer.add_scalar("test/vol_idw_hausdorff_95", idw_surf["hausdorff_95"], iters)
        writer.add_scalar("test/vol_raw_f1_1v", raw_surf["f1_1v"], iters)
        writer.add_scalar("test/vol_idw_f1_1v", idw_surf["f1_1v"], iters)
        writer.add_scalar("test/vol_raw_f1_2v", raw_surf["f1_2v"], iters)
        writer.add_scalar("test/vol_idw_f1_2v", idw_surf["f1_2v"], iters)

        with open(f"{out_dir}/metrics.txt", "a") as f:
            f.write(f"Vol Raw PSNR: {raw_psnr_3d:.4f}\n")
            f.write(f"Vol Raw SSIM: {raw_ssim_3d:.6f}\n")
            f.write(f"Vol IDW PSNR: {idw_psnr_3d:.4f}\n")
            f.write(f"Vol IDW SSIM: {idw_ssim_3d:.6f}\n")
            f.write(f"Vol Raw SSIM3D: {raw_ssim3d:.6f}\n")
            f.write(f"Vol IDW SSIM3D: {idw_ssim3d:.6f}\n")
            f.write(f"Vol Raw Sobel PSNR: {sobel_raw_psnr_3d:.4f}\n")
            f.write(f"Vol IDW Sobel PSNR: {sobel_idw_psnr_3d:.4f}\n")
            f.write(f"Vol Raw Sobel SSIM: {sobel_raw_ssim_3d:.6f}\n")
            f.write(f"Vol IDW Sobel SSIM: {sobel_idw_ssim_3d:.6f}\n")
            f.write(f"Vol Raw Dice: {raw_dice:.6f}\n")
            f.write(f"Vol IDW Dice: {idw_dice:.6f}\n")
            f.write(f"Vol Raw CD: {raw_surf['chamfer']:.4f}\n")
            f.write(f"Vol IDW CD: {idw_surf['chamfer']:.4f}\n")
            f.write(f"Vol Raw Hausdorff 95: {raw_surf['hausdorff_95']:.4f}\n")
            f.write(f"Vol IDW Hausdorff 95: {idw_surf['hausdorff_95']:.4f}\n")
            f.write(f"Vol Raw F1 1v: {raw_surf['f1_1v']:.4f}\n")
            f.write(f"Vol IDW F1 1v: {idw_surf['f1_1v']:.4f}\n")
            for ax_i, ax_name in enumerate(["X", "Y", "Z"]):
                f.write(f"Vol Raw SSIM_{ax_name}: {raw_ssim_ax[ax_i]:.6f}\n")
                f.write(f"Vol IDW SSIM_{ax_name}: {idw_ssim_ax[ax_i]:.6f}\n")

        if r2_volume is not None:
            r2_vol_t = torch.from_numpy(r2_volume).float().cuda()
            r2_psnr_3d = compute_volume_psnr(r2_vol_t, vol_gt_t)
            r2_ssim_3d, _ = compute_volume_ssim(r2_vol_t, vol_gt_t)
            print(f"Vol R2   PSNR: {r2_psnr_3d:.4f}, SSIM: {r2_ssim_3d:.6f}")
            writer.add_scalar("test/vol_r2_psnr", r2_psnr_3d, iters)
            writer.add_scalar("test/vol_r2_ssim", r2_ssim_3d, iters)
            with open(f"{out_dir}/metrics.txt", "a") as f:
                f.write(f"Vol R2 PSNR: {r2_psnr_3d:.4f}\n")
                f.write(f"Vol R2 SSIM: {r2_ssim_3d:.6f}\n")

        if pipeline_args.save_volume:
            np.save(f"{out_dir}/volume_raw.npy", raw_vol)
            np.save(f"{out_dir}/volume_idw.npy", idw_vol)


if __name__ == "__main__":
    parser = configargparse.ArgParser(default_config_files=["configs/default.cfg"])
    parser.add_argument("--config", is_config_file=True, help="config file path")
    pipeline_params = PipelineParams(parser)
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    args = parser.parse_args()

    pipeline_args = pipeline_params.extract(args)
    model_args = model_params.extract(args)
    optimizer_args = optimization_params.extract(args)
    dataset_args = dataset_params.extract(args)

    # Volume-training locked defaults for first experiment.
    # per_cell_sigma=True: adaptive sigma proportional to cell size.
    # interp_sigma_v: pure spatial Gaussian (1e6 ≈ disabled bilateral). Change to e.g. 0.35
    #   once the foam is stable to experiment with bilateral IDW.
    if not pipeline_args.per_cell_sigma:
        pipeline_args.per_cell_sigma = True
    # Only override if user hasn't explicitly raised sigma_v above the ray-training default
    if pipeline_args.interp_sigma_v <= 0.35 + 1e-6:
        pipeline_args.interp_sigma_v = 1e6

    print("Optimizing " + str(args))
    train_vol(args, pipeline_args, model_args, optimizer_args, dataset_args)
