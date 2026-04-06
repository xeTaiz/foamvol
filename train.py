import os
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

from skimage.measure import marching_cubes
from scipy.spatial import KDTree

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import CTScene
from visualize_volume import visualize
from vis_foam import (load_density_field, field_from_model, query_density,
                      sample_idw, sample_idw_diagnostic,
                      visualize_idw_diagnostics,
                      make_slice_coords, compute_cell_density_slice,
                      compute_voronoi_edges, visualize_cell_heatmap,
                      visualize_slices, load_gt_volume, load_r2_volume,
                      sample_gt_slice, render_volume_drr,
                      voxelize_volumes, log_density_histogram)
import radfoam


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def compute_psnr(pred, gt):
    """PSNR between two tensors using the ground-truth data range."""
    mse = ((pred - gt) ** 2).mean()
    if mse == 0:
        return float("inf")
    data_range = gt.max() - gt.min()
    return (10 * torch.log10(data_range ** 2 / mse)).item()


def compute_ssim(pred, gt, window_size=11):
    """SSIM between two (H, W, C) projection images."""
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return 1.0

    C = pred.shape[-1]
    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    gt_4d = gt.permute(2, 0, 1).unsqueeze(0)

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).expand(C, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(pred_4d, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt_4d, kernel, padding=pad, groups=C)

    sigma1_sq = F.conv2d(pred_4d ** 2, kernel, padding=pad, groups=C) - mu1 ** 2
    sigma2_sq = F.conv2d(gt_4d ** 2, kernel, padding=pad, groups=C) - mu2 ** 2
    sigma12 = F.conv2d(pred_4d * gt_4d, kernel, padding=pad, groups=C) - mu1 * mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


@torch.no_grad()
def compute_volume_psnr(pred, gt):
    """3D PSNR matching R2-Gaussian: 10*log10(pixel_max^2 / MSE).

    Uses gt.max() as pixel_max (R2 default when pixel_max=None).
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    pred, gt = pred.float(), gt.float()
    pixel_max = gt.max()
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(pixel_max ** 2 / mse)).item()


@torch.no_grad()
def compute_volume_ssim(pred, gt, window_size=11):
    """3D SSIM matching R2-Gaussian: slice-by-slice 2D SSIM averaged over 3 axes.

    Skips slices where gt.max() <= 0.

    Returns:
        (mean_ssim, [ssim_x, ssim_y, ssim_z])
    """
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


def sobel_filter_3d(vol):
    """Sobel gradient magnitude of a (D, H, W) torch tensor."""
    v = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    v = F.pad(v, (1, 1, 1, 1, 1, 1), mode="replicate")
    smooth = torch.tensor([1, 2, 1], dtype=torch.float32, device=vol.device)
    diff = torch.tensor([-1, 0, 1], dtype=torch.float32, device=vol.device)
    # Kernel along X: smooth_z ⊗ smooth_y ⊗ diff_x
    kx = (smooth[:, None, None] * smooth[None, :, None] * diff[None, None, :]).reshape(1, 1, 3, 3, 3)
    # Kernel along Y: smooth_z ⊗ diff_y ⊗ smooth_x
    ky = (smooth[:, None, None] * diff[None, :, None] * smooth[None, None, :]).reshape(1, 1, 3, 3, 3)
    # Kernel along Z: diff_z ⊗ smooth_y ⊗ smooth_x
    kz = (diff[:, None, None] * smooth[None, :, None] * smooth[None, None, :]).reshape(1, 1, 3, 3, 3)
    gx = F.conv3d(v, kx)
    gy = F.conv3d(v, ky)
    gz = F.conv3d(v, kz)
    return torch.log1p(1.0 * torch.sqrt(gx**2 + gy**2 + gz**2)).clamp(0, 1).squeeze()


def _gauss_conv3d_separable(x, gauss_1d, pad):
    """Apply separable 3D Gaussian smoothing using three 1D conv3d passes."""
    ws = gauss_1d.shape[0]
    kx = gauss_1d.reshape(1, 1, ws, 1, 1)
    ky = gauss_1d.reshape(1, 1, 1, ws, 1)
    kz = gauss_1d.reshape(1, 1, 1, 1, ws)
    x = F.conv3d(x, kx, padding=(pad, 0, 0))
    x = F.conv3d(x, ky, padding=(0, pad, 0))
    x = F.conv3d(x, kz, padding=(0, 0, pad))
    return x


@torch.no_grad()
def compute_volume_ssim_3d(pred, gt, window_size=11):
    """True 3D SSIM using separable 3D Gaussian kernel."""
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    pred, gt = pred.float(), gt.float()

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss_1d = torch.exp(-coords**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    pad = window_size // 2

    p = pred.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    g = gt.unsqueeze(0).unsqueeze(0)

    data_range = gt.max() - gt.min()
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    mu1 = _gauss_conv3d_separable(p, gauss_1d, pad)
    mu2 = _gauss_conv3d_separable(g, gauss_1d, pad)
    sigma1_sq = _gauss_conv3d_separable(p**2, gauss_1d, pad) - mu1**2
    sigma2_sq = _gauss_conv3d_separable(g**2, gauss_1d, pad) - mu2**2
    sigma12 = _gauss_conv3d_separable(p * g, gauss_1d, pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


@torch.no_grad()
def compute_dice(pred, gt, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)):
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
def compute_surface_metrics(pred, gt, threshold=0.5, f_thresholds=(1.0, 2.0)):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    try:
        verts_p, _, _, _ = marching_cubes(pred, level=threshold)
        verts_g, _, _, _ = marching_cubes(gt, level=threshold)
    except ValueError:
        return {"chamfer": float("inf"), **{f"f1_{d:.0f}v": 0.0 for d in f_thresholds}}
    tree_p = KDTree(verts_p)
    tree_g = KDTree(verts_g)
    d_pred_to_gt, _ = tree_g.query(verts_p)
    d_gt_to_pred, _ = tree_p.query(verts_g)
    chamfer = 0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean())
    result = {"chamfer": float(chamfer)}
    for d in f_thresholds:
        prec = (d_pred_to_gt <= d).mean()
        rec = (d_gt_to_pred <= d).mean()
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        result[f"f1_{d:.0f}v"] = float(f1)
    return result


def log_diagnostics(model, writer, step):
    with torch.no_grad():
        _, cell_radius = radfoam.farthest_neighbor(
            model.primal_points,
            model.point_adjacency,
            model.point_adjacency_offsets,
        )
        writer.add_histogram("diagnostics/cell_radius", cell_radius, step)


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
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
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset
    train_data_handler = DataHandler(
        dataset_args, rays_per_batch=pipeline_args.rays_per_batch, device=device
    )
    train_data_handler.reload(split="train")

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(split="test")
    test_ray_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_proj_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.projections, batch_size=1, shuffle=False
    )

    train_ray_batch_fetcher = radfoam.BatchFetcher(
        train_data_handler.rays, batch_size=1, shuffle=False
    )
    train_proj_batch_fetcher = radfoam.BatchFetcher(
        train_data_handler.projections, batch_size=1, shuffle=False
    )

    # Setting up loss
    if pipeline_args.loss_type == "l1":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    # Setting up model
    model = CTScene(
        args=model_args,
        device=device,
    )

    # Override initial points if a file is provided
    init_points_file = getattr(model_args, "init_points_file", "")
    if init_points_file:
        pts = torch.load(init_points_file, map_location=device, weights_only=True)
        print(f"Overriding initial points from {init_points_file}: {pts.shape}")
        model.triangulation = radfoam.Triangulation(pts.float().contiguous().to(device))
        perm = model.triangulation.permutation().to(torch.long)
        model.primal_points = nn.Parameter(pts[perm].to(device))
        model.update_triangulation(rebuild=False)
        init_val = model.init_density
        density = torch.full(
            (pts.shape[0], 1), init_val, device=device, dtype=torch.float32
        )
        model.density = nn.Parameter(density[perm])

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    gt_volume = load_gt_volume(dataset_args.data_path, dataset_args.dataset)
    if gt_volume is not None:
        print(f"Loaded GT volume: shape={gt_volume.shape}")

    r2_volume = load_r2_volume(dataset_args.data_path)
    if r2_volume is not None:
        print(f"Loaded R2 volume: shape={r2_volume.shape}")

    def eval_views(data_handler, ray_batch_fetcher, proj_batch_fetcher,
                   return_images=False):
        rays = data_handler.rays
        points, *_ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        rmse_list = []
        psnr_list = []
        ssim_list = []
        pred_imgs = []
        gt_imgs = []
        ray_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                proj_batch = proj_batch_fetcher.next()[0]
                proj_output, _, _, _, _ = model(ray_batch, start_points[i])

                mse = ((proj_output - proj_batch) ** 2).mean()
                rmse_list.append(torch.sqrt(mse).item())
                psnr_list.append(compute_psnr(proj_output, proj_batch))
                ssim_list.append(compute_ssim(proj_output, proj_batch))
                if return_images:
                    pred_imgs.append(proj_output.squeeze(-1).cpu().numpy())
                    gt_imgs.append(proj_batch.squeeze(-1).cpu().numpy())
                    ray_list.append(ray_batch.cpu().numpy())
                torch.cuda.synchronize()

        result = {
            "rmse": sum(rmse_list) / len(rmse_list),
            "psnr": sum(psnr_list) / len(psnr_list),
            "ssim": sum(ssim_list) / len(ssim_list),
        }
        if return_images:
            result["pred_imgs"] = pred_imgs
            result["gt_imgs"] = gt_imgs
            result["rays"] = ray_list
        return result

    def log_basic(step, loss_val=None, tv_loss_val=None, tv_scale_val=None):
        """Log eval metrics, LR, point count. Called at step 0, log_interval, final."""
        test_metrics = eval_views(
            test_data_handler,
            test_ray_batch_fetcher,
            test_proj_batch_fetcher,
        )
        writer.add_scalar("test/rmse", test_metrics["rmse"], step)
        writer.add_scalar("test/psnr", test_metrics["psnr"], step)
        writer.add_scalar("test/ssim", test_metrics["ssim"], step)

        train_metrics = eval_views(
            train_data_handler,
            train_ray_batch_fetcher,
            train_proj_batch_fetcher,
        )
        writer.add_scalar("train/rmse", train_metrics["rmse"], step)
        writer.add_scalar("train/psnr", train_metrics["psnr"], step)
        writer.add_scalar("train/ssim", train_metrics["ssim"], step)

        num_points = model.primal_points.shape[0]
        writer.add_scalar("train/num_points", num_points, step)

        if loss_val is not None:
            writer.add_scalar("train/loss", loss_val, step)
        if tv_loss_val is not None and optimizer_args.tv_weight > 0:
            writer.add_scalar("train/tv_loss", tv_loss_val, step)
            if optimizer_args.tv_anneal and tv_scale_val is not None:
                writer.add_scalar("train/tv_scale", tv_scale_val, step)

        if hasattr(model, '_triangulation_retries'):
            writer.add_scalar("diagnostics/triangulation_retries", model._triangulation_retries, step)

        if (pipeline_args.bf_start >= 0
                and pipeline_args.bf_start <= step < pipeline_args.bf_until):
            t = (step - pipeline_args.bf_start) / max(
                1, pipeline_args.bf_until - pipeline_args.bf_start - 1
            )
            writer.add_scalar("train/bf_sigma",
                              pipeline_args.bf_sigma_init + t * (pipeline_args.bf_sigma_final - pipeline_args.bf_sigma_init), step)
            writer.add_scalar("train/bf_sigma_v",
                              pipeline_args.bf_sigma_v_final + 0.5 * (pipeline_args.bf_sigma_v_init - pipeline_args.bf_sigma_v_final) * (1 + math.cos(math.pi * t)), step)

        writer.add_scalar("lr/points_lr", model.xyz_scheduler_args(step), step)
        writer.add_scalar("lr/density_lr", model.den_scheduler_args(step), step)
        if model.grad_scheduler_args is not None:
            writer.add_scalar("lr/gradient_lr",
                              model.grad_scheduler_args(step - model._gradient_start), step)
        if hasattr(model, 'peak_scheduler_args'):
            writer.add_scalar("lr/peak_lr",
                              model.peak_scheduler_args(step - model._gaussian_start), step)
        if getattr(model, '_gaussian_active', False) and hasattr(model, 'density_peak'):
            import torch.nn.functional as _F
            mu_peak = _F.softplus(model.density_peak, beta=10)
            writer.add_scalar("gaussian/mu_peak_mean", mu_peak.mean().item(), step)
            writer.add_scalar("gaussian/mu_peak_max", mu_peak.max().item(), step)
            offset_mag = (model.delta_raw.detach().tanh()).norm(dim=-1)
            writer.add_scalar("gaussian/offset_mag_mean", offset_mag.mean().item(), step)

        if pipeline_args.interpolation_start >= 0 and step >= pipeline_args.densify_until:
            ramp_length = max(1, pipeline_args.interpolation_start - pipeline_args.densify_until)
            frac = min(1.0, (step - pipeline_args.densify_until) / ramp_length)
            writer.add_scalar("train/interp_fraction", frac, step)

        print(f"Step {step}: test PSNR={test_metrics['psnr']:.2f}, "
              f"train PSNR={train_metrics['psnr']:.2f}, "
              f"points={num_points}")
        return test_metrics, train_metrics

    def log_diag(step, hit_count=None):
        """Log slice visualizations, diagnostics, error comparison. Called at step 0, diag_interval, final."""
        log_density_histogram(model, writer, step)
        log_diagnostics(model, writer, step)
        cell_entropy = model.compute_neighbor_entropy(n_bins=pipeline_args.entropy_bins)
        writer.add_histogram("diagnostics/cell_entropy", cell_entropy, step)
        writer.add_scalar("diagnostics/entropy_mean", cell_entropy.mean().item(), step)
        writer.add_scalar("diagnostics/entropy_max", cell_entropy.max().item(), step)

        with torch.no_grad():
            field = field_from_model(model)
            _, cell_radius = radfoam.farthest_neighbor(
                model.primal_points,
                model.point_adjacency,
                model.point_adjacency_offsets,
            )
            sigma = pipeline_args.interp_sigma_scale * cell_radius.median().item()
            sigma_v = pipeline_args.interp_sigma_v

            # Slice visualizations
            axes = [0, 1, 2]
            slice_coords = [-0.2, 0.0, 0.2]
            d_slices, idw_slices, cd_slices = [], [], []
            gt_slices, r2_slices_list, ve_slices = [], [], []
            for a in axes:
                for c in slice_coords:
                    coords_2d = make_slice_coords(a, c, 256, 1.0)
                    d_slices.append(query_density(field, coords_2d))
                    idw_slices.append(sample_idw(field, coords_2d,
                                                 sigma=sigma, sigma_v=sigma_v))
                    cd_slices.append(
                        compute_cell_density_slice(field["points"], a, c, 64, 1.0)
                    )
                    gt_slices.append(sample_gt_slice(gt_volume, a, c, 256, 1.0))
                    r2_slices_list.append(sample_gt_slice(r2_volume, a, c, 256, 1.0))
                    ve_slices.append(compute_voronoi_edges(field, a, c, 256, 1.0))

            log_fig_il = partial(writer.add_figure, f"slices_interleaved/{experiment_name}", global_step=step)
            log_fig_sobel = partial(writer.add_figure, f"slices_sobel/{experiment_name}", global_step=step)
            metrics = visualize_slices(
                d_slices, idw_slices, cd_slices,
                gt_slices=gt_slices,
                r2_slices=r2_slices_list if r2_volume is not None else None,
                writer_fn_interleaved=log_fig_il,
                writer_fn_sobel=log_fig_sobel,
                voronoi_edges=ve_slices,
            )
            if metrics is not None:
                for key, val in metrics.items():
                    parts = key.split('_')
                    if len(parts) == 2:
                        tag = f"slice_{parts[1]}/{parts[0]}"
                    else:
                        tag = f"slice_{parts[-1]}/{'_'.join(parts[:-1])}"
                    writer.add_scalar(tag, val, step)

            # Cell heatmap
            log_fig_hm = partial(writer.add_figure, f"cell_heatmap/{experiment_name}", global_step=step)
            visualize_cell_heatmap(cd_slices, writer_fn=log_fig_hm)

            # IDW diagnostics
            diag_coords = make_slice_coords(axis=2, coord=0.0, resolution=256, extent=1.0)
            diag = sample_idw_diagnostic(field, diag_coords,
                                         sigma=sigma, sigma_v=sigma_v)
            diag_writer = partial(writer.add_figure, f"idw_diagnostics/{experiment_name}", global_step=step)
            visualize_idw_diagnostics(diag, writer_fn=diag_writer)
            n_holes = (diag["diff"] > 0.05).sum()
            writer.add_scalar("diagnostics/idw_hole_pixels", n_holes, step)
            writer.add_scalar("diagnostics/idw_mean_cell_weight", diag["cell_weight"].mean(), step)
            writer.add_scalar("diagnostics/idw_mean_neighbor_count", diag["neighbor_count"].mean(), step)

            # Starvation diagnostics
            if hasattr(model, '_starvation_count'):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                writer.add_histogram("diagnostics/starvation_count",
                                     model._starvation_count, step)
                if hasattr(model, '_starvation_lifetimes') and model._starvation_lifetimes:
                    all_lengths = torch.cat([t[0] for t in model._starvation_lifetimes])
                    all_radii = torch.cat([t[1] for t in model._starvation_lifetimes])
                    lengths_float = all_lengths.float()
                    max_len = int(all_lengths.max().item())
                    bins = torch.arange(0, max_len + 2, dtype=torch.float32) - 0.5
                    writer.add_histogram("diagnostics/starvation_lifetimes", lengths_float, step, bins=bins)
                    writer.add_scalar("diagnostics/starvation_lifetime_mean", all_lengths.float().mean().item(), step)
                    writer.add_scalar("diagnostics/starvation_lifetime_median", all_lengths.float().median().item(), step)
                    writer.add_scalar("diagnostics/starvation_lifetime_max", all_lengths.max().item(), step)
                    writer.add_scalar("diagnostics/starvation_lifetime_count", all_lengths.shape[0], step)
                    writer.add_scalar("diagnostics/starvation_lifetime_p95",
                                       torch.quantile(lengths_float, 0.95).item(), step)
                    writer.add_scalar("diagnostics/starvation_lifetime_p99",
                                       torch.quantile(lengths_float, 0.99).item(), step)
                    long_count = (all_lengths > 10).sum().item()
                    writer.add_scalar("diagnostics/starvation_lifetime_long_count", long_count, step)

                    log_r_lt = torch.log10(all_radii.clamp(min=1e-6))
                    lt_bin_edges = torch.linspace(log_r_lt.min(), log_r_lt.max(), 21)
                    lt_bin_means, lt_bin_centers = [], []
                    for b in range(20):
                        mask = (log_r_lt >= lt_bin_edges[b]) & (log_r_lt < lt_bin_edges[b + 1])
                        if mask.any():
                            lt_bin_means.append(all_lengths[mask].float().mean().item())
                            lt_bin_centers.append(((lt_bin_edges[b] + lt_bin_edges[b + 1]) / 2).item())
                    if lt_bin_centers:
                        fig_lt, ax_lt = plt.subplots(figsize=(6, 4))
                        bw_lt = (lt_bin_centers[1] - lt_bin_centers[0]) * 0.9 if len(lt_bin_centers) > 1 else 0.1
                        ax_lt.bar(lt_bin_centers, lt_bin_means, width=bw_lt)
                        ax_lt.set_xlabel("log10(cell_radius)")
                        ax_lt.set_ylabel("mean starvation lifetime")
                        ax_lt.set_title("Starvation Lifetime vs Cell Radius")
                        writer.add_figure(f"diagnostics/starvation_lifetime_vs_radius/{experiment_name}", fig_lt, step)
                        plt.close(fig_lt)

                    model._starvation_lifetimes.clear()

                starvation = model._starvation_count.float()
                log_r = torch.log10(cell_radius.clamp(min=1e-6))
                n_bins = 20
                bin_edges = torch.linspace(log_r.min(), log_r.max(), n_bins + 1)
                bin_means, bin_centers = [], []
                for b in range(n_bins):
                    mask = (log_r >= bin_edges[b]) & (log_r < bin_edges[b + 1])
                    if mask.any():
                        bin_means.append(starvation[mask].mean().item())
                        bin_centers.append(((bin_edges[b] + bin_edges[b + 1]) / 2).item())
                if bin_centers:
                    fig_sv, ax_sv = plt.subplots(figsize=(6, 4))
                    bw = (bin_centers[1] - bin_centers[0]) * 0.9 if len(bin_centers) > 1 else 0.1
                    ax_sv.bar(bin_centers, bin_means, width=bw)
                    ax_sv.set_xlabel("log10(cell_radius)")
                    ax_sv.set_ylabel("mean starvation count")
                    ax_sv.set_title("Starvation vs Cell Radius")
                    writer.add_figure(f"diagnostics/starvation_vs_radius/{experiment_name}", fig_sv, step)
                    plt.close(fig_sv)

            # Rays-per-cell diagnostics
            if hit_count is not None:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                hc = hit_count.squeeze().float()
                writer.add_histogram("diagnostics/rays_per_cell", hc, step)
                writer.add_scalar("diagnostics/rays_per_cell_mean", hc.mean().item(), step)
                writer.add_scalar("diagnostics/rays_per_cell_median", hc.median().item(), step)
                writer.add_scalar("diagnostics/rays_per_cell_zero_frac",
                                   (hc == 0).float().mean().item(), step)

                log_r_hc = torch.log10(cell_radius.clamp(min=1e-6))
                hc_bin_edges = torch.linspace(log_r_hc.min(), log_r_hc.max(), 21)
                hc_bin_means, hc_bin_centers = [], []
                for b in range(20):
                    mask = (log_r_hc >= hc_bin_edges[b]) & (log_r_hc < hc_bin_edges[b + 1])
                    if mask.any():
                        hc_bin_means.append(hc[mask].mean().item())
                        hc_bin_centers.append(((hc_bin_edges[b] + hc_bin_edges[b + 1]) / 2).item())
                if hc_bin_centers:
                    fig_hc, ax_hc = plt.subplots(figsize=(6, 4))
                    bw_hc = (hc_bin_centers[1] - hc_bin_centers[0]) * 0.9 if len(hc_bin_centers) > 1 else 0.1
                    ax_hc.bar(hc_bin_centers, hc_bin_means, width=bw_hc)
                    ax_hc.set_xlabel("log10(cell_radius)")
                    ax_hc.set_ylabel("mean rays per cell")
                    ax_hc.set_title("Rays Per Cell vs Cell Radius")
                    writer.add_figure(f"diagnostics/rays_per_cell_vs_radius/{experiment_name}", fig_hc, step)
                    plt.close(fig_hc)

            # Error comparison figure (projection error vs volume error)
            if gt_volume is not None:
                test_with_imgs = eval_views(
                    test_data_handler,
                    test_ray_batch_fetcher,
                    test_proj_batch_fetcher,
                    return_images=True,
                )
                n_views = len(test_with_imgs["pred_imgs"])
                n_show = min(8, n_views)
                view_indices = np.linspace(0, n_views - 1, n_show, dtype=int)

                # Voxelize at 128³ for speed
                raw_vol, _ = voxelize_volumes(field, resolution=128, extent=1.0,
                                              sigma=sigma, sigma_v=sigma_v)
                # Downsample GT to match
                from skimage.transform import resize
                gt_vol_ds = resize(gt_volume, (128, 128, 128), order=1,
                                   preserve_range=True).astype(np.float32)
                error_vol = np.abs(raw_vol - gt_vol_ds)

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.colors import LinearSegmentedColormap
                # White-to-blue colormap for volume errors on white background
                blues_on_white = LinearSegmentedColormap.from_list(
                    "blues_on_white", ["white", "royalblue", "darkblue"])

                fig, axs = plt.subplots(6, n_show, figsize=(3 * n_show, 18))
                if n_show == 1:
                    axs = axs[:, None]

                proj_vmax = max(max(test_with_imgs["gt_imgs"][vi].max() for vi in view_indices), max(test_with_imgs["pred_imgs"][vi].max() for vi in view_indices)) + 1e-9
                err_max = max(abs(test_with_imgs["pred_imgs"][vi] - test_with_imgs["gt_imgs"][vi]).max() for vi in view_indices) + 1e-9

                # Pre-compute volume DRRs for shared scaling
                vol_err_drrs = []
                for col, vi in enumerate(view_indices):
                    rays_np = test_with_imgs["rays"][vi]
                    vol_err_drrs.append(render_volume_drr(error_vol, rays_np,
                                                          extent=1.0, num_samples=128))
                vol_err_vmax = max(d.max() for d in vol_err_drrs) + 1e-6

                for col, vi in enumerate(view_indices):
                    gt_img = test_with_imgs["gt_imgs"][vi]
                    pred_img = test_with_imgs["pred_imgs"][vi]
                    proj_err = pred_img - gt_img
                    rays_np = test_with_imgs["rays"][vi]
                    vol_err_drr = vol_err_drrs[col]

                    # Row 0: GT projection
                    axs[0, col].imshow(gt_img.T, origin="lower", cmap="gray",
                                       vmin=0, vmax=proj_vmax)
                    axs[0, col].set_title(f"v{vi}", fontsize=8)
                    axs[0, col].axis("off")

                    # Row 1: Predicted projection
                    axs[1, col].imshow(pred_img.T, origin="lower", cmap="gray",
                                       vmin=0, vmax=proj_vmax)
                    axs[1, col].axis("off")

                    # Row 2: Voxelized volume Beer-Lambert DRR (sanity check)
                    vol_drr = render_volume_drr(raw_vol, rays_np, extent=1.0,
                                               num_samples=128)
                    axs[2, col].imshow(vol_drr.T, origin="lower", cmap="gray",
                                       vmin=0, vmax=proj_vmax)
                    axs[2, col].axis("off")

                    # Row 3: Projection error (signed, bwr)
                    axs[3, col].imshow(proj_err.T, origin="lower", cmap="bwr",
                                       vmin=-err_max, vmax=err_max)
                    axs[3, col].axis("off")

                    # Row 4: Volume error DRR (blue on white)
                    axs[4, col].imshow(vol_err_drr.T, origin="lower",
                                       cmap=blues_on_white, vmin=0, vmax=vol_err_vmax)
                    axs[4, col].axis("off")

                    # Row 5: |proj error| - |vol error DRR|
                    abs_proj_err = np.abs(proj_err)
                    vol_err_norm = vol_err_drr / (vol_err_vmax + 1e-8) * (err_max + 1e-8)
                    diff = abs_proj_err - vol_err_norm
                    diff_max = max(np.abs(diff).max(), 1e-6)
                    axs[5, col].imshow(diff.T, origin="lower", cmap="bwr",
                                       vmin=-diff_max, vmax=diff_max)
                    axs[5, col].axis("off")

                fig.tight_layout(rect=[0.08, 0, 1, 1])  # leave space on left for labels

                # Row labels on left side using fig.text (independent of axis state)
                row_labels = ["GT Projection", "Predicted Projection",
                              "Voxelized Volume DRR",
                              "Projection Error", "Volume Error DRR",
                              "|Proj Err| \u2212 |Vol Err|"]
                for row, label in enumerate(row_labels):
                    bbox = axs[row, 0].get_position()
                    y_center = (bbox.y0 + bbox.y1) / 2
                    fig.text(0.01, y_center, label, fontsize=9, rotation=90,
                             ha="left", va="center")
                writer.add_figure(f"error_comparison/{experiment_name}", fig, step)
                plt.close(fig)

        return metrics

    def train_loop(viewer):
        print("Training")

        log_interval = max(1, pipeline_args.iterations * pipeline_args.log_percent // 100)
        diag_interval = max(1, pipeline_args.iterations * pipeline_args.diag_percent // 100)

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        ray_batch, proj_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        # Log initial state (step 0, before any optimization)
        if not pipeline_args.debug:
            log_basic(0)
            log_diag(0)

        _loss_cpu = None
        with tqdm.trange(pipeline_args.iterations) as train:
            for i in train:
                return_diag = (i % diag_interval == diag_interval - 1 and not pipeline_args.debug)
                proj_output, contribution, hit_count, _, _ = model(ray_batch, return_contribution=return_diag)

                loss = loss_fn(proj_output, proj_batch)

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

                model.optimizer.zero_grad(set_to_none=True)

                # Hide latency of data loading behind the backward pass
                loss.backward()
                event = torch.cuda.Event()
                event.record()

                ray_batch, proj_batch = next(data_iterator)
                event.synchronize()

                if optimizer_args.density_grad_clip > 0 and model.density.grad is not None:
                    model.density.grad.clamp_(-optimizer_args.density_grad_clip, optimizer_args.density_grad_clip)

                model.optimizer.step()
                model.update_starvation_count()

                if i < pipeline_args.densify_until:
                    model.density.data.clamp_(min=-1.0)

                # Bilateral filter: direct density smoothing
                if (pipeline_args.bf_start >= 0
                        and pipeline_args.bf_start <= i < pipeline_args.bf_until
                        and i % pipeline_args.bf_period == 0):
                    t = (i - pipeline_args.bf_start) / max(
                        1, pipeline_args.bf_until - pipeline_args.bf_start - 1
                    )
                    bf_sigma = (pipeline_args.bf_sigma_init
                                + t * (pipeline_args.bf_sigma_final - pipeline_args.bf_sigma_init))
                    # Cosine anneal for sigma_v: stays high longer, then drops
                    bf_sigma_v = (pipeline_args.bf_sigma_v_final
                                  + 0.5 * (pipeline_args.bf_sigma_v_init - pipeline_args.bf_sigma_v_final)
                                  * (1 + math.cos(math.pi * t)))
                    model.apply_bilateral_filter(bf_sigma, bf_sigma_v)

                model.update_learning_rate(i)

                # Interpolation interleaving schedule
                if pipeline_args.interpolation_start >= 0:
                    if i >= pipeline_args.interpolation_start:
                        # 100% interpolation
                        if not getattr(model, '_interpolation_mode', False):
                            model.set_interpolation_mode(True)
                            print(f"Full interpolation mode at iter {i}")
                    elif pipeline_args.interp_ramp and i >= pipeline_args.densify_until:
                        # Linear ramp: fraction of steps that use interpolation
                        ramp_length = pipeline_args.interpolation_start - pipeline_args.densify_until
                        frac = (i - pipeline_args.densify_until) / ramp_length
                        period = 10
                        use_interp = (i % period) < (frac * period)
                        model.set_interpolation_mode(use_interp)
                    else:
                        model.set_interpolation_mode(False)

                if _loss_cpu is not None:
                    train.set_postfix(loss=f"{_loss_cpu.item():.5f}")
                _loss_cpu = loss.detach().to("cpu", non_blocking=True)

                if i % log_interval == log_interval - 1 and not pipeline_args.debug:
                    tv_loss_val = tv_loss.item() if optimizer_args.tv_weight > 0 and i >= optimizer_args.tv_start else None
                    tv_scale_val = tv_scale if optimizer_args.tv_anneal and tv_loss_val is not None else None
                    log_basic(i, loss_val=loss.item(), tv_loss_val=tv_loss_val,
                              tv_scale_val=tv_scale_val)

                if i % diag_interval == diag_interval - 1 and not pipeline_args.debug:
                    log_diag(i, hit_count=hit_count)

                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                    # Refresh targeting state if active (radius changed)
                    if (hasattr(train_data_handler, '_target_weights')
                            and pipeline_args.targeted_fraction > 0):
                        cell_weights = model.compute_cell_importance()
                        if cell_weights.sum() > 0:
                            points_t, *_ = model.get_trace_data()
                            train_data_handler.update_targeting(
                                cell_weights, points_t.detach(),
                                model._cached_cell_radius.detach(),
                                pipeline_args.targeted_fraction,
                            )

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler,
                        contrast_alpha=pipeline_args.contrast_alpha,
                    )

                    densify_stats = model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                        gradient_fraction=pipeline_args.gradient_fraction,
                        idw_fraction=pipeline_args.idw_fraction,
                        entropy_fraction=pipeline_args.entropy_fraction,
                        entropy_bins=pipeline_args.entropy_bins,
                        redundancy_threshold=pipeline_args.redundancy_threshold,
                        redundancy_cap=pipeline_args.redundancy_cap,
                        sigma_scale=pipeline_args.interp_sigma_scale,
                        sigma_v=pipeline_args.interp_sigma_v,
                    )

                    if not pipeline_args.debug and densify_stats is not None:
                        for key, val in densify_stats.items():
                            writer.add_scalar(f"densify/{key}", val, i)

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

                    # Rebuild targeted sampling pool (radius is fresh from update_triangulation)
                    targeted_start = pipeline_args.targeted_start
                    if targeted_start < 0:
                        targeted_start = pipeline_args.densify_from
                    if (pipeline_args.targeted_fraction > 0
                            and i >= targeted_start):
                        cell_weights = model.compute_cell_importance()
                        if cell_weights.sum() > 0:
                            points_t, *_ = model.get_trace_data()
                            train_data_handler.update_targeting(
                                cell_weights, points_t.detach(),
                                model._cached_cell_radius.detach(),
                                pipeline_args.targeted_fraction,
                            )
                            data_iterator = train_data_handler.get_targeted_iter()
                            ray_batch, proj_batch = next(data_iterator)

                    # Linear growth
                    iters_since_densification = 0
                    next_densification_after = int(
                        (
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (
                                pipeline_args.densify_until
                                - pipeline_args.densify_from
                            )
                        )
                        / (model.num_final_points - model.num_init_points)
                    )
                    next_densification_after = max(
                        next_densification_after, 100
                    )

                if i == pipeline_args.densify_until:
                    model.update_triangulation(incremental=False)
                    n_standalone_pruned = model.prune_only(train_data_handler)
                    if not pipeline_args.debug:
                        writer.add_scalar("densify/standalone_pruned", n_standalone_pruned, i)
                        writer.add_scalar("densify/points_after", model.primal_points.shape[0], i)

                    if pipeline_args.interpolation_start >= 0:
                        use_adaptive = pipeline_args.per_cell_sigma or pipeline_args.per_neighbor_sigma
                        if use_adaptive:
                            # Pass raw scale factor — kernel multiplies by per-cell radius
                            sigma = pipeline_args.interp_sigma_scale
                        else:
                            _, cell_radius = radfoam.farthest_neighbor(
                                model.primal_points,
                                model.point_adjacency,
                                model.point_adjacency_offsets,
                            )
                            sigma = pipeline_args.interp_sigma_scale * cell_radius.median().item()
                        model.set_interpolation_mode(
                            False, sigma=sigma, sigma_v=pipeline_args.interp_sigma_v,
                            per_cell_sigma=pipeline_args.per_cell_sigma,
                            per_neighbor_sigma=pipeline_args.per_neighbor_sigma,
                        )
                        print(f"Prepared interpolation sigma={sigma:.6f} "
                              f"adaptive={use_adaptive} at densify_until={i}")

                if (
                    optimizer_args.gradient_start >= 0
                    and i == optimizer_args.gradient_start
                ):
                    model.initialize_gradients(optimizer_args)

                # Gaussian mode schedule
                if (
                    optimizer_args.gaussian_start >= 0
                    and i == optimizer_args.gaussian_start
                ):
                    model.initialize_gaussian(optimizer_args)
                    if optimizer_args.freeze_base_at_gaussian:
                        model.density.requires_grad_(False)
                        print(f"Froze base density at iter {i}")

                if (
                    optimizer_args.joint_finetune_start >= 0
                    and i == optimizer_args.joint_finetune_start
                    and getattr(model, '_gaussian_active', False)
                ):
                    model.density.requires_grad_(True)
                    print(f"Unfroze base density for joint fine-tuning at iter {i}")

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)
                    #model.prune_only(train_data_handler)

                if viewer is not None and viewer.is_closed():
                    break

        if not pipeline_args.debug:
            model.save_ply(f"{out_dir}/scene.ply")
            model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    train_loop(viewer=None)

    iters = pipeline_args.iterations

    if not pipeline_args.debug:
        # Final basic + diag logging
        test_metrics, train_metrics = log_basic(iters)
        slice_metrics = log_diag(iters)

        with open(f"{out_dir}/metrics.txt", "w") as f:
            f.write(f"Test  RMSE: {test_metrics['rmse']:.6f}\n")
            f.write(f"Test  PSNR: {test_metrics['psnr']:.4f}\n")
            f.write(f"Test  SSIM: {test_metrics['ssim']:.6f}\n")
            f.write(f"Train RMSE: {train_metrics['rmse']:.6f}\n")
            f.write(f"Train PSNR: {train_metrics['psnr']:.4f}\n")
            f.write(f"Train SSIM: {train_metrics['ssim']:.6f}\n")

        if slice_metrics is not None:
            with open(f"{out_dir}/metrics.txt", "a") as f:
                for key, val in slice_metrics.items():
                    f.write(f"Slice {key}: {val:.4f}\n")

        model_path = f"{out_dir}/model.pt"

        # Compute interp sigma from live model for Python-side IDW
        with torch.no_grad():
            _, cell_radius = radfoam.farthest_neighbor(
                model.primal_points,
                model.point_adjacency,
                model.point_adjacency_offsets,
            )
            interp_sigma = pipeline_args.interp_sigma_scale * cell_radius.median().item()
            interp_sigma_v = pipeline_args.interp_sigma_v

        field = load_density_field(model_path)

        # 3D volume metrics (matching R2-Gaussian evaluation)
        if gt_volume is not None:
            vol_res = gt_volume.shape[0]
            print(f"Voxelizing at {vol_res}³ for 3D volume metrics...")
            raw_vol, idw_vol = voxelize_volumes(
                field, resolution=vol_res, extent=1.0,
                sigma=interp_sigma,
                sigma_v=interp_sigma_v,
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

            iters = pipeline_args.iterations
            writer.add_scalar("test/vol_raw_psnr", raw_psnr_3d, iters)
            writer.add_scalar("test/vol_raw_ssim", raw_ssim_3d, iters)
            writer.add_scalar("test/vol_idw_psnr", idw_psnr_3d, iters)
            writer.add_scalar("test/vol_idw_ssim", idw_ssim_3d, iters)
            for ax_i, ax_name in enumerate(["x", "y", "z"]):
                writer.add_scalar(f"test/vol_raw_ssim_{ax_name}", raw_ssim_ax[ax_i], iters)
                writer.add_scalar(f"test/vol_idw_ssim_{ax_name}", idw_ssim_ax[ax_i], iters)

            with open(f"{out_dir}/metrics.txt", "a") as f:
                f.write(f"Vol Raw PSNR: {raw_psnr_3d:.4f}\n")
                f.write(f"Vol Raw SSIM: {raw_ssim_3d:.6f}\n")
                f.write(f"Vol IDW PSNR: {idw_psnr_3d:.4f}\n")
                f.write(f"Vol IDW SSIM: {idw_ssim_3d:.6f}\n")
                for ax_i, ax_name in enumerate(["X", "Y", "Z"]):
                    f.write(f"Vol Raw SSIM_{ax_name}: {raw_ssim_ax[ax_i]:.6f}\n")
                    f.write(f"Vol IDW SSIM_{ax_name}: {idw_ssim_ax[ax_i]:.6f}\n")

            # Sobel-filtered volume metrics
            gt_sobel_vol = sobel_filter_3d(vol_gt_t)
            raw_sobel_vol = sobel_filter_3d(raw_vol_t)
            idw_sobel_vol = sobel_filter_3d(idw_vol_t)

            sobel_raw_psnr_3d = compute_volume_psnr(raw_sobel_vol, gt_sobel_vol)
            sobel_raw_ssim_3d, _ = compute_volume_ssim(raw_sobel_vol, gt_sobel_vol)
            sobel_idw_psnr_3d = compute_volume_psnr(idw_sobel_vol, gt_sobel_vol)
            sobel_idw_ssim_3d, _ = compute_volume_ssim(idw_sobel_vol, gt_sobel_vol)

            # True 3D SSIM
            raw_ssim3d = compute_volume_ssim_3d(raw_vol_t, vol_gt_t)
            idw_ssim3d = compute_volume_ssim_3d(idw_vol_t, vol_gt_t)

            print(f"Vol Raw  Sobel PSNR: {sobel_raw_psnr_3d:.4f}, Sobel SSIM: {sobel_raw_ssim_3d:.6f}")
            print(f"Vol IDW  Sobel PSNR: {sobel_idw_psnr_3d:.4f}, Sobel SSIM: {sobel_idw_ssim_3d:.6f}")
            print(f"Vol Raw  SSIM3D: {raw_ssim3d:.6f}")
            print(f"Vol IDW  SSIM3D: {idw_ssim3d:.6f}")

            writer.add_scalar("test/vol_raw_sobel_psnr", sobel_raw_psnr_3d, iters)
            writer.add_scalar("test/vol_raw_sobel_ssim", sobel_raw_ssim_3d, iters)
            writer.add_scalar("test/vol_idw_sobel_psnr", sobel_idw_psnr_3d, iters)
            writer.add_scalar("test/vol_idw_sobel_ssim", sobel_idw_ssim_3d, iters)
            writer.add_scalar("test/vol_raw_ssim3d", raw_ssim3d, iters)
            writer.add_scalar("test/vol_idw_ssim3d", idw_ssim3d, iters)

            with open(f"{out_dir}/metrics.txt", "a") as f:
                f.write(f"Vol Raw Sobel PSNR: {sobel_raw_psnr_3d:.4f}\n")
                f.write(f"Vol Raw Sobel SSIM: {sobel_raw_ssim_3d:.6f}\n")
                f.write(f"Vol IDW Sobel PSNR: {sobel_idw_psnr_3d:.4f}\n")
                f.write(f"Vol IDW Sobel SSIM: {sobel_idw_ssim_3d:.6f}\n")
                f.write(f"Vol Raw SSIM3D: {raw_ssim3d:.6f}\n")
                f.write(f"Vol IDW SSIM3D: {idw_ssim3d:.6f}\n")

            # Geometric metrics
            raw_dice, _ = compute_dice(raw_vol_t, vol_gt_t)
            idw_dice, _ = compute_dice(idw_vol_t, vol_gt_t)
            raw_surf = compute_surface_metrics(raw_vol_t, vol_gt_t)
            idw_surf = compute_surface_metrics(idw_vol_t, vol_gt_t)

            print(f"Vol Raw  Dice: {raw_dice:.6f}, CD: {raw_surf['chamfer']:.4f}v, "
                  f"F1@1v: {raw_surf['f1_1v']:.4f}, F1@2v: {raw_surf['f1_2v']:.4f}")
            print(f"Vol IDW  Dice: {idw_dice:.6f}, CD: {idw_surf['chamfer']:.4f}v, "
                  f"F1@1v: {idw_surf['f1_1v']:.4f}, F1@2v: {idw_surf['f1_2v']:.4f}")

            writer.add_scalar("test/vol_raw_dice", raw_dice, iters)
            writer.add_scalar("test/vol_idw_dice", idw_dice, iters)
            writer.add_scalar("test/vol_raw_chamfer", raw_surf["chamfer"], iters)
            writer.add_scalar("test/vol_idw_chamfer", idw_surf["chamfer"], iters)
            writer.add_scalar("test/vol_raw_f1_1v", raw_surf["f1_1v"], iters)
            writer.add_scalar("test/vol_idw_f1_1v", idw_surf["f1_1v"], iters)
            writer.add_scalar("test/vol_raw_f1_2v", raw_surf["f1_2v"], iters)
            writer.add_scalar("test/vol_idw_f1_2v", idw_surf["f1_2v"], iters)

            with open(f"{out_dir}/metrics.txt", "a") as f:
                f.write(f"Vol Raw Dice: {raw_dice:.6f}\n")
                f.write(f"Vol IDW Dice: {idw_dice:.6f}\n")
                f.write(f"Vol Raw CD: {raw_surf['chamfer']:.4f}\n")
                f.write(f"Vol IDW CD: {idw_surf['chamfer']:.4f}\n")
                f.write(f"Vol Raw F1 1v: {raw_surf['f1_1v']:.4f}\n")
                f.write(f"Vol IDW F1 1v: {idw_surf['f1_1v']:.4f}\n")
                f.write(f"Vol Raw F1 2v: {raw_surf['f1_2v']:.4f}\n")
                f.write(f"Vol IDW F1 2v: {idw_surf['f1_2v']:.4f}\n")

            # R2-Gaussian volume metrics
            if r2_volume is not None:
                r2_vol_t = torch.from_numpy(r2_volume).float().cuda()

                r2_psnr_3d = compute_volume_psnr(r2_vol_t, vol_gt_t)
                r2_ssim_3d, r2_ssim_ax = compute_volume_ssim(r2_vol_t, vol_gt_t)

                r2_sobel_vol = sobel_filter_3d(r2_vol_t)
                sobel_r2_psnr_3d = compute_volume_psnr(r2_sobel_vol, gt_sobel_vol)
                sobel_r2_ssim_3d, _ = compute_volume_ssim(r2_sobel_vol, gt_sobel_vol)

                r2_ssim3d = compute_volume_ssim_3d(r2_vol_t, vol_gt_t)

                r2_dice, _ = compute_dice(r2_vol_t, vol_gt_t)
                r2_surf = compute_surface_metrics(r2_vol_t, vol_gt_t)

                print(f"Vol R2   PSNR: {r2_psnr_3d:.4f}, SSIM: {r2_ssim_3d:.6f}")
                print(f"Vol R2   Sobel PSNR: {sobel_r2_psnr_3d:.4f}, Sobel SSIM: {sobel_r2_ssim_3d:.6f}")
                print(f"Vol R2   SSIM3D: {r2_ssim3d:.6f}")
                print(f"Vol R2   Dice: {r2_dice:.6f}, CD: {r2_surf['chamfer']:.4f}v, "
                      f"F1@1v: {r2_surf['f1_1v']:.4f}, F1@2v: {r2_surf['f1_2v']:.4f}")

                writer.add_scalar("test/vol_r2_psnr", r2_psnr_3d, iters)
                writer.add_scalar("test/vol_r2_ssim", r2_ssim_3d, iters)
                for ax_i, ax_name in enumerate(["x", "y", "z"]):
                    writer.add_scalar(f"test/vol_r2_ssim_{ax_name}", r2_ssim_ax[ax_i], iters)
                writer.add_scalar("test/vol_r2_sobel_psnr", sobel_r2_psnr_3d, iters)
                writer.add_scalar("test/vol_r2_sobel_ssim", sobel_r2_ssim_3d, iters)
                writer.add_scalar("test/vol_r2_ssim3d", r2_ssim3d, iters)
                writer.add_scalar("test/vol_r2_dice", r2_dice, iters)
                writer.add_scalar("test/vol_r2_chamfer", r2_surf["chamfer"], iters)
                writer.add_scalar("test/vol_r2_f1_1v", r2_surf["f1_1v"], iters)
                writer.add_scalar("test/vol_r2_f1_2v", r2_surf["f1_2v"], iters)

                with open(f"{out_dir}/metrics.txt", "a") as f:
                    f.write(f"Vol R2 PSNR: {r2_psnr_3d:.4f}\n")
                    f.write(f"Vol R2 SSIM: {r2_ssim_3d:.6f}\n")
                    for ax_i, ax_name in enumerate(["X", "Y", "Z"]):
                        f.write(f"Vol R2 SSIM_{ax_name}: {r2_ssim_ax[ax_i]:.6f}\n")
                    f.write(f"Vol R2 Sobel PSNR: {sobel_r2_psnr_3d:.4f}\n")
                    f.write(f"Vol R2 Sobel SSIM: {sobel_r2_ssim_3d:.6f}\n")
                    f.write(f"Vol R2 SSIM3D: {r2_ssim3d:.6f}\n")
                    f.write(f"Vol R2 Dice: {r2_dice:.6f}\n")
                    f.write(f"Vol R2 CD: {r2_surf['chamfer']:.4f}\n")
                    f.write(f"Vol R2 F1 1v: {r2_surf['f1_1v']:.4f}\n")
                    f.write(f"Vol R2 F1 2v: {r2_surf['f1_2v']:.4f}\n")

                del r2_vol_t, r2_sobel_vol

            if pipeline_args.save_volume:
                np.save(f"{out_dir}/volume_raw.npy", raw_vol)
                np.save(f"{out_dir}/volume_idw.npy", idw_vol)
                print(f"Saved volumes to {out_dir}/volume_raw.npy, volume_idw.npy")

        writer.close()


def main():
    parser = configargparse.ArgParser()

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
