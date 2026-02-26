import os
import uuid
import yaml
import gc
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

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import CTScene
from voxelize import voxelize
from visualize_volume import visualize
from vis_foam import (load_density_field, field_from_model, query_density,
                      sample_idw, sample_idw_diagnostic,
                      visualize_idw_diagnostics, supersample_slice,
                      make_slice_coords, compute_cell_density_slice,
                      visualize_slices, load_gt_volume, sample_gt_slice)
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
        dataset_args, rays_per_batch=1_000_000, device=device
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

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    gt_volume = load_gt_volume(dataset_args.data_path, dataset_args.dataset)
    if gt_volume is not None:
        print(f"Loaded GT volume: shape={gt_volume.shape}")

    def eval_views(data_handler, ray_batch_fetcher, proj_batch_fetcher):
        rays = data_handler.rays
        points, _, _, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        rmse_list = []
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                proj_batch = proj_batch_fetcher.next()[0]
                proj_output, _, _, _ = model(ray_batch, start_points[i])

                mse = ((proj_output - proj_batch) ** 2).mean()
                rmse_list.append(torch.sqrt(mse).item())
                psnr_list.append(compute_psnr(proj_output, proj_batch))
                ssim_list.append(compute_ssim(proj_output, proj_batch))
                torch.cuda.synchronize()

        return {
            "rmse": sum(rmse_list) / len(rmse_list),
            "psnr": sum(psnr_list) / len(psnr_list),
            "ssim": sum(ssim_list) / len(ssim_list),
        }

    def train_loop(viewer):
        print("Training")

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        ray_batch, proj_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        with tqdm.trange(pipeline_args.iterations) as train:
            for i in train:
                proj_output, _, _, _ = model(ray_batch)

                loss = loss_fn(proj_output, proj_batch)

                if optimizer_args.tv_weight > 0 and i >= optimizer_args.tv_start:
                    if optimizer_args.tv_border:
                        tv_loss = model.tv_border_regularization(
                            epsilon=optimizer_args.tv_epsilon,
                            area_weighted=optimizer_args.tv_area_weighted,
                        )
                    else:
                        tv_loss = model.tv_regularization(
                            epsilon=optimizer_args.tv_epsilon,
                            area_weighted=optimizer_args.tv_area_weighted,
                        )
                    loss = loss + optimizer_args.tv_weight * tv_loss

                model.optimizer.zero_grad(set_to_none=True)

                # Hide latency of data loading behind the backward pass
                event = torch.cuda.Event()
                event.record()
                loss.backward()

                event.synchronize()
                ray_batch, proj_batch = next(data_iterator)

                model.optimizer.step()
                model.update_learning_rate(i)

                train.set_postfix(loss=f"{loss.item():.5f}")

                if i % 100 == 99 and not pipeline_args.debug:
                    writer.add_scalar("train/loss", loss.item(), i)
                    if optimizer_args.tv_weight > 0 and i >= optimizer_args.tv_start:
                        writer.add_scalar("train/tv_loss", tv_loss.item(), i)
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("train/num_points", num_points, i)

                    if hasattr(model, '_triangulation_retries'):
                        writer.add_scalar("diagnostics/triangulation_retries", model._triangulation_retries, i)

                    test_metrics = eval_views(
                        test_data_handler,
                        test_ray_batch_fetcher,
                        test_proj_batch_fetcher,
                    )
                    writer.add_scalar("test/rmse", test_metrics["rmse"], i)
                    writer.add_scalar("test/psnr", test_metrics["psnr"], i)
                    writer.add_scalar("test/ssim", test_metrics["ssim"], i)

                    train_metrics = eval_views(
                        train_data_handler,
                        train_ray_batch_fetcher,
                        train_proj_batch_fetcher,
                    )
                    writer.add_scalar("train/rmse", train_metrics["rmse"], i)
                    writer.add_scalar("train/psnr", train_metrics["psnr"], i)
                    writer.add_scalar("train/ssim", train_metrics["ssim"], i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )
                    if model.grad_scheduler_args is not None:
                        writer.add_scalar(
                            "lr/gradient_lr",
                            model.grad_scheduler_args(i - model._gradient_start),
                            i,
                        )

                if i % 1000 == 999 and not pipeline_args.debug:
                    log_diagnostics(model, writer, i)
                    with torch.no_grad():
                        field = field_from_model(model)
                        axes = [0, 1, 2]
                        slice_coords = [-0.2, 0.0, 0.2]
                        d_slices = []
                        idw_slices = []
                        cd_slices = []
                        gt_slices = []
                        for a in axes:
                            for c in slice_coords:
                                d_slices.append(supersample_slice(query_density, field, a, c, 256, 1.0, ss=2))
                                idw_slices.append(supersample_slice(sample_idw, field, a, c, 256, 1.0, ss=2))
                                cd_slices.append(
                                    compute_cell_density_slice(field["points"], a, c, 64, 1.0)
                                )
                                gt_slices.append(sample_gt_slice(gt_volume, a, c, 256, 1.0))
                        log_fig = partial(writer.add_figure, f"slices/{experiment_name}", global_step=i)
                        log_fig_il = partial(writer.add_figure, f"slices_interleaved/{experiment_name}", global_step=i)
                        metrics = visualize_slices(
                            d_slices, idw_slices, cd_slices,
                            gt_slices=gt_slices, writer_fn=log_fig,
                            writer_fn_interleaved=log_fig_il,
                        )
                        if metrics is not None:
                            for key, val in metrics.items():
                                tag = f"slice_{key.split('_')[1]}/{key.split('_')[0]}"
                                writer.add_scalar(tag, val, i)

                        # IDW diagnostic for a single Z=0 slice
                        diag_coords = make_slice_coords(axis=2, coord=0.0, resolution=256, extent=1.0)
                        diag = sample_idw_diagnostic(field, diag_coords)
                        diag_writer = partial(writer.add_figure, f"idw_diagnostics/{experiment_name}", global_step=i)
                        visualize_idw_diagnostics(diag, writer_fn=diag_writer)
                        n_holes = (diag["diff"] > 0.05).sum()
                        writer.add_scalar("diagnostics/idw_hole_pixels", n_holes, i)
                        writer.add_scalar("diagnostics/idw_mean_cell_weight", diag["cell_weight"].mean(), i)
                        writer.add_scalar("diagnostics/idw_mean_neighbor_count", diag["neighbor_count"].mean(), i)

                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler
                    )
                    model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                        pipeline_args.contrast_fraction,
                        pipeline_args.contrast_power,
                    )

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

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

                if (
                    optimizer_args.gradient_start >= 0
                    and i == optimizer_args.gradient_start
                ):
                    model.initialize_gradients(optimizer_args)

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

        if not pipeline_args.debug:
            model.save_ply(f"{out_dir}/scene.ply")
            model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    train_loop(viewer=None)

    test_metrics = eval_views(
        test_data_handler,
        test_ray_batch_fetcher,
        test_proj_batch_fetcher,
    )
    train_metrics = eval_views(
        train_data_handler,
        train_ray_batch_fetcher,
        train_proj_batch_fetcher,
    )

    if not pipeline_args.debug:
        with open(f"{out_dir}/metrics.txt", "w") as f:
            f.write(f"Test  RMSE: {test_metrics['rmse']:.6f}\n")
            f.write(f"Test  PSNR: {test_metrics['psnr']:.4f}\n")
            f.write(f"Test  SSIM: {test_metrics['ssim']:.6f}\n")
            f.write(f"Train RMSE: {train_metrics['rmse']:.6f}\n")
            f.write(f"Train PSNR: {train_metrics['psnr']:.4f}\n")
            f.write(f"Train SSIM: {train_metrics['ssim']:.6f}\n")

        model_path = f"{out_dir}/model.pt"

        # Direct slice evaluation (no full volume needed)
        field = load_density_field(model_path)
        axes = [0, 1, 2]
        coords = [-0.2, 0.0, 0.2]
        density_slices = []
        idw_slices = []
        cell_density_slices = []
        gt_slices_final = []
        for a in axes:
            for c in coords:
                density_slices.append(supersample_slice(query_density, field, a, c, 256, 1.0, ss=2))
                idw_slices.append(supersample_slice(sample_idw, field, a, c, 256, 1.0, ss=2))
                cell_density_slices.append(
                    compute_cell_density_slice(field["points"], a, c, 64, 1.0)
                )
                gt_slices_final.append(sample_gt_slice(gt_volume, a, c, 256, 1.0))

        log_fig = partial(writer.add_figure, f"slices/{experiment_name}", global_step=pipeline_args.iterations)
        log_fig_il = partial(writer.add_figure, f"slices_interleaved/{experiment_name}", global_step=pipeline_args.iterations)
        slice_metrics = visualize_slices(
            density_slices, idw_slices, cell_density_slices,
            gt_slices=gt_slices_final, writer_fn=log_fig,
            writer_fn_interleaved=log_fig_il,
            out_path=f"{out_dir}/vis.jpg",
        )
        if slice_metrics is not None:
            for key, val in slice_metrics.items():
                tag = f"slice_{key.split('_')[1]}/{key.split('_')[0]}"
                writer.add_scalar(tag, val, pipeline_args.iterations)
            with open(f"{out_dir}/metrics.txt", "a") as f:
                for key, val in slice_metrics.items():
                    f.write(f"Slice {key}: {val:.4f}\n")

        # Final IDW diagnostics
        diag_coords = make_slice_coords(axis=2, coord=0.0, resolution=256, extent=1.0)
        diag = sample_idw_diagnostic(field, diag_coords)
        diag_writer = partial(writer.add_figure, f"idw_diagnostics/{experiment_name}", global_step=pipeline_args.iterations)
        visualize_idw_diagnostics(diag, writer_fn=diag_writer,
                                  out_path=f"{out_dir}/idw_diagnostics.jpg")

        # Optionally save full volume
        if pipeline_args.save_volume:
            volume_path = f"{out_dir}/volume.npy"
            voxelize(model_path, resolution=256, output_path=volume_path, extent=1.0)

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
