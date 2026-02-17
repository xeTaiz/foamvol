import os
import uuid
import yaml
import gc
import numpy as np
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import CTScene
from voxelize import voxelize
from visualize_volume import visualize
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
        points = model.primal_points.detach()
        _, cell_radius = radfoam.farthest_neighbor(
            points,
            model.point_adjacency,
            model.point_adjacency_offsets,
        )
        density = model.get_primal_density().squeeze()

        # --- Scalar statistics ---
        inside_mask = (points.abs() <= 1.0).all(dim=-1)
        n_inside = inside_mask.sum().item()
        n_total = points.shape[0]
        writer.add_scalar("diagnostics/points_inside", n_inside, step)
        writer.add_scalar("diagnostics/points_outside", n_total - n_inside, step)
        writer.add_scalar("diagnostics/frac_inside", n_inside / n_total, step)

        writer.add_scalar("diagnostics/cell_radius_mean", cell_radius.mean().item(), step)
        writer.add_scalar("diagnostics/cell_radius_median", cell_radius.median().item(), step)
        writer.add_scalar("diagnostics/cell_radius_min", cell_radius.min().item(), step)
        writer.add_scalar("diagnostics/cell_radius_max", cell_radius.max().item(), step)

        writer.add_scalar("diagnostics/density_mean", density.mean().item(), step)
        writer.add_scalar("diagnostics/density_max", density.max().item(), step)

        # --- Histograms ---
        writer.add_histogram("diagnostics/cell_radius", cell_radius, step)
        writer.add_histogram("diagnostics/density", density, step)
        writer.add_histogram("diagnostics/pos_x", points[:, 0], step)
        writer.add_histogram("diagnostics/pos_y", points[:, 1], step)
        writer.add_histogram("diagnostics/pos_z", points[:, 2], step)

        # --- 2x2 figure ---
        pts_cpu = points.cpu().numpy()
        cr_cpu = cell_radius.cpu().numpy()
        n_total_pts = pts_cpu.shape[0]

        # Subsample for speed
        max_pts = 5000
        if n_total_pts > max_pts:
            idx = np.random.choice(n_total_pts, max_pts, replace=False)
            pts_sub = pts_cpu[idx]
            cr_sub = cr_cpu[idx]
        else:
            pts_sub = pts_cpu
            cr_sub = cr_cpu

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Top-left: XY scatter colored by cell radius
        ax = axes[0, 0]
        sc = ax.scatter(pts_sub[:, 0], pts_sub[:, 1], c=cr_sub, s=1, alpha=0.3,
                        cmap="coolwarm", rasterized=True)
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", linewidth=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("XY (color=cell radius)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # Top-right: XZ scatter colored by cell radius
        ax = axes[0, 1]
        sc = ax.scatter(pts_sub[:, 0], pts_sub[:, 2], c=cr_sub, s=1, alpha=0.3,
                        cmap="coolwarm", rasterized=True)
        ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", linewidth=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title("XZ (color=cell radius)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # Bottom-left: Radial density profile
        ax = axes[1, 0]
        dist = np.linalg.norm(pts_cpu, axis=1)
        bins = np.linspace(0, dist.max(), 50)
        counts, edges = np.histogram(dist, bins=bins)
        frac = counts / n_total_pts
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, frac, width=np.diff(edges), edgecolor="none", alpha=0.7)
        ax.set_xlabel("Distance from origin")
        ax.set_ylabel("Fraction of cells")
        ax.set_title("Radial cell density")

        # Bottom-right: Cell radius vs distance from origin
        ax = axes[1, 1]
        dist_sub = np.linalg.norm(pts_sub, axis=1)
        ax.scatter(dist_sub, cr_sub, s=1, alpha=0.3, rasterized=True)
        ax.set_xlabel("Distance from origin")
        ax.set_ylabel("Cell radius")
        ax.set_title("Cell radius vs distance")

        fig.suptitle(f"Cell diagnostics — step {step} — {n_total} points", fontsize=13)
        fig.tight_layout()
        writer.add_figure("diagnostics/cell_overview", fig, step)
        plt.close(fig)


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

    def eval_views(data_handler, ray_batch_fetcher, proj_batch_fetcher):
        rays = data_handler.rays
        points, _, _, _ = model.get_trace_data()
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
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("train/num_points", num_points, i)

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

                if i % 1000 == 999 and not pipeline_args.debug:
                    log_diagnostics(model, writer, i)

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

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

        if not pipeline_args.debug:
            model.save_ply(f"{out_dir}/scene.ply")
            model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

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
        volume_path = f"{out_dir}/volume.npy"
        voxelize(model_path, resolution=512, output_path=volume_path, extent=1.0)
        visualize(volume_path)


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
