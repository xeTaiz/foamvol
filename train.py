import os
import uuid
import yaml
import gc
import numpy as np
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import CTScene
from radfoam_model.utils import psnr
from voxelize import voxelize
from visualize_volume import visualize
import radfoam


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


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

    def test_render(test_data_handler, ray_batch_fetcher, proj_batch_fetcher, debug=False):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        rmse_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                proj_batch = proj_batch_fetcher.next()[0]
                proj_output, _, _, _ = model(ray_batch, start_points[i])

                mse = ((proj_output - proj_batch) ** 2).mean()
                rmse = torch.sqrt(mse)
                rmse_list.append(rmse.item())
                torch.cuda.synchronize()

        average_rmse = sum(rmse_list) / len(rmse_list)
        if not debug and not pipeline_args.debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average RMSE: {average_rmse:.6f}")
            f.close()

        return average_rmse

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
                    writer.add_scalar("test/num_points", num_points, i)

                    test_rmse = test_render(
                        test_data_handler,
                        test_ray_batch_fetcher,
                        test_proj_batch_fetcher,
                        True,
                    )
                    writer.add_scalar("test/rmse", test_rmse, i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )

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

    test_render(
        test_data_handler,
        test_ray_batch_fetcher,
        test_proj_batch_fetcher,
        pipeline_args.debug,
    )

    if not pipeline_args.debug:
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
