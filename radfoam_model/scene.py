import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
import tqdm

import radfoam
from radfoam_model.render import TraceRays
from radfoam_model.utils import *


class CTScene(torch.nn.Module):

    def __init__(
        self,
        args,
        device=torch.device("cuda"),
    ):
        super().__init__()

        self.device = device
        self.num_init_points = args.init_points
        self.num_final_points = args.final_points
        self.activation_scale = args.activation_scale
        self.init_scale = getattr(args, "init_scale", 1.1)
        self.init_type = getattr(args, "init_type", "random")

        if self.init_type == "regular":
            self.regular_initialize()
        else:
            self.random_initialize()

        self.pipeline = radfoam.create_ct_pipeline()

    def regular_initialize(self):
        s = self.init_scale
        pt_per_axis = int(round(self.num_init_points ** (1.0 / 3.0)))
        ax = torch.linspace(-s, s, pt_per_axis, device=self.device)
        mg = torch.stack(torch.meshgrid([ax,ax,ax]), dim=-1).reshape(-1, 3)
        print(mg.shape, mg.min(), mg.max())
        if mg.size(0) < self.num_init_points:
            mg = torch.cat([mg, torch.rand(self.num_init_points - mg.size(0), 3, device=self.device) * 2 * s - s], dim=0)
        print(mg.shape, mg.min(), mg.max())
        self.triangulation = radfoam.Triangulation(mg.float().contiguous())
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = mg[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)
        density = torch.zeros(mg.size(0), 1, device=self.device, dtype=torch.float32)
        self.density = nn.Parameter(density[perm])

    def random_initialize(self):
        s = self.init_scale
        primal_points = (
            torch.rand(self.num_init_points, 3, device=self.device) * 2 * s - s
        )
        print(primal_points.shape, primal_points.dtype, primal_points.min(), primal_points.max())
        self.triangulation = radfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        density = torch.zeros(
            self.num_init_points, 1, device=self.device, dtype=torch.float32
        )
        self.density = nn.Parameter(density[perm])

    def permute_points(self, permutation):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(
                group["params"][0], None
            )
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][
                    permutation
                ]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][
                    permutation
                ]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][permutation].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][permutation].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]

    def update_triangulation(self, rebuild=True, incremental=False):
        if not self.primal_points.isfinite().all():
            raise RuntimeError("NaN in points")

        needs_permute = False
        del_points = self.primal_points
        failures = 0
        while rebuild:
            if failures > 10:
                raise RuntimeError("aborted triangulation after 10 attempts")
            try:
                needs_permute = self.triangulation.rebuild(
                    del_points, incremental=incremental
                )
                break
            except radfoam.TriangulationFailedError as e:
                print("caught: ", e)
                failures += 1
                incremental = False
                # Adaptive perturbation: scale relative to point cloud extent
                extent = self.primal_points.abs().max().item()
                perturbation = extent * 1e-5 * (3.0 ** failures)
                with torch.no_grad():
                    del_points = (
                        self.primal_points
                        + perturbation * torch.randn_like(self.primal_points)
                    )

        self._triangulation_retries = failures

        if failures > 3:
            with torch.no_grad():
                self.primal_points.copy_(del_points)

        if needs_permute:
            perm = self.triangulation.permutation().to(torch.long)
            self.permute_points(perm)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)

        self.point_adjacency = self.triangulation.point_adjacency()
        self.point_adjacency_offsets = (
            self.triangulation.point_adjacency_offsets()
        )

    def get_primal_density(self):
        return self.activation_scale * F.softplus(self.density, beta=10)

    def tv_regularization(self, epsilon=1e-3, area_weighted=False):
        """Charbonnier (smooth L1) TV loss over Voronoi neighbor edges."""
        density = self.get_primal_density().squeeze()  # (N,)
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = density.shape[0]

        counts = offsets[1:] - offsets[:-1]
        source = torch.repeat_interleave(
            torch.arange(N, device=density.device), counts
        )

        diff = density[source] - density[adj]
        edge_loss = torch.sqrt(diff ** 2 + epsilon ** 2) - epsilon

        if area_weighted:
            with torch.no_grad():
                _, cell_radius = radfoam.farthest_neighbor(
                    self.primal_points,
                    self.point_adjacency,
                    self.point_adjacency_offsets,
                )
                cr = cell_radius.squeeze()
                w = cr[source] * cr[adj]
                w = w / w.sum()
            return (w * edge_loss).sum()

        return edge_loss.mean()

    def get_trace_data(self):
        points = self.primal_points
        density = self.get_primal_density()
        point_adjacency = self.point_adjacency
        point_adjacency_offsets = self.point_adjacency_offsets

        return points, density, point_adjacency, point_adjacency_offsets

    def get_starting_point(self, rays, points, aabb_tree):
        with torch.no_grad():
            camera_origins = rays[..., :3]
            unique_cameras, inverse_indices = torch.unique(
                camera_origins, dim=0, return_inverse=True
            )

            nn_inds = radfoam.nn(points, aabb_tree, unique_cameras).long()

            start_point = nn_inds[inverse_indices]
            return start_point.type(torch.uint32)

    def forward(
        self,
        rays,
        start_point=None,
        return_contribution=False,
    ):
        points, density, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data()
        )

        if start_point is None:
            start_point = self.get_starting_point(rays, points, self.aabb_tree)
        else:
            start_point = torch.broadcast_to(start_point, rays.shape[:-1])
        return TraceRays.apply(
            self.pipeline,
            points,
            density,
            point_adjacency,
            point_adjacency_offsets,
            rays,
            start_point,
            return_contribution,
        )

    def declare_optimizer(self, args, warmup, max_iterations):
        params = [
            {
                "params": self.primal_points,
                "lr": args.points_lr_init,
                "name": "primal_points",
            },
            {
                "params": self.density,
                "lr": args.density_lr_init,
                "name": "density",
            },
        ]

        self.optimizer = torch.optim.Adam(params, eps=1e-15)
        self.xyz_scheduler_args = get_cosine_lr_func(
            lr_init=args.points_lr_init,
            lr_final=args.points_lr_final,
            max_steps=args.freeze_points,
        )
        self.den_scheduler_args = get_cosine_lr_func(
            lr_init=args.density_lr_init,
            lr_final=args.density_lr_final,
            warmup_steps=warmup,
            max_steps=max_iterations,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density":
                lr = self.den_scheduler_args(iteration)
                param_group["lr"] = lr

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, prune_mask):
        valid_points_mask = ~prune_mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]

    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in new_params.keys():
                assert len(group["params"]) == 1
                stored_tensor = group["params"][0]
                extension_tensor = new_params[group["name"]]
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_params):
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]

    def prune_and_densify(
        self, point_error, point_contribution, upsample_factor=1.2, contrast_fraction=0.5
    ):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            primal_error_accum = point_error.clip(min=0).squeeze()
            points, _, point_adjacency, point_adjacency_offsets = (
                self.get_trace_data()
            )
            ################### Farthest neighbor ###################
            farthest_neighbor, cell_radius = radfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            ################### Edge contrast ###################
            activated = self.get_primal_density().squeeze()  # [N]
            offsets = point_adjacency_offsets.long()
            adj = point_adjacency.long()

            # Build source index for each edge
            counts = offsets[1:] - offsets[:-1]
            source = torch.repeat_interleave(
                torch.arange(num_curr_points, device=points.device), counts
            )

            # Per-edge contrast
            edge_contrast = (activated[source] - activated[adj]).abs()

            # Per-point max contrast
            max_contrast = torch.zeros(num_curr_points, device=points.device)
            max_contrast.scatter_reduce_(0, source, edge_contrast, reduce="amax", include_self=False)

            # Per-point argmax neighbor (which neighbor achieves the max contrast)
            max_contrast_neighbor = torch.zeros(num_curr_points, dtype=torch.long, device=points.device)
            is_max = edge_contrast == max_contrast[source]
            max_contrast_neighbor[source[is_max]] = adj[is_max]

            ######################## Pruning ########################
            self_mask = point_contribution > 1e-2
            neighbor_mask = self_mask.long()[point_adjacency.long()]
            neighbor_mask = torch.cat(
                [neighbor_mask, torch.zeros_like(neighbor_mask[:1])], dim=0
            )
            nsum = torch.cumsum(neighbor_mask, dim=0)

            n_masked_adj = nsum[offsets[1:]] - nsum[offsets[:-1]]

            contrib_mask = ((n_masked_adj == 0) & ~self_mask).squeeze()
            cell_size_mask = cell_radius < 1e-1
            prune_mask = contrib_mask * cell_size_mask

            ######################## Sampling ########################
            primal_contribution_accum = point_contribution.squeeze()
            mask = primal_contribution_accum < 1e-3
            self.density[mask] = -1

            perturbation = 0.25 * (points[farthest_neighbor] - points)
            delta = torch.randn_like(perturbation)
            delta /= delta.norm(dim=-1, keepdim=True)
            perturbation += (
                0.1 * perturbation.norm(dim=-1, keepdim=True) * delta
            )

            ################### Split budget ########################
            num_contrast_points = int(contrast_fraction * num_new_points)
            num_gradient_points = num_new_points - num_contrast_points

            sampled_points_list = []
            sampled_inds_list = []

            # --- Gradient-based sampling (existing strategy, reduced budget) ---
            if num_gradient_points > 0:
                grad_inds = torch.multinomial(
                    primal_error_accum * cell_radius,
                    num_gradient_points,
                    replacement=False,
                )
                sampled_points_list.append((points + perturbation)[grad_inds])
                sampled_inds_list.append(grad_inds)

            # --- Contrast-based sampling (new strategy) ---
            if num_contrast_points > 0:
                # Distance to the max-contrast neighbor
                max_contrast_dist = (points - points[max_contrast_neighbor]).norm(dim=-1)
                # Weight by contrast, cell size, AND neighbor distance —
                # deprioritize edges that are too short to warrant a new point
                contrast_weight = max_contrast * cell_radius.squeeze() * max_contrast_dist
                # Hard floor: zero out pairs closer than 1e-3
                contrast_weight[max_contrast_dist < 1e-3] = 0.0
                # Fallback: if all contrasts are zero, redirect budget to gradient strategy
                if contrast_weight.sum() < 1e-10:
                    if num_gradient_points > 0:
                        extra_inds = torch.multinomial(
                            primal_error_accum * cell_radius,
                            num_contrast_points,
                            replacement=False,
                        )
                        sampled_points_list.append((points + perturbation)[extra_inds])
                        sampled_inds_list.append(extra_inds)
                else:
                    contrast_inds = torch.multinomial(
                        contrast_weight,
                        num_contrast_points,
                        replacement=False,
                    )
                    # Place at midpoint between point and its max-contrast neighbor
                    midpoints = 0.5 * (points[contrast_inds] + points[max_contrast_neighbor[contrast_inds]])
                    sampled_points_list.append(midpoints)
                    sampled_inds_list.append(contrast_inds)

            sampled_inds = torch.cat(sampled_inds_list, dim=0)
            sampled_points = torch.cat(sampled_points_list, dim=0)

            ################### Filter near-duplicates ###################
            nn_inds = radfoam.nn(points, self.aabb_tree, sampled_points).long()
            nn_dists = (sampled_points - points[nn_inds]).norm(dim=-1)
            # Minimum separation: 2% of the source point's cell radius
            min_sep = 0.02 * cell_radius[sampled_inds].squeeze()
            keep_mask = nn_dists > min_sep

            if not keep_mask.all():
                n_filtered = (~keep_mask).sum().item()
                print(f"Filtered {n_filtered}/{sampled_points.shape[0]} new points (too close to existing)")
                sampled_points = sampled_points[keep_mask]
                sampled_inds = sampled_inds[keep_mask]

            new_params = {
                "primal_points": sampled_points,
                "density": self.density[sampled_inds],
            }

            prune_mask = torch.cat(
                (
                    prune_mask,
                    torch.zeros(
                        sampled_points.shape[0],
                        device=prune_mask.device,
                        dtype=bool,
                    ),
                )
            )

            self.densification_postfix(new_params)
            self.prune_points(prune_mask)

    def collect_error_map(self, data_handler, downsample=2):
        rays, projections = data_handler.rays, data_handler.projections

        points, _, _, _ = self.get_trace_data()
        start_points = self.get_starting_point(
            rays[:, 0, 0].cuda(), points, self.aabb_tree
        )

        ray_batch_fetcher = radfoam.BatchFetcher(
            rays, batch_size=1, shuffle=False
        )
        proj_batch_fetcher = radfoam.BatchFetcher(
            projections, batch_size=1, shuffle=False
        )

        point_error_accum = torch.zeros_like(self.primal_points[..., 0:1])
        point_contribution_accum = torch.zeros_like(
            self.primal_points[..., 0:1]
        )
        proj_loss = nn.L1Loss(reduction="none")

        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()
            proj_batch = proj_batch_fetcher.next()

            d = torch.randint(0, downsample, (2,))
            ray_batch = ray_batch[:, d[0] :: downsample, d[1] :: downsample, :]
            proj_batch = proj_batch[:, d[0] :: downsample, d[1] :: downsample, :]

            proj_output, contribution, _, errbox = self.forward(
                ray_batch, start_points[i], return_contribution=True
            )

            loss = proj_loss(proj_batch, proj_output).mean(dim=-1)

            loss.sum().backward()
            point_error_accum += self.primal_points.grad.norm(
                dim=-1, keepdim=True
            ).detach()
            point_contribution_accum = torch.maximum(
                point_contribution_accum, contribution.detach()
            )
            torch.cuda.synchronize()

            self.optimizer.zero_grad(set_to_none=True)

        return point_error_accum, point_contribution_accum

    def save_ply(self, ply_path):
        points = self.primal_points.detach().float().cpu().numpy()
        density = self.get_primal_density().detach().float().cpu().numpy()
        adjacency = self.point_adjacency.cpu().numpy()
        adjacency_offsets = self.point_adjacency_offsets.cpu().numpy()

        vertex_data = []
        for i in tqdm.trange(points.shape[0]):
            vertex_data.append(
                (
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    density[i, 0],
                    adjacency_offsets[i + 1],
                )
            )

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("density", np.float32),
            ("adjacency_offset", np.uint32),
        ]

        vertex_data = np.array(vertex_data, dtype=dtype)
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        adjacency_data = np.array(adjacency, dtype=[("adjacency", np.uint32)])
        adjacency_element = PlyElement.describe(adjacency_data, "adjacency")

        PlyData([vertex_element, adjacency_element]).write(ply_path)

    def save_pt(self, pt_path):
        points = self.primal_points.detach().float().cpu()
        density = self.density.detach().float().cpu()
        adjacency = self.point_adjacency.cpu()
        adjacency_offsets = self.point_adjacency_offsets.cpu()

        scene_data = {
            "xyz": points,
            "density": density,
            "adjacency": adjacency.long(),
            "adjacency_offsets": adjacency_offsets.long(),
        }
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        scene_data = torch.load(pt_path)

        self.primal_points = nn.Parameter(scene_data["xyz"].to(self.device))
        self.density = nn.Parameter(scene_data["density"].to(self.device))

        self.point_adjacency = scene_data["adjacency"].to(self.device).to(
            torch.uint32)
        self.point_adjacency_offsets = scene_data["adjacency_offsets"].to(
            self.device
        ).to(torch.uint32)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)
