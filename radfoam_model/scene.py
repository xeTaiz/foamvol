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
        self.init_density = getattr(args, "init_density", 0.0)

        if self.init_type == "regular":
            self.regular_initialize()
        else:
            self.random_initialize()

        self.pipeline = radfoam.create_ct_pipeline()

    def regular_initialize(self):
        s = self.init_scale
        pt_per_axis = int(self.num_init_points ** (1.0 / 3.0))
        ax = torch.linspace(-s, s, pt_per_axis, device=self.device)
        mg = torch.stack(torch.meshgrid([ax,ax,ax]), dim=-1).reshape(-1, 3)
        # Jitter to avoid coplanar/collinear degeneracies in triangulation
        spacing = 2 * s / pt_per_axis
        mg = mg + spacing * 1e-3 * torch.randn_like(mg)
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

        init_val = self.init_density
        density = torch.full(
            (self.num_init_points, 1), init_val, device=self.device, dtype=torch.float32
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
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]

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
        if on_raw:
            density = self.density.squeeze()  # raw params, no activation
        else:
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

    def tv_border_regularization(self, epsilon=1e-3, area_weighted=False, on_raw=False):
        """Charbonnier TV on density evaluated at Voronoi cell borders."""
        if on_raw:
            mu_base = self.density.squeeze()  # raw params, no activation
        else:
            mu_base = self.get_primal_density().squeeze()  # (N,) activated density
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = mu_base.shape[0]
        points = self.primal_points

        counts = offsets[1:] - offsets[:-1]
        source = torch.repeat_interleave(
            torch.arange(N, device=mu_base.device), counts
        )

        # Displacement from source center to midpoint: 0.5 * (p_j - p_i)
        dx = 0.5 * (points[adj] - points[source])  # (E, 3)

        has_grad = hasattr(self, "density_grad") and self.density_grad is not None
        if has_grad:
            max_slope = getattr(self, "_gradient_max_slope", 5.0)
            slope_i = max_slope * torch.tanh(self.density_grad[source])  # (E, 3)
            slope_j = max_slope * torch.tanh(self.density_grad[adj])     # (E, 3)
            mu_i = (mu_base[source] + (slope_i * dx).sum(dim=-1)).clamp(min=0)
            mu_j = (mu_base[adj] + (slope_j * (-dx)).sum(dim=-1)).clamp(min=0)
        else:
            # No gradient active — falls back to center values
            mu_i = mu_base[source]
            mu_j = mu_base[adj]

        diff = mu_i - mu_j
        edge_loss = torch.sqrt(diff ** 2 + epsilon ** 2) - epsilon

        if area_weighted:
            with torch.no_grad():
                _, cell_radius = radfoam.farthest_neighbor(
                    points, self.point_adjacency, self.point_adjacency_offsets,
                )
                cr = cell_radius.squeeze()
                w = cr[source] * cr[adj]
                w = w / w.sum()
            return (w * edge_loss).sum()

        return edge_loss.mean()

    @torch.no_grad()
    def compute_redundancy_error(self, cell_radius, sigma_scale, sigma_v):
        """Per-cell leave-one-out IDW error: |density_i - interp_from_neighbors|."""
        activated = self.get_primal_density().squeeze()  # (N,)
        points = self.primal_points                       # (N, 3)
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = points.shape[0]

        sigma = sigma_scale * cell_radius.median().item()
        sigma_sq = sigma ** 2

        counts = offsets[1:] - offsets[:-1]
        source = torch.repeat_interleave(
            torch.arange(N, device=points.device), counts
        )

        # Gaussian spatial weight
        d_sq = (points[adj] - points[source]).pow(2).sum(dim=-1)
        # Gaussian bilateral weight (density similarity)
        dmu = activated[source] - activated[adj]
        w = torch.exp(-d_sq / sigma_sq - dmu * dmu / (sigma_v * sigma_v))

        # Per-cell weighted sum
        w_sum = torch.zeros(N, device=points.device).scatter_add_(0, source, w)
        w_mu_sum = torch.zeros(N, device=points.device).scatter_add_(
            0, source, w * activated[adj]
        )

        interp = w_mu_sum / w_sum.clamp(min=1e-8)
        return (activated - interp).abs()

    def set_interpolation_mode(self, enabled, sigma=None, sigma_v=None,
                               per_cell_sigma=None, per_neighbor_sigma=None):
        self._interpolation_mode = enabled
        if sigma is not None:
            self._idw_sigma = sigma
        if sigma_v is not None:
            self._idw_sigma_v = sigma_v
        if per_cell_sigma is not None:
            self._per_cell_sigma = per_cell_sigma
        if per_neighbor_sigma is not None:
            self._per_neighbor_sigma = per_neighbor_sigma

    def get_trace_data(self):
        points = self.primal_points
        density = self.density  # raw — kernel applies softplus
        point_adjacency = self.point_adjacency
        point_adjacency_offsets = self.point_adjacency_offsets
        density_grad = getattr(self, "density_grad", None)
        gradient_max_slope = getattr(self, "_gradient_max_slope", 5.0)

        return points, density, point_adjacency, point_adjacency_offsets, density_grad, gradient_max_slope

    @torch.no_grad()
    def _get_cell_radius(self):
        """Compute per-cell radius (cached until triangulation changes)."""
        _, cell_radius = radfoam.farthest_neighbor(
            self.primal_points,
            self.point_adjacency,
            self.point_adjacency_offsets,
        )
        return cell_radius.squeeze()

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
        points, density, point_adjacency, point_adjacency_offsets, density_grad, gradient_max_slope = (
            self.get_trace_data()
        )

        interpolation_mode = getattr(self, "_interpolation_mode", False)
        idw_sigma = getattr(self, "_idw_sigma", 0.01)
        idw_sigma_v = getattr(self, "_idw_sigma_v", 0.1)
        per_cell_sigma = getattr(self, "_per_cell_sigma", False)
        per_neighbor_sigma = getattr(self, "_per_neighbor_sigma", False)

        # Compute cell_radius on demand when adaptive sigma is active
        cell_radius = None
        if interpolation_mode and (per_cell_sigma or per_neighbor_sigma):
            cell_radius = self._get_cell_radius()

        # When interpolation is active, suppress the linear gradient feature
        if interpolation_mode:
            density_grad = None

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
            density_grad,
            gradient_max_slope,
            interpolation_mode,
            idw_sigma,
            idw_sigma_v,
            per_cell_sigma,
            per_neighbor_sigma,
            cell_radius,
        )

    def declare_optimizer(self, args, warmup, max_iterations):
        self._optimizer_args = args
        self._max_iterations = max_iterations
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
        self.grad_scheduler_args = None

    def initialize_gradients(self, args):
        N = self.primal_points.shape[0]
        self.density_grad = nn.Parameter(
            torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        )
        self.optimizer.add_param_group({
            "params": self.density_grad,
            "lr": args.gradient_lr_init,
            "name": "density_grad",
        })
        self.grad_scheduler_args = get_cosine_lr_func(
            lr_init=args.gradient_lr_init,
            lr_final=args.gradient_lr_final,
            warmup_steps=args.gradient_warmup,
            max_steps=self._max_iterations - args.gradient_start,
        )
        self._gradient_start = args.gradient_start
        self._gradient_freeze_points_until = args.gradient_start + args.gradient_freeze_points
        self._gradient_max_slope = args.gradient_max_slope
        print(f"Initialized density_grad: {N} x 3 "
              f"(warmup={args.gradient_warmup}, freeze_points={args.gradient_freeze_points}, "
              f"max_slope={args.gradient_max_slope})")

    def update_learning_rate(self, iteration):
        # Freeze positions while density gradients stabilize
        freeze_for_grad = (
            hasattr(self, "_gradient_freeze_points_until")
            and iteration < self._gradient_freeze_points_until
        )
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                if freeze_for_grad:
                    param_group["lr"] = 0.0
                else:
                    param_group["lr"] = self.xyz_scheduler_args(iteration)
            elif param_group["name"] == "density":
                lr = self.den_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density_grad":
                if self.grad_scheduler_args is not None:
                    lr = self.grad_scheduler_args(
                        iteration - self._gradient_start
                    )
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
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]

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
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]

    def prune_and_densify(
        self, point_error, point_contribution, upsample_factor=1.2,
        contrast_fraction=0.5,
        redundancy_threshold=0.0, redundancy_cap=0.0,
        sigma_scale=0.5, sigma_v=0.1,
    ):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            primal_error_accum = point_error.clip(min=0).squeeze()
            points, _, point_adjacency, point_adjacency_offsets, _, _ = (
                self.get_trace_data()
            )
            ################### Farthest neighbor ###################
            farthest_neighbor, cell_radius = radfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            ################### Edge weights ###################
            activated = self.get_primal_density().squeeze()  # [N]
            offsets = point_adjacency_offsets.long()
            adj = point_adjacency.long()

            counts = offsets[1:] - offsets[:-1]
            source = torch.repeat_interleave(
                torch.arange(num_curr_points, device=points.device), counts
            )

            # Deduplicate: keep only edges where source < target
            edge_mask = source < adj
            src = source[edge_mask]
            tgt = adj[edge_mask]

            edge_vec = points[src] - points[tgt]
            edge_length = edge_vec.norm(dim=-1)

            # Per-cell bilateral prediction error as interface score
            cell_error = self.compute_redundancy_error(cell_radius, sigma_scale, sigma_v)
            contrast_weight = (cell_error[src] + cell_error[tgt]) * edge_length

            ######################## Pruning ########################
            low_contrib = point_contribution.squeeze() < 1e-2
            tiny_radius = cell_radius < 1e-4
            prune_mask = low_contrib | tiny_radius
            n_pruned_low_contrib = (low_contrib & ~tiny_radius).sum().item()
            n_pruned_tiny_radius = (tiny_radius & ~low_contrib).sum().item()
            n_pruned_both = (low_contrib & tiny_radius).sum().item()
            n_redundant = 0
            n_added_gradient = 0
            n_added_contrast = 0
            n_filtered_dupes = 0
            n_basic_pruned = prune_mask.sum().item()
            if n_basic_pruned > 0:
                print(f"Pruning {n_basic_pruned}/{num_curr_points} cells "
                      f"(low_contrib={n_pruned_low_contrib}, tiny_radius={n_pruned_tiny_radius}, both={n_pruned_both})")

            ################ Redundancy pruning ################
            if redundancy_cap > 0:
                density_scale = torch.quantile(activated, 0.95).item()
                candidates = cell_error < redundancy_threshold * density_scale
                # Don't re-mark cells already pruned
                candidates = candidates & ~prune_mask

                if candidates.sum() > 0:
                    # Independent set: error-based priority (most redundant neighbor wins)
                    priorities = cell_error.clone()
                    priorities[~candidates] = float('inf')
                    neighbor_min = torch.full(
                        (num_curr_points,), float('inf'), device=points.device
                    ).scatter_reduce_(0, source, priorities[adj], reduce='amin')
                    removable = candidates & (priorities < neighbor_min)

                    # Cap: at most redundancy_cap fraction of total cells
                    max_remove = int(redundancy_cap * num_curr_points)
                    n_removable = removable.sum().item()
                    if n_removable > max_remove:
                        err_vals = cell_error.clone()
                        err_vals[~removable] = float('inf')
                        _, topk = err_vals.topk(max_remove, largest=False)
                        removable = torch.zeros_like(removable)
                        removable[topk] = True

                    n_redundant_here = removable.sum().item()
                    if n_redundant_here > 0:
                        n_redundant = n_redundant_here
                        print(f"Redundancy prune: {n_redundant}/{num_curr_points} cells "
                              f"(threshold={redundancy_threshold * density_scale:.4f})")
                        prune_mask = prune_mask | removable

            ######################## Sampling ########################
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
            sampled_density_list = []
            sampled_density_grad_list = []
            has_density_grad = hasattr(self, "density_grad") and self.density_grad is not None

            # --- Gradient-based sampling (existing strategy, reduced budget) ---
            if num_gradient_points > 0:
                grad_inds = torch.multinomial(
                    primal_error_accum * cell_radius,
                    num_gradient_points,
                    replacement=False,
                )
                sampled_points_list.append((points + perturbation)[grad_inds])
                sampled_inds_list.append(grad_inds)
                sampled_density_list.append(self.density[grad_inds])
                if has_density_grad:
                    sampled_density_grad_list.append(self.density_grad[grad_inds])
                n_added_gradient += num_gradient_points

            # --- Contrast-based sampling (edge-based strategy) ---
            if num_contrast_points > 0:
                num_viable = (contrast_weight > 0).sum().item()
                if num_viable == 0:
                    # Fallback: redirect budget to gradient strategy
                    if num_gradient_points > 0:
                        extra_inds = torch.multinomial(
                            primal_error_accum * cell_radius,
                            num_contrast_points,
                            replacement=False,
                        )
                        sampled_points_list.append((points + perturbation)[extra_inds])
                        sampled_inds_list.append(extra_inds)
                        sampled_density_list.append(self.density[extra_inds])
                        if has_density_grad:
                            sampled_density_grad_list.append(self.density_grad[extra_inds])
                        n_added_gradient += num_contrast_points
                else:
                    n_sample = min(num_contrast_points, num_viable)
                    edge_inds = torch.multinomial(
                        contrast_weight, n_sample, replacement=False,
                    )
                    n_added_contrast += n_sample
                    # Radius-ratio placement: bias towards the larger cell
                    p_a = points[src[edge_inds]]
                    p_b = points[tgt[edge_inds]]
                    r_a = cell_radius[src[edge_inds]].squeeze(-1)
                    r_b = cell_radius[tgt[edge_inds]].squeeze(-1)
                    t = r_b / (r_a + r_b + 1e-12)  # closer to A when A is larger
                    edge_vec = p_b - p_a
                    edge_len = edge_vec.norm(dim=-1, keepdim=True)
                    # Jitter: 10% of edge length to avoid co-spherical degeneracies
                    jitter = 0.10 * edge_len * torch.randn_like(p_a)
                    new_points = p_a + t.unsqueeze(-1) * edge_vec + jitter
                    avg_density = 0.5 * (
                        self.density[src[edge_inds]] + self.density[tgt[edge_inds]]
                    )
                    sampled_points_list.append(new_points)
                    sampled_inds_list.append(src[edge_inds])
                    sampled_density_list.append(avg_density)
                    if has_density_grad:
                        sampled_density_grad_list.append(
                            torch.zeros(n_sample, 3, device=self.device)
                        )

            sampled_inds = torch.cat(sampled_inds_list, dim=0)
            sampled_points = torch.cat(sampled_points_list, dim=0)
            sampled_density = torch.cat(sampled_density_list, dim=0)
            if has_density_grad:
                sampled_dg = torch.cat(sampled_density_grad_list, dim=0)

            ################### Filter near-duplicates ###################
            nn_inds = radfoam.nn(points, self.aabb_tree, sampled_points).long()
            nn_dists = (sampled_points - points[nn_inds]).norm(dim=-1)
            # Minimum separation: 5% of the source point's cell radius
            min_sep = 0.05 * cell_radius[sampled_inds].squeeze()
            keep_mask = nn_dists > min_sep

            n_filtered_dupes = (~keep_mask).sum().item()
            if n_filtered_dupes > 0:
                print(f"Filtered {n_filtered_dupes}/{sampled_points.shape[0]} new points (too close to existing)")
                sampled_points = sampled_points[keep_mask]
                sampled_inds = sampled_inds[keep_mask]
                sampled_density = sampled_density[keep_mask]
                if has_density_grad:
                    sampled_dg = sampled_dg[keep_mask]

            new_params = {
                "primal_points": sampled_points,
                "density": sampled_density,
            }
            if has_density_grad:
                new_params["density_grad"] = sampled_dg

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

            return {
                "points_before": num_curr_points,
                "pruned_low_contrib": n_pruned_low_contrib,
                "pruned_tiny_radius": n_pruned_tiny_radius,
                "pruned_both": n_pruned_both,
                "pruned_redundancy": n_redundant,
                "added_gradient": n_added_gradient,
                "added_contrast": n_added_contrast,
                "filtered_duplicates": n_filtered_dupes,
                "points_after": self.primal_points.shape[0],
            }

    def prune_only(self, data_handler):
        """Standalone prune pass: remove cells with negligible contribution or tiny radius."""
        _, point_contribution = self.collect_error_map(data_handler)
        with torch.no_grad():
            points, _, point_adjacency, point_adjacency_offsets, _, _ = self.get_trace_data()
            _, cell_radius = radfoam.farthest_neighbor(
                points, point_adjacency, point_adjacency_offsets,
            )
            prune_mask = torch.logical_or(
                point_contribution.squeeze() < 1e-2, cell_radius < 1e-3
            )
            n_pruned = prune_mask.sum().item()
            if n_pruned > 0:
                print(f"Standalone prune: {n_pruned}/{points.shape[0]} cells")
                self.prune_points(prune_mask)
                self.update_triangulation(incremental=False)
            return n_pruned

    def collect_error_map(self, data_handler):
        rays, projections = data_handler.rays, data_handler.projections

        points, _, _, _, _, _ = self.get_trace_data()
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

        has_grad = hasattr(self, "density_grad") and self.density_grad is not None
        if has_grad:
            dg = self.density_grad.detach().float().cpu().numpy()

        vertex_data = []
        for i in tqdm.trange(points.shape[0]):
            row = (
                points[i, 0],
                points[i, 1],
                points[i, 2],
                density[i, 0],
                adjacency_offsets[i + 1],
            )
            if has_grad:
                row = row + (dg[i, 0], dg[i, 1], dg[i, 2])
            vertex_data.append(row)

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("density", np.float32),
            ("adjacency_offset", np.uint32),
        ]
        if has_grad:
            dtype += [
                ("grad_x", np.float32),
                ("grad_y", np.float32),
                ("grad_z", np.float32),
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
        if hasattr(self, "density_grad") and self.density_grad is not None:
            scene_data["density_grad"] = self.density_grad.detach().float().cpu()
            scene_data["gradient_max_slope"] = getattr(self, "_gradient_max_slope", 5.0)
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        scene_data = torch.load(pt_path)

        self.primal_points = nn.Parameter(scene_data["xyz"].to(self.device))
        self.density = nn.Parameter(scene_data["density"].to(self.device))

        if "density_grad" in scene_data:
            self.density_grad = nn.Parameter(
                scene_data["density_grad"].to(self.device)
            )
            self._gradient_max_slope = scene_data.get("gradient_max_slope", 5.0)

        self.point_adjacency = scene_data["adjacency"].to(self.device).to(
            torch.uint32)
        self.point_adjacency_offsets = scene_data["adjacency_offsets"].to(
            self.device
        ).to(torch.uint32)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)
