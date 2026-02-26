import os

import numpy as np
import einops
import torch

import radfoam

from .ct_synthetic import CTSyntheticDataset
from .r2_gaussian import R2GaussianDataset


def jitter_rays_parallel(rays, pixel_size):
    """Perturb parallel-beam ray origins by uniform random offset within one pixel."""
    dirs = rays[..., 3:6]
    # Perpendicular axis in XY plane
    perp = torch.stack([-dirs[..., 1], dirs[..., 0], torch.zeros_like(dirs[..., 0])], dim=-1)
    vert = torch.zeros_like(dirs)
    vert[..., 2] = 1.0
    ju = (torch.rand(rays.shape[:-1], device=rays.device) - 0.5) * pixel_size
    jv = (torch.rand(rays.shape[:-1], device=rays.device) - 0.5) * pixel_size
    jittered = rays.clone()
    jittered[..., :3] += ju.unsqueeze(-1) * perp + jv.unsqueeze(-1) * vert
    return jittered


def jitter_rays_cone(rays, pixel_ang_size):
    """Perturb cone-beam ray directions by uniform random offset within one pixel's angular extent."""
    dirs = rays[..., 3:6]
    # Build a local frame: "right" and "up" perpendicular to each ray direction
    # Use world-up (0,0,1) as reference; fall back to (1,0,0) for vertical rays
    world_up = torch.zeros_like(dirs)
    world_up[..., 2] = 1.0
    right = torch.cross(dirs, world_up, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True)
    # Handle degenerate case (ray parallel to world-up)
    fallback = torch.zeros_like(dirs)
    fallback[..., 0] = 1.0
    right = torch.where(right_norm > 1e-6, right, torch.cross(dirs, fallback, dim=-1))
    right = right / right.norm(dim=-1, keepdim=True)
    up = torch.cross(right, dirs, dim=-1)
    up = up / up.norm(dim=-1, keepdim=True)
    # Uniform angular jitter within pixel bounds
    ju = (torch.rand(rays.shape[:-1], device=rays.device) - 0.5) * pixel_ang_size[0]
    jv = (torch.rand(rays.shape[:-1], device=rays.device) - 0.5) * pixel_ang_size[1]
    jittered = rays.clone()
    new_dirs = dirs + ju.unsqueeze(-1) * right + jv.unsqueeze(-1) * up
    jittered[..., 3:6] = new_dirs / new_dirs.norm(dim=-1, keepdim=True)
    return jittered


dataset_dict = {
    "ct_synthetic": CTSyntheticDataset,
    "r2_gaussian": R2GaussianDataset,
}


class DataHandler:
    def __init__(self, dataset_args, rays_per_batch, device="cuda"):
        self.args = dataset_args
        self.rays_per_batch = rays_per_batch
        self.device = torch.device(device)

    def reload(self, split, downsample=None):
        dataset = dataset_dict[self.args.dataset]

        kwargs = {}
        if hasattr(self.args, "num_angles"):
            kwargs["num_angles"] = self.args.num_angles
        if hasattr(self.args, "detector_size"):
            kwargs["detector_size"] = self.args.detector_size

        split_dataset = dataset(
            data_dir=self.args.data_path, split=split, **kwargs
        )

        self.rays = split_dataset.all_rays
        self.projections = split_dataset.all_projections
        self.pixel_size = getattr(split_dataset, "pixel_size", None)
        self.pixel_ang_size = getattr(split_dataset, "pixel_ang_size", None)
        self.beam_type = getattr(split_dataset, "beam_type", None)

        if split == "train":
            self.train_rays = einops.rearrange(
                self.rays, "n h w r -> (n h w) r"
            )
            self.train_projections = einops.rearrange(
                self.projections, "n h w c -> (n h w) c"
            )
            self.batch_size = self.rays_per_batch

    def get_iter(self):
        ray_batch_fetcher = radfoam.BatchFetcher(
            self.train_rays, self.batch_size, shuffle=True
        )
        proj_batch_fetcher = radfoam.BatchFetcher(
            self.train_projections, self.batch_size, shuffle=True
        )

        while True:
            ray_batch = ray_batch_fetcher.next()
            proj_batch = proj_batch_fetcher.next()
            if self.beam_type == "parallel" and self.pixel_size is not None:
                ray_batch = jitter_rays_parallel(ray_batch, self.pixel_size)
            elif self.beam_type == "cone" and self.pixel_ang_size is not None:
                ray_batch = jitter_rays_cone(ray_batch, self.pixel_ang_size)
            yield ray_batch, proj_batch
