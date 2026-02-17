import os

import numpy as np
import einops
import torch

import radfoam

from .ct_synthetic import CTSyntheticDataset
from .r2_gaussian import R2GaussianDataset


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

            yield ray_batch, proj_batch
