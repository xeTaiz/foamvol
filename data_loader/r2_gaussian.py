import json
import os

import numpy as np
import torch


def angle2pose(DSO, angle):
    """Transfer angle to c2w pose matrix.

    Copied from R2 Gaussian (dataset_readers.py:156-191).
    Applies three rotations:
      1. rotate -90 deg around x-axis
      2. rotate +90 deg around z-axis
      3. rotate `angle` around z-axis
    Translation = source position at (DSO*cos(angle), DSO*sin(angle), 0).
    """
    phi1 = -np.pi / 2
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi1), -np.sin(phi1)],
        [0.0, np.sin(phi1), np.cos(phi1)],
    ])
    phi2 = np.pi / 2
    R2 = np.array([
        [np.cos(phi2), -np.sin(phi2), 0.0],
        [np.sin(phi2), np.cos(phi2), 0.0],
        [0.0, 0.0, 1.0],
    ])
    R3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot = R3 @ R2 @ R1
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0.0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


class R2GaussianDataset:
    """Loads cone-beam CT data in R2 Gaussian format.

    Data layout:
        data_dir/
            meta_data.json
            proj_train/proj_train_NNNN.npy   (512x512 float32 each)
            proj_test/proj_test_NNNN.npy

    Produces:
        self.all_rays         — [num_views, nv, nu, 6] float32
        self.all_projections  — [num_views, nv, nu, 1] float32
    """

    def __init__(self, data_dir, split="train", **kwargs):
        with open(os.path.join(data_dir, "meta_data.json")) as f:
            meta = json.load(f)

        scanner = meta["scanner"]
        DSO = scanner["DSO"]
        DSD = scanner["DSD"]
        nDetector = scanner["nDetector"]  # [nv, nu]
        sDetector = scanner["sDetector"]  # [sv, su] physical size
        sVoxel = scanner["sVoxel"]

        # Scene scale normalization (R2 Gaussian convention)
        scene_scale = 2.0 / max(sVoxel)
        DSO *= scene_scale
        DSD *= scene_scale
        sDetector = [s * scene_scale for s in sDetector]

        nv, nu = nDetector
        # Focal lengths in pixel units
        fx = nu * DSD / sDetector[1]
        fy = nv * DSD / sDetector[0]

        # Select split
        entries = meta[f"proj_{split}"]

        # Load projections
        projections = []
        angles = []
        for entry in entries:
            proj = np.load(os.path.join(data_dir, entry["file_path"]))
            projections.append(proj)
            angles.append(entry["angle"])

        projections = np.stack(projections, axis=0)  # [N, nv, nu]
        projections = projections * scene_scale
        angles = np.array(angles)

        # Build cone-beam rays for all views (vectorized per view)
        # Pixel grid: (iv, iu) indices
        iu = np.arange(nu, dtype=np.float32)  # [nu]
        iv = np.arange(nv, dtype=np.float32)  # [nv]
        iv_grid, iu_grid = np.meshgrid(iv, iu, indexing="ij")  # [nv, nu]

        # Camera-space directions per pixel
        x_cam = (iu_grid + 0.5 - nu / 2.0) / fx
        y_cam = (iv_grid + 0.5 - nv / 2.0) / fy
        z_cam = np.ones_like(x_cam)
        dir_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [nv, nu, 3]
        dir_cam = dir_cam / np.linalg.norm(dir_cam, axis=-1, keepdims=True)

        all_rays = []
        for angle in angles:
            c2w = angle2pose(DSO, angle)
            rot = c2w[:3, :3]  # [3, 3]
            origin = c2w[:3, 3]  # [3]

            # Rotate directions to world space: [nv, nu, 3] @ [3, 3]^T
            dir_world = dir_cam @ rot.T  # [nv, nu, 3]

            # Origin is the same for all pixels (source position)
            origins = np.broadcast_to(origin, (nv, nu, 3)).copy()

            rays = np.concatenate([origins, dir_world], axis=-1)  # [nv, nu, 6]
            all_rays.append(rays)

        all_rays = np.stack(all_rays, axis=0)  # [N, nv, nu, 6]

        self.all_rays = torch.from_numpy(all_rays).float()
        self.all_projections = torch.from_numpy(
            projections[..., np.newaxis]
        ).float()
