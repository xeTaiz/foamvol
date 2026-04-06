import math

import numpy as np
import torch


def angle2pose(DSO, angle):
    """Transfer angle to c2w pose matrix (from r2_gaussian.py)."""
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


def ray_box_intersect(origins, directions, box_min, box_max):
    """Slab method for ray-AABB intersection.

    Args:
        origins: (..., 3)
        directions: (..., 3)
        box_min: (3,) tensor
        box_max: (3,) tensor

    Returns:
        t_min, t_max: (...,) entry/exit distances (t_max < t_min means miss)
    """
    inv_dir = 1.0 / (directions + 1e-12)

    t1 = (box_min - origins) * inv_dir
    t2 = (box_max - origins) * inv_dir

    t_near = torch.minimum(t1, t2)
    t_far = torch.maximum(t1, t2)

    t_min = t_near.max(dim=-1).values
    t_max = t_far.min(dim=-1).values

    return t_min, t_max


def compute_projections(origins, directions, boxes, densities):
    """Compute line integrals through a set of axis-aligned boxes.

    Args:
        origins: (H, W, 3)
        directions: (H, W, 3)
        boxes: list of (box_min, box_max) tuples, each (3,) tensors
        densities: list of float, one per box

    Returns:
        projections: (H, W) accumulated line integral
    """
    proj = torch.zeros(origins.shape[:-1], dtype=origins.dtype)

    for (box_min, box_max), density in zip(boxes, densities):
        t_min, t_max = ray_box_intersect(origins, directions, box_min, box_max)
        hit = t_max > t_min
        path_length = torch.clamp(t_max - t_min, min=0.0)
        proj += density * path_length * hit.float()

    return proj


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

def make_single_cube_scene(cube_half=0.25, density=1.0):
    """One cube centered at origin."""
    box_min = torch.tensor([-cube_half, -cube_half, -cube_half])
    box_max = torch.tensor([cube_half, cube_half, cube_half])
    return [(box_min, box_max)], [density]


def make_2x2x2_scene(block_half=0.25):
    """2x2x2 grid of cubes centered at origin, each with different density.

    Densities are offset from standard thresholds (0.1, 0.3, 0.5, 0.7, 0.9)
    to avoid isosurfaces landing inside cubes during evaluation.
    """
    densities_list = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    boxes = []
    densities = []
    idx = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                x0 = -block_half + ix * block_half
                y0 = -block_half + iy * block_half
                z0 = -block_half + iz * block_half
                box_min = torch.tensor([x0, y0, z0])
                box_max = torch.tensor([x0 + block_half, y0 + block_half, z0 + block_half])
                boxes.append((box_min, box_max))
                densities.append(densities_list[idx])
                idx += 1
    return boxes, densities


# ---------------------------------------------------------------------------
# Utility: compute GT volume for evaluation
# ---------------------------------------------------------------------------

def make_gt_volume(boxes, densities, resolution=256, extent=1.0):
    """Create a ground truth volume on a regular grid.

    Returns:
        volume: (resolution, resolution, resolution) numpy array
    """
    coords = np.linspace(-extent, extent, resolution, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    vol = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    for (box_min, box_max), density in zip(boxes, densities):
        bmin = box_min.numpy()
        bmax = box_max.numpy()
        mask = ((x >= bmin[0]) & (x <= bmax[0]) &
                (y >= bmin[1]) & (y <= bmax[1]) &
                (z >= bmin[2]) & (z <= bmax[2]))
        vol[mask] = density

    return vol


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CTCubeDataset:
    """Generates cone-beam CT projection data from analytical box phantoms.

    Scenes:
        "cube_single" — one cube centered at origin (density=1.0)
        "cube_2x2x2"  — 2×2×2 grid of cubes with densities 0.2–0.9

    Beam geometry matches R2 Gaussian conventions (cone beam, angle2pose).
    """

    def __init__(self, data_dir="cube_single", split="train",
                 num_angles=180, detector_size=128, **kwargs):
        scene_type = data_dir

        if scene_type == "cube_single":
            self.boxes, self.densities = make_single_cube_scene()
        elif scene_type == "cube_2x2x2":
            self.boxes, self.densities = make_2x2x2_scene()
        else:
            raise ValueError(f"Unknown cube scene: {scene_type}")

        # Cone beam geometry (similar to R2 Gaussian)
        DSO = 4.0   # source-to-origin distance
        DSD = 8.0   # source-to-detector distance
        det_phys = 3.0  # physical detector size (covers [-1.5, 1.5])

        nv = nu = detector_size
        fx = nu * DSD / det_phys
        fy = nv * DSD / det_phys

        # Generate angles
        all_angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        if split == "train":
            angles = all_angles[::2]  # even indices
        else:
            angles = all_angles[1::2]  # odd indices

        # Pixel grid (camera space)
        iu = np.arange(nu, dtype=np.float32)
        iv = np.arange(nv, dtype=np.float32)
        iv_grid, iu_grid = np.meshgrid(iv, iu, indexing="ij")

        x_cam = (iu_grid + 0.5 - nu / 2.0) / fx
        y_cam = (iv_grid + 0.5 - nv / 2.0) / fy
        z_cam = np.ones_like(x_cam)
        dir_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        dir_cam = dir_cam / np.linalg.norm(dir_cam, axis=-1, keepdims=True)
        dir_cam_t = torch.from_numpy(dir_cam).float()

        all_rays = []
        all_projections = []

        for angle in angles:
            c2w = angle2pose(DSO, angle)
            rot = torch.from_numpy(c2w[:3, :3]).float()
            origin = torch.from_numpy(c2w[:3, 3]).float()

            dir_world = dir_cam_t @ rot.T  # (nv, nu, 3)
            origins = origin.unsqueeze(0).unsqueeze(0).expand(nv, nu, 3).contiguous()
            rays = torch.cat([origins, dir_world], dim=-1)  # (nv, nu, 6)

            proj = compute_projections(origins, dir_world, self.boxes, self.densities)
            all_rays.append(rays)
            all_projections.append(proj)

        self.all_rays = torch.stack(all_rays, dim=0).float()
        self.all_projections = torch.stack(all_projections, dim=0).unsqueeze(-1).float()
        self.beam_type = "cone"
        self.pixel_ang_size = (1.0 / fx, 1.0 / fy)
        self.fx = fx
        self.fy = fy
        self.c2ws = torch.from_numpy(
            np.stack([angle2pose(DSO, a) for a in angles], axis=0)
        ).float()

        # Store scene info for GT volume generation
        self.scene_type = scene_type
