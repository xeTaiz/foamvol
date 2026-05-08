import os

import numpy as np
import einops
import torch

import radfoam

from .ct_cube import CTCubeDataset
from .ct_synthetic import CTSyntheticDataset
from .r2_gaussian import R2GaussianDataset
from .lodopab import LoDoPaBDataset
from .two_detectct import TwoDeteCTDataset
from .more import MOREDataset
from .aapm_mayo import AAPMMayoDataset
from .inveon_ct import InveonDataset
from .acr_dicomctpd import ACRPhantomDataset
from .targeted_sampler import build_targeted_batch
from .utils import bilinear_proj_lookup


def jitter_rays_parallel(rays, pixel_size):
    """Perturb parallel-beam ray origins by uniform random offset within one pixel.

    Returns (jittered_rays, du_pix, dv_pix) where du/dv are in [-0.5, 0.5].
    """
    dirs = rays[..., 3:6]
    perp = torch.stack([-dirs[..., 1], dirs[..., 0], torch.zeros_like(dirs[..., 0])], dim=-1)
    vert = torch.zeros_like(dirs)
    vert[..., 2] = 1.0
    du_pix = torch.rand(rays.shape[:-1], device=rays.device) - 0.5
    dv_pix = torch.rand(rays.shape[:-1], device=rays.device) - 0.5
    jittered = rays.clone()
    jittered[..., :3] += (du_pix * pixel_size).unsqueeze(-1) * perp + (dv_pix * pixel_size).unsqueeze(-1) * vert
    return jittered, du_pix, dv_pix


def jitter_rays_cone(rays, pixel_ang_size):
    """Perturb cone-beam ray directions by uniform random offset within one pixel's angular extent.

    Returns (jittered_rays, du_pix, dv_pix) where du/dv are in [-0.5, 0.5].
    """
    dirs = rays[..., 3:6]
    world_up = torch.zeros_like(dirs)
    world_up[..., 2] = 1.0
    right = torch.cross(dirs, world_up, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(dirs)
    fallback[..., 0] = 1.0
    right = torch.where(right_norm > 1e-6, right, torch.cross(dirs, fallback, dim=-1))
    right = right / right.norm(dim=-1, keepdim=True)
    up = torch.cross(right, dirs, dim=-1)
    up = up / up.norm(dim=-1, keepdim=True)
    du_pix = torch.rand(rays.shape[:-1], device=rays.device) - 0.5
    dv_pix = torch.rand(rays.shape[:-1], device=rays.device) - 0.5
    jittered = rays.clone()
    new_dirs = dirs + (du_pix * pixel_ang_size[0]).unsqueeze(-1) * right + (dv_pix * pixel_ang_size[1]).unsqueeze(-1) * up
    jittered[..., 3:6] = new_dirs / new_dirs.norm(dim=-1, keepdim=True)
    return jittered, du_pix, dv_pix


def _apply_jitter(rays, beam_type, pixel_size, pixel_ang_size, device):
    """Apply jitter for the given beam type. Returns (jittered_rays, du_pix, dv_pix)."""
    if beam_type == "parallel" and pixel_size is not None:
        return jitter_rays_parallel(rays, pixel_size)
    elif beam_type == "cone" and pixel_ang_size is not None:
        return jitter_rays_cone(rays, pixel_ang_size)
    else:
        zero = torch.zeros(rays.shape[0], device=device)
        return rays, zero, zero


dataset_dict = {
    "ct_cube": CTCubeDataset,
    "ct_synthetic": CTSyntheticDataset,
    "r2_gaussian": R2GaussianDataset,
    "lodopab": LoDoPaBDataset,
    "two_detectct": TwoDeteCTDataset,
    "more": MOREDataset,
    "aapm_mayo": AAPMMayoDataset,
    "inveon_ct": InveonDataset,
    "acr_phantom": ACRPhantomDataset,
}


class DataHandler:
    def __init__(self, dataset_args, rays_per_batch, device="cuda"):
        self.args = dataset_args
        self.rays_per_batch = rays_per_batch
        self.device = torch.device(device)

    def reload(self, split, downsample=None):
        dataset = dataset_dict[self.args.dataset]

        import inspect
        accepted = set(inspect.signature(dataset.__init__).parameters)

        kwargs = {}
        if "num_angles" in accepted and hasattr(self.args, "num_angles") and self.args.num_angles > 0:
            kwargs["num_angles"] = self.args.num_angles
        if "detector_size" in accepted and hasattr(self.args, "detector_size"):
            kwargs["detector_size"] = self.args.detector_size
        if "sample_index" in accepted and hasattr(self.args, "sample_index"):
            kwargs["sample_index"] = self.args.sample_index
        if "mode" in accepted and hasattr(self.args, "mode"):
            kwargs["mode"] = self.args.mode
        if "split_override" in accepted and hasattr(self.args, "split_override") and self.args.split_override:
            kwargs["split_override"] = self.args.split_override

        split_dataset = dataset(
            data_dir=self.args.data_path, split=split, **kwargs
        )

        self.rays = split_dataset.all_rays
        self.projections = split_dataset.all_projections
        self.pixel_size = getattr(split_dataset, "pixel_size", None)
        self.pixel_ang_size = getattr(split_dataset, "pixel_ang_size", None)
        self.beam_type = getattr(split_dataset, "beam_type", None)
        self.fx = getattr(split_dataset, "fx", None)
        self.fy = getattr(split_dataset, "fy", None)
        self.c2ws = getattr(split_dataset, "c2ws", None)

        if split == "train":
            self.train_rays = einops.rearrange(
                self.rays, "n h w r -> (n h w) r"
            )
            self.train_projections = einops.rearrange(
                self.projections, "n h w c -> (n h w) c"
            )
            self.batch_size = self.rays_per_batch

            num_angles, det_h, det_w, _ = self.rays.shape

            # GPU tensors — sole source of truth for training
            self._train_rays_gpu = self.train_rays.to(self.device, non_blocking=True)
            self._train_projections_gpu = self.train_projections.to(self.device, non_blocking=True)
            # View-shaped projection tensor for bilinear lookup
            self._proj_nchw_gpu = self._train_projections_gpu.reshape(num_angles, det_h, det_w, 1)

            # Pre-compute beam geometry lookup tables on GPU
            self._beam_geom = {
                "num_angles": num_angles,
                "det_h": det_h,
                "det_w": det_w,
            }
            if self.beam_type == "parallel":
                self._beam_geom["center_rays"] = self.rays[:, det_h // 2, det_w // 2].to(self.device, non_blocking=True)
                self._beam_geom["pixel_size"] = self.pixel_size
            elif self.beam_type == "cone" and self.c2ws is not None:
                self._beam_geom["c2ws"] = self.c2ws.to(self.device, non_blocking=True)
                self._beam_geom["fx"] = self.fx
                self._beam_geom["fy"] = self.fy

    def _sample_uniform_batch(self, batch_size):
        """Sample a random batch of rays with bilinear GT. Returns (rays, proj, flat_idx)."""
        N, H, W = self.rays.shape[:3]
        flat_idx = torch.randint(0, N * H * W, (batch_size,), device=self.device)
        rays = self._train_rays_gpu[flat_idx]
        view = flat_idx // (H * W)
        rem = flat_idx % (H * W)
        iy = rem // W
        ix = rem % W
        rays, du_pix, dv_pix = _apply_jitter(rays, self.beam_type, self.pixel_size, self.pixel_ang_size, self.device)
        proj = bilinear_proj_lookup(self._proj_nchw_gpu, view, ix.float() + du_pix, iy.float() + dv_pix)
        return rays, proj, flat_idx

    def update_targeting(self, cell_weights, points, cell_radii,
                         targeted_fraction=0.2):
        """Update GPU-side targeting state (called at triangulation updates)."""
        self._target_weights = cell_weights
        self._target_points = points
        self._target_radii = cell_radii
        self._targeted_batch_size = int(self.batch_size * targeted_fraction)

    # --- High-error sampling ---

    def init_high_error_sampling(self, high_error_fraction, high_error_power=1.0):
        """Initialize high-error ray sampling state."""
        num_angles, det_h, det_w, _ = self.rays.shape
        if det_h < 2:
            if high_error_fraction > 0:
                import warnings
                warnings.warn(
                    "high_error_fraction is not supported when det_h=1 (e.g. LoDoPaB). "
                    "Set high_error_fraction=0 in your config.",
                    stacklevel=2,
                )
            self._he_fraction = 0.0
            self._he_batch_size = 0
            return
        err_h, err_w = det_h // 2, det_w // 2  # 256² error map
        self._he_fraction = high_error_fraction
        self._he_power = high_error_power
        self._he_batch_size = int(self.batch_size * high_error_fraction)
        self._he_err_h = err_h
        self._he_err_w = err_w
        self._he_det_h = det_h
        self._he_det_w = det_w
        self._he_num_angles = num_angles

        # Error map at half resolution, bfloat16
        self._he_error_map = torch.rand(
            num_angles, err_h, err_w, dtype=torch.bfloat16, device=self.device,
        ) * 1e-2

        # Pre-compute downsampled GT projections at 256² (average pool 512→256)
        gt_full = self.projections  # (N, H, W, 1)
        gt_4d = gt_full.squeeze(-1).unsqueeze(1).float()  # (N, 1, H, W)
        self._he_gt_ds = torch.nn.functional.avg_pool2d(
            gt_4d, kernel_size=2,
        ).squeeze(1).to(self.device)  # (N, err_h, err_w) on GPU

        # Pre-sampled pool (filled on first refresh)
        self._he_pool = None
        self._he_pool_cursor = 0

    def update_high_error_map(self, flat_indices, errors):
        """Update error map entries for sampled pixels.

        Args:
            flat_indices: (B,) int64 — flat indices into (N*H*W) at full 512² res
            errors: (B,) float — |pred - gt| per sampled pixel
        """
        if not hasattr(self, '_he_error_map'):
            return
        # Map 512² flat indices to 256² error map indices
        H, W = self._he_det_h, self._he_det_w
        eh, ew = self._he_err_h, self._he_err_w
        view_idx = flat_indices // (H * W)
        pixel_in_view = flat_indices % (H * W)
        iy = (pixel_in_view // W) // 2  # 512→256
        ix = (pixel_in_view % W) // 2
        # Clamp to valid range
        iy = iy.clamp(0, eh - 1)
        ix = ix.clamp(0, ew - 1)
        self._he_error_map[view_idx, iy, ix] = errors.to(torch.bfloat16)

    def _refresh_high_error_pool(self):
        """Re-sample the high-error ray pool from current error map."""
        N = self._he_num_angles
        eh, ew = self._he_err_h, self._he_err_w
        H, W = self._he_det_h, self._he_det_w

        # Work at 256² — apply power scaling and filter to above median
        weights = self._he_error_map.float().reshape(-1)  # (N*eh*ew,)
        # nan/inf from corrupted loss steps → treat as 0 (uniform fallback)
        weights = weights.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-8) ** self._he_power
        median = weights.median()
        candidate_idx = (weights >= median).nonzero(as_tuple=True)[0]
        if candidate_idx.shape[0] > 2**24:
            keep = torch.randperm(candidate_idx.shape[0], device=candidate_idx.device)[:2**24]
            candidate_idx = candidate_idx[keep]
        sub_weights = weights[candidate_idx]
        sub_weights = sub_weights / sub_weights.sum()

        # Pool lasts one epoch — sample at 256², expand to all 4 pixels of 2×2
        # Ensure pool is always at least as large as one batch (handles small datasets)
        min_pool_lo = (self._he_batch_size + 3) // 4
        pool_size_lo = max(min_pool_lo, int(self._he_fraction * N * H * W) // 4)
        pool_sub = torch.multinomial(sub_weights, pool_size_lo, replacement=True)
        lowres_flat = candidate_idx[pool_sub]  # indices into (N*eh*ew)

        # Map each 256² sample to all 4 corresponding 512² pixels
        view_idx = lowres_flat // (eh * ew)
        pixel_in_view = lowres_flat % (eh * ew)
        iy_lo = pixel_in_view // ew
        ix_lo = pixel_in_view % ew
        # Expand to 2×2 block: (pool_size_lo,) → (pool_size_lo, 4)
        iy_hi = (iy_lo * 2).unsqueeze(1) + torch.tensor([0, 0, 1, 1], device=iy_lo.device)
        ix_hi = (ix_lo * 2).unsqueeze(1) + torch.tensor([0, 1, 0, 1], device=ix_lo.device)
        iy_hi = iy_hi.clamp(0, H - 1)
        ix_hi = ix_hi.clamp(0, W - 1)
        view_exp = view_idx.unsqueeze(1).expand(-1, 4)
        flat_indices = view_exp * (H * W) + iy_hi * W + ix_hi  # (pool_size_lo, 4)
        self._he_pool = flat_indices.reshape(-1)  # (pool_size_lo * 4,)
        self._he_pool_cursor = 0

    def _get_high_error_batch(self):
        """Get a batch of high-error rays from the pre-sampled pool."""
        if self._he_batch_size == 0:
            empty = torch.zeros(0, 6, device=self.device)
            empty_proj = torch.zeros(0, 1, device=self.device)
            return empty, empty_proj
        if self._he_pool is None or self._he_pool_cursor + self._he_batch_size > self._he_pool.shape[0]:
            self._refresh_high_error_pool()
        start = self._he_pool_cursor
        end = start + self._he_batch_size
        flat_idx = self._he_pool[start:end]
        self._he_pool_cursor = end
        rays = self._train_rays_gpu[flat_idx]
        H, W = self._he_det_h, self._he_det_w
        view = flat_idx // (H * W)
        rem = flat_idx % (H * W)
        iy = rem // W
        ix = rem % W
        rays, du_pix, dv_pix = _apply_jitter(rays, self.beam_type, self.pixel_size, self.pixel_ang_size, self.device)
        proj = bilinear_proj_lookup(self._proj_nchw_gpu, view, ix.float() + du_pix, iy.float() + dv_pix)
        return rays, proj

    def set_batch_size(self, new_batch_size):
        self.rays_per_batch = new_batch_size
        self.batch_size = new_batch_size
        if hasattr(self, '_he_fraction'):
            self._he_batch_size = int(self.batch_size * self._he_fraction)
            # _he_pool cursor logic auto-refreshes when remaining < _he_batch_size

    def get_high_error_iter(self):
        """Iterator yielding (uniform_rays + high_error_rays, uniform_proj + high_error_proj)."""
        while True:
            u_rays, u_proj, _ = self._sample_uniform_batch(self.batch_size)
            he_rays, he_proj = self._get_high_error_batch()
            yield torch.cat([u_rays, he_rays], 0), torch.cat([u_proj, he_proj], 0)

    def get_iter(self):
        while True:
            rays, proj, _ = self._sample_uniform_batch(self.batch_size)
            yield rays, proj

    def get_targeted_iter(self):
        """Iterator yielding (uniform_rays + targeted_rays, uniform_proj + targeted_proj)."""
        while True:
            u_rays, u_proj, _ = self._sample_uniform_batch(self.batch_size)
            t_rays, t_proj = build_targeted_batch(
                self._target_weights, self._target_points,
                self._target_radii,
                self._train_rays_gpu, self._proj_nchw_gpu,
                self.beam_type, self._beam_geom,
                self._targeted_batch_size,
            )
            yield torch.cat([u_rays, t_rays], 0), torch.cat([u_proj, t_proj], 0)
