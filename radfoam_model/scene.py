import os
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement
import tqdm

import radfoam
from radfoam_model.render import TraceRays
from radfoam_model.utils import *


IDWResult = namedtuple("IDWResult", [
    "nn_idx",      # (B,) containing cell indices
    "pad_idx",     # (B, K+1) padded neighbor indices (slot 0 = self)
    "valid",       # (B, K+1) validity mask
    "weights",     # (B, K+1) normalized bilateral weights
    "vals",        # (B, K+1) activated density values
    "dist_sq",     # (B, K+1) squared distances to neighbors
    "counts",      # (B,) neighbor counts per cell
    "idw_result",  # (B,) weighted average
])


def idw_query(query, points, adjacency, adjacency_offsets, aabb_tree,
              activated, sigma, sigma_v, global_max_k=None,
              per_cell_sigma=False, per_neighbor_sigma=False,
              cell_radius=None):
    """Bilateral IDW interpolation for a batch of query points.

    Matches the CUDA kernel: exp(-d²/σ²) spatial × exp(-Δμ²/σ_v²) bilateral.

    Args:
        query: (B, 3) tensor of query positions
        points: (N, 3) cell centers
        adjacency: (E,) CSR column indices
        adjacency_offsets: (N+1,) CSR row pointers
        aabb_tree: AABB tree for NN queries
        activated: (N,) precomputed softplus-activated densities
        sigma: spatial Gaussian scale (or scale factor when per_cell_sigma=True)
        sigma_v: bilateral value-similarity scale (None=disabled)
        global_max_k: max neighbor count (computed if None)
        per_cell_sigma: if True, sigma is a scale factor × cell_radius
        per_neighbor_sigma: each neighbor slot uses its own cell's radius
        cell_radius: (N,) required when per_cell_sigma=True

    Returns:
        IDWResult namedtuple
    """
    device = query.device
    adj = adjacency.long()
    adj_off = adjacency_offsets.long()
    B = query.shape[0]

    if global_max_k is None:
        global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())

    nn_idx = radfoam.nn(points, aabb_tree, query).long()

    counts = adj_off[nn_idx + 1] - adj_off[nn_idx]
    offsets = adj_off[nn_idx]

    pad_idx = torch.zeros(B, global_max_k + 1, dtype=torch.long, device=device)
    valid = torch.zeros(B, global_max_k + 1, dtype=torch.bool, device=device)
    pad_idx[:, 0] = nn_idx
    valid[:, 0] = True

    k_range = torch.arange(global_max_k, device=device)
    has_k = counts.unsqueeze(1) > k_range.unsqueeze(0)
    flat_offsets = offsets.unsqueeze(1) + k_range.unsqueeze(0)
    flat_offsets = flat_offsets.clamp(max=adj.shape[0] - 1)
    pad_idx[:, 1:] = adj[flat_offsets]
    valid[:, 1:] = has_k

    centers = points[pad_idx]
    diff = query.unsqueeze(1) - centers
    dist_sq = diff.pow(2).sum(dim=-1)

    if per_cell_sigma and cell_radius is not None:
        if per_neighbor_sigma:
            sigma_sq = (sigma * cell_radius[pad_idx]).pow(2)
        else:
            sigma_sq = (sigma * cell_radius[nn_idx]).pow(2).unsqueeze(1)
    else:
        sigma_sq = sigma * sigma

    w = torch.exp(-dist_sq / sigma_sq)

    vals = activated[pad_idx]
    if sigma_v is not None:
        ref_val = activated[nn_idx]
        val_diff = vals - ref_val.unsqueeze(1)
        w = w * torch.exp(-val_diff * val_diff / (sigma_v * sigma_v))

    w[~valid] = 0.0
    w = w + valid.float() * 1e-6
    weights = w / w.sum(dim=1, keepdim=True)

    masked_vals = vals.clone()
    masked_vals[~valid] = 0.0
    idw_result = (weights * masked_vals).sum(dim=1)

    return IDWResult(
        nn_idx=nn_idx, pad_idx=pad_idx, valid=valid, weights=weights,
        vals=vals, dist_sq=dist_sq, counts=counts, idw_result=idw_result,
    )


def projection_contrast(proj, normalize=True):
    """Sobel gradient magnitude on a (..., H, W, C) projection (batch-aware)."""
    leading = proj.shape[:-3]
    H, W, C = proj.shape[-3], proj.shape[-2], proj.shape[-1]
    img = proj.reshape(-1, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=img.dtype, device=img.device).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1)
    gx = F.conv2d(img, sobel_x, padding=1, groups=C)
    gy = F.conv2d(img, sobel_y, padding=1, groups=C)
    mag = (gx**2 + gy**2).sqrt()  # (B, C, H, W)
    if normalize:
        # Normalize per-image
        mag = mag / (mag.flatten(1).max(dim=1).values[:, None, None, None] + 1e-8)
    mag = mag.permute(0, 2, 3, 1).reshape(*leading, H, W, C)
    return mag


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
        init_val = self.init_density
        density = torch.full((mg.size(0), 1), init_val, device=self.device, dtype=torch.float32)
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

    @torch.no_grad()
    def initialize_from_volume(self, vol_path, ref_resolution=64, ref_blur_sigma=2.0):
        """Initialize cell densities by sampling a pre-computed volume (e.g. FDK).

        The volume must be a (R, R, R) float32 numpy array covering [-1, 1]^3,
        stored in (X, Y, Z) axis order (same convention as vis_foam DRR rendering).

        Negative values (FDK ring artifacts) are clamped to zero before inversion.
        The volume is Gaussian-blurred before sampling to remove high-frequency noise
        and FDK streak artifacts. A blurred+downsampled copy is also stored as the
        reference volume for reference_volume_loss().

        Args:
            vol_path: path to .npy volume
            ref_resolution: target resolution for the stored reference volume
            ref_blur_sigma: Gaussian blur sigma applied before sampling (source voxels)
        """
        import math
        vol_np = np.load(vol_path).astype(np.float32)
        vol_5d = torch.from_numpy(vol_np).to(self.device).unsqueeze(0).unsqueeze(0)

        if ref_blur_sigma > 0:
            ks = max(3, 2 * int(math.ceil(2 * ref_blur_sigma)) + 1)
            pad = ks // 2
            coords = torch.arange(ks, dtype=torch.float32, device=vol_5d.device) - pad
            gauss_1d = torch.exp(-coords ** 2 / (2 * ref_blur_sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            vol_5d = gauss_conv3d_separable(vol_5d, gauss_1d, pad)

        # Store blurred+downsampled reference volume
        raw_res = vol_np.shape[0]
        stride = max(1, raw_res // ref_resolution)
        t_ref = F.avg_pool3d(vol_5d, kernel_size=stride, stride=stride) if stride > 1 else vol_5d
        if t_ref.shape[-1] != ref_resolution:
            t_ref = F.interpolate(t_ref, size=ref_resolution, mode="trilinear", align_corners=True)
        self._ref_volume = t_ref.squeeze().detach().float()
        self._ref_weight = None

        pts = self.primal_points.detach()                      # (N, 3) as (x, y, z)
        # grid_sample: grid[..., 0]→W, grid[..., 1]→H, grid[..., 2]→D
        # volume is (D=X, H=Y, W=Z) so we need to pass (z, y, x) — flip world (x,y,z)
        grid = pts.flip(-1).reshape(1, 1, 1, -1, 3)           # (1, 1, 1, N, 3)
        sampled = F.grid_sample(
            vol_5d, grid, mode="bilinear", padding_mode="border", align_corners=True
        )                                                       # (1, 1, 1, 1, N)
        fdk_mu = sampled.reshape(-1).clamp(1e-6, 1.0)         # (N,) — clamp negatives

        raw = self.softplus_inv(fdk_mu / self.activation_scale)
        self.density.data.copy_(raw.unsqueeze(1))

        print(f"[FDK init] loaded {vol_path} (blur σ={ref_blur_sigma}, ref_res={ref_resolution})")
        print(f"  cells: {pts.shape[0]}  density [{fdk_mu.min():.4f}, {fdk_mu.max():.4f}]"
              f"  mean: {fdk_mu.mean():.4f}")

    @torch.no_grad()
    def load_reference_volume(self, path, resolution=64, blur_sigma=2.0,
                              edge_mask=False, edge_alpha=10.0):
        """Load a reference volume from a .npy or .pt file for reference_volume_loss().

        The volume is Gaussian-blurred and downsampled to `resolution` before storage.

        Args:
            path: path to .npy volume or .pt model checkpoint
            resolution: target voxel grid resolution (stored at this resolution)
            blur_sigma: Gaussian blur sigma applied to .npy volumes (source voxels)
            edge_mask: if True, weight loss by inverse gradient magnitude of ref
            edge_alpha: sensitivity of edge mask — weight = 1/(1 + alpha*|∇ref|)
        """
        import math
        if path.endswith(".pt"):
            ckpt = torch.load(path, map_location="cpu")
            pts = ckpt["xyz"].to(self.device)
            raw = ckpt["density"].to(self.device)
            adjacency = ckpt["adjacency"].to(self.device).to(torch.int32)
            adjacency_offsets = ckpt["adjacency_offsets"].to(self.device).to(torch.int32)
            mu = F.softplus(raw.squeeze(), beta=10).detach()

            aabb_tree = radfoam.build_aabb_tree(pts)
            _, cell_radius = radfoam.farthest_neighbor(pts, adjacency, adjacency_offsets)

            # Sample at 128³ to avoid undersampling the Voronoi, then blur+downsample
            idw_res = 128
            voxel_size = 2.0 / idw_res
            centers = torch.linspace(
                -1 + voxel_size / 2, 1 - voxel_size / 2, idw_res,
                device=self.device,
            )
            xx, yy, zz = torch.meshgrid(centers, centers, centers, indexing="ij")
            query = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

            adj_off_long = adjacency_offsets.long()
            global_max_k = int((adj_off_long[1:] - adj_off_long[:-1]).max().item())

            result = torch.zeros(idw_res ** 3, device=self.device)
            batch = 500_000
            for start in range(0, idw_res ** 3, batch):
                end = min(start + batch, idw_res ** 3)
                r = idw_query(
                    query[start:end], pts, adjacency, adjacency_offsets,
                    aabb_tree, mu, sigma=0.7, sigma_v=None,
                    global_max_k=global_max_k,
                    per_cell_sigma=True,
                    cell_radius=cell_radius,
                )
                result[start:end] = r.idw_result

            n_cells = pts.shape[0]
            print(f"  IDW grid: {n_cells} cells → {idw_res}³ (no gaps), then blur+downsample → {resolution}³")

            t = result.reshape(1, 1, idw_res, idw_res, idw_res)
            blur_sigma = max(blur_sigma, 1.0)  # always blur to anti-alias the downsample
        else:
            vol_np = np.load(path).astype(np.float32)
            t = torch.from_numpy(vol_np).to(self.device).unsqueeze(0).unsqueeze(0)

        # Shared blur + downsample for both .pt and .npy paths
        if blur_sigma > 0:
            ks = max(3, 2 * int(math.ceil(2 * blur_sigma)) + 1)
            pad = ks // 2
            coords = torch.arange(ks, dtype=torch.float32, device=t.device) - pad
            gauss_1d = torch.exp(-coords ** 2 / (2 * blur_sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            t = gauss_conv3d_separable(t, gauss_1d, pad)
        if t.shape[2] != resolution:
            t = F.interpolate(t, size=resolution, mode="trilinear", align_corners=True)
        vol = t.squeeze()
        self.set_reference_volume(vol, edge_mask=edge_mask, edge_alpha=edge_alpha)
        print(f"[ref vol] loaded {path} → {resolution}³ grid")

    def set_reference_volume(self, tensor, edge_mask=False, edge_alpha=10.0):
        """Set a pre-computed reference volume for reference_volume_loss().

        Can be called directly with a (R, R, R) tensor — useful for setting
        an intermediate snapshot without file I/O:
            model.set_reference_volume(model._scatter_voxelize(64)[0].detach())

        Args:
            tensor: (R, R, R) float tensor (any resolution — resampled at loss time)
            edge_mask: if True, weight loss by inverse gradient magnitude of ref
            edge_alpha: sensitivity of edge mask
        """
        self._ref_volume = tensor.detach().float()
        if edge_mask:
            grad_mag = self._gradient_magnitude_3d(tensor).detach()
            self._ref_weight = 1.0 / (1.0 + edge_alpha * grad_mag)
        else:
            self._ref_weight = None

    @staticmethod
    def _gradient_magnitude_3d(vol):
        """3D gradient magnitude via central differences. vol: (R,R,R) → (R,R,R)."""
        v = vol.float().unsqueeze(0).unsqueeze(0)  # (1,1,R,R,R)
        k = torch.tensor([-0.5, 0.0, 0.5], device=vol.device, dtype=torch.float32)
        gx = F.conv3d(v, k.reshape(1, 1, 3, 1, 1), padding=(1, 0, 0))
        gy = F.conv3d(v, k.reshape(1, 1, 1, 3, 1), padding=(0, 1, 0))
        gz = F.conv3d(v, k.reshape(1, 1, 1, 1, 3), padding=(0, 0, 1))
        return (gx ** 2 + gy ** 2 + gz ** 2).sqrt().squeeze()

    def _scatter_voxelize(self, resolution=64, extent=1.0):
        """Bin cells into a voxel grid weighted by cell area (radius²), differentiable.

        Returns:
            vol: (res, res, res) float tensor with gradient through density
            occupied: (res, res, res) bool mask — True for voxels with ≥1 cell
        """
        res = resolution
        voxel_size = 2.0 * extent / res
        points = self.primal_points.detach()            # (N, 3), no positional grad
        mu = self.get_primal_density().squeeze()        # (N,) grad through density

        grid_coords = ((points + extent) / voxel_size).long().clamp(0, res - 1)
        voxel_idx = (grid_coords[:, 0] * res * res
                     + grid_coords[:, 1] * res
                     + grid_coords[:, 2])               # (N,) int64

        inside = (points.abs() <= extent).all(dim=1)
        voxel_idx_in = voxel_idx[inside]
        mu_in = mu[inside]

        if self._cached_cell_radius is not None:
            w_in = self._cached_cell_radius[inside].detach() ** 2
        else:
            w_in = torch.ones(inside.sum(), device=mu.device, dtype=mu.dtype)

        num_voxels = res ** 3
        weighted_sum = torch.zeros(num_voxels, device=mu.device, dtype=mu.dtype)
        total_weight = torch.zeros(num_voxels, device=mu.device, dtype=mu.dtype)
        weighted_sum.scatter_add_(0, voxel_idx_in, mu_in * w_in)
        total_weight.scatter_add_(0, voxel_idx_in, w_in)

        occupied_flat = total_weight > 0
        safe_weight = total_weight.clamp(min=1e-10)
        vol_flat = weighted_sum / safe_weight           # (num_voxels,), grad through weighted_sum

        return vol_flat.reshape(res, res, res), occupied_flat.reshape(res, res, res)

    @staticmethod
    @torch.no_grad()
    def _scatter_voxelize_from(points, mu_detached, resolution=64, extent=1.0):
        """Scatter-voxelize external points+densities (no grad, uniform cell weights).

        Used when loading a reference from a model.pt checkpoint where cell_radius
        is not available without rebuilding the triangulation.
        """
        res = resolution
        voxel_size = 2.0 * extent / res
        grid_coords = ((points + extent) / voxel_size).long().clamp(0, res - 1)
        voxel_idx = (grid_coords[:, 0] * res * res
                     + grid_coords[:, 1] * res
                     + grid_coords[:, 2])

        inside = (points.abs() <= extent).all(dim=1)
        voxel_idx_in = voxel_idx[inside]
        mu_in = mu_detached[inside]

        num_voxels = res ** 3
        weighted_sum = torch.zeros(num_voxels, device=points.device, dtype=torch.float32)
        count = torch.zeros(num_voxels, device=points.device, dtype=torch.float32)
        weighted_sum.scatter_add_(0, voxel_idx_in, mu_in.float())
        count.scatter_add_(0, voxel_idx_in, torch.ones(voxel_idx_in.shape[0],
                                                        device=points.device))

        safe_count = count.clamp(min=1.0)
        return (weighted_sum / safe_count).reshape(res, res, res)

    def reference_volume_loss(self, resolution=64):
        """L2 loss between scatter-voxelized current model and the stored reference volume.

        Only occupied voxels (those containing at least one cell) contribute to the loss.
        If edge_mask was set via set_reference_volume(), the loss is weighted by
        1/(1+alpha*|∇ref|) so high-frequency regions are regularized less strongly.

        Returns:
            scalar loss tensor with gradient through model density
        """
        if not hasattr(self, "_ref_volume") or self._ref_volume is None:
            return torch.tensor(0.0, device=self.density.device)

        vol, occupied = self._scatter_voxelize(resolution)  # (R,R,R), (R,R,R) bool

        ref = self._ref_volume
        if ref.shape[0] != resolution:
            ref = F.interpolate(
                ref.unsqueeze(0).unsqueeze(0),
                size=resolution, mode="trilinear", align_corners=True,
            ).squeeze()

        diff = vol - ref  # (R,R,R)

        if self._ref_weight is not None:
            w = self._ref_weight
            if w.shape[0] != resolution:
                w = F.interpolate(
                    w.unsqueeze(0).unsqueeze(0),
                    size=resolution, mode="trilinear", align_corners=True,
                ).squeeze()
            return (w[occupied] * diff[occupied] ** 2).mean()

        return diff[occupied].pow(2).mean()

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
        if "density_peak" in optimizable_tensors:
            self.density_peak = optimizable_tensors["density_peak"]
        if "delta_raw" in optimizable_tensors:
            self.delta_raw = optimizable_tensors["delta_raw"]
        if "cov_raw" in optimizable_tensors:
            self.cov_raw = optimizable_tensors["cov_raw"]
        if hasattr(self, '_frozen_mask'):
            self._frozen_mask = self._frozen_mask[permutation]

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

        # Cache cell radius for starvation tracking (cheap here, avoids per-iter recompute)
        _, cr = radfoam.farthest_neighbor(
            self.primal_points, self.point_adjacency, self.point_adjacency_offsets,
        )
        self._cached_cell_radius = cr.squeeze()

    def get_primal_density(self):
        return self.activation_scale * F.softplus(self.density, beta=10)

    def tv_regularization(self, epsilon=1e-3, area_weighted=False, on_raw=False):
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

    def voxel_variance_regularization(self, resolution=32, sigma_v=0.2, extent=1.0):
        """Bilateral variance loss on a randomly-offset voxel grid.

        Bins cells into a voxel grid, computes weighted mean density per voxel,
        then penalizes each cell's deviation from its voxel mean — weighted
        bilaterally so cells near density boundaries are not smoothed.

        Random grid offset each call prevents persistent binning artifacts.

        Args:
            resolution: grid resolution per axis
            sigma_v: bilateral value sigma — cells with density difference
                     > sigma_v from the voxel mean get reduced weight
            extent: half-extent of the volume (grid spans [-extent, extent]^3)
        """
        res = resolution
        voxel_size = 2.0 * extent / res
        points = self.primal_points.detach()  # (N, 3) — no grad through positions
        mu = self.get_primal_density().squeeze()  # (N,) — grad through density

        # Random grid offset for stochastic binning
        offset = (torch.rand(3, device=points.device) - 0.5) * voxel_size

        # Compute voxel index per cell
        shifted = points + offset
        grid_coords = ((shifted + extent) / voxel_size).long().clamp(0, res - 1)
        voxel_idx = grid_coords[:, 0] * res * res + grid_coords[:, 1] * res + grid_coords[:, 2]

        # Exclude cells outside the volume
        inside = (points.abs() <= extent).all(dim=1)
        voxel_idx = voxel_idx[inside]
        mu_inside = mu[inside]

        # Compute weighted mean per voxel (uniform weights for the mean)
        num_voxels = res ** 3
        sum_mu = torch.zeros(num_voxels, device=mu.device, dtype=mu.dtype)
        count = torch.zeros(num_voxels, device=mu.device, dtype=mu.dtype)
        sum_mu.scatter_add_(0, voxel_idx, mu_inside)
        count.scatter_add_(0, voxel_idx, torch.ones_like(mu_inside))

        # Per-voxel mean (only occupied voxels)
        occupied = count > 1  # need at least 2 cells for variance
        voxel_mean = torch.zeros(num_voxels, device=mu.device, dtype=mu.dtype)
        voxel_mean[occupied] = sum_mu[occupied] / count[occupied]

        # Per-cell deviation from voxel mean
        cell_mean = voxel_mean[voxel_idx]  # (N_inside,)
        diff = mu_inside - cell_mean

        # Bilateral weight: suppress smoothing across density boundaries
        bilateral_w = torch.exp(-diff.detach() ** 2 / (sigma_v ** 2))

        # Weighted variance loss
        loss = (bilateral_w * diff ** 2).mean()

        return loss

    def _neighbor_smooth_target(self, mu_detached, hops):
        """K-hop smoothed density target via iterated message passing (O(k×E)).

        Each hop replaces each cell's value with the mean of its neighbors' current
        values. k=1 = immediate neighbor mean; k=2 = neighbors' neighbor mean, etc.
        Operates on detached mu to avoid grad accumulation across hops.
        """
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = mu_detached.shape[0]
        counts = (offsets[1:] - offsets[:-1]).float().clamp(min=1)
        src = torch.repeat_interleave(
            torch.arange(N, device=mu_detached.device),
            offsets[1:] - offsets[:-1],
        )
        smooth = mu_detached.clone()
        for _ in range(hops):
            nbr_sum = torch.zeros(N, device=smooth.device, dtype=smooth.dtype)
            nbr_sum.scatter_add_(0, src, smooth[adj])
            smooth = nbr_sum / counts
        return smooth  # [N], detached k-hop neighborhood mean

    def neighbor_variance_regularization(self, sigma_v=1.0, hops=1):
        """Bilateral variance loss over the Voronoi neighbor graph.

        Each cell is penalized for deviating from its k-hop neighborhood mean.
        Bilaterally weighted by sigma_v: large sigma_v = plain L2 smoothing;
        small sigma_v = edge-preserving (boundaries down-weighted).
        Combined with sigma annealing in train.py this starts as full smoothing
        and transitions to edge-preserving as training progresses.
        """
        mu = self.get_primal_density().squeeze()        # [N], with grad
        target = self._neighbor_smooth_target(mu.detach(), hops)
        diff = mu - target
        bilateral_w = torch.exp(-(diff.detach() ** 2) / (sigma_v ** 2))
        return (bilateral_w * diff ** 2).mean()

    @torch.no_grad()
    def compute_neighborhood_variance(self, cell_radius=None, hops=1):
        """Per-cell neighborhood variance score for variance-based pruning.

        Returns per-cell score = (mu - k_hop_mean)^2 * max(radius, p10_radius).
        Combined score targets cells that are BOTH smooth (low variance) AND
        small-to-medium sized. Large empty-space cells (large radius) score high
        → protected. Tiny densification-placed cells near boundaries (high variance
        even if small) score high → kept.

        Args:
            cell_radius: [N] tensor of per-cell radii. If None, returns raw variance.
            hops: k-hop neighborhood depth for smoothing target.
        """
        mu = self.get_primal_density().squeeze().detach()
        target = self._neighbor_smooth_target(mu, hops)
        var = (mu - target) ** 2  # [N]

        if cell_radius is not None:
            p10 = torch.quantile(cell_radius, 0.1)
            size_factor = cell_radius.clamp(min=p10)
            return var * size_factor
        return var

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

    @torch.no_grad()
    def compute_neighbor_entropy(self, n_bins=5):
        """Per-cell Shannon entropy of neighbor density distribution.

        High entropy = diverse neighborhood (edges, under-resolved).
        Uses random bin offset so edges aligned with bin interiors
        are caught across multiple calls.
        """
        activated = self.get_primal_density().squeeze()  # (N,)
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = activated.shape[0]

        counts = offsets[1:] - offsets[:-1]
        source = torch.repeat_interleave(
            torch.arange(N, device=activated.device), counts
        )

        # Random bin offset to avoid persistent alignment artifacts
        bin_width = 1.0 / n_bins
        offset = torch.rand(1, device=activated.device).item() * bin_width
        boundaries = torch.arange(1, n_bins, device=activated.device).float() * bin_width + offset

        # Bin neighbor densities
        neighbor_bins = torch.bucketize(activated[adj].clamp(0, 1), boundaries)  # (E,) in [0, K-1]

        # Count per (cell, bin) via scatter into (N, K) matrix
        flat_idx = source * n_bins + neighbor_bins
        bin_counts = torch.zeros(N * n_bins, device=activated.device)
        bin_counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        bin_counts = bin_counts.reshape(N, n_bins)

        # Also count the cell itself
        self_bins = torch.bucketize(activated.clamp(0, 1), boundaries)
        self_idx = torch.arange(N, device=activated.device) * n_bins + self_bins
        bin_counts.view(-1).scatter_add_(0, self_idx, torch.ones(N, device=activated.device))

        # Shannon entropy: H = -Σ p_k log(p_k)
        total = bin_counts.sum(dim=-1, keepdim=True)
        p = bin_counts / total.clamp(min=1)
        log_p = torch.log(p.clamp(min=1e-10))
        entropy = -(p * log_p).sum(dim=-1)  # (N,)

        return entropy

    @staticmethod
    def softplus_inv(x, beta=10):
        """Numerically stable inverse of softplus."""
        return torch.where(
            beta * x > 20,
            x,
            torch.log(torch.expm1(beta * x)) / beta,
        )

    @torch.no_grad()
    def apply_bilateral_filter(self, sigma_scale, sigma_v, extent=1.0):
        """Apply bilateral filter to cell densities in-place.

        Only filters cells within [-extent, extent]^3; cells outside
        (and neighbors outside) are left untouched.
        Uses per-cell radius so sigma adapts to local cell density:
            sigma_i = sigma_scale * cell_radius_i
        """
        activated = self.get_primal_density().squeeze()  # (N,)
        points = self.primal_points
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = points.shape[0]

        # Mask: only cells inside the reconstruction volume
        inside = (points.abs() <= extent).all(dim=-1)  # (N,)

        _, cell_radius = radfoam.farthest_neighbor(
            points, self.point_adjacency, self.point_adjacency_offsets,
        )
        cr = cell_radius.squeeze()  # (N,)
        # Per-cell spatial sigma: sigma_i = sigma_scale * cell_radius_i
        sigma_sq = (sigma_scale * cr) ** 2  # (N,)

        counts = offsets[1:] - offsets[:-1]
        source = torch.repeat_interleave(
            torch.arange(N, device=points.device), counts
        )

        # Zero out edges where either endpoint is outside the volume
        edge_valid = inside[source] & inside[adj]

        # Bilateral weights: spatial (per-cell sigma) x value similarity
        d_sq = (points[adj] - points[source]).pow(2).sum(dim=-1)
        dmu = activated[source] - activated[adj]
        w = torch.exp(-d_sq / sigma_sq[source] - dmu * dmu / (sigma_v * sigma_v))
        w = w * edge_valid  # discard outside neighbors

        # Per-cell weighted average (include self with weight 1:
        # d_sq=0, dmu=0 → exp(0)=1, guarantees w_sum >= 1)
        w_sum = torch.ones(N, device=points.device).scatter_add_(0, source, w)
        w_mu = activated.clone().scatter_add_(
            0, source, w * activated[adj]
        )
        filtered = w_mu / w_sum

        # Only write back cells inside the volume
        inv = self.softplus_inv(filtered / self.activation_scale)
        self.density.data[inside, 0] = inv[inside]

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
        density_peak = getattr(self, "density_peak", None)
        delta_raw = getattr(self, "delta_raw", None)
        cov_raw = getattr(self, "cov_raw", None)

        return (points, density, point_adjacency, point_adjacency_offsets,
                density_grad, gradient_max_slope,
                density_peak, delta_raw, cov_raw)

    @torch.no_grad()
    def _get_cell_radius(self):
        """Compute per-cell radius (cached until triangulation changes)."""
        _, cell_radius = radfoam.farthest_neighbor(
            self.primal_points,
            self.point_adjacency,
            self.point_adjacency_offsets,
        )
        return cell_radius.squeeze()

    @torch.no_grad()
    def update_starvation_count(self):
        """Update per-cell starvation counter and record completed episodes.

        When a starving cell (count > 0) gets re-visited (nonzero gradient),
        its starvation length and cell radius (from cache) are recorded.
        """
        N = self.primal_points.shape[0]
        if not hasattr(self, '_starvation_count'):
            self._starvation_count = torch.zeros(N, dtype=torch.int32, device=self.device)
        if not hasattr(self, '_starvation_lifetimes'):
            self._starvation_lifetimes = []  # list of (lengths, radii) tuples

        hit = self.density.grad.squeeze().abs() > 0

        # Record completed starvation episodes with radius snapshot
        revisited = hit & (self._starvation_count > 0)
        if revisited.any():
            indices = revisited.nonzero(as_tuple=True)[0]
            self._starvation_lifetimes.append((
                self._starvation_count[indices].clone(),
                self._cached_cell_radius[indices].clone(),
            ))

        self._starvation_count[hit] = 0
        self._starvation_count[~hit] += 1

    @torch.no_grad()
    def compute_cell_importance(self):
        """Per-cell sampling weight based on inverse cross-section (1/r²).

        Small cells have small cross-section → low probability of random ray
        intersection → need more targeted sampling. Weight ∝ 1/r².
        """
        r = self._cached_cell_radius  # (N,) — updated at every triangulation rebuild
        weights = 1.0 / (r * r + 1e-12)

        # Zero out cells outside the reconstruction volume (|coord| > 1)
        inside = (self.primal_points.abs() <= 1.0).all(dim=-1)
        weights = weights * inside.float()

        # Normalize to probability distribution
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

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
        (points, density, point_adjacency, point_adjacency_offsets,
         density_grad, gradient_max_slope,
         density_peak, delta_raw, cov_raw) = self.get_trace_data()

        interpolation_mode = getattr(self, "_interpolation_mode", False)
        idw_sigma = getattr(self, "_idw_sigma", 0.01)
        idw_sigma_v = getattr(self, "_idw_sigma_v", 0.1)
        per_cell_sigma = getattr(self, "_per_cell_sigma", False)
        per_neighbor_sigma = getattr(self, "_per_neighbor_sigma", False)
        gaussian_mode = getattr(self, "_gaussian_active", False)

        # Compute cell_radius on demand when adaptive sigma or gaussian mode is active
        cell_radius = None
        if interpolation_mode and (per_cell_sigma or per_neighbor_sigma):
            cell_radius = self._get_cell_radius()
        if gaussian_mode and density_peak is not None:
            cell_radius = self._get_cell_radius()

        # When interpolation is active, suppress the linear gradient feature
        if interpolation_mode:
            density_grad = None

        # When gaussian mode is active, suppress the linear gradient feature
        if gaussian_mode:
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
            gaussian_mode,
            density_peak,
            delta_raw,
            cov_raw,
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
            warmup_steps=warmup,
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

    def initialize_gaussian(self, args):
        N = self.primal_points.shape[0]
        cell_r = self._get_cell_radius()  # (N,)
        raw_diag = self.softplus_inv(cell_r)  # sigma ~ cell_radius

        self.density_peak = nn.Parameter(
            torch.zeros(N, 1, device=self.device, dtype=torch.float32)
        )
        self.delta_raw = nn.Parameter(
            torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        )
        cov = torch.zeros(N, 6, device=self.device, dtype=torch.float32)
        cov[:, 0] = raw_diag
        cov[:, 2] = raw_diag
        cov[:, 5] = raw_diag
        self.cov_raw = nn.Parameter(cov)

        self.optimizer.add_param_group({
            "params": self.density_peak,
            "lr": args.peak_lr_init,
            "name": "density_peak",
        })
        self.optimizer.add_param_group({
            "params": self.delta_raw,
            "lr": args.offset_lr_init,
            "name": "delta_raw",
        })
        self.optimizer.add_param_group({
            "params": self.cov_raw,
            "lr": args.cov_lr_init,
            "name": "cov_raw",
        })

        self.peak_scheduler_args = get_cosine_lr_func(
            lr_init=args.peak_lr_init,
            lr_final=args.peak_lr_final,
            max_steps=self._max_iterations - args.gaussian_start,
        )
        self.offset_scheduler_args = get_cosine_lr_func(
            lr_init=args.offset_lr_init,
            lr_final=args.offset_lr_final,
            max_steps=self._max_iterations - args.gaussian_start,
        )
        self.cov_scheduler_args = get_cosine_lr_func(
            lr_init=args.cov_lr_init,
            lr_final=args.cov_lr_final,
            max_steps=self._max_iterations - args.gaussian_start,
        )
        self._gaussian_start = args.gaussian_start
        self._gaussian_active = True

        print(f"Initialized Gaussian params: {N} cells "
              f"(peak_lr={args.peak_lr_init}, offset_lr={args.offset_lr_init}, "
              f"cov_lr={args.cov_lr_init})")

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
            elif param_group["name"] == "density_peak":
                if hasattr(self, "peak_scheduler_args"):
                    lr = self.peak_scheduler_args(
                        iteration - self._gaussian_start
                    )
                    param_group["lr"] = lr
            elif param_group["name"] == "delta_raw":
                if hasattr(self, "offset_scheduler_args"):
                    lr = self.offset_scheduler_args(
                        iteration - self._gaussian_start
                    )
                    param_group["lr"] = lr
            elif param_group["name"] == "cov_raw":
                if hasattr(self, "cov_scheduler_args"):
                    lr = self.cov_scheduler_args(
                        iteration - self._gaussian_start
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
        if "density_peak" in optimizable_tensors:
            self.density_peak = optimizable_tensors["density_peak"]
        if "delta_raw" in optimizable_tensors:
            self.delta_raw = optimizable_tensors["delta_raw"]
        if "cov_raw" in optimizable_tensors:
            self.cov_raw = optimizable_tensors["cov_raw"]
        if hasattr(self, '_starvation_count'):
            self._starvation_count = self._starvation_count[valid_points_mask]
        if hasattr(self, '_frozen_mask'):
            self._frozen_mask = self._frozen_mask[valid_points_mask]

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
        n_new = new_params["primal_points"].shape[0]
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]
        if "density_peak" in optimizable_tensors:
            self.density_peak = optimizable_tensors["density_peak"]
        if "delta_raw" in optimizable_tensors:
            self.delta_raw = optimizable_tensors["delta_raw"]
        if "cov_raw" in optimizable_tensors:
            self.cov_raw = optimizable_tensors["cov_raw"]
        if hasattr(self, '_starvation_count'):
            self._starvation_count = torch.cat([
                self._starvation_count,
                torch.zeros(n_new, dtype=torch.int32, device=self.device),
            ])
        if hasattr(self, '_frozen_mask'):
            self._frozen_mask = torch.cat([
                self._frozen_mask,
                torch.zeros(n_new, dtype=torch.bool, device=self.device),
            ])

    def prune_and_densify(
        self, point_error, point_contribution, upsample_factor=1.2,
        gradient_fraction=0.4, idw_fraction=0.3,
        entropy_fraction=0.3, entropy_bins=5,
        redundancy_threshold=0.0, redundancy_cap=0.0,
        sigma_scale=0.5, sigma_v=0.1,
        variance_pruning=False, prune_hops=1,
    ):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            primal_error_accum = point_error.clip(min=0).squeeze()
            points, _, point_adjacency, point_adjacency_offsets, *_ = (
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

            # IDW-based weight (bilateral prediction error)
            cell_error = self.compute_redundancy_error(cell_radius, sigma_scale, sigma_v)
            idw_weight = (cell_error[src] + cell_error[tgt]) * edge_length

            # Per-cell neighbor entropy
            cell_entropy = self.compute_neighbor_entropy(n_bins=entropy_bins)

            ######################## Pruning ########################
            low_contrib = point_contribution.squeeze() < 1e-2
            tiny_radius = cell_radius < 1e-4
            prune_mask = low_contrib | tiny_radius
            if hasattr(self, '_frozen_mask'):
                prune_mask = prune_mask & ~self._frozen_mask
            n_pruned_low_contrib = (low_contrib & ~tiny_radius).sum().item()
            n_pruned_tiny_radius = (tiny_radius & ~low_contrib).sum().item()
            n_pruned_both = (low_contrib & tiny_radius).sum().item()
            n_redundant = 0
            n_added_gradient = 0
            n_added_idw = 0
            n_added_entropy = 0
            n_filtered_dupes = 0
            n_basic_pruned = prune_mask.sum().item()
            if n_basic_pruned > 0:
                print(f"Pruning {n_basic_pruned}/{num_curr_points} cells "
                      f"(low_contrib={n_pruned_low_contrib}, tiny_radius={n_pruned_tiny_radius}, both={n_pruned_both})")

            ################ Redundancy pruning ################
            if redundancy_cap > 0:
                if variance_pruning:
                    # Variance-based criterion: score = neighborhood_var × clamped_radius
                    # All non-pruned cells are candidates (no threshold); purely cap-based.
                    cell_score = self.compute_neighborhood_variance(
                        cell_radius, hops=prune_hops
                    )
                    candidates = ~prune_mask
                    if hasattr(self, '_frozen_mask'):
                        candidates = candidates & ~self._frozen_mask
                    prune_label = "variance"
                else:
                    # IDW leave-one-out criterion (original)
                    density_scale = torch.quantile(activated, 0.95).item()
                    cell_score = cell_error
                    candidates = cell_score < redundancy_threshold * density_scale
                    candidates = candidates & ~prune_mask
                    if hasattr(self, '_frozen_mask'):
                        candidates = candidates & ~self._frozen_mask
                    prune_label = f"IDW threshold={redundancy_threshold * density_scale:.4f}"

                if candidates.sum() > 0:
                    # Independent set: lowest-score neighbor wins (most redundant wins locally)
                    priorities = cell_score.clone()
                    priorities[~candidates] = float('inf')
                    neighbor_min = torch.full(
                        (num_curr_points,), float('inf'), device=points.device
                    ).scatter_reduce_(0, source, priorities[adj], reduce='amin')
                    removable = candidates & (priorities < neighbor_min)

                    # Cap: at most redundancy_cap fraction of total cells
                    max_remove = int(redundancy_cap * num_curr_points)
                    n_removable = removable.sum().item()
                    if n_removable > max_remove:
                        score_vals = cell_score.clone()
                        score_vals[~removable] = float('inf')
                        _, topk = score_vals.topk(max_remove, largest=False)
                        removable = torch.zeros_like(removable)
                        removable[topk] = True

                    n_redundant_here = removable.sum().item()
                    if n_redundant_here > 0:
                        n_redundant = n_redundant_here
                        print(f"Redundancy prune ({prune_label}): "
                              f"{n_redundant}/{num_curr_points} cells")
                        prune_mask = prune_mask | removable

            ######################## Sampling ########################
            perturbation = 0.25 * (points[farthest_neighbor] - points)
            delta = torch.randn_like(perturbation)
            delta /= delta.norm(dim=-1, keepdim=True)
            perturbation += (
                0.1 * perturbation.norm(dim=-1, keepdim=True) * delta
            )

            ################### Split budget ########################
            num_gradient_points = int(gradient_fraction * num_new_points)
            num_idw_points = int(idw_fraction * num_new_points)
            num_entropy_points = num_new_points - num_gradient_points - num_idw_points

            sampled_points_list = []
            sampled_inds_list = []
            sampled_density_list = []
            sampled_density_grad_list = []
            sampled_density_peak_list = []
            sampled_delta_raw_list = []
            sampled_cov_raw_list = []
            has_density_grad = hasattr(self, "density_grad") and self.density_grad is not None
            has_gaussian = hasattr(self, "density_peak") and self.density_peak is not None

            def _append_gaussian_for_inds(inds, n):
                """Append Gaussian params for sampled indices (zeros for new cells)."""
                if has_gaussian:
                    sampled_density_peak_list.append(
                        torch.zeros(n, 1, device=self.device))
                    sampled_delta_raw_list.append(
                        torch.zeros(n, 3, device=self.device))
                    # Init cov diagonal from parent cell radius
                    cr_new = cell_radius[inds].squeeze()
                    cov_new = torch.zeros(n, 6, device=self.device)
                    raw_diag = self.softplus_inv(cr_new)
                    cov_new[:, 0] = raw_diag
                    cov_new[:, 2] = raw_diag
                    cov_new[:, 5] = raw_diag
                    sampled_cov_raw_list.append(cov_new)

            def _sample_edges(weight, n_budget, counter_name):
                """Sample points along edges weighted by `weight`. Returns count added."""
                nonlocal n_added_idw
                num_viable = (weight > 0).sum().item()
                if num_viable == 0:
                    # Fallback: redirect budget to gradient strategy
                    extra_inds = torch.multinomial(
                        primal_error_accum * cell_radius,
                        n_budget,
                        replacement=False,
                    )
                    sampled_points_list.append((points + perturbation)[extra_inds])
                    sampled_inds_list.append(extra_inds)
                    sampled_density_list.append(self.density[extra_inds])
                    if has_density_grad:
                        sampled_density_grad_list.append(self.density_grad[extra_inds])
                    _append_gaussian_for_inds(extra_inds, n_budget)
                    return 0  # added to gradient fallback, not this strategy
                n_sample = min(n_budget, num_viable)
                # Filter to above-median edges to stay within multinomial limits
                candidate_idx = (weight > weight.median()).nonzero(as_tuple=True)[0]
                sub_weights = weight[candidate_idx]
                sub_inds = torch.multinomial(sub_weights, min(n_sample, candidate_idx.shape[0]), replacement=False)
                edge_inds = candidate_idx[sub_inds]
                # Radius-ratio placement: bias towards the larger cell
                p_a = points[src[edge_inds]]
                p_b = points[tgt[edge_inds]]
                r_a = cell_radius[src[edge_inds]].squeeze(-1)
                r_b = cell_radius[tgt[edge_inds]].squeeze(-1)
                t = r_b / (r_a + r_b + 1e-12)  # closer to A when A is larger
                ev = p_b - p_a
                el = ev.norm(dim=-1, keepdim=True)
                jitter = 0.10 * el * torch.randn_like(p_a)
                new_points = p_a + t.unsqueeze(-1) * ev + jitter
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
                _append_gaussian_for_inds(src[edge_inds], n_sample)
                return n_sample

            # --- Gradient-based sampling (position error × cell radius) ---
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
                _append_gaussian_for_inds(grad_inds, num_gradient_points)
                n_added_gradient += num_gradient_points

            # --- IDW-based sampling (bilateral prediction error × edge length) ---
            if num_idw_points > 0:
                n_added_idw += _sample_edges(idw_weight, num_idw_points, "idw")

            # --- Entropy-based sampling (neighbor density entropy × cell radius) ---
            if num_entropy_points > 0:
                entropy_weight = cell_entropy * cell_radius.squeeze()
                num_viable = (entropy_weight > 0).sum().item()
                if num_viable >= num_entropy_points:
                    entropy_inds = torch.multinomial(
                        entropy_weight, num_entropy_points, replacement=False,
                    )
                else:
                    entropy_inds = torch.multinomial(
                        primal_error_accum * cell_radius,
                        num_entropy_points, replacement=False,
                    )
                sampled_points_list.append((points + perturbation)[entropy_inds])
                sampled_inds_list.append(entropy_inds)
                sampled_density_list.append(self.density[entropy_inds])
                if has_density_grad:
                    sampled_density_grad_list.append(self.density_grad[entropy_inds])
                _append_gaussian_for_inds(entropy_inds, num_entropy_points)
                n_added_entropy = num_entropy_points

            sampled_inds = torch.cat(sampled_inds_list, dim=0)
            sampled_points = torch.cat(sampled_points_list, dim=0)

            # Initialize new cell densities via IDW interpolation at their positions.
            # This gives each new cell the smooth field value rather than a parent's raw density.
            result = idw_query(
                sampled_points, points,
                self.point_adjacency, self.point_adjacency_offsets,
                self.aabb_tree, activated,
                sigma=sigma_scale, sigma_v=sigma_v,
                per_cell_sigma=True, per_neighbor_sigma=True,
                cell_radius=cell_radius,
            )
            idw_activated = result.idw_result
            beta = 10.0
            raw = torch.log((idw_activated * beta).exp().clamp(min=1.0 + 1e-6) - 1.0) / beta
            sampled_density = raw.unsqueeze(-1)
            if has_density_grad:
                sampled_dg = torch.cat(sampled_density_grad_list, dim=0)

            ################### Filter near-duplicates ###################
            nn_inds = radfoam.nn(points, self.aabb_tree, sampled_points).long()
            nn_dists = (sampled_points - points[nn_inds]).norm(dim=-1)
            # Minimum separation: 5% of the source point's cell radius
            min_sep = 0.05 * cell_radius[sampled_inds].squeeze()
            keep_mask = nn_dists > min_sep

            if has_gaussian:
                sampled_peak = torch.cat(sampled_density_peak_list, dim=0)
                sampled_dr = torch.cat(sampled_delta_raw_list, dim=0)
                sampled_cov = torch.cat(sampled_cov_raw_list, dim=0)

            n_filtered_dupes = (~keep_mask).sum().item()
            if n_filtered_dupes > 0:
                print(f"Filtered {n_filtered_dupes}/{sampled_points.shape[0]} new points (too close to existing)")
                sampled_points = sampled_points[keep_mask]
                sampled_inds = sampled_inds[keep_mask]
                sampled_density = sampled_density[keep_mask]
                if has_density_grad:
                    sampled_dg = sampled_dg[keep_mask]
                if has_gaussian:
                    sampled_peak = sampled_peak[keep_mask]
                    sampled_dr = sampled_dr[keep_mask]
                    sampled_cov = sampled_cov[keep_mask]

            new_params = {
                "primal_points": sampled_points,
                "density": sampled_density,
            }
            if has_density_grad:
                new_params["density_grad"] = sampled_dg
            if has_gaussian:
                new_params["density_peak"] = sampled_peak
                new_params["delta_raw"] = sampled_dr
                new_params["cov_raw"] = sampled_cov

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
                "added_idw": n_added_idw,
                "added_entropy": n_added_entropy,
                "filtered_duplicates": n_filtered_dupes,
                "points_after": self.primal_points.shape[0],
            }

    def prune_only(self, data_handler):
        """Standalone prune pass: remove cells with negligible contribution or tiny radius."""
        _, point_contribution = self.collect_error_map(data_handler)
        with torch.no_grad():
            points, _, point_adjacency, point_adjacency_offsets, *_ = self.get_trace_data()
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

    def collect_error_map(self, data_handler, contrast_alpha=0.0):
        rays, projections = data_handler.rays, data_handler.projections

        points, *_ = self.get_trace_data()
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

            proj_output, contribution, _, _, errbox = self.forward(
                ray_batch, start_points[i], return_contribution=True
            )

            pixel_loss = proj_loss(proj_batch, proj_output)  # (H, W, 1)

            # Weight by projection contrast if enabled
            if contrast_alpha > 0:
                contrast = projection_contrast(proj_batch)  # (H, W, 1)
                pixel_loss = pixel_loss * (1.0 + contrast_alpha * contrast)

            loss = pixel_loss.mean(dim=-1)

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
        if hasattr(self, "density_peak") and self.density_peak is not None:
            scene_data["density_peak"] = self.density_peak.detach().float().cpu()
            scene_data["delta_raw"] = self.delta_raw.detach().float().cpu()
            scene_data["cov_raw"] = self.cov_raw.detach().float().cpu()
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

        if "density_peak" in scene_data:
            self.density_peak = nn.Parameter(
                scene_data["density_peak"].to(self.device)
            )
            self.delta_raw = nn.Parameter(
                scene_data["delta_raw"].to(self.device)
            )
            self.cov_raw = nn.Parameter(
                scene_data["cov_raw"].to(self.device)
            )
            self._gaussian_active = True

        self.point_adjacency = scene_data["adjacency"].to(self.device).to(
            torch.uint32)
        self.point_adjacency_offsets = scene_data["adjacency_offsets"].to(
            self.device
        ).to(torch.uint32)

        self.aabb_tree = radfoam.build_aabb_tree(self.primal_points)

    def load_frozen_checkpoint(self, pt_path, n_new_points, freeze_density=True):
        """Load xyz+density from pt_path as frozen seed, add n_new_points fresh random cells.

        After this call, self._frozen_mask[i] is True for the N_f loaded points (permuted).
        init_points in the stage config refers to n_new_points (the fresh additions only).
        """
        scene_data = torch.load(pt_path, map_location=self.device)
        frozen_xyz = scene_data["xyz"].to(self.device)      # (N_f, 3)
        frozen_den = scene_data["density"].to(self.device)  # (N_f, 1)
        N_f = frozen_xyz.shape[0]

        s = self.init_scale
        new_xyz = torch.rand(n_new_points, 3, device=self.device) * 2 * s - s
        new_den = torch.full((n_new_points, 1), self.init_density,
                             device=self.device, dtype=torch.float32)

        all_xyz = torch.cat([frozen_xyz, new_xyz])
        all_den = torch.cat([frozen_den, new_den])

        self.triangulation = radfoam.Triangulation(all_xyz.float().contiguous())
        perm = self.triangulation.permutation().to(torch.long)
        self.primal_points = nn.Parameter(all_xyz[perm])
        self.density = nn.Parameter(all_den[perm])
        self.update_triangulation(rebuild=False)

        mask = torch.zeros(N_f + n_new_points, dtype=torch.bool, device=self.device)
        mask[:N_f] = True
        self._frozen_mask = mask[perm].clone()
        self._freeze_density = freeze_density
        # Fix scheduler denominator: interval formula uses num_final_points - num_init_points
        # to estimate how many cells will be added. Actual start is N_f + n_new, not n_new.
        self.num_init_points = N_f + n_new_points
        print(f"[frozen init] {N_f} frozen + {n_new_points} new = {N_f + n_new_points} total "
              f"({N_f / (N_f + n_new_points):.1%} frozen)")

    @torch.no_grad()
    def apply_frozen_mask(self):
        """Zero gradients for all frozen points. Call after backward(), before optimizer.step()."""
        if not hasattr(self, '_frozen_mask') or not self._frozen_mask.any():
            return
        mask = self._frozen_mask
        if self.primal_points.grad is not None:
            self.primal_points.grad[mask] = 0.0
        if getattr(self, '_freeze_density', True) and self.density.grad is not None:
            self.density.grad[mask] = 0.0

    def unfreeze_all(self):
        """Remove per-point freeze. Called at frozen_unfreeze_step."""
        if hasattr(self, '_frozen_mask'):
            n = self._frozen_mask.sum().item()
            del self._frozen_mask
            print(f"[unfreeze] released {n} previously-frozen points")
