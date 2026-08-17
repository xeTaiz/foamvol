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

# K values with a verified forward+backward gradcheck. K=8 is blocked until the
# CUDA backward's fixed-size w_arr[8] path and adjoint are gradient-checked
# (see specs/SPLIT-CELL-EXPERIMENT-PLAN-v2.md, Phase 0 item 4).
_SUPPORTED_THIN_K = (4,)
# Hard cap from the backward kernel's stack buffer `w_arr[8]` (pipeline.cu).
_THIN_K_HARD_CAP = 8


def assert_supported_thin_K(K: int) -> None:
    """Validate thin_surface_K. K=4 is the only verified value for now."""
    if not isinstance(K, int) or K <= 0:
        raise ValueError(f"thin_surface_K must be a positive int, got {K!r}")
    if K > _THIN_K_HARD_CAP:
        raise ValueError(
            f"thin_surface_K={K} exceeds hard cap {_THIN_K_HARD_CAP} "
            f"(CUDA backward w_arr size)."
        )
    if K not in _SUPPORTED_THIN_K:
        raise ValueError(
            f"thin_surface_K={K} is not yet verified. Supported: "
            f"{_SUPPORTED_THIN_K}. Extend _SUPPORTED_THIN_K only after a "
            f"finite-difference gradcheck passes for that K."
        )
from radfoam_model.utils import *


def quaternion_to_normals(q: torch.Tensor) -> torch.Tensor:
    """Surface normal vectors implied by per-cell orientation quaternions.

    The thin-surface model interprets each quaternion as the rotation that
    maps the reference direction [1, 0, 0] onto the cell's outward surface
    normal (see ``initialize_thin_surface``'s half-angle warm-start, which
    builds exactly this rotation).  Rotating [1, 0, 0] by the unit quaternion
    (w, x, y, z) gives the first column of the rotation matrix:

        n = (1 - 2(y^2 + z^2),  2(x y + w z),  2(x z - w y))

    The result is renormalized so a drifted (non-unit) quaternion still maps
    to a unit direction; this is a pure diagnostic transform and does NOT
    feed any loss or the optimizer, so it cannot change rendering or math.
    """
    q = q.detach()
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    nx = 1.0 - 2.0 * (y * y + z * z)
    ny = 2.0 * (x * y + w * z)
    nz = 2.0 * (x * z - w * y)
    n = torch.stack([nx, ny, nz], dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return n


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
              cell_radius=None, hop=1):
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
        hop: neighborhood depth (1=direct neighbors only, 2=include neighbors-of-neighbors)

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

    if hop == 2:
        K1 = global_max_k
        K2_max = global_max_k

        # Gather 2-hop candidates: for each valid 1-hop neighbor, expand its CSR row
        one_hop_nbs = pad_idx[:, 1:]      # (B, K1)
        one_hop_valid = valid[:, 1:]      # (B, K1)

        hop2_starts = adj_off[one_hop_nbs]  # (B, K1)
        hop2_counts = adj_off[one_hop_nbs + 1] - hop2_starts  # (B, K1)
        hop2_counts = hop2_counts * one_hop_valid.long()

        k_range2 = torch.arange(K2_max, device=device)
        has_k2 = hop2_counts.unsqueeze(2) > k_range2[None, None, :]         # (B, K1, K2_max)
        flat_off2 = (hop2_starts.unsqueeze(2) + k_range2[None, None, :]).clamp(max=adj.shape[0] - 1)
        pad_idx_2 = adj[flat_off2].view(B, K1 * K2_max)                     # (B, K1*K2_max)
        valid_2 = has_k2.view(B, K1 * K2_max)

        pad_idx = torch.cat([pad_idx, pad_idx_2], dim=1)  # (B, 1+K1+K1*K2_max)
        valid = torch.cat([valid, valid_2], dim=1)

        # Strict dedup: mark any slot whose index already appeared in an earlier slot as invalid.
        # Invalid slots get unique sentinels above N so they never match valid indices.
        M_total = pad_idx.shape[1]
        N_pts = points.shape[0]
        col_ids = torch.arange(M_total, device=device).unsqueeze(0).expand(B, M_total)
        pad_safe = torch.where(valid, pad_idx, N_pts + col_ids)
        sorted_safe, sort_order = pad_safe.sort(dim=1, stable=True)
        is_dup = torch.zeros(B, M_total, dtype=torch.bool, device=device)
        is_dup[:, 1:] = sorted_safe[:, 1:] == sorted_safe[:, :-1]
        _, unsort = sort_order.sort(dim=1, stable=True)
        is_dup = is_dup.gather(1, unsort)
        valid = valid & ~is_dup

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

        # LC64 plan v3 -- density-mode discriminator. Defaults to
        # "scalar" (legacy behavior: density is the only degree and
        # is trained; thin-surface remains inactive). Independent
        # mode flips this to "independent" via
        # initialize_independent_sides; absolute/relative modes flip
        # to their respective labels via initialize_thin_surface.
        # Used by save_pt / load_pt for round-trip and by forward()
        # for the independent-mode fail-fast gate.
        self._thin_surface_density_mode = "scalar"

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
        self._reapply_hard_freeze()
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
        self._reapply_hard_freeze()
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
            model.set_reference_volume(model._idw_voxelize(64)[0].detach())

        Args:
            tensor: (R, R, R) float tensor (any resolution — resampled at loss time)
            edge_mask: if True, weight loss by inverse gradient magnitude of ref
            edge_alpha: sensitivity of edge mask

        Note: _ref_weight is always computed regardless of edge_mask so it can be
        used for ref-guided pruning/densification. edge_mask only controls whether
        it is applied inside reference_volume_loss().
        """
        self._ref_volume = tensor.detach().float()
        grad_mag = self._gradient_magnitude_3d(tensor).detach()
        self._ref_weight = 1.0 / (1.0 + edge_alpha * grad_mag)
        self._ref_weight_in_loss = bool(edge_mask)

    def _sample_ref_weight_at_points(self):
        """Sample _ref_weight at current primal_point positions via grid_sample.

        Returns (N,) tensor in [0,1]: high = smooth/homogeneous region, low = edge region.
        Returns None if _ref_weight is not set.
        Points outside [-1,1]³ return 0 (unknown region — no pruning bias, no densify suppression).
        """
        if getattr(self, '_ref_weight', None) is None:
            return None
        points = self.primal_points.detach()
        vol = self._ref_weight.unsqueeze(0).unsqueeze(0)              # (1,1,R,R,R)
        grid = points.flip(-1).reshape(1, 1, 1, -1, 3)                # ZYX flip — volume stored (X,Y,Z)=(D,H,W)
        sampled = F.grid_sample(
            vol, grid, mode='bilinear',
            padding_mode='zeros', align_corners=True,
        )
        return sampled.reshape(-1)                                     # (N,) in [0,1]

    @staticmethod
    def _gradient_magnitude_3d(vol):
        """3D gradient magnitude via central differences. vol: (R,R,R) → (R,R,R)."""
        v = vol.float().unsqueeze(0).unsqueeze(0)  # (1,1,R,R,R)
        k = torch.tensor([-0.5, 0.0, 0.5], device=vol.device, dtype=torch.float32)
        gx = F.conv3d(v, k.reshape(1, 1, 3, 1, 1), padding=(1, 0, 0))
        gy = F.conv3d(v, k.reshape(1, 1, 1, 3, 1), padding=(0, 1, 0))
        gz = F.conv3d(v, k.reshape(1, 1, 1, 1, 3), padding=(0, 0, 1))
        return (gx ** 2 + gy ** 2 + gz ** 2).sqrt().squeeze()

    def _idw_voxelize(self, resolution=64, supersample=1, extent=1.0, hop=1):
        """Evaluate scene density on a regular voxel grid via IDW interpolation.

        Uses the same IDW parameters as inference-mode interpolation (_idw_sigma,
        _idw_sigma_v, _per_cell_sigma, _per_neighbor_sigma), set on the model via
        set_interpolation_mode(). This means density at each sample point is the
        bilateral-weighted average of the NN cell and its Voronoi graph neighbors —
        not just the raw nearest-cell value.

        supersample=1: evaluate at each voxel center (deterministic).
        supersample>1: evaluate at k uniform random points per voxel and average.

        Returns:
            vol: (res, res, res) float tensor with gradient through density
            occupied: (res, res, res) bool — True for voxels inside [-extent, extent]³
        """
        res = resolution
        voxel_size = 2.0 * extent / res
        mu = self.get_primal_density().squeeze()  # (N,) with grad

        sigma = getattr(self, '_idw_sigma', 0.7)
        sigma_v = getattr(self, '_idw_sigma_v', None)
        per_cell_sigma = getattr(self, '_per_cell_sigma', False)
        per_neighbor_sigma = getattr(self, '_per_neighbor_sigma', False)
        cell_radius = self._cached_cell_radius if per_cell_sigma else None

        adj = self.point_adjacency
        adj_off = self.point_adjacency_offsets
        global_max_k = int((adj_off.long()[1:] - adj_off.long()[:-1]).max().item())

        # Build all voxel centers in world space
        ax = torch.arange(res, device=mu.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing='ij')
        vox_centers = (torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3) + 0.5) * voxel_size - extent

        if supersample <= 1:
            sample_pts = vox_centers.contiguous()
        else:
            rand_shifts = (torch.rand(res**3, supersample, 3, device=mu.device) - 0.5) * voxel_size
            sample_pts = (vox_centers.unsqueeze(1) + rand_shifts).reshape(-1, 3).contiguous()

        # IDW query in batches to avoid OOM on large grids
        B = sample_pts.shape[0]
        chunks = []
        batch_size = 500_000
        for start in range(0, B, batch_size):
            r = idw_query(
                sample_pts[start:start + batch_size],
                self.primal_points.detach(),
                adj, adj_off, self.aabb_tree, mu,
                sigma=sigma, sigma_v=sigma_v,
                global_max_k=global_max_k,
                per_cell_sigma=per_cell_sigma,
                per_neighbor_sigma=per_neighbor_sigma,
                cell_radius=cell_radius,
                hop=hop,
            )
            chunks.append(r.idw_result)
        sample_dens = torch.cat(chunks)  # (B,) with grad

        if supersample > 1:
            vol_flat = sample_dens.reshape(res**3, supersample).mean(dim=1)
        else:
            vol_flat = sample_dens

        occupied = (vox_centers.abs() <= extent).all(dim=1).reshape(res, res, res)
        return vol_flat.reshape(res, res, res), occupied

    @torch.no_grad()
    def voxelize_per_cell_field(self, field: torch.Tensor, resolution: int = 128,
                                extent: float = 1.0) -> torch.Tensor:
        """Nearest-cell scatter of a per-cell scalar field onto a regular voxel grid.

        Much cheaper than _idw_voxelize — assigns each voxel center to its nearest
        cell via NN lookup and reads off the cell's field value directly.  Used for
        diagnostic purposes (per-cell field vs 3D error spatial correlations).

        Args:
            field:      (N,) or (N,1) tensor of per-cell scalar values.
            resolution: voxel grid side length R; output is (R, R, R).
            extent:     voxel centers sampled from [-extent, extent]³.

        Returns:
            (R, R, R) float32 CPU tensor.
        """
        res = resolution
        device = self.primal_points.device
        voxel_size = 2.0 * extent / res
        ax = torch.arange(res, device=device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing='ij')
        vox_centers = (
            torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3) + 0.5
        ) * voxel_size - extent
        nn_idx = radfoam.nn(self.primal_points.detach(), self.aabb_tree,
                            vox_centers).long()
        field_f = field.to(device=device, dtype=torch.float32).detach().squeeze()
        return field_f[nn_idx].reshape(res, res, res).cpu()

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

        supersample = getattr(self, '_ref_vol_supersample', 1)
        vol, occupied = self._idw_voxelize(resolution, supersample=supersample)  # (R,R,R), (R,R,R) bool

        ref = self._ref_volume
        if ref.shape[0] != resolution:
            ref = F.interpolate(
                ref.unsqueeze(0).unsqueeze(0),
                size=resolution, mode="trilinear", align_corners=True,
            ).squeeze()

        diff = vol - ref  # (R,R,R)

        if getattr(self, '_ref_weight_in_loss', False) and getattr(self, '_ref_weight', None) is not None:
            w = self._ref_weight
            if w.shape[0] != resolution:
                w = F.interpolate(
                    w.unsqueeze(0).unsqueeze(0),
                    size=resolution, mode="trilinear", align_corners=True,
                ).squeeze()
            return (w[occupied] * diff[occupied] ** 2).mean()

        return diff[occupied].pow(2).mean()

    def _idw_query_at(self, query, hop=1):
        """Differentiable IDW density at arbitrary [B, 3] query points.

        Unlike _idw_voxelize, primal_points are NOT detached so that
        gradients flow through point positions (spatial weights) as well as
        density values — enabling the position-update signal in volume training.
        """
        mu = self.get_primal_density().squeeze()

        sigma = getattr(self, '_idw_sigma', 0.7)
        sigma_v = getattr(self, '_idw_sigma_v', None)
        per_cell_sigma = getattr(self, '_per_cell_sigma', False)
        per_neighbor_sigma = getattr(self, '_per_neighbor_sigma', False)
        cell_radius = self._cached_cell_radius if per_cell_sigma else None

        adj = self.point_adjacency
        adj_off = self.point_adjacency_offsets

        B = query.shape[0]
        chunk_size = 500_000
        chunks = []
        for start in range(0, B, chunk_size):
            result = idw_query(
                query[start:start + chunk_size],
                self.primal_points,
                adj, adj_off, self.aabb_tree, mu,
                sigma=sigma, sigma_v=sigma_v,
                per_cell_sigma=per_cell_sigma,
                per_neighbor_sigma=per_neighbor_sigma,
                cell_radius=cell_radius,
                hop=hop,
            )
            chunks.append(result.idw_result)
        return torch.cat(chunks)

    def collect_error_map_volume(self, vol_gt_5d, n_query=4_000_000,
                                  batch_size=1_000_000, extent=1.0):
        """Volume-based alternative to collect_error_map.

        Samples n_query random 3D points, back-props |IDW_pred - GT|, and
        returns per-cell position-gradient norms and a normalized sample count.

        The contribution is normalized so that an average-sized cell has
        contribution ~1.0, making it compatible with the < 1e-2 pruning
        threshold used in prune_and_densify.
        """
        import math
        self.optimizer.zero_grad(set_to_none=True)
        n_points = self.primal_points.shape[0]
        contribution = torch.zeros(n_points, device=self.device)

        total = 0
        while total < n_query:
            bs = min(batch_size, n_query - total)
            query = (torch.rand(bs, 3, device=self.device) * 2 - 1) * extent

            mu_pred = self._idw_query_at(query)
            grid = (query / extent).flip(-1)[None, None, None]
            mu_gt = F.grid_sample(
                vol_gt_5d, grid, mode='bilinear',
                align_corners=True, padding_mode='zeros',
            ).reshape(-1).detach()

            (mu_pred - mu_gt).abs().sum().backward()

            with torch.no_grad():
                nn_idx = radfoam.nn(
                    self.primal_points.detach(), self.aabb_tree, query
                ).long()
                contribution.scatter_add_(0, nn_idx, torch.ones(bs, device=self.device))

            total += bs

        grad = (self.primal_points.grad if self.primal_points.grad is not None
                else torch.zeros_like(self.primal_points))
        point_error = grad.norm(dim=-1, keepdim=True).clamp_min_(0).detach()
        contribution_norm = (contribution / max(n_query, 1) * n_points).unsqueeze(-1)

        self.optimizer.zero_grad(set_to_none=True)
        return point_error, contribution_norm

    def prune_only_volume(self, vol_gt_5d, n_query=2_000_000,
                          batch_size=1_000_000, extent=1.0):
        """Standalone prune pass using volume-based sample-count contribution."""
        _, point_contribution = self.collect_error_map_volume(
            vol_gt_5d, n_query=n_query, batch_size=batch_size, extent=extent)
        with torch.no_grad():
            points, _, point_adjacency, point_adjacency_offsets, *_ = self.get_trace_data()
            _, cell_radius = radfoam.farthest_neighbor(
                points, point_adjacency, point_adjacency_offsets,
            )
            prune_mask = torch.logical_or(
                point_contribution.squeeze() < 1e-2,
                cell_radius < 1e-3,
            )
            n_pruned = prune_mask.sum().item()
            if n_pruned > 0:
                print(f"Standalone prune: {n_pruned}/{points.shape[0]} cells")
                self.prune_points(prune_mask)
                self.update_triangulation(incremental=False)
            return n_pruned

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
        self._reapply_hard_freeze()  # Re-apply after primal-points replacement
        # LC64 plan v3 -- base density may have been frozen out of
        # the optimizer (independent mode); restore the latest
        # permuted tensor only if it was registered.
        if "density" in optimizable_tensors:
            self.density = optimizable_tensors["density"]
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]
        if "density_peak" in optimizable_tensors:
            self.density_peak = optimizable_tensors["density_peak"]
        if "delta_raw" in optimizable_tensors:
            self.delta_raw = optimizable_tensors["delta_raw"]
        if "cov_raw" in optimizable_tensors:
            self.cov_raw = optimizable_tensors["cov_raw"]
        # LC64 plan v3 -- permute the independent raw sides in lock-step
        # so they stay aligned with primal_points after a triangulation
        # rebuild. The optimizer group identity is also rebuilt so a
        # later optimizer.step still reaches the new tensor.
        if "raw_plus" in optimizable_tensors:
            self.raw_plus = optimizable_tensors["raw_plus"]
        if "raw_minus" in optimizable_tensors:
            self.raw_minus = optimizable_tensors["raw_minus"]
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
        if needs_permute or failures > 3:
            # Any cell-row permutation or in-place perturbation invalidates the
            # cached Delaunay-edge/Voronoi-face correspondence.
            self._thin_surface_face_cache = None
            self._thin_surface_face_cache_signature = None
        self._reapply_hard_freeze()

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

    def voxel_variance_regularization(self, resolution=32, sigma_v=0.2, extent=1.0,
                                       supersample=1):
        """Bilateral variance loss on a randomly-offset voxel grid.

        Assigns cells to voxels via a stochastic grid offset, then estimates each
        voxel's density using full bilateral IDW (NN cell + graph neighbors, same
        spatial/bilateral params as interpolation mode) at sample points within the voxel.
        Each cell is penalized for deviating from its voxel's estimated mean, bilaterally
        weighted so cells near density edges are smoothed less aggressively.

        supersample=1: evaluate at the voxel center (deterministic single sample).
        supersample>1: k random points per voxel — better Monte-Carlo estimator,
            gradient flows through multiple contributing cells per voxel.

        The stochastic grid offset randomizes cell-to-voxel assignment across iterations,
        preventing persistent artifacts. For k>1 the random sample points add further
        decorrelation within each voxel.

        Args:
            resolution: grid resolution per axis
            sigma_v: bilateral value sigma (large = plain smoothing, small = edge-preserving)
            extent: half-extent of the volume (grid spans [-extent, extent]^3)
            supersample: IDW sample points per voxel (1 = deterministic center)
        """
        res = resolution
        voxel_size = 2.0 * extent / res
        points = self.primal_points.detach()  # (N, 3) — no grad through positions
        mu = self.get_primal_density().squeeze()  # (N,) — grad through density

        # Random grid offset for stochastic binning
        offset = (torch.rand(3, device=points.device) - 0.5) * voxel_size

        # Voxel assignment for all inside cells
        shifted = points + offset
        inside = (points.abs() <= extent).all(dim=1)
        shifted_inside = shifted[inside]
        mu_inside = mu[inside]
        grid_coords = ((shifted_inside + extent) / voxel_size).long().clamp(0, res - 1)
        voxel_idx = grid_coords[:, 0] * res * res + grid_coords[:, 1] * res + grid_coords[:, 2]

        # IDW-NN path: k sample points per occupied voxel (k=1 → voxel center, k>1 → random)
        unique_vox_ids, inverse = torch.unique(voxel_idx, return_inverse=True)
        M = unique_vox_ids.shape[0]

        vox_z = unique_vox_ids % res
        vox_y = (unique_vox_ids // res) % res
        vox_x = unique_vox_ids // (res * res)
        vox_3d = torch.stack([vox_x, vox_y, vox_z], dim=1).float()

        # Voxel centers in world space (undo the stochastic grid offset)
        vox_centers = (vox_3d + 0.5) * voxel_size - extent - offset  # (M, 3)

        k = max(1, supersample)
        if k <= 1:
            sample_pts = vox_centers.contiguous()  # (M, 3)
        else:
            rand_shifts = (torch.rand(M, k, 3, device=mu.device) - 0.5) * voxel_size
            sample_pts = (vox_centers.unsqueeze(1) + rand_shifts).reshape(-1, 3).contiguous()

        # Full bilateral IDW (NN cell + graph neighbors) — same params as interpolation mode
        idw_sigma = getattr(self, '_idw_sigma', 0.7)
        idw_sigma_v = getattr(self, '_idw_sigma_v', None)
        per_cell_sigma = getattr(self, '_per_cell_sigma', False)
        per_neighbor_sigma = getattr(self, '_per_neighbor_sigma', False)
        cell_radius = self._cached_cell_radius if per_cell_sigma else None

        adj = self.point_adjacency
        adj_off = self.point_adjacency_offsets
        global_max_k = int((adj_off.long()[1:] - adj_off.long()[:-1]).max().item())

        B_pts = sample_pts.shape[0]
        chunks = []
        batch_size = 500_000
        for start in range(0, B_pts, batch_size):
            r = idw_query(
                sample_pts[start:start + batch_size],
                self.primal_points.detach(),
                adj, adj_off, self.aabb_tree, mu,
                sigma=idw_sigma, sigma_v=idw_sigma_v,
                global_max_k=global_max_k,
                per_cell_sigma=per_cell_sigma,
                per_neighbor_sigma=per_neighbor_sigma,
                cell_radius=cell_radius,
            )
            chunks.append(r.idw_result)
        sample_densities = torch.cat(chunks)  # (M,) or (M*k,) with grad

        if k > 1:
            sample_densities = sample_densities.reshape(M, k).mean(dim=1)  # (M,)

        cell_mean = sample_densities[inverse]  # (N_inside,) with grad

        diff = mu_inside - cell_mean
        bilateral_w = torch.exp(-diff.detach() ** 2 / (sigma_v ** 2))
        return (bilateral_w * diff ** 2).mean()

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

    def _neighbor_median_target(self, mu_detached, hops):
        """K-hop median density target via iterated 1-hop median passes (O(k×E)).

        Each hop replaces each cell's value with the median of itself + immediate
        neighbors. More robust than mean to outlier cells (shot noise): a single
        noisy cell among correct neighbors gets pulled toward the correct value
        even if one other neighbor is also noisy, since the majority is clean.
        """
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = mu_detached.shape[0]
        device = mu_detached.device

        deg = offsets[1:] - offsets[:-1]  # (N,) neighbor counts (excl. self)
        max_deg = int(deg.max().item())

        # Build padded index matrix (N, max_deg+1): col 0 = self, cols 1..deg = neighbors
        padded_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, max_deg + 1).clone()
        k_range = torch.arange(max_deg, device=device)
        has_k = deg.unsqueeze(1) > k_range.unsqueeze(0)              # (N, max_deg)
        flat_off = (offsets[:-1].unsqueeze(1) + k_range.unsqueeze(0)).clamp(max=adj.shape[0] - 1)
        padded_idx[:, 1:] = torch.where(has_k, adj[flat_off], padded_idx[:, 1:])

        valid_count = (deg + 1).clamp(min=1)  # self + true neighbors
        # is_padding[i, k] = True for slots beyond the real deg[i]+1 entries
        is_padding = torch.arange(max_deg + 1, device=device).unsqueeze(0) >= valid_count.unsqueeze(1)

        smooth = mu_detached.clone()
        for _ in range(hops):
            vals = smooth[padded_idx]                    # (N, max_deg+1)
            vals = vals + is_padding.float() * 1e6       # push padding above real densities
            sorted_vals, _ = vals.sort(dim=1)
            mid_idx = ((valid_count - 1) // 2).clamp(min=0)  # 0-indexed median position
            smooth = sorted_vals.gather(1, mid_idx.unsqueeze(1)).squeeze(1)

        return smooth  # (N,), detached

    def neighbor_variance_regularization(self, sigma_v=1.0, hops=1,
                                          reg_type='bilateral_var', huber_delta=0.1):
        """Variance-style loss over the Voronoi neighbor graph.

        Each cell is penalized for deviating from its k-hop neighborhood mean or median.

        reg_type options:
          'bilateral_var' (default): bilateral-weighted L2 vs. k-hop mean. sigma_v controls
             the edge-preservation boundary; large = plain smoothing, small = edge-preserving.
          'huber': Huber loss on the residual vs. k-hop mean. Aggressively kills sub-delta
             noise while capping large residuals at edges. huber_delta sets the noise scale.
          'bilateral_huber': Huber × bilateral weight — both outlier robustness AND
             edge preservation. Strongest regularizer; may over-smooth at surfaces.
          'median': L1 loss vs. k-hop iterated median target. Median is robust to outlier
             neighbors — a single noisy cell does not pull the target for its clean neighbors.
          'bilateral_median': L1 × bilateral weight vs. k-hop median. Edge-preserving
             version of 'median'.
        """
        mu = self.get_primal_density().squeeze()        # [N], with grad

        # Choose target: mean-based or median-based
        if reg_type in ('median', 'bilateral_median'):
            target = self._neighbor_median_target(mu.detach(), hops)
        else:
            target = self._neighbor_smooth_target(mu.detach(), hops)

        diff = mu - target

        if reg_type == 'bilateral_var':
            bilateral_w = torch.exp(-(diff.detach() ** 2) / (sigma_v ** 2))
            return (bilateral_w * diff ** 2).mean()
        elif reg_type == 'huber':
            return F.huber_loss(mu, target.detach(), delta=huber_delta, reduction='mean')
        elif reg_type == 'bilateral_huber':
            bilateral_w = torch.exp(-(diff.detach() ** 2) / (sigma_v ** 2))
            huber = F.huber_loss(mu, target.detach(), delta=huber_delta, reduction='none')
            return (bilateral_w * huber).mean()
        elif reg_type == 'median':
            return diff.abs().mean()
        elif reg_type == 'bilateral_median':
            bilateral_w = torch.exp(-(diff.detach() ** 2) / (sigma_v ** 2))
            return (bilateral_w * diff.abs()).mean()
        else:
            raise ValueError(f"Unknown neighbor reg_type: {reg_type}")

    def _boundary_top_eigvec(self, sigma_v: float):
        """Shared helper for top-eigvec-based boundary losses (A and C).

        Builds M_i = Σ_j w_ij n_ij n_ij^T (weighted scatter of high-jump edge directions),
        normalises by trace for numerical stability, then extracts the top eigenvector v_i
        via eigh.  Gradients flow into primal_points through n_ij and through eigh; all
        density-derived weights are detached.

        Returns:
            v      (N, 3)  top eigenvector of M_i per cell
            valid  (N,)    bool, tr(M_i) > 0.01 * median(tr)
            sim    (E,)    same-density gate exp(-Δμ²/σ_v²), detached
            src    (E,)    long, source cell index per directed edge
            adj    (E,)    long, destination cell index per directed edge
        """
        points = self.primal_points                                         # (N, 3)
        mu     = self.get_primal_density().detach()                         # (N, 1)
        cr     = self._cached_cell_radius                                   # (N,)

        offsets = self.point_adjacency_offsets.long()
        adj     = self.point_adjacency.long()
        counts  = offsets[1:] - offsets[:-1]
        N       = points.shape[0]
        src     = torch.repeat_interleave(
                      torch.arange(N, device=points.device), counts)        # (E,)

        dx = points[adj] - points[src]                                      # (E, 3)
        n  = dx / dx.norm(dim=-1, keepdim=True).clamp_min(1e-12)           # (E, 3)

        dmu_sq = (mu[adj] - mu[src]).pow(2).squeeze(-1)                     # (E,) detached
        w_face = (cr[src] * cr[adj]).detach()                               # (E,) detached
        w      = dmu_sq * w_face                                            # (E,)

        # Scatter outer products into per-cell M_i; gradients flow through n → points.
        outer_flat = (w[:, None, None] * (n.unsqueeze(2) * n.unsqueeze(1))).reshape(-1, 9)
        M_flat = torch.zeros((N, 9), device=points.device,
                              dtype=points.dtype).index_add(0, src, outer_flat)
        M = M_flat.reshape(N, 3, 3)                                         # (N, 3, 3)

        tr    = M.diagonal(dim1=-2, dim2=-1).sum(-1)                        # (N,)
        tau   = (tr.detach().median() * 0.01).clamp_min(1e-12)
        valid = tr.detach() > tau                                            # (N,)

        # Normalise M so Rayleigh quotients are in [0, 1].
        M_hat = M / tr.detach().unsqueeze(-1).unsqueeze(-1).clamp_min(1e-12)

        # Estimate top eigenvector via power iteration on the detached matrix.
        # Gradients must NOT flow through the iteration itself (that would give
        # an 8-step unrolled graph); instead the losses route gradients through
        # M_hat via Rayleigh quotients, using v only as a detached probe direction.
        with torch.no_grad():
            v = torch.randn(N, 3, device=points.device, dtype=points.dtype)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for _ in range(8):
                v = torch.einsum('nij,nj->ni', M_hat, v)
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        sim = torch.exp(-(mu[src] - mu[adj]).pow(2).squeeze(-1) / (sigma_v ** 2))

        # Cache detached per-cell intermediates for diagnostic correlation logging.
        self._last_M_trace    = tr.detach()     # (N,) boundary edge strength
        self._last_M_valid    = valid           # (N,) bool — cells near density boundary
        self._last_top_eigvec = v               # (N, 3) — already detached (power-iter)

        return v, M_hat, valid, sim, src, adj

    def top_eigvec_alignment_regularization(self, sigma_v: float = 0.2) -> torch.Tensor:
        """Loss A: pairwise top-eigvec alignment between same-density Voronoi neighbors.

        For each edge (i,j) measures how well j's probe direction explains i's M_hat via
        the Rayleigh quotient R = v_j^T M_hat_i v_j (∈ [0,1] for normalised M_hat).
        Loss = 1 − (R_ij + R_ji)/2.  Zero when probe directions are top eigenvectors of
        each other's M_hat (i.e. surface normals agree); one when they are perpendicular.

        Gradients flow through M_hat (→ n_ij → primal_points); v is a detached probe
        so no eigenvalue-gap blow-up.  Still a first-order pairwise loss — fires on
        both kinks and smooth curvature.
        """
        v, M_hat, valid, sim, src, adj = self._boundary_top_eigvec(sigma_v)

        # Rayleigh quotient: how well does each cell's probe explain the partner's M_hat?
        Mv_adj = torch.einsum('eij,ej->ei', M_hat[src], v[adj])            # (E, 3)
        R_adj_in_src = (Mv_adj * v[adj]).sum(dim=-1)                       # (E,)

        Mv_src = torch.einsum('eij,ej->ei', M_hat[adj], v[src])            # (E, 3)
        R_src_in_adj = (Mv_src * v[src]).sum(dim=-1)                       # (E,)

        loss  = 1.0 - (R_adj_in_src + R_src_in_adj) * 0.5                 # (E,)
        mask  = (valid[src] & valid[adj]).float()

        # Skip cells where the thin-surface sub-cell geometry is already active
        # (height norm > τ) — they don't need bisector alignment on top.
        if getattr(self, "_thin_surface_active", False) and hasattr(self, "texel_heights"):
            with torch.no_grad():
                h_norm = self.texel_heights.detach().abs().sum(dim=-1)  # (N,)
                thin_tau = getattr(self, "_thin_surface_gate_tau", 0.01)
                active_ts = (h_norm > thin_tau).float()
            ts_mask = 1.0 - (active_ts[src].clamp(max=1.0) + active_ts[adj].clamp(max=1.0)).clamp(max=1.0)
            mask = mask * ts_mask

        denom = mask.sum().clamp_min(1.0)
        return (mask * sim * loss).sum() / denom

    def normal_laplacian_regularization(self, sigma_v: float = 0.2) -> torch.Tensor:
        """Loss C: graph-Laplacian smoothness on the surface-normal direction field.

        Computes a sign-aligned neighbourhood-mean probe v̄_i from detached power-
        iteration eigenvectors, then measures 1 − v̄_i^T M_hat_i v̄_i (Rayleigh
        quotient of the mean direction against the cell's own boundary tensor).

        On a smoothly curved iso-surface v̄_i ≈ v_i ≈ top eigvec of M_hat_i, so the
        Rayleigh quotient ≈ 1 and loss ≈ 0.  On a kink v̄_i averages across the
        direction discontinuity, reducing the quotient and firing the loss.
        Unlike pairwise losses (BA, A) this does NOT penalise smooth curvature.

        Gradients flow through M_hat (→ n_ij → primal_points); v and v̄ are detached.
        """
        v, M_hat, valid, sim, src, adj = self._boundary_top_eigvec(sigma_v)
        N = v.shape[0]

        # Sign-align each neighbour probe to the same hemisphere as the source.
        dot_sg = (v[src] * v[adj]).sum(dim=-1)                             # (E,)
        sigma  = dot_sg.sign().masked_fill(dot_sg == 0, 1.0)               # (E,) ∈ {-1,+1}

        # Weighted mean of sign-aligned probes: v̄_i (fully detached).
        w = sim * valid[adj].float()                                        # (E,)
        v_num = torch.zeros((N, 3), device=v.device, dtype=v.dtype).index_add(
                    0, src, (w * sigma).unsqueeze(-1) * v[adj])
        v_den = torch.zeros(N, device=v.device, dtype=v.dtype).index_add(0, src, w)

        has_nbrs = v_den > 1e-8
        v_bar    = v_num / v_den.unsqueeze(-1).clamp_min(1e-8)             # (N, 3), detached
        # Normalize before Rayleigh quotient so ||v̄|| == 1; otherwise a flat
        # boundary with slightly-varying local normals gives ||v̄|| < 1 → loss > 0
        # even without any kink, collapsing the tolerance advantage over loss A.
        v_bar    = v_bar / v_bar.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Rayleigh quotient: 1 − v̄_i^T M_hat_i v̄_i.
        Mv_bar   = torch.einsum('nij,nj->ni', M_hat, v_bar)                # (N, 3)
        R        = (Mv_bar * v_bar).sum(dim=-1)                            # (N,)
        residual = 1.0 - R                                                  # (N,)

        self._last_normal_lap_residual = residual.detach()                  # (N,) cached for diag

        active = (valid & has_nbrs).float()

        # Skip cells where the thin-surface sub-cell geometry is already active.
        if getattr(self, "_thin_surface_active", False) and hasattr(self, "texel_heights"):
            with torch.no_grad():
                h_norm = self.texel_heights.detach().abs().sum(dim=-1)  # (N,)
                thin_tau = getattr(self, "_thin_surface_gate_tau", 0.01)
                active_ts = (h_norm > thin_tau).float()
            active = active * (1.0 - active_ts.clamp(max=1.0))

        denom  = active.sum().clamp_min(1.0)
        return (active * residual).sum() / denom

    def build_thin_surface_face_cache(
            self, num_samples: int = 12, domain_extent: float = 1.0,
            max_vertices: int = 32):
        """Construct/reuse the exact finite-face cache on the current GPU.

        When ``primal_points`` is trainable (not hard-frozen), the cached
        geometry can go silently stale: Adam mutates the parameter storage
        in place, so ``data_ptr()``/shape alone cannot detect that points
        moved between calls. In that case always rebuild from the live
        triangulation (~0.07s, no re-triangulation) rather than reuse the
        signature-keyed cache. Frozen points keep the cheap reuse path.
        """
        if not getattr(self, "_thin_surface_active", False):
            raise RuntimeError("face continuity requires active thin surfaces")
        required = ("density_delta", "quaternions", "texel_sites_2d",
                    "texel_heights", "_cached_cell_radius")
        if any(getattr(self, name, None) is None for name in required):
            raise RuntimeError("face continuity thin-surface state is incomplete")

        from radfoam_model.face_continuity import build_voronoi_face_cache
        points_trainable = bool(self.primal_points.requires_grad)
        signature = (self.primal_points.data_ptr(), self.primal_points.shape[0],
                     int(num_samples), float(domain_extent), int(max_vertices))
        cache = getattr(self, "_thin_surface_face_cache", None)
        stale = (cache is None
                 or getattr(self, "_thin_surface_face_cache_signature", None) != signature
                 or points_trainable)
        if stale:
            # CTScene already permutes every parameter row into the live
            # triangulation's internal sorted order. Therefore live tets index
            # primal_points directly; applying triangulation.permutation again
            # would be a double permutation.
            identity = torch.arange(
                self.primal_points.shape[0], device=self.primal_points.device)
            cache = build_voronoi_face_cache(
                self.primal_points.detach(), self.triangulation.tets(), identity,
                num_samples=int(num_samples), domain_extent=float(domain_extent),
                max_vertices=int(max_vertices),
            )
            self._thin_surface_face_cache = cache
            self._thin_surface_face_cache_signature = signature
            rebuild_count = getattr(self, "_thin_surface_face_cache_rebuilds", 0) + 1
            self._thin_surface_face_cache_rebuilds = rebuild_count
            if not points_trainable or rebuild_count <= 1 or rebuild_count % 200 == 0:
                print(
                    f"[face-continuity] GPU cache rebuild #{rebuild_count} "
                    f"(points_trainable={points_trainable}): "
                    f"{cache.num_faces:,} faces from "
                    f"{cache.num_finite_tets:,} finite tets in "
                    f"{cache.build_seconds:.3f}s")
        return cache

    def thin_surface_face_continuity_regularization(
            self, step: int, density_scale: float, **kwargs):
        """Robust continuity of meaningful split surfaces across shared faces."""
        from radfoam_model.face_continuity import face_continuity_loss
        cache = self.build_thin_surface_face_cache(
            num_samples=kwargs.pop("num_samples", 12),
            domain_extent=kwargs.pop("domain_extent", 1.0),
            max_vertices=kwargs.pop("max_vertices", 32))
        loss, diagnostics = face_continuity_loss(
            self, cache, step=step, density_scale=density_scale, **kwargs)
        diagnostics.update({
            "cache_faces": float(cache.num_faces),
            "cache_build_seconds": float(cache.build_seconds),
        })
        return loss, diagnostics

    def cvt_regularization(self, hops: int = 1) -> torch.Tensor:
        """CVT centroidal regularization (Laplacian/Lloyd proxy).

        Pulls each point toward the mean of its k-hop Delaunay neighbours with
        the target fully detached, equivalent to one SGD step of Lloyd's algorithm.
        Loss is normalized by r_i² (cached cell radius squared) so the weight is
        scale-invariant across point counts and scene scales.

        Caches self._last_cvt_residual = ||p_i − centroid_i|| / r_i (N,) for
        per-cell correlation diagnostics.
        """
        pts = self.primal_points                               # (N, 3), gradient flows here
        N = pts.shape[0]
        device = pts.device

        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        counts = (offsets[1:] - offsets[:-1]).float().clamp(min=1)
        src = torch.repeat_interleave(
            torch.arange(N, device=device),
            offsets[1:] - offsets[:-1],
        )

        # K-hop centroid estimate via iterated mean (detached neighbour positions).
        centroid = pts.detach().clone()
        for _ in range(hops):
            nbr_sum = torch.zeros(N, 3, device=device, dtype=centroid.dtype)
            nbr_sum.index_add_(0, src, centroid[adj])
            centroid = nbr_sum / counts.unsqueeze(-1)          # (N, 3), fully detached

        r2 = self._cached_cell_radius.to(device=device, dtype=pts.dtype).detach() ** 2
        r2 = r2.clamp_min(1e-12)

        diff = pts - centroid                                  # (N, 3); gradient into pts
        sq_dist = (diff * diff).sum(dim=-1)                   # (N,)

        # Cache per-cell residual (displacement / radius) for diag correlations.
        self._last_cvt_residual = (sq_dist / r2).sqrt().detach()   # (N,)

        loss = (sq_dist / r2).mean()
        return loss

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

    def smooth_density_grad(self, hops=1, eps=1e-12):
        """Agreement-weighted density gradient smoothing (in-place).

        For each cell, compute the neighborhood mean (including self) of density.grad
        and scale the cell's own gradient by w = min(1, |mean| / (|own| + eps)).
        w ∈ [0, 1] acts as a trust weight:
          - Coherent region (own ≈ mean): w ≈ 1, gradient preserved.
          - Zero-mean noise (mean ≈ 0): w ≈ 0, noisy gradient suppressed.
          - Empty cell near active region (|mean| > |own|): w = 1 (capped), grad unchanged.
          - Noisy cell in otherwise-zero region (|mean| < |own|): w < 1, magnitude capped.

        Never boosts a gradient beyond its own magnitude — the weight is a shrinkage
        factor, not an amplifier. Applied to density.grad only, after backward() and
        before optimizer.step().

        Restricted to cells within the nominal scene volume [-1, 1]^3. Outside cells
        (background) keep their original gradient unchanged, avoiding interactions
        between large background cells and interior signal.

        Iterated `hops` times — each hop recomputes the mean from the updated grad.
        """
        if self.density.grad is None:
            return
        grad = self.density.grad.squeeze()  # (N,) view into density.grad
        N = grad.shape[0]

        # Bail out if adjacency is stale (e.g. right after densification before
        # update_triangulation rebuilds); smoothing on mismatched shapes is unsafe.
        if self.point_adjacency_offsets.shape[0] != N + 1:
            return

        # Clean NaN/Inf at entry (uses returned tensor for unambiguous semantics).
        grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        offsets = self.point_adjacency_offsets.long()
        # Clamp adjacency indices to [0, N-1] as a defensive guard.
        adj = self.point_adjacency.long().clamp_(0, N - 1)

        deg = offsets[1:] - offsets[:-1]
        counts = deg.float() + 1.0  # include self in the neighborhood
        src = torch.repeat_interleave(
            torch.arange(N, device=grad.device), deg,
        )

        # Inside-volume mask: the nominal scene extent is [-1, 1]^3.
        pts = self.primal_points.detach()
        inside = (pts.abs() <= 1.0).all(dim=1)

        current = grad_clean.clone()
        for _ in range(hops):
            # Neighborhood mean including self: (g_i + Σ_j g_j) / (1 + deg_i)
            g_sum = current.clone()
            g_sum.scatter_add_(0, src, current[adj])
            g_mean = g_sum / counts

            # Agreement weight, clamped to [0, 1].
            w = (g_mean.abs() / (current.abs() + eps)).clamp_(max=1.0)
            smoothed = w * current

            # Only update cells whose centers lie inside [-1, 1]^3.
            current = torch.where(inside, smoothed, current)

        # Safety net: if anything non-finite slipped through, fall back for those entries.
        bad = ~current.isfinite()
        if bad.any():
            n_bad = int(bad.sum().item())
            print(f"[smooth_density_grad] WARN: {n_bad}/{N} non-finite after smoothing "
                  f"— falling back to cleaned raw grad for those cells")
            current = torch.where(bad, grad_clean, current)
            current = torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)

        # Copy into the existing grad buffer to preserve its identity.
        self.density.grad.copy_(current.unsqueeze(1))

        # Stash agreement weights for periodic diagnostic visualization.
        # Outside cells keep their gradient unchanged (effective weight = 1).
        if hops > 0:
            eff_w = torch.where(inside, w, torch.ones(N, device=w.device))
        else:
            eff_w = torch.ones(N, device=grad.device)
        self._last_grad_weights = eff_w.detach().cpu()

    @torch.no_grad()
    def compute_redundancy_error(self, cell_radius, sigma_scale, sigma_v):
        """Per-cell leave-one-out IDW error: |density_i - interp_from_neighbors|."""
        activated = self.get_primal_density().squeeze()  # (N,)
        points = self.primal_points                       # (N, 3)
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = points.shape[0]

        # Intentionally uses scale × median(radius), not interp_sigma_abs.
        # Pruning quality is relative to local mesh density: a cell is redundant when its
        # neighbors (at the current mesh resolution) predict it well. Fixing to a physical
        # absolute sigma would penalize fine-mesh regions more aggressively than coarse ones.
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
        density_delta = getattr(self, "density_delta", None)
        quaternions = getattr(self, "quaternions", None)
        texel_sites_2d = getattr(self, "texel_sites_2d", None)
        texel_heights = getattr(self, "texel_heights", None)
        # LC64 plan v3 -- independent-side raw logits. Always present in
        # the tuple (None when not registered) so forward() can unpack
        # unconditionally and the ABI is stable across modes.
        raw_plus = getattr(self, "raw_plus", None)
        raw_minus = getattr(self, "raw_minus", None)

        return (points, density, point_adjacency, point_adjacency_offsets,
                density_grad, gradient_max_slope,
                density_peak, delta_raw, cov_raw,
                density_delta, quaternions, texel_sites_2d, texel_heights,
                raw_plus, raw_minus)

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

        if getattr(self, "_thin_surface_density_mode", "scalar") == "independent":
            gp = getattr(self.raw_plus, "grad", None)
            gm = getattr(self.raw_minus, "grad", None)
            if gp is None and gm is None:
                return
            gp_abs = (gp.squeeze(-1).abs() if gp is not None
                      else torch.zeros(N, device=self.device))
            gm_abs = (gm.squeeze(-1).abs() if gm is not None
                      else torch.zeros(N, device=self.device))
            hit = (gp_abs + gm_abs) > 0
        elif self.density.grad is not None:
            hit = self.density.grad.squeeze(-1).abs() > 0
        else:
            return

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
        # LC64 plan v3 -- independent-side mode is now implemented
        # (Commit 2A: forward-only CUDA dispatch).  The discriminator
        # drives the dispatch: when set to "independent" we activate
        # the thin-surface path AND flip the independent-mode flag so
        # the kernel reads raw_plus/raw_minus instead of density_delta.
        # The legacy absolute / relative paths are unchanged.
        _mode = getattr(self, "_thin_surface_density_mode", "scalar")
        independent_mode = (_mode == "independent")

        (points, density, point_adjacency, point_adjacency_offsets,
         density_grad, gradient_max_slope,
         density_peak, delta_raw, cov_raw,
         density_delta, quaternions, texel_sites_2d, texel_heights,
         raw_plus, raw_minus) = self.get_trace_data()

        interpolation_mode = getattr(self, "_interpolation_mode", False)
        idw_sigma = getattr(self, "_idw_sigma", 0.01)
        idw_sigma_v = getattr(self, "_idw_sigma_v", 0.1)
        per_cell_sigma = getattr(self, "_per_cell_sigma", False)
        per_neighbor_sigma = getattr(self, "_per_neighbor_sigma", False)
        gaussian_mode = getattr(self, "_gaussian_active", False)
        # Independent mode REQUIRES the thin-surface geometry
        # (quaternion + K texel sites + heights + cell_radius), so the
        # thin-surface master flag is forced on when independent mode
        # is active.  The kernel dispatch in C++ keys on the union
        # (thin_surface_mode && thin_surface_independent_mode).
        thin_surface_mode = (getattr(self, "_thin_surface_active", False)
                              or independent_mode)

        # Compute cell_radius on demand when adaptive sigma, gaussian, or thin-surface mode is active
        cell_radius = None
        if interpolation_mode and (per_cell_sigma or per_neighbor_sigma):
            cell_radius = self._get_cell_radius()
        if gaussian_mode and density_peak is not None:
            cell_radius = self._get_cell_radius()
        if thin_surface_mode and density_delta is not None:
            cell_radius = self._get_cell_radius()
        # Independent mode also needs cell_radius (the surface geometry
        # uses it for the soft-Voronoi height eval).  The geometry
        # tensors (quaternions / texel_sites_2d / texel_heights) are
        # always present in independent mode because the user must
        # have either initialized thin_surface or loaded a checkpoint
        # -- initialize_independent_sides does not create them, so
        # reach into the scene to verify they are wired.
        if independent_mode:
            if (quaternions is None or texel_sites_2d is None
                    or texel_heights is None):
                raise RuntimeError(
                    "CTScene.forward: independent mode requires the "
                    "thin-surface geometry (quaternions, texel_sites_2d, "
                    "texel_heights) to be registered. Call "
                    "initialize_thin_surface(args, K=4) first (it will "
                    "route to initialize_independent_sides under "
                    "thin_surface_density_mode='independent') or load a "
                    "checkpoint that carries the geometry tensors."
                )
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
            thin_surface_mode,
            # LC64 plan v3 Commit 2A -- independent mode is mutually
            # exclusive with density_delta (the legacy absolute/relative
            # thin-surface path).  Pass None for density_delta under
            # independent mode so the C++ binding rejects no-call-input
            # before any kernel launch.
            density_delta if not independent_mode else None,
            quaternions,
            texel_sites_2d,
            texel_heights,
            getattr(self, "_thin_K", 4),
            getattr(self, "_thin_temp", 10.0),
            getattr(self, "_thin_height_eps", 1e-4),
            # M5 chest rescue: relative-delta parameterization
            # (delta = rho * mu_bar * tanh(raw)).  Persisted through
            # initialize_thin_surface's _thin_surface_relative_delta flag
            # so eval / resume use the same interpretation as training.
            getattr(self, "_thin_surface_relative_delta", False),
            float(getattr(self, "_thin_surface_delta_max_frac", 0.5)),
            # LC64 plan v3 Commit 2A -- independent-side raw logits.
            # When independent_mode is True, raw_plus / raw_minus are
            # forwarded to the CUDA kernel (each (N,1)) and the kernel
            # dispatches to ct_independent_forward.  When False, the
            # legacy thin-surface / scalar paths are unchanged.
            raw_plus if independent_mode else None,
            raw_minus if independent_mode else None,
            independent_mode,
            float(self.activation_scale),
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

        # True stationary-frame control (LC64 plan v2).  When
        # iteration >= points_hard_freeze_at, the primal-points
        # param group becomes non-trainable: LR=0,
        # primal_points.requires_grad_(False), and Adam state
        # cleared.  -1 disables; legacy freeze_points schedule is
        # unchanged.
        self._points_hard_freeze_at = int(
            getattr(args, "points_hard_freeze_at", -1))
        # Sticky "freeze is currently active" flag (set by
        # enforce_hard_point_freeze when iter >= T) and the last
        # iteration seen by enforce_hard_point_freeze.  Both are
        # used by the frozen-state helper so replacement paths
        # (permute / prune / densify / load) can re-apply the freeze
        # without being passed an iteration bound.
        self._hard_freeze_active = False
        self._last_iteration = None

    def initialize_independent_sides(self, args):
        """Register the LC64 plan v3 independent-side raw logits.

        Schema/setup only -- this commit does NOT touch rendering or
        CUDA. The two raw side parameters (`raw_plus`, `raw_minus`,
        each (N,1)) are initialized by cloning the existing scalar
        raw `density`, so softplus(raw_plus) == softplus(raw_minus)
        == softplus(density) at iteration 0 (physical-side equality
        with the scalar baseline).

        The base density is then FROZEN as a third density degree:
        requires_grad=False and removed from the optimizer so it
        cannot be stepped by Adam. The two raw side parameters
        become separate ordinary Adam groups (`raw_plus`, `raw_minus`)
        with identical native raw-side LR schedule (a single shared
        cosine scheduler). No mean_raw is introduced and we do not
        claim coordinate-matched Adam; the equal schedule is the
        only claim.

        Independent rendering/backward use the CUDA-native thin-surface path;
        this initializer owns raw-side optimizer registration while
        initialize_thin_surface() owns shared geometry registration.
        Mutually exclusive with the relative-delta path
        (`thin_surface_relative_delta=True`).
        """
        # Mutually exclusive with the bounded relative-delta path.
        # The E0 amendment (v3) chose either relative OR independent;
        # a config that activates both is a misuse and we reject it
        # rather than silently picking one.
        if getattr(args, "thin_surface_relative_delta", False):
            raise ValueError(
                "initialize_independent_sides: mutually exclusive with "
                "thin_surface_relative_delta=True (LC64 plan v3 E0 "
                "amendment). Pick exactly one of relative or "
                "independent -- they are alternative parameterizations "
                "for the same comparison contract."
            )

        if not hasattr(self, "optimizer"):
            raise RuntimeError(
                "initialize_independent_sides: must be called AFTER "
                "declare_optimizer(args, warmup, max_iterations) so "
                "the optimizer is available to attach param groups to."
            )

        # Fresh runs clone the scalar raw density into both sides, establishing
        # exact zero-split equivalence. Checkpoint resumes keep the loaded raw
        # tensors and only rebuild optimizer groups/schedulers.
        N = self.density.shape[0]
        resume = (
            getattr(self, "_thin_surface_density_mode", "scalar") == "independent"
            and getattr(self, "raw_plus", None) is not None
            and getattr(self, "raw_minus", None) is not None
        )
        if not resume:
            with torch.no_grad():
                self.raw_plus = nn.Parameter(self.density.detach().clone())
                self.raw_minus = nn.Parameter(self.density.detach().clone())
        if not (torch.isfinite(self.raw_plus).all()
                and torch.isfinite(self.raw_minus).all()):
            raise RuntimeError(
                "initialize_independent_sides: raw_plus/raw_minus contain "
                "non-finite values; refusing to register."
            )
        if tuple(self.raw_plus.shape) != (N, 1) or tuple(self.raw_minus.shape) != (N, 1):
            raise RuntimeError(
                f"initialize_independent_sides: raw shape mismatch "
                f"(raw_plus={tuple(self.raw_plus.shape)}, "
                f"raw_minus={tuple(self.raw_minus.shape)}, expected "
                f"(N,1) with N={N})."
            )
        if not resume:
            self._raw_plus_init = self.raw_plus.detach().clone()
            self._raw_minus_init = self.raw_minus.detach().clone()

        # Freeze the base density as a third density degree. Adam
        # state for density is dropped so a hypothetical later
        # requires_grad_(True) cannot drag the base density through
        # stale momentum; the group is removed from the optimizer
        # entirely so update_learning_rate / step cannot touch it.
        self.density.requires_grad_(False)
        self.optimizer.state.pop(self.density, None)
        self.optimizer.param_groups = [
            g for g in self.optimizer.param_groups if g["name"] != "density"
        ]
        # Mark the (now-removed) density group on the scene so a
        # subsequent re-registration (e.g. on a stale checkpoint)
        # can detect the conflict.
        self._density_frozen = True

        # Attach raw_plus / raw_minus as ordinary Adam groups with
        # an identical native raw-side LR schedule. Same LR init and
        # same cosine schedule on both -> schedules are equal at
        # every iteration by construction (single shared scheduler).
        existing = {g["name"] for g in self.optimizer.param_groups}
        raw_side_lr_init = float(getattr(
            args, "thin_surface_raw_side_lr_init", args.density_lr_init))
        raw_side_lr_final = float(getattr(
            args, "thin_surface_raw_side_lr_final", args.density_lr_final))
        for name, param in (("raw_plus", self.raw_plus),
                            ("raw_minus", self.raw_minus)):
            if name not in existing:
                self.optimizer.add_param_group({
                    "params": param,
                    "lr": raw_side_lr_init,
                    "name": name,
                    "eps": 1e-8,
                })
            else:
                for group in self.optimizer.param_groups:
                    if group["name"] == name:
                        group["params"][0] = param
                        group["lr"] = raw_side_lr_init
                        group["eps"] = 1e-8
        # A single shared cosine scheduler; both groups evaluate it
        # at the same iteration, so the per-group LRs are bit-equal.
        # Use the legacy density schedule's max_iterations so the
        # raw-side cosine decays on the same horizon as the rest of
        # the model. Iteration offset 0 (no thin_surface_start gating
        # for the raw-side schedule; raw-side is always-on once
        # activated and the cosine uses the configured max_iter).
        self.raw_side_scheduler_args = get_cosine_lr_func(
            lr_init=raw_side_lr_init,
            lr_final=raw_side_lr_final,
            warmup_steps=getattr(args, "warmup_steps", 0),
            max_steps=int(self._max_iterations),
        )

        # Discriminator + bookkeeping. Independent mode does NOT
        # flip `_thin_surface_active` (rendering is not active), so
        # the existing kernel dispatch is unaffected. The mode flag
        # drives the fail-fast gate in forward().
        self._thin_surface_density_mode = "independent"
        # Schema-only initialization has no geometry yet; full
        # initialize_thin_surface() creates it immediately afterward. A loaded
        # active checkpoint keeps its restored geometry/active flag.
        if not resume:
            self._thin_surface_active = False
        # Activation iter (so a future commit can gate raw-side
        # logging / freezing on this; the raw-side schedule itself
        # uses the legacy max_iter horizon, not this offset).
        self._thin_surface_start = int(
            getattr(args, "thin_surface_start", -1))
        # Persisted scheduler cfg (matches the bounded-delta
        # convention so save_pt can round-trip it).
        self._raw_side_scheduler_cfg = {
            "lr_init": raw_side_lr_init,
            "lr_final": raw_side_lr_final,
            "max_steps": int(self._max_iterations),
        }
        print(f"Initialized independent-side raw logits: "
              f"N={N}, resume={resume}, plus/minus max abs diff="
              f"{(self.raw_plus - self.raw_minus).abs().max().item():.2e}, "
              f"raw_side_lr=[{raw_side_lr_init:.2e} -> "
              f"{raw_side_lr_final:.2e}], base density frozen "
              f"(excluded from optimizer as third degree).")

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

    def initialize_thin_surface(self, args, K: int = 4):
        """Activate the two-sided thin-surface sub-cell partition.

        Called once at iteration M0 (Stage 1 init), or after `load_pt` when
        resuming training (in which case the four tensors already exist on the
        scene and are kept as-is — only the optimizer param groups and the LR
        scheduler are (re)attached).

        Registers density_delta, quaternions, texel_sites_2d, texel_heights as
        learnable parameters and adds them to the optimizer with separate
        (lower) learning rates.

        Mode dispatch (LC64 plan v3): independent mode replaces
        ``density``/``density_delta`` with native ``raw_plus``/``raw_minus``
        while sharing the same quaternion/site/height surface geometry.
        Scalar / absolute / relative modes keep the legacy density path.
        """
        _mode = getattr(args, "thin_surface_density_mode", "scalar")
        independent_mode = (_mode == "independent")
        if independent_mode:
            # Register/re-attach raw-side density parameters first; geometry
            # is initialized by the shared path below.
            self.initialize_independent_sides(args)
        # Default / legacy path. Record the discriminator so save_pt /
        # load_pt can round-trip it explicitly (legacy checkpoints
        # without the discriminator will infer this on load). If the
        # caller explicitly passed a non-scalar density_mode we
        # respect it; otherwise infer from thin_surface_relative_delta
        # (which is the legacy discriminator for the bounded-delta arm).
        _explicit_mode = getattr(args, "thin_surface_density_mode", "scalar")
        if not independent_mode:
            if _explicit_mode in ("absolute", "relative"):
                self._thin_surface_density_mode = _explicit_mode
            else:
                self._thin_surface_density_mode = (
                    "relative" if getattr(args, "thin_surface_relative_delta", False)
                    else "absolute")
        assert_supported_thin_K(K)
        N = self.primal_points.shape[0]
        device = self.device

        resume = (
            getattr(self, "_thin_surface_active", False)
            and getattr(self, "quaternions", None) is not None
            and getattr(self, "texel_sites_2d", None) is not None
            and getattr(self, "texel_heights", None) is not None
            and (independent_mode
                 or getattr(self, "density_delta", None) is not None)
        )

        if not resume:
            # Legacy absolute/relative modes have a base+delta density pair.
            # Independent mode has no density_delta third degree.
            if not independent_mode:
                self.density_delta = nn.Parameter(
                    torch.zeros(N, 1, device=device, dtype=torch.float32)
                )

            # Quaternions: identity [w=1, x=0, y=0, z=0] → normal along +X until
            # boundary-alignment eigenvectors are available for warm-start.
            q0 = torch.zeros(N, 4, device=device, dtype=torch.float32)
            q0[:, 0] = 1.0
            # Warm-start from `_last_top_eigvec` (a (N, 3) tensor cached by
            # `_boundary_top_eigvec`).  Defensively validate its shape: the
            # cache is keyed to the point count at the time it was populated
            # and may not have been updated by an intermediate prune/densify
            # path.  If it disagrees with the current N we discard it and
            # fall back to identity quaternions rather than crash on a shape
            # mismatch in `torch.cross(ref, v, dim=-1)` (CH4 reproducer:
            # standalone prune between iters 5999->6000 shrunk primal_points
            # but left the cache at its pre-prune N).
            v_cache = getattr(self, "_last_top_eigvec", None)
            if v_cache is not None:
                v_shape = tuple(v_cache.shape)
                if len(v_shape) >= 1 and v_shape[0] == N:
                    v = v_cache  # (N, 3) unit vectors, detached
                    # Quaternion rotating [1,0,0] onto v via the half-angle formula.
                    ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
                    ref = ref.unsqueeze(0).expand(N, -1)
                    cross = torch.cross(ref, v, dim=-1)         # (N, 3)
                    dot   = (ref * v).sum(dim=-1, keepdim=True) # (N, 1)
                    # sin(θ) = |cross|, cos(θ) = dot
                    w = torch.sqrt(((dot + 1.0) * 0.5).clamp_min(0.0))  # cos(θ/2)
                    xyz = cross / (2.0 * w.clamp_min(1e-12))             # sin(θ/2) * axis
                    q0 = torch.cat([w, xyz], dim=-1)
                    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    # Flip to upper hemisphere (w ≥ 0) for sign consistency.
                    q0 = q0 * torch.sign(q0[:, :1]).clamp_min(1.0)
                else:
                    # Stale cache (e.g. left over from a prune or densify
                    # that did not permute/invalidate it).  Drop it and
                    # fall through to the identity-quaternion fallback so
                    # the activation never crashes; the warm-start will
                    # be rebuilt on the next boundary-loss call.
                    print(
                        f"[thin-surface] WARNING: _last_top_eigvec shape "
                        f"{v_shape} disagrees with current N={N}; discarding "
                        f"stale warm-start cache and falling back to identity "
                        f"quaternions.  Re-run a boundary-alignment loss to "
                        f"repopulate the cache."
                    )
                    delattr(self, "_last_top_eigvec")
            self.quaternions = nn.Parameter(q0)

            # Texel sites: small jittered ring (radius 0.4) in the unit disc.
            angles = torch.linspace(0, 2 * 3.14159265, K + 1, device=device)[:-1]
            base_sites = torch.stack([
                torch.cos(angles) * 0.4,
                torch.sin(angles) * 0.4,
            ], dim=-1)  # (K, 2)
            jitter = (torch.rand(N, K, 2, device=device) - 0.5) * 0.05
            self.texel_sites_2d = nn.Parameter(
                base_sites.unsqueeze(0).expand(N, -1, -1) + jitter
            )

            # Heights: zero → flat surface, no contribution until δ grows.
            self.texel_heights = nn.Parameter(
                torch.zeros(N, K, device=device, dtype=torch.float32)
            )
        else:
            # Resuming from a checkpoint that already restored the tensors.
            # Validate shapes against the current point count and K.
            _resume_shapes = [
                ("quaternions", (N, 4)),
                ("texel_sites_2d", (N, K, 2)),
                ("texel_heights", (N, K)),
            ]
            if not independent_mode:
                _resume_shapes.insert(0, ("density_delta", (N, 1)))
            for name, shp in _resume_shapes:
                t = getattr(self, name)
                if tuple(t.shape) != shp:
                    raise RuntimeError(
                        f"resume thin-surface: {name} shape {tuple(t.shape)} "
                        f"!= expected {shp} (N={N}, K={K}). The checkpoint "
                        f"was saved with a different K or point count."
                    )
            print(f"[thin-surface] resuming with loaded tensors (N={N}, K={K})")

        thin_surface_lr = args.density_lr_init * 0.1
        # Cube-rescue LR overrides (see experiments/queue.md Batch B1-R):
        #   thin_surface_lr_scale        : global multiplier on all four thin LRs
        #   *_delta_lr_scale / *_quat_lr_scale / *_sites_lr_scale /
        #     *_heights_lr_scale : per-group multipliers (applied after global)
        # A scale of 0.0 freezes that group's LR (R1 freezes all; R2 scales
        # delta; R3 freezes geometry). Defaults (1.0) preserve the failed recipe.
        _global_scale = float(getattr(args, "thin_surface_lr_scale", 1.0))
        self._thin_surface_lr_scale = _global_scale
        self._thin_surface_group_lr_scale = {
            "density_delta": float(getattr(args, "thin_surface_delta_lr_scale", 1.0)),
            "quaternions":   float(getattr(args, "thin_surface_quat_lr_scale", 1.0)),
            "texel_sites_2d": float(getattr(args, "thin_surface_sites_lr_scale", 1.0)),
            "texel_heights": float(getattr(args, "thin_surface_heights_lr_scale", 1.0)),
        }
        # Effective initial per-group LR for the cosine scheduler. Each group
        # gets its own scheduler so frozen groups (scale 0) stay at 0.
        _thin_group_names = ["quaternions", "texel_sites_2d", "texel_heights"]
        if not independent_mode:
            _thin_group_names.insert(0, "density_delta")
        self._thin_surface_group_lr_init = {}
        self._thin_surface_group_scheduler = {}
        for _name in _thin_group_names:
            self._thin_surface_group_lr_init[_name] = (
                thin_surface_lr * _global_scale
                * self._thin_surface_group_lr_scale[_name])
            self._thin_surface_group_scheduler[_name] = get_cosine_lr_func(
                lr_init=self._thin_surface_group_lr_init[_name],
                lr_final=self._thin_surface_group_lr_init[_name] * 0.1,
                max_steps=max(1, self._max_iterations - args.thin_surface_start),
            )

        # Attach optimizer param groups (skip names already present — idempotent
        # across repeated initialize_thin_surface calls on the same optimizer).
        existing = {g["name"] for g in self.optimizer.param_groups}
        for name in _thin_group_names:
            p = getattr(self, name)
            _lr0 = self._thin_surface_group_lr_init[name]
            if name not in existing:
                self.optimizer.add_param_group({
                    "params": p, "lr": _lr0, "name": name,
                    "eps": 1e-8,
                })
            else:
                for _g in self.optimizer.param_groups:
                    if _g["name"] == name:
                        _g["params"][0] = p
                        _g["lr"] = _lr0
                        _g["eps"] = 1e-8

        # Kept for backwards compatibility with code that reads the shared
        # scheduler (use the group schedulers above for per-group LRs).
        _primary_group = ("quaternions" if independent_mode else "density_delta")
        self.thin_surface_scheduler_args = self._thin_surface_group_scheduler[_primary_group]
        self._thin_surface_scheduler_cfg = {
            "lr_init": self._thin_surface_group_lr_init[_primary_group],
            "lr_final": self._thin_surface_group_lr_init[_primary_group] * 0.1,
            "max_steps": max(1, self._max_iterations - args.thin_surface_start),
        }
        self._thin_surface_start = args.thin_surface_start
        self._thin_surface_active = True
        self._thin_K = K
        self._thin_surface_delta_clip = float(
            getattr(args, "thin_surface_delta_clip", 2.0))
        self._thin_surface_grad_clip = float(
            getattr(args, "thin_surface_grad_clip", 1.0))
        # M5 relative-delta parameterization: store the chosen mode so it
        # propagates through the autograd function call and the checkpoint
        # metadata (save_pt / load_pt).  Defaults to False (legacy absolute
        # delta = raw additive offset); opt in per-config / per-run.
        self._thin_surface_relative_delta = (
            False if independent_mode else bool(
                getattr(args, "thin_surface_relative_delta", False)))
        self._thin_surface_delta_max_frac = float(
            getattr(args, "thin_surface_delta_max_frac", 0.5))

        _ginfo = ", ".join(
            f"{n}={self._thin_surface_group_lr_init[n]:.2e}"
            for n in _thin_group_names)
        print(f"Initialized thin-surface params: {N} cells, K={K} texels "
              f"(density_mode={self._thin_surface_density_mode}, "
              f"global_lr_scale={_global_scale}, group_lr=[{_ginfo}], "
              f"resume={resume}, delta_clip={self._thin_surface_delta_clip}, "
              f"grad_clip={self._thin_surface_grad_clip}, "
              f"relative_delta={self._thin_surface_relative_delta}, "
              f"rho={self._thin_surface_delta_max_frac})")

    @torch.no_grad()
    def clamp_thin_surface_params(self):
        """Post-step safety bounds on thin-surface params.

        density_delta clamping semantics depend on parameterization:
          absolute (legacy):  delta_val = raw_delta,  so clamp |raw_delta|
          relative (M5):      delta_val = rho * mu_bar * tanh(raw_delta);
                              tanh already bounds it to (-1,+1), so the
                              resulting delta stays within rho * mu_bar no
                              matter how raw_delta grows. We still clamp
                              raw_delta itself as a soft Adam LR cap (off if
                              thin_surface_delta_clip <= 0).  Quaternions
                              are normalized in the kernel already; we only
                              rescale if their norm has drifted far from 1
                              to keep the parameter manifold sane (no effect
                              on the forward).
        """
        if not getattr(self, "_thin_surface_active", False):
            return
        c = getattr(self, "_thin_surface_delta_clip", 0.0)
        # In the relative parameterization the kernel's `tanh` already
        # bounds the effective delta to (-rho*mu_bar, +rho*mu_bar), so the
        # absolute post-step raw clamp is redundant; skip it to preserve
        # optimizer headroom (otherwise tanh would saturate prematurely).
        if (c and c > 0
            and getattr(self, "density_delta", None) is not None
            and not getattr(self, "_thin_surface_relative_delta", False)):
            self.density_delta.data.clamp_(-c, c)
        if getattr(self, "quaternions", None) is not None:
            qn = self.quaternions.data.norm(dim=-1, keepdim=True)
            # Rescale only where the norm has drifted out of [0.5, 2.0].
            drift = (qn < 0.5) | (qn > 2.0)
            if drift.any():
                inv = torch.where(drift, 1.0 / qn.clamp_min(1e-12),
                                  torch.ones_like(qn))
                self.quaternions.data.mul_(inv)

    def clip_thin_surface_grads(self):
        """Clip gradients on the four thin-surface params (mirrors the existing
        density grad clip). No-op if _thin_surface_grad_clip <= 0."""
        c = float(getattr(self, "_thin_surface_grad_clip", 0.0))
        if not (c and c > 0):
            return
        for name in ("density_delta", "quaternions",
                     "texel_sites_2d", "texel_heights"):
            p = getattr(self, name, None)
            if p is not None and p.grad is not None:
                p.grad.clamp_(-c, c)

    @torch.no_grad()
    def independent_side_diagnostics(self):
        """Read-only diagnostics for the LC64 plan v3 independent-side
        raw logits. Returns a dict of numeric stats (TensorBoard-safe)
        or None when the mode is not active. Does NOT alter losses or
        rendering -- a pure observability hook.

        Reported keys:
          - side_raw_plus_mean / p95 / max     raw logit stats
          - side_raw_minus_mean / p95 / max    raw logit stats
          - side_physical_plus_mean           softplus(raw_plus) mean
          - side_physical_minus_mean          softplus(raw_minus) mean
          - side_physical_contrast_mean       mean of physical
                                              |mu_plus - mu_minus|
          - side_physical_contrast_p95        p95 of same
          - side_physical_contrast_max        max of same
          - side_raw_diff_mean                raw |plus - minus| mean
          - base_density_frozen               1 if the base density is
                                              currently frozen (third
                                              degree excluded from the
                                              optimizer), else 0
        """
        if not self._raw_side_active():
            return None
        rp = self.raw_plus.detach().squeeze(-1)
        rm = self.raw_minus.detach().squeeze(-1)
        if not (torch.isfinite(rp).all() and torch.isfinite(rm).all()):
            return {"raw_nonfinite": 1.0}
        scale = float(getattr(self, "activation_scale", 1.0))
        mu_p = scale * torch.nn.functional.softplus(rp, beta=10.0)
        mu_m = scale * torch.nn.functional.softplus(rm, beta=10.0)
        contrast = (mu_p - mu_m).abs()
        raw_diff = (rp - rm).abs()
        rp0 = getattr(self, "_raw_plus_init", None)
        rm0 = getattr(self, "_raw_minus_init", None)
        rp_disp = ((rp - rp0.squeeze(-1)).abs() if rp0 is not None
                   and rp0.shape == self.raw_plus.shape else None)
        rm_disp = ((rm - rm0.squeeze(-1)).abs() if rm0 is not None
                   and rm0.shape == self.raw_minus.shape else None)
        q = 0.95
        return {
            "side_raw_plus_mean": rp.mean().item(),
            "side_raw_plus_p95": (rp.float().quantile(q).item()
                                   if rp.numel() else 0.0),
            "side_raw_plus_max": rp.abs().max().item(),
            "side_raw_minus_mean": rm.mean().item(),
            "side_raw_minus_p95": (rm.float().quantile(q).item()
                                    if rm.numel() else 0.0),
            "side_raw_minus_max": rm.abs().max().item(),
            "side_physical_plus_mean": mu_p.mean().item(),
            "side_physical_minus_mean": mu_m.mean().item(),
            "side_physical_contrast_mean": contrast.mean().item(),
            "side_physical_contrast_p95": (contrast.float().quantile(q).item()
                                            if contrast.numel() else 0.0),
            "side_physical_contrast_max": contrast.max().item(),
            "side_raw_diff_mean": raw_diff.mean().item(),
            "side_raw_plus_grad_norm": (
                self.raw_plus.grad.detach().float().norm().item()
                if self.raw_plus.grad is not None else float("nan")),
            "side_raw_minus_grad_norm": (
                self.raw_minus.grad.detach().float().norm().item()
                if self.raw_minus.grad is not None else float("nan")),
            "side_raw_plus_displacement_mean": (
                rp_disp.mean().item() if rp_disp is not None else float("nan")),
            "side_raw_minus_displacement_mean": (
                rm_disp.mean().item() if rm_disp is not None else float("nan")),
            "base_density_frozen": 1.0 if getattr(self, "_density_frozen",
                                                    False) else 0.0,
        }

    def thin_surface_diagnostics(self):
        """P0-F stats for TensorBoard. Returns dict or None if inactive.

        For the relative-delta parameterization (M5) the recorded `dd` is
        the effective delta = rho * mu_bar * tanh(raw), not the raw learnable
        parameter.  We surface an extra `delta_raw_*` triplet for diagnosing
        Adam-side saturation if desired, and tag the dict with `delta_mode`.
        """
        if not getattr(self, "_thin_surface_active", False):
            return None
        if getattr(self, "density_delta", None) is None:
            return None
        raw = self.density_delta.detach().squeeze(-1)       # (N,) raw learnable
        mu_bar = self.get_primal_density().detach().squeeze(-1)  # (N,)
        if getattr(self, "_thin_surface_relative_delta", False):
            rho = float(getattr(self, "_thin_surface_delta_max_frac", 0.5))
            dd = rho * mu_bar * torch.tanh(raw)             # effective delta
            mode = "relative"
        else:
            dd = raw                                          # effective delta
            mode = "absolute"
        mu_p = torch.clamp(mu_bar + dd, min=0.0)
        mu_n = torch.clamp(mu_bar - dd, min=0.0)
        h = self.texel_heights.detach()                     # (N,K)
        h_l1 = h.abs().sum(dim=-1)                           # (N,)
        qn = self.quaternions.detach().norm(dim=-1)          # (N,)
        tau = getattr(self, "_thin_surface_gate_tau", 0.01)
        active = (h_l1 > tau).float()
        warm = 1.0 if hasattr(self, "_last_top_eigvec") else 0.0
        q = 0.95

        # --- Geometry-health diagnostics (additive, TensorBoard-safe). --------
        # These are read-only numeric summaries surfaced for the LC64
        # oriented-height run diagnosis.  They never feed a loss or the
        # optimizer and use only already-detached tensors, so they cannot
        # change rendering, initialization, or optimization behaviour.
        # All new keys are numeric (Python float); `float('nan')` is a valid
        # float for `writer.add_scalar` and only appears when a quantity is
        # genuinely undefined (no gradients yet / no neighbours / no radius).

        # (a) Per-group gradient norms before the optimizer step.
        # `optimizer.zero_grad(set_to_none=True)` runs at the START of the
        # next iteration, so the grad tensors driving this step are still
        # resident in `.grad` when diagnostics run (after step, before the
        # next zero_grad).  These therefore reflect the post-clip,
        # pre-step gradients -- i.e. exactly what Adam consumed.  NaN if no
        # backward has populated `.grad` for that group yet.
        grad_norms = {}
        for _gname in ("density_delta", "quaternions",
                       "texel_sites_2d", "texel_heights"):
            _p = getattr(self, _gname, None)
            if _p is not None and _p.grad is not None:
                grad_norms[_gname] = _p.grad.detach().float().norm().item()
            else:
                grad_norms[_gname] = float("nan")

        # (b) Quaternion-implied surface-normal neighbour coherence using
        # the CSR Delaunay adjacency.  `coh_sq` = mean of (n_i . n_j)^2 over
        # oriented edges (sign-insensitive, in [0,1]; 1 = neighbours share a
        # direction).  `flip_frac` = fraction of edges whose normals point in
        # opposite hemispheres (n_i . n_j < 0) -- a large value indicates the
        # orientation field is flipping rather than smoothly rotating.
        normals = quaternion_to_normals(self.quaternions)     # (N, 3)
        coh_sq, flip_frac = self._normal_neighbor_coherence(normals)

        # (c) Height mean/std and a uniform-vs-curved measure.
        # `curvedness` per cell = std_K / (|mean_K| + std_K + eps), in [0,1].
        # All-K-equal (pure parallel translation) -> std_K = 0 -> curvedness 0.
        # Spread across texels (tilt/curvature) -> curvedness > 0.
        h_mean = h.mean().item()
        h_std = h.std(unbiased=False).item() if h.numel() > 1 else 0.0
        per_cell_std = h.std(dim=-1, unbiased=False) if h.shape[-1] > 1 \
            else torch.zeros_like(h_l1)
        per_cell_abs_mean = h.abs().mean(dim=-1)
        curvedness = (per_cell_std /
                      (per_cell_abs_mean + per_cell_std + 1e-12))

        # (d) Height extent diagnostics.
        # `texel_heights` are DIMENSIONLESS raw learnable parameters; the
        # forward kernel applies the world-space offset as `r * h_k`
        # (src/tracing/pipeline.cu, ct_thinsurface_forward: the height eval
        # accumulates `w * (r * texel_heights[...])`). The physical surface
        # extent per cell is therefore `cell_radius * h_l1`, NOT `h_l1` alone.
        #
        # Two complementary, correctly-dimensioned measures:
        #   (a) dimensionless normalized height L1 = h_l1 / p95(h_l1):
        #       scale-invariant relative magnitude (the p95 cell normalises
        #       to ~1). It does NOT divide by a world length, so it does not
        #       confound scene scale or cell count.
        #   (b) world height extent = cell_radius * h_l1:
        #       the physical protrusion of the sub-cell surface in scene units.
        # The previous `height_radius_ratio = h_l1 / cell_radius` had units of
        # 1/length and confounded cell count; it is replaced by the pair above.
        h_l1_p95 = h_l1.float().quantile(q).item() if h_l1.numel() else 0.0
        if h_l1.numel() and h_l1_p95 > 0.0:
            norm_ratio = h_l1 / h_l1_p95
            height_l1_norm_mean = norm_ratio.mean().item()
            height_l1_norm_p95 = (
                norm_ratio.float().quantile(q).item()
                if norm_ratio.numel() else 0.0)
        else:
            height_l1_norm_mean = float("nan")
            height_l1_norm_p95 = float("nan")

        cr = getattr(self, "_cached_cell_radius", None)
        if cr is not None and h_l1.numel():
            cr = cr.detach().to(device=h_l1.device, dtype=h_l1.dtype)
            extent = cr * h_l1
            height_extent_mean = extent.mean().item()
            height_extent_p95 = (
                extent.float().quantile(q).item() if extent.numel() else 0.0)
        else:
            height_extent_mean = float("nan")
            height_extent_p95 = float("nan")

        return {
            "delta_mode": mode,
            "delta_abs_mean": dd.abs().mean().item(),
            "delta_abs_p95": dd.abs().float().quantile(q).item() if dd.numel() else 0.0,
            "delta_abs_max": dd.abs().max().item(),
            "mu_plus_max": mu_p.max().item(),
            "mu_plus_mean": mu_p.mean().item(),
            "mu_minus_max": mu_n.max().item(),
            "height_l1_mean": h_l1.mean().item(),
            "height_l1_max": h_l1.max().item(),
            "quat_norm_mean": qn.mean().item(),
            "quat_norm_max": qn.max().item(),
            "active_frac": active.mean().item(),
            "delta_nonzero_frac": (dd.abs() > 1e-6).float().mean().item(),
            # Raw learnable param stats (M5): under the relative parameterization
            # this can drift without bound since tanh already handles the
            # semantic clipping. Useful to detect Adam-side saturation.
            "delta_raw_abs_mean": raw.abs().mean().item(),
            "delta_raw_abs_p95": raw.abs().float().quantile(q).item()
                                  if raw.numel() else 0.0,
            "delta_raw_abs_max": raw.abs().max().item(),
            "warm_start": warm,
            # (a) per-group pre-step gradient norms (NaN if no grad yet).
            "grad_norm_density_delta": grad_norms["density_delta"],
            "grad_norm_quaternions": grad_norms["quaternions"],
            "grad_norm_texel_sites_2d": grad_norms["texel_sites_2d"],
            "grad_norm_texel_heights": grad_norms["texel_heights"],
            # (b) normal neighbour coherence over the CSR adjacency.
            "quat_normal_coherence_sq": coh_sq,
            "quat_normal_flip_frac": flip_frac,
            # (c) height distribution + uniform-vs-curved measure.
            "height_mean": h_mean,
            "height_std": h_std,
            "height_curvedness": curvedness.mean().item(),
            # (d) height extent: (a) dimensionless normalized height L1
            #     h_l1 / p95(h_l1); (b) world height extent cell_radius * h_l1.
            "height_l1_norm_mean": height_l1_norm_mean,
            "height_l1_norm_p95": height_l1_norm_p95,
            "height_extent_mean": height_extent_mean,
            "height_extent_p95": height_extent_p95,
        }

    @torch.no_grad()
    def _normal_neighbor_coherence(self, normals: torch.Tensor):
        """Sign-insensitive neighbour coherence of a per-cell direction field.

        Uses the CSR Delaunay adjacency (``point_adjacency`` /
        ``point_adjacency_offsets``).  For every oriented edge (i -> j) it
        forms d = n_i . n_j and returns:

          coh_sq    = mean of d^2 over non-self edges   (in [0,1]; 1 = aligned)
          flip_frac = fraction of edges with d < 0       (hemisphere flips)

        Both are returned as python floats.  When the mesh has no neighbour
        edges (or only self-loops) the quantities are undefined and returned
        as ``float('nan')``.  Read-only / no-grad -- a pure diagnostic.
        """
        nan = float("nan")
        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        N = normals.shape[0]
        counts = offsets[1:] - offsets[:-1]
        if counts.numel() == 0 or counts.sum().item() == 0:
            return nan, nan
        src = torch.repeat_interleave(
            torch.arange(N, device=normals.device), counts)
        ni = normals[src]
        nj = normals[adj]
        dot = (ni * nj).sum(dim=-1)
        valid = src != adj
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return nan, nan
        dot = dot[valid]
        coh_sq = (dot * dot).mean().item()
        flip_frac = (dot < 0).float().mean().item()
        return coh_sq, flip_frac

    def _raw_side_active(self):
        """True iff the LC64 plan v3 independent-side raw logits are
        registered on the scene. Used by replacement paths
        (permute / prune / densify) to decide whether to align
        raw_plus / raw_minus with primal_points, and by save_pt /
        load_pt to decide whether to write the discriminator +
        raw tensors into the checkpoint.

        Independent mode is opt-in via
        ``initialize_independent_sides``; until then this returns
        False and the raw sides are silently absent from every
        replacement / checkpoint path. Safe to call before
        ``initialize_independent_sides`` (no AttributeError).
        """
        return (getattr(self, "_thin_surface_density_mode", "scalar")
                == "independent")

    def _hard_freeze_threshold(self):
        """Safe getter for the hard-freeze threshold (default -1)."""
        return int(getattr(self, "_points_hard_freeze_at", -1))

    def _should_hard_freeze(self, iteration=None):
        """Robust frozen-state helper.

        Returns True iff the hard point freeze should be active RIGHT NOW.

          - If ``iteration`` is provided: ``iteration >= T`` (where T is
            the configured ``points_hard_freeze_at``).
          - If ``iteration`` is None: rely on the sticky
            ``_hard_freeze_active`` flag set by a previous
            ``enforce_hard_point_freeze`` call with ``iter >= T``.

        Always returns False when T < 0 (default-disabled sentinel).
        Safe to call before optimizer declaration.
        """
        T = self._hard_freeze_threshold()
        if T < 0:
            return False
        if iteration is not None:
            return int(iteration) >= T
        return bool(getattr(self, "_hard_freeze_active", False))

    def _reapply_hard_freeze(self):
        """Replacement-lifecycle hook helper.

        Re-applies the hard freeze IF the frozen-state helper says so.
        Uses ``_last_iteration`` (set by previous enforce calls) as the
        iteration bound, falling back to the threshold itself if no
        iteration has been seen yet.  Safe to call from any replacement
        path (permute / prune / densify / load) and before optimizer
        declaration (no-op when T < 0 or before any enforce call).
        """
        if not self._should_hard_freeze():
            return
        iter_for_enforce = getattr(self, "_last_iteration", None)
        if iter_for_enforce is None:
            # No iteration seen yet (e.g., a checkpoint reload that
            # immediately triggers a replacement path).  Fall back to
            # the threshold itself so the freeze fires on the very
            # first replacement when T is non-negative.
            iter_for_enforce = self._hard_freeze_threshold()
        self.enforce_hard_point_freeze(iter_for_enforce)

    def enforce_hard_point_freeze(self, iteration):
        """Enforce the true stationary-frame control gate.

        Called at the start of every iteration (and defensively right
        before `optimizer.step`).  When iteration >= the threshold T:
          - re-resolves the CURRENT primal-points optimizer param group
            (identity may have changed since the last call due to
            permute / prune / densify / load_pt);
          - atomically rebinds the param group to the CURRENT
            ``self.primal_points`` tensor if identity changed;
          - sets its LR to 0;
          - sets primal_points.requires_grad_(False)  (idempotent);
          - clears the Adam state entry for BOTH the old (rebound-out)
            and the new (current) primal-points parameter so a future
            step (or a hypothetical toggle of requires_grad) cannot
            drag the points.

        Idempotent and free when T < 0 (disabled default sentinel).

        Side effects (always, even when T<0 or iter<T):
          - ``self._last_iteration`` is updated to the integer iteration
            so replacement paths called between iterations can re-apply
            the freeze via ``_reapply_hard_freeze()`` without needing a
            hard-coded iteration bound.
          - ``self._hard_freeze_active`` tracks the sticky "is the
            freeze currently engaged" flag, used by ``_should_hard_freeze``
            when called without an explicit iteration.
        """
        T = self._hard_freeze_threshold()
        self._last_iteration = int(iteration)
        if T < 0:
            # Disabled sentinel: explicitly clear the sticky flag so a
            # later re-enable (e.g., re-declare the optimizer with a
            # different points_hard_freeze_at) starts clean.
            self._hard_freeze_active = False
            return
        if iteration < T:
            # Below threshold: leave _hard_freeze_active alone (sticky
            # once tripped) so replacement paths called later can still
            # detect the freeze through the helper.
            return
        # Re-resolve every time: the primal_points tensor identity can
        # change via permute_points / prune_points / densification_postfix
        # / load_pt / load_frozen_checkpoint / initialize_gradients.  We
        # always target the CURRENT tensor by reading
        # `self.primal_points` and walking the optimizer's param_groups.
        pp = getattr(self, "primal_points", None)
        if pp is None:
            return
        self._hard_freeze_active = True
        # Atomically (re)bind the primal-points optimizer param group to
        # the CURRENT self.primal_points tensor.  This guarantees the
        # invariant ``group['params'][0] is self.primal_points`` after
        # every replacement path.  Old Adam state (keyed by the OLD
        # tensor id) is dropped according to the existing clear-state
        # policy (``state.pop(p, None)``).
        for _g in self.optimizer.param_groups:
            if _g["name"] == "primal_points":
                _old_param = _g["params"][0]
                if _old_param is not pp:
                    _g["params"][0] = pp
                    self.optimizer.state.pop(_old_param, None)
                _g["lr"] = 0.0
                break
        # Idempotent requires_grad_(False).  This re-asserts the freeze
        # on the CURRENT tensor after every replacement path (which is
        # how the post-replacement hooks below propagate it).
        pp.requires_grad_(False)
        # Clear Adam state for the current primal-points parameter.
        # PyTorch Adam lazily re-creates state on the next .step();
        # dropping the entry here ensures stale momentum from before
        # the freeze cannot drag the points on a hypothetical step
        # (e.g. if some downstream code toggles requires_grad back).
        self.optimizer.state.pop(pp, None)

    def pre_step(self, iteration):
        """Train-loop hook.  Forwards to enforce_hard_point_freeze.

        Idempotent; cheap when disabled.  The reviewer contract uses
        `enforce_hard_point_freeze` as the canonical name; `pre_step` is
        the historical name kept for backwards compatibility with the
        prior uncommitted patch and is a thin alias.
        """
        self.enforce_hard_point_freeze(iteration)

    def update_learning_rate(self, iteration):
        # Freeze positions while density gradients stabilize
        freeze_for_grad = (
            hasattr(self, "_gradient_freeze_points_until")
            and iteration < self._gradient_freeze_points_until
        )
        # True stationary-frame control (LC64 plan v2): if hard freeze
        # is active at/after the threshold, the primal-points LR MUST
        # remain exactly 0.0 -- the scheduler must never temporarily
        # restore a positive LR.  This makes update_learning_rate safe
        # to call even if enforce_hard_point_freeze was skipped at the
        # start of the iteration (e.g., during a replacement-only path).
        hard_freeze_active = self._should_hard_freeze(iteration)
        # Keep the last-iteration tracker fresh so replacement paths
        # called after this point can re-apply the freeze via
        # _reapply_hard_freeze().
        self._last_iteration = int(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                if hard_freeze_active or freeze_for_grad:
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
            elif param_group["name"] in ("density_delta", "quaternions",
                                         "texel_sites_2d", "texel_heights"):
                # Per-group cosine scheduler, honoring rescue LR scales (a scale
                # of 0 keeps that group's LR at 0 for the whole run -- R1/R3).
                _gn = param_group["name"]
                _scheds = getattr(self, "_thin_surface_group_scheduler", None)
                if _scheds is not None and _gn in _scheds:
                    param_group["lr"] = _scheds[_gn](
                        iteration - self._thin_surface_start
                    )
                elif hasattr(self, "thin_surface_scheduler_args"):
                    # Backwards-compat fallback (single shared scheduler).
                    param_group["lr"] = self.thin_surface_scheduler_args(
                        iteration - self._thin_surface_start
                    )
            elif param_group["name"] in ("raw_plus", "raw_minus"):
                # LC64 plan v3 -- native raw-side Adam schedule. Both
                # groups share a single scheduler (identical LR at
                # every iteration by construction); the cosine LR is
                # exactly the same value for raw_plus and raw_minus,
                # so the per-group "equal schedule" contract holds
                # without any per-group argmax/argmin claim.
                _sched = getattr(self, "raw_side_scheduler_args", None)
                if _sched is not None:
                    param_group["lr"] = _sched(iteration)

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
        # LC64 plan v3 -- density is only present in the optimizer
        # when not in independent mode (initialize_independent_sides
        # drops it). Guard the assignment to keep scalar / relative /
        # absolute paths unchanged.
        if "density" in optimizable_tensors:
            self.density = optimizable_tensors["density"]
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]
        if "density_peak" in optimizable_tensors:
            self.density_peak = optimizable_tensors["density_peak"]
        if "delta_raw" in optimizable_tensors:
            self.delta_raw = optimizable_tensors["delta_raw"]
        if "cov_raw" in optimizable_tensors:
            self.cov_raw = optimizable_tensors["cov_raw"]
        if "density_delta" in optimizable_tensors:
            self.density_delta = optimizable_tensors["density_delta"]
        if "quaternions" in optimizable_tensors:
            self.quaternions = optimizable_tensors["quaternions"]
        if "texel_sites_2d" in optimizable_tensors:
            self.texel_sites_2d = optimizable_tensors["texel_sites_2d"]
        if "texel_heights" in optimizable_tensors:
            self.texel_heights = optimizable_tensors["texel_heights"]
        # LC64 plan v3 -- prune the independent raw sides in lock-step
        # so they stay aligned with the surviving primal_points rows.
        if "raw_plus" in optimizable_tensors:
            self.raw_plus = optimizable_tensors["raw_plus"]
        if "raw_minus" in optimizable_tensors:
            self.raw_minus = optimizable_tensors["raw_minus"]
        if hasattr(self, '_starvation_count'):
            self._starvation_count = self._starvation_count[valid_points_mask]
        if hasattr(self, '_frozen_mask'):
            self._frozen_mask = self._frozen_mask[valid_points_mask]

        # Re-apply the hard point freeze after primal-points replacement
        # (prune creates a fresh nn.Parameter, so the optimizer group
        # identity must be rebound and the freeze re-asserted on the
        # new tensor).  Uses _reapply_hard_freeze so the call is safe
        # whether or not the freeze is currently active.
        self._reapply_hard_freeze()

        # Per-cell boundary-loss caches (set by `_boundary_top_eigvec`) are
        # keyed by row index to primal_points.  After a prune the surviving
        # rows are a strict subset, so the caches must be permuted by the
        # same `valid_points_mask` to stay aligned with primal_points --
        # otherwise `initialize_thin_surface` would read pre-prune warm-start
        # vectors and `torch.cross(ref, v, dim=-1)` would crash with a shape
        # mismatch at thin activation (CH4 reproducer).
        for cache_name in (
            "_last_top_eigvec",
            "_last_M_trace",
            "_last_M_valid",
            "_last_normal_lap_residual",
        ):
            if not hasattr(self, cache_name):
                continue
            t = getattr(self, cache_name)
            if t is None:
                continue
            if t.shape[0] == valid_points_mask.shape[0]:
                setattr(self, cache_name, t[valid_points_mask])
            else:
                # Cache shape disagrees with point count -- this means the
                # cache was left behind by a previous prune/densify path that
                # forgot to update it.  Drop it; the next boundary-loss call
                # repopulates it.  initialize_thin_surface also has its own
                # shape guard for `_last_top_eigvec`.
                delattr(self, cache_name)

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
        # LC64 plan v3 -- density is absent from the optimizer under
        # independent mode; new_params never contains "density" in
        # that case (initialize_independent_sides dropped it). Guard
        # so legacy / absolute / relative paths are unaffected.
        if "density" in optimizable_tensors:
            self.density = optimizable_tensors["density"]
        if "density_grad" in optimizable_tensors:
            self.density_grad = optimizable_tensors["density_grad"]
        if "density_peak" in optimizable_tensors:
            self.density_peak = optimizable_tensors["density_peak"]
        if "delta_raw" in optimizable_tensors:
            self.delta_raw = optimizable_tensors["delta_raw"]
        if "cov_raw" in optimizable_tensors:
            self.cov_raw = optimizable_tensors["cov_raw"]
        if "density_delta" in optimizable_tensors:
            self.density_delta = optimizable_tensors["density_delta"]
        if "quaternions" in optimizable_tensors:
            self.quaternions = optimizable_tensors["quaternions"]
        if "texel_sites_2d" in optimizable_tensors:
            self.texel_sites_2d = optimizable_tensors["texel_sites_2d"]
        if "texel_heights" in optimizable_tensors:
            self.texel_heights = optimizable_tensors["texel_heights"]
        # LC64 plan v3 -- densify the independent raw sides. New
        # cells get fresh raw_plus/raw_minus = 0 (matching the
        # legacy IDW-init pattern for `density`: a new cell starts
        # at the densify init value rather than inheriting a
        # parent's value, so the optimizer learns it from scratch).
        if ("raw_plus" in optimizable_tensors
                and "raw_minus" in optimizable_tensors):
            self.raw_plus = optimizable_tensors["raw_plus"]
            self.raw_minus = optimizable_tensors["raw_minus"]
        elif self._raw_side_active():
            # Densification fires while independent mode is active:
            # neither raw_plus nor raw_minus was in the densify
            # payload, so allocate zero-initialised extensions in
            # lock-step with the new primal_points rows. Without
            # this, the optimizer's cat_tensors_to_optimizer would
            # skip the raw sides and the next step would crash on
            # a shape mismatch with the Adam state.
            dev = self.primal_points.device
            new_raw_plus = torch.zeros(n_new, 1, device=dev)
            new_raw_minus = torch.zeros(n_new, 1, device=dev)
            if "raw_plus" not in optimizable_tensors:
                optimizable_tensors.update(self.cat_tensors_to_optimizer({
                    "raw_plus": new_raw_plus,
                    "raw_minus": new_raw_minus,
                }))
                self.raw_plus = optimizable_tensors["raw_plus"]
                self.raw_minus = optimizable_tensors["raw_minus"]
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

        # Re-apply the hard point freeze after primal-points replacement
        # (densification_postfix builds a fresh nn.Parameter for the
        # new cells).  Uses _reapply_hard_freeze so the call is safe
        # whether or not the freeze is currently active.
        self._reapply_hard_freeze()

        # Per-cell boundary-loss caches (set by `_boundary_top_eigvec`) are
        # row-aligned with primal_points at the time they are populated.
        # Newly densified cells have no boundary graph (and no cache entry)
        # until `_boundary_top_eigvec` runs again, so any cached tensor
        # would be structurally misaligned with the new row range.  Drop
        # the caches entirely -- they will be re-populated on the next
        # call to the boundary losses / `initialize_thin_surface` warm
        # start.  This protects the warm-start shape from the moment a
        # densification step fires before thin activation.
        for cache_name in (
            "_last_top_eigvec",
            "_last_M_trace",
            "_last_M_valid",
            "_last_normal_lap_residual",
        ):
            if hasattr(self, cache_name):
                delattr(self, cache_name)

    def prune_and_densify(
        self, point_error, point_contribution, upsample_factor=1.2,
        gradient_fraction=0.4, idw_fraction=0.3,
        entropy_fraction=0.3, entropy_bins=5,
        redundancy_threshold=0.0, redundancy_cap=0.0,
        sigma_scale=0.5, sigma_v=0.1,
        variance_pruning=False, prune_hops=1,
        ref_guided_pruning=False, ref_guided_densify=False, ref_guided_eps=0.05,
        grad_thresh=0.0,
        var_thresh=0.0, var_power=0.0, var_hops=1,
    ):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            # Sample reference weight at all cell positions once, before any pruning/densify
            ref_w = self._sample_ref_weight_at_points() if (ref_guided_pruning or ref_guided_densify) else None

            primal_error_accum = point_error.clip(min=0).squeeze()

            # Threshold mode: cap budget by eligible count and remaining final_points headroom.
            eligible_count = num_curr_points
            eligible_mask = None
            cell_var = None
            if grad_thresh > 0:
                grad_mask = primal_error_accum > grad_thresh
                if var_thresh > 0:
                    cell_var = self.compute_neighborhood_variance(cell_radius=None, hops=var_hops)
                    eligible_mask = grad_mask & (cell_var > var_thresh)
                else:
                    eligible_mask = grad_mask
                eligible_count = int(eligible_mask.sum().item())
                num_new_points = min(
                    int((upsample_factor - 1) * num_curr_points),
                    eligible_count,
                    max(0, self.num_final_points - num_curr_points),
                )
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

            # Ref-guided densification multiplier: suppress smooth regions, amplify edges
            if ref_guided_densify and ref_w is not None:
                ref_factor = (1.0 - ref_w).clamp(min=ref_guided_eps)  # (N,) with floor
                idw_weight = idw_weight * 0.5 * (ref_factor[src] + ref_factor[tgt])
            else:
                ref_factor = None

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
            mean_ref_w_pruned = None
            mean_ref_w_gradient = None
            mean_ref_w_idw = None
            n_basic_pruned = prune_mask.sum().item()
            if n_basic_pruned > 0:
                print(f"Pruning {n_basic_pruned}/{num_curr_points} cells "
                      f"(low_contrib={n_pruned_low_contrib}, tiny_radius={n_pruned_tiny_radius}, both={n_pruned_both})")

            ################ Redundancy pruning ################
            if redundancy_cap > 0:
                if ref_guided_pruning and ref_w is not None:
                    # Ref-guided criterion: smooth cells (high ref_w) score low → pruned first.
                    # Replaces noisy variance/IDW estimate with stable reference volume signal.
                    cell_score = 1.0 - ref_w
                    candidates = ~prune_mask
                    if hasattr(self, '_frozen_mask'):
                        candidates = candidates & ~self._frozen_mask
                    prune_label = "ref_weight"
                elif variance_pruning:
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
                        if ref_w is not None and removable.any():
                            mean_ref_w_pruned = ref_w[removable].mean().item()

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
                grad_weight = primal_error_accum * cell_radius
                if eligible_mask is not None:
                    if var_power > 0 and cell_var is not None:
                        grad_weight = grad_weight * cell_var.clamp(min=1e-12) ** var_power
                    grad_weight = grad_weight * eligible_mask.float()
                if ref_factor is not None:
                    grad_weight = grad_weight * ref_factor
                grad_inds = torch.multinomial(
                    grad_weight,
                    num_gradient_points,
                    replacement=False,
                )
                if ref_w is not None:
                    mean_ref_w_gradient = ref_w[grad_inds].mean().item()
                sampled_points_list.append((points + perturbation)[grad_inds])
                sampled_inds_list.append(grad_inds)
                sampled_density_list.append(self.density[grad_inds])
                if has_density_grad:
                    sampled_density_grad_list.append(self.density_grad[grad_inds])
                _append_gaussian_for_inds(grad_inds, num_gradient_points)
                n_added_gradient += num_gradient_points

            # --- IDW-based sampling (bilateral prediction error × edge length) ---
            if num_idw_points > 0:
                if ref_w is not None and ref_factor is not None:
                    edge_ref = 0.5 * (ref_factor[src] + ref_factor[tgt])
                    mean_ref_w_idw = (1.0 - edge_ref).mean().item()
                gated_idw_weight = idw_weight
                if eligible_mask is not None:
                    edge_eligible = (eligible_mask[src] | eligible_mask[tgt]).float()
                    gated_idw_weight = idw_weight * edge_eligible
                n_added_idw += _sample_edges(gated_idw_weight, num_idw_points, "idw")

            # --- Entropy-based sampling (neighbor density entropy × cell radius) ---
            if num_entropy_points > 0:
                entropy_weight = cell_entropy * cell_radius.squeeze()
                if eligible_mask is not None:
                    entropy_weight = entropy_weight * eligible_mask.float()
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
            # LC64 plan v3 -- densify independent raw sides in lock-step
            # with primal_points. New cells get fresh zero-initialised
            # raw_plus/raw_minus (zero activation via softplus), so a
            # densified cell starts as a transparent air cell rather
            # than inheriting a parent's value. The optimizer learns
            # both sides from scratch.
            if self._raw_side_active():
                n_s = sampled_points.shape[0]
                dev = sampled_points.device
                new_params["raw_plus"] = torch.zeros(n_s, 1, device=dev)
                new_params["raw_minus"] = torch.zeros(n_s, 1, device=dev)
            if getattr(self, "_thin_surface_active", False):
                n_s = sampled_points.shape[0]
                K = getattr(self, "_thin_K", 4)
                dev = sampled_points.device
                new_params["density_delta"] = torch.zeros(n_s, 1, device=dev)
                q0 = torch.zeros(n_s, 4, device=dev)
                q0[:, 0] = 1.0
                new_params["quaternions"] = q0
                angles = torch.linspace(0, 2 * 3.14159265, K + 1, device=dev)[:-1]
                base_sites = torch.stack([
                    torch.cos(angles) * 0.4,
                    torch.sin(angles) * 0.4,
                ], dim=-1)
                new_params["texel_sites_2d"] = base_sites.unsqueeze(0).expand(n_s, -1, -1).clone()
                new_params["texel_heights"] = torch.zeros(n_s, K, device=dev)

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

            stats = {
                "points_before": num_curr_points,
                "pruned_low_contrib": n_pruned_low_contrib,
                "pruned_tiny_radius": n_pruned_tiny_radius,
                "pruned_both": n_pruned_both,
                "pruned_redundancy": n_redundant,
                "added_gradient": n_added_gradient,
                "added_idw": n_added_idw,
                "added_entropy": n_added_entropy,
                "thresh_eligible": eligible_count,
                "thresh_n_sample": num_new_points,
                "filtered_duplicates": n_filtered_dupes,
                "points_after": self.primal_points.shape[0],
            }
            if mean_ref_w_pruned is not None:
                stats["ref_w_pruned"] = mean_ref_w_pruned
            if mean_ref_w_gradient is not None:
                stats["ref_w_gradient"] = mean_ref_w_gradient
            if mean_ref_w_idw is not None:
                stats["ref_w_idw"] = mean_ref_w_idw
            return stats

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
        # Thin-surface sub-cell partition (all four tensors + metadata).
        # Required so test.py / eval_vol.py / voxelize / resumed training can
        # reconstruct the surface; without this the checkpoint silently drops
        # to the scalar baseline.
        if (getattr(self, "_thin_surface_active", False)
                and getattr(self, "density_delta", None) is not None
                and getattr(self, "_thin_surface_density_mode", "scalar")
                    != "independent"):
            scene_data["density_delta"] = self.density_delta.detach().float().cpu()
            scene_data["quaternions"] = self.quaternions.detach().float().cpu()
            scene_data["texel_sites_2d"] = self.texel_sites_2d.detach().float().cpu()
            scene_data["texel_heights"] = self.texel_heights.detach().float().cpu()
            scene_data["thin_surface"] = {
                "active": True,
                "K": int(self._thin_K),
                "start": int(getattr(self, "_thin_surface_start", -1)),
                "scheduler_cfg": getattr(self, "_thin_surface_scheduler_cfg", None),
                # M5 chest rescue: persist the relative-delta parameterization
                # so eval/resume don't silently reinterpret density_delta.
                # Default False (legacy absolute) when missing -- safe for
                # checkpoints saved before this field existed.
                "relative_delta": bool(
                    getattr(self, "_thin_surface_relative_delta", False)),
                "delta_max_frac": float(
                    getattr(self, "_thin_surface_delta_max_frac", 0.5)),
            }
        # LC64 plan v3 -- independent-side raw logits + discriminator.
        # The discriminator is stored alongside the bounded-delta
        # metadata (under the `thin_surface` key for backward-compat
        # layout) so a single `thin_surface` block carries the full
        # density-mode state. A mixed-state checkpoint (relative_delta
        # and density_mode="independent") is rejected at save time so
        # the discriminator never observes a contradiction.
        if self._raw_side_active():
            # Mutually exclusive with the bounded relative-delta path.
            if getattr(self, "_thin_surface_relative_delta", False):
                raise RuntimeError(
                    "save_pt: _thin_surface_density_mode='independent' is "
                    "active but _thin_surface_relative_delta is also True; "
                    "this is a mutually-exclusive mixed state and the "
                    "checkpoint would be malformed. Pick exactly one."
                )
            rp = self.raw_plus.detach().float().cpu()
            rm = self.raw_minus.detach().float().cpu()
            # Shape / finite validation: independent tensors must be
            # (N, 1) and finite. A regression here would silently
            # round-trip a NaN into the next training run.
            if not (torch.isfinite(rp).all() and torch.isfinite(rm).all()):
                raise RuntimeError(
                    "save_pt: raw_plus / raw_minus contain non-finite "
                    "values; refusing to write a malformed checkpoint."
                )
            N_pp = self.primal_points.shape[0]
            if tuple(rp.shape) != (N_pp, 1) or tuple(rm.shape) != (N_pp, 1):
                raise RuntimeError(
                    f"save_pt: raw shape mismatch (raw_plus={tuple(rp.shape)}, "
                    f"raw_minus={tuple(rm.shape)}, expected (N,1) with "
                    f"N={N_pp})."
                )
            scene_data["raw_plus"] = rp
            scene_data["raw_minus"] = rm
            # Independent rendering shares the thin-surface geometry but has
            # no density_delta/base-density degree. Persist all geometry so a
            # loaded checkpoint reproduces both projections and hard-side
            # queries, then reattach optimizer groups on training resume.
            geometry_active = bool(getattr(self, "_thin_surface_active", False))
            if geometry_active:
                required_geometry = {
                    "quaternions": (N_pp, 4),
                    "texel_sites_2d": (N_pp, int(self._thin_K), 2),
                    "texel_heights": (N_pp, int(self._thin_K)),
                }
                for name, shape in required_geometry.items():
                    value = getattr(self, name, None)
                    if value is None or tuple(value.shape) != shape:
                        raise RuntimeError(
                            f"save_pt: independent mode requires {name} with "
                            f"shape {shape}, got "
                            f"{None if value is None else tuple(value.shape)}")
                    scene_data[name] = value.detach().float().cpu()
            ts_meta = scene_data.get("thin_surface", None)
            if not isinstance(ts_meta, dict):
                ts_meta = {}
            ts_meta.update({
                "active": geometry_active,
                "K": int(getattr(self, "_thin_K", 4)),
                "start": int(getattr(self, "_thin_surface_start", -1)),
                "density_mode": "independent",
                "relative_delta": False,
                "scheduler_cfg": getattr(
                    self, "_thin_surface_scheduler_cfg", None),
                "raw_side_scheduler_cfg": getattr(
                    self, "_raw_side_scheduler_cfg", None),
                "raw_side_start": int(
                    getattr(self, "_thin_surface_start", -1)),
            })
            scene_data["thin_surface"] = ts_meta
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        scene_data = torch.load(pt_path)

        self.primal_points = nn.Parameter(scene_data["xyz"].to(self.device))
        self._reapply_hard_freeze()  # Re-apply after primal-points replacement
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

        # Thin-surface sub-cell partition. Restores the four tensors and the
        # metadata flags so `forward()` keys surface mode on. Optimizer param
        # groups are NOT rebuilt here — call `initialize_thin_surface(args, K)`
        # after `declare_optimizer` to resume training (it detects the loaded
        # tensors and only re-attaches the LR scheduler + param groups).
        if ("thin_surface" in scene_data
                and scene_data["thin_surface"].get("active")
                and scene_data["thin_surface"].get("density_mode")
                    != "independent"):
            meta = scene_data["thin_surface"]
            K = int(meta.get("K", 4))
            assert_supported_thin_K(K)
            self.density_delta = nn.Parameter(scene_data["density_delta"].to(self.device))
            self.quaternions = nn.Parameter(scene_data["quaternions"].to(self.device))
            self.texel_sites_2d = nn.Parameter(scene_data["texel_sites_2d"].to(self.device))
            self.texel_heights = nn.Parameter(scene_data["texel_heights"].to(self.device))
            self._thin_K = K
            self._thin_surface_active = True
            self._thin_surface_start = int(meta.get("start", -1))
            self._thin_surface_scheduler_cfg = meta.get("scheduler_cfg", None)
            # M5 relative-delta parameterization (legacy default = absolute).
            # A naive reinterpretation of an absolute-delta checkpoint under
            # relative mode would silently feed raw delta values into tanh,
            # which is why this flag MUST round-trip through the checkpoint.
            self._thin_surface_relative_delta = bool(
                meta.get("relative_delta", False))
            self._thin_surface_delta_max_frac = float(
                meta.get("delta_max_frac", 0.5))
            if self._thin_surface_scheduler_cfg is not None:
                self.thin_surface_scheduler_args = get_cosine_lr_func(
                    lr_init=self._thin_surface_scheduler_cfg["lr_init"],
                    lr_final=self._thin_surface_scheduler_cfg["lr_final"],
                    max_steps=self._thin_surface_scheduler_cfg["max_steps"],
                )
            print(f"[load_pt] restored thin-surface state: K={K}, "
                  f"N={self.primal_points.shape[0]}, "
                  f"relative_delta={self._thin_surface_relative_delta}, "
                  f"rho={self._thin_surface_delta_max_frac}")

        # LC64 plan v3 -- independent-side raw logits + discriminator.
        # The discriminator (`density_mode`) lives inside the
        # `thin_surface` metadata block; when set to "independent" the
        # load path restores raw_plus / raw_minus as Parameters and
        # wires the scheduler cfg. Mixed-state rejection (relative_delta
        # AND density_mode='independent') is enforced here so a
        # malformed checkpoint fails loudly on resume instead of
        # silently picking one parameterization. Legacy checkpoints
        # without a density_mode key infer the existing mode from the
        # pre-existing thin_surface fields -- "scalar" when the
        # bounded-delta block is inactive, "absolute" / "relative" when
        # it's active and the relative flag is set accordingly.
        if "thin_surface" in scene_data and isinstance(
                scene_data["thin_surface"], dict):
            meta = scene_data["thin_surface"]
            loaded_mode = meta.get("density_mode", None)
            if loaded_mode is None:
                # Legacy inference -- never expose "independent" for a
                # checkpoint that didn't carry the discriminator.
                if meta.get("active", False):
                    self._thin_surface_density_mode = (
                        "relative" if meta.get("relative_delta", False)
                        else "absolute")
                else:
                    self._thin_surface_density_mode = "scalar"
            else:
                # Explicit discriminator. Mixed state is a malformed
                # checkpoint -- reject with a clear error.
                if loaded_mode == "independent":
                    if meta.get("relative_delta", False):
                        raise RuntimeError(
                            "load_pt: malformed checkpoint -- "
                            "thin_surface.density_mode='independent' "
                            "but relative_delta is also True. The two "
                            "modes are mutually exclusive (LC64 plan "
                            "v3); this checkpoint cannot be resumed."
                        )
                    if "raw_plus" not in scene_data or "raw_minus" not in scene_data:
                        raise RuntimeError(
                            "load_pt: malformed checkpoint -- "
                            "density_mode='independent' requires raw_plus "
                            "and raw_minus tensors; missing one or both."
                        )
                    rp = scene_data["raw_plus"].to(self.device)
                    rm = scene_data["raw_minus"].to(self.device)
                    N_pp = self.primal_points.shape[0]
                    if (tuple(rp.shape) != (N_pp, 1)
                            or tuple(rm.shape) != (N_pp, 1)):
                        raise RuntimeError(
                            f"load_pt: malformed checkpoint -- raw shape "
                            f"mismatch (raw_plus={tuple(rp.shape)}, "
                            f"raw_minus={tuple(rm.shape)}, expected "
                            f"(N,1) with N={N_pp})."
                        )
                    if not (torch.isfinite(rp).all()
                            and torch.isfinite(rm).all()):
                        raise RuntimeError(
                            "load_pt: malformed checkpoint -- raw_plus / "
                            "raw_minus contain non-finite values."
                        )
                    geometry_active = bool(meta.get("active", False))
                    K = int(meta.get("K", 4))
                    assert_supported_thin_K(K)
                    if geometry_active:
                        geometry_shapes = {
                            "quaternions": (N_pp, 4),
                            "texel_sites_2d": (N_pp, K, 2),
                            "texel_heights": (N_pp, K),
                        }
                        for name, shape in geometry_shapes.items():
                            if name not in scene_data:
                                raise RuntimeError(
                                    f"load_pt: independent checkpoint missing "
                                    f"required geometry tensor {name}")
                            value = scene_data[name].to(self.device)
                            if tuple(value.shape) != shape:
                                raise RuntimeError(
                                    f"load_pt: independent {name} shape "
                                    f"{tuple(value.shape)} != {shape}")
                            setattr(self, name, nn.Parameter(value))
                    self.raw_plus = nn.Parameter(rp)
                    self.raw_minus = nn.Parameter(rm)
                    self._thin_surface_density_mode = "independent"
                    self._thin_surface_active = geometry_active
                    self._thin_K = K
                    self._thin_surface_scheduler_cfg = meta.get(
                        "scheduler_cfg", None)
                    self._raw_side_scheduler_cfg = meta.get(
                        "raw_side_scheduler_cfg", None)
                    self._thin_surface_start = int(meta.get(
                        "raw_side_start",
                        meta.get("start", -1)))
                    # Force the legacy relative flag off (mixed state
                    # was already rejected above; this is the safe
                    # secondary guard).
                    self._thin_surface_relative_delta = False
                    print(f"[load_pt] restored independent-side state: "
                          f"N={N_pp}, raw_side_lr_cfg="
                          f"{self._raw_side_scheduler_cfg}")
                else:
                    # Explicit non-independent discriminator -- only
                    # safe values are "scalar", "absolute", "relative".
                    if loaded_mode not in ("scalar", "absolute", "relative"):
                        raise RuntimeError(
                            f"load_pt: unknown thin_surface.density_mode="
                            f"{loaded_mode!r}; expected one of "
                            f"'scalar', 'absolute', 'relative', "
                            f"'independent'."
                        )
                    self._thin_surface_density_mode = loaded_mode
        else:
            # No thin_surface metadata at all -- definitely legacy.
            # Don't override the __init__ default ("scalar") unless a
            # legacy thin-surface block was loaded above, in which case
            # the legacy inference already set the discriminator.
            if not hasattr(self, "_thin_surface_density_mode"):
                self._thin_surface_density_mode = "scalar"

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


import types as _types


def load_model_for_mesh(model_path, activation_scale=1.0, device="cuda"):
    """Load a CTScene checkpoint ready for IDW mesh extraction.

    Restores points/density, rebuilds the Delaunay triangulation (applying
    the resulting permutation to parameters in-place), and refreshes
    adjacency/aabb_tree so idw_query can be called immediately.

    Returns a fully-initialized CTScene in eval mode.
    """
    args = _types.SimpleNamespace(
        init_points=64000,
        final_points=512000,
        activation_scale=activation_scale,
        init_scale=1.05,
        init_type="random",
        init_density=0.0,
    )
    model = CTScene(args, device=torch.device(device))
    model.load_pt(model_path)

    with torch.no_grad():
        pts = model.primal_points.detach().contiguous()
        try:
            model.triangulation = radfoam.Triangulation(pts)
        except radfoam.TriangulationFailedError as e:
            if "duplicate" not in str(e):
                raise
            import numpy as np
            pts_np = pts.cpu().numpy()
            sort_idx = np.lexsort(pts_np.T[::-1])
            sorted_pts = pts_np[sort_idx]
            dists = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)
            extent = float(np.abs(pts_np).max())
            eps = max(1e-6, extent * 1e-5)
            keep_in_sorted = np.concatenate([[True], dists > eps])
            keep_idx = np.sort(sort_idx[keep_in_sorted])
            keep_idx = torch.from_numpy(keep_idx).to(pts.device)
            print(f"Removed {pts.shape[0] - len(keep_idx)} near-duplicate points (eps={eps:.2e}) before triangulation")
            pts = pts[keep_idx].contiguous()
            model.primal_points = torch.nn.Parameter(pts)
            model.density = torch.nn.Parameter(model.density.detach()[keep_idx])
            for attr in ("density_grad", "density_peak", "delta_raw", "cov_raw"):
                if hasattr(model, attr) and getattr(model, attr) is not None:
                    setattr(model, attr, torch.nn.Parameter(getattr(model, attr).detach()[keep_idx]))
            model.triangulation = radfoam.Triangulation(pts)
        perm = model.triangulation.permutation().to(torch.long)
        model.primal_points = torch.nn.Parameter(pts[perm])
        model.density = torch.nn.Parameter(model.density.detach()[perm])

    model.update_triangulation(rebuild=False)
    model.eval()
    return model
