"""Direct slice visualization of a trained CTScene foam.

Queries only 2D slice coordinates (no full 3D volume), plus cell density
(point count per pixel bin) for diagnosing foam structure.

Usage from train.py or standalone:
    from vis_foam import (load_density_field, query_density,
                          make_slice_coords, compute_cell_density_slice,
                          visualize_slices)
"""

import numpy as np
import torch
import torch.nn.functional as F
import radfoam

from voxelize import sample_interpolated
from radfoam_model.scene import IDWResult, idw_query


def field_from_model(model):
    """Build a field dict from a live CTScene (no checkpoint save/load)."""
    with torch.no_grad():
        adj = model.point_adjacency.long()
        adj_off = model.point_adjacency_offsets.long()
        _, cell_radius = radfoam.farthest_neighbor(
            model.primal_points, adj.to(torch.int32), adj_off.to(torch.int32)
        )
        return {
            "points": model.primal_points,
            "density_flat": model.density.squeeze(-1),
            "gradients": getattr(model, "density_grad", None),
            "grad_max_slope": getattr(model, "_gradient_max_slope", None),
            "adjacency": adj,
            "adjacency_offsets": adj_off,
            "aabb_tree": model.aabb_tree,
            "cell_radius": cell_radius,
            "device": model.primal_points.device,
        }


def load_density_field(model_path, device="cuda"):
    """Load a checkpoint and build an AABB tree for NN queries.

    Returns a dict with keys: points, density_flat, gradients,
    grad_max_slope, aabb_tree, device.
    """
    device = torch.device(device)
    scene_data = torch.load(model_path)
    points = scene_data["xyz"].to(device)
    density_flat = scene_data["density"].to(device).squeeze(-1)

    gradients = None
    grad_max_slope = None
    if "density_grad" in scene_data:
        gradients = scene_data["density_grad"].to(device)
        grad_max_slope = scene_data.get("gradient_max_slope", 5.0)

    adjacency = scene_data["adjacency"].to(device).to(torch.int32)
    adjacency_offsets = scene_data["adjacency_offsets"].to(device).to(torch.int32)
    aabb_tree = radfoam.build_aabb_tree(points)
    _, cell_radius = radfoam.farthest_neighbor(points, adjacency, adjacency_offsets)

    return {
        "points": points,
        "density_flat": density_flat,
        "gradients": gradients,
        "grad_max_slope": grad_max_slope,
        "adjacency": adjacency,
        "adjacency_offsets": adjacency_offsets,
        "aabb_tree": aabb_tree,
        "cell_radius": cell_radius,
        "device": device,
    }


def query_density(field, coordinates):
    """Evaluate the density field at arbitrary coordinates.

    Args:
        field: dict from load_density_field()
        coordinates: numpy or torch array of shape (..., 3)

    Returns:
        numpy array of shape (...) with density values
    """
    original_shape = coordinates.shape[:-1]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coords_flat = coordinates.reshape(-1, 3).to(field["device"])

    result = torch.zeros(coords_flat.shape[0], device=field["device"])
    batch_size = 4_000_000

    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        query = coords_flat[start:end]
        nn_indices = radfoam.nn(field["points"], field["aabb_tree"], query).long()

        if field["gradients"] is not None:
            result[start:end] = sample_interpolated(
                query, nn_indices,
                field["points"], field["density_flat"],
                field["gradients"], field["grad_max_slope"],
            )
        else:
            result[start:end] = F.softplus(
                field["density_flat"][nn_indices], beta=10
            )

    return result.reshape(original_shape).cpu().numpy()


def _idw_query(query, field, activated, sigma, sigma_v, global_max_k=None,
               per_cell_sigma=False, per_neighbor_sigma=False):
    """Core bilateral IDW for a batch of query points. Thin wrapper around scene.idw_query."""
    return idw_query(
        query, field["points"], field["adjacency"], field["adjacency_offsets"],
        field["aabb_tree"], activated, sigma, sigma_v,
        global_max_k=global_max_k,
        per_cell_sigma=per_cell_sigma,
        per_neighbor_sigma=per_neighbor_sigma,
        cell_radius=field.get("cell_radius"),
    )


def sample_idw(field, coordinates, sigma=0.01, sigma_v=None,
               per_cell_sigma=False, per_neighbor_sigma=False):
    """Inverse-distance weighted interpolation over Voronoi neighbors.

    For each query point, finds the containing cell, gathers its Voronoi
    neighbors, and returns the IDW-weighted average of their activated
    densities (softplus of raw values).

    When sigma_v is set, applies Gaussian bilateral weighting: neighbors with
    dissimilar density to the containing cell are downweighted by
    exp(-(mu_i - mu_ref)² / sigma_v²).

    Args:
        field: dict from load_density_field() or field_from_model()
        coordinates: numpy or torch array of shape (..., 3)
        sigma: length scale for exp(-dist/sigma) spatial weighting
            (or sigma_scale when per_cell_sigma=True)
        sigma_v: value-similarity scale for bilateral weighting (None=disabled)
        per_cell_sigma: use per-cell adaptive sigma instead of global
        per_neighbor_sigma: Mode B (each neighbor uses its own radius)
            vs Mode A (all slots use containing cell's radius)

    Returns:
        numpy array of shape (...) with interpolated density values
    """
    original_shape = coordinates.shape[:-1]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coords_flat = coordinates.reshape(-1, 3).to(field["device"])

    activated = F.softplus(field["density_flat"], beta=10)
    adj_off = field["adjacency_offsets"]
    global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())

    result = torch.zeros(coords_flat.shape[0], device=field["device"])
    batch_size = 2_000_000

    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        res = _idw_query(coords_flat[start:end], field, activated,
                         sigma, sigma_v, global_max_k,
                         per_cell_sigma=per_cell_sigma,
                         per_neighbor_sigma=per_neighbor_sigma)
        result[start:end] = res.idw_result

    return torch.nan_to_num(result).reshape(original_shape).cpu().numpy()


def sample_idw_diagnostic(field, coordinates, sigma=0.001, sigma_v=None):
    """Like sample_idw but returns diagnostic channels.

    Runs WITHOUT batching (designed for a single 256x256 slice, ~65k queries).

    Args:
        field: density field dict from load_density_field() or field_from_model()
        coordinates: (H, W, 3) numpy array — single slice only
        sigma: spatial distance scale
        sigma_v: value-similarity scale for bilateral weighting (None=disabled)

    Returns:
        dict of (H, W) numpy arrays:
            nn_density      — softplus(density[nn_idx]), containing cell value
            idw_result      — final IDW weighted average
            diff            — nn_density - idw_result
            cell_weight     — weight of the containing cell (slot 0)
            min_neighbor_val — lowest activated density among valid neighbors
            neighbor_count  — number of Voronoi neighbors for containing cell
            max_neighbor_dist — largest distance to any valid neighbor
            value_weight    — mean bilateral value-similarity factor (if sigma_v set)
    """
    H, W = coordinates.shape[:2]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coords_flat = coordinates.reshape(-1, 3).to(field["device"])

    activated = F.softplus(field["density_flat"], beta=10)
    res = _idw_query(coords_flat, field, activated, sigma, sigma_v)

    nn_density = activated[res.nn_idx]
    idw_result = torch.nan_to_num(res.idw_result)
    cell_weight = torch.nan_to_num(res.weights[:, 0])

    # Min neighbor val: lowest activated density among valid neighbors (slots 1+)
    neighbor_vals = res.vals[:, 1:].clone()
    neighbor_valid = res.valid[:, 1:]
    neighbor_vals[~neighbor_valid] = float("inf")
    min_neighbor_val = neighbor_vals.min(dim=1).values
    no_neighbors = ~neighbor_valid.any(dim=1)
    min_neighbor_val[no_neighbors] = nn_density[no_neighbors]

    # Neighbor count
    neighbor_count = res.counts.float()

    # Max neighbor dist (excluding slot 0)
    neighbor_dist = res.dist_sq[:, 1:].sqrt()
    neighbor_dist_masked = neighbor_dist.clone()
    neighbor_dist_masked[~neighbor_valid] = 0.0
    max_neighbor_dist = neighbor_dist_masked.max(dim=1).values

    def to_hw(t):
        return t.reshape(H, W).cpu().numpy()

    diff = nn_density - idw_result

    result = {
        "nn_density": to_hw(nn_density),
        "idw_result": to_hw(idw_result),
        "diff": to_hw(diff),
        "cell_weight": to_hw(cell_weight),
        "min_neighbor_val": to_hw(min_neighbor_val),
        "neighbor_count": to_hw(neighbor_count),
        "max_neighbor_dist": to_hw(max_neighbor_dist),
    }

    if sigma_v is not None:
        ref_val = activated[res.nn_idx]
        val_diff = res.vals - ref_val.unsqueeze(1)
        vw = torch.exp(-val_diff * val_diff / (sigma_v * sigma_v))
        vw[~res.valid] = 0.0
        n_valid = res.valid.float().sum(dim=1).clamp(min=1)
        result["value_weight"] = to_hw(vw.sum(dim=1) / n_valid)

    return result


def visualize_idw_diagnostics(diag, diff_threshold=0.05, writer_fn=None,
                              out_path=None):
    """Create a multi-panel figure from sample_idw_diagnostic output.

    Args:
        diag: dict from sample_idw_diagnostic()
        diff_threshold: pixels with diff > this are highlighted as holes
        writer_fn: optional callable(fig) for TensorBoard logging
        out_path: optional file path for saving

    Returns:
        matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = [
        ("nn_density", "NN Density (containing cell)", "gray", None),
        ("idw_result", "IDW Result", "gray", None),
        ("diff", "Diff (NN - IDW)", "bwr", "symmetric"),
        ("cell_weight", "Cell Weight (slot 0)", "viridis", None),
        ("min_neighbor_val", "Min Neighbor Val", "gray", None),
        ("neighbor_count", "Neighbor Count", "plasma", None),
        ("max_neighbor_dist", "Max Neighbor Dist", "inferno", None),
    ]

    has_value_weight = "value_weight" in diag
    if has_value_weight:
        channels.append(("value_weight", "Value Weight (bilateral)", "viridis", None))

    ncols = 4 if not has_value_weight else 5
    fig, axs = plt.subplots(2, ncols, figsize=(5 * ncols, 10))
    axs_flat = axs.ravel()

    for i, (key, label, cmap, mode) in enumerate(channels):
        ax = axs_flat[i]
        data = diag[key]
        kwargs = {"origin": "lower", "cmap": cmap}
        if mode == "symmetric":
            abs_max = max(np.abs(data).max(), 1e-6)
            kwargs["vmin"] = -abs_max
            kwargs["vmax"] = abs_max
        elif key in ("nn_density", "idw_result", "min_neighbor_val"):
            kwargs["vmin"] = 0
            kwargs["vmax"] = max(diag["nn_density"].max(), 1e-6)
        elif key == "value_weight":
            kwargs["vmin"] = 0
            kwargs["vmax"] = 1
        im = ax.imshow(data.T, **kwargs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Hole mask overlay in the panel after channels
    hole_panel_idx = len(channels)
    ax = axs_flat[hole_panel_idx]
    hole_mask = diag["diff"] > diff_threshold
    n_holes = hole_mask.sum()
    ax.imshow(diag["idw_result"].T, origin="lower", cmap="gray",
              vmin=0, vmax=max(diag["nn_density"].max(), 1e-6))
    if n_holes > 0:
        ax.imshow(hole_mask.T, origin="lower", cmap="Reds", alpha=0.5)
    ax.set_title(f"Hole Mask (diff>{diff_threshold}, n={n_holes})", fontsize=9)
    ax.axis("off")

    # Hide any unused panels
    for j in range(hole_panel_idx + 1, len(axs_flat)):
        axs_flat[j].axis("off")

    fig.suptitle("IDW Diagnostic Channels", fontsize=13)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
    if writer_fn is not None:
        writer_fn(fig)

    plt.close(fig)
    return fig


def supersample_slice(sample_fn, field, axis, coord, resolution, extent,
                      ss=2, **kwargs):
    """Run a slice sampling function at ss× resolution and avg-pool back.

    Args:
        sample_fn: callable(field, coords, **kwargs) -> (H, W) numpy array
        field: density field dict
        axis, coord, resolution, extent: same args as make_slice_coords
        ss: supersample factor (1 = no supersampling)
        **kwargs: forwarded to sample_fn (e.g. sigma, sigma_v)

    Returns:
        (resolution, resolution) numpy array
    """
    if ss <= 1:
        return sample_fn(field, make_slice_coords(axis, coord, resolution, extent), **kwargs)
    coords_hi = make_slice_coords(axis, coord, resolution * ss, extent)
    hi = sample_fn(field, coords_hi, **kwargs)  # (resolution*ss, resolution*ss)
    t = torch.from_numpy(hi).float().unsqueeze(0).unsqueeze(0)
    return F.avg_pool2d(t, kernel_size=ss).squeeze().numpy()


def make_slice_coords(axis, coord, resolution, extent):
    """Build a 2D grid of 3D query positions for a single slice.

    Args:
        axis: 0, 1, or 2 for X, Y, Z
        coord: world-space position along the sliced axis
        resolution: number of pixels per side
        extent: half-extent of the grid (spans [-extent, extent])

    Returns:
        (resolution, resolution, 3) numpy array
    """
    lin = np.linspace(-extent, extent, resolution)
    other = [a for a in range(3) if a != axis]
    u, v = np.meshgrid(lin, lin, indexing="ij")

    coords = np.zeros((resolution, resolution, 3), dtype=np.float32)
    coords[:, :, axis] = coord
    coords[:, :, other[0]] = u
    coords[:, :, other[1]] = v
    return coords


def compute_voronoi_edges(field, axis, coord, resolution, extent):
    """Compute Voronoi cell borders for a 2D slice.

    Returns:
        (resolution, resolution) bool array — True at border pixels.
    """
    coords = make_slice_coords(axis, coord, resolution, extent)
    coords_flat = torch.from_numpy(coords).reshape(-1, 3).to(field["device"])

    # NN lookup in batches to avoid OOM at high resolution
    cell_map = torch.empty(coords_flat.shape[0], dtype=torch.int64,
                           device=field["device"])
    batch_size = 4_000_000
    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        cell_map[start:end] = radfoam.nn(
            field["points"], field["aabb_tree"], coords_flat[start:end]
        ).long()
    cell_map = cell_map.reshape(resolution, resolution)

    # Detect borders: pixel differs from any 4-connected neighbor
    border = torch.zeros(resolution, resolution, dtype=torch.bool,
                         device=cell_map.device)
    border[:-1, :] |= (cell_map[:-1, :] != cell_map[1:, :])
    border[1:, :]  |= (cell_map[1:, :] != cell_map[:-1, :])
    border[:, :-1] |= (cell_map[:, :-1] != cell_map[:, 1:])
    border[:, 1:]  |= (cell_map[:, 1:] != cell_map[:, :-1])

    return border.cpu().numpy()


def visualize_cell_heatmap(cell_density_slices, writer_fn=None):
    """Render cell heatmaps as a separate 3x3 figure.

    Args:
        cell_density_slices: list of 9 (res, res) arrays (3 axes x 3 coords)
        writer_fn: optional callable(fig) for TensorBoard
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    axes_labels = ["X", "Y", "Z"]
    coords = [-0.2, 0.0, 0.2]
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            ax = axs[row, col]
            ax.imshow(cell_density_slices[idx].T, origin="lower", cmap="hot")
            ax.set_title(f"{axes_labels[row]}={coords[col]:.1f}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Cell Density Heatmap", fontsize=12)
    fig.tight_layout()
    if writer_fn:
        writer_fn(fig)
    plt.close(fig)


def visualize_grad_weights(points, grad_weights, axes=(0, 1, 2),
                           slice_coords=(-0.2, 0.0, 0.2),
                           resolution=128, extent=1.0,
                           writer_fn=None):
    """Render per-cell gradient agreement weights as a 3×3 slice figure.

    Each pixel shows the mean agreement weight of cells in a thin slab.
    0 (red) = noisy / suppressed, 1 (green) = coherent / preserved.
    NaN (gray) = no cells in pixel.

    Args:
        points: (N, 3) CPU tensor of cell centres
        grad_weights: (N,) CPU tensor in [0, 1]
        axes: sequence of 3 axis indices (rows)
        slice_coords: sequence of 3 slice positions (columns)
        resolution: pixel resolution per axis
        extent: half-extent of the scene volume
        writer_fn: optional callable(fig) for TensorBoard
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thickness = 15 * (2 * extent / resolution)
    axes_labels = ["X", "Y", "Z"]

    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    for row, ax in enumerate(axes):
        other = [a for a in range(3) if a != ax]
        for col, coord in enumerate(slice_coords):
            slab_mask = (points[:, ax] - coord).abs() < thickness / 2
            slab_pts = points[slab_mask]
            slab_w = grad_weights[slab_mask].float()

            w_sum = torch.zeros(resolution, resolution)
            cnt = torch.zeros(resolution, resolution)
            if slab_pts.shape[0] > 0:
                ix = ((slab_pts[:, other[0]] + extent) / (2 * extent) * resolution).long().clamp(0, resolution - 1)
                iy = ((slab_pts[:, other[1]] + extent) / (2 * extent) * resolution).long().clamp(0, resolution - 1)
                w_sum.index_put_((ix, iy), slab_w, accumulate=True)
                cnt.index_put_((ix, iy), torch.ones(slab_pts.shape[0]), accumulate=True)

            mean_w = (w_sum / cnt.clamp(min=1.0)).numpy()
            mean_w[cnt.numpy() == 0] = float("nan")

            axs[row, col].imshow(mean_w.T, origin="lower", cmap="RdYlGn", vmin=0.0, vmax=1.0)
            axs[row, col].set_title(f"{axes_labels[ax]}={coord:.1f}", fontsize=8)
            axs[row, col].axis("off")

    fig.suptitle("Grad agreement weight (0=suppressed, 1=coherent)", fontsize=11)
    fig.tight_layout()
    if writer_fn:
        writer_fn(fig)
    plt.close(fig)


def compute_cell_density_slice(points, axis, coord, resolution, extent,
                               slab_thickness=None, device="cuda"):
    """Count cell centers per pixel bin in a thin slab around a slice.

    Args:
        points: (N, 3) tensor of cell centers
        axis: 0, 1, or 2
        coord: slice position along axis
        resolution: grid resolution
        extent: half-extent
        slab_thickness: thickness of slab (default: 5 voxel widths)
        device: torch device

    Returns:
        (resolution, resolution) numpy array of point counts per bin
    """
    if slab_thickness is None:
        slab_thickness = 15 * (2 * extent / resolution)

    pts = points.to(device)
    mask = (pts[:, axis] - coord).abs() < slab_thickness / 2
    slab = pts[mask]

    other = [a for a in range(3) if a != axis]
    grid = torch.zeros(resolution, resolution, device=device)

    if slab.shape[0] > 0:
        ix = ((slab[:, other[0]] + extent) / (2 * extent) * resolution).long().clamp(0, resolution - 1)
        iy = ((slab[:, other[1]] + extent) / (2 * extent) * resolution).long().clamp(0, resolution - 1)
        ones = torch.ones(slab.shape[0], device=device)
        grid.index_put_((ix, iy), ones, accumulate=True)

    # Gaussian blur for smoother heatmap
    sigma = 0.75
    ks = 5
    coords = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
    grid = F.conv2d(grid.unsqueeze(0).unsqueeze(0), kernel, padding=ks // 2).squeeze()

    return grid.cpu().numpy()


def render_volume_drr(volume, rays, extent=1.0, num_samples=256):
    """Render a DRR (digitally reconstructed radiograph) by ray-summing through a volume.

    Uses PyTorch grid_sample on GPU for fast trilinear interpolation.

    Args:
        volume: (R, R, R) numpy array — the 3D volume to project
        rays: (H, W, 6) numpy array — ray origins (3) + directions (3)
        extent: half-extent of the volume (grid spans [-extent, extent]^3)
        num_samples: number of sample points along each ray

    Returns:
        projection: (H, W) numpy array — accumulated ray-sum values
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vol_t = torch.from_numpy(volume).float().to(device)  # (R, R, R)
    rays_t = torch.from_numpy(rays).float().to(device)   # (H, W, 6)
    H, W = rays_t.shape[:2]
    origins = rays_t[..., :3]   # (H, W, 3)
    dirs = rays_t[..., 3:6]     # (H, W, 3)

    # Ray-AABB intersection via slab method
    inv_dir = 1.0 / (dirs + 1e-12)
    t_min_all = (-extent - origins) * inv_dir
    t_max_all = (extent - origins) * inv_dir
    t_near = torch.minimum(t_min_all, t_max_all)
    t_far = torch.maximum(t_min_all, t_max_all)
    t_entry = t_near.max(dim=-1).values.clamp(min=0.0)  # (H, W)
    t_exit = t_far.min(dim=-1).values                    # (H, W)
    valid = t_exit > t_entry

    # Sample points along rays: (H, W, S, 3)
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    t_samples = t_entry[..., None] + t_vals * (t_exit - t_entry)[..., None]  # (H, W, S)
    sample_pts = origins[..., None, :] + dirs[..., None, :] * t_samples[..., None]

    # grid_sample for 5D: volume is (1, 1, D, H, W) where D=axis0, H=axis1, W=axis2
    # grid coords are (x_W, y_H, z_D) — reversed relative to our (x, y, z) world coords
    grid = sample_pts / extent  # [-1, 1] range, shape (H, W, S, 3) as (x, y, z)
    grid = grid.flip(-1)  # reverse to (z, y, x) for grid_sample convention
    grid_5d = grid.reshape(1, -1, 1, 1, 3)   # (1, H*W*S, 1, 1, 3)
    vol_5d = vol_t.unsqueeze(0).unsqueeze(0)  # (1, 1, R, R, R)
    sampled = F.grid_sample(vol_5d, grid_5d, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    vals = sampled.reshape(H, W, num_samples)  # (H, W, S)

    # Ray-sum with step length
    step_length = (t_exit - t_entry) / num_samples  # (H, W)
    projection = vals.sum(dim=-1) * step_length
    projection[~valid] = 0.0

    return projection.cpu().numpy().astype(np.float32)


def load_gt_volume(data_path, dataset_type, dataset_args=None):
    """Load or generate a ground-truth 3D density volume.

    Args:
        data_path: path to the dataset directory
        dataset_type: 'r2_gaussian', 'ct_synthetic', 'lodopab', etc.
        dataset_args: full dataset args namespace (used for lodopab sample_index / split_override)

    Returns:
        (G,G,G) numpy array, or None if GT not available.
        Bbox is assumed [-1,1]^3.
    """
    if dataset_type == "r2_gaussian":
        import os
        vol_path = os.path.join(data_path, "vol_gt.npy")
        if os.path.exists(vol_path):
            return np.load(vol_path)
        return None
    elif dataset_type == "ct_synthetic":
        G = 256
        lin = np.linspace(-1, 1, G)
        x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
        vol = np.zeros((G, G, G), dtype=np.float32)
        vol[x ** 2 + y ** 2 + z ** 2 <= 1.0] = 1.0
        return vol
    elif dataset_type == "ct_cube":
        from data_loader.ct_cube import make_single_cube_scene, make_2x2x2_scene, make_gt_volume
        if "2x2x2" in data_path:
            boxes, densities = make_2x2x2_scene()
        else:
            boxes, densities = make_single_cube_scene()
        return make_gt_volume(boxes, densities, resolution=256, extent=1.0)
    elif dataset_type == "lodopab":
        import h5py, os
        G = 256
        SAMPLES_PER_FILE = 128
        sample_index = getattr(dataset_args, "sample_index", 0) if dataset_args is not None else 0
        split = getattr(dataset_args, "split_override", "") or "train"
        file_idx = sample_index // SAMPLES_PER_FILE
        in_file_idx = sample_index % SAMPLES_PER_FILE
        gt_path = os.path.join(data_path, f"ground_truth_{split}_{file_idx:03d}.hdf5")
        if not os.path.exists(gt_path):
            return None
        with h5py.File(gt_path, "r") as f:
            gt2d = f["data"][in_file_idx].astype(np.float32)  # (362, 362), values in [0,1]
        # Resize to G×G
        from skimage.transform import resize
        gt2d = resize(gt2d, (G, G), order=1, anti_aliasing=True).astype(np.float32)
        # Embed 2D slice into a 3D volume by replicating across z.
        # Axis convention: volume[x, y, z] — the CT image is in the xy-plane.
        # LoDoPaB image axes: row=y (top→bottom), col=x (left→right).
        gt3d = np.stack([gt2d] * G, axis=2)  # (G, G, G): same slice at every z
        return gt3d
    elif dataset_type == "two_detectct":
        import os, tifffile
        G = 256
        sample_index = getattr(dataset_args, "sample_index", 1) if dataset_args is not None else 1
        mode = getattr(dataset_args, "mode", 1) if dataset_args is not None else 1
        # 2DeteCT includes FBP reconstructions for slices 2001-3000 in the RecSeg package.
        # Look for reconstruction TIFF in slice dir.
        slice_dir = os.path.join(data_path, f"slice{sample_index:05d}", f"mode{mode}")
        recon_path = os.path.join(slice_dir, "reconstruction.tif")
        if not os.path.exists(recon_path):
            return None
        gt2d = tifffile.imread(recon_path).astype(np.float32)  # (H, W)
        from skimage.transform import resize
        gt2d = resize(gt2d, (G, G), order=1, anti_aliasing=True).astype(np.float32)
        # Normalize to [0, 1]
        gt2d = (gt2d - gt2d.min()) / (gt2d.max() - gt2d.min() + 1e-9)
        gt3d = np.stack([gt2d] * G, axis=2)
        return gt3d
    elif dataset_type in ("more", "aapm_mayo"):
        # GT is the input volume itself (we forward-projected from it).
        # Reconstruct a 256³ volume from the source slices for evaluation.
        import os, glob, cv2
        from skimage.transform import resize as sk_resize

        G = 256
        split = getattr(dataset_args, "split_override", "") or "train"
        sample_index = getattr(dataset_args, "sample_index", 0) if dataset_args is not None else 0

        if dataset_type == "more":
            split_dir = os.path.join(data_path, split)
            if not os.path.isdir(split_dir):
                return None
            patients = sorted(set(
                os.path.basename(f).split("_")[0]
                for f in glob.glob(os.path.join(split_dir, "*.png"))
            ))
            if sample_index >= len(patients):
                return None
            patient_id = patients[sample_index]
            files = sorted(glob.glob(os.path.join(split_dir, f"{patient_id}_*.png")))
            slices = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                      for f in files]
        else:  # aapm_mayo npy layout
            npy_dir = os.path.join(data_path, split)
            if not os.path.isdir(npy_dir):
                return None
            files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
            if not files:
                return None
            slices = [np.load(f).squeeze().astype(np.float32) for f in files]

        if not slices:
            return None

        # Resize each slice to G×G and stack into (G, G, N_slices)
        resized = np.stack([
            sk_resize(s, (G, G), order=1, anti_aliasing=True).astype(np.float32)
            for s in slices
        ], axis=0)   # (N_slices, G, G)

        # Interpolate N_slices → G in z using numpy
        n_slices = resized.shape[0]
        if n_slices != G:
            z_in = np.linspace(0, 1, n_slices)
            z_out = np.linspace(0, 1, G)
            from scipy.interpolate import interp1d
            interp = interp1d(z_in, resized, axis=0, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
            resized = interp(z_out).astype(np.float32)   # (G, G, G)

        # Normalize to [0, 1]
        vmin, vmax = resized.min(), resized.max()
        if vmax > vmin:
            resized = (resized - vmin) / (vmax - vmin)

        # Return as (G, G, G): axes are (x, y, z)
        return resized.transpose(1, 2, 0)   # (G, G, N_z→G)
    return None


def load_r2_volume(data_path):
    """Load R2-Gaussian prediction volume if available.

    Args:
        data_path: path to the dataset directory

    Returns:
        (G,G,G) numpy array, or None if not found.
    """
    import os
    vol_path = os.path.join(data_path, "vol_r2.npy")
    if os.path.exists(vol_path):
        return np.load(vol_path)
    return None


def voxelize_volumes(field, resolution, extent, sigma, sigma_v):
    """Voxelize the field into two 3D volumes in one pass.

    Raw volume: softplus(density[nearest_cell]) — constant per Voronoi cell.
    IDW volume: Gaussian bilateral natural-neighbor interpolation matching
    the CUDA tracing kernel (exp(-d²/σ²) spatial, exp(-Δμ²/σ_v²) bilateral).

    Args:
        field: dict from load_density_field() or field_from_model()
        resolution: grid resolution per axis
        extent: half-extent (grid spans [-extent, extent]^3)
        sigma: spatial scale for Gaussian weighting
        sigma_v: bilateral value-similarity scale

    Returns:
        (raw_volume, idw_volume) — both (resolution, resolution, resolution) numpy arrays
    """
    device = field["device"]
    activated = F.softplus(field["density_flat"], beta=10)

    lin = torch.linspace(-extent, extent, resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    coords_flat = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    num_voxels = coords_flat.shape[0]
    raw_vol = torch.zeros(num_voxels, device=device)
    idw_vol = torch.zeros(num_voxels, device=device)

    adj_off = field["adjacency_offsets"]
    global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())
    batch_size = 2_000_000

    for start in range(0, num_voxels, batch_size):
        end = min(start + batch_size, num_voxels)
        res = _idw_query(coords_flat[start:end], field, activated,
                         sigma, sigma_v, global_max_k)
        raw_vol[start:end] = activated[res.nn_idx]
        idw_vol[start:end] = torch.nan_to_num(res.idw_result)

    raw_vol = raw_vol.reshape(resolution, resolution, resolution)
    idw_vol = idw_vol.reshape(resolution, resolution, resolution)

    return raw_vol.cpu().numpy(), idw_vol.cpu().numpy()


def sample_gt_slice(gt_volume, axis, coord, resolution, extent):
    """Extract a 2D slice from a GT volume at a world-space coordinate.

    Args:
        gt_volume: (G,G,G) numpy array in [-extent, extent]^3
        axis: 0, 1, or 2
        coord: world-space position along axis
        resolution: output resolution
        extent: half-extent of the volume

    Returns:
        (resolution, resolution) numpy array, or None if gt_volume is None.
    """
    if gt_volume is None:
        return None
    G = gt_volume.shape[0]
    # Map world coord to voxel index
    idx = int((coord + extent) / (2 * extent) * (G - 1) + 0.5)
    idx = max(0, min(G - 1, idx))

    if axis == 0:
        raw_slice = gt_volume[idx, :, :]
    elif axis == 1:
        raw_slice = gt_volume[:, idx, :]
    else:
        raw_slice = gt_volume[:, :, idx]

    # Resize to target resolution if needed
    if raw_slice.shape[0] != resolution or raw_slice.shape[1] != resolution:
        t = torch.from_numpy(raw_slice).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=(resolution, resolution), mode="bilinear",
                          align_corners=False)
        return t.squeeze().numpy()
    return raw_slice.copy()


def compute_slice_psnr(pred, gt):
    """PSNR between two (res, res) numpy arrays."""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_slice_ssim(pred, gt, window_size=11):
    """SSIM between two (res, res) numpy arrays (single-channel)."""
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return 1.0
    # Use torch conv2d for windowed SSIM
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)

    pad = window_size // 2
    mu1 = F.conv2d(pred_t, kernel, padding=pad)
    mu2 = F.conv2d(gt_t, kernel, padding=pad)

    sigma1_sq = F.conv2d(pred_t ** 2, kernel, padding=pad) - mu1 ** 2
    sigma2_sq = F.conv2d(gt_t ** 2, kernel, padding=pad) - mu2 ** 2
    sigma12 = F.conv2d(pred_t * gt_t, kernel, padding=pad) - mu1 * mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def sobel_filter_2d(img):
    """Sobel gradient magnitude of a (res, res) numpy array."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    t = F.pad(t, (1, 1, 1, 1), mode="replicate")
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).reshape(1, 1, 3, 3)
    gx = F.conv2d(t, sobel_x)
    gy = F.conv2d(t, sobel_y)
    mag = torch.log1p(1.0 * torch.sqrt(gx**2 + gy**2)).clamp(0, 1).squeeze().numpy()
    return mag


def visualize_slices(density_slices, idw_slices, cell_density_slices,
                     gt_slices=None, r2_slices=None, vmax=1.0, writer_fn=None,
                     writer_fn_interleaved=None, writer_fn_sobel=None,
                     out_path=None, title=None, voronoi_edges=None):
    """Plot density slices. 3x9 without GT, 6x9 with GT comparison.

    Also produces an interleaved view (grouped by slice instead of by
    vis type) if writer_fn_interleaved is provided, and a Sobel gradient
    magnitude view if writer_fn_sobel is provided.

    Args:
        density_slices: list of 9 (res, res) arrays (3 axes x 3 coords)
        idw_slices: list of 9 matching arrays (natural neighbor IDW)
        cell_density_slices: list of 9 matching arrays
        gt_slices: optional list of 9 (res, res) GT arrays (or Nones)
        vmax: density colorbar max
        writer_fn: optional callable(fig) for TensorBoard (by-type layout)
        writer_fn_interleaved: optional callable(fig) for TensorBoard (per-slice layout)
        writer_fn_sobel: optional callable(fig) for TensorBoard (Sobel-filtered view)
        out_path: optional file path for saving
        title: optional figure title
        voronoi_edges: optional list of 9 (res, res) bool arrays for Voronoi borders

    Returns:
        dict of average metrics if GT available, else None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_labels = ["X", "Y", "Z"]
    coords = [-0.2, 0.0, 0.2]

    has_gt = (gt_slices is not None
              and any(g is not None for g in gt_slices))
    nrows = 6 if has_gt else 3
    fig, axs = plt.subplots(nrows, 9, figsize=(27, nrows * 3))

    # Check if R2 slices are available
    has_r2 = (r2_slices is not None
              and any(s is not None for s in r2_slices))

    # Collect per-slice metrics
    raw_psnrs, raw_ssims = [], []
    idw_psnrs, idw_ssims = [], []
    blend_psnrs, blend_ssims = [], []
    r2_psnrs, r2_ssims = [], []
    sobel_raw_psnrs, sobel_raw_ssims = [], []
    sobel_idw_psnrs, sobel_idw_ssims = [], []
    sobel_blend_psnrs, sobel_blend_ssims = [], []
    sobel_r2_psnrs, sobel_r2_ssims = [], []
    # Store Sobel-filtered images for visualization
    sobel_raw_imgs, sobel_idw_imgs, sobel_gt_imgs = [], [], []
    sobel_blend_imgs, sobel_r2_imgs = [], []

    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            gt = gt_slices[idx] if has_gt else None

            # --- Row 0-2: Raw density (left 3 cols) ---
            ax = axs[row, col]
            ax.imshow(density_slices[idx].T, origin="lower", cmap="gray",
                      vmin=0, vmax=vmax)
            lbl = f"{axes_labels[row]}={coords[col]:.1f}"
            if gt is not None:
                p = compute_slice_psnr(density_slices[idx], gt)
                s = compute_slice_ssim(density_slices[idx], gt)
                raw_psnrs.append(p)
                raw_ssims.append(s)
                lbl += f" P={p:.1f} S={s:.2f}"
            ax.set_title(lbl, fontsize=8)
            ax.axis("off")

            # --- Row 0-2: R2 or IDW (middle 3 cols) ---
            ax = axs[row, col + 3]
            r2_slice = r2_slices[idx] if has_r2 else None
            if r2_slice is not None:
                ax.imshow(r2_slice.T, origin="lower", cmap="gray",
                          vmin=0, vmax=vmax)
                lbl = f"R2 {axes_labels[row]}={coords[col]:.1f}"
                if gt is not None:
                    p = compute_slice_psnr(r2_slice, gt)
                    s = compute_slice_ssim(r2_slice, gt)
                    r2_psnrs.append(p)
                    r2_ssims.append(s)
                    lbl += f" P={p:.1f} S={s:.2f}"
            else:
                ax.imshow(idw_slices[idx].T, origin="lower", cmap="gray",
                          vmin=0, vmax=vmax)
                lbl = f"IDW {axes_labels[row]}={coords[col]:.1f}"
                if gt is not None:
                    p = compute_slice_psnr(idw_slices[idx], gt)
                    s = compute_slice_ssim(idw_slices[idx], gt)
                    idw_psnrs.append(p)
                    idw_ssims.append(s)
                    lbl += f" P={p:.1f} S={s:.2f}"
            ax.set_title(lbl, fontsize=8)
            ax.axis("off")

            # --- Row 0-2: Density + Voronoi borders (right 3 cols) ---
            ax = axs[row, col + 6]
            ax.imshow(density_slices[idx].T, origin="lower", cmap="gray",
                      vmin=0, vmax=vmax)
            if voronoi_edges is not None and voronoi_edges[idx] is not None:
                edge_rgba = np.zeros((*voronoi_edges[idx].shape, 4))
                edge_rgba[voronoi_edges[idx], :] = [1, 0.3, 0, 0.7]  # orange, 70% opacity
                ax.imshow(edge_rgba.transpose(1, 0, 2), origin="lower")
            ax.set_title(f"borders {axes_labels[row]}={coords[col]:.1f}",
                         fontsize=8)
            ax.axis("off")

            if has_gt:
                # --- Row 3-5: GT slices (left 3 cols) ---
                ax = axs[row + 3, col]
                if gt is not None:
                    ax.imshow(gt.T, origin="lower", cmap="gray",
                              vmin=0, vmax=vmax)
                ax.set_title(f"GT {axes_labels[row]}={coords[col]:.1f}",
                             fontsize=8)
                ax.axis("off")

                # --- Row 3-5: IDW or Blend (middle 3 cols) ---
                ax = axs[row + 3, col + 3]
                if has_r2:
                    # When R2 is in top row, show IDW here
                    mid_img = idw_slices[idx]
                    lbl = f"IDW {axes_labels[row]}={coords[col]:.1f}"
                    if gt is not None:
                        p = compute_slice_psnr(mid_img, gt)
                        s = compute_slice_ssim(mid_img, gt)
                        idw_psnrs.append(p)
                        idw_ssims.append(s)
                        lbl += f" P={p:.1f} S={s:.2f}"
                else:
                    # Fallback: blend
                    mid_img = 0.5 * density_slices[idx] + 0.5 * idw_slices[idx]
                    lbl = f"blend {axes_labels[row]}={coords[col]:.1f}"
                    if gt is not None:
                        p = compute_slice_psnr(mid_img, gt)
                        s = compute_slice_ssim(mid_img, gt)
                        blend_psnrs.append(p)
                        blend_ssims.append(s)
                        lbl += f" P={p:.1f} S={s:.2f}"
                ax.imshow(mid_img.T, origin="lower", cmap="gray",
                          vmin=0, vmax=vmax)
                ax.set_title(lbl, fontsize=8)
                ax.axis("off")

                # --- Row 3-5: Difference GT-IDW or GT-raw (right 3 cols) ---
                ax = axs[row + 3, col + 6]
                if gt is not None:
                    if has_r2:
                        diff = gt - idw_slices[idx]
                    else:
                        diff = gt - density_slices[idx]
                    abs_max = max(np.abs(diff).max(), 1e-6)
                    ax.imshow(diff.T, origin="lower", cmap="bwr",
                              vmin=-abs_max, vmax=abs_max)
                ax.set_title(f"diff {axes_labels[row]}={coords[col]:.1f}",
                             fontsize=8)
                ax.axis("off")

                # --- Sobel-filtered metrics ---
                if gt is not None:
                    gt_sobel = sobel_filter_2d(gt)
                    raw_sobel = sobel_filter_2d(density_slices[idx])
                    idw_sobel = sobel_filter_2d(idw_slices[idx])
                    sobel_raw_psnrs.append(compute_slice_psnr(raw_sobel, gt_sobel))
                    sobel_raw_ssims.append(compute_slice_ssim(raw_sobel, gt_sobel))
                    sobel_idw_psnrs.append(compute_slice_psnr(idw_sobel, gt_sobel))
                    sobel_idw_ssims.append(compute_slice_ssim(idw_sobel, gt_sobel))
                    sobel_raw_imgs.append(raw_sobel)
                    sobel_idw_imgs.append(idw_sobel)
                    sobel_gt_imgs.append(gt_sobel)
                    if has_r2 and r2_slices[idx] is not None:
                        r2_sobel = sobel_filter_2d(r2_slices[idx])
                        sobel_r2_psnrs.append(compute_slice_psnr(r2_sobel, gt_sobel))
                        sobel_r2_ssims.append(compute_slice_ssim(r2_sobel, gt_sobel))
                        sobel_r2_imgs.append(r2_sobel)
                    else:
                        blend = 0.5 * density_slices[idx] + 0.5 * idw_slices[idx]
                        blend_sobel = sobel_filter_2d(blend)
                        sobel_blend_psnrs.append(compute_slice_psnr(blend_sobel, gt_sobel))
                        sobel_blend_ssims.append(compute_slice_ssim(blend_sobel, gt_sobel))
                        sobel_blend_imgs.append(blend_sobel)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
        print(f"Saved {out_path}")
    if writer_fn is not None:
        writer_fn(fig)

    # Build interleaved view by rearranging rendered tiles
    if writer_fn_interleaved is not None and has_gt:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        h, w, _ = buf.shape
        th, tw = h // nrows, w // 9
        new_buf = np.zeros_like(buf)
        for a in range(3):
            for ci in range(3):
                moves = [
                    ((a, ci),     (a * 2, ci * 3)),
                    ((a, ci + 3), (a * 2, ci * 3 + 1)),
                    ((a, ci + 6), (a * 2, ci * 3 + 2)),
                ]
                if has_gt:
                    moves += [
                        ((a + 3, ci),     (a * 2 + 1, ci * 3)),
                        ((a + 3, ci + 3), (a * 2 + 1, ci * 3 + 1)),
                        ((a + 3, ci + 6), (a * 2 + 1, ci * 3 + 2)),
                    ]
                for (sr, sc), (dr, dc) in moves:
                    new_buf[dr*th:(dr+1)*th, dc*tw:(dc+1)*tw] = \
                        buf[sr*th:(sr+1)*th, sc*tw:(sc+1)*tw]
        fig2, ax2 = plt.subplots(figsize=(27, nrows * 3))
        ax2.imshow(new_buf)
        ax2.axis("off")
        fig2.tight_layout(pad=0)
        writer_fn_interleaved(fig2)
        plt.close(fig2)

    # Build Sobel-filtered visualization (interleaved layout: grouped by slice)
    # With R2: top = Raw/R2/DiffRaw-GT, bottom = GT/IDW/DiffIDW-GT
    # Without R2: top = Raw/IDW/Blend, bottom = GT/DiffRaw/DiffIDW
    if writer_fn_sobel is not None and sobel_gt_imgs:
        sobel_vmax = max(
            max((s.max() for s in sobel_gt_imgs), default=1.0),
            max((s.max() for s in sobel_raw_imgs), default=1.0),
            max((s.max() for s in sobel_idw_imgs), default=1.0),
            1e-6,
        )
        sfig, saxs = plt.subplots(6, 9, figsize=(27, 18))
        si = 0
        for a in range(3):
            for ci in range(3):
                r0 = a * 2
                c0 = ci * 3

                if has_r2 and sobel_r2_imgs:
                    # Top row: Sobel Raw, Sobel R2, Diff Raw-GT
                    ax = saxs[r0, c0]
                    ax.imshow(sobel_raw_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    p, s = sobel_raw_psnrs[si], sobel_raw_ssims[si]
                    ax.set_title(f"SRaw {axes_labels[a]}={coords[ci]:.1f}"
                                 f" P={p:.1f} S={s:.2f}", fontsize=7)
                    ax.axis("off")

                    ax = saxs[r0, c0 + 1]
                    ax.imshow(sobel_r2_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    p, s = sobel_r2_psnrs[si], sobel_r2_ssims[si]
                    ax.set_title(f"SR2 {axes_labels[a]}={coords[ci]:.1f}"
                                 f" P={p:.1f} S={s:.2f}", fontsize=7)
                    ax.axis("off")

                    diff_raw = sobel_raw_imgs[si] - sobel_gt_imgs[si]
                    abs_max_r = max(np.abs(diff_raw).max(), 1e-6)
                    ax = saxs[r0, c0 + 2]
                    ax.imshow(diff_raw.T, origin="lower", cmap="bwr",
                              vmin=-abs_max_r, vmax=abs_max_r)
                    ax.set_title(f"SDiff Raw {axes_labels[a]}={coords[ci]:.1f}",
                                 fontsize=7)
                    ax.axis("off")

                    # Bottom row: Sobel GT, Sobel IDW, Diff IDW-GT
                    ax = saxs[r0 + 1, c0]
                    ax.imshow(sobel_gt_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    ax.set_title(f"SGT {axes_labels[a]}={coords[ci]:.1f}",
                                 fontsize=7)
                    ax.axis("off")

                    ax = saxs[r0 + 1, c0 + 1]
                    ax.imshow(sobel_idw_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    p, s = sobel_idw_psnrs[si], sobel_idw_ssims[si]
                    ax.set_title(f"SIDW {axes_labels[a]}={coords[ci]:.1f}"
                                 f" P={p:.1f} S={s:.2f}", fontsize=7)
                    ax.axis("off")

                    diff_idw = sobel_idw_imgs[si] - sobel_gt_imgs[si]
                    abs_max_i = max(np.abs(diff_idw).max(), 1e-6)
                    ax = saxs[r0 + 1, c0 + 2]
                    ax.imshow(diff_idw.T, origin="lower", cmap="bwr",
                              vmin=-abs_max_i, vmax=abs_max_i)
                    ax.set_title(f"SDiff IDW {axes_labels[a]}={coords[ci]:.1f}",
                                 fontsize=7)
                    ax.axis("off")
                else:
                    # Fallback: top = Raw/IDW/Blend, bottom = GT/DiffRaw/DiffIDW
                    ax = saxs[r0, c0]
                    ax.imshow(sobel_raw_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    p, s = sobel_raw_psnrs[si], sobel_raw_ssims[si]
                    ax.set_title(f"SRaw {axes_labels[a]}={coords[ci]:.1f}"
                                 f" P={p:.1f} S={s:.2f}", fontsize=7)
                    ax.axis("off")

                    ax = saxs[r0, c0 + 1]
                    ax.imshow(sobel_idw_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    p, s = sobel_idw_psnrs[si], sobel_idw_ssims[si]
                    ax.set_title(f"SIDW {axes_labels[a]}={coords[ci]:.1f}"
                                 f" P={p:.1f} S={s:.2f}", fontsize=7)
                    ax.axis("off")

                    ax = saxs[r0, c0 + 2]
                    ax.imshow(sobel_blend_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    p, s = sobel_blend_psnrs[si], sobel_blend_ssims[si]
                    ax.set_title(f"SBlend {axes_labels[a]}={coords[ci]:.1f}"
                                 f" P={p:.1f} S={s:.2f}", fontsize=7)
                    ax.axis("off")

                    ax = saxs[r0 + 1, c0]
                    ax.imshow(sobel_gt_imgs[si].T, origin="lower", cmap="gray",
                              vmin=0, vmax=sobel_vmax)
                    ax.set_title(f"SGT {axes_labels[a]}={coords[ci]:.1f}",
                                 fontsize=7)
                    ax.axis("off")

                    diff_raw = sobel_raw_imgs[si] - sobel_gt_imgs[si]
                    abs_max_r = max(np.abs(diff_raw).max(), 1e-6)
                    ax = saxs[r0 + 1, c0 + 1]
                    ax.imshow(diff_raw.T, origin="lower", cmap="bwr",
                              vmin=-abs_max_r, vmax=abs_max_r)
                    ax.set_title(f"SDiff Raw {axes_labels[a]}={coords[ci]:.1f}",
                                 fontsize=7)
                    ax.axis("off")

                    diff_idw = sobel_idw_imgs[si] - sobel_gt_imgs[si]
                    abs_max_i = max(np.abs(diff_idw).max(), 1e-6)
                    ax = saxs[r0 + 1, c0 + 2]
                    ax.imshow(diff_idw.T, origin="lower", cmap="bwr",
                              vmin=-abs_max_i, vmax=abs_max_i)
                    ax.set_title(f"SDiff IDW {axes_labels[a]}={coords[ci]:.1f}",
                                 fontsize=7)
                    ax.axis("off")

                si += 1

        sfig.suptitle("Sobel Gradient Magnitude", fontsize=13)
        sfig.tight_layout()
        writer_fn_sobel(sfig)
        plt.close(sfig)

    plt.close(fig)

    if has_gt and raw_psnrs:
        metrics = {
            "raw_psnr": np.mean(raw_psnrs),
            "raw_ssim": np.mean(raw_ssims),
        }
        if idw_psnrs:
            metrics["idw_psnr"] = np.mean(idw_psnrs)
            metrics["idw_ssim"] = np.mean(idw_ssims)
        if blend_psnrs:
            metrics["blend_psnr"] = np.mean(blend_psnrs)
            metrics["blend_ssim"] = np.mean(blend_ssims)
        if sobel_raw_psnrs:
            metrics["sobel_raw_psnr"] = np.mean(sobel_raw_psnrs)
            metrics["sobel_raw_ssim"] = np.mean(sobel_raw_ssims)
        if sobel_idw_psnrs:
            metrics["sobel_idw_psnr"] = np.mean(sobel_idw_psnrs)
            metrics["sobel_idw_ssim"] = np.mean(sobel_idw_ssims)
        if sobel_blend_psnrs:
            metrics["sobel_blend_psnr"] = np.mean(sobel_blend_psnrs)
            metrics["sobel_blend_ssim"] = np.mean(sobel_blend_ssims)
        return metrics
    return None


def log_density_histogram(model, writer, step):
    """Log histogram of raw density values with 0.5-wide bins from -10 to 10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        raw = model.density.detach().squeeze().cpu().numpy()
        bin_edges = np.arange(-10, 10.5, 0.5)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(raw.clip(-10, 10), bins=bin_edges)
        ax.set_xlabel("Raw density")
        ax.set_ylabel("Count")
        ax.set_xlim(-10, 10)
        writer.add_figure("diagnostics/density_histogram", fig, step)
        plt.close(fig)


def log_volume_slices(model, writer, gt_volume, step, experiment_name):
    """Log slices of the stored reference/init volume, edge weight map, and GT.

    Reads model._ref_volume (R,R,R) and optionally model._ref_weight (R,R,R).
    Intended to be called at step 0 to inspect the init/reference density prior
    and verify which regions are strongly vs. weakly regularized.
    """
    if not hasattr(model, "_ref_volume") or model._ref_volume is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_np = model._ref_volume.cpu().numpy()   # (R, R, R)
    has_gt = gt_volume is not None
    has_weight = getattr(model, "_ref_weight", None) is not None
    weight_np = model._ref_weight.cpu().numpy() if has_weight else None

    axes = [0, 1, 2]
    positions = [-0.2, 0.0, 0.2]
    axis_names = ["X", "Y", "Z"]
    n_cols = len(axes) * len(positions)        # 9
    row_labels = ["ref/init vol"]
    if has_weight:
        row_labels.append("reg weight")
    if has_gt:
        row_labels.append("GT vol")
    n_rows = len(row_labels)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axs = axs[None, :]

    for col, (a, c) in enumerate((a, c) for a in axes for c in positions):
        row = 0
        axs[row, col].set_title(f"{axis_names[a]}={c:.1f}", fontsize=7)

        ref_sl = sample_gt_slice(ref_np, a, c, ref_np.shape[0], 1.0)
        axs[row, col].imshow(ref_sl.T, origin="lower", cmap="gray", vmin=0, vmax=1.0)
        axs[row, col].axis("off")
        row += 1

        if has_weight:
            w_sl = sample_gt_slice(weight_np, a, c, weight_np.shape[0], 1.0)
            # viridis: yellow=high weight (strongly regularized), purple=low (free)
            axs[row, col].imshow(w_sl.T, origin="lower", cmap="viridis", vmin=0, vmax=1.0)
            axs[row, col].axis("off")
            row += 1

        if has_gt:
            gt_sl = sample_gt_slice(gt_volume, a, c, gt_volume.shape[0], 1.0)
            if gt_sl is not None:
                axs[row, col].imshow(gt_sl.T, origin="lower", cmap="gray", vmin=0, vmax=1.0)
            axs[row, col].axis("off")

    for row, label in enumerate(row_labels):
        axs[row, 0].set_ylabel(label, fontsize=8)

    fig.suptitle(f"Reference volume slices (step {step})", fontsize=9)
    fig.tight_layout()
    writer.add_figure(f"ref_vol_slices/{experiment_name}", fig, global_step=step)
    plt.close(fig)
