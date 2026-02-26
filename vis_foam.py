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


def field_from_model(model):
    """Build a field dict from a live CTScene (no checkpoint save/load)."""
    with torch.no_grad():
        return {
            "points": model.primal_points,
            "density_flat": model.density.squeeze(-1),
            "gradients": getattr(model, "density_grad", None),
            "grad_max_slope": getattr(model, "_gradient_max_slope", None),
            "adjacency": model.point_adjacency.long(),
            "adjacency_offsets": model.point_adjacency_offsets.long(),
            "aabb_tree": model.aabb_tree,
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

    adjacency = scene_data["adjacency"].to(device).long()
    adjacency_offsets = scene_data["adjacency_offsets"].to(device).long()
    aabb_tree = radfoam.build_aabb_tree(points)

    return {
        "points": points,
        "density_flat": density_flat,
        "gradients": gradients,
        "grad_max_slope": grad_max_slope,
        "adjacency": adjacency,
        "adjacency_offsets": adjacency_offsets,
        "aabb_tree": aabb_tree,
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


def sample_idw(field, coordinates):
    """Inverse-distance weighted interpolation over Voronoi neighbors.

    For each query point, finds the containing cell, gathers its Voronoi
    neighbors, and returns the IDW-weighted average of their activated
    densities (softplus of raw values).

    Args:
        field: dict from load_density_field() or field_from_model()
        coordinates: numpy or torch array of shape (..., 3)

    Returns:
        numpy array of shape (...) with interpolated density values
    """
    original_shape = coordinates.shape[:-1]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coords_flat = coordinates.reshape(-1, 3).to(field["device"])

    points = field["points"]
    density_flat = field["density_flat"]
    adj = field["adjacency"]
    adj_off = field["adjacency_offsets"]
    activated = F.softplus(density_flat, beta=10)  # (N,)

    result = torch.zeros(coords_flat.shape[0], device=field["device"])
    batch_size = 2_000_000
    global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())

    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        query = coords_flat[start:end]  # (B, 3)
        B = query.shape[0]

        # Find containing cell
        nn_idx = radfoam.nn(points, field["aabb_tree"], query).long()  # (B,)

        # Gather neighbor counts and build padded neighbor tensor
        counts = adj_off[nn_idx + 1] - adj_off[nn_idx]  # (B,)
        max_k = global_max_k
        offsets = adj_off[nn_idx]  # (B,) start offset per cell

        # Padded (B, max_k+1) index tensor: slot 0 = cell itself, 1..max_k = neighbors
        pad_idx = torch.zeros(B, max_k + 1, dtype=torch.long, device=field["device"])
        valid = torch.zeros(B, max_k + 1, dtype=torch.bool, device=field["device"])

        # Slot 0: the containing cell
        pad_idx[:, 0] = nn_idx
        valid[:, 0] = True

        # Slots 1..max_k: Voronoi neighbors
        k_range = torch.arange(max_k, device=field["device"])
        has_k = counts.unsqueeze(1) > k_range.unsqueeze(0)  # (B, max_k)
        # Gather indices where valid
        flat_offsets = offsets.unsqueeze(1) + k_range.unsqueeze(0)  # (B, max_k)
        flat_offsets = flat_offsets.clamp(max=adj.shape[0] - 1)
        pad_idx[:, 1:] = adj[flat_offsets]
        valid[:, 1:] = has_k

        # Compute squared distances from query to each candidate center
        centers = points[pad_idx]  # (B, max_k+1, 3)
        dist_sq = ((query.unsqueeze(1) - centers) ** 2).sum(dim=-1)  # (B, max_k+1)

        # IDW weights: 1/dist^4 (sharper weighting toward nearest cell)
        eps = 1e-10
        inv_dist = 1.0 / (dist_sq + eps) ** 2
        inv_dist[~valid] = 0.0
        weights = inv_dist / inv_dist.sum(dim=1, keepdim=True)  # (B, max_k+1)

        # Weighted sum of activated densities
        vals = activated[pad_idx]  # (B, max_k+1)
        vals[~valid] = 0.0
        result[start:end] = (weights * vals).sum(dim=1)

    return result.reshape(original_shape).cpu().numpy()


def sample_idw_diagnostic(field, coordinates):
    """Like sample_idw but returns diagnostic channels.

    Runs WITHOUT batching (designed for a single 256x256 slice, ~65k queries).

    Args:
        field: density field dict from load_density_field() or field_from_model()
        coordinates: (H, W, 3) numpy array — single slice only

    Returns:
        dict of (H, W) numpy arrays:
            nn_density      — softplus(density[nn_idx]), containing cell value
            idw_result      — final IDW weighted average
            diff            — nn_density - idw_result
            cell_weight     — weight of the containing cell (slot 0)
            min_neighbor_val — lowest activated density among valid neighbors
            neighbor_count  — number of Voronoi neighbors for containing cell
            max_neighbor_dist — largest distance to any valid neighbor
    """
    H, W = coordinates.shape[:2]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coords_flat = coordinates.reshape(-1, 3).to(field["device"])

    points = field["points"]
    density_flat = field["density_flat"]
    adj = field["adjacency"]
    adj_off = field["adjacency_offsets"]
    activated = F.softplus(density_flat, beta=10)  # (N,)

    B = coords_flat.shape[0]
    global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())

    # Find containing cell
    nn_idx = radfoam.nn(points, field["aabb_tree"], coords_flat).long()  # (B,)

    # NN density: containing cell's activated value
    nn_density = activated[nn_idx]  # (B,)

    # Gather neighbor counts and build padded neighbor tensor
    counts = adj_off[nn_idx + 1] - adj_off[nn_idx]  # (B,)
    max_k = global_max_k
    offsets = adj_off[nn_idx]  # (B,)

    # Padded (B, max_k+1) index tensor: slot 0 = cell itself, 1..max_k = neighbors
    pad_idx = torch.zeros(B, max_k + 1, dtype=torch.long, device=field["device"])
    valid = torch.zeros(B, max_k + 1, dtype=torch.bool, device=field["device"])

    # Slot 0: the containing cell
    pad_idx[:, 0] = nn_idx
    valid[:, 0] = True

    # Slots 1..max_k: Voronoi neighbors
    k_range = torch.arange(max_k, device=field["device"])
    has_k = counts.unsqueeze(1) > k_range.unsqueeze(0)  # (B, max_k)
    flat_offsets = offsets.unsqueeze(1) + k_range.unsqueeze(0)  # (B, max_k)
    flat_offsets = flat_offsets.clamp(max=adj.shape[0] - 1)
    pad_idx[:, 1:] = adj[flat_offsets]
    valid[:, 1:] = has_k

    # Compute squared distances from query to each candidate center
    centers = points[pad_idx]  # (B, max_k+1, 3)
    dist_sq = ((coords_flat.unsqueeze(1) - centers) ** 2).sum(dim=-1)  # (B, max_k+1)

    # IDW weights: 1/dist^4
    eps = 1e-10
    inv_dist = 1.0 / (dist_sq + eps) ** 2
    inv_dist[~valid] = 0.0
    weights = inv_dist / inv_dist.sum(dim=1, keepdim=True)  # (B, max_k+1)

    # Activated values
    vals = activated[pad_idx]  # (B, max_k+1)
    vals[~valid] = 0.0

    # IDW result
    idw_result = (weights * vals).sum(dim=1)  # (B,)

    # Cell weight (slot 0)
    cell_weight = weights[:, 0]  # (B,)

    # Min neighbor val: lowest activated density among valid neighbors (slots 1+)
    neighbor_vals = vals[:, 1:].clone()  # (B, max_k)
    neighbor_valid = valid[:, 1:]  # (B, max_k)
    neighbor_vals[~neighbor_valid] = float("inf")
    min_neighbor_val = neighbor_vals.min(dim=1).values  # (B,)
    # For cells with zero neighbors, use the cell's own value
    no_neighbors = ~neighbor_valid.any(dim=1)
    min_neighbor_val[no_neighbors] = nn_density[no_neighbors]

    # Neighbor count
    neighbor_count = counts.float()  # (B,)

    # Max neighbor dist (sqrt of max dist_sq among valid entries, excluding slot 0)
    neighbor_dist_sq = dist_sq[:, 1:].clone()  # (B, max_k)
    neighbor_dist_sq[~neighbor_valid] = 0.0
    max_neighbor_dist = neighbor_dist_sq.max(dim=1).values.sqrt()  # (B,)

    # Reshape all to (H, W) numpy
    def to_hw(t):
        return t.reshape(H, W).cpu().numpy()

    diff = nn_density - idw_result

    return {
        "nn_density": to_hw(nn_density),
        "idw_result": to_hw(idw_result),
        "diff": to_hw(diff),
        "cell_weight": to_hw(cell_weight),
        "min_neighbor_val": to_hw(min_neighbor_val),
        "neighbor_count": to_hw(neighbor_count),
        "max_neighbor_dist": to_hw(max_neighbor_dist),
    }


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

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
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
        im = ax.imshow(data.T, **kwargs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Last panel: hole mask overlay
    ax = axs_flat[7]
    hole_mask = diag["diff"] > diff_threshold
    n_holes = hole_mask.sum()
    ax.imshow(diag["idw_result"].T, origin="lower", cmap="gray",
              vmin=0, vmax=max(diag["nn_density"].max(), 1e-6))
    if n_holes > 0:
        ax.imshow(hole_mask.T, origin="lower", cmap="Reds", alpha=0.5)
    ax.set_title(f"Hole Mask (diff>{diff_threshold}, n={n_holes})", fontsize=9)
    ax.axis("off")

    fig.suptitle("IDW Diagnostic Channels", fontsize=13)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
    if writer_fn is not None:
        writer_fn(fig)

    plt.close(fig)
    return fig


def supersample_slice(sample_fn, field, axis, coord, resolution, extent,
                      ss=2):
    """Run a slice sampling function at ss× resolution and avg-pool back.

    Args:
        sample_fn: callable(field, coords) -> (H, W) numpy array
        field: density field dict
        axis, coord, resolution, extent: same args as make_slice_coords
        ss: supersample factor (1 = no supersampling)

    Returns:
        (resolution, resolution) numpy array
    """
    if ss <= 1:
        return sample_fn(field, make_slice_coords(axis, coord, resolution, extent))
    coords_hi = make_slice_coords(axis, coord, resolution * ss, extent)
    hi = sample_fn(field, coords_hi)  # (resolution*ss, resolution*ss)
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


def load_gt_volume(data_path, dataset_type):
    """Load or generate a ground-truth 3D density volume.

    Args:
        data_path: path to the dataset directory
        dataset_type: 'r2_gaussian' or 'ct_synthetic'

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
    return None


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


def visualize_slices(density_slices, idw_slices, cell_density_slices,
                     gt_slices=None, vmax=1.0, writer_fn=None,
                     writer_fn_interleaved=None, out_path=None, title=None):
    """Plot density slices. 3x9 without GT, 6x9 with GT comparison.

    Also produces an interleaved view (grouped by slice instead of by
    vis type) if writer_fn_interleaved is provided.

    Args:
        density_slices: list of 9 (res, res) arrays (3 axes x 3 coords)
        idw_slices: list of 9 matching arrays (natural neighbor IDW)
        cell_density_slices: list of 9 matching arrays
        gt_slices: optional list of 9 (res, res) GT arrays (or Nones)
        vmax: density colorbar max
        writer_fn: optional callable(fig) for TensorBoard (by-type layout)
        writer_fn_interleaved: optional callable(fig) for TensorBoard (per-slice layout)
        out_path: optional file path for saving
        title: optional figure title

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

    # Collect per-slice metrics
    raw_psnrs, raw_ssims = [], []
    idw_psnrs, idw_ssims = [], []
    blend_psnrs, blend_ssims = [], []

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

            # --- Row 0-2: IDW interpolated (middle 3 cols) ---
            ax = axs[row, col + 3]
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

            # --- Row 0-2: Cell density overlay (right 3 cols) ---
            ax = axs[row, col + 6]
            cd = cell_density_slices[idx]
            cd_res = cd.shape[0]
            ds_t = torch.from_numpy(density_slices[idx]).unsqueeze(0).unsqueeze(0)
            ds = F.avg_pool2d(ds_t, kernel_size=ds_t.shape[-1] // cd_res).squeeze().numpy()
            ax.imshow(ds.T, origin="lower", cmap="gray",
                      vmin=0, vmax=vmax)
            ax.imshow(cd.T, origin="lower",
                      cmap="hot", alpha=0.5)
            ax.set_title(f"cells {axes_labels[row]}={coords[col]:.1f}",
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

                # --- Row 3-5: Blended 50/50 (middle 3 cols) ---
                ax = axs[row + 3, col + 3]
                blend = 0.5 * density_slices[idx] + 0.5 * idw_slices[idx]
                ax.imshow(blend.T, origin="lower", cmap="gray",
                          vmin=0, vmax=vmax)
                lbl = f"blend {axes_labels[row]}={coords[col]:.1f}"
                if gt is not None:
                    p = compute_slice_psnr(blend, gt)
                    s = compute_slice_ssim(blend, gt)
                    blend_psnrs.append(p)
                    blend_ssims.append(s)
                    lbl += f" P={p:.1f} S={s:.2f}"
                ax.set_title(lbl, fontsize=8)
                ax.axis("off")

                # --- Row 3-5: Difference GT - raw (right 3 cols) ---
                ax = axs[row + 3, col + 6]
                if gt is not None:
                    diff = gt - density_slices[idx]
                    abs_max = max(np.abs(diff).max(), 1e-6)
                    ax.imshow(diff.T, origin="lower", cmap="bwr",
                              vmin=-abs_max, vmax=abs_max)
                ax.set_title(f"diff {axes_labels[row]}={coords[col]:.1f}",
                             fontsize=8)
                ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
        print(f"Saved {out_path}")
    if writer_fn is not None:
        writer_fn(fig)

    # Build interleaved view by rearranging rendered tiles
    if writer_fn_interleaved is not None:
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

    plt.close(fig)

    if has_gt and raw_psnrs:
        return {
            "raw_psnr": np.mean(raw_psnrs),
            "raw_ssim": np.mean(raw_ssims),
            "idw_psnr": np.mean(idw_psnrs),
            "idw_ssim": np.mean(idw_ssims),
            "blend_psnr": np.mean(blend_psnrs),
            "blend_ssim": np.mean(blend_ssims),
        }
    return None
