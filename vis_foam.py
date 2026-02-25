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

    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        query = coords_flat[start:end]  # (B, 3)
        B = query.shape[0]

        # Find containing cell
        nn_idx = radfoam.nn(points, field["aabb_tree"], query).long()  # (B,)

        # Gather neighbor counts and build padded neighbor tensor
        counts = adj_off[nn_idx + 1] - adj_off[nn_idx]  # (B,)
        max_k = min(int(counts.max().item()), 40)
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

        # IDW weights: 1/dist^2
        eps = 1e-16
        inv_dist = 1.0 / (dist_sq + eps)
        inv_dist[~valid] = 0.0
        weights = inv_dist / inv_dist.sum(dim=1, keepdim=True)  # (B, max_k+1)

        # Weighted sum of activated densities
        vals = activated[pad_idx]  # (B, max_k+1)
        vals[~valid] = 0.0
        result[start:end] = (weights * vals).sum(dim=1)

    return result.reshape(original_shape).cpu().numpy()


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

    return grid.cpu().numpy()


def visualize_slices(density_slices, idw_slices, cell_density_slices,
                     vmax=1.0, writer_fn=None, out_path=None, title=None):
    """Plot density, IDW-interpolated, and cell-density slices in a 3x9 figure.

    Args:
        density_slices: list of 9 (res, res) arrays (3 axes x 3 coords)
        idw_slices: list of 9 matching arrays (natural neighbor IDW)
        cell_density_slices: list of 9 matching arrays
        vmax: density colorbar max
        writer_fn: optional callable(fig) for TensorBoard
        out_path: optional file path for saving
        title: optional figure title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_labels = ["X", "Y", "Z"]
    coords = [-0.2, 0.0, 0.2]

    fig, axs = plt.subplots(3, 9, figsize=(27, 9))

    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            # Raw density (left 3 cols)
            ax = axs[row, col]
            ax.imshow(density_slices[idx].T, origin="lower", cmap="gray",
                      vmin=0, vmax=vmax)
            ax.set_title(f"{axes_labels[row]}={coords[col]:.1f}")
            ax.axis("off")

            # IDW interpolated (middle 3 cols)
            ax = axs[row, col + 3]
            ax.imshow(idw_slices[idx].T, origin="lower", cmap="gray",
                      vmin=0, vmax=vmax)
            ax.set_title(f"IDW {axes_labels[row]}={coords[col]:.1f}")
            ax.axis("off")

            # Cell density overlaid on density (right 3 cols)
            ax = axs[row, col + 6]
            ax.imshow(density_slices[idx].T, origin="lower", cmap="gray",
                      vmin=0, vmax=vmax)
            ax.imshow(cell_density_slices[idx].T, origin="lower",
                      cmap="hot", alpha=0.5)
            ax.set_title(f"cells {axes_labels[row]}={coords[col]:.1f}")
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
        print(f"Saved {out_path}")
    if writer_fn is not None:
        writer_fn(fig)
    plt.close(fig)
