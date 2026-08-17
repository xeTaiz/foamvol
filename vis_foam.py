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


def _build_tet_topology(points):
    """Build Delaunay tet topology from a point cloud.

    Returns a dict of int64 GPU tensors keyed by:
      tets           (T, 4)  Triangulation vertex indices per tet
      tet_adjacency  (T, 4)  packed: 4*nbr_tet + face_slot; 4294967295 = boundary
      vert_to_tet    (M,)    one incident tet index per vertex (M = dedup count ≤ N)
      tri_perm       (M,)    perm[tri_vertex] = original point idx
      tri_inv_perm   (N,)    inv_perm[original_idx] = Triangulation vertex idx
    """
    N = points.shape[0]
    orig_to_dedup = None
    keep_t = None

    try:
        tri = radfoam.Triangulation(points)
    except radfoam.TriangulationFailedError as e:
        if "duplicate" not in str(e):
            raise
        # Deduplicate: identical logic to scene.py load_model
        pts_np = points.detach().cpu().numpy()
        sort_idx = np.lexsort(pts_np.T[::-1])
        diffs = np.linalg.norm(np.diff(pts_np[sort_idx], axis=0), axis=1)
        eps = max(1e-6, float(np.abs(pts_np).max()) * 1e-5)
        keep_in_sorted = np.concatenate([[True], diffs > eps])
        keep_sorted_pos = np.where(keep_in_sorted)[0]  # positions in sort order that are kept

        # For each sorted position, find the nearest preceding kept position
        pred = np.searchsorted(keep_sorted_pos, np.arange(N), side="right") - 1
        pred = np.clip(pred, 0, len(keep_sorted_pos) - 1)
        # rep_orig[orig_idx] = original idx of its kept representative
        rep_orig = np.empty(N, dtype=np.int64)
        rep_orig[sort_idx] = sort_idx[keep_sorted_pos[pred]]

        keep_orig = np.sort(sort_idx[keep_in_sorted])  # (M,) kept original indices
        M = len(keep_orig)
        orig_to_keep_pos = np.empty(N, dtype=np.int64)
        orig_to_keep_pos[keep_orig] = np.arange(M)
        orig_to_dedup = orig_to_keep_pos[rep_orig]  # (N,) → dedup index in [0, M)

        keep_t = torch.from_numpy(keep_orig).to(points.device)
        tri = radfoam.Triangulation(points[keep_t].contiguous())
        print(f"_build_tet_topology: removed {N - M} near-duplicate points before triangulation")

    tri_perm_raw = tri.permutation().long()
    M_tri = tri_perm_raw.shape[0]

    if orig_to_dedup is None:
        tri_perm = tri_perm_raw  # (N,) original indices
        tri_inv_perm = torch.empty(N, dtype=torch.long, device=points.device)
        tri_inv_perm[tri_perm] = torch.arange(N, dtype=torch.long, device=points.device)
    else:
        tri_perm = keep_t[tri_perm_raw]  # (M,) maps tri vertex → original idx
        # Inverse: dedup idx → tri vertex idx
        tri_inv_dedup = torch.empty(M_tri, dtype=torch.long, device=points.device)
        tri_inv_dedup[tri_perm_raw] = torch.arange(M_tri, dtype=torch.long, device=points.device)
        # Full inverse: original idx → tri vertex idx (removed points map to their representative)
        orig_to_dedup_t = torch.from_numpy(orig_to_dedup).to(points.device)
        tri_inv_perm = tri_inv_dedup[orig_to_dedup_t]  # (N,)

    return {
        "tets": tri.tets().long(),
        "tet_adjacency": tri.tet_adjacency().long(),
        "vert_to_tet": tri.vert_to_tet().long(),
        "tri_perm": tri_perm,
        "tri_inv_perm": tri_inv_perm,
    }


def _topology_from_live_triangulation(tri, num_points, device):
    """Extract topology from an already-valid live ``Triangulation``.

    Rebuilding solely for diagnostics can fail for an otherwise renderable
    near-degenerate point cloud.  The live triangulation is the one currently
    backing the renderer, so reuse it whenever its permutation covers all
    model points.  Callers may fall back to ``_build_tet_topology`` if that
    invariant is unavailable (e.g. a lightweight test stub).
    """
    tri_perm = tri.permutation().long()
    if tri_perm.numel() != num_points:
        raise ValueError("live triangulation does not cover all model points")
    tri_inv_perm = torch.empty(num_points, dtype=torch.long, device=device)
    tri_inv_perm[tri_perm] = torch.arange(num_points, dtype=torch.long, device=device)
    return {
        "tets": tri.tets().long(),
        "tet_adjacency": tri.tet_adjacency().long(),
        "vert_to_tet": tri.vert_to_tet().long(),
        "tri_perm": tri_perm,
        "tri_inv_perm": tri_inv_perm,
    }


def _thin_surface_query_config(field):
    """Return validated split-query state, or None for scalar fallback."""
    try:
        active = field.get("thin_surface_active", False)
        mode = field.get("thin_surface_density_mode")
        relative = field.get("thin_surface_relative_delta")
        if (not isinstance(active, (bool, np.bool_)) or not bool(active)
                or mode not in ("absolute", "relative", "independent")
                or not isinstance(relative, (bool, np.bool_))
                or bool(relative) != (mode == "relative")):
            return None

        points = field["points"]
        density = field["density_flat"]
        cell_radius = field["cell_radius"]
        quaternions = field["quaternions"]
        texel_sites_2d = field["texel_sites_2d"]
        texel_heights = field["texel_heights"]
        tensors = (points, density, cell_radius, quaternions,
                   texel_sites_2d, texel_heights)
        if mode == "independent":
            raw_plus = field["raw_plus"]
            raw_minus = field["raw_minus"]
            side_tensors = (raw_plus, raw_minus)
        else:
            density_delta = field["density_delta"]
            side_tensors = (density_delta,)
        tensors += side_tensors
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            return None

        n_cells = points.shape[0] if points.ndim == 2 else 0
        valid_side_shapes = all(
            tuple(value.shape) in ((n_cells,), (n_cells, 1))
            for value in side_tensors
        )
        if (points.shape != (n_cells, 3)
                or density.ndim not in (1, 2) or density.numel() != n_cells
                or density.shape[0] != n_cells
                or cell_radius.numel() != n_cells
                or not valid_side_shapes
                or quaternions.shape != (n_cells, 4)
                or texel_sites_2d.ndim != 3
                or texel_sites_2d.shape[0] != n_cells
                or texel_sites_2d.shape[1] < 1
                or texel_sites_2d.shape[2] != 2
                or texel_heights.shape != texel_sites_2d.shape[:2]
                or not all(value.is_floating_point() for value in tensors)
                or any(value.device != points.device for value in tensors)):
            return None

        activation_scale = float(field["activation_scale"])
        delta_max_frac = float(field["thin_surface_delta_max_frac"])
        if (not np.isfinite(activation_scale) or activation_scale < 0.0
                or not np.isfinite(delta_max_frac) or delta_max_frac < 0.0):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    return mode, activation_scale, delta_max_frac


def field_from_model(model):
    """Build a field dict from a live CTScene (no checkpoint save/load)."""
    with torch.no_grad():
        adj = model.point_adjacency.long()
        adj_off = model.point_adjacency_offsets.long()
        _, cell_radius = radfoam.farthest_neighbor(
            model.primal_points, adj.to(torch.int32), adj_off.to(torch.int32)
        )
        try:
            topology = _topology_from_live_triangulation(
                model.triangulation, model.primal_points.shape[0],
                model.primal_points.device,
            )
        except (AttributeError, ValueError):
            # Retain the standalone/test fallback when no compatible live
            # triangulation is available.
            topology = _build_tet_topology(model.primal_points)

        try:
            activation_scale = float(getattr(model, "activation_scale", 1.0))
        except (TypeError, ValueError, OverflowError):
            activation_scale = 1.0
        if not np.isfinite(activation_scale):
            activation_scale = 1.0

        field = {
            "points": model.primal_points,
            "density_flat": model.density.squeeze(-1),
            "gradients": getattr(model, "density_grad", None),
            "grad_max_slope": getattr(model, "_gradient_max_slope", None),
            "adjacency": adj,
            "adjacency_offsets": adj_off,
            "aabb_tree": model.aabb_tree,
            "cell_radius": cell_radius,
            "device": model.primal_points.device,
            "activation_scale": activation_scale,
            **topology,
        }

        thin_names = (
            "quaternions", "texel_sites_2d", "texel_heights",
            "_thin_surface_density_mode", "_thin_surface_relative_delta",
            "_thin_surface_delta_max_frac",
        )
        if (getattr(model, "_thin_surface_active", False)
                and all(hasattr(model, name) for name in thin_names)):
            density_mode = model._thin_surface_density_mode
            thin_field = {
                "thin_surface_active": True,
                "thin_surface_density_mode": density_mode,
                "thin_surface_relative_delta": model._thin_surface_relative_delta,
                "thin_surface_delta_max_frac": model._thin_surface_delta_max_frac,
                "quaternions": model.quaternions,
                "texel_sites_2d": model.texel_sites_2d,
                "texel_heights": model.texel_heights,
            }
            if density_mode == "independent":
                thin_field.update({
                    "raw_plus": getattr(model, "raw_plus", None),
                    "raw_minus": getattr(model, "raw_minus", None),
                })
            else:
                thin_field["density_delta"] = getattr(
                    model, "density_delta", None)
            candidate = {**field, **thin_field}
            if _thin_surface_query_config(candidate) is not None:
                field.update(thin_field)

        return field


def load_density_field(model_path, device="cuda", load_thin_surface=True):
    """Load a checkpoint and build an AABB tree for NN queries.

    Returns a dict with keys: points, density_flat, gradients,
    grad_max_slope, aabb_tree, device, activation_scale. When
    load_thin_surface is True (default) and the checkpoint carries active
    thin-surface state, also sets the same thin_surface_*/quaternions/
    texel_*/density_delta (or raw_plus/raw_minus) keys field_from_model()
    sets for a live model, so query_density()/voxelize_volumes() resolve
    the two-sided split instead of silently falling back to flat
    per-cell density.
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

    field = {
        "points": points,
        "density_flat": density_flat,
        "gradients": gradients,
        "grad_max_slope": grad_max_slope,
        "adjacency": adjacency,
        "adjacency_offsets": adjacency_offsets,
        "aabb_tree": aabb_tree,
        "cell_radius": cell_radius,
        "device": device,
        "activation_scale": float(scene_data.get("activation_scale", 1.0)),
        **_build_tet_topology(points),
    }

    ts_meta = scene_data.get("thin_surface") if load_thin_surface else None
    if isinstance(ts_meta, dict) and ts_meta.get("active", False):
        density_mode = ts_meta.get("density_mode")
        if density_mode is None:
            density_mode = ("relative" if ts_meta.get("relative_delta", False)
                            else "absolute")
        thin_field = {
            "thin_surface_active": True,
            "thin_surface_density_mode": density_mode,
            "thin_surface_relative_delta": (density_mode == "relative"),
            "thin_surface_delta_max_frac": float(ts_meta.get("delta_max_frac", 0.5)),
            "quaternions": scene_data["quaternions"].to(device),
            "texel_sites_2d": scene_data["texel_sites_2d"].to(device),
            "texel_heights": scene_data["texel_heights"].to(device),
        }
        if density_mode == "independent":
            thin_field["raw_plus"] = scene_data["raw_plus"].to(device)
            thin_field["raw_minus"] = scene_data["raw_minus"].to(device)
        else:
            thin_field["density_delta"] = scene_data["density_delta"].to(device)
        candidate = {**field, **thin_field}
        if _thin_surface_query_config(candidate) is not None:
            field.update(thin_field)

    return field


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

        thin_config = _thin_surface_query_config(field)
        if thin_config is not None:
            # Keep the split dependency local to live thin-surface slices.
            from split_voxelize import split_cell_query
            density_mode, activation_scale, delta_max_frac = thin_config
            if density_mode == "independent":
                value, _, _ = split_cell_query(
                    query, field["points"], nn_indices, field["density_flat"],
                    None, field["quaternions"], field["texel_sites_2d"],
                    field["texel_heights"], field["cell_radius"],
                    thin_temp=10.0, activation_scale=activation_scale,
                    blend_eps=0.0, density_mode="independent",
                    raw_plus=field["raw_plus"], raw_minus=field["raw_minus"],
                    delta_max_frac=delta_max_frac,
                )
            else:
                value, _, _ = split_cell_query(
                    query, field["points"], nn_indices, field["density_flat"],
                    field["density_delta"], field["quaternions"],
                    field["texel_sites_2d"], field["texel_heights"],
                    field["cell_radius"], thin_temp=10.0,
                    activation_scale=activation_scale, blend_eps=0.0,
                    density_mode=density_mode, delta_max_frac=delta_max_frac,
                )
            result[start:end] = value
        elif field["gradients"] is not None:
            result[start:end] = sample_interpolated(
                query, nn_indices,
                field["points"], field["density_flat"],
                field["gradients"], field["grad_max_slope"],
            )
        else:
            try:
                activation_scale = float(field.get("activation_scale", 1.0))
            except (TypeError, ValueError, OverflowError):
                activation_scale = 1.0
            if not np.isfinite(activation_scale):
                activation_scale = 1.0
            result[start:end] = activation_scale * F.softplus(
                field["density_flat"][nn_indices], beta=10
            )

    return result.reshape(original_shape).cpu().numpy()


def _idw_query(query, field, activated, sigma, sigma_v, global_max_k=None,
               per_cell_sigma=False, per_neighbor_sigma=False, hop=1):
    """Core bilateral IDW for a batch of query points. Thin wrapper around scene.idw_query."""
    return idw_query(
        query, field["points"], field["adjacency"], field["adjacency_offsets"],
        field["aabb_tree"], activated, sigma, sigma_v,
        global_max_k=global_max_k,
        per_cell_sigma=per_cell_sigma,
        per_neighbor_sigma=per_neighbor_sigma,
        cell_radius=field.get("cell_radius"),
        hop=hop,
    )


def sample_idw(field, coordinates, sigma=0.01, sigma_v=None,
               per_cell_sigma=False, per_neighbor_sigma=False, hop=1):
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
                         per_neighbor_sigma=per_neighbor_sigma,
                         hop=hop)
        result[start:end] = res.idw_result

    return torch.nan_to_num(result).reshape(original_shape).cpu().numpy()


def sample_linear_idw(field, coordinates, sigma_v=0.0, eps_frac=0.05):
    """Linear inverse-distance-weighted interpolation over Voronoi neighbours.

    Mirrors the CUDA ``ct_visualization_linear_idw`` kernel exactly.

    For each query point, finds the containing cell and its Voronoi neighbours,
    then computes:
        w_j = 1 / (d_j² + eps²)
    where eps² = (eps_frac * cell_radius)².  With sigma_v > 0 each weight is
    additionally multiplied by exp(-(mu_j - mu_ref)² / sigma_v²).

    Args:
        field: dict from load_density_field() or field_from_model()
        coordinates: numpy or torch array of shape (..., 3)
        sigma_v: bilateral value sigma (0 = disabled; mirrors CUDA idw_sigma_v)
        eps_frac: eps = eps_frac * cell_radius (default 0.05 = 5% of cell radius)

    Returns:
        numpy array of shape (...) with interpolated density values
    """
    original_shape = coordinates.shape[:-1]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    coords_flat = coordinates.reshape(-1, 3).to(field["device"])
    device = field["device"]

    activated = F.softplus(field["density_flat"], beta=10)
    points = field["points"]                          # (N, 3)
    adj    = field["adjacency"]                       # (E,)
    adj_off = field["adjacency_offsets"]              # (N+1,)
    cell_radius = field.get("cell_radius")            # (N,) or None
    aabb_tree   = field["aabb_tree"]

    use_bilateral = sigma_v > 0.0
    sigma_v_sq = sigma_v ** 2 if use_bilateral else 1.0

    batch_size = 500_000
    result = torch.zeros(coords_flat.shape[0], device=device)

    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        q = coords_flat[start:end]                    # (B, 3)
        B = q.shape[0]

        # Find containing cell via nearest-neighbour lookup
        nn_idx = radfoam.nn(q, aabb_tree, points)     # (B,)

        # Self distance squared
        p_self  = points[nn_idx]                      # (B, 3)
        d_sq_self = ((q - p_self) ** 2).sum(-1)       # (B,)

        # eps² scaled to containing cell radius
        if cell_radius is not None:
            r_self  = cell_radius[nn_idx]             # (B,)
            eps_sq  = (eps_frac * r_self) ** 2        # (B,)
        else:
            eps_sq  = torch.full((B,), (eps_frac * 1e-3) ** 2, device=device)

        mu_ref  = activated[nn_idx]                   # (B,)
        w_sum   = 1.0 / (d_sq_self + eps_sq)          # (B,)
        mu_w    = w_sum * mu_ref                      # (B,)

        # Iterate over neighbour slots using CSR adjacency
        # Build a dense (B, max_degree) neighbour matrix
        deg     = (adj_off[nn_idx + 1] - adj_off[nn_idx])  # (B,)
        max_deg = int(deg.max().item()) if B > 0 else 0

        if max_deg > 0:
            # Gather neighbour indices: (B, max_deg) — pad with self (safe)
            nb_idx = torch.zeros(B, max_deg, dtype=torch.long, device=device)
            for k in range(max_deg):
                slot = adj_off[nn_idx] + k
                valid = k < deg
                slot_clamped = torch.where(valid, slot,
                                           adj_off[nn_idx])   # fallback to row start
                nb_idx[:, k] = torch.where(valid, adj[slot_clamped], nn_idx)

            p_nb  = points[nb_idx]                    # (B, max_deg, 3)
            mu_nb = activated[nb_idx]                 # (B, max_deg)

            diff_nb   = q.unsqueeze(1) - p_nb        # (B, max_deg, 3)
            d_sq_nb   = (diff_nb ** 2).sum(-1)       # (B, max_deg)
            w_nb      = 1.0 / (d_sq_nb + eps_sq.unsqueeze(1))  # (B, max_deg)

            if use_bilateral:
                dmu = mu_nb - mu_ref.unsqueeze(1)    # (B, max_deg)
                w_nb = w_nb * torch.exp(-dmu ** 2 / sigma_v_sq)

            # Mask padding (where k >= deg)
            valid_mask = torch.arange(max_deg, device=device).unsqueeze(0) < deg.unsqueeze(1)
            w_nb = w_nb * valid_mask.float()
            mu_nb_contrib = (w_nb * mu_nb).sum(1)
            w_nb_sum      = w_nb.sum(1)

            w_sum = w_sum + w_nb_sum
            mu_w  = mu_w + mu_nb_contrib

        result[start:end] = torch.clamp(mu_w / w_sum.clamp(min=1e-7), min=0.0)

    return torch.nan_to_num(result).reshape(original_shape).cpu().numpy()


def _nn_sibson_batch(query, field, activated, k_samples=64, sample_radius_scale=1.5,
                     sigma_v=None, global_max_k=None):
    """Discrete Sibson natural-neighbor query for a batch of query points.

    For each query Q, samples K points in a ball around Q.  A sample P belongs
    to the "stolen" region of candidate i when Q is closer to P than any existing
    candidate.  The fraction of stolen samples per candidate gives the Sibson weight.
    """
    device = query.device
    B = query.shape[0]
    adj = field["adjacency"].long()
    adj_off = field["adjacency_offsets"].long()
    points = field["points"]
    cell_radius = field.get("cell_radius")

    if global_max_k is None:
        global_max_k = int((adj_off[1:] - adj_off[:-1]).max().item())

    nn_idx = radfoam.nn(points, field["aabb_tree"], query).long()

    counts = adj_off[nn_idx + 1] - adj_off[nn_idx]
    offsets = adj_off[nn_idx]
    M = global_max_k + 1

    pad_idx = torch.zeros(B, M, dtype=torch.long, device=device)
    valid = torch.zeros(B, M, dtype=torch.bool, device=device)
    pad_idx[:, 0] = nn_idx
    valid[:, 0] = True

    k_range = torch.arange(global_max_k, device=device)
    has_k = counts.unsqueeze(1) > k_range.unsqueeze(0)
    flat_offsets = (offsets.unsqueeze(1) + k_range.unsqueeze(0)).clamp(max=adj.shape[0] - 1)
    pad_idx[:, 1:] = adj[flat_offsets]
    valid[:, 1:] = has_k

    if cell_radius is not None:
        r = sample_radius_scale * cell_radius[nn_idx]  # (B,)
    else:
        cand_pts = points[pad_idx]
        d2 = (query.unsqueeze(1) - cand_pts).pow(2).sum(-1)
        d2[~valid] = 0.0
        r = sample_radius_scale * d2.max(dim=1).values.sqrt()

    K = k_samples
    direction = torch.randn(B, K, 3, device=device)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    u = torch.rand(B, K, device=device).pow(1.0 / 3.0)
    samples = query.unsqueeze(1) + direction * (u * r.unsqueeze(1)).unsqueeze(-1)  # (B, K, 3)

    cand_centers = points[pad_idx]  # (B, M, 3)
    # Find nearest candidate for each sample — serial over M to avoid (B,K,M,3) tensor
    nearest_d2 = torch.full((B, K), float("inf"), device=device)
    nearest_slot = torch.zeros(B, K, dtype=torch.long, device=device)
    for m in range(M):
        d2_m = (samples - cand_centers[:, m:m+1, :]).pow(2).sum(-1)  # (B, K)
        d2_m = d2_m + (~valid[:, m]).float().unsqueeze(1) * 1e10
        closer = d2_m < nearest_d2
        nearest_d2 = torch.where(closer, d2_m, nearest_d2)
        nearest_slot = torch.where(closer, torch.full_like(nearest_slot, m), nearest_slot)

    dist_to_q_sq = (samples - query.unsqueeze(1)).pow(2).sum(-1)  # (B, K)
    stolen = dist_to_q_sq < nearest_d2  # True when Q owns the sample

    stolen_counts = torch.zeros(B, M, device=device)
    stolen_counts.scatter_add_(1, nearest_slot, stolen.float())
    stolen_counts = stolen_counts * valid.float()

    total = stolen_counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    w = stolen_counts / total  # (B, M) Sibson weights

    # Fallback to containing cell when no samples are stolen
    no_stolen = stolen.sum(dim=1) == 0  # (B,)
    if no_stolen.any():
        w[no_stolen] = 0.0
        w[no_stolen, 0] = 1.0

    vals = activated[pad_idx]  # (B, M)
    if sigma_v is not None:
        ref_val = activated[nn_idx]
        val_diff = vals - ref_val.unsqueeze(1)
        w = w * torch.exp(-val_diff.pow(2) / (sigma_v * sigma_v)) * valid.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-10)

    result = (w * vals * valid.float()).sum(dim=1)
    return torch.nan_to_num(result), nn_idx


def sample_nn_sibson(field, coordinates, k_samples=64, sample_radius_scale=1.5, sigma_v=None):
    """Discrete Sibson natural-neighbor interpolation over Voronoi cells.

    Weights are determined by the fraction of randomly-sampled space that each
    neighbor would lose to Q if Q were inserted into the Voronoi diagram — the
    geometric definition of Sibson's natural-neighbor weights.  No spatial
    bandwidth parameter is needed; the support set is the same as IDW (containing
    cell + its Delaunay neighbors).

    Args:
        field: dict from load_density_field() or field_from_model()
        coordinates: numpy or torch array of shape (..., 3)
        k_samples: Monte Carlo samples per query point (more → lower variance)
        sample_radius_scale: ball radius as multiple of cell_radius[containing cell]
        sigma_v: optional bilateral value-similarity scale (None=disabled)

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
    # Smaller batch than IDW: M serial passes over (B, K) tensors
    batch_size = 4096

    for start in range(0, coords_flat.shape[0], batch_size):
        end = min(start + batch_size, coords_flat.shape[0])
        res, _ = _nn_sibson_batch(
            coords_flat[start:end], field, activated,
            k_samples=k_samples, sample_radius_scale=sample_radius_scale,
            sigma_v=sigma_v, global_max_k=global_max_k,
        )
        result[start:end] = res

    return result.reshape(original_shape).cpu().numpy()


def sample_tet_barycentric(field, coordinates, sigma_v_intra=None, use_smoothed=False):
    """C⁰ barycentric-linear density via Delaunay tet walk.

    For each query point, walks the Delaunay tet adjacency graph from the
    nearest cell site until the containing tet is found, then returns the
    barycentric-linear blend of the 4 vertex densities (softplus-activated).
    Continuous everywhere by construction — zero parameters.

    Requires tet topology keys in field (built automatically by
    load_density_field / field_from_model): tets, tet_adjacency, vert_to_tet,
    tri_perm, tri_inv_perm.

    Points outside the convex hull of the foam return 0.

    Args:
        field: dict from load_density_field() or field_from_model()
        coordinates: numpy or torch array of shape (..., 3)
        sigma_v_intra: if set, bilateral reweight the 4 barycentric coords by
            exp(-(μ_k − μ_ref)² / σ²) where μ_ref is the unweighted blend.
            C⁰ is preserved: the absent vertex (λ=0) cancels in numerator and
            denominator, and μ_ref is itself C⁰.
        use_smoothed: if True, use field["density_flat_smooth"] (pre-activated,
            produced by smooth_density_graph) instead of density_flat.

    Returns:
        numpy array of shape (...) with interpolated density values
    """
    original_shape = coordinates.shape[:-1]
    if isinstance(coordinates, np.ndarray):
        coordinates = torch.from_numpy(coordinates).float()
    q = coordinates.reshape(-1, 3).to(field["device"])
    N = q.shape[0]

    points = field["points"]
    tets = field["tets"]            # (T, 4) int64 Triangulation vertex indices
    tet_adj = field["tet_adjacency"]  # (T, 4) int64; UINT32_MAX=4294967295 → boundary
    tri_perm = field["tri_perm"]      # (P,) int64; perm[new_idx] = original_idx
    tri_inv_perm = field["tri_inv_perm"]  # (P,) int64; inv[orig] = new_idx
    vert_to_tet = field["vert_to_tet"]    # (P,) int64

    if use_smoothed and "density_flat_smooth" in field:
        dens = field["density_flat_smooth"]  # already activated
    else:
        dens = F.softplus(field["density_flat"], beta=10)

    BOUNDARY = (1 << 32) - 1  # uint32 max stored as int64

    # Seed: nearest point in original space → Triangulation vertex → starting tet
    nn_old = radfoam.nn(points, field["aabb_tree"], q).long()
    nn_new = tri_inv_perm[nn_old]
    cur_tet = vert_to_tet[nn_new].clone()

    lam_result = torch.zeros(N, 4, device=q.device)
    final_tet = cur_tet.clone()
    done = torch.zeros(N, dtype=torch.bool, device=q.device)

    for _ in range(32):
        active_mask = ~done
        if not active_mask.any():
            break

        idx = active_mask.nonzero(as_tuple=True)[0]  # (A,) active query indices
        qa = q[idx]         # (A, 3)
        ta = cur_tet[idx]   # (A,) tet indices

        # Vertex positions via double index: Triangulation idx → original idx → xyz
        vi_new = tets[ta]              # (A, 4) Triangulation vertex indices
        vi_old = tri_perm[vi_new]      # (A, 4) original point indices
        v = points[vi_old]             # (A, 4, 3)

        # Barycentric: solve [v1-v0 | v2-v0 | v3-v0] @ [λ1;λ2;λ3] = q-v0
        v0 = v[:, 0]
        M = torch.stack([v[:, 1] - v0, v[:, 2] - v0, v[:, 3] - v0], dim=2)  # (A,3,3)
        rhs = (qa - v0).unsqueeze(2)      # (A, 3, 1)
        lam_123 = torch.linalg.solve(M, rhs).squeeze(2)  # (A, 3)
        lam0 = 1.0 - lam_123.sum(-1, keepdim=True)
        lam = torch.cat([lam0, lam_123], dim=-1)  # (A, 4)

        # Mark degenerate tets (singular matrix → NaN) as boundary
        bad = ~torch.isfinite(lam).all(-1)  # (A,)
        done[idx[bad]] = True

        # Contained: all λ ≥ -eps (small slack for floating point at faces)
        contained = (~bad) & (lam.min(-1).values >= -1e-6)

        done[idx[contained]] = True
        lam_result[idx[contained]] = lam[contained]
        final_tet[idx[contained]] = ta[contained]

        # Step remaining active queries to the neighbor tet via the most-negative face
        moving = ~contained & ~bad
        if not moving.any():
            continue

        idx_m = idx[moving]
        ta_m = ta[moving]
        lam_m = lam[moving]

        # Face slot with most-negative λ is the face to cross
        k_neg = lam_m.argmin(-1)                 # (M,)
        packed = tet_adj[ta_m, k_neg]             # (M,) packed adjacency entry

        at_bnd = packed == BOUNDARY
        done[idx_m[at_bnd]] = True               # outside convex hull → density 0

        go = ~at_bnd
        if go.any():
            cur_tet[idx_m[go]] = packed[go] >> 2  # decode neighbor tet index

    # Evaluate: linear blend of 4 vertex densities using converged λ
    vi_new = tets[final_tet]          # (N, 4) Triangulation vertex indices
    vi_old = tri_perm[vi_new]         # (N, 4) original point indices
    mu_verts = dens[vi_old]           # (N, 4)

    if sigma_v_intra is not None:
        mu_ref = (lam_result * mu_verts).sum(-1, keepdim=True)  # (N, 1) plain blend
        b = torch.exp(-(mu_verts - mu_ref).pow(2) / (sigma_v_intra * sigma_v_intra + 1e-12))
        lam_w = lam_result * b
        lam_w = lam_w / lam_w.sum(-1, keepdim=True).clamp(min=1e-10)
        result = (lam_w * mu_verts).sum(-1)
    else:
        result = (lam_result * mu_verts).sum(-1)  # (N,)

    return result.reshape(original_shape).cpu().numpy()


def smooth_density_graph(field, alpha=1.0, sigma_v=0.3, sigma_s_scale=2.0, T=3):
    """Bilateral Jacobi smoothing of per-cell densities on the Delaunay graph.

    Runs T Jacobi iterations with spatial + value-bilateral weights:
        μ̃_i ← (μ_i + α · Σ_j w_ij μ̃_j) / (1 + α · Σ_j w_ij)
        w_ij = exp(−d_ij²/(σ_s_scale·r_i)²) · exp(−(μ_i−μ_j)²/σ_v²)

    Uses the CSR Delaunay adjacency (field["adjacency"] / ["adjacency_offsets"]).
    Spatial sigma is per-cell-adaptive: σ_s_i = σ_s_scale × cell_radius[i].

    The result is cached on the field dict under "density_flat_smooth" together with
    the cache key so that subsequent calls with the same params return immediately.

    Returns the smoothed, already-activated density tensor (N,).
    """
    cache_key = (alpha, sigma_v, sigma_s_scale, T)
    if field.get("_smooth_cache_key") == cache_key and "density_flat_smooth" in field:
        return field["density_flat_smooth"]

    device = field["device"]
    adj = field["adjacency"].long()
    adj_off = field["adjacency_offsets"].long()
    points = field["points"]
    cell_radius = field["cell_radius"]
    N = points.shape[0]

    degrees = adj_off[1:] - adj_off[:-1]                            # (N,)
    src = torch.arange(N, device=device).repeat_interleave(degrees)  # (E,) row index
    dst = adj                                                         # (E,) col index

    d2 = (points[src] - points[dst]).pow(2).sum(-1)                 # (E,)
    sigma_s_sq = (sigma_s_scale * cell_radius[src]).pow(2).clamp(min=1e-12)
    w_spatial = torch.exp(-d2 / sigma_s_sq)                          # (E,) static part

    mu = F.softplus(field["density_flat"], beta=10).clone()

    for _ in range(T):
        mu_diff_sq = (mu[src] - mu[dst]).pow(2)
        w = w_spatial * torch.exp(-mu_diff_sq / (sigma_v * sigma_v + 1e-12))

        w_sum = torch.zeros(N, device=device).scatter_add_(0, src, w)
        w_mu = torch.zeros(N, device=device).scatter_add_(0, src, w * mu[dst])

        mu = (mu + alpha * w_mu) / (1.0 + alpha * w_sum + 1e-12)

    field["density_flat_smooth"] = mu
    field["_smooth_cache_key"] = cache_key
    return mu


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
        import os, glob
        from torchvision.io import read_image, ImageReadMode
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
            slices = [read_image(f, mode=ImageReadMode.GRAY).squeeze().numpy().astype(np.float32) / 255.0
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


def voxelize_volumes(field, resolution, extent, sigma, sigma_v, hop=1):
    """Voxelize the field into two 3D volumes in one pass.

    Raw volume: split-aware when the field carries active thin-surface state
    (see load_density_field/field_from_model) — each voxel takes its owning
    cell's split-resolved density via split_cell_query, the same evaluation
    query_density() uses for the live slices_interleaved panel. Falls back to
    softplus(density[nearest_cell]) — constant per Voronoi cell — otherwise.
    IDW volume: Gaussian bilateral natural-neighbor interpolation matching
    the CUDA tracing kernel (exp(-d²/σ²) spatial, exp(-Δμ²/σ_v²) bilateral)
    over the flat per-cell density. NOT split-aware: blending would require
    resolving each blended neighbor's own split side before combining, which
    this helper does not attempt.

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

    thin_config = _thin_surface_query_config(field)
    if thin_config is not None:
        from split_voxelize import split_cell_query
        density_mode, activation_scale, delta_max_frac = thin_config

    for start in range(0, num_voxels, batch_size):
        end = min(start + batch_size, num_voxels)
        query = coords_flat[start:end]
        res = _idw_query(query, field, activated,
                         sigma, sigma_v, global_max_k, hop=hop)
        if thin_config is not None:
            if density_mode == "independent":
                value, _, _ = split_cell_query(
                    query, field["points"], res.nn_idx, field["density_flat"],
                    None, field["quaternions"], field["texel_sites_2d"],
                    field["texel_heights"], field["cell_radius"],
                    thin_temp=10.0, activation_scale=activation_scale,
                    blend_eps=0.0, density_mode="independent",
                    raw_plus=field["raw_plus"], raw_minus=field["raw_minus"],
                    delta_max_frac=delta_max_frac,
                )
            else:
                value, _, _ = split_cell_query(
                    query, field["points"], res.nn_idx, field["density_flat"],
                    field["density_delta"], field["quaternions"],
                    field["texel_sites_2d"], field["texel_heights"],
                    field["cell_radius"], thin_temp=10.0,
                    activation_scale=activation_scale, blend_eps=0.0,
                    density_mode=density_mode, delta_max_frac=delta_max_frac,
                )
            raw_vol[start:end] = torch.nan_to_num(value)
        else:
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


def select_gt_sobel_anchors(gt_volume, count=6, seed=42,
                            center_fraction=0.6, min_separation=0.16):
    """Select fixed high-gradient anchors from a 3-D GT volume.

    Returned coordinates are in the normalized ``(x, y, z)`` convention used
    by ``grid_sample`` (each component is in [-1, 1]).  Sobel score is the
    primary ordering; ``seed`` only supplies a deterministic tie-break order.
    If the separation constraint prevents selecting ``count`` anchors, the
    remaining highest-scoring central candidates are used without separation.
    Invalid inputs produce an empty ``(0, 3)`` array.
    """
    empty = np.empty((0, 3), dtype=np.float32)
    if (not isinstance(count, (int, np.integer)) or isinstance(count, bool)
            or count <= 0):
        return empty
    try:
        center_fraction = float(center_fraction)
        min_separation = float(min_separation)
    except (TypeError, ValueError, OverflowError):
        return empty
    if (not np.isfinite(center_fraction) or center_fraction <= 0
            or center_fraction > 1 or not np.isfinite(min_separation)
            or min_separation < 0):
        return empty

    try:
        if isinstance(gt_volume, torch.Tensor):
            if (gt_volume.ndim != 3 or gt_volume.numel() == 0
                    or gt_volume.is_complex()):
                return empty
            volume = gt_volume.detach().to(device="cpu", dtype=torch.float32)
        else:
            volume_np = np.array(gt_volume, dtype=np.float32, order="C",
                                 copy=True)
            if volume_np.ndim != 3 or volume_np.size == 0:
                return empty
            volume = torch.from_numpy(volume_np)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return empty
    if not torch.isfinite(volume).all():
        return empty

    # Separable 3-D Sobel derivatives.  The input's logical dimension order is
    # already (x, y, z); only the magnitude matters here, so no image-layout
    # permutation is needed.
    source = F.pad(volume[None, None], (1, 1, 1, 1, 1, 1),
                   mode="replicate")
    smooth = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
    diff = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    kernels = (
        diff[:, None, None] * smooth[None, :, None] * smooth[None, None, :],
        smooth[:, None, None] * diff[None, :, None] * smooth[None, None, :],
        smooth[:, None, None] * smooth[None, :, None] * diff[None, None, :],
    )
    derivatives = [F.conv3d(source, k.reshape(1, 1, 3, 3, 3))
                   for k in kernels]
    magnitude = torch.sqrt(sum(component.square()
                               for component in derivatives))[0, 0].numpy()

    # Use the same centered fraction on all three normalized axes.  Index
    # bounds (rather than a coordinate threshold) keep tiny volumes usable.
    slices = []
    normalized_axes = []
    for size in volume.shape:
        start = int(np.floor(0.5 * (1.0 - center_fraction) * size))
        stop = int(np.ceil(0.5 * (1.0 + center_fraction) * size))
        start = max(0, min(start, size - 1))
        stop = max(start + 1, min(stop, size))
        slices.append(slice(start, stop))
        indices = np.arange(start, stop, dtype=np.float64)
        if size == 1:
            normalized_axes.append(np.zeros_like(indices))
        else:
            normalized_axes.append(2.0 * indices / (size - 1) - 1.0)

    central_scores = magnitude[tuple(slices)].reshape(-1).astype(np.float64)
    mesh = np.meshgrid(*normalized_axes, indexing="ij")
    candidates = np.stack(mesh, axis=-1).reshape(-1, 3)
    if candidates.shape[0] == 0:
        return empty

    # Score-descending greedy traversal, with seeded ties and flat index as a
    # final stable key.  This remains deterministic even for constant volumes.
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError):
        return empty
    tie_order = rng.random(candidates.shape[0])
    flat_order = np.arange(candidates.shape[0])
    order = np.lexsort((flat_order, tie_order, -central_scores))

    selected = []
    selected_set = set()
    separation_sq = float(min_separation) ** 2
    for candidate_index in order:
        coordinate = candidates[candidate_index]
        if all(np.sum((coordinate - candidates[other]) ** 2) >= separation_sq
               for other in selected):
            selected.append(int(candidate_index))
            selected_set.add(int(candidate_index))
            if len(selected) == count:
                break

    # A small/flat central region may not support the requested packing.
    for candidate_index in order:
        candidate_index = int(candidate_index)
        if len(selected) == count:
            break
        if candidate_index not in selected_set:
            selected.append(candidate_index)
            selected_set.add(candidate_index)

    return candidates[selected].astype(np.float32, copy=False)


def log_thin_surface_zoom_panels(
        model, gt_volume, writer, step, anchors, resolution=192,
        extent_scale=2.2, tag="thin_surface_zoom"):
    """Log learned/GT oblique zooms centered on fixed GT-space anchors.

    At every call each normalized anchor is mapped to its current nearest cell;
    that owner supplies the radius and orientation while the anchor remains the
    plane center.  Unsupported or incomplete state is skipped silently.
    """
    # Keep this helper safe to call unconditionally from train.py.  In
    # particular, independent-side checkpoints are intentionally not handled.
    if not getattr(model, "_thin_surface_active", False):
        return
    density_mode = getattr(model, "_thin_surface_density_mode", None)
    if density_mode not in ("absolute", "relative"):
        return
    # The bool is the legacy renderer discriminator; honor it if an older live
    # model labels the same state as absolute.
    relative_delta = bool(getattr(model, "_thin_surface_relative_delta", False))
    if relative_delta:
        density_mode = "relative"

    required = (
        "primal_points", "density", "density_delta", "quaternions",
        "texel_sites_2d", "texel_heights", "point_adjacency",
        "point_adjacency_offsets", "aabb_tree",
    )
    if any(getattr(model, name, None) is None for name in required):
        return
    if writer is None or gt_volume is None:
        return
    if (not isinstance(resolution, (int, np.integer)) or isinstance(resolution, bool)
            or resolution < 2 or not np.isfinite(extent_scale)
            or extent_scale <= 0):
        return

    points = model.primal_points.detach()
    density = model.density.detach()
    density_delta = model.density_delta.detach()
    quaternions = model.quaternions.detach()
    texel_sites_2d = model.texel_sites_2d.detach()
    texel_heights = model.texel_heights.detach()
    adjacency = model.point_adjacency.detach()
    adjacency_offsets = model.point_adjacency_offsets.detach()
    n_cells = points.shape[0] if points.ndim == 2 else 0

    # Validate all row-indexed state before invoking RadFoam.  A partially
    # initialized thin-surface model should remain a clean no-op.
    if (points.shape != (n_cells, 3)
            or density.ndim not in (1, 2) or density.shape[0] != n_cells
            or density.numel() != n_cells
            or density_delta.ndim not in (1, 2)
            or density_delta.shape[0] != n_cells
            or density_delta.numel() != n_cells
            or quaternions.shape != (n_cells, 4)
            or texel_sites_2d.ndim != 3
            or texel_sites_2d.shape[0] != n_cells
            or texel_sites_2d.shape[-1] != 2
            or texel_heights.shape != texel_sites_2d.shape[:2]
            or adjacency.ndim != 1
            or adjacency_offsets.shape != (n_cells + 1,)):
        return

    selected_anchors = []
    try:
        anchor_candidates = iter(anchors)
    except TypeError:
        return
    for value in anchor_candidates:
        try:
            if isinstance(value, torch.Tensor):
                coordinate = value.detach().to(device="cpu", dtype=torch.float64).numpy()
            else:
                coordinate = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, RuntimeError, OverflowError):
            continue
        if (coordinate.shape != (3,) or not np.isfinite(coordinate).all()
                or np.any(coordinate < -1.0) or np.any(coordinate > 1.0)):
            continue
        selected_anchors.append(coordinate.copy())
    if not selected_anchors:
        return

    if isinstance(gt_volume, torch.Tensor):
        gt_xyz = gt_volume.detach()
        if gt_xyz.ndim != 3 or gt_xyz.numel() == 0 or gt_xyz.is_complex():
            return
        gt_xyz = gt_xyz.to(device=points.device, dtype=torch.float32)
    else:
        try:
            gt_np = np.array(gt_volume, dtype=np.float32, order="C", copy=True)
        except (TypeError, ValueError):
            return
        if gt_np.ndim != 3 or gt_np.size == 0:
            return
        gt_xyz = torch.from_numpy(gt_np).to(device=points.device)
    if not torch.isfinite(gt_xyz).all():
        return

    # NumPy/torch GT is logically (x,y,z); grid_sample's source is (z,y,x),
    # while its grid coordinates remain q=(x,y,z).
    gt_source = gt_xyz.permute(2, 1, 0).contiguous()[None, None]
    gt_p99 = float(torch.quantile(gt_xyz.float().reshape(-1), 0.99).item())
    if not np.isfinite(gt_p99):
        return

    # Safe local imports keep normal vis_foam users independent of the split
    # diagnostics and Matplotlib backend setup.
    from split_voxelize import quat_to_frame, split_cell_query
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pixel_boundaries(labels):
        border = np.zeros(labels.shape, dtype=bool)
        changed = labels[1:, :] != labels[:-1, :]
        border[1:, :] |= changed
        border[:-1, :] |= changed
        changed = labels[:, 1:] != labels[:, :-1]
        border[:, 1:] |= changed
        border[:, :-1] |= changed
        return border

    def draw_panel(ax, image, owner_np, signed_np, focus, vmax, title):
        ax.imshow(image, origin="lower", extent=(-1, 1, -1, 1), cmap="gray",
                  vmin=0.0, vmax=vmax, interpolation="nearest")

        borders = pixel_boundaries(owner_np)
        border_rgba = np.zeros((*borders.shape, 4), dtype=np.float32)
        border_rgba[borders] = (0.7, 0.7, 0.7, 0.8)
        ax.imshow(border_rgba, origin="lower", extent=(-1, 1, -1, 1),
                  interpolation="nearest")

        focus_border = pixel_boundaries((owner_np == focus).astype(np.int8))
        focus_rgba = np.zeros((*focus_border.shape, 4), dtype=np.float32)
        focus_rgba[focus_border] = (1.0, 1.0, 0.0, 1.0)
        ax.imshow(focus_rgba, origin="lower", extent=(-1, 1, -1, 1),
                  interpolation="nearest")

        # Contour each owner's signed field only where that owner is exact.
        # This prevents discontinuities at owner seams from becoming false
        # magenta zero contours.
        coords = np.linspace(-1.0, 1.0, resolution)
        for owner_id in np.unique(owner_np):
            owner_mask = owner_np == owner_id
            values = signed_np[owner_mask]
            if (values.size < 4 or not np.isfinite(values).all()
                    or values.min() >= 0.0 or values.max() <= 0.0):
                continue
            ax.contour(coords, coords,
                       np.ma.masked_where(~owner_mask, signed_np),
                       levels=[0.0], colors=["magenta"], linewidths=0.45)
        ax.plot(0.0, 0.0, marker="+", color="cyan", markersize=7,
                markeredgewidth=1.0, linestyle="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    activation_scale = float(getattr(model, "activation_scale", 1.0))
    delta_max_frac = float(getattr(model, "_thin_surface_delta_max_frac", 0.5))
    if (not np.isfinite(activation_scale) or not np.isfinite(delta_max_frac)):
        return

    with torch.inference_mode():
        _, cell_radius = radfoam.farthest_neighbor(
            points, adjacency.to(torch.int32), adjacency_offsets.to(torch.int32))
        cell_radius = cell_radius.reshape(-1)
        if cell_radius.shape != (n_cells,):
            return

        uv = torch.linspace(-1.0, 1.0, resolution, device=points.device,
                            dtype=points.dtype)
        uu, vv = torch.meshgrid(uv, uv, indexing="xy")
        density_flat = density.reshape(-1)
        delta_flat = density_delta.reshape(-1)
        mu_bar = activation_scale * F.softplus(density_flat, beta=10.0)
        if density_mode == "relative":
            effective_delta = (delta_max_frac * mu_bar
                               * torch.tanh(delta_flat))
        else:
            effective_delta = delta_flat
        mu_plus = torch.clamp(mu_bar + effective_delta, min=0.0)
        mu_minus = torch.clamp(mu_bar - effective_delta, min=0.0)

        for anchor_index, anchor_np in enumerate(selected_anchors):
            anchor = torch.as_tensor(anchor_np, device=points.device,
                                     dtype=points.dtype)
            current_owner = radfoam.nn(
                points, model.aabb_tree, anchor.reshape(1, 3)).long().reshape(-1)
            if (current_owner.numel() != 1 or current_owner[0] < 0
                    or current_owner[0] >= n_cells):
                continue
            owner_cell = int(current_owner[0].item())
            radius = cell_radius[owner_cell]
            if not torch.isfinite(radius) or radius <= 0:
                continue
            normal = quat_to_frame(
                quaternions[owner_cell:owner_cell + 1])[0][0]
            ref = torch.tensor([0.0, 0.0, 1.0], device=points.device,
                               dtype=points.dtype)
            if torch.abs(torch.dot(normal, ref)) > 0.9:
                ref = torch.tensor([0.0, 1.0, 0.0], device=points.device,
                                   dtype=points.dtype)
            tangent = F.normalize(torch.linalg.cross(ref, normal), dim=0)
            extent = extent_scale * radius
            q = anchor + extent * (
                uu.reshape(-1, 1) * tangent + vv.reshape(-1, 1) * normal)

            owner = radfoam.nn(points, model.aabb_tree, q).long()
            learned, _, signed = split_cell_query(
                q, points, owner, density_flat, density_delta, quaternions,
                texel_sites_2d, texel_heights, cell_radius,
                thin_temp=10.0, activation_scale=activation_scale,
                blend_eps=0.0, density_mode=density_mode,
                delta_max_frac=delta_max_frac)
            gt_grid = q.to(dtype=gt_source.dtype).reshape(
                1, 1, resolution, resolution, 3)
            gt_values = F.grid_sample(
                gt_source, gt_grid, mode="bilinear", padding_mode="zeros",
                align_corners=True).reshape(resolution, resolution)

            learned_np = torch.nan_to_num(learned).reshape(
                resolution, resolution).cpu().numpy()
            gt_np = torch.nan_to_num(gt_values).cpu().numpy()
            owner_np = owner.reshape(resolution, resolution).cpu().numpy()
            signed_np = signed.reshape(resolution, resolution).cpu().numpy()
            plus = float(mu_plus[owner_cell].item())
            minus = float(mu_minus[owner_cell].item())
            abs_delta = float(effective_delta[owner_cell].abs().item())
            vmax = 1.02 * max(gt_p99, plus, minus, np.finfo(np.float32).eps)
            anchor_text = ", ".join(f"{component:.3f}" for component in anchor_np)
            diagnostic = f"anchor ({anchor_text})  current owner {owner_cell}"

            fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2))
            draw_panel(axes[0], learned_np, owner_np, signed_np, owner_cell,
                       vmax, f"Learned hard split — {diagnostic}")
            draw_panel(axes[1], gt_np, owner_np, signed_np, owner_cell, vmax,
                       f"GT — {diagnostic}")
            fig.suptitle(
                f"{diagnostic}  mu-/mu+ {minus:.4g}/{plus:.4g}  "
                f"abs delta {abs_delta:.4g}  step {step}\n"
                f"shared density range [0, {vmax:.4g}]",
                fontsize=11)
            fig.tight_layout()
            try:
                writer.add_figure(f"{tag}/anchor_{anchor_index:02d}", fig,
                                  global_step=step)
            finally:
                plt.close(fig)


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


def visualize_cells_vs_gradient(points, gt_volume, extent=1.0, n_bins=32,
                                count_res=64, writer_fn=None):
    """Plot mean cells-per-voxel vs. GT gradient magnitude (binned).

    Confirms the Voronoi representation concentrates cells at structure
    boundaries. Also returns Spearman ρ between per-voxel cell count and
    gradient magnitude.

    The GT gradient is computed at full resolution then max-pooled to
    count_res³, and cells are binned at the same count_res³ grid.
    This avoids the sparse-count problem that arises when the full 256³
    grid has far more voxels than cells (~0.03 cells/voxel).

    Args:
        points:     (N, 3) float32 tensor (any device) — cell positions in
                    world coords [-extent, extent]^3
        gt_volume:  (R, R, R) numpy float32 array
        extent:     half-side of world box (default 1.0)
        n_bins:     number of linearly-spaced bins over the log1p gradient
        count_res:  resolution of the counting grid (default 64); the gradient
                    volume is max-pooled from R to count_res before comparison
        writer_fn:  optional callable(fig) for TensorBoard; if None, returns fig

    Returns:
        dict with "spearman_rho" and "spearman_pval" (always),
        plus figure when writer_fn is None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    R = gt_volume.shape[0]
    device = points.device if hasattr(points, 'device') else torch.device('cpu')

    # ── 1. Gradient magnitude at full res, then max-pool to count_res³ ─────
    vol_t = torch.from_numpy(gt_volume).float().to(device)
    v = vol_t.unsqueeze(0).unsqueeze(0)
    v = F.pad(v, (1, 1, 1, 1, 1, 1), mode='replicate')
    sm = torch.tensor([1, 2, 1], dtype=torch.float32, device=device)
    df = torch.tensor([-1, 0, 1], dtype=torch.float32, device=device)
    kx = (sm[:, None, None] * sm[None, :, None] * df[None, None, :]).reshape(1, 1, 3, 3, 3)
    ky = (sm[:, None, None] * df[None, :, None] * sm[None, None, :]).reshape(1, 1, 3, 3, 3)
    kz = (df[:, None, None] * sm[None, :, None] * sm[None, None, :]).reshape(1, 1, 3, 3, 3)
    mag = torch.sqrt(F.conv3d(v, kx) ** 2 +
                     F.conv3d(v, ky) ** 2 +
                     F.conv3d(v, kz) ** 2)  # (1,1,R,R,R)
    # Max-pool to count_res: captures "any edge in this region?"
    stride = R // count_res
    if stride > 1:
        mag = F.max_pool3d(mag, kernel_size=stride, stride=stride)
    grad_log = torch.log1p(mag).squeeze().cpu().numpy().ravel()  # (count_res³,)

    # ── 2. Voxelize point positions into count_res³ cell-count grid ─────────
    pts = points.detach().float().cpu()
    C = mag.shape[-1]  # actual grid size after pooling (may differ slightly from count_res)
    vox = ((pts + extent) / (2.0 * extent) * C).long().clamp(0, C - 1)  # (N,3)
    flat_idx = vox[:, 0] * C * C + vox[:, 1] * C + vox[:, 2]
    count_grid = torch.zeros(C ** 3, dtype=torch.float32)
    count_grid.scatter_add_(0, flat_idx, torch.ones(flat_idx.shape[0]))
    count_flat = count_grid.numpy()  # (C³,)

    # ── 3. Spearman ρ on per-voxel arrays ──────────────────────────────────
    rho, pval = spearmanr(grad_log, count_flat)

    # ── 4. Bin voxels by log1p gradient, compute mean cells/voxel ± 95% CI ─
    g_max = float(grad_log.max())
    bin_edges = np.linspace(0.0, g_max + 1e-8, n_bins + 1)
    bin_idx = np.clip(np.digitize(grad_log, bin_edges) - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    means = np.full(n_bins, np.nan)
    cis   = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = count_flat[bin_idx == b]
        if len(vals) > 0:
            means[b] = vals.mean()
            cis[b]   = 1.96 * vals.std() / max(np.sqrt(len(vals)), 1)

    # ── 5. Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ~np.isnan(means)
    ax.plot(bin_centers[valid], means[valid], color='steelblue', linewidth=2)
    ax.fill_between(bin_centers[valid],
                    (means - cis)[valid], (means + cis)[valid],
                    alpha=0.25, color='steelblue')
    ax.set_xlabel("GT gradient magnitude  (log1p, max-pooled)")
    ax.set_ylabel(f"Mean cells per voxel  ({C}³ grid)")
    ax.set_title(f"Cell allocation vs. GT gradient  (Spearman ρ = {rho:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    stats = {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "bin_centers": bin_centers,
        "bin_means": means,
        "bin_cis": cis,
        "count_res": C,
    }
    if writer_fn:
        writer_fn(fig)
        plt.close(fig)
        return stats
    return fig, stats
