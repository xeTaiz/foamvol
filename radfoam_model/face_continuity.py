"""Efficient shared-Voronoi-face continuity for thin split-cell surfaces.

The cache builder is GPU-native: tetrahedron circumcenters, Delaunay edge
incidence grouping, convex-hull edge rejection, face polygon ordering, and
fixed quadrature sampling are all batched torch operations.  Geometry is
strictly detached; the runtime loss remains differentiable with respect to the
thin-surface density, quaternion, texel-site, and height parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F


_EDGE_TEMPLATE = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACE_TEMPLATE = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_FACE_EDGE_TEMPLATE = ((0, 1), (0, 2), (1, 2))


@dataclass
class VoronoiFaceCache:
    pairs: torch.Tensor       # (F,2), model-row cell IDs
    samples: torch.Tensor     # (F,Q,3), area-stratified interior quadrature
    vertices: torch.Tensor    # (sum_f V_f,3), exact ordered polygon vertices
    vertex_offsets: torch.Tensor  # (F+1,), ragged vertex CSR offsets
    area: torch.Tensor        # (F,)
    scale: torch.Tensor       # (F,), equivalent circular face radius
    build_seconds: float
    num_input_tets: int
    num_finite_tets: int
    num_faces_before_domain_filter: int
    max_vertices: int
    candidate_faces: torch.Tensor | None = None
    candidate_refresh_step: int = -1

    @property
    def num_faces(self) -> int:
        return int(self.pairs.shape[0])


def _encoded_edge(edges: torch.Tensor, n_points: int) -> torch.Tensor:
    edges = edges.sort(dim=-1).values.to(torch.int64)
    return edges[..., 0] * int(n_points) + edges[..., 1]


def _encoded_triangle(faces: torch.Tensor, n_points: int) -> torch.Tensor:
    faces = faces.sort(dim=-1).values.to(torch.int64)
    n = int(n_points)
    return (faces[..., 0] * n + faces[..., 1]) * n + faces[..., 2]


def _decode_edge(keys: torch.Tensor, n_points: int) -> torch.Tensor:
    n = int(n_points)
    return torch.stack((torch.div(keys, n, rounding_mode="floor"), keys % n), dim=-1)


def _circumcenters(tet_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched tetrahedron circumcenters with a finite/nondegenerate mask."""
    p0 = tet_points[:, 0]
    other = tet_points[:, 1:]
    a = 2.0 * (other - p0[:, None, :])
    b = other.square().sum(-1) - p0.square().sum(-1, keepdim=True)
    centers, info = torch.linalg.solve_ex(a, b.unsqueeze(-1), check_errors=False)
    centers = centers.squeeze(-1)
    finite = (info == 0) & torch.isfinite(centers).all(dim=-1)
    return centers, finite


@torch.no_grad()
def build_voronoi_face_cache(
    points: torch.Tensor,
    tetrahedra: torch.Tensor,
    permutation: torch.Tensor,
    num_samples: int = 12,
    domain_extent: float = 1.0,
    max_vertices: int = 32,
    min_area: float = 1e-10,
) -> VoronoiFaceCache:
    """Build finite shared Voronoi faces from the dual Delaunay tetrahedra.

    ``tetrahedra`` index triangulation rows; ``permutation`` maps triangulation
    rows to model parameter rows.  Faces dual to convex-hull Delaunay edges are
    unbounded and are rejected.  Remaining circumcenter polygons are ordered in
    their bisector planes and sampled deterministically.
    """
    if points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point():
        raise ValueError("points must be floating (N,3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (T,4)")
    if permutation.shape != (points.shape[0],):
        raise ValueError("permutation must map every triangulation row")
    if num_samples < 4 or max_vertices < 3:
        raise ValueError("num_samples>=4 and max_vertices>=3 are required")
    if not math.isfinite(domain_extent) or domain_extent <= 0:
        raise ValueError("domain_extent must be finite and positive")

    device = points.device
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        started = time.perf_counter()

    n_points = points.shape[0]
    tets_tri = tetrahedra.long()
    valid_index = ((tets_tri >= 0) & (tets_tri < permutation.numel())).all(dim=-1)
    tets_tri = tets_tri[valid_index]
    tets = permutation.long()[tets_tri]
    distinct = tets.sort(dim=-1).values.diff(dim=-1).ne(0).all(dim=-1)
    tets = tets[distinct]

    centers, finite = _circumcenters(points[tets])
    tets = tets[finite]
    centers = centers[finite]
    if tets.shape[0] == 0:
        raise RuntimeError("no finite nondegenerate Delaunay tetrahedra")

    edge_template = torch.tensor(_EDGE_TEMPLATE, device=device, dtype=torch.long)
    tet_edges = tets[:, edge_template].sort(dim=-1).values               # (T,6,2)
    edge_keys = _encoded_edge(tet_edges, n_points).reshape(-1)
    edge_centers = centers[:, None, :].expand(-1, 6, -1).reshape(-1, 3)
    order = edge_keys.argsort()
    edge_keys = edge_keys[order]
    edge_centers = edge_centers[order]
    unique_edges, counts = torch.unique_consecutive(edge_keys, return_counts=True)
    starts = counts.cumsum(0) - counts

    # A Delaunay edge belonging to any convex-hull triangle has an unbounded
    # dual Voronoi face. Detect hull triangles as tetrahedron faces appearing
    # exactly once, then encode their three edges.
    face_template = torch.tensor(_FACE_TEMPLATE, device=device, dtype=torch.long)
    tet_faces = tets[:, face_template].sort(dim=-1).values.reshape(-1, 3)
    face_keys = _encoded_triangle(tet_faces, n_points)
    face_order = face_keys.argsort()
    sorted_face_keys = face_keys[face_order]
    sorted_faces = tet_faces[face_order]
    _, face_counts = torch.unique_consecutive(sorted_face_keys, return_counts=True)
    face_starts = face_counts.cumsum(0) - face_counts
    boundary_faces = sorted_faces[face_starts[face_counts == 1]]
    face_edge_template = torch.tensor(
        _FACE_EDGE_TEMPLATE, device=device, dtype=torch.long)
    boundary_edge_keys = torch.unique(_encoded_edge(
        boundary_faces[:, face_edge_template], n_points).reshape(-1), sorted=True)
    boundary_pos = torch.searchsorted(boundary_edge_keys, unique_edges)
    boundary_pos = boundary_pos.clamp(max=max(boundary_edge_keys.numel() - 1, 0))
    if boundary_edge_keys.numel():
        is_boundary = boundary_edge_keys[boundary_pos] == unique_edges
    else:
        is_boundary = torch.zeros_like(unique_edges, dtype=torch.bool)

    eligible_group = ((counts >= 3) & (counts <= max_vertices) & ~is_boundary)
    eligible_old_ids = torch.nonzero(eligible_group, as_tuple=False).squeeze(-1)
    if eligible_old_ids.numel() == 0:
        raise RuntimeError("no finite bounded shared Voronoi faces")

    # Convert the sorted ragged circumcenter groups to a temporary padded tensor.
    group_ids = torch.repeat_interleave(
        torch.arange(unique_edges.numel(), device=device), counts)
    local_ids = torch.arange(edge_keys.numel(), device=device) - torch.repeat_interleave(
        starts, counts)
    old_to_new = torch.full(
        (unique_edges.numel(),), -1, device=device, dtype=torch.long)
    old_to_new[eligible_old_ids] = torch.arange(
        eligible_old_ids.numel(), device=device)
    record_keep = eligible_group[group_ids]
    new_groups = old_to_new[group_ids[record_keep]]
    local_kept = local_ids[record_keep]
    padding_width = int(counts[eligible_old_ids].max().item())
    vertices = torch.zeros(
        (eligible_old_ids.numel(), padding_width, 3),
        device=device, dtype=points.dtype)
    vertices[new_groups, local_kept] = edge_centers[record_keep]
    vertex_counts = counts[eligible_old_ids].long()
    vertex_mask = (torch.arange(padding_width, device=device)[None, :]
                   < vertex_counts[:, None])

    kept_keys = unique_edges[eligible_old_ids]
    pairs = _decode_edge(kept_keys, n_points).long()
    centroid = (vertices * vertex_mask[..., None]).sum(dim=1) / vertex_counts[:, None]
    edge_direction = F.normalize(points[pairs[:, 1]] - points[pairs[:, 0]], dim=-1)
    ref_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=points.dtype).expand_as(
        edge_direction).clone()
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=points.dtype)
    use_y = (edge_direction * ref_z).sum(-1).abs() > 0.9
    ref_z[use_y] = ref_y
    basis_u = F.normalize(torch.linalg.cross(ref_z, edge_direction), dim=-1)
    basis_v = torch.linalg.cross(edge_direction, basis_u)
    centered = vertices - centroid[:, None, :]
    angle = torch.atan2(
        (centered * basis_v[:, None, :]).sum(-1),
        (centered * basis_u[:, None, :]).sum(-1))
    angle = angle.masked_fill(~vertex_mask, float("inf"))
    polygon_order = angle.argsort(dim=-1)
    ordered = vertices.gather(1, polygon_order[..., None].expand(-1, -1, 3))

    row = torch.arange(ordered.shape[0], device=device)[:, None]
    valid_positions = torch.arange(padding_width, device=device)[None, :] < vertex_counts[:, None]
    next_positions = ((torch.arange(padding_width, device=device)[None, :] + 1)
                      % vertex_counts[:, None])
    next_vertex = ordered[row, next_positions]
    triangle_area = 0.5 * torch.linalg.cross(
        ordered - centroid[:, None, :], next_vertex - centroid[:, None, :]
    ).norm(dim=-1)
    triangle_area = triangle_area.masked_fill(~valid_positions, 0.0)
    area = triangle_area.sum(dim=-1)

    cumulative_area = triangle_area.cumsum(dim=-1).contiguous()
    quantiles = ((torch.arange(num_samples, device=device, dtype=points.dtype) + 0.5)
                 / num_samples)
    targets = (area[:, None] * quantiles[None, :]).contiguous()
    triangle_ids = torch.searchsorted(cumulative_area, targets, right=False)
    triangle_ids = triangle_ids.clamp(max=padding_width - 1)
    tri_a = ordered[row, triangle_ids]
    tri_b = next_vertex[row, triangle_ids]
    # Equal-weight, area-stratified samples. Low-discrepancy barycentric
    # coordinates avoid repeated triangle centroids on large fan triangles.
    bary_u = quantiles.sqrt()
    golden = 0.6180339887498949
    bary_v = torch.frac(
        (torch.arange(num_samples, device=device, dtype=points.dtype) + 0.5)
        * golden)
    samples = ((1.0 - bary_u)[None, :, None] * centroid[:, None, :]
               + (bary_u * (1.0 - bary_v))[None, :, None] * tri_a
               + (bary_u * bary_v)[None, :, None] * tri_b)

    vertex_abs_max = ordered.abs().masked_fill(~valid_positions[..., None], 0.0).amax(
        dim=(1, 2))
    geometric_ok = (torch.isfinite(samples).all(dim=(1, 2))
                    & torch.isfinite(area) & (area > min_area)
                    & (vertex_abs_max <= domain_extent))
    before_domain = int(samples.shape[0])
    pairs = pairs[geometric_ok]
    samples = samples[geometric_ok]
    ordered = ordered[geometric_ok]
    vertex_counts = vertex_counts[geometric_ok]
    valid_positions = valid_positions[geometric_ok]
    area = area[geometric_ok]
    scale = torch.sqrt(area / math.pi).clamp_min(1e-8)
    flat_vertices = ordered[valid_positions]
    vertex_offsets = torch.cat((
        torch.zeros(1, device=device, dtype=torch.long),
        vertex_counts.cumsum(0)))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return VoronoiFaceCache(
        pairs=pairs.contiguous(), samples=samples.contiguous(),
        vertices=flat_vertices.contiguous(),
        vertex_offsets=vertex_offsets.contiguous(),
        area=area.contiguous(), scale=scale.contiguous(),
        build_seconds=elapsed, num_input_tets=int(tetrahedra.shape[0]),
        num_finite_tets=int(tets.shape[0]),
        num_faces_before_domain_filter=before_domain,
        max_vertices=int(vertex_counts.max().item()),
    )


def quaternion_to_frame(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable normalized quaternion frame matching the renderer."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    n = torch.stack((
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y)), dim=-1)
    t = torch.stack((
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x)), dim=-1)
    b = torch.stack((
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y)), dim=-1)
    return n, t, b


def evaluate_surface_field(
    cell_ids: torch.Tensor,
    query: torch.Tensor,
    points: torch.Tensor,
    quaternions: torch.Tensor,
    texel_sites_2d: torch.Tensor,
    texel_heights: torch.Tensor,
    cell_radius: torch.Tensor,
    thin_temp: float = 10.0,
    return_normal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate signed field and analytic local normal for explicit cells.

    Args:
        cell_ids: (B,)
        query: (B,Q,3)
    Returns:
        signed: (B,Q)
        local_normal: (B,Q,3), normalized gradient of the implicit field
    """
    cp = points[cell_ids]
    radius = cell_radius[cell_ids].reshape(-1).clamp_min(1e-12)
    n, t, b = quaternion_to_frame(quaternions[cell_ids])
    rel = query - cp[:, None, :]
    u = (rel * t[:, None, :]).sum(-1) / radius[:, None]
    v = (rel * b[:, None, :]).sum(-1) / radius[:, None]
    sites = texel_sites_2d[cell_ids]
    du = u[:, :, None] - sites[:, None, :, 0]
    dv = v[:, :, None] - sites[:, None, :, 1]
    weights = torch.exp(-float(thin_temp) * (du.square() + dv.square()))
    weight_sum = weights.sum(-1).clamp_min(1e-20)
    heights = texel_heights[cell_ids][:, None, :]
    height = (weights * heights).sum(-1) / weight_sum
    signed = (rel * n[:, None, :]).sum(-1) - radius[:, None] * height
    if not return_normal:
        return signed, None

    # H_u = sum(dw/du * (h_k-H))/sum(w); world-space r cancels with du/dx=1/r.
    centered_height = heights - height[:, :, None]
    dweight_du = -2.0 * float(thin_temp) * du * weights
    dweight_dv = -2.0 * float(thin_temp) * dv * weights
    height_u = (dweight_du * centered_height).sum(-1) / weight_sum
    height_v = (dweight_dv * centered_height).sum(-1) / weight_sum
    gradient = (n[:, None, :] - height_u[:, :, None] * t[:, None, :]
                - height_v[:, :, None] * b[:, None, :])
    local_normal = F.normalize(gradient, dim=-1)
    return signed, local_normal


def _smooth_l1(value: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=beta, reduction="none")


def face_continuity_loss(
    model,
    cache: VoronoiFaceCache,
    step: int,
    batch_size: int = 8192,
    density_scale: float = 1.0,
    abs_contrast_fraction: float = 0.01,
    relative_contrast_threshold: float = 0.10,
    base_density_fraction: float = 0.05,
    crossing_margin_fraction: float = 0.005,
    side_agreement_threshold: float = 0.60,
    normal_dot_threshold: float = 0.0,
    zero_bandwidth: float = 0.20,
    huber_beta: float = 0.05,
    zero_weight: float = 1.0,
    normal_weight: float = 0.25,
    density_weight: float = 0.10,
    seed: int = 42,
    candidate_refresh: int = 50,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stochastic robust shared-face continuity loss for relative/absolute splits."""
    if cache.num_faces == 0:
        raise RuntimeError("empty Voronoi face cache")
    if batch_size <= 0 or density_scale <= 0 or zero_bandwidth <= 0 or huber_beta <= 0:
        raise ValueError("invalid face-continuity scale/batch parameters")
    mode = getattr(model, "_thin_surface_density_mode", "absolute")
    if mode not in ("absolute", "relative"):
        raise ValueError("face continuity currently supports absolute/relative split modes")

    device = model.primal_points.device
    # Contrast is cheap to evaluate for all cells and changes slowly. Refresh a
    # detached candidate-face pool periodically, then spend surface evaluations
    # only on pairs that can pass the meaningful-density gate. This improves
    # eligible samples per millisecond by roughly the inverse meaningful fraction.
    if (candidate_refresh <= 0 or cache.candidate_faces is None
            or int(step) - cache.candidate_refresh_step >= int(candidate_refresh)):
        with torch.no_grad():
            all_density = float(getattr(model, "activation_scale", 1.0)) * F.softplus(
                model.density.reshape(-1), beta=10.0)
            all_raw_delta = model.density_delta.reshape(-1)
            if mode == "relative":
                all_delta = (float(getattr(model, "_thin_surface_delta_max_frac", 0.5))
                             * all_density * torch.tanh(all_raw_delta))
            else:
                all_delta = all_raw_delta
            all_plus = torch.clamp(all_density + all_delta, min=0.0)
            all_minus = torch.clamp(all_density - all_delta, min=0.0)
            all_contrast = (all_plus - all_minus).abs()
            meaningful_cells = (
                (all_contrast >= float(abs_contrast_fraction) * float(density_scale))
                & (all_contrast / all_density.clamp_min(1e-12)
                   >= relative_contrast_threshold)
                & (all_density >= float(base_density_fraction) * float(density_scale)))
            candidate_mask = (meaningful_cells[cache.pairs[:, 0]]
                              & meaningful_cells[cache.pairs[:, 1]])
            cache.candidate_faces = torch.nonzero(
                candidate_mask, as_tuple=False).squeeze(-1)
            cache.candidate_refresh_step = int(step)
    candidates = cache.candidate_faces
    if candidates is None or candidates.numel() == 0:
        zero = (model.quaternions.sum() + model.texel_heights.sum()
                + model.density_delta.sum() + model.density.sum()) * 0.0
        return zero, {"candidate_faces": torch.zeros((), device=device)}

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(step))
    if int(batch_size) >= candidates.numel():
        chosen = candidates
    else:
        positions = torch.randint(
            candidates.numel(), (int(batch_size),), generator=generator, device=device)
        chosen = candidates[positions]
    pairs = cache.pairs[chosen]
    query = cache.samples[chosen]
    face_scale = cache.scale[chosen]
    face_area = cache.area[chosen]
    i, j = pairs[:, 0], pairs[:, 1]

    # Gather only sampled endpoints; avoid O(N) activation work per iteration.
    raw_density_i = model.density.reshape(-1)[i]
    raw_density_j = model.density.reshape(-1)[j]
    activation_scale = float(getattr(model, "activation_scale", 1.0))
    density_i = activation_scale * F.softplus(raw_density_i, beta=10.0)
    density_j = activation_scale * F.softplus(raw_density_j, beta=10.0)
    raw_delta_i = model.density_delta.reshape(-1)[i]
    raw_delta_j = model.density_delta.reshape(-1)[j]
    if mode == "relative":
        rho = float(getattr(model, "_thin_surface_delta_max_frac", 0.5))
        delta_i = rho * density_i * torch.tanh(raw_delta_i)
        delta_j = rho * density_j * torch.tanh(raw_delta_j)
    else:
        delta_i, delta_j = raw_delta_i, raw_delta_j
    plus_i = torch.clamp(density_i + delta_i, min=0.0)
    minus_i = torch.clamp(density_i - delta_i, min=0.0)
    plus_j = torch.clamp(density_j + delta_j, min=0.0)
    minus_j = torch.clamp(density_j - delta_j, min=0.0)
    contrast_i, contrast_j = (plus_i - minus_i).abs(), (plus_j - minus_j).abs()
    relative_i = contrast_i / density_i.clamp_min(1e-12)
    relative_j = contrast_j / density_j.clamp_min(1e-12)
    high_i, high_j = torch.maximum(plus_i, minus_i), torch.maximum(plus_j, minus_j)
    low_i, low_j = torch.minimum(plus_i, minus_i), torch.minimum(plus_j, minus_j)
    orientation_i = torch.where(
        plus_i >= minus_i, torch.ones_like(plus_i), -torch.ones_like(plus_i)).detach()
    orientation_j = torch.where(
        plus_j >= minus_j, torch.ones_like(plus_j), -torch.ones_like(plus_j)).detach()

    signed_i, normals_i = evaluate_surface_field(
        i, query, model.primal_points, model.quaternions,
        model.texel_sites_2d, model.texel_heights, model._cached_cell_radius)
    signed_j, normals_j = evaluate_surface_field(
        j, query, model.primal_points, model.quaternions,
        model.texel_sites_2d, model.texel_heights, model._cached_cell_radius)
    phi_i = orientation_i[:, None] * signed_i
    phi_j = orientation_j[:, None] * signed_j

    with torch.no_grad():
        # Exact polygon-vertex crossing test, using the cache's ragged CSR.
        vertex_counts = cache.vertex_offsets[chosen + 1] - cache.vertex_offsets[chosen]
        local = torch.arange(cache.max_vertices, device=device)[None, :]
        vertex_mask = local < vertex_counts[:, None]
        vertex_index = cache.vertex_offsets[chosen, None] + local
        vertex_index = vertex_index.clamp(max=cache.vertices.shape[0] - 1)
        face_vertices = cache.vertices[vertex_index]
        cross_i_values, _ = evaluate_surface_field(
            i, face_vertices, model.primal_points, model.quaternions,
            model.texel_sites_2d, model.texel_heights, model._cached_cell_radius,
            return_normal=False)
        cross_j_values, _ = evaluate_surface_field(
            j, face_vertices, model.primal_points, model.quaternions,
            model.texel_sites_2d, model.texel_heights, model._cached_cell_radius,
            return_normal=False)
        cross_i_values = (orientation_i[:, None] * cross_i_values).masked_fill(
            ~vertex_mask, float("nan"))
        cross_j_values = (orientation_j[:, None] * cross_j_values).masked_fill(
            ~vertex_mask, float("nan"))
        abs_threshold = float(abs_contrast_fraction) * float(density_scale)
        base_threshold = float(base_density_fraction) * float(density_scale)
        meaningful_i = ((contrast_i >= abs_threshold)
                        & (relative_i >= relative_contrast_threshold)
                        & (density_i >= base_threshold))
        meaningful_j = ((contrast_j >= abs_threshold)
                        & (relative_j >= relative_contrast_threshold)
                        & (density_j >= base_threshold))
        margin = crossing_margin_fraction * face_scale
        crossing_i = ((torch.nan_to_num(cross_i_values, nan=float("inf")).amin(dim=-1) < -margin)
                      & (torch.nan_to_num(cross_i_values, nan=-float("inf")).amax(dim=-1) > margin))
        crossing_j = ((torch.nan_to_num(cross_j_values, nan=float("inf")).amin(dim=-1) < -margin)
                      & (torch.nan_to_num(cross_j_values, nan=-float("inf")).amax(dim=-1) > margin))
        away = ((phi_i.abs() > margin[:, None])
                & (phi_j.abs() > margin[:, None]))
        side_same = ((phi_i * phi_j) > 0) & away
        side_agreement = side_same.sum(-1).float() / away.sum(-1).clamp_min(1)

    near_i = torch.softmax(-phi_i.abs() / (zero_bandwidth * face_scale[:, None]), dim=-1).detach()
    near_j = torch.softmax(-phi_j.abs() / (zero_bandwidth * face_scale[:, None]), dim=-1).detach()
    normal_i = F.normalize((near_i[..., None] * normals_i).sum(dim=1), dim=-1)
    normal_j = F.normalize((near_j[..., None] * normals_j).sum(dim=1), dim=-1)
    high_normal_i = orientation_i[:, None] * normal_i
    high_normal_j = orientation_j[:, None] * normal_j
    normal_dot = (high_normal_i * high_normal_j).sum(-1).clamp(-1.0, 1.0)

    with torch.no_grad():
        eligible = (meaningful_i & meaningful_j & crossing_i & crossing_j
                    & (side_agreement >= side_agreement_threshold)
                    & (normal_dot.detach() > normal_dot_threshold))
        area_weight = (face_area / face_area.mean().clamp_min(1e-12)).clamp(0.25, 4.0)
        pair_weight = eligible.float() * area_weight
        denominator = pair_weight.sum().clamp_min(1.0)

    near_either = torch.exp(
        -torch.minimum(phi_i.abs(), phi_j.abs())
        / (zero_bandwidth * face_scale[:, None])).detach()
    zero_residual = (phi_i - phi_j) / face_scale[:, None]
    zero_pair = (_smooth_l1(zero_residual, huber_beta) * near_either).sum(-1) \
        / near_either.sum(-1).clamp_min(1e-12)
    normal_pair = 1.0 - normal_dot
    density_pair = (
        _smooth_l1((high_i - high_j) / density_scale, huber_beta)
        + _smooth_l1((low_i - low_j) / density_scale, huber_beta))

    zero_loss = (pair_weight * zero_pair).sum() / denominator
    normal_loss = (pair_weight * normal_pair).sum() / denominator
    density_loss = (pair_weight * density_pair).sum() / denominator
    total = (float(zero_weight) * zero_loss
             + float(normal_weight) * normal_loss
             + float(density_weight) * density_loss)

    eligible_float = eligible.float()
    eligible_den = eligible_float.sum().clamp_min(1.0)
    diagnostics = {
        # Keep diagnostics as detached GPU tensors; train.py materializes them
        # only at logging events, avoiding per-step host synchronization.
        "candidate_faces": torch.as_tensor(candidates.numel(), device=device),
        "sampled_faces": torch.as_tensor(pairs.shape[0], device=device),
        "meaningful_pairs": (meaningful_i & meaningful_j).sum().detach(),
        "both_crossing_pairs": (crossing_i & crossing_j).sum().detach(),
        "side_compatible_pairs": (
            side_agreement >= side_agreement_threshold).sum().detach(),
        "eligible_pairs": eligible.sum().detach(),
        "eligible_fraction": eligible_float.mean().detach(),
        "zero_loss": zero_loss.detach(),
        "normal_loss": normal_loss.detach(),
        "density_loss": density_loss.detach(),
        "normal_dot_eligible_mean": (
            normal_dot.detach() * eligible_float).sum() / eligible_den,
        "side_agreement_eligible_mean": (
            side_agreement * eligible_float).sum() / eligible_den,
    }
    return total, diagnostics
