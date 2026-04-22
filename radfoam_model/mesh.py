"""Marching tetrahedra iso-surface extraction from the Voronoi/Delaunay foam.

Extracts an iso-surface at a given activated-density threshold directly from
the Delaunay tet connectivity, with no voxelization step.

Public API:
    marching_tets(points, density, tets, threshold) -> (vertices, faces)
    write_ply(path, vertices, faces)
    surface_metrics_vs_gt_volume(points, density, tets, gt_volume, ...) -> dict
"""

import numpy as np
import torch

# The 6 edges of a tet: pairs (local_i, local_j) with i < j
EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Marching tetrahedra lookup table.
# Row index = 4-bit mask of which vertices are inside (density >= threshold).
# Each row: [t0_e0, t0_e1, t0_e2,  t1_e0, t1_e1, t1_e2]
#   where values are edge indices 0..5 (from EDGE_PAIRS), -1 = no triangle.
# Winding: cases 1..7 have normals pointing toward the "outside" region;
#           cases 8..14 are their bitwise complements with reversed winding.
_TRI_TABLE = [
    [-1, -1, -1,  -1, -1, -1],  # 0:  all outside
    [ 0,  2,  1,  -1, -1, -1],  # 1:  v0 in
    [ 0,  3,  4,  -1, -1, -1],  # 2:  v1 in
    [ 1,  2,  4,   1,  4,  3],  # 3:  v0,v1 in
    [ 1,  3,  5,  -1, -1, -1],  # 4:  v2 in
    [ 0,  2,  5,   0,  5,  3],  # 5:  v0,v2 in
    [ 0,  1,  5,   0,  5,  4],  # 6:  v1,v2 in
    [ 2,  4,  5,  -1, -1, -1],  # 7:  v0,v1,v2 in
    [ 2,  5,  4,  -1, -1, -1],  # 8:  v3 in
    [ 0,  5,  1,   0,  4,  5],  # 9:  v0,v3 in
    [ 0,  5,  2,   0,  3,  5],  # 10: v1,v3 in
    [ 1,  5,  3,  -1, -1, -1],  # 11: v0,v1,v3 in
    [ 1,  4,  2,   1,  3,  4],  # 12: v2,v3 in
    [ 0,  4,  3,  -1, -1, -1],  # 13: v0,v2,v3 in
    [ 0,  1,  2,  -1, -1, -1],  # 14: v1,v2,v3 in
    [-1, -1, -1,  -1, -1, -1],  # 15: all inside
]
TRI_TABLE = torch.tensor(_TRI_TABLE, dtype=torch.long)


def marching_tets(points, density, tets, threshold):
    """Extract an iso-surface using marching tetrahedra on the Delaunay complex.

    Vertices are deduplicated: two triangles sharing a Delaunay edge share
    the same vertex in the output mesh (keyed by sorted global vertex pair).

    Args:
        points:    (N, 3) float32 CUDA — cell positions
        density:   (N,)  float32 CUDA — activated density per cell
        tets:      (T, 4) integer CUDA — tet vertex indices from Triangulation.tets()
        threshold: float — iso-value in activated density space

    Returns:
        vertices: (V, 3) float32 numpy array
        faces:    (F, 3) int64  numpy array  (indices into vertices)
    """
    device = points.device
    N = points.shape[0]
    tets_l = tets.long()

    # Filter degenerate tets (sentinel indices >= N used by some triangulations)
    valid_tets = tets_l.max(dim=1).values < N
    if not valid_tets.all():
        print(f"  Filtering {(~valid_tets).sum().item()} degenerate tets (index >= N)")
        tets_l = tets_l[valid_tets]

    T = tets_l.shape[0]
    tet_d = density[tets_l]   # (T, 4) activated density at tet vertices
    tet_p = points[tets_l]    # (T, 4, 3)

    # Case index: bitmask of which local vertices are inside
    inside = tet_d >= threshold  # (T, 4) bool
    case_idx = (inside[:, 0].long()
                | (inside[:, 1].long() << 1)
                | (inside[:, 2].long() << 2)
                | (inside[:, 3].long() << 3))  # (T,)

    # Global vertex index pairs for each of the 6 local edges
    edge_vi = torch.stack([tets_l[:, i] for i, _ in EDGE_PAIRS], dim=1)  # (T, 6)
    edge_vj = torch.stack([tets_l[:, j] for _, j in EDGE_PAIRS], dim=1)  # (T, 6)

    # Interpolated crossing position for every edge of every tet
    edge_verts = torch.zeros(T, 6, 3, device=device, dtype=torch.float32)
    for e_idx, (i, j) in enumerate(EDGE_PAIRS):
        p_i = tet_p[:, i]   # (T, 3)
        p_j = tet_p[:, j]
        d_i = tet_d[:, i]   # (T,)
        d_j = tet_d[:, j]
        denom = d_j - d_i
        t = torch.where(
            denom.abs() > 1e-12,
            ((threshold - d_i) / denom).clamp(0.0, 1.0),
            torch.zeros_like(denom),
        )
        edge_verts[:, e_idx] = p_i + t.unsqueeze(-1) * (p_j - p_i)

    # Lookup per-tet triangle edge assignments
    table = TRI_TABLE.to(device)
    tri_edges = table[case_idx]  # (T, 6)

    # Collect valid triangles from both slots (up to 2 per tet)
    all_tris_list = []
    for slot in range(2):
        e0 = tri_edges[:, slot * 3]
        valid = e0 >= 0
        if valid.any():
            v_idx = torch.where(valid)[0]
            all_tris_list.append(torch.stack([
                v_idx,
                tri_edges[v_idx, slot * 3],
                tri_edges[v_idx, slot * 3 + 1],
                tri_edges[v_idx, slot * 3 + 2],
            ], dim=1))  # (K, 4): [tet_idx, e0, e1, e2]

    if not all_tris_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    all_tris = torch.cat(all_tris_list, dim=0)  # (F, 4)
    F_count = all_tris.shape[0]

    # Flatten: one row per (triangle, vertex slot)
    t_flat = all_tris[:, 0].repeat_interleave(3)  # (F*3,) tet indices
    e_flat = all_tris[:, 1:].reshape(-1)           # (F*3,) local edge indices

    # Global Delaunay edge key: unique per undirected edge (vi, vj)
    vi_flat = edge_vi[t_flat, e_flat]
    vj_flat = edge_vj[t_flat, e_flat]
    key_flat = torch.minimum(vi_flat, vj_flat) * N + torch.maximum(vi_flat, vj_flat)

    unique_keys, inverse = torch.unique(key_flat, return_inverse=True)
    num_verts = unique_keys.shape[0]

    # Crossing positions (identical for the same Delaunay edge; average for safety)
    pos_flat = edge_verts[t_flat, e_flat]  # (F*3, 3)
    unique_pos = torch.zeros(num_verts, 3, device=device)
    cnt = torch.zeros(num_verts, device=device)
    unique_pos.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), pos_flat)
    cnt.scatter_add_(0, inverse, torch.ones(F_count * 3, device=device))
    unique_pos = unique_pos / cnt.unsqueeze(1)

    faces = inverse.reshape(F_count, 3)  # (F, 3) vertex indices

    return unique_pos.cpu().numpy(), faces.cpu().numpy().astype(np.int64)


def surface_metrics_vs_gt_volume(
    points,
    density,
    tets,
    gt_volume,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    f_thresholds_vox=(1.0, 2.0),
):
    """Direct Voronoi surface metrics vs GT volume, both in world coords [-1, 1]^3.

    For each threshold, extracts the Voronoi iso-surface via marching_tets (no
    voxelization) and the GT iso-surface via marching_cubes on gt_volume (converted
    from voxel to world coords). Computes Chamfer, Hausdorff 95, and F1 distances
    in world units. F1 thresholds are expressed in voxel-equivalent units so results
    are directly comparable to compute_surface_metrics() from train.py.

    Args:
        points:           (N, 3) float32 CUDA, cell positions in world coords
        density:          (N,)   float32 CUDA, activated density per cell
        tets:             (T, 4) uint32  CUDA, Delaunay tet vertex indices
        gt_volume:        (R, R, R) float32 numpy, GT volume covering [-1, 1]^3
        thresholds:       iso-values applied to both Voronoi density and gt_volume
        f_thresholds_vox: F1-score distance thresholds in voxel units

    Returns:
        dict with keys "chamfer", "hausdorff", "hausdorff_95",
                       "f1_1v", "f1_2v" (matching compute_surface_metrics keys)
    """
    from skimage.measure import marching_cubes
    from scipy.spatial import KDTree

    R = gt_volume.shape[0]
    vox_to_world = 2.0 / (R - 1)  # one voxel in world units
    f_thresholds_world = [fv * vox_to_world for fv in f_thresholds_vox]

    per_level = []
    for t in thresholds:
        try:
            verts_g_vox, _, _, _ = marching_cubes(gt_volume, level=t)
        except (ValueError, RuntimeError):
            continue
        if len(verts_g_vox) < 3:
            continue
        verts_g = verts_g_vox * vox_to_world - 1.0  # → world coords

        verts_v, _ = marching_tets(points, density, tets, threshold=t)
        if len(verts_v) < 3:
            continue

        tree_g = KDTree(verts_g)
        tree_v = KDTree(verts_v)
        d_v_to_g, _ = tree_g.query(verts_v)
        d_g_to_v, _ = tree_v.query(verts_g)

        world_to_vox = (R - 1) / 2.0
        chamfer = 0.5 * (float(d_v_to_g.mean()) + float(d_g_to_v.mean())) * world_to_vox
        hausdorff_95 = float(max(np.percentile(d_v_to_g, 95),
                                  np.percentile(d_g_to_v, 95))) * world_to_vox
        hausdorff_max = float(max(d_v_to_g.max(), d_g_to_v.max())) * world_to_vox

        result = {"chamfer": chamfer, "hausdorff": hausdorff_max,
                  "hausdorff_95": hausdorff_95}
        for fw, fv in zip(f_thresholds_world, f_thresholds_vox):
            prec = float((d_v_to_g <= fw).mean())
            rec  = float((d_g_to_v <= fw).mean())
            result[f"f1_{fv:.0f}v"] = 2 * prec * rec / (prec + rec + 1e-8)

        per_level.append(result)

    if not per_level:
        return {"chamfer": float("inf"), "hausdorff": float("inf"),
                "hausdorff_95": float("inf"),
                **{f"f1_{fv:.0f}v": 0.0 for fv in f_thresholds_vox}}

    keys = per_level[0].keys()
    return {k: float(np.mean([m[k] for m in per_level])) for k in keys}


def write_ply(path, vertices, faces):
    """Write a binary little-endian PLY mesh."""
    V = len(vertices)
    F = len(faces)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {V}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {F}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(np.asarray(vertices, dtype=np.float32).tobytes())
        # Each face: 1 byte count (=3) + 3 × int32 vertex indices = 13 bytes
        faces_i32 = np.asarray(faces, dtype="<i4")  # (F, 3) little-endian
        row = np.empty((F, 13), dtype=np.uint8)
        row[:, 0] = 3
        row[:, 1:5] = faces_i32[:, 0:1].view(np.uint8).reshape(F, 4)
        row[:, 5:9] = faces_i32[:, 1:2].view(np.uint8).reshape(F, 4)
        row[:, 9:13] = faces_i32[:, 2:3].view(np.uint8).reshape(F, 4)
        f.write(row.tobytes())
