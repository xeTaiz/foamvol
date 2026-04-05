"""Targeted ray sampling: build targeted batches on GPU each iteration."""

import torch

VIEWS_PER_CELL = 10


def build_targeted_batch(cell_weights, points, cell_radii,
                         train_rays_gpu, train_projections_gpu,
                         beam_type, beam_geom,
                         n_target_rays):
    """Build a single targeted batch entirely on GPU.

    Args:
        cell_weights: (N,) importance weights on GPU
        points: (N, 3) cell positions on GPU
        cell_radii: (N,) cell radii on GPU
        train_rays_gpu: (total_rays, 6) flat training rays on GPU
        train_projections_gpu: (total_rays, C) flat projections on GPU
        beam_type: "parallel" or "cone"
        beam_geom: dict with geometry constants (all on GPU):
            num_angles, det_h, det_w, and beam-specific params
        n_target_rays: number of targeted rays to generate

    Returns (rays, projections) GPU tensors.
    """
    num_angles = beam_geom["num_angles"]
    det_h = beam_geom["det_h"]
    det_w = beam_geom["det_w"]
    total_rays = num_angles * det_h * det_w

    n_cells = max(1, n_target_rays // VIEWS_PER_CELL)
    n_rays = n_cells * VIEWS_PER_CELL

    # Sample cells proportional to importance
    cell_idx = torch.multinomial(cell_weights, n_cells, replacement=True)
    cell_pos = points[cell_idx]
    cell_r = cell_radii[cell_idx]

    # Expand: each cell × VIEWS_PER_CELL random angles
    cell_pos_exp = cell_pos.unsqueeze(1).expand(-1, VIEWS_PER_CELL, -1).reshape(-1, 3)
    cell_r_exp = cell_r.unsqueeze(1).expand(-1, VIEWS_PER_CELL).reshape(-1)
    angle_idx = torch.randint(0, num_angles, (n_rays,), device=cell_weights.device)

    # Jitter within half cell radius
    jitter = (torch.rand(n_rays, 3, device=cell_weights.device) * 2 - 1) * (0.5 * cell_r_exp[:, None])
    jittered_pos = cell_pos_exp + jitter

    if beam_type == "parallel":
        center_rays_all = beam_geom["center_rays"]  # (num_angles, 6) on GPU
        center_ray = center_rays_all[angle_idx]
        beam_dir = center_ray[:, 3:6]
        perp = torch.stack([-beam_dir[:, 1], beam_dir[:, 0],
                            torch.zeros_like(beam_dir[:, 0])], dim=-1)
        u = (jittered_pos * perp).sum(-1)
        v = jittered_pos[:, 2]

        pixel_size = beam_geom["pixel_size"]
        extent = pixel_size * det_w
        iu = ((u / extent + 0.5) * det_w).long().clamp(0, det_w - 1)
        iv = ((v / extent + 0.5) * det_h).long().clamp(0, det_h - 1)

    elif beam_type == "cone":
        c2ws = beam_geom["c2ws"]  # (num_angles, 4, 4) on GPU
        fx = beam_geom["fx"]
        fy = beam_geom["fy"]

        origin = c2ws[angle_idx, :3, 3]
        rot = c2ws[angle_idx, :3, :3]

        d_world = jittered_pos - origin
        d_cam = torch.einsum('nij,nj->ni', rot.transpose(-1, -2), d_world)

        z = d_cam[:, 2].clamp(min=1e-6)
        px = d_cam[:, 0] / z * fx + det_w / 2.0
        py = d_cam[:, 1] / z * fy + det_h / 2.0

        iu = px.long().clamp(0, det_w - 1)
        iv = py.long().clamp(0, det_h - 1)

    else:
        flat_idx = torch.randint(0, total_rays, (n_rays,), device=cell_weights.device)
        return train_rays_gpu[flat_idx], train_projections_gpu[flat_idx]

    flat_idx = (angle_idx * (det_h * det_w) + iv * det_w + iu).clamp(0, total_rays - 1)
    return train_rays_gpu[flat_idx], train_projections_gpu[flat_idx]
