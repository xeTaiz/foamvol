import torch


def bilinear_proj_lookup(proj_nchw, view_idx, px, py):
    """Bilinearly sample projection values at continuous pixel coordinates.

    Args:
        proj_nchw: (N, H, W, 1) float tensor on GPU
        view_idx: (B,) int64 — view index per ray
        px: (B,) float — continuous column coord; 0.0 = left edge of pixel 0
        py: (B,) float — continuous row coord;    0.0 = top  edge of pixel 0

    Returns:
        (B, 1) bilinearly interpolated projection values
    """
    H, W = proj_nchw.shape[1], proj_nchw.shape[2]
    ix_lo = px.long().clamp(0, W - 1)
    iy_lo = py.long().clamp(0, H - 1)
    ix_hi = (ix_lo + 1).clamp(0, W - 1)
    iy_hi = (iy_lo + 1).clamp(0, H - 1)
    fx = (px - ix_lo.float()).unsqueeze(-1)  # (B, 1)
    fy = (py - iy_lo.float()).unsqueeze(-1)
    p00 = proj_nchw[view_idx, iy_lo, ix_lo]  # (B, 1)
    p10 = proj_nchw[view_idx, iy_lo, ix_hi]
    p01 = proj_nchw[view_idx, iy_hi, ix_lo]
    p11 = proj_nchw[view_idx, iy_hi, ix_hi]
    return (1 - fx) * (1 - fy) * p00 + fx * (1 - fy) * p10 + (1 - fx) * fy * p01 + fx * fy * p11
