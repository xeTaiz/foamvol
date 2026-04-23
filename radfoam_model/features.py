"""Per-cell feature assignment for the Voronoi foam.

Public API:
    assign_cell_features(points, feature_volume, extent=1.0) -> (N, F) float16 CPU
    save_cell_features(path, features, meta=None)
    load_cell_features(path) -> features, meta
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def assign_cell_features(points, feature_volume, extent=1.0):
    """Trilinear sample feature_volume at each cell's world position.

    Axis convention matches CTScene FDK init: the volume is stored in
    (X, Y, Z) = (D, H, W) order, so world (x,y,z) maps to grid_sample
    coords (z, y, x) after normalizing to [-1, 1].

    Args:
        points:         (N, 3) float32 CUDA — world coords in [-extent, extent]^3
        feature_volume: (R, R, R, F) numpy float16 or float32, or equivalent tensor
        extent:         half-side of world box (default 1.0)

    Returns:
        (N, F) float16 on CPU
    """
    device = points.device

    if isinstance(feature_volume, np.ndarray):
        feat_np = feature_volume
    else:
        feat_np = feature_volume.cpu().numpy()

    R = feat_np.shape[0]
    F_dim = feat_np.shape[-1]

    feat_t = torch.from_numpy(feat_np.astype(np.float32)).to(device)  # (R,R,R,F)
    feat_t = feat_t.permute(3, 0, 1, 2).unsqueeze(0)                  # (1,F,R,R,R) = (1,F,X,Y,Z)

    # Normalize world coords to [-1, 1], then flip (x,y,z)→(z,y,x) for grid_sample
    grid = (points / extent).flip(-1)                # (N,3): z_n, y_n, x_n
    grid = grid.reshape(1, 1, 1, -1, 3)              # (1,1,1,N,3)

    sampled = F.grid_sample(
        feat_t, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (1, F, 1, 1, N)

    cell_feats = sampled.reshape(F_dim, -1).T.half().cpu()  # (N, F) float16
    return cell_feats


def save_cell_features(path, features, meta=None):
    """Save cell features to a .npz file.

    Args:
        path:     output path (e.g. output/<run>/features_dino.npz)
        features: (N, F) numpy or tensor float16
        meta:     optional dict of extra info (backbone, n_cells, …)
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if features.dtype != np.float16:
        features = features.astype(np.float16)
    kwargs = {"features": features}
    if meta:
        for k, v in meta.items():
            kwargs[f"meta_{k}"] = np.array([v] if not hasattr(v, "__len__") else v)
    np.savez_compressed(path, **kwargs)


def load_cell_features(path):
    """Load cell features saved by save_cell_features.

    Returns:
        features: (N, F) float16 numpy
        meta:     dict (may be empty)
    """
    d = np.load(path, allow_pickle=False)
    features = d["features"]
    meta = {k[5:]: v for k, v in d.items() if k.startswith("meta_")}
    return features, meta
