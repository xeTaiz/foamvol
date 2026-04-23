"""Click-based segmentation via cosine similarity over cell features.

Given a 3D click position (world coordinates) and pre-extracted cell features,
finds the queried cell and ranks all other cells by cosine similarity to it.
Outputs:
  - A matplotlib figure showing axial/coronal/sagittal slices of the GT volume
    with colour-coded cosine similarity overlaid.
  - (Optional) per-cell similarity .npy for downstream use.

Usage
-----
    # Manual clicks (world coords in [-1,1]³)
    micromamba run -n radfoam python demos/click_segment.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_dino_vits8_axial.npz \\
        --click 0.1 -0.3 0.05 \\
        [--topk 0.05] \\
        [--out output/<run>/click_segment.png]

    # Multiple clicks (averaged query)
    micromamba run -n radfoam python demos/click_segment.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_dino_vits8_axial.npz \\
        --click 0.1 -0.3 0.05  0.15 -0.28 0.02

    # Auto-sample clicks from GT labels + compute Dice (CT-ORG)
    micromamba run -n radfoam python demos/click_segment.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_dino_vits8_axial.npz \\
        --gt-labels r2_data/ct_org/CT_ORG_case_00_cone/labels.npy \\
        --sample-class 3 \\        # class 3 = lungs
        --n-clicks 5 \\
        --topk 0.05

    # Evaluate all classes in one pass
    micromamba run -n radfoam python demos/click_segment.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_dino_vits8_axial.npz \\
        --gt-labels r2_data/ct_org/CT_ORG_case_00_cone/labels.npy \\
        --eval-all-classes
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import radfoam
from vis_foam import load_gt_volume, load_density_field
from radfoam_model.features import load_cell_features


def cosine_similarity_batch(query_feat, cell_feats):
    """Cosine similarity between a query vector and all cell features.

    Args:
        query_feat: (F,) float32 tensor
        cell_feats: (N, F) float32 tensor

    Returns:
        (N,) float32 tensor in [-1, 1]
    """
    q = query_feat / (query_feat.norm() + 1e-8)
    c = cell_feats / (cell_feats.norm(dim=1, keepdim=True) + 1e-8)
    return (c @ q)


def find_cells_near_click(click_xyz, points, aabb_tree, k=1):
    """Return indices of the k nearest cells to click_xyz.

    Args:
        click_xyz: (3,) world coordinate tensor
        points:    (N, 3) cell positions
        aabb_tree: prebuilt AABB tree
        k:         number of neighbours

    Returns:
        (k,) long tensor of cell indices
    """
    q = click_xyz.unsqueeze(0)  # (1, 3)
    idx = radfoam.nn(points, aabb_tree, q).long()   # (1,)
    if k == 1:
        return idx
    # For k>1, fall back to torch brute-force (rare use case, small N)
    dists = (points - click_xyz.unsqueeze(0)).norm(dim=1)
    return dists.topk(k, largest=False).indices


def project_sim_to_slices(sim_np, points_np, gt_volume, slice_coords=None):
    """Project per-cell cosine similarity onto GT volume axial/coronal/sagittal slices.

    Voxelizes the similarity values using nearest-cell assignment for each voxel
    slice, returning three 2D arrays.

    Args:
        sim_np:       (N,) float32
        points_np:    (N, 3) float32 world coords
        gt_volume:    (R, R, R) float32 GT volume
        slice_coords: dict with keys "axial", "coronal", "sagittal"
                      each a float in [-1,1] (world coord of slice plane).
                      Defaults to 0.0 for all.

    Returns:
        dict: {"axial": (R,R), "coronal": (R,R), "sagittal": (R,R)} float32
    """
    R = gt_volume.shape[0]
    if slice_coords is None:
        slice_coords = {"axial": 0.0, "coronal": 0.0, "sagittal": 0.0}

    def world_to_vox(w):
        return int(round((w + 1.0) / 2.0 * (R - 1)))

    results = {}
    for name, axis, fixed_w in [
        ("axial",    0, slice_coords.get("axial",    0.0)),
        ("coronal",  1, slice_coords.get("coronal",  0.0)),
        ("sagittal", 2, slice_coords.get("sagittal", 0.0)),
    ]:
        fixed_v = world_to_vox(fixed_w)

        # Select points within ±2 voxels of this slice plane
        coord_v = ((points_np[:, axis] + 1.0) / 2.0 * (R - 1)).round().astype(int)
        mask = np.abs(coord_v - fixed_v) <= 2
        if mask.sum() == 0:
            results[name] = np.zeros((R, R), dtype=np.float32)
            continue

        pts_2d = np.delete(points_np[mask], axis, axis=1)  # (M, 2) world
        sim_2d = sim_np[mask]

        # Voxelize onto (R, R) grid
        grid = np.zeros((R, R), dtype=np.float32)
        cnt  = np.zeros((R, R), dtype=np.int32)
        ui = ((pts_2d[:, 0] + 1.0) / 2.0 * (R - 1)).round().astype(int).clip(0, R - 1)
        vi = ((pts_2d[:, 1] + 1.0) / 2.0 * (R - 1)).round().astype(int).clip(0, R - 1)
        np.add.at(grid, (ui, vi), sim_2d)
        np.add.at(cnt,  (ui, vi), 1)
        cnt = np.maximum(cnt, 1)
        results[name] = grid / cnt

    return results


def make_figure(sim_slices, gt_slices, click_xyz, click_vox, threshold=None):
    """Produce a 3×2 figure: GT slice (left) + similarity overlay (right)."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    cmap_sim = plt.get_cmap("hot")
    cmap_gt  = plt.get_cmap("gray")
    names = ["axial", "coronal", "sagittal"]

    for row, name in enumerate(names):
        gt  = gt_slices[name]
        sim = sim_slices[name]

        axes[row, 0].imshow(gt.T,  cmap=cmap_gt,  origin="lower",
                            vmin=0, vmax=gt.max() or 1)
        axes[row, 0].set_title(f"GT {name}")
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(sim.T, cmap=cmap_sim, origin="lower",
                                 vmin=0, vmax=1, alpha=0.9)
        axes[row, 1].set_title(f"Cosine sim {name}")
        axes[row, 1].axis("off")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Mark click position
        ax_idx = ["axial", "coronal", "sagittal"].index(name)
        other = [i for i in range(3) if i != ax_idx]
        R = gt.shape[0]
        cx = int((click_vox[other[0]] / (R - 1)) * R)
        cy = int((click_vox[other[1]] / (R - 1)) * R)
        for ax_col in range(2):
            axes[row, ax_col].plot(cx, cy, "c+", markersize=12, markeredgewidth=2)

    fig.suptitle(
        f"Click @ world ({click_xyz[0]:.3f}, {click_xyz[1]:.3f}, {click_xyz[2]:.3f})",
        fontsize=11
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CT-ORG label class names (used when --gt-labels is provided)
# ---------------------------------------------------------------------------

CTORG_LABEL_NAMES = {
    0: "background",
    1: "liver",
    2: "bladder",
    3: "lungs",
    4: "kidneys",
    5: "bone",
    6: "brain",
}


def sample_clicks_from_labels(labels_np, class_id, n_clicks, rng=None, extent=1.0):
    """Sample n_clicks voxel positions from the label mask and convert to world coords.

    Volume convention: (X, Y, Z) order matching vol_gt.npy and assign_cell_features.
    World coords: vox_i / (R-1) * 2*extent - extent  (same as grid_sample convention).

    Args:
        labels_np: (R, R, R) int16 label volume
        class_id:  integer label to sample from
        n_clicks:  number of click positions to sample
        rng:       optional numpy random Generator (for reproducibility)
        extent:    world half-extent (default 1.0)

    Returns:
        (n_clicks, 3) float32 world coords in [-extent, extent]³
        or fewer rows if the class has fewer voxels than n_clicks
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mask = labels_np == class_id
    vox_idx = np.argwhere(mask)   # (M, 3) voxel indices (X, Y, Z)
    if len(vox_idx) == 0:
        return np.empty((0, 3), dtype=np.float32)

    chosen = vox_idx[rng.choice(len(vox_idx), size=min(n_clicks, len(vox_idx)),
                                replace=False)]
    R = labels_np.shape[0]
    world = (chosen.astype(np.float32) / (R - 1)) * (2 * extent) - extent
    return world


def voxelize_sim_mask(sim_np, points_np, topk_frac, R, extent=1.0):
    """Convert per-cell similarity to a binary (R,R,R) mask at the topk threshold.

    Args:
        sim_np:    (N,) float32 cosine similarity scores
        points_np: (N, 3) float32 world coords
        topk_frac: float in (0,1] — top fraction to include
        R:         grid resolution
        extent:    world half-extent

    Returns:
        (R, R, R) bool numpy mask
    """
    thresh = float(np.partition(sim_np, -max(1, int(topk_frac * len(sim_np))))
                   [-max(1, int(topk_frac * len(sim_np)))])
    mask_cells = sim_np >= thresh

    grid = np.zeros((R, R, R), dtype=np.bool_)
    vi = ((points_np + extent) / (2 * extent) * (R - 1)).round().astype(int).clip(0, R - 1)
    grid[vi[mask_cells, 0], vi[mask_cells, 1], vi[mask_cells, 2]] = True
    return grid


def dice_binary(pred_mask, ref_mask):
    """Binary Dice between two boolean arrays."""
    inter = (pred_mask & ref_mask).sum()
    denom = pred_mask.sum() + ref_mask.sum()
    return float(2.0 * inter / (denom + 1e-8)) if denom > 0 else 0.0


def run_click_eval(class_id, class_name, labels_np, points, aabb_tree, cell_feats,
                   gt_volume, topk, n_clicks, device, out_prefix, rng):
    """Run click segmentation for one class and optionally compute Dice.

    Returns a dict with keys: class_id, class_name, n_clicks_used, dice_topk,
    sim_max, sim_mean.
    """
    clicks = sample_clicks_from_labels(labels_np, class_id, n_clicks, rng=rng)
    if len(clicks) == 0:
        print(f"  Class {class_id} ({class_name}): not present in labels, skipping.")
        return None

    click_tensors = torch.from_numpy(clicks).to(device)
    hit_indices = []
    for c in click_tensors:
        idx = find_cells_near_click(c, points, aabb_tree, k=1)
        hit_indices.append(idx.item())

    query_feat = cell_feats[hit_indices].mean(0)
    sim = cosine_similarity_batch(query_feat, cell_feats)
    sim_np = sim.cpu().numpy()

    # Dice evaluation
    R = labels_np.shape[0]
    points_np = points.cpu().numpy()
    pred_mask = voxelize_sim_mask(sim_np, points_np, topk, R)
    ref_mask   = labels_np == class_id
    dice = dice_binary(pred_mask, ref_mask)

    print(f"  Class {class_id:2d} ({class_name:<12s})  "
          f"clicks={len(clicks)}  Dice@{topk:.2f}={dice:.3f}  "
          f"sim_max={sim_np.max():.3f}  sim_mean={sim_np.mean():.3f}")

    # Save per-class visualization (only for manually triggered single-class mode)
    if out_prefix is not None:
        thresh_val = float(sim.topk(max(1, int(topk * len(sim_np)))).values[-1].item())
        sim_clipped = np.clip(sim_np, thresh_val, 1.0)
        sim_norm = (sim_clipped - thresh_val) / (1.0 - thresh_val + 1e-8)

        click_mean = clicks.mean(0)
        R_gt = gt_volume.shape[0]

        def world_to_vox_f(w):
            return (w + 1.0) / 2.0 * (R_gt - 1)

        click_vox = world_to_vox_f(click_mean)
        slice_coords = {
            "axial":    float(click_mean[0]),
            "coronal":  float(click_mean[1]),
            "sagittal": float(click_mean[2]),
        }

        def slice_vol(vol, axis, coord):
            idx = int(round((coord + 1.0) / 2.0 * (R_gt - 1)))
            idx = np.clip(idx, 0, R_gt - 1)
            return np.take(vol, idx, axis=axis)

        gt_slices = {
            "axial":    slice_vol(gt_volume, 0, click_mean[0]),
            "coronal":  slice_vol(gt_volume, 1, click_mean[1]),
            "sagittal": slice_vol(gt_volume, 2, click_mean[2]),
        }

        sim_slices = project_sim_to_slices(sim_norm, points_np, gt_volume,
                                           slice_coords=slice_coords)
        fig = make_figure(sim_slices, gt_slices, click_mean, click_vox)

        import matplotlib.pyplot as plt
        png_path = f"{out_prefix}_class{class_id}_{class_name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {png_path}")
        np.save(f"{out_prefix}_class{class_id}_{class_name}_sim.npy", sim_np)

    return {
        "class_id":       class_id,
        "class_name":     class_name,
        "n_clicks_used":  int(len(clicks)),
        "dice":           dice,
        "topk":           topk,
        "sim_max":        float(sim_np.max()),
        "sim_mean":       float(sim_np.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True, help="output/<run>/config.yaml")
    parser.add_argument("--features", required=True, help="output/<run>/features_*.npz")
    parser.add_argument("--click",    nargs="+", type=float, default=None,
                        help="x y z [x y z …] world-coord click(s) (manual mode)")
    parser.add_argument("--topk",     type=float, default=0.1,
                        help="Fraction of top-similarity cells to highlight (default: 0.1)")
    # GT-label options
    parser.add_argument("--gt-labels",      default=None,
                        help="Path to integer label volume (e.g. labels.npy from CT-ORG)")
    parser.add_argument("--sample-class",   type=int, default=None,
                        help="Label class ID to auto-sample clicks from (requires --gt-labels)")
    parser.add_argument("--n-clicks",       type=int, default=5,
                        help="Number of auto-sampled clicks per class (default: 5)")
    parser.add_argument("--eval-all-classes", action="store_true",
                        help="Evaluate all non-background classes in the GT label volume")
    parser.add_argument("--label-names",    default=None,
                        help="Optional JSON/npy file mapping class_id → name "
                             "(default: use CT-ORG names if ≤7 classes)")
    # Output
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--out",      default=None,
                        help="Output PNG path (default: <run_dir>/click_segment.png)")
    parser.add_argument("--out-csv",  default=None,
                        help="Output Dice CSV (default: <run_dir>/click_dice.csv; "
                             "only written in eval mode)")
    args = parser.parse_args()

    # Validate
    if args.click is None and args.sample_class is None and not args.eval_all_classes:
        parser.error("Provide --click OR --sample-class / --eval-all-classes with --gt-labels")
    if args.click is not None and len(args.click) % 3 != 0:
        parser.error("--click requires groups of 3 floats (x y z)")
    if (args.sample_class is not None or args.eval_all_classes) and args.gt_labels is None:
        parser.error("--sample-class and --eval-all-classes require --gt-labels")

    config_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(config_path)
    out_path = args.out or os.path.join(run_dir, "click_segment.png")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_path = cfg.get("data_path", "")
    dataset   = cfg.get("dataset", "r2_gaussian")

    # Load model checkpoint
    model_pt = os.path.join(run_dir, "model.pt")
    print(f"Loading checkpoint: {model_pt}")
    field  = load_density_field(model_pt, device=args.device)
    points = field["points"]
    aabb   = field["aabb_tree"]
    N = points.shape[0]
    print(f"Points: {N:,}")

    # Load cell features
    print(f"Loading features: {args.features}")
    cell_feats_np, meta = load_cell_features(args.features)
    if cell_feats_np.shape[0] != N:
        print(f"WARNING: feature file has {cell_feats_np.shape[0]} cells, "
              f"checkpoint has {N}. They may be mismatched.")
    print(f"Cell features: {cell_feats_np.shape}  backbone={meta.get('backbone', '?')}")

    cell_feats = torch.from_numpy(cell_feats_np.astype(np.float32)).to(args.device)

    # Load GT volume for visualization
    gt_volume = load_gt_volume(data_path, dataset)
    if gt_volume is None:
        print("No GT volume — slices will show empty background.")
        gt_volume = np.zeros((128, 128, 128), dtype=np.float32)

    # Load GT labels if provided
    labels_np = None
    if args.gt_labels:
        print(f"Loading GT labels: {args.gt_labels}")
        labels_np = np.load(args.gt_labels).astype(np.int16)
        print(f"Labels: {labels_np.shape}  classes: {np.unique(labels_np)}")

    # Determine label names
    label_names = dict(CTORG_LABEL_NAMES)
    if labels_np is not None:
        max_cls = int(labels_np.max())
        if max_cls > 6:
            # Not CT-ORG — use generic names for unknown classes
            for c in range(7, max_cls + 1):
                if c not in label_names:
                    label_names[c] = f"class_{c}"
    if args.label_names:
        import json
        with open(args.label_names) as f:
            extra = json.load(f)
        label_names.update({int(k): v for k, v in extra.items()})

    # -----------------------------------------------------------------------
    # Eval-all-classes mode
    # -----------------------------------------------------------------------
    if args.eval_all_classes:
        print(f"\nEvaluating all classes with {args.n_clicks} clicks each …")
        rng = np.random.default_rng(42)
        classes = [int(c) for c in np.unique(labels_np) if c != 0]
        out_csv = args.out_csv or os.path.join(run_dir, "click_dice.csv")
        out_prefix = out_path.replace(".png", "")

        results = []
        for cid in classes:
            cname = label_names.get(cid, f"class_{cid}")
            r = run_click_eval(
                cid, cname, labels_np, points, aabb, cell_feats,
                gt_volume, args.topk, args.n_clicks, args.device,
                out_prefix, rng
            )
            if r:
                results.append(r)

        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "class_name", "n_clicks", "topk", "dice",
                        "sim_max", "sim_mean"])
            for r in results:
                w.writerow([r["class_id"], r["class_name"], r["n_clicks_used"],
                            r["topk"], f"{r['dice']:.4f}",
                            f"{r['sim_max']:.4f}", f"{r['sim_mean']:.4f}"])

        valid = [r["dice"] for r in results if r["dice"] is not None]
        print(f"\nMean click-Dice@{args.topk}: {np.mean(valid):.4f}  "
              f"over {len(valid)} classes")
        print(f"Saved: {out_csv}")
        return

    # -----------------------------------------------------------------------
    # Single-class auto-sample mode
    # -----------------------------------------------------------------------
    if args.sample_class is not None:
        rng = np.random.default_rng(42)
        cname = label_names.get(args.sample_class, f"class_{args.sample_class}")
        out_prefix = out_path.replace(".png", "")
        r = run_click_eval(
            args.sample_class, cname, labels_np, points, aabb, cell_feats,
            gt_volume, args.topk, args.n_clicks, args.device,
            out_prefix, rng
        )
        if r:
            out_csv = args.out_csv or os.path.join(run_dir, "click_dice.csv")
            import csv
            write_header = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["class_id", "class_name", "n_clicks", "topk", "dice",
                                "sim_max", "sim_mean"])
                w.writerow([r["class_id"], r["class_name"], r["n_clicks_used"],
                            r["topk"], f"{r['dice']:.4f}",
                            f"{r['sim_max']:.4f}", f"{r['sim_mean']:.4f}"])
            print(f"Appended to: {out_csv}")
        return

    # -----------------------------------------------------------------------
    # Manual click mode (original behaviour)
    # -----------------------------------------------------------------------
    clicks = np.array(args.click, dtype=np.float32).reshape(-1, 3)

    # Compute query feature = mean of clicked cell features
    click_tensors = torch.from_numpy(clicks).to(args.device)
    hit_indices = []
    for c in click_tensors:
        idx = find_cells_near_click(c, points, aabb, k=1)
        hit_indices.append(idx.item())
    print(f"Hit cells: {hit_indices}")

    query_feat = cell_feats[hit_indices].mean(0)  # (F,)
    sim = cosine_similarity_batch(query_feat, cell_feats)  # (N,)

    # Threshold to top-k fraction
    thresh_val = float(sim.topk(max(1, int(args.topk * N))).values[-1].item())
    sim_np = sim.cpu().numpy()
    sim_clipped = np.clip(sim_np, thresh_val, 1.0)
    sim_norm = (sim_clipped - thresh_val) / (1.0 - thresh_val + 1e-8)
    print(f"Similarity: min={sim_np.min():.3f}  max={sim_np.max():.3f}  "
          f"threshold={thresh_val:.3f}")

    # Load GT volume for slice visualization
    gt_volume = load_gt_volume(data_path, dataset)
    if gt_volume is None:
        print("No GT volume — slices will show empty background.")
        gt_volume = np.zeros((128, 128, 128), dtype=np.float32)

    R = gt_volume.shape[0]
    click_mean = clicks.mean(0)

    def world_to_vox_f(w):
        return (w + 1.0) / 2.0 * (R - 1)

    click_vox = world_to_vox_f(click_mean)
    slice_coords = {
        "axial":    float(click_mean[0]),
        "coronal":  float(click_mean[1]),
        "sagittal": float(click_mean[2]),
    }

    # Build GT slices at click planes
    def slice_vol(vol, axis, coord):
        idx = int(round(world_to_vox_f(coord)[axis] if hasattr(coord, '__len__') else
                        (coord + 1.0) / 2.0 * (R - 1)))
        idx = np.clip(idx, 0, R - 1)
        return np.take(vol, idx, axis=axis)

    gt_slices = {
        "axial":    slice_vol(gt_volume, 0, click_mean[0]),
        "coronal":  slice_vol(gt_volume, 1, click_mean[1]),
        "sagittal": slice_vol(gt_volume, 2, click_mean[2]),
    }

    # Project similarity onto 2D slices
    points_np = points.cpu().numpy()
    sim_slices = project_sim_to_slices(sim_norm, points_np, gt_volume,
                                       slice_coords=slice_coords)

    fig = make_figure(sim_slices, gt_slices, click_mean, click_vox)

    import matplotlib.pyplot as plt
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also save raw similarity scores
    sim_out = out_path.replace(".png", "_sim.npy")
    np.save(sim_out, sim_np)
    print(f"Saved similarity scores: {sim_out}")


if __name__ == "__main__":
    main()
