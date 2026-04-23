"""Quantitative feature-field evaluation: prototype-based organ segmentation Dice.

Uses cell features to assign organ labels to each Voronoi cell via cosine-similarity
prototype matching, then voxelizes and computes Dice vs a reference segmentation.

The reference segmentation is the TotalSegmentator output on the GT volume. If
pre-computed TotalSegmentator features are provided via --ref-features, those are
used directly as the reference label map.  Otherwise a fresh TotalSegmentator run
is launched to generate labels.

Usage
-----
    # Evaluate DINO features against TotalSegmentator organ labels
    micromamba run -n radfoam python analysis/feature_dice_eval.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_dino_vits8_axial.npz \\
        [--ref-features output/<run>/features_totalseg_fast.npz] \\
        [--out output/<run>/dice_dino.csv]

    # Also evaluate the upper bound (features sampled directly on GT volume)
    micromamba run -n radfoam python analysis/feature_dice_eval.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_dino_vits8_axial.npz \\
        --upper-bound
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vis_foam import load_gt_volume, load_density_field
from radfoam_model.features import load_cell_features, assign_cell_features, save_cell_features


def get_totalseg_label_map():
    """Return {class_id: name} for TotalSegmentator total task."""
    try:
        from totalsegmentator.map_to_binary import class_map
        return class_map["total"]
    except ImportError:
        return {}


def run_totalseg_labels(gt_volume, device="cuda", fast=True):
    """Run TotalSegmentator on gt_volume and return (R,R,R) integer label array.

    Args:
        gt_volume: (R,R,R) float32, values in [0,1] normalised density
        device:    torch device string

    Returns:
        (R,R,R) int16 array with organ label indices (0=background)
        list of class names in label order (index 1..N_classes)
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        from totalsegmentator.map_to_binary import class_map
        import nibabel as nib
    except ImportError:
        raise ImportError("TotalSegmentator is required for generating reference labels. "
                          "Run: pip install TotalSegmentator")

    import tempfile

    R = gt_volume.shape[0]
    hu_vol = (gt_volume * 4000.0 - 1000.0).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(hu_vol, affine)

    with tempfile.TemporaryDirectory(prefix="totalseg_ref_") as tmpdir:
        dev_str = "gpu" if str(device).startswith("cuda") else "cpu"
        result_nii = totalsegmentator(
            nii, output=tmpdir, task="total",
            fast=fast, ml=True, quiet=True, device=dev_str,
        )
        if result_nii is None:
            import nibabel as nib2
            result_nii = nib2.load(os.path.join(tmpdir, "s01.nii.gz"))

        seg_data = np.asarray(result_nii.dataobj).astype(np.int16)

    label_map = class_map["total"]                     # {int: "name"}
    class_names = ["background"] + [label_map[k] for k in sorted(label_map.keys())]
    return seg_data, class_names


def labels_to_onehot(seg_data, n_classes):
    """Convert integer (R,R,R) segmentation to (R,R,R,n_classes) float32 one-hot."""
    R = seg_data.shape[0]
    onehot = np.zeros((R, R, R, n_classes), dtype=np.float32)
    for c in range(n_classes):
        onehot[..., c] = (seg_data == c).astype(np.float32)
    return onehot


def prototype_classify(cell_feats, ref_cell_labels, n_classes):
    """Classify each cell by cosine similarity to per-class prototype vectors.

    Args:
        cell_feats:      (N, F) float32 tensor
        ref_cell_labels: (N,) int tensor — reference label per cell (0=background)
        n_classes:       total number of classes including background

    Returns:
        (N,) int tensor — predicted class per cell
    """
    device = cell_feats.device
    F_dim = cell_feats.shape[1]

    prototypes = torch.zeros(n_classes, F_dim, device=device)
    counts = torch.zeros(n_classes, device=device)
    for c in range(n_classes):
        mask = ref_cell_labels == c
        if mask.any():
            prototypes[c] = cell_feats[mask].float().mean(0)
            counts[c] = mask.sum()

    # Normalise prototypes and features
    proto_norm = torch.nn.functional.normalize(prototypes, dim=1)
    feat_norm  = torch.nn.functional.normalize(cell_feats.float(), dim=1)

    # Cosine similarity matrix: (N, n_classes)
    sim = feat_norm @ proto_norm.T  # (N, n_classes)

    # Zero out classes with no training samples
    sim[:, counts == 0] = -1.0

    predicted = sim.argmax(dim=1)
    return predicted


def voxelize_labels(points, labels, R, extent=1.0):
    """Scatter cell integer labels onto an (R,R,R) voxel grid (nearest-cell).

    Args:
        points: (N,3) float32 numpy world coords in [-extent, extent]^3
        labels: (N,) int numpy
        R:      grid resolution

    Returns:
        (R,R,R) int16 numpy
    """
    grid = np.zeros((R, R, R), dtype=np.int16)
    vi = ((points + extent) / (2 * extent) * (R - 1)).round().astype(int).clip(0, R - 1)
    grid[vi[:, 0], vi[:, 1], vi[:, 2]] = labels.astype(np.int16)
    return grid


def dice_per_class(pred_vol, ref_vol, n_classes, ignore_bg=True):
    """Compute Dice coefficient for each class.

    Args:
        pred_vol: (R,R,R) int
        ref_vol:  (R,R,R) int
        n_classes: total classes (0=background)
        ignore_bg: skip class 0

    Returns:
        (n_classes,) float array — Dice per class (NaN if class absent)
    """
    dices = np.full(n_classes, np.nan)
    start = 1 if ignore_bg else 0
    for c in range(start, n_classes):
        pred_c = pred_vol == c
        ref_c  = ref_vol  == c
        if not ref_c.any():
            continue
        inter = (pred_c & ref_c).sum()
        denom = pred_c.sum() + ref_c.sum()
        dices[c] = 2.0 * inter / (denom + 1e-8) if denom > 0 else 0.0
    return dices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True)
    parser.add_argument("--features",     required=True,
                        help="Cell feature .npz from feature_extract.py")
    parser.add_argument("--ref-features", default=None,
                        help="TotalSegmentator cell features .npz to use as GT labels "
                             "(if omitted, runs TotalSegmentator from scratch)")
    parser.add_argument("--upper-bound",  action="store_true",
                        help="Also compute upper bound: features sampled directly from "
                             "GT volume (no Voronoi representation involved)")
    parser.add_argument("--fast",         action="store_true",
                        help="Use fast TotalSegmentator model when generating labels")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--out",          default=None,
                        help="CSV output path (default: <run_dir>/dice_<backbone>.csv)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_path = cfg.get("data_path", "")
    dataset   = cfg.get("dataset", "r2_gaussian")

    # Load checkpoint
    model_pt = os.path.join(run_dir, "model.pt")
    print(f"Loading checkpoint: {model_pt}")
    field  = load_density_field(model_pt, device=args.device)
    points = field["points"].cpu().numpy()  # (N,3)
    N = points.shape[0]
    print(f"Points: {N:,}")

    # Load features to evaluate
    print(f"Loading features: {args.features}")
    cell_feats_np, feat_meta = load_cell_features(args.features)
    backbone = str(feat_meta.get("backbone", "unknown"))
    if backbone.startswith("['") or backbone.startswith("[b'"):
        backbone = backbone.strip("[]b'\" ")   # unwrap numpy string array repr
    F_dim = cell_feats_np.shape[1]
    print(f"Features: {cell_feats_np.shape}  backbone={backbone}")

    # Get reference GT labels (cell-level integer class assignments)
    if args.ref_features is not None:
        print(f"Loading reference TotalSegmentator features: {args.ref_features}")
        ref_feats_np, _ = load_cell_features(args.ref_features)
        # One-hot → argmax → integer labels
        ref_labels = ref_feats_np.astype(np.float32).argmax(axis=1) + 1
        # Cells with all-zero features → background (0)
        ref_labels[ref_feats_np.max(axis=1) == 0] = 0
        ref_labels = ref_labels.astype(np.int16)
        label_map = get_totalseg_label_map()
        class_names = ["background"] + [label_map.get(k, f"class_{k}")
                                        for k in sorted(label_map.keys())]
    else:
        print("Running TotalSegmentator to generate reference labels …")
        gt_volume = load_gt_volume(data_path, dataset)
        if gt_volume is None:
            print("No GT volume found. Exiting.")
            sys.exit(1)
        seg_vol, class_names = run_totalseg_labels(
            gt_volume, device=args.device, fast=args.fast
        )
        R = gt_volume.shape[0]
        # Assign voxel labels to cells via nearest-cell (voxelize)
        pts_vox = ((points + 1.0) / 2.0 * (R - 1)).round().astype(int).clip(0, R - 1)
        ref_labels = seg_vol[pts_vox[:, 0], pts_vox[:, 1], pts_vox[:, 2]]

    n_classes = int(ref_labels.max()) + 1
    print(f"Reference: {n_classes} classes, "
          f"{(ref_labels > 0).sum()} foreground cells / {N}")

    # ---------------------
    # Evaluate Voronoi cells
    # ---------------------
    dev = torch.device(args.device)
    cell_feats_t = torch.from_numpy(cell_feats_np.astype(np.float32)).to(dev)
    ref_labels_t = torch.from_numpy(ref_labels.astype(np.int64)).to(dev)

    print("Computing prototype classification …")
    pred_labels_t = prototype_classify(cell_feats_t, ref_labels_t, n_classes)
    pred_labels = pred_labels_t.cpu().numpy().astype(np.int16)

    # Voxelize both predictions and reference for Dice
    gt_volume_tmp = load_gt_volume(data_path, dataset)
    if gt_volume_tmp is not None:
        R = gt_volume_tmp.shape[0]
    else:
        R = 128
    pred_vol = voxelize_labels(points, pred_labels, R)
    ref_vol  = voxelize_labels(points, ref_labels,  R)

    dices = dice_per_class(pred_vol, ref_vol, n_classes)

    present = np.where(~np.isnan(dices[1:]))[0]
    mean_dice = float(np.nanmean(dices[1:]))
    print(f"\nMean Dice ({len(present)} classes present): {mean_dice:.4f}")

    # ---------------------
    # Upper bound (optional)
    # ---------------------
    upper_bound_dices = None
    if args.upper_bound:
        print("\nComputing upper bound (features directly on GT volume) …")
        gt_volume_ub = load_gt_volume(data_path, dataset)
        if gt_volume_ub is None:
            print("  No GT volume — skipping upper bound.")
        else:
            lo, hi = gt_volume_ub.min(), gt_volume_ub.max()
            if hi > lo:
                gt_volume_ub = (gt_volume_ub - lo) / (hi - lo)
            from feature_extract import extract_features_dino, _DINO_HUB_REPOS
            if backbone in _DINO_HUB_REPOS:
                print(f"  Re-extracting {backbone} features on GT volume …")
                ub_feat_vol = extract_features_dino(
                    gt_volume_ub, backbone, axes=("axial",), device=args.device
                )
            else:
                print(f"  Backbone {backbone!r} not DINO — using same features as Voronoi eval.")
                ub_feat_vol = None

            if ub_feat_vol is not None:
                # Sample at voxel centres of the reference volume
                pts_t = torch.from_numpy(points).to(dev)
                ub_cell_feats = assign_cell_features(pts_t, ub_feat_vol).to(dev).float()
                ub_pred_t = prototype_classify(ub_cell_feats, ref_labels_t, n_classes)
                ub_pred_vol = voxelize_labels(points,
                                              ub_pred_t.cpu().numpy().astype(np.int16), R)
                upper_bound_dices = dice_per_class(ub_pred_vol, ref_vol, n_classes)
                print(f"  Upper bound mean Dice: {np.nanmean(upper_bound_dices[1:]):.4f}")

    # ---------------------
    # Save CSV
    # ---------------------
    backbone_tag = backbone.replace("/", "-")
    out_path = args.out or os.path.join(run_dir, f"dice_{backbone_tag}.csv")

    header_parts = ["class_id", "class_name", f"dice_{backbone_tag}"]
    if upper_bound_dices is not None:
        header_parts.append(f"dice_upperbound_{backbone_tag}")

    with open(out_path, "w") as f:
        f.write(",".join(header_parts) + "\n")
        for c in range(1, n_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            d = dices[c]
            row = [str(c), name, f"{d:.4f}" if not np.isnan(d) else "nan"]
            if upper_bound_dices is not None:
                ub = upper_bound_dices[c]
                row.append(f"{ub:.4f}" if not np.isnan(ub) else "nan")
            f.write(",".join(row) + "\n")

    print(f"Saved: {out_path}")

    # Print top / bottom classes
    valid = [(c, dices[c], class_names[c] if c < len(class_names) else f"class_{c}")
             for c in range(1, n_classes) if not np.isnan(dices[c])]
    valid.sort(key=lambda x: -x[1])
    print("\nTop 10 Dice classes:")
    for c, d, name in valid[:10]:
        print(f"  {name:40s}  Dice={d:.4f}")
    print("Bottom 10 Dice classes:")
    for c, d, name in valid[-10:]:
        print(f"  {name:40s}  Dice={d:.4f}")


if __name__ == "__main__":
    main()
