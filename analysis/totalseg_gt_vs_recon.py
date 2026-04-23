"""TotalSegmentator comparison: GT volume vs. RadFoam reconstruction.

Runs TotalSegmentator (fast, 3mm model) on both the ground-truth volume and the
voxelized RadFoam reconstruction, then computes per-class Dice to quantify how
well the reconstruction preserves organ-level structure.

This is independent of feature extraction: it's a direct reconstruction-fidelity
metric, not a feature-quality metric.

Usage
-----
    # Run TotalSeg on GT + auto-voxelise the reconstruction:
    micromamba run -n radfoam python analysis/totalseg_gt_vs_recon.py \\
        --config output/<run>/config.yaml

    # Supply a pre-voxelized reconstruction (skips voxelization):
    micromamba run -n radfoam python analysis/totalseg_gt_vs_recon.py \\
        --config output/<run>/config.yaml \\
        --recon  output/<run>/volume.npy

    # Skip TotalSeg on GT if you already have a pre-run label volume:
    micromamba run -n radfoam python analysis/totalseg_gt_vs_recon.py \\
        --config output/<run>/config.yaml \\
        --gt-labels output/<run>/totalseg_gt.npy
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vis_foam import load_gt_volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_totalseg(volume_np, label_out_path=None, fast=True, device="cuda"):
    """Run TotalSegmentator on a [0,1]-normalised volume.

    Returns:
        (R,R,R) int16 label array (0=background, 1..N=organs)
        list of class names (index 0=background)
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        from totalsegmentator.map_to_binary import class_map
        import nibabel as nib
    except ImportError:
        raise ImportError("pip install TotalSegmentator")

    import tempfile

    R = volume_np.shape[0]
    hu_vol = (volume_np * 4000.0 - 1000.0).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(hu_vol, affine)

    dev_str = "gpu" if str(device).startswith("cuda") else "cpu"
    print(f"  TotalSegmentator (fast={fast}, device={dev_str}) …")

    with tempfile.TemporaryDirectory(prefix="totalseg_") as tmpdir:
        result_nii = totalsegmentator(
            nii, output=tmpdir, task="total",
            fast=fast, ml=True, quiet=True, device=dev_str,
        )
        if result_nii is None:
            ml_path = os.path.join(tmpdir, "s01.nii.gz")
            result_nii = nib.load(ml_path)
        seg_data = np.asarray(result_nii.dataobj).astype(np.int16)

    if seg_data.shape != (R, R, R):
        from scipy.ndimage import zoom
        seg_data = zoom(seg_data.astype(np.float32),
                        np.array([R, R, R]) / np.array(seg_data.shape),
                        order=0, prefilter=False).astype(np.int16)

    label_map = class_map["total"]
    class_names = ["background"] + [label_map[k] for k in sorted(label_map.keys())]

    if label_out_path:
        np.save(label_out_path, seg_data)
        print(f"  Saved labels: {label_out_path}")

    return seg_data, class_names


def voxelize_recon(model_pt, resolution=256, device="cuda"):
    """Voxelize a RadFoam model.pt into a (resolution³) numpy array."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from voxelize import voxelize

    import tempfile, os
    with tempfile.TemporaryDirectory(prefix="voxelize_") as tmpdir:
        out_npy = os.path.join(tmpdir, "volume.npy")
        vol = voxelize(model_pt, resolution, out_npy, extent=1.0,
                       blur_sigma=0.0, supersample=3, interpolate=True)
    return vol


def dice_per_class(pred, ref, n_classes):
    """Binary Dice for each class (0=background skipped).

    Returns (n_classes,) float array with NaN for absent classes.
    """
    dices = np.full(n_classes, np.nan)
    for c in range(1, n_classes):
        ref_c  = ref  == c
        if not ref_c.any():
            continue
        pred_c = pred == c
        inter = (pred_c & ref_c).sum()
        denom = pred_c.sum() + ref_c.sum()
        dices[c] = 2.0 * inter / (denom + 1e-8) if denom > 0 else 0.0
    return dices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare TotalSegmentator on GT volume vs. RadFoam reconstruction"
    )
    parser.add_argument("--config",    required=True,
                        help="output/<run>/config.yaml")
    parser.add_argument("--recon",     default=None,
                        help="Pre-voxelized reconstruction .npy (skips voxelization)")
    parser.add_argument("--gt-labels", default=None,
                        help="Pre-computed GT TotalSeg labels .npy (skips GT TotalSeg run)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Voxelization resolution (default: 256)")
    parser.add_argument("--fast",      action="store_true", default=True,
                        help="Use fast (3mm) TotalSegmentator model (default: True)")
    parser.add_argument("--no-fast",   dest="fast", action="store_false")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--out",       default=None,
                        help="Output CSV path (default: <run_dir>/totalseg_gt_vs_recon.csv)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir     = os.path.dirname(config_path)
    out_csv     = args.out or os.path.join(run_dir, "totalseg_gt_vs_recon.csv")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_path = cfg.get("data_path", "")
    dataset   = cfg.get("dataset", "r2_gaussian")

    # -----------------------------------------------------------------------
    # GT labels
    # -----------------------------------------------------------------------
    if args.gt_labels and os.path.exists(args.gt_labels):
        print(f"Loading pre-computed GT labels: {args.gt_labels}")
        gt_labels = np.load(args.gt_labels).astype(np.int16)
        # Infer class names from TotalSegmentator map
        try:
            from totalsegmentator.map_to_binary import class_map
            label_map = class_map["total"]
            class_names = ["background"] + [label_map[k] for k in sorted(label_map.keys())]
        except ImportError:
            class_names = [f"class_{i}" for i in range(int(gt_labels.max()) + 1)]
    else:
        print("Running TotalSegmentator on GT volume …")
        gt_volume = load_gt_volume(data_path, dataset)
        if gt_volume is None:
            print("GT volume not found.")
            sys.exit(1)
        # Normalize to [0,1]
        lo, hi = gt_volume.min(), gt_volume.max()
        if hi > lo:
            gt_volume = (gt_volume - lo) / (hi - lo)
        gt_save = os.path.join(run_dir, "totalseg_gt_labels.npy")
        gt_labels, class_names = run_totalseg(
            gt_volume, label_out_path=gt_save,
            fast=args.fast, device=args.device
        )

    n_classes = len(class_names)
    present_gt = set(int(c) for c in np.unique(gt_labels) if c > 0)
    print(f"GT: {len(present_gt)} organ classes present")

    # -----------------------------------------------------------------------
    # Reconstruction labels
    # -----------------------------------------------------------------------
    model_pt = os.path.join(run_dir, "model.pt")

    if args.recon and os.path.exists(args.recon):
        print(f"Loading pre-voxelized reconstruction: {args.recon}")
        recon_vol = np.load(args.recon).astype(np.float32)
    elif os.path.exists(model_pt):
        print(f"Voxelizing reconstruction from {model_pt} …")
        recon_vol = voxelize_recon(model_pt, resolution=args.resolution,
                                   device=args.device)
    else:
        print(f"No model.pt at {model_pt} and --recon not supplied.")
        sys.exit(1)

    # Normalize recon to [0, 1] before TotalSeg
    lo, hi = recon_vol.min(), recon_vol.max()
    if hi > lo:
        recon_vol = (recon_vol - lo) / (hi - lo)

    recon_save = os.path.join(run_dir, "totalseg_recon_labels.npy")
    recon_labels, _ = run_totalseg(
        recon_vol, label_out_path=recon_save,
        fast=args.fast, device=args.device
    )

    # -----------------------------------------------------------------------
    # Dice
    # -----------------------------------------------------------------------
    dices = dice_per_class(recon_labels, gt_labels, n_classes)

    rows = []
    for c in range(1, n_classes):
        if c not in present_gt:
            continue
        d = dices[c]
        rows.append((class_names[c], c, float(d) if not np.isnan(d) else float("nan")))

    rows.sort(key=lambda r: -r[2] if not np.isnan(r[2]) else -999)

    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organ", "class_id", "dice"])
        for name, cid, d in rows:
            w.writerow([name, cid, f"{d:.4f}" if not np.isnan(d) else "nan"])

    valid = [r[2] for r in rows if not np.isnan(r[2])]
    mean_dice = float(np.mean(valid)) if valid else float("nan")
    print(f"\nTotalSeg GT vs. Recon — {len(valid)} classes:")
    print(f"  Mean Dice: {mean_dice:.4f}")
    for name, cid, d in rows[:10]:
        print(f"  {name:<30s} {d:.3f}")
    if len(rows) > 10:
        print(f"  … ({len(rows) - 10} more classes in {out_csv})")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
