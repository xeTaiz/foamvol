"""Download and resample CT-ORG segmentation labels to match a vol_gt.npy grid.

CT-ORG label classes:
  0=background  1=liver  2=bladder  3=lungs  4=kidneys  5=bone  6=brain

Sources tried (in order):
  1. HF Hub: mrmrx/CADS-dataset  (use --source hf to force)
  2. Zenodo record 3546986  (use --source zenodo to force)

Usage
-----
    python analysis/fetch_ctorg_labels.py \\
        --vol r2_data/ct_org/CT_ORG_case_00_cone/vol_gt.npy \\
        --case 0 --verify

    # Force a specific source
    python analysis/fetch_ctorg_labels.py --vol ... --case 0 --source zenodo
"""

import argparse
import os
import sys

import numpy as np


CTORG_LABEL_NAMES = {
    0: "background",
    1: "liver",
    2: "bladder",
    3: "lungs",
    4: "kidneys",
    5: "bone",
    6: "brain",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_hf(case_id, tmpdir):
    """Download labels-{case_id}.nii.gz from mrmrx/CADS-dataset on HF Hub."""
    from huggingface_hub import HfFileSystem
    import re

    fs = HfFileSystem()
    repo_prefix = "datasets/mrmrx/CADS-dataset"

    # Recursively search for a file matching labels-{case_id} anywhere in the repo
    print(f"  Searching HF repo {repo_prefix} for case {case_id} labels …")
    target_re = re.compile(rf"labels[-_]0*{case_id}[^/]*\.nii(\.gz)?$", re.IGNORECASE)

    def search_dir(path, depth=0):
        if depth > 5:
            return None
        try:
            entries = fs.ls(path, detail=False)
        except Exception:
            return None
        for entry in entries:
            if target_re.search(os.path.basename(entry)):
                return entry
        for entry in entries:
            if fs.isdir(entry):
                result = search_dir(entry, depth + 1)
                if result:
                    return result
        return None

    hf_path = search_dir(repo_prefix)
    if hf_path is None:
        raise RuntimeError(
            f"Could not find labels-{case_id} in {repo_prefix}.\n"
            "Inspect manually:\n"
            "  from huggingface_hub import HfFileSystem\n"
            "  fs = HfFileSystem()\n"
            f"  fs.ls('datasets/mrmrx/CADS-dataset', detail=False)"
        )

    local_name = f"labels-{case_id}.nii.gz"
    local_path = os.path.join(tmpdir, local_name)
    print(f"  Found: {hf_path} — downloading …")
    with fs.open(hf_path, "rb") as src, open(local_path, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    print(f"  Done: {local_path}")
    return local_path


def _download_zenodo(case_id, tmpdir):
    """Download labels-{case_id}.nii.gz from Zenodo."""
    import urllib.request

    # CT-ORG is on Zenodo record 3546986 (Rister et al. 2020).
    candidate_urls = [
        f"https://zenodo.org/record/3546986/files/labels-{case_id}.nii.gz",
        f"https://zenodo.org/records/3546986/files/labels-{case_id}.nii.gz",
    ]

    local_path = os.path.join(tmpdir, f"labels-{case_id}.nii.gz")
    last_err = None
    for url in candidate_urls:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, local_path)
            print(f"  Downloaded: {local_path}")
            return local_path
        except Exception as e:
            last_err = e
            print(f"  Failed: {e}")

    raise RuntimeError(
        f"Could not download labels-{case_id}.nii.gz from Zenodo.\n"
        "Download manually from https://zenodo.org/record/3546986\n"
        f"and place as: {local_path}"
    ) from last_err


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_labels(labels_data, target_shape):
    """Resample integer label volume to target_shape with nearest-neighbor.

    The R2-Gaussian CT-ORG preprocessing crops to a centered cube FOV and
    resamples to 256³. We mirror that here: crop non-cubic volumes to the
    shortest dimension, then zoom to target_shape.

    Args:
        labels_data: (H, W, D) int16 numpy from nibabel
        target_shape: (R, R, R) target shape

    Returns:
        (R, R, R) int16 numpy
    """
    from scipy.ndimage import zoom

    data = labels_data.astype(np.int16)
    src = np.array(data.shape[:3], dtype=float)
    tgt = np.array(target_shape[:3], dtype=float)

    # Crop non-cubic source to centered cube (matches R2-Gaussian convention)
    if src.max() / src.min() > 1.1:
        min_d = int(src.min())
        print(f"  Non-cubic source {data.shape} — cropping to centered {min_d}³")
        offsets = ((src - min_d) / 2).astype(int)
        data = data[
            offsets[0]: offsets[0] + min_d,
            offsets[1]: offsets[1] + min_d,
            offsets[2]: offsets[2] + min_d,
        ]
        src = np.array([min_d, min_d, min_d], dtype=float)

    factors = tgt / src
    print(f"  Resampling {data.shape} → {tuple(int(t) for t in tgt)}  "
          f"(factors: {factors.round(3)})")
    resampled = zoom(data, factors, order=0, prefilter=False)
    return resampled.astype(np.int16)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def save_verification_png(labels_np, vol_gt_np, out_path):
    """Save a 3×2 overlay (axial/coronal/sagittal mid-slices)."""
    import matplotlib.pyplot as plt

    R = vol_gt_np.shape[0]
    mid = R // 2
    n_classes = len(CTORG_LABEL_NAMES)

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    lut = plt.get_cmap("tab10")

    slices = [
        ("axial",    (slice(None), slice(None), mid)),
        ("coronal",  (slice(None), mid, slice(None))),
        ("sagittal", (mid, slice(None), slice(None))),
    ]
    for row, (name, sl) in enumerate(slices):
        gt = vol_gt_np[sl].T
        lb = labels_np[sl].T

        axes[row, 0].imshow(gt, cmap="gray", origin="lower",
                            vmin=0, vmax=max(gt.max(), 1e-6))
        axes[row, 0].set_title(f"GT {name}")
        axes[row, 0].axis("off")

        # Semi-transparent label overlay on GT
        rgba = lut(lb.astype(float) / max(n_classes - 1, 1))
        rgba[lb == 0, 3] = 0.0      # transparent background
        axes[row, 0].imshow(rgba, origin="lower", alpha=0.55)

        axes[row, 1].imshow(lb, cmap="tab10", origin="lower",
                            vmin=0, vmax=n_classes - 1, interpolation="nearest")
        axes[row, 1].set_title(f"Labels {name}")
        axes[row, 1].axis("off")

    present = {k: CTORG_LABEL_NAMES[k] for k in np.unique(labels_np)
               if k in CTORG_LABEL_NAMES and k != 0}
    fig.suptitle(
        "CT-ORG labels resampled to 256³\n"
        + "  ".join(f"{k}={v}" for k, v in present.items()),
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved verification: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download + resample CT-ORG labels")
    parser.add_argument("--vol", required=True,
                        help="Path to vol_gt.npy (reference grid shape)")
    parser.add_argument("--case", type=int, default=0,
                        help="CT-ORG case number 0–21 (default: 0)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: labels.npy next to vol_gt.npy)")
    parser.add_argument("--source", choices=["hf", "zenodo", "auto"], default="auto",
                        help="Download source (default: try HF first, then Zenodo)")
    parser.add_argument("--verify", action="store_true",
                        help="Save a verification overlay PNG")
    args = parser.parse_args()

    vol_dir = os.path.dirname(os.path.abspath(args.vol))
    out_path = args.out or os.path.join(vol_dir, "labels.npy")

    vol_gt = np.load(args.vol)
    print(f"Reference volume: {vol_gt.shape}  "
          f"range [{vol_gt.min():.3f}, {vol_gt.max():.3f}]")

    import tempfile
    with tempfile.TemporaryDirectory(prefix="ctorg_labels_") as tmpdir:
        labels_path = None

        if args.source in ("hf", "auto"):
            try:
                labels_path = _download_hf(args.case, tmpdir)
            except Exception as e:
                print(f"HF download failed: {e}")
                if args.source == "hf":
                    sys.exit(1)

        if labels_path is None and args.source in ("zenodo", "auto"):
            try:
                labels_path = _download_zenodo(args.case, tmpdir)
            except Exception as e:
                print(f"Zenodo download failed: {e}")
                sys.exit(1)

        import nibabel as nib
        nii = nib.load(labels_path)
        raw = np.asarray(nii.dataobj)
        print(f"Loaded labels NIfTI: shape={nii.shape}  "
              f"dtype={nii.get_data_dtype()}  "
              f"unique={np.unique(raw)}")

        labels_np = resample_labels(raw, vol_gt.shape)

    np.save(out_path, labels_np)
    print(f"Saved: {out_path}")

    present = {k: CTORG_LABEL_NAMES.get(k, f"class_{k}")
               for k in np.unique(labels_np)}
    counts = {v: int((labels_np == k).sum()) for k, v in present.items()}
    print(f"Present labels: {present}")
    print(f"Voxel counts:   {counts}")

    if args.verify:
        save_verification_png(labels_np, vol_gt,
                              out_path.replace(".npy", "_verify.png"))


if __name__ == "__main__":
    main()
