"""Extract 3D feature volumes from a trained RadFoam checkpoint.

Supported backbones
-------------------
dino_vits8         DINOv1 ViT-S/8  (patch 8,  dim 384)  torch.hub
dino_vitb8         DINOv1 ViT-B/8  (patch 8,  dim 768)  torch.hub
dino_vits16        DINOv1 ViT-S/16 (patch 16, dim 384)  torch.hub
dino_vitb16        DINOv1 ViT-B/16 (patch 16, dim 768)  torch.hub
dinov2_vits14      DINOv2 ViT-S/14 (patch 14, dim 384)  torch.hub
dinov2_vitb14      DINOv2 ViT-B/14 (patch 14, dim 768)  torch.hub
dinov2_vitl14      DINOv2 ViT-L/14 (patch 14, dim 1024) torch.hub
dinov3_vits14      DINOv3 ViT-S/14 (patch 14, dim 384)  HF transformers
dinov3_vitb14      DINOv3 ViT-B/14 (patch 14, dim 768)  HF transformers
dinov3_vitl14      DINOv3 ViT-L/14 (patch 14, dim 1024) HF transformers
totalsegmentator   TotalSegmentator nnUNet encoder        pip install TotalSegmentator

Usage
-----
    micromamba run -n radfoam python feature_extract.py \\
        --config output/<run>/config.yaml \\
        --backbone dino_vits8 \\
        [--axes axial coronal sagittal] \\
        [--batch 32] \\
        [--out output/<run>/features_dino_vits8.npz]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as TF
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vis_foam import load_gt_volume, load_density_field
from radfoam_model.features import assign_cell_features, save_cell_features


# ---------------------------------------------------------------------------
# DINO / ViT backbone (torch.hub — no extra installs needed)
# ---------------------------------------------------------------------------

_DINO_HUB_REPOS = {
    "dino_vits8":    ("facebookresearch/dino:main",  "dino_vits8"),
    "dino_vits16":   ("facebookresearch/dino:main",  "dino_vits16"),
    "dino_vitb8":    ("facebookresearch/dino:main",  "dino_vitb8"),
    "dino_vitb16":   ("facebookresearch/dino:main",  "dino_vitb16"),
    "dinov2_vits14": ("facebookresearch/dinov2",     "dinov2_vits14"),
    "dinov2_vitb14": ("facebookresearch/dinov2",     "dinov2_vitb14"),
    "dinov2_vitl14": ("facebookresearch/dinov2",     "dinov2_vitl14"),
}

# ---------------------------------------------------------------------------
# DINOv3 — HF transformers (Meta 2025 release)
# Update the HF repo IDs below once confirmed; current values are placeholders.
# ---------------------------------------------------------------------------

# Map backbone name → (hf_repo_id, patch_size, embed_dim)
# TODO: replace with confirmed HF repo IDs once Meta publishes DINOv3 weights.
_DINOV3_HF_REPOS = {
    "dinov3_vits14": ("facebook/dinov3-vit-small-14", 14, 384),
    "dinov3_vitb14": ("facebook/dinov3-vit-base-14",  14, 768),
    "dinov3_vitl14": ("facebook/dinov3-vit-large-14", 14, 1024),
}

# ImageNet stats (DINO models expect ImageNet-normalized RGB)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def _load_dino_model(backbone, device):
    if backbone not in _DINO_HUB_REPOS:
        raise ValueError(f"Unknown backbone '{backbone}'. "
                         f"Choices: {list(_DINO_HUB_REPOS)}")
    repo, name = _DINO_HUB_REPOS[backbone]
    print(f"Loading {backbone} from torch.hub ({repo}) …")
    model = torch.hub.load(repo, name, pretrained=True)
    model.eval().to(device)
    return model


def _patch_size(model):
    """Infer patch size from a DINO / ViT model."""
    if hasattr(model, "patch_embed"):
        ps = model.patch_embed.patch_size
        return ps if isinstance(ps, int) else ps[0]
    raise AttributeError("Cannot determine patch size from model")


def _feat_dim(model):
    """Infer feature dimension (embed_dim) from a DINO / ViT model."""
    if hasattr(model, "embed_dim"):
        return model.embed_dim
    raise AttributeError("Cannot determine embed_dim from model")


@torch.no_grad()
def _extract_slices_dino(model, slices_np, patch_size, feat_dim,
                         device, batch_size=32):
    """Forward a stack of 2D slices through a DINO model.

    Args:
        slices_np: (S, H, W) float32, values in [0, 1]
        patch_size: model patch size (int)
        feat_dim:   model feature dimension (int)

    Returns:
        (S, H, W, feat_dim) float16 numpy  — patch tokens upsampled to original H×W
    """
    S, H, W = slices_np.shape

    # Pad H and W to be multiples of patch_size
    pad_h = (-H) % patch_size
    pad_w = (-W) % patch_size
    pH, pW = H + pad_h, W + pad_w

    mean = _IMAGENET_MEAN.to(device)
    std  = _IMAGENET_STD.to(device)

    out = np.empty((S, H, W, feat_dim), dtype=np.float16)

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)
        batch = slices_np[start:end]   # (B, H, W) float32

        # Grayscale → RGB, normalize
        t = torch.from_numpy(batch).to(device)           # (B, H, W)
        t = t.unsqueeze(1).expand(-1, 3, -1, -1)         # (B, 3, H, W)
        t = TF.pad(t, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else t
        t = (t - mean[:, None, None]) / std[:, None, None]

        # Get patch tokens (skip CLS token)
        if hasattr(model, "get_intermediate_layers"):
            # DINOv1 API
            feats = model.get_intermediate_layers(t, n=1)[0]  # (B, n_patches+1, D)
            feats = feats[:, 1:]                               # drop CLS
        else:
            # DINOv2 API
            feats = model.forward_features(t)["x_norm_patchtokens"]  # (B, n_patches, D)

        n_h = pH // patch_size
        n_w = pW // patch_size
        B = end - start
        feats = feats.reshape(B, n_h, n_w, feat_dim)   # (B, n_h, n_w, D)
        feats = feats.permute(0, 3, 1, 2).float()       # (B, D, n_h, n_w)

        # Upsample to (pH, pW) then crop to (H, W)
        feats = TF.interpolate(feats, size=(pH, pW), mode="bilinear", align_corners=False)
        feats = feats[:, :, :H, :W]                     # (B, D, H, W)
        feats = feats.permute(0, 2, 3, 1).half().cpu().numpy()  # (B, H, W, D)
        out[start:end] = feats

        print(f"  slices {start}–{end-1}/{S}", end="\r", flush=True)

    print()
    return out


def _load_hf_vit_model(hf_repo, device):
    """Load a ViT from HF transformers (used for DINOv3 and future HF-hosted models)."""
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError:
        raise ImportError(
            "Install transformers to use HF-hosted backbones:\n"
            "  pip install transformers"
        )
    print(f"Loading from HF transformers: {hf_repo} …")
    model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def _extract_slices_hf(model, slices_np, patch_size, feat_dim, device, batch_size=32):
    """Forward a stack of 2D slices through an HF ViT model.

    Mirrors _extract_slices_dino but uses the HF transformers API.  Tries
    three common output formats:
      1. model(pixel_values=t).last_hidden_state   (standard HF ViT)
      2. model(pixel_values=t).patch_embeddings     (some DINOv2 HF wrappers)
      3. model.get_intermediate_layers(t, n=1)[0]   (DINO-style hub API)

    Returns:
        (S, H, W, feat_dim) float16 numpy
    """
    S, H, W = slices_np.shape

    pad_h = (-H) % patch_size
    pad_w = (-W) % patch_size
    pH, pW = H + pad_h, W + pad_w

    mean = _IMAGENET_MEAN.to(device)
    std  = _IMAGENET_STD.to(device)

    out = np.empty((S, H, W, feat_dim), dtype=np.float16)

    for start in range(0, S, batch_size):
        end = min(start + batch_size, S)
        batch = slices_np[start:end]

        t = torch.from_numpy(batch).to(device)              # (B, H, W)
        t = t.unsqueeze(1).expand(-1, 3, -1, -1)            # (B, 3, H, W)
        if pad_h or pad_w:
            t = TF.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        t = (t - mean[:, None, None]) / std[:, None, None]

        B = end - start

        # Try HF transformers standard API first
        try:
            out_obj = model(pixel_values=t)
            if hasattr(out_obj, "last_hidden_state"):
                feats = out_obj.last_hidden_state[:, 1:]   # drop CLS token
            elif hasattr(out_obj, "patch_embeddings"):
                feats = out_obj.patch_embeddings
            else:
                feats = list(out_obj.values())[0][:, 1:]
        except TypeError:
            # Fall back to torch.hub DINO-style API
            if hasattr(model, "get_intermediate_layers"):
                feats = model.get_intermediate_layers(t, n=1)[0][:, 1:]
            else:
                feats = model(t)[:, 1:]

        n_h = pH // patch_size
        n_w = pW // patch_size
        feats = feats.reshape(B, n_h, n_w, feat_dim)
        feats = feats.permute(0, 3, 1, 2).float()
        feats = TF.interpolate(feats, size=(pH, pW), mode="bilinear", align_corners=False)
        feats = feats[:, :, :H, :W]
        out[start:end] = feats.permute(0, 2, 3, 1).half().cpu().numpy()

        print(f"  slices {start}–{end-1}/{S}", end="\r", flush=True)

    print()
    return out


def extract_features_dinov3(volume_np, backbone, axes=("axial",),
                             device="cuda", batch_size=16):
    """Stack-wise DINOv3 feature extraction (HF transformers backend).

    Same interface as extract_features_dino but loads from HF Hub.
    """
    if backbone not in _DINOV3_HF_REPOS:
        raise ValueError(f"Unknown DINOv3 backbone '{backbone}'.")
    hf_repo, ps, fdim = _DINOV3_HF_REPOS[backbone]

    dev = torch.device(device)
    model = _load_hf_vit_model(hf_repo, dev)
    R = volume_np.shape[0]

    accum = np.zeros((R, R, R, fdim), dtype=np.float32)
    count = 0

    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    for ax_name in axes:
        ax = axis_map[ax_name]
        print(f"  Axis: {ax_name} (dim {ax}), {R} slices …")
        slices = np.moveaxis(volume_np, ax, 0).copy()
        feat_slices = _extract_slices_hf(model, slices, ps, fdim, dev,
                                          batch_size=batch_size)
        feat_vol = np.moveaxis(feat_slices, 0, ax)
        accum += feat_vol.astype(np.float32)
        count += 1

    del model
    torch.cuda.empty_cache()
    return (accum / count).astype(np.float16)


def extract_features_dino(volume_np, backbone, axes=("axial",),
                          device="cuda", batch_size=32):
    """Stack-wise DINO feature extraction from a 3D volume.

    Applies the model slice-by-slice along each requested axis and averages
    the results. Volume is assumed (X, Y, Z) = (D, H, W), values in [0, 1].

    Args:
        volume_np: (R, R, R) float32 numpy
        backbone:  model name from _DINO_HUB_REPOS
        axes:      tuple of "axial" / "coronal" / "sagittal"
        device:    torch device string
        batch_size: slices per GPU forward pass

    Returns:
        (R, R, R, F) float16 numpy feature volume, same spatial convention as volume_np
    """
    dev = torch.device(device)
    model = _load_dino_model(backbone, dev)
    ps   = _patch_size(model)
    fdim = _feat_dim(model)
    R    = volume_np.shape[0]

    accum = np.zeros((R, R, R, fdim), dtype=np.float32)
    count = 0

    axis_map = {
        "axial":    0,   # iterate over X, each slice is (Y,Z)
        "coronal":  1,   # iterate over Y, each slice is (X,Z)
        "sagittal": 2,   # iterate over Z, each slice is (X,Y)
    }

    for ax_name in axes:
        ax = axis_map[ax_name]
        print(f"  Axis: {ax_name} (dim {ax}), {R} slices …")

        # Extract slices along this axis: always (R, R, R) → (R, R, R)
        slices = np.moveaxis(volume_np, ax, 0).copy()   # (R, slice_H, slice_W)

        feat_slices = _extract_slices_dino(
            model, slices, ps, fdim, dev, batch_size=batch_size
        )  # (R, R, R, F)

        # Move slice axis back to its original position
        feat_vol = np.moveaxis(feat_slices, 0, ax)   # (R, R, R, F)
        accum += feat_vol.astype(np.float32)
        count += 1

    del model
    torch.cuda.empty_cache()

    result = (accum / count).astype(np.float16)
    return result


# ---------------------------------------------------------------------------
# TotalSegmentator backend — one-hot organ label features
# ---------------------------------------------------------------------------

# 117-class TotalSegmentator label map (class_map["total_v2"])
_TOTALSEG_TASK = "total"   # use the standard total body task

def _volume_to_nifti(volume_np, voxel_spacing_mm=1.0):
    """Wrap a (R,R,R) numpy array as a nibabel NIfTI with isotropic spacing."""
    import nibabel as nib
    s = float(voxel_spacing_mm)
    affine = np.diag([s, s, s, 1.0])
    # volume_np is (X,Y,Z); nibabel expects (I,J,K) = (X,Y,Z) — no transpose needed
    return nib.Nifti1Image(volume_np, affine)


def extract_features_totalseg(volume_np, device="cuda",
                               fast=True,
                               hu_range=(0.0, 1.0)):
    """One-hot organ label features via TotalSegmentator.

    Runs TotalSegmentator's body-part segmentation on the volume and returns
    a one-hot feature volume with one channel per organ class (117 classes for
    the "total" task). Features are binary {0,1} indicating organ membership.

    The volume is assumed to be a normalised CT density (values in `hu_range`).
    It is linearly mapped to approximate HU [-1000, 3000] before inference so
    that TotalSegmentator's intensity normalisation works correctly.

    Args:
        volume_np:  (R, R, R) float32 — CT density / LAC values
        device:     "cuda" or "cpu"
        fast:       if True, use fast (3mm) model (task 297); else full (1.5mm)
        hu_range:   (lo, hi) expected value range of volume_np. Will be mapped
                    to [-1000, 3000] HU. Default (0, 1) for normalised volumes.

    Returns:
        (R, R, R, F) float16 numpy  where F = number of TotalSegmentator classes
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        from totalsegmentator.map_to_binary import class_map
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "TotalSegmentator is not installed.\n"
            "Run: pip install TotalSegmentator"
        )

    import tempfile, os

    R = volume_np.shape[0]
    lo, hi = hu_range
    hu_vol = ((volume_np - lo) / (hi - lo + 1e-8) * 4000.0 - 1000.0).astype(np.float32)

    # voxel_spacing: assume the volume covers ~[-1,1]^3 in world coords → 2m span
    # For a 256³ R2-Gaussian CT at typical 256mm FOV: spacing ≈ 1 mm
    voxel_spacing_mm = 1.0
    nii = _volume_to_nifti(hu_vol, voxel_spacing_mm=voxel_spacing_mm)

    with tempfile.TemporaryDirectory(prefix="totalseg_feat_") as tmpdir:
        out_dir = os.path.join(tmpdir, "segs")
        os.makedirs(out_dir, exist_ok=True)

        dev_str = "gpu" if str(device).startswith("cuda") else "cpu"
        print(f"  Running TotalSegmentator (fast={fast}, device={dev_str}) …")
        result_nii = totalsegmentator(
            nii, output=out_dir,
            task=_TOTALSEG_TASK,
            fast=fast,
            ml=True,          # multilabel output
            quiet=True,
            device=dev_str,
        )

        # result_nii is a multilabel NIfTI: integer voxels 0..N_classes
        if result_nii is None:
            ml_path = os.path.join(out_dir, "s01.nii.gz")
            if not os.path.exists(ml_path):
                raise RuntimeError(f"TotalSegmentator did not produce output at {ml_path}")
            result_nii = nib.load(ml_path)

        seg_data = np.asarray(result_nii.dataobj).astype(np.int16)   # (R,R,R)

    label_map = class_map[_TOTALSEG_TASK]          # {voxel_val: "organ_name"}
    class_ids = sorted(label_map.keys())            # e.g. 1..117
    F_dim = len(class_ids)
    print(f"  TotalSegmentator: {F_dim} organ classes, seg shape {seg_data.shape}")

    if seg_data.shape != (R, R, R):
        # Resample to original volume size if needed (shouldn't happen with ml=True)
        from skimage.transform import resize
        seg_resized = resize(seg_data.astype(np.float32), (R, R, R),
                             order=0, anti_aliasing=False, preserve_range=True)
        seg_data = seg_resized.astype(np.int16)

    feat_vol = np.zeros((R, R, R, F_dim), dtype=np.float16)
    for f_idx, cls_id in enumerate(class_ids):
        feat_vol[..., f_idx] = (seg_data == cls_id).astype(np.float16)

    return feat_vol


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_features(volume_np, backbone, device="cuda", **kwargs):
    """Dispatch to the correct backend.

    Args:
        volume_np: (R, R, R) float32, values in [0, 1] (DINO) or same range (TotalSeg maps internally)
        backbone:  one of the dino_* names or "totalsegmentator"
        device:    torch device string
        **kwargs:  forwarded to backend (axes, batch_size, fast, hu_range…)

    Returns:
        (R, R, R, F) float16 numpy
    """
    if backbone == "totalsegmentator":
        fast     = kwargs.pop("fast",     True)
        hu_range = kwargs.pop("hu_range", (0.0, 1.0))
        return extract_features_totalseg(volume_np, device=device,
                                         fast=fast, hu_range=hu_range)
    if backbone in _DINO_HUB_REPOS:
        axes = kwargs.pop("axes", ("axial",))
        return extract_features_dino(volume_np, backbone, axes=axes,
                                     device=device, **kwargs)
    if backbone in _DINOV3_HF_REPOS:
        axes = kwargs.pop("axes", ("axial",))
        return extract_features_dinov3(volume_np, backbone, axes=axes,
                                       device=device, **kwargs)
    all_choices = list(_DINO_HUB_REPOS) + list(_DINOV3_HF_REPOS) + ["totalsegmentator"]
    raise ValueError(f"Unknown backbone '{backbone}'. Choices: {all_choices}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract cell features from a checkpoint")
    parser.add_argument("--config",   required=True, help="Path to output/<run>/config.yaml")
    parser.add_argument("--backbone", default="dino_vits8",
                        choices=list(_DINO_HUB_REPOS) + list(_DINOV3_HF_REPOS) + ["totalsegmentator"],
                        help="Feature backbone (default: dino_vits8)")
    parser.add_argument("--axes",     nargs="+",
                        default=["axial"],
                        choices=["axial", "coronal", "sagittal"],
                        help="Axes to slice along for 2D backbones (default: axial)")
    parser.add_argument("--batch",    type=int, default=32,
                        help="Slices per GPU forward pass (default: 32)")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--fast",     action="store_true",
                        help="TotalSegmentator: use fast (3mm) model")
    parser.add_argument("--out",      default=None,
                        help="Output .npz path (default: <run_dir>/features_<backbone>.npz)")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data_path", "")
    dataset   = cfg.get("dataset", "r2_gaussian")

    if args.backbone == "totalsegmentator":
        out_fname = f"features_totalseg_{'fast' if args.fast else 'full'}.npz"
    elif args.backbone in _DINO_HUB_REPOS or args.backbone in _DINOV3_HF_REPOS:
        out_fname = f"features_{args.backbone}_{'_'.join(args.axes)}.npz"
    else:
        out_fname = f"features_{args.backbone}.npz"
    out_path = args.out or os.path.join(run_dir, out_fname)

    # Load GT volume
    gt_volume = load_gt_volume(data_path, dataset)
    if gt_volume is None:
        print("No GT volume found. Exiting.")
        sys.exit(1)
    print(f"GT volume: {gt_volume.shape}, "
          f"range [{gt_volume.min():.4f}, {gt_volume.max():.4f}]")

    # Normalize to [0, 1] for DINO-type backbones (TotalSeg does its own HU mapping)
    if args.backbone != "totalsegmentator":
        lo, hi = gt_volume.min(), gt_volume.max()
        if hi > lo:
            gt_volume = (gt_volume - lo) / (hi - lo)

    # Extract feature volume
    print(f"\nExtracting features with {args.backbone} (axes={args.axes}) …")
    extra = {}
    if args.backbone in _DINO_HUB_REPOS:
        extra["axes"] = tuple(args.axes)
        extra["batch_size"] = args.batch
    elif args.backbone in _DINOV3_HF_REPOS:
        extra["axes"] = tuple(args.axes)
        extra["batch_size"] = args.batch
    elif args.backbone == "totalsegmentator":
        extra["fast"] = args.fast
    feat_vol = extract_features(gt_volume, args.backbone, device=args.device, **extra)
    R, _, _, F_dim = feat_vol.shape
    print(f"Feature volume: {R}³ × {F_dim}  "
          f"({feat_vol.nbytes / 1e9:.2f} GB float16)")

    # Load model checkpoint and assign features to cells
    model_pt = os.path.join(run_dir, "model.pt")
    if not os.path.exists(model_pt):
        print(f"model.pt not found at {model_pt}. Saving feature volume only.")
        np.save(out_path.replace(".npz", "_volume.npy"), feat_vol)
        print(f"Saved feature volume to {out_path.replace('.npz', '_volume.npy')}")
        return

    print(f"Loading checkpoint: {model_pt}")
    field = load_density_field(model_pt, device=args.device)
    points = field["points"]
    print(f"Points: {points.shape[0]:,}")

    cell_feats = assign_cell_features(points, feat_vol)
    print(f"Cell features: {cell_feats.shape}  dtype={cell_feats.dtype}")

    save_cell_features(out_path, cell_feats, meta={
        "backbone": args.backbone,
        "axes": "_".join(args.axes),
        "n_cells": points.shape[0],
        "feat_dim": F_dim,
        "run_dir": run_dir,
    })
    print(f"Saved: {out_path}  "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
