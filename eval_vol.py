"""Standalone volume metric evaluation script.

Computes all metrics used at end-of-training against a ground-truth volume:
  PSNR, SSIM (2D slice-wise + 3D), Sobel-filtered PSNR/SSIM,
  Dice coefficient, Chamfer / Hausdorff / F1 surface metrics.

Usage:
  # Single pair
  python eval_vol.py vol_pred.npy vol_gt.npy

  # Recursive search: finds every folder containing both files
  python eval_vol.py --scan /path/to/output/dir

  # Custom filenames
  python eval_vol.py --scan /path/to/output/dir --pred-name ct_pred.npy --gt-name vol_gt.npy
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
from scipy.spatial import KDTree


# ─── metric helpers ──────────────────────────────────────────────────────────

def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float().to(device)


def compute_volume_psnr(pred, gt):
    pred, gt = pred.float(), gt.float()
    pixel_max = gt.max()
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(pixel_max ** 2 / mse)).item()


@torch.no_grad()
def compute_volume_ssim(pred, gt, window_size=11):
    """Slice-wise 2D SSIM averaged over 3 axes (R2-Gaussian convention)."""
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    axis_ssims = []
    for axis in range(3):
        n = pred.shape[axis]
        ssim_sum, count = 0.0, 0
        for i in range(n):
            if axis == 0:
                s_pred, s_gt = pred[i], gt[i]
            elif axis == 1:
                s_pred, s_gt = pred[:, i], gt[:, i]
            else:
                s_pred, s_gt = pred[:, :, i], gt[:, :, i]
            if s_gt.max() <= 0:
                continue
            img1 = s_pred.unsqueeze(0).unsqueeze(0)
            img2 = s_gt.unsqueeze(0).unsqueeze(0)
            mu1 = F.conv2d(img1, kernel, padding=pad)
            mu2 = F.conv2d(img2, kernel, padding=pad)
            s1 = F.conv2d(img1**2, kernel, padding=pad) - mu1**2
            s2 = F.conv2d(img2**2, kernel, padding=pad) - mu2**2
            s12 = F.conv2d(img1 * img2, kernel, padding=pad) - mu1 * mu2
            ssim_map = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
            ssim_sum += ssim_map.mean().item()
            count += 1
        axis_ssims.append(ssim_sum / count if count > 0 else 0.0)
    return float(np.mean(axis_ssims)), axis_ssims


def _gauss_conv3d_separable(x, gauss_1d, pad):
    ws = gauss_1d.shape[0]
    kx = gauss_1d.reshape(1, 1, ws, 1, 1)
    ky = gauss_1d.reshape(1, 1, 1, ws, 1)
    kz = gauss_1d.reshape(1, 1, 1, 1, ws)
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, pad, pad), mode="replicate"), kx)
    x = F.conv3d(F.pad(x, (0, 0, pad, pad, 0, 0), mode="replicate"), ky)
    x = F.conv3d(F.pad(x, (pad, pad, 0, 0, 0, 0), mode="replicate"), kz)
    return x


@torch.no_grad()
def compute_volume_ssim_3d(pred, gt, window_size=11):
    """True 3D SSIM using separable Gaussian kernel."""
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    gauss_1d = torch.exp(-coords**2 / (2 * sigma**2))
    gauss_1d /= gauss_1d.sum()
    pad = window_size // 2
    p = pred.unsqueeze(0).unsqueeze(0)
    g = gt.unsqueeze(0).unsqueeze(0)
    data_range = gt.max() - gt.min()
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2
    mu1 = _gauss_conv3d_separable(p, gauss_1d, pad)
    mu2 = _gauss_conv3d_separable(g, gauss_1d, pad)
    s1 = _gauss_conv3d_separable(p**2, gauss_1d, pad) - mu1**2
    s2 = _gauss_conv3d_separable(g**2, gauss_1d, pad) - mu2**2
    s12 = _gauss_conv3d_separable(p * g, gauss_1d, pad) - mu1 * mu2
    ssim_map = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


def sobel_filter_3d(vol):
    v = vol.unsqueeze(0).unsqueeze(0)
    v = F.pad(v, (1,1,1,1,1,1), mode="replicate")
    smooth = torch.tensor([1, 2, 1], dtype=torch.float32, device=vol.device)
    diff   = torch.tensor([-1, 0, 1], dtype=torch.float32, device=vol.device)
    kx = (smooth[:,None,None] * smooth[None,:,None] * diff[None,None,:]).reshape(1,1,3,3,3)
    ky = (smooth[:,None,None] * diff[None,:,None]   * smooth[None,None,:]).reshape(1,1,3,3,3)
    kz = (diff[:,None,None]   * smooth[None,:,None] * smooth[None,None,:]).reshape(1,1,3,3,3)
    gx = F.conv3d(v, kx)
    gy = F.conv3d(v, ky)
    gz = F.conv3d(v, kz)
    return torch.log1p(torch.sqrt(gx**2 + gy**2 + gz**2)).clamp(0, 1).squeeze()


@torch.no_grad()
def compute_dice(pred, gt, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    dices = {}
    for t in thresholds:
        p, g = pred > t, gt > t
        inter = (p & g).sum()
        dices[t] = 2 * inter / (p.sum() + g.sum() + 1e-8)
    return float(np.mean(list(dices.values()))), dices


def _surface_metrics_at_level(pred, gt, level, f_thresholds=(1.0, 2.0)):
    try:
        verts_p, _, _, _ = marching_cubes(pred, level=level)
        verts_g, _, _, _ = marching_cubes(gt, level=level)
    except (ValueError, RuntimeError):
        return None
    if len(verts_p) < 3 or len(verts_g) < 3:
        return None
    tree_p = KDTree(verts_p)
    tree_g = KDTree(verts_g)
    d_p2g, _ = tree_g.query(verts_p)
    d_g2p, _ = tree_p.query(verts_g)
    chamfer = 0.5 * (d_p2g.mean() + d_g2p.mean())
    hausdorff = float(max(d_p2g.max(), d_g2p.max()))
    hausdorff_95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
    result = {"chamfer": float(chamfer), "hausdorff": hausdorff, "hausdorff_95": hausdorff_95}
    for d in f_thresholds:
        prec = (d_p2g <= d).mean()
        rec  = (d_g2p <= d).mean()
        result[f"f1_{d:.0f}v"] = float(2 * prec * rec / (prec + rec + 1e-8))
    return result


@torch.no_grad()
def compute_surface_metrics(pred, gt,
                            thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            f_thresholds=(1.0, 2.0)):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    per_level = [m for t in thresholds
                 if (m := _surface_metrics_at_level(pred, gt, t, f_thresholds)) is not None]
    if not per_level:
        return {"chamfer": float("inf"), "hausdorff": float("inf"),
                "hausdorff_95": float("inf"),
                **{f"f1_{d:.0f}v": 0.0 for d in f_thresholds}}
    keys = per_level[0].keys()
    return {k: float(np.mean([m[k] for m in per_level])) for k in keys}


# ─── core evaluation ─────────────────────────────────────────────────────────

def evaluate(pred_path, gt_path, device="cuda"):
    pred_np = np.load(pred_path).astype(np.float32)
    gt_np   = np.load(gt_path).astype(np.float32)

    if pred_np.shape != gt_np.shape:
        print(f"  [WARN] shape mismatch: pred={pred_np.shape}, gt={gt_np.shape} — skipping")
        return None

    pred = _to_tensor(pred_np, device)
    gt   = _to_tensor(gt_np,   device)

    results = {}

    # PSNR / SSIM
    results["psnr"]   = compute_volume_psnr(pred, gt)
    results["ssim"], ax_ssims = compute_volume_ssim(pred, gt)
    results["ssim_x"], results["ssim_y"], results["ssim_z"] = ax_ssims
    results["ssim_3d"] = compute_volume_ssim_3d(pred, gt)

    # Sobel
    gt_sobel   = sobel_filter_3d(gt)
    pred_sobel = sobel_filter_3d(pred)
    results["sobel_psnr"]   = compute_volume_psnr(pred_sobel, gt_sobel)
    results["sobel_ssim"], _ = compute_volume_ssim(pred_sobel, gt_sobel)

    # Dice
    results["dice"], _ = compute_dice(pred, gt)

    # Surface metrics (Chamfer / Hausdorff / F1)
    print("  computing surface metrics (marching cubes)...")
    surf = compute_surface_metrics(pred, gt)
    results.update(surf)

    return results


def print_results(label, r):
    if r is None:
        return
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  PSNR:        {r['psnr']:.4f}")
    print(f"  SSIM (2D):   {r['ssim']:.6f}  (X={r['ssim_x']:.4f}  Y={r['ssim_y']:.4f}  Z={r['ssim_z']:.4f})")
    print(f"  SSIM (3D):   {r['ssim_3d']:.6f}")
    print(f"  Sobel PSNR:  {r['sobel_psnr']:.4f}")
    print(f"  Sobel SSIM:  {r['sobel_ssim']:.6f}")
    print(f"  Dice:        {r['dice']:.6f}")
    print(f"  Chamfer:     {r['chamfer']:.4f} v")
    print(f"  Hausdorff:   {r['hausdorff']:.4f} v")
    print(f"  HD95:        {r['hausdorff_95']:.4f} v")
    print(f"  F1@1v:       {r['f1_1v']:.4f}")
    print(f"  F1@2v:       {r['f1_2v']:.4f}")


def save_results(out_path, label, r):
    with open(out_path, "w") as f:
        f.write(f"# {label}\n")
        for k, v in r.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"  saved → {out_path}")


# ─── search ──────────────────────────────────────────────────────────────────

def find_pairs(root, pred_name, gt_name):
    pairs = []
    for dirpath, _, files in os.walk(root):
        if pred_name in files and gt_name in files:
            pairs.append((
                os.path.join(dirpath, pred_name),
                os.path.join(dirpath, gt_name),
                dirpath,
            ))
    return sorted(pairs)


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate vol_pred.npy vs vol_gt.npy")
    p.add_argument("files", nargs="*", metavar="FILE",
                   help="vol_pred.npy vol_gt.npy (positional pair)")
    p.add_argument("--scan", metavar="DIR",
                   help="Recursively scan DIR for matching file pairs")
    p.add_argument("--pred-name", default="vol_pred.npy")
    p.add_argument("--gt-name",   default="vol_gt.npy")
    p.add_argument("--cpu", action="store_true", help="Force CPU (default: CUDA if available)")
    p.add_argument("--save", action="store_true", help="Write metrics.txt next to each pred file")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip folders where vol_metrics.txt already exists")
    args = p.parse_args()

    if args.scan and args.files:
        p.error("Use either positional files or --scan, not both")
    if not args.scan and not args.files:
        p.error("Provide either two positional files or --scan DIR")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device: {device}")

    if args.scan:
        pairs = find_pairs(args.scan, args.pred_name, args.gt_name)
        if not pairs:
            print(f"No folders found under {args.scan} containing both "
                  f"{args.pred_name} and {args.gt_name}")
            sys.exit(1)
        import random
        random.shuffle(pairs)
        print(f"Found {len(pairs)} pair(s) under {args.scan}")
        for pred_path, gt_path, folder in pairs:
            out = os.path.join(folder, "vol_metrics.txt")
            if args.skip_existing and os.path.exists(out):
                print(f"\n[{folder}] skipped (vol_metrics.txt exists)")
                continue
            print(f"\n[{folder}]")
            r = evaluate(pred_path, gt_path, device=device)
            print_results(folder, r)
            if args.save and r is not None:
                save_results(out, folder, r)
    else:
        if len(args.files) != 2:
            p.error("Provide exactly two positional arguments: vol_pred.npy vol_gt.npy")
        pred_path, gt_path = args.files
        r = evaluate(pred_path, gt_path, device=device)
        print_results(f"{pred_path} vs {gt_path}", r)
        if args.save and r is not None:
            out = os.path.splitext(pred_path)[0] + "_metrics.txt"
            save_results(out, f"{pred_path} vs {gt_path}", r)


if __name__ == "__main__":
    main()
