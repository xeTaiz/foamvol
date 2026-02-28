"""Sigma scale investigation: visualize how different IDW sigma values affect slice quality."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import radfoam
from vis_foam import (
    load_density_field,
    make_slice_coords,
    query_density,
    sample_idw,
    supersample_slice,
)


def median_cell_radius(field):
    """Compute median cell radius from a density field dict."""
    _, cell_radius = radfoam.farthest_neighbor(
        field["points"], field["adjacency"], field["adjacency_offsets"]
    )
    return cell_radius.median().item()


def single_slice(field, mcr, axis, coord, sigma_scale, sigma_v,
                 resolution, ss, extent, vmax, out):
    """Generate NN vs IDW vs difference for a single slice."""
    sigma = sigma_scale * mcr
    axis_name = "XYZ"[axis]

    nn_slice = supersample_slice(
        query_density, field, axis, coord, resolution, extent, ss=ss,
    )
    idw_slice = supersample_slice(
        sample_idw, field, axis, coord, resolution, extent, ss=ss,
        sigma=sigma, sigma_v=sigma_v,
    )
    diff = nn_slice - idw_slice

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(nn_slice.T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title(f"NN  {axis_name}={coord}")
    axes[0].axis("off")

    axes[1].imshow(idw_slice.T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    axes[1].set_title(f"IDW  {axis_name}={coord}  scale={sigma_scale}")
    axes[1].axis("off")

    abs_max = max(np.abs(diff).max(), 1e-6)
    axes[2].imshow(diff.T, origin="lower", cmap="bwr", vmin=-abs_max, vmax=abs_max)
    axes[2].set_title("Diff (NN - IDW)")
    axes[2].axis("off")

    fig.suptitle(f"sigma = {sigma:.6f}  (scale={sigma_scale} x mcr={mcr:.6f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}  (sigma={sigma:.6f})")


def comparison_grid(field, mcr, sigma_v, resolution, ss, extent, vmax, out):
    """Generate 9x8 grid: 9 slices x (6 global sigma + 2 per-cell sigma)."""
    # (scale, label, per_cell_sigma, per_neighbor_sigma)
    columns = [
        (None,  "No interp",     False, False),
        (2.0,   "scale=2.0",    False, False),
        (1.0,   "scale=1.0",    False, False),
        (0.5,   "scale=0.5",    False, False),
        (0.25,  "scale=0.25",   False, False),
        (0.1,   "scale=0.1",    False, False),
        (0.5,   "cell σ (A)",   True,  False),
        (0.5,   "cell σ (B)",   True,  True),
    ]

    slice_specs = []
    for axis in range(3):
        for coord in [-0.2, 0.0, 0.2]:
            slice_specs.append((axis, coord))

    nrows = len(slice_specs)   # 9
    ncols = len(columns)       # 8
    axis_names = "XYZ"

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    for ri, (axis, coord) in enumerate(slice_specs):
        for ci, (scale, label, pcs, pns) in enumerate(columns):
            ax = axes[ri, ci]
            if scale is None:
                sl = supersample_slice(
                    query_density, field, axis, coord, resolution, extent, ss=ss,
                )
            elif pcs:
                # per-cell sigma: pass scale directly (not scale*mcr)
                sl = supersample_slice(
                    sample_idw, field, axis, coord, resolution, extent, ss=ss,
                    sigma=scale, sigma_v=sigma_v,
                    per_cell_sigma=True, per_neighbor_sigma=pns,
                )
            else:
                sigma = scale * mcr
                sl = supersample_slice(
                    sample_idw, field, axis, coord, resolution, extent, ss=ss,
                    sigma=sigma, sigma_v=sigma_v,
                )
            ax.imshow(sl.T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(f"{axis_names[axis]}={coord:.1f} {label}", fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"Sigma scale comparison  (mcr={mcr:.6f}, sigma_v={sigma_v})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["single", "grid", "both"], default="both")
    parser.add_argument("--axis", type=int, default=0, help="Slice axis (0=X, 1=Y, 2=Z)")
    parser.add_argument("--coord", type=float, default=0.2, help="Slice coordinate")
    parser.add_argument("--sigma-scale", type=float, default=0.5)
    parser.add_argument("--sigma-v", type=float, default=0.1, help="Bilateral scale (0 to disable)")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--ss", type=int, default=2, help="Supersample factor")
    parser.add_argument("--extent", type=float, default=1.0)
    parser.add_argument("--vmax", type=float, default=1.2)
    parser.add_argument("--out-single", default="sigma_single.png")
    parser.add_argument("--out-grid", default="sigma_grid.png")
    args = parser.parse_args()

    sigma_v = args.sigma_v if args.sigma_v > 0 else None

    print(f"Loading {args.model} ...")
    field = load_density_field(args.model)
    mcr = median_cell_radius(field)
    print(f"Points:             {field['points'].shape[0]:,}")
    print(f"Median cell radius: {mcr:.6f}")

    if args.mode in ("single", "both"):
        single_slice(field, mcr, args.axis, args.coord, args.sigma_scale,
                     sigma_v, args.resolution, args.ss, args.extent,
                     args.vmax, args.out_single)

    if args.mode in ("grid", "both"):
        comparison_grid(field, mcr, sigma_v, args.resolution, args.ss,
                        args.extent, args.vmax, args.out_grid)


if __name__ == "__main__":
    main()
