"""Visualize slices of a voxelized CT reconstruction.

Produces a 3x3 grid: rows = X/Y/Z axis, columns = slices at -0.5, 0.0, +0.5
(in normalized [-1,1] coordinates).

Usage:
    python visualize_volume.py volume.npy
    python visualize_volume.py volume.npy --gt vol_gt.npy
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def coord_to_index(coord, resolution):
    """Map coordinate in [-1, 1] to voxel index."""
    return int((coord + 1) / 2 * (resolution - 1))


def main():
    parser = argparse.ArgumentParser(description="Visualize volume slices")
    parser.add_argument("volume", type=str, help="Path to volume.npy")
    parser.add_argument("--gt", type=str, default=None, help="Optional ground truth volume.npy")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max (auto if not set)")
    args = parser.parse_args()

    vol = np.load(args.volume)
    res = vol.shape[0]
    vmax = args.vmax if args.vmax is not None else vol.max()

    axes = ["X", "Y", "Z"]
    coords = [-0.5, 0.0, 0.5]

    fig, axs = plt.subplots(3, 3, figsize=(9, 9))

    for row, axis in enumerate(axes):
        for col, c in enumerate(coords):
            idx = coord_to_index(c, res)
            if axis == "X":
                slc = vol[idx, :, :]
            elif axis == "Y":
                slc = vol[:, idx, :]
            else:
                slc = vol[:, :, idx]

            ax = axs[row, col]
            ax.imshow(slc.T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(f"{axis}={c:.1f} (i={idx})")
            ax.axis("off")

    fig.suptitle(os.path.basename(args.volume), fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(args.volume), "vis.jpg")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
