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


def visualize(volume_path, vmax=1.0):
    vol = np.load(volume_path)
    res = vol.shape[0]
    if vmax is None:
        vmax = vol.max()

    axes = ["X", "Y", "Z"]
    coords = [-0.2, 0.0, 0.2]

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

    fig.suptitle(os.path.basename(volume_path), fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(volume_path), "vis.jpg")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize volume slices")
    parser.add_argument("volume", type=str, help="Path to volume.npy")
    parser.add_argument("--gt", type=str, default=None, help="Optional ground truth volume.npy")
    parser.add_argument("--vmax", type=float, default=1.0, help="Color scale max (auto if not set)")
    args = parser.parse_args()
    visualize(args.volume, vmax=args.vmax)


if __name__ == "__main__":
    main()
