"""Generate 3D Shepp-Logan head phantom as 256^3 float32 volume.

Uses Toft's 3D extension of the classic Shepp-Logan phantom (10 ellipsoids).
Output values are in [0, 1] (rescaled from [-0.8, 1.0]).
Volume is saved as (X, Y, Z) = (256, 256, 256) float32.
"""

import argparse
import os
import numpy as np


# Toft's 3D Shepp-Logan table.
# Each row: (a, b, c, x0, y0, z0, phi_deg, theta_deg, psi_deg, rho)
# (a, b, c) are semi-axes; (x0, y0, z0) is center; angles rotate the ellipsoid; rho is density.
SHEPP_LOGAN_TABLE = np.array([
    # a      b      c      x0      y0     z0     phi   theta  psi    rho
    [0.6900, 0.920, 0.810,  0.000,  0.000, 0.000, 0.0,  0.0,   0.0,   2.00],
    [0.6624, 0.874, 0.780,  0.000, -0.184, 0.000, 0.0,  0.0,   0.0,  -0.98],
    [0.4100, 0.160, 0.210, -0.220,  0.000, 0.000, 108., 0.0,   0.0,  -0.02],
    [0.3100, 0.110, 0.220,  0.220,  0.000, 0.000, 72.,  0.0,   0.0,  -0.02],
    [0.2100, 0.250, 0.500,  0.000,  0.350, -0.150, 0.0,  0.0,   0.0,   0.01],
    [0.0460, 0.046, 0.046,  0.000,  0.100, 0.250, 0.0,  0.0,   0.0,   0.01],
    [0.0460, 0.023, 0.020, -0.080, -0.605, 0.000, 0.0,  0.0,   0.0,   0.01],
    [0.0460, 0.023, 0.020,  0.060, -0.605, 0.000, 90.,  0.0,   0.0,   0.01],
    [0.0560, 0.040, 0.100,  0.060, -0.105, 0.625, 90.,  0.0,   0.0,   0.02],
    [0.0560, 0.056, 0.100,  0.000,  0.100, 0.625, 0.0,  0.0,   0.0,  -0.02],
], dtype=np.float64)


def rotation_matrix(phi, theta, psi):
    """Euler rotation (ZYZ convention) in radians."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)
    Rz1 = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]], dtype=np.float64)
    Ry  = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]], dtype=np.float64)
    Rz2 = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]], dtype=np.float64)
    return Rz2 @ Ry @ Rz1


def generate(n=256):
    vol = np.zeros((n, n, n), dtype=np.float64)
    lin = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")  # [n, n, n] each

    for row in SHEPP_LOGAN_TABLE:
        a, b, c, x0, y0, z0, phi_d, theta_d, psi_d, rho = row
        phi, theta, psi = np.radians(phi_d), np.radians(theta_d), np.radians(psi_d)
        R = rotation_matrix(phi, theta, psi)

        dx = X - x0
        dy = Y - y0
        dz = Z - z0
        # Rotate the difference vector
        xr = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
        yr = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
        zr = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz

        inside = (xr / a) ** 2 + (yr / b) ** 2 + (zr / c) ** 2 <= 1.0
        vol[inside] += rho

    # Rescale to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    vol = (vol - vmin) / (vmax - vmin)
    return vol.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/mnt/hdd/r2_data/debug_phantoms/volumes/shepp_logan.npy")
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    vol = generate(args.n)
    np.save(args.output, vol)
    print(f"Saved {args.output}  shape={vol.shape}  min={vol.min():.3f}  max={vol.max():.3f}")


if __name__ == "__main__":
    main()
