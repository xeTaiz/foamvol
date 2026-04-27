"""Generate NEMA NU-2 IEC body phantom as 256^3 float32 volume.

Simplified analytical model:
  - Elliptical cylinder background at density 0.3
  - 6 hot spheres (diameters: 10, 13, 17, 22, 28, 37 mm) at density 0.8
    arranged in a ring at standard NEMA IEC positions

Physical dimensions of the phantom body:
  - Elliptical cross-section: a=147 mm, b=174 mm (IQ cylinder)
  - Length: 180 mm
  - Sphere ring radius: 57.2 mm from center, positioned at mid-height

All coordinates normalised to the unit cube [-1, 1]^3.
Physical scale: 300 mm per side (so 1 unit = 150 mm).
"""

import argparse
import os
import numpy as np

# Physical dimensions (mm), will be normalised to [-1,1] where 1 unit = 150 mm
PHYS_SCALE = 150.0

# IEC body phantom ellipse
BODY_A_MM = 147.0  # semi-axis x
BODY_B_MM = 174.0  # semi-axis y
BODY_LEN_MM = 180.0  # half-length z

# Sphere diameters (mm) and ring radius
SPHERE_DIAMETERS_MM = [10.0, 13.0, 17.0, 22.0, 28.0, 37.0]
SPHERE_RING_RADIUS_MM = 57.2
SPHERE_Z_MM = 0.0  # at mid-height

# Density values
RHO_BACKGROUND = 0.30
RHO_SPHERE = 0.80


def generate(n=256):
    vol = np.zeros((n, n, n), dtype=np.float32)
    lin = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

    # Normalise physical dims to unit-cube coords
    a = BODY_A_MM / PHYS_SCALE
    b = BODY_B_MM / PHYS_SCALE
    half_z = BODY_LEN_MM / PHYS_SCALE

    # Elliptical cylinder
    in_cylinder = ((X / a) ** 2 + (Y / b) ** 2 <= 1.0) & (np.abs(Z) <= half_z)
    vol[in_cylinder] = RHO_BACKGROUND

    # Hot spheres at 60° intervals starting at 0°
    ring_r = SPHERE_RING_RADIUS_MM / PHYS_SCALE
    sz = SPHERE_Z_MM / PHYS_SCALE
    angles = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    for angle, diam_mm in zip(angles, SPHERE_DIAMETERS_MM):
        cx = ring_r * np.cos(angle)
        cy = ring_r * np.sin(angle)
        radius = (diam_mm / 2.0) / PHYS_SCALE
        in_sphere = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - sz) ** 2 <= radius ** 2
        vol[in_sphere] = RHO_SPHERE

    return vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/mnt/hdd/r2_data/debug_phantoms/volumes/nema_iec.npy")
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    vol = generate(args.n)
    np.save(args.output, vol)
    print(f"Saved {args.output}  shape={vol.shape}  min={vol.min():.3f}  max={vol.max():.3f}")


if __name__ == "__main__":
    main()
