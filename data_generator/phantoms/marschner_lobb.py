"""Generate Marschner-Lobb phantom as 256^3 float32 volume.

Standard analytical signal (Marschner & Lobb, 1994):
  rho_r(r) = cos(2 * pi * f_M * cos(pi * r / 2))
  ML(x,y,z) = (1 - sin(pi*z/2) + alpha*(1 + rho_r(sqrt(x^2+y^2)))) / (2*(1+alpha))

with alpha=0.25 and f_M=6, sampled on [-1,1]^3.

Values are analytically in [0,1] so no rescaling needed.
"""

import argparse
import os
import numpy as np


ALPHA = 0.25
FM = 6.0


def generate(n=256):
    lin = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

    r = np.sqrt(X ** 2 + Y ** 2)
    rho_r = np.cos(2.0 * np.pi * FM * np.cos(np.pi * r / 2.0))
    vol = (1.0 - np.sin(np.pi * Z / 2.0) + ALPHA * (1.0 + rho_r)) / (2.0 * (1.0 + ALPHA))
    vol = np.clip(vol, 0.0, 1.0).astype(np.float32)
    return vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/mnt/hdd/r2_data/debug_phantoms/volumes/marschner_lobb.npy")
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    vol = generate(args.n)
    np.save(args.output, vol)
    print(f"Saved {args.output}  shape={vol.shape}  min={vol.min():.3f}  max={vol.max():.3f}")


if __name__ == "__main__":
    main()
