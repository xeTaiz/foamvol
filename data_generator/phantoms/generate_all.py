"""Drive projection generation for all debug phantoms.

For each (phantom, n_train, noise) combination this script:
  1. Checks whether the output directory already has meta_data.json (idempotent).
  2. Invokes generate_data.py inside the r2_gaussian docker container.

Volume files must already exist (run shepp_logan_3d.py / nema_iec.py /
marschner_lobb.py first).

Usage (from the radfoam repo root):
    python data_generator/phantoms/generate_all.py
    python data_generator/phantoms/generate_all.py --dry-run
    python data_generator/phantoms/generate_all.py --phantoms shepp_logan nema_iec
    python data_generator/phantoms/generate_all.py --n-train 75 --noise clean
"""

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

R2_REPO = "/home/engeld/Dev/r2_gaussian"
DATA_ROOT = "/mnt/hdd/r2_data/debug_phantoms"
DOCKER_IMAGE = "xetaiz/r2:latest"

PHANTOMS = ["shepp_logan", "nema_iec", "marschner_lobb"]

SCANNER_MAP = {
    "clean": "/data/debug_phantoms/scanner/cone_beam_clean.yml",
    "low":   "/data/debug_phantoms/scanner/cone_beam_low.yml",
    "mid":   "/data/debug_phantoms/scanner/cone_beam_mid.yml",
}

VARIANTS = [
    (500, "clean"),
    (500, "low"),
    (500, "mid"),
    (75,  "clean"),
    (75,  "low"),
    (75,  "mid"),
]

N_TEST = 100


def run_variant(phantom, n_train, noise_label, dry_run=False):
    out_dir_name = f"{phantom}_n{n_train}_{noise_label}"
    out_host = os.path.join(DATA_ROOT, out_dir_name)
    # generate_data.py appends <phantom>_cone/ as a case subdirectory
    case_subdir = os.path.join(out_host, f"{phantom}_cone")
    meta_path = os.path.join(case_subdir, "meta_data.json")

    if os.path.exists(meta_path):
        print(f"[SKIP]  {out_dir_name}  (already exists)")
        return

    vol_docker = f"/data/debug_phantoms/volumes/{phantom}.npy"
    scanner_docker = SCANNER_MAP[noise_label]
    out_docker = f"/data/debug_phantoms/{out_dir_name}"

    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "-v", f"{R2_REPO}:/workspace/r2_gaussian",
        "-v", "/mnt/hdd/r2_data:/data",
        "-w", "/workspace/r2_gaussian",
        DOCKER_IMAGE,
        "python", "data_generator/synthetic_dataset/generate_data.py",
        "--vol",     vol_docker,
        "--scanner", scanner_docker,
        "--output",  out_docker,
        "--n_train", str(n_train),
        "--n_test",  str(N_TEST),
    ]

    print(f"[RUN]   {out_dir_name}")
    if dry_run:
        print("  " + " ".join(cmd))
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: generation failed for {out_dir_name}", file=sys.stderr)
        sys.exit(1)


def copy_scanner_configs():
    """Copy scanner YAMLs to /mnt/hdd/r2_data/debug_phantoms/scanner/ for docker access."""
    src_dir = os.path.join(os.path.dirname(__file__), "scanner")
    dst_dir = os.path.join(DATA_ROOT, "scanner")
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.endswith(".yml"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            with open(src) as f:
                content = f.read()
            with open(dst, "w") as f:
                f.write(content)
    print(f"Scanner YAMLs copied to {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phantoms", nargs="+", default=PHANTOMS, choices=PHANTOMS)
    parser.add_argument("--n-train", type=int, default=None, help="Only run this n_train value")
    parser.add_argument("--noise", default=None, choices=list(SCANNER_MAP.keys()), help="Only run this noise variant")
    args = parser.parse_args()

    copy_scanner_configs()

    variants = [
        (n, noise) for (n, noise) in VARIANTS
        if (args.n_train is None or n == args.n_train)
        and (args.noise is None or noise == args.noise)
    ]

    for phantom in args.phantoms:
        vol_path = os.path.join(DATA_ROOT, "volumes", f"{phantom}.npy")
        if not os.path.exists(vol_path):
            print(f"ERROR: volume not found: {vol_path}  — run the phantom generator first", file=sys.stderr)
            sys.exit(1)

        for n_train, noise_label in variants:
            run_variant(phantom, n_train, noise_label, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
