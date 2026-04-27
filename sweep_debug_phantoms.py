#!/usr/bin/env python3
"""Debug phantom sweep — grittiness / noise investigation.

Runs radfoam on three analytical phantoms (Shepp-Logan, NEMA IEC, Marschner-Lobb)
across all simulated noise/projection-count variants, plus the two real-scanner
datasets (Siemens Inveon NEMA and TCIA ACR phantom).

Purpose: isolate whether reconstruction artefacts arise from the representation,
the optimisation, undersampling, or measurement noise.

Simulated variants (18 runs):
  Phantoms : shepp_logan, nema_iec, marschner_lobb
  Variants : n500_clean, n500_low, n500_mid, n75_clean, n75_low, n75_mid

Real-scanner datasets (2 runs):
  inveon_nema — Siemens Inveon preclinical cone-beam CT of NEMA IQ phantom
  acr_phantom — Siemens SOMATOM helical fan-beam CT of ACR phantom (TCIA LDCT)

Total: 20 runs.

Usage:
    python sweep_debug_phantoms.py --list
    python sweep_debug_phantoms.py
    python sweep_debug_phantoms.py --runs shepp_logan-n75_clean inveon_nema
    python sweep_debug_phantoms.py --summarize
    python sweep_debug_phantoms.py --worker 1 --of 4
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep_debug_phantoms"

PHANTOM_BASE = "/mnt/hdd/r2_data/debug_phantoms"

INVEON_PREFIX = (
    "/mnt/hdd/r2_data/debug_phantoms/real_scans/"
    "NEMA_S6_18F_Inveon_190M_counts_XAT_Attenuation-CT_NEMA_s6_v1"
)
ACR_DIR = (
    "/mnt/hdd/r2_data/debug_phantoms/real_scans/acr_phantom/"
    "1.3.6.1.4.1.9590.100.1.2.79685099111720316617921522283988355980"
)

# ── Base configs ──────────────────────────────────────────────────────────────

def _common_base():
    """Shared hyperparams for all runs."""
    return {
        "iterations": 10000,
        "rays_per_batch_late": 2_000_000,
        "rays_per_batch_late_start": 9000,
        "activation_scale": 1.0,
        "init_scale": 1.05,
        "init_type": "random",
        "init_density": 2.0,
        "init_volume_path": "",
        "device": "cuda",
        "debug": False,
        "viewer": False,
        "save_volume": False,
        "loss_type": "l1",
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "density_grad_clip": 10.0,
        "freeze_points": 9500,
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "gradient_fraction": 0.4,
        "idw_fraction": 0.3,
        "entropy_fraction": 0.3,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.03,
        "redundancy_cap_init": 0.0,
        "redundancy_cap_final": 0.0,
        "prune_variance_criterion": False,
        "prune_hops": 1,
        "ref_guided_pruning": False,
        "ref_guided_densify": False,
        "ref_guided_eps": 0.01,
        "targeted_fraction": 0.1,
        "targeted_start": -1,
        "high_error_fraction": 0.2,
        "high_error_power": 1.0,
        "high_error_start": -1,
        "tv_weight": 0.0,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        "voxel_var_weight": 0.0,
        "voxel_var_weight_final": -1.0,
        "voxel_var_resolution": 64,
        "voxel_var_start": 0,
        "voxel_var_supersample": 4,
        "neighbor_var_weight": 0.0,
        "neighbor_var_weight_final": -1.0,
        "neighbor_var_hops": 1,
        "neighbor_var_start": 0,
        "neighbor_reg_type": "bilateral_var",
        "neighbor_huber_delta": 0.1,
        "var_sigma_v_init": 50.0,
        "var_sigma_v_final": 0.2,
        "grad_smooth_hops": 1,
        "ref_volume_path": "",
        "ref_volume_weight": 0.0,
        "ref_volume_weight_final": -1.0,
        "ref_volume_start": 0,
        "ref_volume_until": -1,
        "ref_volume_resolution": 64,
        "ref_volume_blur_sigma": 0.0,
        "ref_volume_edge_mask": True,
        "ref_volume_edge_alpha": 10.0,
        "ref_volume_supersample": 4,
        "interpolation_start": 9000,
        "interp_ramp": False,
        "interp_sigma_scale": 0.7,
        "interp_sigma_v": 0.2,
        "per_cell_sigma": False,
        "per_neighbor_sigma": False,
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        "gaussian_start": -1,
        "freeze_base_at_gaussian": False,
        "joint_finetune_start": -1,
        "peak_lr_init": 1e-2,
        "peak_lr_final": 1e-3,
        "offset_lr_init": 1e-3,
        "offset_lr_final": 1e-4,
        "cov_lr_init": 1e-2,
        "cov_lr_final": 1e-3,
        "gradient_start": -1,
        "gradient_lr_init": 1e-2,
        "gradient_lr_final": 1e-3,
        "gradient_warmup": 500,
        "gradient_max_slope": 5.0,
        "gradient_freeze_points": 500,
    }


def simulated_config(phantom, n_train, noise_label):
    """Config for one simulated phantom variant."""
    data_path = (
        f"{PHANTOM_BASE}/{phantom}_n{n_train}_{noise_label}/{phantom}_cone"
    )
    cfg = _common_base()
    cfg.update({
        "dataset": "r2_gaussian",
        "data_path": data_path,
        "init_points": 64_000,
        "final_points": 512_000,
        "rays_per_batch": 1_000_000,
    })
    return cfg


def inveon_config():
    cfg = _common_base()
    cfg.update({
        "dataset": "inveon_ct",
        "data_path": INVEON_PREFIX,
        "init_points": 64_000,
        "final_points": 512_000,
        # 363 projections × 512 × 768 ≈ 143M rays; 1M batch is fine
        "rays_per_batch": 1_000_000,
    })
    return cfg


def acr_config():
    cfg = _common_base()
    cfg.update({
        "dataset": "acr_phantom",
        "data_path": ACR_DIR,
        "init_points": 64_000,
        "final_points": 512_000,
        # 1800 views × 736 × 64 ≈ 85M rays; 1M batch is fine
        "rays_per_batch": 1_000_000,
        # Helical fan-beam: no ground-truth volume → skip volume metrics at eval
        "save_volume": True,
    })
    return cfg


# ── Build run table ───────────────────────────────────────────────────────────

SIMULATED_PHANTOMS = ["shepp_logan", "nema_iec", "marschner_lobb"]
SIMULATED_VARIANTS = [
    (500, "clean"),
    (500, "low"),
    (500, "mid"),
    (75,  "clean"),
    (75,  "low"),
    (75,  "mid"),
]

ALL_RUNS = {}

for phantom in SIMULATED_PHANTOMS:
    for n_train, noise in SIMULATED_VARIANTS:
        name = f"{phantom}-n{n_train}_{noise}"
        ALL_RUNS[name] = simulated_config(phantom, n_train, noise)

ALL_RUNS["inveon_nema"] = inveon_config()
ALL_RUNS["acr_phantom"] = acr_config()


# ── Infrastructure (same pattern as other sweeps) ─────────────────────────────

def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def parse_metrics(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                metrics[key] = float(m.group(2))
    return metrics


def run_experiment(name, cfg):
    mpath = metrics_path(name)
    if os.path.exists(mpath):
        print(f"[SKIP] {name} — already complete")
        return True

    out_dir = os.path.join(SWEEP_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep_debug_phantoms/{name}",
    ]
    print(f"[RUN]  {name}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"[FAIL] {name} — exit code {result.returncode}")
        return False
    if not os.path.exists(mpath):
        print(f"[WARN] {name} — finished but metrics.txt missing")
        return False
    return True


def collect_summary(names, sort_key="psnr"):
    rows = []
    for name in names:
        mpath = metrics_path(name)
        if not os.path.exists(mpath):
            continue
        rows.append({"name": name, **parse_metrics(mpath)})

    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
    if not rows:
        print("[WARN] No completed runs to summarize")
        return rows

    seen = set()
    all_keys = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    fieldnames = ["name"] + [k for k in all_keys if k != "name"]
    out_csv = os.path.join(SWEEP_DIR, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] {len(rows)} runs → {out_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Debug phantom sweep (sweep_debug_phantoms)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID",
                        help="Run only specific named runs")
    parser.add_argument("--summarize", action="store_true",
                        help="Collect metrics.txt files into summary.csv (also runs after training)")
    parser.add_argument("--list", action="store_true",
                        help="Print all run names and key params, then exit")
    parser.add_argument("--worker", type=int, metavar="W",
                        help="Worker index (1-based) for parallelism")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers",
                        help="Total number of workers")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\nDebug phantom sweep — {len(ALL_RUNS)} runs total\n")
        print("── Simulated phantoms ──────────────────────────────────────────")
        for phantom in SIMULATED_PHANTOMS:
            for n_train, noise in SIMULATED_VARIANTS:
                name = f"{phantom}-n{n_train}_{noise}"
                cfg = ALL_RUNS[name]
                pts = cfg["final_points"] // 1000
                rpb = cfg["rays_per_batch"] // 1_000_000
                print(f"  {name:<36}  pts={pts}k  rpb={rpb}M")
        print()
        print("── Real-scanner datasets ──────────────────────────────────────")
        for name in ("inveon_nema", "acr_phantom"):
            cfg = ALL_RUNS[name]
            pts = cfg["final_points"] // 1000
            print(f"  {name:<36}  pts={pts}k  dataset={cfg['dataset']}")
        return

    os.makedirs(SWEEP_DIR, exist_ok=True)
    all_names = list(ALL_RUNS.keys())

    if args.runs:
        selected = set(args.runs)
        names = [n for n in all_names if n in selected]
        for u in selected - set(all_names):
            print(f"[WARN] Unknown run ID: {u}")
    else:
        names = all_names

    if args.worker is not None:
        names = names[args.worker - 1::args.num_workers]
        if not names:
            print(f"[WARN] worker {args.worker}/{args.num_workers} — nothing to do")
            return
        print(f"Worker {args.worker}/{args.num_workers} — {len(names)} runs: {names}")
    else:
        print(f"Running {len(names)}/{len(all_names)} runs")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names)


if __name__ == "__main__":
    main()
