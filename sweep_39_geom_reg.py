#!/usr/bin/env python3
"""Sweep 39: Geometric regularisers (A, C) on 75-view CT — best428_nointerp baseline.

Tests top-eigvec alignment (Loss A) and normal-Laplacian (Loss C) across:
  - Wide weight ranges (1e-4 → 1.0, ×3 log steps) to find optimum and break-point
  - Combinations with each other
  - Combinations with density-side regs (vvar, nvar, tv)
    → grad_smooth_hops=0 for all density-combo groups to avoid confounding
  - Extended iterations (13000) on promising configs

Baseline: configs/best428_nointerp.yaml (all density regs OFF, grad_smooth_hops=1,
variance-based pruning, HE=0.2, targeted=0.1, no interpolation).

Density-reg controls use grad_smooth_hops=0 to isolate effect from grad smoothing.

Usage:
    python sweep_39_geom_reg.py
    python sweep_39_geom_reg.py --runs V05 Vc1
    python sweep_39_geom_reg.py --list
    python sweep_39_geom_reg.py --summarize
    python sweep_39_geom_reg.py --worker 1 --of 4
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import yaml

SWEEP_DIR = "output/sweep39_geom_reg"
DATA_PATH = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"

# Weight grid: 9 values spanning 1e-4 → 1.0
W = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def base_config(
    # --- Geometric regularisers (new; off by default) ---
    top_eig_align_weight=0.0,
    top_eig_align_start=-1,
    top_eig_align_until=-1,
    normal_lap_weight=0.0,
    normal_lap_start=-1,
    normal_lap_until=-1,
    cvt_weight=0.0,
    cvt_start=-1,
    cvt_until=-1,
    cvt_hops=1,
    # --- Density-side regularisers ---
    voxel_var_weight=0.0,
    voxel_var_resolution=64,
    neighbor_var_weight=0.0,
    neighbor_var_hops=1,
    tv_weight=0.0,
    # --- Grad smoothing ---
    grad_smooth_hops=1,
    # --- Iteration count / freeze ---
    iterations=10000,
    freeze_points=9500,
    # --- Misc overrides ---
    **kwargs,
):
    cfg = {
        # ---- Training ----
        "iterations": iterations,
        "rays_per_batch": 1_000_000,
        "rays_per_batch_late": 8_000_000,
        "rays_per_batch_late_start": 9000,
        "init_points": 64000,
        "final_points": 512000,
        "activation_scale": 1.0,
        "init_scale": 1.05,
        "init_type": "random",
        "init_density": 2.0,
        "init_volume_path": "",
        "device": "cuda",
        "debug": False,
        "viewer": False,
        "save_volume": False,
        "dataset": "r2_gaussian",
        "data_path": DATA_PATH,
        "diag": True,
        "corr_diag": True,
        # ---- Optimisation ----
        "loss_type": "l1",
        "points_lr_init": 2e-4,
        "points_lr_final": 5e-6,
        "density_lr_init": 5e-2,
        "density_lr_final": 1e-2,
        "freeze_points": freeze_points,
        "density_grad_clip": 10.0,
        # ---- TV ----
        "tv_weight": tv_weight,
        "tv_start": 0,
        "tv_epsilon": 1e-4,
        "tv_area_weighted": False,
        "tv_border": False,
        "tv_anneal": False,
        "tv_on_raw": True,
        # ---- Voxel variance ----
        "voxel_var_weight": voxel_var_weight,
        "voxel_var_weight_final": -1.0,
        "voxel_var_resolution": voxel_var_resolution,
        "voxel_var_start": 0,
        "voxel_var_supersample": 4,
        # ---- Neighbour variance ----
        "neighbor_var_weight": neighbor_var_weight,
        "neighbor_var_weight_final": -1.0,
        "neighbor_var_hops": neighbor_var_hops,
        "neighbor_var_start": 0,
        "neighbor_reg_type": "bilateral_var",
        "neighbor_huber_delta": 0.1,
        # ---- Sigma schedule (for vvar/nvar bilateral gate) ----
        "var_sigma_v_init": 50.0,
        "var_sigma_v_final": 0.2,
        # ---- Grad smoothing ----
        "grad_smooth_hops": grad_smooth_hops,
        # ---- Geometric regularisers ----
        "top_eig_align_weight": top_eig_align_weight,
        "top_eig_align_start": top_eig_align_start,
        "top_eig_align_until": top_eig_align_until,
        "normal_lap_weight": normal_lap_weight,
        "normal_lap_start": normal_lap_start,
        "normal_lap_until": normal_lap_until,
        "cvt_weight": cvt_weight,
        "cvt_start": cvt_start,
        "cvt_until": cvt_until,
        "cvt_hops": cvt_hops,
        # ---- Densification + pruning ----
        "densify_from": 1000,
        "densify_until": 6000,
        "densify_factor": 1.15,
        "gradient_fraction": 0.4,
        "idw_fraction": 0.3,
        "entropy_fraction": 0.3,
        "entropy_bins": 5,
        "contrast_alpha": 0.0,
        "redundancy_threshold": 0.0,
        "redundancy_cap": 0.05,
        "redundancy_cap_init": 0.0,
        "redundancy_cap_final": 0.0,
        "prune_variance_criterion": True,
        "prune_hops": 1,
        "ref_guided_pruning": False,
        "ref_guided_densify": False,
        "ref_guided_eps": 0.01,
        "targeted_fraction": 0.1,
        "targeted_start": -1,
        # ---- Ref volume off ----
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
        # ---- Sampling ----
        "high_error_fraction": 0.2,
        "high_error_power": 1.0,
        "high_error_start": -1,
        # ---- Interpolation off ----
        "interpolation_start": -1,
        "interp_ramp": False,
        "interp_sigma_abs": 0.010,
        "interp_sigma_scale": 1.0,
        "interp_sigma_v": 0.05,
        "per_cell_sigma": False,
        "per_neighbor_sigma": False,
        # ---- Bilateral filter off ----
        "bf_start": -1,
        "bf_until": 6000,
        "bf_period": 10,
        "bf_sigma_init": 2.0,
        "bf_sigma_final": 0.3,
        "bf_sigma_v_init": 10.0,
        "bf_sigma_v_final": 0.1,
        # ---- Gaussians off ----
        "gaussian_start": -1,
        "freeze_base_at_gaussian": False,
        "joint_finetune_start": -1,
        "peak_lr_init": 1e-2,
        "peak_lr_final": 1e-3,
        "offset_lr_init": 1e-3,
        "offset_lr_final": 1e-4,
        "cov_lr_init": 1e-2,
        "cov_lr_final": 1e-3,
        # ---- Linear gradient off ----
        "gradient_start": -1,
        "gradient_lr_init": 1e-2,
        "gradient_lr_final": 1e-3,
        "gradient_warmup": 500,
        "gradient_max_slope": 5.0,
        "gradient_freeze_points": 500,
    }
    cfg.update(kwargs)
    return cfg


# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

ALL_RUNS = {}

# ---- G0: Baseline (best428_nointerp verbatim, no new regs) ----
ALL_RUNS["V00-baseline"] = base_config()

# ---- Gctrl: Density-reg-only controls (grad_smooth OFF for isolation) ----
ALL_RUNS["Vc1-vvar1e-3-gs0"] = base_config(voxel_var_weight=1e-3,  grad_smooth_hops=0)
ALL_RUNS["Vc2-nvar1e-3-gs0"] = base_config(neighbor_var_weight=1e-3, grad_smooth_hops=0)
ALL_RUNS["Vc3-tv1e-3-gs0"]   = base_config(tv_weight=1e-3,           grad_smooth_hops=0)

# ---- G1: Loss A alone — weight sweep ----
for _i, _w in enumerate(W):
    _wstr = f"{_w:.0e}".replace("e-0", "e-").replace("e+0", "e")
    ALL_RUNS[f"V{_i+1:02d}-A_{_wstr}"] = base_config(top_eig_align_weight=_w)

# ---- G2: Loss C alone — weight sweep ----
for _i, _w in enumerate(W):
    _wstr = f"{_w:.0e}".replace("e-0", "e-").replace("e+0", "e")
    ALL_RUNS[f"V{_i+10:02d}-C_{_wstr}"] = base_config(normal_lap_weight=_w)

# ---- G3: A + C matched weights ----
for _w in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    _wstr = f"{_w:.0e}".replace("e-0", "e-").replace("e+0", "e")
    ALL_RUNS[f"V{19 + [1e-3,3e-3,1e-2,3e-2,1e-1].index(_w):02d}-AC_{_wstr}"] = base_config(
        top_eig_align_weight=_w, normal_lap_weight=_w)

# ---- G4: A × vvar (grad_smooth OFF) ----
_idx = 24
for _aw in [1e-3, 3e-3, 1e-2]:
    for _dw in [1e-3, 3e-3, 1e-2]:
        _as = f"{_aw:.0e}".replace("e-0", "e-")
        _ds = f"{_dw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-A{_as}_vvar{_ds}_gs0"] = base_config(
            top_eig_align_weight=_aw, voxel_var_weight=_dw, grad_smooth_hops=0)
        _idx += 1

# ---- G5: A × nvar (vvar OFF, grad_smooth OFF) ----
for _aw in [1e-3, 3e-3, 1e-2]:
    for _dw in [1e-3, 3e-3, 1e-2]:
        _as = f"{_aw:.0e}".replace("e-0", "e-")
        _ds = f"{_dw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-A{_as}_nvar{_ds}_gs0"] = base_config(
            top_eig_align_weight=_aw, neighbor_var_weight=_dw, grad_smooth_hops=0)
        _idx += 1

# ---- G6: A × tv (vvar OFF, grad_smooth OFF) ----
for _aw in [1e-3, 3e-3, 1e-2]:
    for _tw in [1e-4, 1e-3, 1e-2]:
        _as = f"{_aw:.0e}".replace("e-0", "e-")
        _ts = f"{_tw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-A{_as}_tv{_ts}_gs0"] = base_config(
            top_eig_align_weight=_aw, tv_weight=_tw, grad_smooth_hops=0)
        _idx += 1

# ---- G7: C × vvar (grad_smooth OFF) ----
for _cw in [1e-3, 3e-3, 1e-2]:
    for _dw in [1e-3, 3e-3, 1e-2]:
        _cs = f"{_cw:.0e}".replace("e-0", "e-")
        _ds = f"{_dw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-C{_cs}_vvar{_ds}_gs0"] = base_config(
            normal_lap_weight=_cw, voxel_var_weight=_dw, grad_smooth_hops=0)
        _idx += 1

# ---- G8: C × nvar (vvar OFF, grad_smooth OFF) ----
for _cw in [1e-3, 3e-3, 1e-2]:
    for _dw in [1e-3, 3e-3, 1e-2]:
        _cs = f"{_cw:.0e}".replace("e-0", "e-")
        _ds = f"{_dw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-C{_cs}_nvar{_ds}_gs0"] = base_config(
            normal_lap_weight=_cw, neighbor_var_weight=_dw, grad_smooth_hops=0)
        _idx += 1

# ---- G9: C × tv (vvar OFF, grad_smooth OFF) ----
for _cw in [1e-3, 3e-3, 1e-2]:
    for _tw in [1e-4, 1e-3, 1e-2]:
        _cs = f"{_cw:.0e}".replace("e-0", "e-")
        _ts = f"{_tw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-C{_cs}_tv{_ts}_gs0"] = base_config(
            normal_lap_weight=_cw, tv_weight=_tw, grad_smooth_hops=0)
        _idx += 1

# ---- G10: A+C+density best combos (pre-selected, grad_smooth OFF) ----
_g10 = [
    ("AC3e3_vvar1e3",  dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, voxel_var_weight=1e-3)),
    ("AC1e2_vvar1e3",  dict(top_eig_align_weight=1e-2, normal_lap_weight=1e-2, voxel_var_weight=1e-3)),
    ("A1e2C3e3_vvar1e3", dict(top_eig_align_weight=1e-2, normal_lap_weight=3e-3, voxel_var_weight=1e-3)),
    ("A3e3C1e2_vvar1e3", dict(top_eig_align_weight=3e-3, normal_lap_weight=1e-2, voxel_var_weight=1e-3)),
    ("AC3e3_vvar3e3",  dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, voxel_var_weight=3e-3)),
    ("AC3e3_nvar1e3",  dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, neighbor_var_weight=1e-3)),
    ("A1e2C3e3_nvar1e3", dict(top_eig_align_weight=1e-2, normal_lap_weight=3e-3, neighbor_var_weight=1e-3)),
    ("AC3e3_tv1e4",    dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, tv_weight=1e-4)),
    ("AC3e3_tv1e3",    dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, tv_weight=1e-3)),
    ("AC3e3_only",     dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3)),
    ("AC1e2_only",     dict(top_eig_align_weight=1e-2, normal_lap_weight=1e-2)),
    ("AC3e3_vv_nv",    dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3,
                            voxel_var_weight=1e-3, neighbor_var_weight=1e-3)),
    ("A1e2C1e2_nvar1e3", dict(top_eig_align_weight=1e-2, normal_lap_weight=1e-2, neighbor_var_weight=1e-3)),
    ("AC3e3_vv3e3_nv1e3", dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3,
                                voxel_var_weight=3e-3, neighbor_var_weight=1e-3)),
    ("AC3e2_vvar1e3",  dict(top_eig_align_weight=3e-2, normal_lap_weight=3e-2, voxel_var_weight=1e-3)),
]
for _tag, _kw in _g10:
    ALL_RUNS[f"V{_idx:02d}-{_tag}_gs0"] = base_config(grad_smooth_hops=0, **_kw)
    _idx += 1

# ---- G11: Extended iterations (13000, freeze at 9500, extra density-finalization) ----
_g11 = [
    ("V00-base",        dict()),
    ("Vc1-vvar1e3",     dict(voxel_var_weight=1e-3, grad_smooth_hops=0)),
    ("Vc2-nvar1e3",     dict(neighbor_var_weight=1e-3, grad_smooth_hops=0)),
    ("V05-A_1e-2",      dict(top_eig_align_weight=1e-2)),
    ("V14-C_1e-2",      dict(normal_lap_weight=1e-2)),
    ("G10-AC3e3_vvar1e3", dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3,
                               voxel_var_weight=1e-3, grad_smooth_hops=0)),
    ("G10-AC3e3_nvar1e3", dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3,
                               neighbor_var_weight=1e-3, grad_smooth_hops=0)),
    ("G10-AC3e3_only",  dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3,
                             grad_smooth_hops=0)),
]
for _src_tag, _kw in _g11:
    ALL_RUNS[f"V{_idx:02d}-iter13k_{_src_tag}"] = base_config(
        iterations=13000, freeze_points=9500, **_kw)
    _idx += 1

# ---- G12: CVT alone (grad_smooth=1, matching G1/G2 baseline) ----
for _i, _w in enumerate(W):
    _wstr = f"{_w:.0e}".replace("e-0", "e-").replace("e+0", "e")
    ALL_RUNS[f"V{_idx:02d}-CVT_{_wstr}"] = base_config(cvt_weight=_w)
    _idx += 1

# ---- G13: A × CVT (3×3 grid, grad_smooth=1) ----
for _aw in [1e-3, 3e-3, 1e-2]:
    for _cw in [1e-3, 3e-3, 1e-2]:
        _as = f"{_aw:.0e}".replace("e-0", "e-")
        _cs = f"{_cw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-A{_as}_CVT{_cs}"] = base_config(
            top_eig_align_weight=_aw, cvt_weight=_cw)
        _idx += 1

# ---- G14: C × CVT (3×3 grid, grad_smooth=1) ----
for _cw in [1e-3, 3e-3, 1e-2]:
    for _vtw in [1e-3, 3e-3, 1e-2]:
        _cs = f"{_cw:.0e}".replace("e-0", "e-")
        _vs = f"{_vtw:.0e}".replace("e-0", "e-")
        ALL_RUNS[f"V{_idx:02d}-C{_cs}_CVT{_vs}"] = base_config(
            normal_lap_weight=_cw, cvt_weight=_vtw)
        _idx += 1

# ---- G15: CVT × density (grad_smooth=0) ----
_g15 = [
    ("CVT3e3_vvar1e3", dict(cvt_weight=3e-3, voxel_var_weight=1e-3)),
    ("CVT3e3_nvar1e3", dict(cvt_weight=3e-3, neighbor_var_weight=1e-3)),
    ("CVT3e3_tv1e3",   dict(cvt_weight=3e-3, tv_weight=1e-3)),
    ("CVT1e2_vvar1e3", dict(cvt_weight=1e-2, voxel_var_weight=1e-3)),
    ("CVT1e2_nvar1e3", dict(cvt_weight=1e-2, neighbor_var_weight=1e-3)),
    ("CVT1e2_tv1e3",   dict(cvt_weight=1e-2, tv_weight=1e-3)),
]
for _tag, _kw in _g15:
    ALL_RUNS[f"V{_idx:02d}-{_tag}_gs0"] = base_config(grad_smooth_hops=0, **_kw)
    _idx += 1

# ---- G16: A+C+CVT cherry picks ----
_g16 = [
    ("AC3e3_CVT3e3",     dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, cvt_weight=3e-3)),
    ("AC1e2_CVT1e2",     dict(top_eig_align_weight=1e-2, normal_lap_weight=1e-2, cvt_weight=1e-2)),
    ("A3e3_C1e2_CVT3e3", dict(top_eig_align_weight=3e-3, normal_lap_weight=1e-2, cvt_weight=3e-3)),
    ("AC3e3_CVT1e2",     dict(top_eig_align_weight=3e-3, normal_lap_weight=3e-3, cvt_weight=1e-2)),
]
for _tag, _kw in _g16:
    ALL_RUNS[f"V{_idx:02d}-{_tag}"] = base_config(**_kw)
    _idx += 1
# Extended-iterations variant of the best A+C+CVT guess
ALL_RUNS[f"V{_idx:02d}-iter13k_AC3e3_CVT3e3"] = base_config(
    iterations=13000, freeze_points=9500,
    top_eig_align_weight=3e-3, normal_lap_weight=3e-3, cvt_weight=3e-3)
_idx += 1


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def metrics_path(name):
    return os.path.join(SWEEP_DIR, name, "metrics.txt")


def parse_metrics(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([\w\s]+):\s+([\d.eE+-]+(?:inf)?)", line.strip())
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                val = float(m.group(2))
                metrics[key] = val
    return metrics


def run_experiment(name, cfg):
    out_dir = os.path.join(SWEEP_DIR, name)
    mpath = metrics_path(name)

    if os.path.exists(mpath):
        print(f"[SKIP] {name} — metrics.txt already exists")
        return True

    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(out_dir, "sweep_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        sys.executable, "train.py",
        "-c", config_file,
        "--experiment_name", f"sweep39_geom_reg/{name}",
    ]
    print(f"[RUN]  {name}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"[FAIL] {name} exited with code {result.returncode}")
        return False
    if not os.path.exists(mpath):
        print(f"[WARN] {name} finished but metrics.txt not found")
        return False
    return True


def collect_summary(names, output_csv, sort_key="vol_idw_psnr"):
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

    fieldnames = ["name"] + [k for k in rows[0] if k != "name"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Summary written to {output_csv} ({len(rows)} runs)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Geometric regulariser sweep on 75-view CT (sweep 39)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", metavar="ID")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", type=int, metavar="W")
    parser.add_argument("--of", type=int, metavar="N", dest="num_workers")
    args = parser.parse_args()

    if (args.worker is None) != (args.num_workers is None):
        parser.error("--worker and --of must be used together")
    if args.worker is not None and not (1 <= args.worker <= args.num_workers):
        parser.error(f"--worker must be between 1 and {args.num_workers}")

    if args.list:
        print(f"\n{len(ALL_RUNS)} sweep 39 runs:")
        for name, cfg in ALL_RUNS.items():
            aw   = cfg.get("top_eig_align_weight", 0.0)
            cw   = cfg.get("normal_lap_weight", 0.0)
            cvtw = cfg.get("cvt_weight", 0.0)
            vv   = cfg.get("voxel_var_weight", 0.0)
            nv   = cfg.get("neighbor_var_weight", 0.0)
            tv   = cfg.get("tv_weight", 0.0)
            gs   = cfg.get("grad_smooth_hops", 1)
            itr  = cfg.get("iterations", 10000)
            parts = []
            if aw   > 0: parts.append(f"A={aw:.0e}")
            if cw   > 0: parts.append(f"C={cw:.0e}")
            if cvtw > 0: parts.append(f"cvt={cvtw:.0e}")
            if vv   > 0: parts.append(f"vvar={vv:.0e}")
            if nv   > 0: parts.append(f"nvar={nv:.0e}")
            if tv   > 0: parts.append(f"tv={tv:.0e}")
            if gs == 0: parts.append("gs0")
            if itr != 10000: parts.append(f"iter={itr}")
            desc = "  ".join(parts) if parts else "baseline"
            print(f"  {name:40s}  {desc}")
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
            print(f"[WARN] worker {args.worker}/{args.num_workers} has no runs "
                  f"(only {len(all_names)} total) — nothing to do")
            return
        print(f"Sweep 39: worker {args.worker}/{args.num_workers} — {len(names)} runs")
    else:
        print(f"Sweep 39: {len(names)}/{len(all_names)} runs selected")

    if not args.summarize:
        for name in names:
            run_experiment(name, ALL_RUNS[name])

    collect_summary(all_names, os.path.join(SWEEP_DIR, "summary.csv"))


if __name__ == "__main__":
    main()
