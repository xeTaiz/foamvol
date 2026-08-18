#!/usr/bin/env python3
"""Generate the sweep_splitcell_v1 config matrix.

Design: the *old* training schedule (densification + late ray-batch ramp +
full 10k iterations) is taken verbatim by loading the already-validated
``configs/fixed_final/{128k,256k,512k}.yaml`` baselines, then overlaying only
the split-cell (thin-surface) and shared-face continuity keys. Loading rather
than re-typing the schedule means the densify/prune/sampling settings cannot
drift from the reference baselines.

Matrix: 3 cell counts x 4 regularization settings + 1 no-split control.

  cells:  128k, 256k, 512k   (init->final point counts from the baselines)
  arms:
    ctrl     continuity off  -> isolates "do split cells help at all"
    w1e-5    1e-5, geometric only (zero-set + normal, no density term)
    w1e-5d   1e-5, geometric + density consistency
    w3e-5d   3e-5, geometric + density consistency (prior safety reference)
  plus:
    SC256_scalar   scalar (unsplit) cells at 256k -> non-split comparison

Schedule ordering that matters:
  densify 1000..6000  ->  splits activate at 6000 (point count settled)
  ->  continuity prior at 6100  ->  ray ramp 9000  ->  points freeze 9500

Points are deliberately NOT hard-frozen early
(``points_hard_freeze_at: -1``): letting geometry keep training past split
activation was by far the largest effect in the 64k/128k face-continuity
matrix (+2.50 dB / +2.29 dB volume PSNR vs the frozen arms), so this sweep
uses the unfrozen setting throughout and relies on the baseline's own late
``freeze_points: 9500``.
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "configs")

BASE_FOR_CELLS = {
    128: "configs/fixed_final/128k.yaml",
    256: "configs/fixed_final/256k.yaml",
    512: "configs/fixed_final/512k.yaml",
}

# Split-cell (thin-surface) parameterization. Matches the validated
# face-continuity / ray-batch split arms: bounded relative delta, same LR
# scales and clips. Splits switch on once densification has finished.
THIN = {
    "thin_surface_start": 6000,
    "thin_surface_K": 4,
    "thin_surface_density_mode": "relative",
    "thin_surface_relative_delta": True,
    "thin_surface_delta_max_frac": 0.5,
    # No auxiliary delta/height priors: the term under test is the shared-face
    # continuity loss, and extra priors would confound it.
    "thin_surface_delta_weight": 0.0,
    "thin_surface_height_weight": 0.0,
    "thin_surface_gate_tau": 0.01,
    "thin_surface_lr_scale": 1.0,
    "thin_surface_delta_lr_scale": 0.1,
    "thin_surface_quat_lr_scale": 0.04,
    "thin_surface_sites_lr_scale": 0.0,
    "thin_surface_heights_lr_scale": 0.04,
    "thin_surface_raw_side_lr_init": 0.0,
    "thin_surface_raw_side_lr_final": 0.0,
    "thin_surface_delta_clip": 2.0,
    "thin_surface_grad_clip": 1.0,
}

# Shared-face continuity regularizer (geometry/gating identical across arms;
# only weight + density_weight vary per arm).
FACE = {
    "thin_surface_face_start": 6100,
    "thin_surface_face_batch": 1024,
    "thin_surface_face_interval": 8,
    "thin_surface_face_candidate_refresh": 50,
    "thin_surface_face_samples": 12,
    "thin_surface_face_max_vertices": 32,
    "thin_surface_face_zero_weight": 1.0,
    "thin_surface_face_normal_weight": 0.25,
    "thin_surface_face_abs_contrast_fraction": 0.01,
    "thin_surface_face_relative_contrast": 0.1,
    "thin_surface_face_base_density_fraction": 0.05,
    "thin_surface_face_crossing_margin": 0.005,
    "thin_surface_face_side_agreement": 0.6,
    "thin_surface_face_normal_dot": 0.0,
    "thin_surface_face_zero_bandwidth": 0.2,
    "thin_surface_face_huber_beta": 0.05,
    "thin_surface_face_seed": 42,
}

# Fixed GT-anchored zoom panels for split-surface inspection.
ZOOM = {
    "thin_surface_zoom_anchor_count": 6,
    "thin_surface_zoom_anchor_seed": 42,
    "thin_surface_zoom_center_fraction": 0.6,
    "thin_surface_zoom_min_separation": 0.16,
    "thin_surface_zoom_resolution": 192,
    "thin_surface_zoom_extent_scale": 2.2,
}

ARMS = {
    "ctrl":   {"thin_surface_face_weight": 0.0,
               "thin_surface_face_density_weight": 0.0},
    "w1e-5":  {"thin_surface_face_weight": 1.0e-5,
               "thin_surface_face_density_weight": 0.0},
    "w1e-5d": {"thin_surface_face_weight": 1.0e-5,
               "thin_surface_face_density_weight": 0.1},
    "w3e-5d": {"thin_surface_face_weight": 3.0e-5,
               "thin_surface_face_density_weight": 0.1},
}

COMMON = {
    # Keep geometry trainable through split activation; the baseline's own
    # freeze_points: 9500 still stops it near the end.
    "points_hard_freeze_at": -1,
    "checkpoint_steps": "6000,8000,10000",
    # Diagnostics on: this sweep exists to be looked at, and every density
    # panel is now split-aware when splits are active.
    "diag": True,
    "log_percent": 5,
    "diag_percent": 10,
    "corr_diag": False,
    "save_volume": False,
}


def load_base(cells):
    with open(os.path.join(REPO, BASE_FOR_CELLS[cells])) as f:
        return yaml.safe_load(f)


def build(cells, arm):
    cfg = load_base(cells)
    cfg.update(COMMON)
    cfg.update(THIN)
    cfg.update(FACE)
    cfg.update(ZOOM)
    cfg.update(ARMS[arm])
    name = f"SC{cells}_{arm}"
    cfg["experiment_name"] = f"sweep_splitcell/{name}"
    return name, cfg


def build_scalar(cells=256):
    """No-split control: scalar cells, everything thin-surface disabled."""
    cfg = load_base(cells)
    cfg.update(COMMON)
    cfg.update(ZOOM)
    cfg.update({
        "thin_surface_start": -1,
        "thin_surface_K": 4,
        "thin_surface_density_mode": "scalar",
        "thin_surface_relative_delta": False,
        "thin_surface_delta_max_frac": 0.5,
        "thin_surface_delta_weight": 0.0,
        "thin_surface_height_weight": 0.0,
        "thin_surface_gate_tau": 0.01,
        "thin_surface_lr_scale": 1.0,
        "thin_surface_delta_lr_scale": 0.0,
        "thin_surface_quat_lr_scale": 0.0,
        "thin_surface_sites_lr_scale": 0.0,
        "thin_surface_heights_lr_scale": 0.0,
        "thin_surface_raw_side_lr_init": 0.0,
        "thin_surface_raw_side_lr_final": 0.0,
        "thin_surface_delta_clip": 2.0,
        "thin_surface_grad_clip": 1.0,
        "thin_surface_face_start": -1,
        "thin_surface_face_weight": 0.0,
        "thin_surface_face_density_weight": 0.0,
    })
    name = f"SC{cells}_scalar"
    cfg["experiment_name"] = f"sweep_splitcell/{name}"
    return name, cfg

# Compressed schedule that still crosses every activation boundary
# (densify -> split activation -> continuity prior -> ray ramp -> freeze) so a
# smoke run exercises the same code paths as a real arm in well under a minute.
SMOKE = {
    "iterations": 60,
    "checkpoint_steps": "60",
    "init_points": 8000,
    "final_points": 12000,
    "densify_from": 5,
    "densify_until": 30,
    "rays_per_batch": 50000,
    "rays_per_batch_late": 100000,
    "rays_per_batch_late_start": 50,
    "freeze_points": 55,
    "thin_surface_start": 30,
    "thin_surface_face_start": 33,
    "thin_surface_face_interval": 2,
    "thin_surface_face_candidate_refresh": 5,
    # Force the diagnostic panels to render mid-run and at the end: these are
    # the split-aware IDW paths under test.
    "log_percent": 50,
    "diag_percent": 50,
}


def build_smoke():
    """Two fast configs: split (all paths active) and scalar (reduction case)."""
    out = []
    _, split_cfg = build(128, "w1e-5d")
    split_cfg.update(SMOKE)
    split_cfg["experiment_name"] = "sweep_splitcell/smoke_split"
    out.append(("smoke_split", split_cfg))

    _, scalar_cfg = build_scalar(256)
    scalar_cfg.update(SMOKE)
    scalar_cfg["thin_surface_start"] = -1
    scalar_cfg["thin_surface_face_start"] = -1
    scalar_cfg["experiment_name"] = "sweep_splitcell/smoke_scalar"
    out.append(("smoke_scalar", scalar_cfg))
    return out



def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for cells in (128, 256, 512):
        for arm in ARMS:
            name, cfg = build(cells, arm)
            path = os.path.join(OUT, f"{name}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
            written.append(name)
    name, cfg = build_scalar()
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
    written.append(name)

    for name, cfg in build_smoke():
        path = os.path.join(OUT, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
        written.append(name)

    print(f"wrote {len(written)} configs to {OUT}")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
