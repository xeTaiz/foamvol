# Voxel-grid convention: one convention, measured consequences, rerun scope

## The convention

A volume of resolution `R` covering `[-extent, extent]` has pitch `2*extent/R`,
and voxel `i` is represented by its **centre**:

```
c_i = -extent + (i + 0.5) * (2*extent/R)
```

Canonical implementation: `voxel_grid.py` (`voxel_center_coords`,
`voxel_center_grid`, `subvoxel_offsets`, `ALIGN_CORNERS`). Pinned by
`test/test_voxel_grid_convention.py` (10 tests, all passing).

The rejected alternative is the **endpoint** convention,
`linspace(-extent, extent, R)`: pitch `2*extent/(R-1)`, samples on the box
faces. Critically, `align_corners=True` in `F.grid_sample` / `F.interpolate`
**is** that same endpoint convention expressed as a boolean — it maps normalized
±1 onto the first/last voxel *centres*. Mixing the two was the bug.

## Which convention the data actually uses — measured, not assumed

Sub-voxel offset scan on `SC128_ctrl`, correct pitch `2/R`, ss1, PSNR vs
`vol_gt.npy`:

| offset t (voxels) | 0.00 | 0.25 | 0.40 | **0.50** | 0.60 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| PSNR | 29.182 | 31.076 | 31.694 | **31.817** | 31.666 | 31.022 | 29.261 |

Symmetric, maximum exactly at `t=0.5`. `vol_gt.npy` is centre-defined; sampling
at voxel edges costs 2.6 dB.

## What the old grid cost

Decomposition of the 3.83 dB gap between TB's `test/vol_raw_psnr` (30.0920) and
the JSON `volume_psnr` (33.9199), same checkpoint, one evaluator:

| config | PSNR |
|---|---|
| `linspace` ss1 | 30.0920 (== TB) |
| `centers` ss1 | 31.8165 → grid alone **+1.72 dB** |
| `centers` ss4 | 33.9199 (== JSON) |
| `centers` ss8 | 33.9575 (converged; ss4 is within 0.04 dB) |
| `centers` ss4, split ignored | 33.9231 (**−0.003 dB**: splits are irrelevant here) |

Per-arm bias across all 13 sweep checkpoints at matched ss1:

| metric | bias (centres − endpoint) |
|---|---|
| volume PSNR | +1.7245 … +1.9380 dB (spread **0.21 dB**) |
| Sobel PSNR | +2.7309 … +3.1212 dB (edge metric, hit ~1.6× harder) |

**The bias is not a constant offset** — it depends on how sharp the
reconstruction is, so it cannot be subtracted post hoc, and its 0.21 dB spread
exceeds the 0.145 dB same-config noise floor at 128k.

### What that did and did not do to conclusions

Honest accounting, because the sweep's four arms per cell count are repeats of
**one effective configuration** (continuity loss was identically zero — see
`SWEEP-SPLITCELL-RESULTS-v1.md`), so their spread *is* the noise estimate:

- **Real effects survive.** Cell-count scaling, a genuine effect: 128k→512k
  ctrl is +0.844 dB on the endpoint grid vs +0.919 dB on centres — the wrong
  grid compressed it by only ~9%. Any conclusion resting on ≥0.3 dB stands.
- **One qualitative flip.** `SC256_scalar` (the only genuinely different config
  in the sweep, no split) sat **+0.076 dB above** the split arms on the endpoint
  grid (≈3σ of within-group spread) and **−0.011 dB below** them on centres
  (≈2σ). The old grid manufactured an apparent scalar-beats-split advantage.
- **Within-group reorderings are not evidence.** Reshuffling statistically
  identical repeats is expected under any perturbation.

Net: the old grid inflated the noise floor and could invert sub-0.25 dB
comparisons. It did **not** hide large real gains.

## Sites changed

Sampling grids (endpoint → centres):
- `vis_foam.py:voxelize_volumes` — the headline: every TB volume metric.
- `vis_foam.py:make_slice_coords` — slice panels and slice PSNR.
- `vis_foam.py:sample_gt_slice` — GT slice index mapping used `/(G-1)`; now `floor(·*G)`.
- `voxelize.py:voxelize` — standalone script whose docstring already *claimed* centres.
- `experiments/orientation_recovery.py:_assign_samples` — world mapping and `np.gradient` spacing.
- `radfoam_model/scene.py:391` and `split_voxelize.py` were already correct; unified onto the helper so there is one spelling.

`align_corners=True` → `ALIGN_CORNERS` (False):
- `radfoam_model/scene.py` — FDK init sampling, ref-volume resize (×2), ref-weight sampling, `mu_gt` backward probe.
- `train_vol.py:_gt_sample` — direct volume supervision.
- `radfoam_model/features.py:assign_cell_features`.
- `vis_foam.py` — DRR ray-sum through the GT volume, split zoom panels.
- `experiments/analyze_split_cells.py` — plus a comment that documented the wrong behaviour as if intended.

GT generators (endpoint → centres): `vis_foam.py:load_gt_volume` (`ct_synthetic`),
`data_loader/ct_cube.py:make_gt_volume`. `r2_gaussian` loads `vol_gt.npy` and was
always centre-defined.

PSNR: `voxelize.py:compute_volume_psnr` used `(gt.max()-gt.min())**2` while
`train.py`, `train_vol.py` and `eval_vol.py` all use R2-Gaussian's
`pixel_max = gt.max()`. Unified on the R2 convention for baseline parity. For
`vol_gt.npy` (`min=0.0, max=1.0`) the two are **numerically identical**, verified
in every row of the decomposition above — so no published number changes. They
diverge only on data with a non-zero air floor, where R2 parity is correct.

## Rerun scope

**Re-score only — no retraining (the large majority, including all 13 sweep arms).**
Standard projection-loss CT training never touches a voxel grid; supervision goes
through the CUDA rasterizer. Only the *reported* metric was wrong. Affected:
`test/vol_raw_psnr`, `test/vol_idw_psnr`, both Sobel variants, all volume SSIMs,
dice, and `train.py`'s final `compute_surface_metrics` (chamfer/HD95/F1, shifted
by ~half a voxel), plus slice panels and DRR projection comparisons.
Re-scoring runs from a saved `model.pt` in seconds to minutes per arm; the 13-arm
volume-PSNR sweep above took ~1 minute total. `test/vol_r2_psnr` is unaffected
(loaded volume, no sampling grid).

**Genuine reruns required — the misregistration entered the optimization target:**
- **FDK-initialized runs** — `scene.py` sampled the FDK volume at cell positions
  with `align_corners=True`, so the density *initialization* was misregistered.
- **`reference_volume_loss` / `--ref_volume` runs** — misregistered target volume
  and misregistered edge weights.
- **`train_vol.py` runs** — `_gt_sample` supervised against a misregistered GT,
  including through the backward pass.
- **Runs consuming `assign_cell_features`** — misregistered per-cell features.

## Operational gotcha found on the way

A stray untracked root-level `test.py` (upstream radfoam, references the
pre-fork `RadFoamScene`) shadows the `test/` directory as module name `test`, so
`importlib.import_module("test.test_x")` always fails with a misleading
`ImportError: cannot import name 'RadFoamScene'`. Load test modules by file path
(`importlib.util.spec_from_file_location`) or invoke pytest on the file.

## Verification

- `test/test_voxel_grid_convention.py`: 10/10 pass, including an exact
  `grid_sample` round-trip under `align_corners=False` and its failure under
  `True`.
- Pre-existing suites still green: `test_split_aware_idw`, `test_face_continuity`,
  `test_split_voxelize_modes`, `test_air_metrics` — 0 failures.
- End-to-end: `vis_foam.voxelize_volumes` on `SC128_ctrl` at ss1 now reports
  **31.8165**, exactly the standalone centres reference, up from 30.0920.
