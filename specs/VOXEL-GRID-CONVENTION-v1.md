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
- `radfoam_model/mesh.py:surface_metrics_vs_gt_volume` — marching-cubes vertex
  index→world used `2/(R-1)` with no half-voxel shift. See the dedicated section
  below: this one is a **metric** error, not cosmetic.
- `eval_sigma_sweep.py:264` — same GT-mesh mapping for the TB mesh export.
- `visualize_volume.py:coord_to_index` — slice picking used `(R-1)`.
- `split_voxelize.py` NIfTI affine — origin was the box face; NIfTI maps index 0
  to the **centre** of voxel 0, so the saved geometry was off by half a voxel in
  external viewers.
- `experiments/analyze_split_cells.py:363` — locator z-axis.
- `demos/click_segment.py` (9 sites: world↔voxel for slice binning, click
  markers, label sampling, mask voxelization, float slice indexing) and
  `demos/text_segment.py:167`. Cosmetic, but fixed so no mixed spelling remains.

`align_corners=True` → `ALIGN_CORNERS` (False):
- `radfoam_model/scene.py` — FDK init sampling, ref-volume resize (×2), ref-weight sampling, `mu_gt` backward probe.
- `train_vol.py:_gt_sample` — direct volume supervision.
- `radfoam_model/features.py:assign_cell_features`.
- `vis_foam.py` — DRR ray-sum through the GT volume, split zoom panels.
- `experiments/analyze_split_cells.py` — plus a comment that documented the wrong behaviour as if intended.

GT generators (endpoint → centres): `vis_foam.py:load_gt_volume` (`ct_synthetic`),
`data_loader/ct_cube.py:make_gt_volume`. `r2_gaussian` loads `vol_gt.npy` and was
always centre-defined.

PSNR: `voxelize.py` used `(gt.max()-gt.min())**2` while `train.py`,
`train_vol.py` and `eval_vol.py` used R2-Gaussian's `pixel_max = gt.max()`.
All consumers now import the single `radfoam_model.utils.compute_volume_psnr`
implementation; NumPy stays in NumPy and tensors stay on their current device.
For `vol_gt.npy` (`min=0.0, max=1.0`) the formulas are numerically identical,
verified in every decomposition row above, so no published number changes. They
diverge only on data with a non-zero air floor, where R2 parity is the reference.

## The two surface-metric paths are affected differently

`cd`/`hd`/`hd95`/`f1`/`dice` reach you through two different code paths, and only
one of them had a registration error in its own right:

1. **`compute_surface_metrics`** (duplicated in `train.py`, `eval_vol.py`,
   `train_vol.py`; used by `eval_hard_surface.py` and therefore by
   `surface_hard_ss4_metrics.json`) runs marching cubes on the **prediction and
   GT arrays in index space**. Both meshes share one indexing, so the metric is
   internally registered. It is only wrong if the *prediction volume* was
   sampled on the wrong grid — true for `train.py`'s TB numbers (fed by
   `voxelize_volumes`), **not** true for the sweep, whose volumes come from
   `split_voxelize` and were always on centres. **The sweep's published chamfer /
   hd95 / f1 numbers stand.**

2. **`surface_metrics_vs_gt_volume`** (`radfoam_model/mesh.py`, reported as
   `Mesh Raw CD / HD95 / F1`) compares an index-derived GT mesh against the
   Voronoi mesh in **true world coordinates**, so the index→world mapping matters
   directly. It used `2/(R-1)` with no half-voxel shift. Measured effect of the
   fix on the GT mesh at R=256: vertices move by **mean 0.375, max 0.708 voxels**.
   Reported chamfer values are 1.14-2.05 voxels, so this was an **18-33%
   systematic error** in those numbers. All `Mesh Raw *` metrics must be
   recomputed.

Dice is a thresholded volume-overlap metric, so it inherits case 1: correct
wherever the prediction volume was on centres, wrong in `train.py`'s TB path.

## Deliberately unchanged

Not every `linspace` or `(N-1)` is a voxel grid. These were audited and left:

- **Detector / ray geometry** — `data_loader/ct_synthetic.py:47-48`,
  `acr_dicomctpd.py:114`. These define projector geometry that must match the
  rays stored with each dataset; changing them would desynchronise the
  projections. The production loaders (`r2_gaussian.py`, `blender.py`,
  `colmap.py`, `inveon_ct.py`, `ct_cube.py` detector code) already use `+0.5`
  pixel centres.
- **2-D tangent-space parameter grids** — `vis_foam.py:2278,2309`,
  `analyze_split_cells.py:275,299`. These sweep a local surface patch in
  `[-1,1]` parameter coordinates, not a voxel lattice; endpoints are correct.
- **Point initialisation lattice** — `radfoam_model/scene.py:265`. Cell seed
  positions with jitter, not a sampling grid.
- **Array-bound clamps** — `adj.shape[0] - 1`, `cache.vertices.shape[0] - 1`, etc.
  Index arithmetic, unrelated to geometry.

## Rerun scope

**Re-score only — no retraining (the large majority, including all 13 sweep arms).**
Standard projection-loss CT training never touches a voxel grid; supervision goes
through the CUDA rasterizer. Only the *reported* metric was wrong. Affected:
`test/vol_raw_psnr`, `test/vol_idw_psnr`, both Sobel variants, all volume SSIMs,
dice, `train.py`'s `compute_surface_metrics` (its prediction volume came from
`voxelize_volumes`), every `Mesh Raw *` metric (see above), plus slice panels and
DRR projection comparisons.
Re-scoring runs from a saved `model.pt` in seconds to minutes per arm; the 13-arm
volume-PSNR sweep above took ~1 minute total. `test/vol_r2_psnr` is unaffected
(loaded volume, no sampling grid), and the sweep's `surface_hard_ss4_metrics.json`
numbers are unaffected (prediction volumes came from `split_voxelize`).

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

- `test/test_voxel_grid_convention.py`: **13/13** pass, including an exact
  `grid_sample` round-trip under `align_corners=False`, its failure under `True`,
  and index↔world inverse round-trips with half-open voxel spans.
- Full run across `test_voxel_grid_convention`, `test_split_aware_idw`,
  `test_face_continuity`, `test_split_voxelize_modes`, `test_air_metrics`:
  **38 tests ran, 0 failures.**
- `test/test_volume_psnr.py`: **4/4** pass for NumPy/Torch parity, R2's
  non-zero-floor convention, zero-MSE `+inf`, and zero-peak/nonzero-error
  `-inf`. Import smoke confirms all six consumers resolve to the canonical
  function object.
- End-to-end: `vis_foam.voxelize_volumes` on `SC128_ctrl` at ss1 now reports
  **31.8165**, exactly the standalone centres reference, up from 30.0920.
- Mesh registration shift measured directly at R=256: mean 0.375, max 0.708
  voxels.
