# Split-cell continuity on the production schedule: 128k/256k/512k results

## Headline

The shared-face continuity regularizer **never activated**. In all 12 split arms,
`face_continuity/candidate_faces` and `face_continuity/raw_total` were `0.0` at
every logged step from 6499 to 9999 — the entire window after `face_start=6100`.
It contributed exactly zero loss and zero gradient on this schedule.

Root cause: the thin-surface split collapses. Median `|tanh(density_delta)|` at
step 10000 is `3e-4` to `3e-3`, while the regularizer's contrast gate requires
`>= 0.1` (10% of a cell's own density). Zero cells clear the gate in 9 of 12
arms. The earlier 64k conclusion that this prior is "a real geometric prior, not
a no-op" **does not transfer** from frozen geometry to densification + unfrozen
points.

Because the loss was identically zero, the four arms at each cell count are
repeats of one effective configuration, so their spread is a free noise estimate.
Every apparent "regularization effect" in the table sits inside that band.

## What ran

13 arms, `sweep_splitcell_v1`, one GPU each across 4 A6000s on `KW60996`.
Schedule taken verbatim from `configs/fixed_final/{128k,256k,512k}.yaml`:
10000 iterations, densify 1000->6000, ray batch 1M->4M at 9000, `freeze_points:
9500`, `points_hard_freeze_at: -1` (no early freeze — geometry trains), thin
surface from 6000, continuity from 6100.

Arms per cell count: `ctrl` (weight 0), `w1e-5` (weight 1e-5, density 0),
`w1e-5d` (1e-5, density 0.1), `w3e-5d` (3e-5, density 0.1), plus `SC256_scalar`
(no split at all). Arms within a cell count differ **only** in
`thin_surface_face_weight`, `thin_surface_face_density_weight`, and
`experiment_name` — verified by config diff.

## Where these numbers live — they are NOT in TensorBoard

Every metric in the table below is read from JSON files on disk, not from TB:

```
output/sweep_splitcell/<ARM>/volume_hard_ss4_metrics.json   -> volume_psnr, ssim, dice, air.*
output/sweep_splitcell/<ARM>/surface_hard_ss4_metrics.json  -> chamfer, hausdorff_95, f1_*
output/sweep_splitcell/<ARM>/face_continuity_eval.json      -> candidate_faces
```

They are produced by `split_voxelize.py --resolution 256 --supersample 4`, which
`run_sweep.sh` runs as a **separate process after training** (lines 51-56). That
script contains no `SummaryWriter` and never calls `add_scalar`, so none of these
values can appear in TensorBoard. It writes `.npy`, `_metrics.json`, a slices
PNG, and a NIfTI.

Three volume PSNRs were historically visible for one run. For `SC128_ctrl`:

| value | number | source | interpretation |
|---|---|---|---|
| `volume_psnr` | **33.920** | `volume_hard_ss4_metrics.json` (table below) | correctly registered hard split voxelization, 256³, SS4 |
| old `test/vol_raw_psnr` | 30.092 | TensorBoard scalar | pre-fix endpoint grid, one sample/voxel; biased low |
| `test/vol_r2_psnr` | **35.851** | TensorBoard scalar | fixed precomputed R2 volume; correctly registered |

The R2 number uses the same `gt.max()` PSNR definition and was never affected by
foam voxelization or supersampling. A fair multi-metric comparison still requires
running the saved foam and R2 volumes through one evaluator, because the original
JSON and TensorBoard paths used different SSIM and Dice definitions.

To reproduce the table on the worker:

```
cd /code/lc64-radfoam
python experiments/sweep_splitcell_v1/summarize.py
# or read one arm directly:
cat output/sweep_splitcell/SC128_ctrl/volume_hard_ss4_metrics.json
```

## Decomposing the 3.8 dB gap (measured, not assumed)

Both numbers were reproduced from the single `SC128_ctrl` checkpoint with one
evaluator, varying only one factor at a time. `linspace ss1` reproduces TB's
`test/vol_raw_psnr` to 4 decimals and `centers ss4` reproduces the JSON
`volume_psnr` to 4 decimals, so the comparison is exact:

| config | PSNR | note |
|---|---|---|
| `linspace` ss1 | **30.0920** | == TB `test/vol_raw_psnr` (30.0920) |
| `centers` ss1 | 31.8165 | grid fix alone: **+1.72 dB** |
| `centers` ss2 | 33.6428 | +1.83 |
| `centers` ss4 | **33.9199** | == JSON `volume_psnr` (33.91995) |
| `centers` ss8 | 33.9575 | +0.038 — converged |
| `linspace` ss4 | 31.7333 | supersampling without the grid fix |
| `centers` ss4, split ignored | 33.9231 | **−0.0032 dB** vs split-aware |

So the gap is `+1.72 dB` grid registration `+ 2.10 dB` anti-aliasing, and
**0.00 dB** from split cells. Two non-factors, both verified rather than
assumed:

- **Split cells contribute nothing.** The TB raw path *is* split-aware
  (`voxelize_volumes:1535-1537` calls `_split_eval`). Evaluating the same grid
  with the split ignored changes PSNR by −0.003 dB, independently confirming the
  delta collapse: the split is not what separates these numbers.
- **The PSNR formulas agree.** All consumers now use
  `radfoam_model.utils.compute_volume_psnr`, with `pixel_max=gt.max()`.
  `vol_gt.npy` has `min=0.0, max=1.0`, so the former peak-to-peak spelling was
  numerically identical and no R2 or published foam PSNR changed.

### The TB grid is misregistered, not merely coarser

`voxelize_volumes:1513` samples `torch.linspace(-extent, extent, resolution)`:
pitch `2/255` and phase starting at a voxel *edge*. `split_voxelize:330` samples
voxel centers at pitch `2/256`. A sub-voxel offset scan at correct pitch `2/R`,
ss1, settles which matches the GT:

| offset t (voxels) | 0.00 | 0.25 | 0.40 | **0.50** | 0.60 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| PSNR | 29.182 | 31.076 | 31.694 | **31.817** | 31.666 | 31.022 | 29.261 |

Symmetric with a maximum exactly at `t=0.5`: `vol_gt.npy` is defined on **voxel
centers**, and sampling at voxel edges costs 2.6 dB. TB's `linspace` grid also
has the wrong pitch (`2/255` vs `2/256`), a 0.4% stretch that is aligned at the
volume centre and drifts to half a voxel at the edges — which is why it lands at
30.09, between the `t=0` and `t=0.5` extremes.

Conclusion: for this dataset the JSON number is the correct one and
`test/vol_raw_psnr` is biased low by ~3.8 dB — ~1.7 dB of that a genuine
registration error against the GT, ~2.1 dB unconverged single-sample quadrature.
`ss4` is within 0.04 dB of converged, so 4x supersampling is the right operating
point; `ss8` costs 8x for +0.038 dB.

This grid mismatch has since been fixed repo-wide: voxel volumes use centres,
`grid_sample` uses `align_corners=False`, and generated GT volumes use the same
centre convention. See `specs/VOXEL-GRID-CONVENTION-v1.md`. The sweep JSON table
does not change because `split_voxelize.py` already sampled centres; the old
TensorBoard foam values do change. The fixed `vol_r2.npy` baseline does not.

## Common-evaluator overview against R2

To compare more than PSNR, `eval_vol.py` was run on each saved 256³ volume and
the same `vol_gt.npy`. This removes the old SSIM/Dice-definition mismatch:
SSIM below is Gaussian-window 3-D SSIM; Dice is the mean over thresholds
0.1, 0.2, ..., 0.9; Chamfer is lower-is-better; F1 is reported at 1/2 voxels.

| Model | Actual cells | Split | Vol PSNR | Vol SSIM3D | Dice | Chamfer | F1@1 / F1@2 |
|---|---:|:---:|---:|---:|---:|---:|---:|
| SC128 control | 121,932 | yes | 33.9199 | .912343 | .835986 | 2.0454 | .7488 / .8447 |
| SC256 scalar | 237,124 | no | 34.8358 | .924820 | .849124 | 1.6204 | .7823 / .8719 |
| SC256 control | 237,394 | yes | 34.8299 | .924534 | .849052 | 1.4388 | .7880 / .8777 |
| SC512 control | 462,712 | yes | 35.0250 | .927302 | .859017 | 1.1396 | .8185 / .9044 |
| R2-Gaussian `vol_r2.npy` | — | — | **35.8512** | **.943398** | **.889397** | **.7283** | **.8838 / .9481** |

Only SC256 scalar/control is a matched split ablation. Splitting changes intensity
metrics by effectively zero (−0.0059 dB PSNR, −.000286 SSIM, −.000072 Dice) but
improves Chamfer by 0.1816 voxels (11.2%) and F1 by about .006. This is
suggestive geometry improvement, not established from one scalar seed.

The strongest foam row here, SC512 split, remains 0.8262 dB PSNR, .0161 SSIM,
.0304 Dice, and .0653 F1@1 behind R2; its Chamfer is 1.1396 versus R2's .7283
(1.56× higher/worse).

## Results

| arm | volPSNR | SSIM3D | sobPSNR | dice | airMAE | airFPR | chamfer | hd95 | f1_1v | cand |
|---|---|---|---|---|---|---|---|---|---|---|
| SC128_ctrl | 33.920 | 0.9281 | 32.520 | 0.9589 | 5.33e-4 | 1.11e-4 | 2.0454 | 19.471 | 0.7488 | 0 |
| SC128_w1e-5 | 34.157 | 0.9317 | 32.783 | 0.9598 | 5.17e-4 | 8.5e-5 | 1.9819 | 18.629 | 0.7528 | 0 |
| SC128_w1e-5d | 34.203 | 0.9324 | 32.854 | 0.9601 | 5.10e-4 | 8.0e-5 | 2.1010 | 20.326 | 0.7495 | 0 |
| SC128_w3e-5d | 34.243 | 0.9334 | 32.904 | 0.9598 | 5.09e-4 | 8.3e-5 | 1.9289 | 18.507 | 0.7529 | 0 |
| SC256_ctrl | 34.830 | 0.9397 | 33.861 | 0.9616 | 4.75e-4 | 7.3e-5 | 1.4388 | 12.706 | 0.7880 | 18 |
| SC256_w1e-5 | 34.849 | 0.9398 | 33.873 | 0.9615 | 4.70e-4 | 6.6e-5 | 1.5362 | 13.267 | 0.7885 | 18 |
| SC256_w1e-5d | 34.846 | 0.9406 | 33.845 | 0.9617 | 4.72e-4 | 7.3e-5 | 1.5459 | 16.009 | 0.7877 | 0 |
| SC256_w3e-5d | 34.818 | 0.9396 | 33.846 | 0.9616 | 4.70e-4 | 7.6e-5 | 1.5581 | 14.752 | 0.7892 | 17 |
| **SC256_scalar** | 34.836 | 0.9399 | 33.867 | 0.9616 | 4.77e-4 | 7.0e-5 | 1.6204 | 16.037 | 0.7823 | - |
| SC512_ctrl | 35.025 | 0.9431 | 34.263 | 0.9612 | 4.47e-4 | 5.7e-5 | 1.1396 | 8.826 | 0.8185 | 0 |
| SC512_w1e-5 | 34.971 | 0.9421 | 34.205 | 0.9611 | 4.49e-4 | 6.4e-5 | 1.2242 | 10.501 | 0.8163 | 0 |
| SC512_w1e-5d | 35.019 | 0.9427 | 34.227 | 0.9612 | 4.54e-4 | 7.4e-5 | 1.1194 | 8.606 | 0.8185 | 0 |
| SC512_w3e-5d | 34.891 | 0.9414 | 34.144 | 0.9611 | 4.57e-4 | 6.4e-5 | 1.1389 | 8.949 | 0.8182 | 0 |

`airMAE`/`airFPR` are `air.mae.strict_air` / `air.strict_air_fpr`; `cand` is the
post-hoc continuity evaluator's mean `candidate_faces`.

## The noise floor this accidentally measured

Four arms per cell count, identical effective optimization (zero-valued loss),
so this is same-config run-to-run variation. Runs are not bitwise reproducible:
nondeterministic CUDA atomics feed densification decisions, which amplify.

| cells | volPSNR sd | volPSNR range | chamfer sd | hd95 sd |
|---|---|---|---|---|
| 128k | 0.145 | 0.323 | 0.0749 | 0.84 |
| 256k | 0.015 | 0.031 | 0.0547 | 1.49 |
| 512k | 0.062 | 0.134 | 0.0467 | 0.87 |

Consequence: the tidy-looking 128k "dose response" (33.920 -> 34.157 -> 34.203
-> 34.243, monotone in weight, with air MAE improving alongside) is **noise**.
The arms are the same configuration; a 0.32 dB range is exactly this group's
spread. Do not report it as a regularization gain. Any future claim on this
schedule needs a margin well beyond ~0.15 dB volume PSNR and ~0.08 chamfer, or
multiple seeds per arm.

## Split vs no split (256k)

`SC256_scalar` lands at volPSNR **34.836** against a split-arm mean of
**34.836** (sd 0.015) — no intensity benefit from splitting whatsoever, which is
what the delta collapse predicts.

Surface metrics mildly favour splitting: chamfer 1.6204 vs split mean 1.5197
(sd 0.0547, ~1.8 sd), hd95 16.037 vs 14.18 (sd 1.49, ~1.2 sd), f1_1v 0.7823 vs
~0.7884. Suggestive but not established from one scalar run against four split
runs; and note the split cells achieving this are nearly degenerate.

## Cell count is the only real effect

Averaged over the four (equivalent) arms:

| cells | volPSNR | chamfer | hd95 |
|---|---|---|---|
| 128k | 34.131 | 2.0143 | 19.23 |
| 256k | 34.836 | 1.5197 | 14.18 |
| 512k | 34.976 | 1.1555 | 9.22 |

Geometry keeps improving faster than intensity: 256k->512k buys only +0.14 dB
volume PSNR (~2 sd) but chamfer 1.52->1.16 and hd95 14.2->9.2. Caveat: the 128k
arms start from `init_points: 32000` versus 64000 for 256k/512k, inherited from
the baselines (see `known_cell_count_caveat` in the manifest), so the cross-count
trend is indicative, not a controlled single-variable sweep.

## Why the split collapsed, quantitatively

The candidate gate (`face_continuity.py:382-388`) requires, for **both** cells of
a Delaunay pair, in `relative` mode with `delta_max_frac=0.5`:

- `contrast >= 0.01 * density_scale` (density_scale = 0.4728, the GT 99th pct)
- `contrast / density >= 0.1`, which reduces to `|tanh(raw_delta)| >= 0.1`
- `density >= 0.05 * density_scale`

Measured at step 10000:

| arm | tanh p50 | tanh p99 | tanh max | cells >= 0.1 | cells passing all |
|---|---|---|---|---|---|
| SC128_ctrl | 0.00031 | 0.02783 | 0.05310 | 0 | 0 |
| SC128_w3e-5d | 0.00026 | 0.01643 | 0.02898 | 0 | 0 |
| SC256_ctrl | 0.00311 | 0.07826 | 0.23285 | 821 | 579 |
| SC256_w1e-5d | 0.00004 | 0.00345 | 0.00543 | 0 | 0 |
| SC256_w3e-5d | 0.00313 | 0.07861 | 0.25586 | 841 | 598 |
| SC512_ctrl | 0.00036 | 0.02787 | 0.10183 | 1 | 0 |
| SC512_w3e-5d | 0.00043 | 0.02982 | 0.08317 | 0 | 0 |

The median split cell has two sides differing by ~0.03% of its own density. Even
the best 256k arms qualify only 579-598 cells out of 237k (0.24%), and requiring
*both* endpoints of a face leaves 17-18 faces out of 1.68M — hence `cand=18` and
a metric too sparse to compare arms. `SC256_w1e-5d` is 100x below its siblings,
the one place the density-weighted term visibly crushed the split; it also has
that group's worst hd95 (16.009).

Interpretation: with points free to move and densify, the optimizer explains the
data by relocating and adding cells rather than developing intra-cell contrast.
The frozen-64k regime had no such option, which is why splits mattered there.

## Failures and fixes made during the sweep

1. `SC512_w1e-5` finished 10000 steps at test PSNR 48.91, saved its checkpoint,
   then died in the *final metrics* block: `train.py` called
   `load_density_field(model_path)`, whose `radfoam.Triangulation(points)`
   from-scratch rebuild raised `TriangulationFailedError: divergent growth
   iterations` on a 462k-point cloud the live incremental triangulation had been
   training on happily. Fixed by building the field from the live model
   (`field_from_model(model)`, which pulls tets from `model.triangulation`);
   also drops a redundant full Delaunay build. The arm was then recovered from
   its checkpoint without retraining.
2. `evaluate_continuity.py` hit the same rebuild and failed deterministically on
   that checkpoint. It now retries with the exact perturbation policy
   `CTScene.update_triangulation` uses (`extent * 1e-5 * 3**failures`) and
   records `triangulation_retries` / `triangulation_perturbation` in its JSON.
   The recovered arm needed one retry at 4.4e-5.
3. Checkpoints store `relative_delta` but no `density_mode` key; consumers
   resolve mode via `relative_delta`. Any analysis defaulting `density_mode` to
   `absolute` silently uses the wrong delta formula.

## Follow-up options

Do not tune the continuity weight further on this schedule — weight is
irrelevant while the candidate pool is empty. The prerequisite question is
whether split contrast can survive densification at all:

1. Instrument `|tanh(density_delta)|` percentiles as a TB scalar, so collapse is
   visible during training instead of by post-hoc checkpoint forensics.
2. Test whether the collapse is a learning-rate/schedule artifact: raise
   `thin_surface_delta_lr_scale`, or start the split before densification ends
   rather than at 6000.
3. If splits genuinely have no work to do once geometry is free, the honest
   conclusion is that thin-surface splitting is a fixed-geometry tool, and the
   continuity prior only matters in that regime.
4. Only then re-run this matrix; and use >= 3 seeds per arm, since the effects
   of interest are smaller than the 0.15 dB noise floor measured here.

## Artifacts

- Runs: `KW60996:/code/lc64-radfoam/output/sweep_splitcell/SC{128,256,512}_*`.
- Per arm: `volume_hard_ss4.npy` + `volume_hard_ss4_metrics.json`,
  `surface_hard_ss4_metrics.json`, `face_continuity_eval.json` (split arms),
  `side_hard_ss4.npy`, `model.pt`, `run.log`, `DONE`.
- TensorBoard (no tunnel needed): `http://kw60996.hs.d0me.xyz:16007/` or
  `http://100.64.0.10:16007/`, all 13 runs under the `sweep_splitcell/` prefix.
- Configs and runner: `experiments/sweep_splitcell_v1/`.
