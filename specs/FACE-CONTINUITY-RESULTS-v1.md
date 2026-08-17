# Shared-face split continuity: implementation, 64k results, and 64k/128k × frozen/unfrozen extension

## Implementation
Branch `face-continuity-v1`; principal commits:
- `9de3e69`: GPU shared-face cache and oriented zero-set/normal/density loss.
- `002b00f`: exact polygon-vertex crossing, candidate filtering, no per-step diagnostic synchronization, cache safety.
- `5dac797`: sparse continuity updates and persistent TensorBoard diagnostics.
- `80aefe8`: deterministic GT-Sobel spatial-anchor zooms (in branch history).

The geometry cache is built entirely with batched Torch operations on GPU: tetrahedron circumcenters, Delaunay-edge grouping, convex-hull edge rejection, face-polygon angular ordering, and area-stratified face quadrature. At runtime, the high-density side orients each local field; the loss is gated to meaningful-contrast neighbor pairs whose two zero sets cross the exact shared-face polygon and agree on high/low sides. It combines face zero-set position, high-side-oriented quaternion normal, and high/high plus low/low density consistency.

Validated tests:
- bounded dual-face construction, including nonidentity external permutation;
- flat-field value and analytic height gradients;
- zero loss for identical surfaces/densities;
- quaternion/density sign ambiguity resolved by the high-density direction;
- offset/density mismatch yields finite nonzero gradients;
- 6 new tests plus 25 existing thin-surface/air/independent tests passed.

## Performance and gradient preflight
On the completed 64k reference checkpoint:
- 388,098 cached finite faces;
- GPU build 0.40 s including a fresh triangulation (0.067–0.071 s from the live-training triangulation);
- 97 MB persistent cache, 1.53 GB temporary construction peak;
- contrast-prefiltered B=1024 update: approximately 127–154 eligible pairs;
- steady update: 7.8 ms forward, 79 ms including backward;
- applied every 8 CT steps, for an estimated average ~9.9 ms/step.

Raw face-only gradient norms at B=1024 were finite: density 0.0319, delta 0.00159, quaternion 0.282, height 0.123. Thus global weights `3e-5`, `1e-4`, and `3e-4` bracket weak through strong influence relative to measured CT quaternion/height gradients. Observed 10k training times were 30.2–31.2 minutes; regularized overhead was approximately 0.6–3.4%, confounded slightly by concurrent execution.

## Matched experiment
All arms:
- 64k fixed cells; same seed and 13B sampled rays;
- hard point freeze and split activation at 1500;
- corrected `densify_from: 0` (no historical exposure-warmup confound);
- face loss begins at 1599, every 8 steps;
- hard split-aware 256³/SS4 evaluation;
- six fixed central GT-Sobel TensorBoard anchor locations with learned/GT oblique panels and all local planes overlaid.

Weights use component ratios `zero=1`, `normal=.25`, `density=.1` unless named otherwise.

### Hard reconstruction metrics

| Arm | Vol PSNR | Sobel PSNR | SSIM3D | Dice | strict-air MAE | strict-air FPR | CD | HD95 | F1@1 / F1@2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 31.4391 | 29.7501 | .89538 | .95399 | .00053380 | .00017717 | 4.3426 | 33.422 | .6097 / .7379 |
| `3e-5` | 31.3539 | 29.6319 | .89366 | .95351 | .00054554 | .00023179 | 4.4133 | 37.796 | .6107 / .7393 |
| `3e-5`, geometry only | 31.2945 | 29.6412 | .89321 | .95317 | .00054794 | .00022790 | 4.9347 | 35.106 | .6057 / .7353 |
| `3e-5`, density ratio 1.0 | 31.2964 | 29.6766 | .89340 | .95345 | .00053934 | .00018576 | 4.7032 | 35.365 | .6098 / .7401 |
| `1e-4` | 30.7569 | 29.2306 | .88450 | .95167 | .00056969 | .00040547 | 4.9079 | 34.700 | .5836 / .7230 |
| `3e-4` | 31.2339 | 29.6286 | .89085 | .95308 | .00055331 | .00022320 | 5.2889 | 39.200 | .5852 / .7224 |

The density-emphasized low arm is the safest regularized variant: versus control, volume −0.143 dB, Sobel −0.073 dB, air MAE +1.0%, FPR +4.9%, F1@1 essentially neutral, and F1@2 +0.0022. It still does not beat control on the priority volume/air metrics.

### Final shared-face statistics
Means over 16 independently sampled B=4096 batches:

| Arm | zero-set loss | normal loss | density mismatch | side agreement |
|---|---:|---:|---:|---:|
| control | .2919 | .0666 | .1428 | .8442 |
| `3e-5` | .2476 | .0644 | .1506 | .8685 |
| `3e-5`, geometry only | .2371 | .0628 | .1326 | .8717 |
| `3e-5`, density ratio 1.0 | .2373 | .0633 | .1566 | .8704 |
| `1e-4` | .2155 | .0624 | .1577 | .8894 |
| `3e-4` | .1887 | .0503 | .1493 | .9057 |

The regularizer clearly changes its intended geometry: from control to `3e-4`, zero-set mismatch drops 35.4%, normal mismatch drops 24.4%, and side agreement rises 6.15 percentage points. The effect is dose-responsive. However, the configured density term does not reduce final density mismatch; increasing its component ratio to 1.0 also fails on the model-specific eligible population.

## Extension: 64k/128k × frozen/unfrozen matrix

The matched experiment above only tested hard-frozen 64k geometry. Two more axes were added: cell count (64k vs 128k, `init_points`/`final_points` scaled together, all else matched) and point-freeze state (`points_hard_freeze_at: 1500` frozen vs `-1` never-frozen; `freeze_points` learning-rate-schedule horizon left at `10000` unchanged, so unfrozen points keep receiving a decaying LR through the full run instead of going stationary at 1500). New configs: `configs/FC64_unfrozen_{control,w3e-5}.yaml`, `configs/FC128_{control,w3e-5,unfrozen_control,unfrozen_w3e-5}.yaml`. `w3e-5` = `zero=1, normal=.25, density=.1` at global weight `3e-5`, the density-emphasized safety reference from the matched experiment.

### Cache-staleness bug found and fixed
`build_thin_surface_face_cache` (`radfoam_model/scene.py`) hard-raised `RuntimeError` whenever `primal_points.requires_grad`, and separately keyed its cache reuse only on `(primal_points.data_ptr(), shape)`. Adam updates `primal_points` in place, so the pointer and shape never change even though the values do — the geometry cache would have silently gone stale (circumcenters/quadrature computed from an old point cloud) the moment points became trainable. Fixed by removing the hard guard and rebuilding the cache unconditionally whenever points are trainable (frozen runs keep the old signature-cached fast path, unchanged). This adds a full cache rebuild (~70–400 ms depending on cell count) at every face-loss application step instead of a no-op cache hit — the only way to get a geometrically-correct prior against moving points. Verified non-regression on the frozen path (0.0 point displacement, byte-identical to the original matched-experiment behavior) and verified unfrozen points actually move: `FC64_unfrozen_control` primal points displaced mean 0.0109 / median 0.00892 / max 0.102 (scene extent `[-1,1]^3`) between step 2000 and the final checkpoint, vs exactly 0.0 for the original hard-frozen control over the same range.

### Hard reconstruction metrics

| Arm | Cells | Frozen | Vol PSNR | Sobel PSNR | SSIM3D | Dice | strict-air MAE | strict-air FPR | CD | HD95 | F1@1 / F1@2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (matched exp.) | 64k | yes | 31.4391 | 29.7501 | .89538 | .95399 | .00053380 | .00017717 | 4.3426 | 33.422 | .6097 / .7379 |
| `3e-5` (matched exp.) | 64k | yes | 31.3539 | 29.6319 | .89366 | .95351 | .00054554 | .00023179 | 4.4133 | 37.796 | .6107 / .7393 |
| unfrozen control | 64k | no | 33.9384 | 32.7362 | .92929 | .96112 | .00047581 | .00006301 | 3.0861 | 25.559 | .6883 / .7911 |
| unfrozen `3e-5` | 64k | no | 34.0042 | 32.8163 | .93048 | .96129 | .00047399 | .00007038 | 2.9852 | 24.712 | .6916 / .7951 |
| control | 128k | yes | 32.1982 | 31.0292 | .91062 | .95728 | .00050247 | .00016203 | 3.9497 | 31.797 | .6556 / .7742 |
| `3e-5` | 128k | yes | 32.1613 | 30.9315 | .91007 | .95694 | .00050069 | .00016100 | 3.7678 | 30.403 | .6604 / .7801 |
| unfrozen control | 128k | no | 34.4896 | 33.5448 | .93685 | .96258 | .00048436 | .00009595 | 3.0076 | 26.798 | .7304 / .8248 |
| unfrozen `3e-5` | 128k | no | 34.5805 | 33.6474 | .93804 | .96282 | .00048111 | .00008163 | 2.9383 | 25.855 | .7287 / .8233 |

### Continuity statistics (mean, final checkpoint)

| Arm | Cells | Frozen | zero-set loss | normal loss | density mismatch | side agreement |
|---|---:|---|---:|---:|---:|---:|
| control (matched exp.) | 64k | yes | .2919 | .0666 | .1428 | .8442 |
| `3e-5` (matched exp.) | 64k | yes | .2476 | .0644 | .1506 | .8685 |
| unfrozen control | 64k | no | .2625 | .0536 | .1813 | .8524 |
| unfrozen `3e-5` | 64k | no | .2142 | .0461 | .1688 | .8790 |
| control | 128k | yes | .2791 | .0488 | .1547 | .8476 |
| `3e-5` | 128k | yes | .2373 | .0431 | .1401 | .8714 |
| unfrozen control | 128k | no | .2464 | .0318 | .1733 | .8583 |
| unfrozen `3e-5` | 128k | no | .1979 | .0273 | .1567 | .8855 |

### Findings
- **Letting geometry keep moving past the split-activation freeze dominates every other lever tested.** Unfreezing points is worth +2.50 dB volume PSNR at 64k and +2.29 dB at 128k (control vs. control) — an order of magnitude larger than any regularizer weight tested, and it also improves every other metric (SSIM3D, Dice, air MAE/FPR, chamfer, HD95, F1@1/F1@2) simultaneously. This is the single largest lever in the whole experiment set.
- **Doubling cell count gives smaller, still-positive gains:** 64k→128k control is +0.759 dB frozen, +0.551 dB unfrozen — diminishing returns, and much smaller than unfreezing.
- **The regularizer's sign flips with freeze state.** With frozen geometry (both 64k, matched experiment, and 128k here), `3e-5` costs volume PSNR (−0.085 dB at 64k, −0.037 dB at 128k), matching the original Decision. With unfrozen geometry, `3e-5` instead *gains* volume PSNR at both cell counts (+0.066 dB at 64k, +0.091 dB at 128k) and also improves Sobel PSNR, SSIM3D, Dice, air MAE, chamfer, and HD95 at both cell counts; F1@1/F1@2 are a wash (+/− ≤0.002 at 128k). This is consistent with the geometric interpretation: forcing continuity onto a *frozen* surface can only ever compromise fit to the CT data, whereas geometry that is still trainable can locally relax to satisfy the shared-face constraint at near-zero cost to data fit, and evidently helps slightly.
- **Correction to the original density-mismatch conclusion (line-67-era text above, based on the single 64k-frozen control/`3e-5` pair): that pair was an outlier, not the rule.** Density mismatch control→`3e-5`: 64k frozen **+5.5%** (worse, the original finding), but 128k frozen **−9.4%**, 64k unfrozen **−6.9%**, 128k unfrozen **−9.6%** (all better). 3 of the 4 (cell count × freeze) replications show the density term reducing mismatch by the expected ~7–10%, in the same direction as the zero-set/normal terms. The density component is not obviously broken; the 64k-frozen matched-experiment pair (also the pair the `1e-4`/`3e-4`/geometry-only/density-ratio-1.0 ablations were built around) looks like single-seed noise or an interaction specific to that exact 64k-frozen configuration, not a general design flaw. Revises the "redesign/normalize density consistency" recommendation below — a seed-replication pass, not a redesign, is the right next step.
- **The regularizer changes the same underlying quantity regardless of freeze state**: zero-set/normal mismatch drop and side agreement rises going from control→`3e-5` in all four (cell count × freeze) cells, at a similar relative magnitude to the original dose-response curve — the fix did not change what the loss optimizes, only whether the geometry is allowed to respond to it.
- Single seed per arm (matching the original matched-experiment protocol). The unfrozen `3e-5` gains are small (0.07–0.09 dB) and directionally consistent across both cell counts but have not been replicated across seeds; treat as suggestive, not conclusive.

## Decision
- Implementation and GPU cache are technically successful and fast enough; no custom CUDA kernel is currently justified.
- The current continuity objective is a real geometric prior, not a no-op.
- On **frozen** geometry (64k matched experiment, 128k extension): do **not** promote it as a reconstruction-quality improvement — every frozen arm loses volume PSNR and worsens strict-air MAE/FPR; most worsen surface distance/F1 metrics.
- On **unfrozen** geometry (64k, 128k extension): the density-emphasized `3e-5` arm is a small net positive (+0.07–0.09 dB volume PSNR, improved SSIM3D/Dice/air MAE/chamfer/HD95, neutral F1) versus its own control. Single-seed evidence only — worth a seed-replication pass before promoting.
- Unfreezing points past the split-activation freeze (`points_hard_freeze_at: -1`) is a far larger and unconditional win (+2.3–2.5 dB volume PSNR) independent of the continuity regularizer; if pursuing further gains, prioritize this over regularizer tuning.
- Preserve the regularizer as an opt-in experimental term (`weight=0` default) either way.
- If continuing, test a gentler `~1e-5` weight or a late ramp. Do not redesign the density-consistency term on the strength of the original 64k-frozen pair alone — the extension matrix shows it reduces density mismatch as intended in 3 of 4 replications (see Findings); that pair looks like an outlier. Use the density-emphasized `3e-5` arm as the present safety reference, not the stronger weights.

## Independent verification of the extension-matrix numbers
Prompted by the volume-PSNR jump looking large versus prior specs (`RAY-BATCH-SCALING-RESULTS-v1.md`'s 64k-scalar reference is 28.98 dB): that doc's own §"Important configuration caveat" says its arms used a nonzero `densify_from` warmup and are ~1.86 dB below a `densify_from: 0` baseline (i.e. not the right comparison point). The correct baseline is this doc's own 64k-frozen control at `densify_from: 0`, 31.4391 dB — the extension's unfrozen arms (33.9–34.6 dB) are +2.3–2.5 dB above *that*, not above the ray-batch-scaling number.
- **Dataset check**: all 8 extension configs use identical `dataset: r2_gaussian`, `data_path: r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone`, `densify_from: 0` — confirmed via direct diff of the shipped YAML files; no accidental dataset/protocol swap.
- **Output-file check**: `volume_hard_ss4.npy` for `FC64_control` vs `FC64_unfrozen_control` are different files (different SHA-256, mean abs difference 0.0095), ruling out a stale/cached-file bug.
- **Visual check**: `volume_hard_ss4_slices.png` for `FC64_unfrozen_control` shows a real chest CT (lungs, heart, mediastinum, ribs, spine) closely matching GT, with residual error concentrated in fine vascular/lung-marking detail and the expected Voronoi-facet pattern — not a degenerate or trivial reconstruction.
- **Second, independent metric family agrees**: TensorBoard's `test/psnr` (2D rendered-image PSNR against held-out camera views, computed inside `train.py` during training, a completely different code path from the `split_voxelize.py`-based hard volumetric eval reported above) shows FC64_control plateauing at 44.87 dB vs FC64_unfrozen_control at 47.58 dB — a +2.71 dB gap, consistent in sign and magnitude with the +2.50 dB volumetric-PSNR gap reported above.
- **A red herring, resolved**: `metrics.txt`'s "Vol R2 PSNR" block is identical (35.8512 dB, etc.) across every run including a 40-iteration smoke test. This is not a caching bug — `train.py` loads it from `load_r2_volume(data_path)`, a fixed precomputed reconstruction from the competing R2-Gaussian baseline method, evaluated against the same GT every run purely for reference; it is unrelated to our model and was never part of any number reported in this doc.
- TensorBoard is live on the worker (`tensorboard --logdir=output`, covers every `FC64_*`/`FC128_*` run) and tunneled; see Artifacts.

## Artifacts
- Remote: `KW60995:/code/lc64-radfoam/output/FC64_*`, `output/FC64_unfrozen_*`, `output/FC128_*`.
- TensorBoard: live at `orchestrator.hs.d0me.xyz:16006` (tunnel `fc-tensorboard` → `KW60995:16006`, `logdir=output`, all runs).
- Hard-surface Chamfer/HD95/F1 metrics for the extension matrix were computed with the new standalone `experiments/face_continuity_v1/eval_hard_surface.py` (reuses `train.py`'s `compute_surface_metrics` against the existing `volume_hard_ss4.npy`/ground truth, independent of `split_voxelize.py`'s own PSNR/SSIM/air path); verified to reproduce the original `FC64_w3e-5` `surface_hard_ss4_metrics.json` byte-for-byte before use on the new arms.
