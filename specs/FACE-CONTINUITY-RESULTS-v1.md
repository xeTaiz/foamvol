# Shared-face split continuity: implementation and 64k results

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

## Decision
- Implementation and GPU cache are technically successful and fast enough; no custom CUDA kernel is currently justified.
- The current continuity objective is a real geometric prior, not a no-op.
- Do **not** promote it as reconstruction-quality improvement: every arm loses volume PSNR and worsens strict-air MAE/FPR; most worsen surface distance/F1 metrics.
- Preserve it as an opt-in experimental regularizer (`weight=0` default).
- If continuing, test a gentler `~1e-5` weight or a late ramp, and redesign/normalize density consistency before increasing it. Use the density-emphasized `3e-5` arm as the present safety reference, not the stronger weights.

## Artifacts
- Remote: `KW60995:/code/lc64-radfoam/output/FC64_*`.
- Local TensorBoard logs and metrics: `output/ray_batch_scaling_v1_tb/KW60995/FC64_*`.
- TensorBoard: `http://camel.hs.d0me.xyz:16006/`.
