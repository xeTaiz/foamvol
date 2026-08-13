# Milestone 5 — COMPLETE: Adaptive Chest Delta Ablations

## Chest relative-delta results (hard-side split query, 256³)
| Arm | Relative rho | delta LR scale | Test PSNR | Split PSNR | Split Sobel | Split Dice | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| CH9 control | .5 | 0 | 49.06 | 35.06 | 17.36 | .858 | stable control |
| CH8 | .5 | .01 | 49.22 | 35.01 | 17.33 | .858 | no meaningful gain |
| CH10 | .5 | .1 | 48.94 | 34.55 | 16.45 | .856 | worse |
| CH11 | .25 | .01 | 48.63 | 35.02 | 17.33 | .858 | no meaningful gain |

## Raw additive-delta negative evidence
- CH3 raw delta .01: test1.03 / split27.94 (catastrophic).
- CH6 raw delta .0001: test32.72 / split34.46 (degrades).
- CH7 raw delta .0001 with delta L2=.1: test30.80 / split34.43 (degrades).

## Decision
- Bounded relative-delta parameterization is GPU-gradient-correct and stable, but all tested rho/LR settings are neutral or worse than no-learning control on chest.
- No candidate warrants seed replication or dataset breadth. Do not spend compute on broad sweeps.
- Geometry remains frozen because nonzero geometry LR degraded cube; it is untested/rejected on chest.

# Milestone 6 — Fixed-64k geometry-learning gate

## Controlled arms
- Dataset: chest, 75-view; fixed 64k cells, no densification; same seed/procedure.
- Scalar: `LC64_scalar_BAseed`; split: `LC64_split_qh_BAseed`.
- Split activation at iteration 1500: bounded relative delta (rho=.5), trainable quaternion + K=4 heights, fixed texel sites. Points remained trainable until the common freeze at 9500.
- Evaluation: `split_voxelize.py --blend_eps 0 --resolution 256 --supersample 4`; this is hard-side split-aware evaluation. Scalar used the same voxelizer's scalar fallback.

## Terminal status / recovery
- Scalar completed normally: final test PSNR 45.7880.
- Split completed all 10,000 optimization steps (final test PSNR 46.49) and wrote `model.pt`/`scene.ply`, then failed only in final diagnostic visualization: a *new* `radfoam.Triangulation(points)` raised `TriangulationFailedError: ambiguous triangulation` in `vis_foam.field_from_model`; no `metrics.txt` was written.
- The checkpoint is usable, but this is not a clean end-to-end completion. Commit `b85af93` reuses the live renderer triangulation for diagnostic topology rather than rebuilding it; it has been deployed to the worker for future runs.

## Hard-side 64k results
| Metric | Scalar | Split q+h | Split − scalar |
|---|---:|---:|---:|
| Volume PSNR | 32.8455 | 31.0343 | -1.8112 |
| SSIM 2D / 3D | .889295 / .889973 | .806042 / .801696 | -.083253 / -.088277 |
| Sobel PSNR / SSIM | 14.1513 / .453633 | 12.6885 / .327213 | -1.4628 / -.126420 |
| Dice | .774416 | .775036 | +.000620 |
| Chamfer (voxels) | 4.6040 | 2.9350 | -1.6690 |
| HD95 (voxels) | 36.6201 | 20.8177 | -15.8024 |
| F1@1v / F1@2v | .6302 / .7402 | .6618 / .7820 | +.0316 / +.0418 |

Artifacts: each arm has `volume_split_hard_ss4.npy` and `volume_split_hard_ss4_metrics.txt`; split additionally has `side_map_hard_ss4.npy` (mu-plus fraction .494).

## Post-hoc split-parameter health
- Mean effective relative delta was .00576 (p95 .02599); mean world-space `radius × L1(height)` was `3.79e-5` (p95 `1.40e-4`) versus median cell radius .1124.
- Quaternion neighbour coherence squared was .371 with sign-flip fraction .500. These values do not demonstrate a coherent, materially displaced learned internal surface; they are diagnostic evidence only, not a causal attribution of the surface metrics.

## Gate decision
- Geometry learning materially improved extracted surfaces (CD -36%, HD95 -43%, F1 +5–6%) but substantially regressed the prioritized hard-side volume and edge metrics (PSNR -1.81 dB; 2D/3D SSIM about -.08; Sobel PSNR -1.46 dB).
- The apparent projection PSNR gain (+.70 dB) does **not** override hard-side evidence.
- **Do not promote to the matched 32k ladder.** This is an explicit surface-vs-volume tradeoff, not evidence that the two-density representation reduces cell count at comparable volumetric quality. A future run requires a new optimization/representation hypothesis, not parameter-count breadth.

# Milestone 7 — Independent-side CUDA backward complete

- Commit `864a361` implements CUDA-native gradients for `raw_plus` / `raw_minus`, preserves point and thin-surface geometry gradients, and returns no legacy base-density gradient in independent mode.
- Fixed C++ binding allocation/output of geometry gradients when independent mode has no `density_delta` tensor.
- Added GPU finite-difference coverage for crossing/non-crossing paths, both `dp` signs, both selected sides, asymmetric logits, and near-air logits. Maximum observed raw-side relative error was `4.05e-4` in the committed fixture (`<3e-2` tolerance).
- Rebuilt for `sm_89`; independent forward/backward, checkpoint/schema, hard-freeze lifecycle, scalar/absolute/relative thin-surface, and crossing-branch regression suites pass.
- Stage A remained blocked on training/checkpoint-resume activation and split-aware voxelizer/evaluator support at this milestone.

# Milestone 8 — Stage A launched

- Commit `cb37d43` adds iteration-zero independent geometry/raw-side training activation, optimizer reattachment on checkpoint resume, independent geometry checkpoint round-trip, hard-side scalar/relative/independent voxelization, fixed GT-air metrics, intermediate checkpoint saves, resolved fixed-64k configs, and a static results site.
- E0 now passes fixed-ray projection/loss/point-gradient equivalence, symmetric raw-side gradients, crossing/noncrossing GPU FD, malformed/mixed checkpoint rejection, projection/query round-trip, no base-density gradient, and scalar/relative-zero/independent-zero hard-volume equivalence.
- Full focused/legacy regression suite and a two-step end-to-end independent training smoke pass. Fixed-64k configs use `init_points=final_points=64000` and `densify_until=-1`, preventing both densification and the iteration-0 `prune_only` path.
- Stage A scalar control `LC64A_scalar_F1500` started on KW60898 GPU 0 at commit `cb37d43` (PID `1407635`). A detached fail-fast chain evaluates it at hard-side 256³/SS4, then runs/evaluates relative-zero and independent-zero sequentially.
- Static results site is live through the worker tunnel at `http://orchestrator.hs.d0me.xyz:18765/`; queued/running/failed/completed arms and resolved config hashes are included.

# Milestone 9 — Stage A complete; Stage B running

- Scalar and relative-zero completed optimization and hard-side 256³/SS4 evaluation. Relative-zero terminal post-processing initially hit an optional frozen-point gradient diagnostic (`primal_points.grad is None`) after valid final artifacts were saved; commit `55468a6` skips that incompatible optional histogram under true hard freeze. The final checkpoint was recovered without retraining and evaluated successfully.
- Stage A hard metrics:
  - scalar: volume PSNR `30.8436`, Sobel PSNR `29.0728`, SSIM3D `.882655`, air MAE `.0005591`, air FPR `.0003406`;
  - relative-zero: volume PSNR `30.9757`, Sobel PSNR `29.1962`, SSIM3D `.881814`, air MAE `.0005638`, air FPR `.0003944`;
  - independent-zero (raw-side LR=0 safety control): volume PSNR `-5.4592`, Sobel PSNR `23.2984`, air MAE `1.99992`, air FPR `1.0`. This intentionally frozen-density arm confirms why native independent sides require a nonzero raw-side LR; it is not a compression candidate.
- Stage A prerequisites/gate passed for starting the density/LR screen: scalar/relative-zero stay within the ±0.30 dB safety window; independent CUDA/query/checkpoint E0 tests remain valid; all three evaluators/checkpoints are finite and structurally valid.
- Stage B six-arm configs are materialized at commit `0fa9414`. Four arms are running concurrently on KW60996 GPUs 0–3 (three relative LRs plus independent `5e-5`). KW60995 GPUs 1–2 are being provisioned for independent `2e-4` and `5e-4`; GPU 0 remains occupied by unrelated work.
- The site now labels future C/D/E entries as `queued (gated)`: these are planned conditional arms, not completed runs, and will only launch when their preceding gate passes.

# Milestone 10 — Controlled plane-orientation recoverability

- Added `experiments/orientation_recovery.py` (commits `db57e64` through `ba88c3b`) to separate CUDA gradient correctness, best-case teacher recoverability, and compatibility with measured CT projections.
- Built a GT-derived oracle on the fixed LC64 tessellation: strided 128³ GT samples, per-cell gradient-directed flat plane + offset + two constants, 4,096 strongest boundary cells retained (median side contrast `.1183`, median scalar-SSE improvement `68.9%`, median 44 GT samples/cell). Positions, both side densities, offsets/heights, and sites were frozen; only quaternions were optimized. Non-selected cells preserve the trained scalar density exactly.
- Directional quaternion Jacobian checks on realistic chest rays pass: relative errors `0.0063–0.0505` for 15°–60° perturbations and `0.0112` at 5°. Thus the production quaternion backward is numerically reasonable in the tested regime.
- Teacher targets generated by the exact split renderer are recoverable locally with 131,072 acquisition rays, 1,000 quaternion-only Adam steps, LR `2e-3`:
  - 5° → median `1.14°`, p90 `4.14°`, 92.9% within 5°;
  - 15° → median `2.96°`, p90 `11.60°`, 65.3% within 5°;
  - 30° → median `10.61°`, p90 `27.48°`;
  - 60° → median `43.96°`, p90 `62.39°`.
  Initial gradients point toward the oracle normal for 64–72% of active cells, degrading with perturbation size. The basin is useful but local rather than globally reliable.
- Against measured synthetic projections from the GT volume, optimization reduces projection MSE (`24–37%`) but does not reliably recover the independently fitted GT plane normals: 5° worsens to median `13.11°`, 15° to `16.01°`, 30° improves only to `24.34°`, and 60° to `52.98°`. This is not evidence of a broken backward; it shows that the independently per-cell GT plane fit is not the projection objective's joint optimum when all other cell values are frozen, and/or that orientation is weakly/non-uniquely identified by the 75-view line integrals.
- During harness validation, independent-mode oracle rendering showed extreme projection outliers even with equal sides. The final controlled experiment therefore uses the stable absolute mean+delta renderer with frozen nonnegative side values. That independent-mode behavior remains a separate implementation/representation concern and was not allowed to confound the orientation-gradient conclusion.

# Milestone 11 — Rays-per-cell orientation supervision

- Added `experiments/orientation_targeted_rays.py` (`400fb0e`) to directly vary distinct rays through each target cell. Selected the same 64 highest-scoring GT-oracle boundary cells, perturbed each normal by 15°, froze every other parameter, and optimized for 1,000 quaternion-only Adam steps at LR `2e-3` with constant 32,768 samples/step. Exact continuous cone rays pass through rejection-sampled points whose nearest Voronoi owner is verified to be the requested cell (`exact_owner_fraction=1.0`); measured projection values use matched continuous detector coordinates. Pools use nested angular coverage.
- Exact-renderer teacher targets confirm that focused rays provide strong local supervision even at very small counts:
  - 2 rays/cell (2 views): median `2.60°`, p90 `14.61°`, 65.6% within 5°;
  - 8 rays/cell (8 views): median approximately `0°`, p90 `5.35°`, 89.1% within 5°;
  - 32 rays/cell: median `1.26°`, p90 `12.58°`, 70.3% within 5°;
  - 128 rays/cell (all 75 views represented): median `.73°`, p90 `13.86°`, 68.8% within 5°.
  The non-monotonic p90/fraction reflects a fixed-step Adam optimization comparison and per-cell conditioning, but the main causal result is clear: guaranteeing just 8 angularly diverse hits per cell is dramatically stronger than sparse global random supervision.
- Measured CT targets do **not** recover the independently fitted GT plane merely by adding focused rays:
  - 2 rays/cell → median `41.19°`; 8 → `27.75°`; 32 → `18.57°`; 128 → `21.47°` (all from 15° initial perturbation).
  More coverage improves the measured result up to 32 rays/cell but does not reach the oracle and remains non-monotonic. This reinforces that the measured line-integral optimum conflicts with the independently per-cell GT plane when all neighboring values are frozen; targeted sampling fixes starvation/gradient strength, not joint-model mismatch or non-identifiability.
- Added `experiments/evaluate_targeted_orientation_volumes.py` (`4938614`) and evaluated every oracle/perturbed/recovered orientation at hard-side 256³/SS4. The tested 64-cell ROI contains 25,681 voxel centers (`0.1531%` of the volume), so global PSNR changes are necessarily tiny. GT-fitted oracle: global `30.5668 dB`, ROI `15.4697 dB`; 15° perturbation: global `30.5535`, ROI `15.2158`. Teacher recovery ROI PSNR by distinct rays/cell: 2=`15.2627` (+.0468 dB; 1.07% ROI-MSE reduction), 8=`15.3557` (+.1399; 3.17%), 32=`15.3583` (+.1424; 3.23%), 128=`15.5216` (+.3057; 6.80%). Measured recovery ROI: 2=`14.4032` (-.8127), 8=`14.9789` (-.2369), 32=`15.1446` (-.0712), 128=`15.3424` (+.1266; 2.87% ROI-MSE reduction). Outside-ROI PSNR remains ~`30.780 dB` for every arm. The tested cells are deliberately the hardest high-contrast boundary cells, explaining their low absolute ROI PSNR. `dB/ray` is not additive; report the nested budget curve and MSE reduction instead.
