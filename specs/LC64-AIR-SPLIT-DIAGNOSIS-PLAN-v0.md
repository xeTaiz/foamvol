# LC64 Air-Artifact Split-Cell Diagnosis — Draft Plan

## Research question
Why do split cells leave visible attenuation artifacts in known-air regions despite multiview CT supervision, and is the cause (a) density parameterization, (b) split/geometry learning-rate starvation, or (c) cell-border versus internal-surface optimization conflict?

This is a new diagnosis cycle, not a cell-count compression claim.

## Evidence motivating the cycle
The matched fixed-64k chest gate used no densification and equal seed/procedure. Split q+h improved extracted surfaces but lost hard-side volume/edge quality (PSNR 31.03 vs 32.85; SSIM3D .802 vs .890). At activation, actual LRs were base density `5e-2`, points `2e-4`, delta `5e-5`, and quaternion/heights `5e-6`. Post-hoc split delta and world-space height offsets were tiny. This makes LR/schedule starvation plausible but does not establish causality.

## Invariants for every scientific arm
- Chest CT, same 75-view data, 64k fixed initial/final cells, no densification or pruning.
- Deterministic seed 42 for screen; finalists repeat seeds 43 and 44.
- `thin_surface_start=0`: split state exists from the beginning. Parameters may be registered at 0 but a scheduled geometry-unfreeze is an explicit factor, not silently changed.
- Scalar control matched to each point-freeze schedule and all non-split settings.
- Hard-side evaluation only: 256^3, supersample 4, `blend_eps=0`; save side map for split arms.
- Report volume PSNR/SSIM2D/SSIM3D, Sobel PSNR/SSIM, Dice, CD, HD95, F1; held-out projection metrics are safety checks only.
- Fixed air-region protocol: derive an air ROI exclusively from GT (threshold/erosion and thresholds frozen before Stage B); report air MAE, air P95/P99 absolute density, and false-positive fraction. Data Expert must specify and test it.
- Each run writes resolved config, commit/build hash, seed, terminal state, TensorBoard logs, hard-side volume, metrics, and fixed-window slices.

## Hypotheses
| ID | Hypothesis | Falsifier |
|---|---|---|
| H1 | Current artifacts are mainly split-LR starvation. | At point-rate or higher split LR, air error does not improve while volume/edge metrics remain worse than matched scalar. |
| H2 | Independent nonnegative side densities optimize air/background better than bounded mean+delta. | At matched mean/difference-coordinate LR and freeze schedule, independent sides fail to improve air metrics or hard-side volume. |
| H3 | A stationary tessellation is required for split geometry. | Early point freeze is no better than late/no early freeze in matched arms. |
| H4 | Internal geometry needs a contrast bootstrap and should not freely chase moving borders at step 0. | Geometry active from step 0 is as stable and effective as a contrast-bootstrap/unfreeze schedule. |

## Engineering gate E0: independent-side parameterization
Before any independent-side experiment, implement a true `mu_plus, mu_minus >= 0` parameterization with raw parameters and softplus activation. It must:
1. preserve scalar equivalence at zero-split initialization (`mu_plus=mu_minus=mu_bar`);
2. expose mean and difference coordinates for matched optimizer scheduling, rather than giving the independent arm an accidental mean-LR advantage;
3. round-trip in checkpoints and be evaluated by the same hard-side query;
4. pass CPU/GPU finite-difference tests for crossing and non-crossing rays, plus an air/zero-density edge case.

Acceptance: all tests pass; zero-split rendered projections and hard-side volumes agree with scalar within numerical tolerance. If E0 fails, Stage B proceeds with relative-delta only and records independent-side comparison as blocked.

## Staged experiment queue

### Stage A — equivalence and instrumentation (3 short/full controls)
- A0 scalar, point freeze 1500.
- A1 relative-delta split at 0, all split LRs 0, point freeze 1500.
- A2 independent sides at zero split / difference LR 0, point freeze 1500 (after E0).

Purpose: establish identical scalar behavior and validate checkpoint/evaluator/website artifacts before changing a learning factor.

### Stage B — density parameterization × difference-LR screen (6 arms + scalar control)
Geometry q/heights fixed. Points train through 1500 then freeze. Split is active at 0.

| Parameterization | Difference-coordinate LR | Intended absolute initial LR |
|---|---:|---:|
| bounded relative delta | .01 | `5e-5` (prior LC64 setting) |
| bounded relative delta | .04 | `2e-4` (= point LR) |
| bounded relative delta | .10 | `5e-4` (2.5× point LR) |
| independent sides | .01 | matched difference LR |
| independent sides | .04 | matched difference LR |
| independent sides | .10 | matched difference LR |

The independent arm's mean coordinate uses the base-density schedule; only the difference coordinate takes this sweep. This separates expressivity from mean-density optimization.

### Stage C — conditional high-LR safety probe (up to 2 arms)
Only Stage-B candidates meeting safety gates receive difference LR `.40` (`2e-3`, 10× point LR), with clipping/bounds unchanged. This tests the user's high-LR hypothesis without spending the full budget on likely unstable arms.

### Stage D — point-freeze handoff (winner + matched scalar)
For the best Stage-B/C parameterization/LR, compare `freeze_points ∈ {0, 500, 1500}`. `freeze_points=0` means fixed tessellation from the start; the scalar is rerun for each schedule.

### Stage E — geometry schedule handoff (winner + matched scalar)
Only if density-only split reduces air artifacts without violating volume gate:
- E1: q/heights train from step 0, geometry LR `.04` (= point LR), points freeze at selected Stage-D value.
- E2: q/heights registered at 0 but unfreeze at 500 after contrast bootstrap; geometry LR `.04`.
- E3: E2 at geometry LR `.10` (2.5× point LR).

Bound height magnitude, gradient clipping, effective-delta L2, and height penalty remain active. Add read-only logs: per-group gradient norms, parameter displacement from initialization, effective delta, world-space height extent, side occupancy, and air metrics at fixed intervals. Do not add cross-cell coherence regularization in this sweep: that is a separate representation hypothesis.

## Gates and stopping rules
- Numerical safety: immediately prune NaN/Inf, side-density negativity, invalid checkpoint, or evaluator failure.
- Projection safety: prune if held-out projection PSNR is >2 dB below matched scalar at a common checkpoint.
- Air safety: prune if air false-positive fraction exceeds 2× scalar or air MAE exceeds 1.5× scalar after the early window.
- Advancement: a learned split candidate must improve air false-positive fraction or air MAE by >=15% **and** be within 0.30 dB in both hard-side volume PSNR and Sobel PSNR of its matched scalar. Surface metrics are reported but cannot override this gate.
- Replication: only finalists get seeds 43/44. Claim only if the direction holds on at least 2/3 seeds and final aggregate meets the gate.

## Web evidence site
Prepare `output/web-results/LC64-air-v1/` as a static site before Stage A. It includes an index/summary table with scalar and split status, resolved configs, per-arm fixed-window scalar/split/GT/air-error/side-map slices, loss/diagnostic curves, final metrics, manifest JSON, and explicit pruned/failed arms. A fresh static server must point at this directory; verify it is reachable from the user's tailnet browser before starting long runs.

## Budget and decisions
- Expected screen: E0 tests + 3 Stage-A arms + 6 Stage-B arms + 0–2 Stage-C arms + 3 Stage-D scalar-matched comparisons + 0–3 Stage-E arms. Run sequential stages, not one confounded monolithic factorial.
- Full final training applies only to passing short safety windows; estimate/update GPU-hour budget after a measured Stage-A runtime.
- Decision outputs: accept a candidate for replication, reject current parameterization/schedule, or introduce one new factor (e.g. continuity prior) in a separately planned cycle.
