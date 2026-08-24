# LC64 Air-Artifact Split-Cell Diagnosis — Approved Execution Plan

**Status:** user-approved on 2026-07-23. Execute stages in order; do not launch a later stage before its gate is met.

## Goal
Diagnose visible nonzero density/artifacts in known-air chest regions at fixed 64k cells. Distinguish density parameterization, split-LR starvation, and moving-cell-border conflict. This is not a compression/cell-count claim.

## Constants and binding evaluation
- Chest CT, 75 views; fixed 64k initial/final cells; densification/pruning off; seed 42 screen then 43/44 finalists.
- Split state registered at iteration 0. Scalar controls are matched to every point-freeze schedule.
- Hard-side `split_voxelize.py --blend_eps 0 --resolution 256 --supersample 4` is binding. Save side maps.
- Report volume PSNR/SSIM2D/SSIM3D, Sobel PSNR/SSIM, Dice, CD, HD95, F1, and GT-defined fixed air-ROI MAE/P95/P99/false-positive rate. Held-out projection metrics are safety checks.

## Hypotheses
- H1: split artifacts arise substantially from LR starvation.
- H2: two independent nonnegative side densities improve air optimization relative to bounded mean+delta.
- H3: early point freeze gives internal surfaces a stationary frame and improves results.
- H4: geometry must be registered from step 0 but may require contrast bootstrap before receiving usable geometry gradients.

## E0 — independent-side implementation and test gate
Implement raw `mu_plus`/`mu_minus` storage with nonnegative activation, mean/difference-coordinate optimizer scheduling, checkpoint support, and same hard-side evaluator. Tests: zero-split scalar equivalence, crossing/noncrossing GPU FD gradients, zero-air edge case, checkpoint round-trip.

**Gate:** all tests pass and zero-split projections/volumes match scalar numerically. If blocked, run relative-only screen and record independent comparison blocked.

## A — equivalence/instrumentation
A0 scalar, A1 relative zero-split, A2 independent zero-split; each fixed 64k, `freeze_points=1500`.

**Gate:** zero-split controls match scalar; air metrics and static web artifact pipeline validate.

## B — density representation × difference-LR screen
q/heights frozen, split at 0, points freeze at 1500.

| Representation | difference LR |
|---|---:|
| bounded relative delta | 5e-5, 2e-4 (=point LR), 5e-4 (2.5× point LR) |
| independent sides | matched difference-coordinate values |

Independent-side mean coordinate uses the base-density schedule; this prevents a mean-learning advantage.

## C — conditional high-LR safety probe
Up to two Stage-B candidates at `2e-3` (10× point LR), bounds/clipping unchanged.

## D — stationary-frame schedule
Best density candidate and scalar controls at `freeze_points={0,500,1500}`.

## E — geometry handoff
Only after a density candidate passes the advancement gate:
- E1: q/heights learn from step 0 at `2e-4`.
- E2: q/heights registered at 0, unfreeze at 500 after contrast bootstrap, `2e-4`.
- E3: E2 at `5e-4`.

Log group gradient norms, parameter displacement, effective delta, world-space height extent, side occupancy, and air metrics. Do not mix a new continuity regularizer into this diagnosis.

## Gates
- Stop immediately: nonfinite values, negative side density, invalid checkpoint/evaluator.
- Prune: held-out projection PSNR >2 dB below matched scalar; air FPR >2× scalar; or air MAE >1.5× scalar after early window.
- Advance only if air FPR **or** air MAE improves >=15% and both hard-side volume PSNR and Sobel PSNR are within 0.30 dB of matched scalar.
- Replicate only finalists (seeds 43/44); claim only if direction holds >=2/3 seeds and aggregate gate passes.

## Static web evidence
Prepare `output/web-results/LC64-air-v1/` before long runs: status table, configs/commit/seed manifest, slices, air error/halo, side maps, curves, metrics, and visible pruned/failed arms. Verify user browser accessibility before Stage A and refresh after each stage.

## Authority
Experiment Designer maintains queue/evidence table; Compute Manager schedules only gate-authorized arms; Paper Writer updates living evidence after each stage. The orchestrator may add tightly controlled follow-ups only when the prior stage yields actionable evidence.