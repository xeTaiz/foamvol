# LC64 Air-Artifact Split-Cell Diagnosis — Executable Queue v1

Authority: `specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v1.md`.
Machine manifest: `experiments/LC64-AIR-SPLIT-DIAGNOSIS-MANIFEST-v1.yaml`.
Evidence table: `experiments/LC64-AIR-EVIDENCE-TEMPLATE-v1.csv`.

## Dependency gates (P0)

- [ ] **E0 independent sides:** matched mean/difference optimizer, zero-split
  equality, crossing/noncrossing GPU FD, zero-air edge, checkpoint/query.
- [ ] **F0 true point freeze:** implement `points_freeze_at`; existing
  `freeze_points` only sets the point-LR scheduler horizon (`scene.py:update_learning_rate`)
  and does not freeze coordinates. Require post-freeze displacement ≤1e-7.
- [ ] **AIR evaluator:** fixed GT air/halo metrics at checkpoints 500/1500/2500/6000/final.
- [ ] **WEB:** browser-accessible `output/web-results/LC64-air-v1/index.html`
  verified before Stage A.

## Stage queue

| Stage | Eligible arms | Gate to next stage |
|---|---|---|
| A | A0 scalar F1500; A1 relative zero split F1500; A2 independent zero split F1500 | zero-split equivalence + AIR + WEB |
| B | Six fixed-F1500 representation × LR arms: relative/independent at 5e-5, 2e-4, 5e-4 | at least one advance winner; retain ≤1/mode |
| C | Up to two mode winners at 2e-3 | keep only if ≥5pp air improvement vs same-mode 5e-4 and advance gate |
| D | Reuse F1500 scalar/winner; launch matched scalar+winner at F500 and F0 | select one best split against schedule-matched scalar |
| E | E1 q+h at 0, 2e-4; E2 q+h unfreeze 500, 2e-4; E3 unfreeze 500, 5e-4 | each must pass before next; any failure stops E |

## Binding gates

Immediate stop: nonfinite, negative side density, invalid checkpoint/evaluator.
Checkpoint prune: held-out projection PSNR < matched scalar−2 dB, air FPR >2×
scalar, or air MAE >1.5× scalar. Advance: air FPR **or** MAE improves ≥15%
and hard-side volume/Sobel PSNR are each within 0.30 dB of matched scalar.

## Runtime

A0 is the timing pilot. Provisional 0.75 GPU-hour/run: 13–17 screen runs,
9.75–12.75 GPU-hours; replicate only final split/scalar pair with seeds 43/44.

No jobs launched by this queue document.
