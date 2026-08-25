# Multi-scene feature follow-up plan v1

## Status and purpose

The corrected, centre-registered 256-cubed SS4 evaluator completed the
`multiscene_feature_v1` screen with 213 non-smoke runs and zero failures across
chest, pepper, and engine. This document records the resulting strategy ranking
and a bounded replicated follow-up. It does not authorize or launch another
sweep.

The evidence sources are:

- `specs/SWEEP-REVALIDATE-RESULTS-v1.md` for replicated chest results and the
  512k capacity check;
- `specs/MULTISCENE-FEATURE-SCREEN-RESULTS-v1.md` for the one-seed three-scene
  screen;
- `output/multiscene_feature_v1/summary.csv` on the retained V100 output tree;
- `output/multiscene_feature_v1_old/<scene>/<tag>/eval_vol.json` for runs moved
  out of the active TensorBoard tree during curation.

The screen has 216 completed rows: 213 non-smoke and three smoke controls. The
curated `summary.csv` currently contains 88 data rows; the other 128 completed
runs remain under `multiscene_feature_v1_old`. Curation changes presentation,
not the evidence set.

## Load-bearing interpretation correction

Two apparent parameter grids were semantic duplicates rather than active knob
sweeps.

1. `P_idw`, `P_idw_cap03`, `P_idw_cap05`, and `P_idw_cap10` all set
   `prune_variance_criterion: false` and `redundancy_threshold: 0.0`.
   `RadFoamScene.prune_and_densify` admits IDW candidates only when
   `cell_error < redundancy_threshold * density_scale`; a zero threshold admits
   none. Their different `redundancy_cap` values therefore did not affect
   redundancy pruning. These rows measure repeated executions with no
   redundancy pruning, not an IDW cap sweep.
2. `REFG_prune_eps005`, `REFG_prune_eps020`, `REFG_prune_eps050`, and
   `REFG_prune_eps100` change only `ref_guided_eps`, while all set
   `ref_guided_densify: false`. The implementation reads `ref_guided_eps` only
   inside the reference-guided densification branch. These rows repeat
   `REFG_prune`; they do not measure an epsilon response.

The corrected PSNR spread among those same-seed semantic duplicates is large:

| semantic duplicate family | chest range / sd | pepper range / sd | engine range / sd |
|---|---:|---:|---:|
| no redundancy pruning (`P_idw*`) | 0.1595 / 0.0672 dB | 0.3022 / 0.1352 dB | 0.1170 / 0.0512 dB |
| reference-guided prune (`REFG_prune*`) | 0.5155 / 0.2014 dB | 0.1055 / 0.0385 dB | 0.1294 / 0.0509 dB |

The complete groups require two archived evaluations: pepper `P_idw_cap03` is
35.1717 dB and chest `REFG_prune` is 34.7732 dB. The spread is direct evidence
of hidden runtime variability at fixed scene, seed, and effective training
configuration. Consequently, the old interpretation of `P_idw_cap05` as a cap
winner and any scene-wise ranking of `ref_guided_eps` are invalid. The measured
rows remain valid as repeated-run observations.

For reference, the five active baseline runs per scene give:

| scene | baseline PSNR mean ± sd | baseline Chamfer mean ± sd |
|---|---:|---:|
| chest | 34.8978 ± 0.0214 dB | 1.487842 ± 0.078984 |
| pepper | 35.0355 ± 0.0876 dB | 0.614972 ± 0.018705 |
| engine | 34.8692 ± 0.0695 dB | 0.451775 ± 0.003843 |

These seed-varying baselines do not directly estimate the same-seed runtime
spread exposed above. Stage 0 measures that spread explicitly.

## Evidence-based strategy ranking

### Retain

1. **Reduce the variance-based redundancy-pruning cap from the baseline's 0.05
   to 0.02–0.03.** This is the cleanest repeated equal-budget signal.
   `P_cap02` gains
   +0.3573/+0.4318/+0.4071 dB on chest/pepper/engine. `P_cap03` gains
   +0.3152/+0.3797/+0.3766 dB, and its earlier three-seed chest confirmation
   gained +0.2823 dB.
2. **TV at `1e-3`, as a scene-specific setting.** It gains +0.3842 dB and
   improves Chamfer by 0.0642 on pepper; it gains +0.5484 dB and improves
   Chamfer by 0.0198 on engine. Chest gains only +0.0366 dB and has materially
   worse Chamfer. This is not a universal default.
3. **Entropy-biased densification as a geometry specialist.** `D_bins3` was the
   replicated chest Chamfer winner among equal-budget feature arms: about
   0.297 lower Chamfer with +0.063 dB PSNR. `D_entonly` improved Chamfer further
   in the initial chest screen but lost PSNR. Multi-scene evidence is mixed;
   test entropy arms alone and in combinations rather than ranking them by one
   scalar.

### Investigate without claiming a win

- **Reference-guided pruning.** The prior three-seed chest PSNR gain did not
  survive the 512k check, and Chamfer worsened. The multi-scene epsilon rows are
  no-op repetitions with large spread. Meaningful pruning knobs are
  `redundancy_cap`, `ref_volume_edge_alpha`, `ref_volume_blur_sigma`, and
  `ref_volume_resolution`.
- **Actual IDW pruning versus no redundancy pruning.** The historical `P_idw*`
  rows did not activate the IDW removal branch. A nonzero
  `redundancy_threshold` is required before `redundancy_cap` can constrain IDW
  removals.
- **Cell-count scaling, in a separate budget class.** 1M/2M strongly improve
  engine and pepper PSNR; chest trades PSNR for surface quality. Capacity arms
  must never be ranked as free equal-budget feature gains.

### Do not prioritize

Thin-surface penalties, neighbor-variance families, border TV, gradient-only
sampling, and split-cell/LC64 ideas either failed replicated corrected
evaluation or use a different protocol/evaluator.

## Shared follow-up protocol

- Base config: `configs/fixed_final/256k.yaml`.
- Baseline pruning: `prune_variance_criterion: true` and
  `redundancy_cap: 0.05`. The `F_CAP*` family changes only this cap.
- Evaluator: corrected hard-SS4 256-cubed volume evaluation.
- Budget: `final_points: 256000` for every equal-budget arm.
- Scenes: chest, pepper, engine.
- Seeds: 42, 43, 44 for every baseline and candidate. Add 45 and 46 only for
  promoted finalists.
- Blocking: keep each scene/seed baseline and its candidates on the same
  worker/GPU and software environment. Randomize candidate order within each
  block. Analyze paired within-block deltas, not only raw means.
- Reference: create one immutable `BASE_s42/volume_hard_ss4.npy` per scene for
  all reference-guided pruning arms. Record its checksum. Keep
  `ref_volume_weight: 0.0`; the reference affects only the pruning mechanism.
- Outcomes: report PSNR and Chamfer jointly, plus exact achieved cell count.
  Keep unequal-budget capacity results in a separate table.
- Provenance: record resolved config hash, commit, worker/GPU, Torch version,
  CUDA version, reference checksum where applicable, and evaluator version for
  every run.

## Stage 0 — same-seed reproducibility gate

Run the unchanged 256k baseline five times per scene with seed 42, changing only
`experiment_name`, and distribute repeats across V100 workers.

Run count: 5 repeats × 3 scenes = **15 runs**.

Report PSNR and Chamfer range and sample sd, config hash, commit, GPU,
Torch/CUDA versions, and exact achieved cell count. Do not promote Stage-1
margins until this gate quantifies the fixed-seed runtime floor.

Stage 0 completed all 15 runs. Per the launch decision, engine `S0_BASE_r2`
on `gpu609-02` (34.1061 dB) is excluded as an outlier; the other four engine
runs give PSNR sd 0.05236 dB and Chamfer sd 0.002915. The resulting Stage-1
engine floors are 0.15 dB PSNR (the protocol minimum dominates `2 × sd`) and
0.005829 Chamfer. Chest and pepper retain their five-run Stage-0 floors:
0.150425/0.162207 and 0.157908/0.026762 for PSNR/Chamfer, respectively. This
exclusion and all exact values are frozen in `stage1_manifest.yaml` before
Stage 1 begins.

## Stage 1 — replicated main-effect matrix

Every setting runs on all three scenes with seeds 42, 43, and 44.
Non-default overrides are relative to `configs/fixed_final/256k.yaml`; scene,
seed, `experiment_name`, and output path are always resolved per run.

| family | tag | non-default overrides |
|---|---|---|
| control | `F_BASE` | none |
| variance cap | `F_CAP015` | `redundancy_cap: 0.015` |
| variance cap | `F_CAP020` | `redundancy_cap: 0.02` |
| variance cap | `F_CAP025` | `redundancy_cap: 0.025` |
| variance cap | `F_CAP030` | `redundancy_cap: 0.03` |
| actual IDW | `F_IDW005` | `prune_variance_criterion: false`, `redundancy_threshold: 0.005`, `redundancy_cap: 0.02` |
| actual IDW | `F_IDW010` | `prune_variance_criterion: false`, `redundancy_threshold: 0.01`, `redundancy_cap: 0.02` |
| actual IDW | `F_IDW020` | `prune_variance_criterion: false`, `redundancy_threshold: 0.02`, `redundancy_cap: 0.02` |
| no redundancy prune | `F_NOPRUNE` | `prune_variance_criterion: false`, `redundancy_threshold: 0.0`, `redundancy_cap: 0.0` |
| REFG | `F_REFG_C02_A5` | scene reference path, `ref_guided_pruning: true`, `redundancy_cap: 0.02`, `ref_volume_edge_alpha: 5`, `ref_volume_blur_sigma: 0.0` |
| REFG | `F_REFG_C02_A10` | scene reference path, `ref_guided_pruning: true`, `redundancy_cap: 0.02`, `ref_volume_edge_alpha: 10`, `ref_volume_blur_sigma: 0.0` |
| REFG | `F_REFG_C02_A20` | scene reference path, `ref_guided_pruning: true`, `redundancy_cap: 0.02`, `ref_volume_edge_alpha: 20`, `ref_volume_blur_sigma: 0.0` |
| REFG | `F_REFG_C03_A10` | scene reference path, `ref_guided_pruning: true`, `redundancy_cap: 0.03`, `ref_volume_edge_alpha: 10`, `ref_volume_blur_sigma: 0.0` |
| REFG | `F_REFG_C02_A10_B1` | scene reference path, `ref_guided_pruning: true`, `redundancy_cap: 0.02`, `ref_volume_edge_alpha: 10`, `ref_volume_blur_sigma: 1.0` |
| TV | `F_TV_3e4` | `tv_weight: 3e-4` |
| TV | `F_TV_1e3` | `tv_weight: 1e-3` |
| TV | `F_TV_3e3` | `tv_weight: 3e-3` |
| entropy geometry | `F_BINS3` | `entropy_bins: 3` |
| entropy geometry | `F_ENT60_B3` | `gradient_fraction: 0.2`, `idw_fraction: 0.2`, `entropy_fraction: 0.6`, `entropy_bins: 3` |
| entropy geometry | `F_ENT80_B3` | `gradient_fraction: 0.1`, `idw_fraction: 0.1`, `entropy_fraction: 0.8`, `entropy_bins: 3` |

The matrix has 19 candidates plus one baseline:
20 settings × 3 scenes × 3 seeds = **180 runs**.

### Activation assertions

Reject or repair an arm before interpreting its metrics when the run evidence
does not prove the intended branch was active.

- `F_IDW*`: the log contains `Redundancy prune (IDW threshold=` with nonzero
  removal at least once.
- `F_REFG*`: the log contains `Redundancy prune (ref_weight):` with nonzero
  removal at least once.
- `F_NOPRUNE`: the log contains no `Redundancy prune (` entry.
- `F_CAP*`: the log contains `Redundancy prune (variance):`; each removal count
  is nonzero and never exceeds the configured fraction of that event's current
  cell count.
- `F_TV*`: TensorBoard contains a nonzero `train/tv_loss` scalar at or after
  `tv_start`.
- Every run retains `final_points: 256000`, completes the unchanged 256k
  schedule without material cell-count underfill relative to its paired
  baseline, records its exact achieved cell count, and emits corrected
  `eval_vol.json`.

### Promotion rule

Use candidate-minus-baseline paired deltas for each scene/seed block.

- **PSNR candidate:** mean paired ΔPSNR exceeds
  `max(0.15 dB, 2 × Stage-0 baseline PSNR sd)` in at least two scenes; at least
  two of three seed deltas are positive in each qualifying scene; no scene mean
  is below -0.10 dB.
- **Geometry candidate:** mean paired Chamfer improvement
  (`baseline - candidate`) exceeds `2 × Stage-0 baseline Chamfer sd` in at
  least two scenes; at least two of three seed deltas have the correct sign; no
  scene loses more than 0.10 dB PSNR.
- Report scene specialists even when they do not promote as universal settings.

## Stage 2A — approved two-way interaction matrix

Stage 1 completed 180/180 runs with all activation assertions passing. Its
scene/seed-paired parents already have three seeds each, so Stage 2A reuses
those parent and baseline measurements on the same nine worker assignments.
Treat the leading pruning settings as a tie band rather than ranking their
sub-0.06 dB spread.

The approved parent set spans distinct outcomes:

- pruning alternatives: `F_CAP030`, `F_IDW020`, `F_NOPRUNE`,
  `F_REFG_C02_A20`;
- TV alternatives: `F_TV_3e4` (joint PSNR/Chamfer promotion) and `F_TV_1e3`
  (PSNR promotion);
- geometry specialist: `F_ENT60_B3` (Chamfer promotion).

Run every pruning × TV pair (8), every pruning × entropy pair (4), and both
TV × entropy pairs (2). Pruning alternatives are mutually exclusive and TV
weights are alternative scalar levels, so neither is crossed within its own
axis.

Run count: 14 combinations × 3 scenes × 3 seeds = **126 new runs**. Keep each
scene/seed block on its Stage-1 worker, randomize the 14 combination runs, and
retain the corrected evaluator, provenance, activation assertions, and exact
achieved cell counts.

A combination is complementary only when it beats its better single parent by
`max(0.10 dB, 2 × paired-difference sd)` in at least two scenes without a loss
greater than 0.10 dB in the third. For Chamfer, improvement is
`min(parent Chamfer) - combination Chamfer`, so positive values are better.
Beating the baseline alone is insufficient.

Three-way pruning × TV × entropy combinations remain gated on Stage-2A
complementarity. Their maximum size is 8 settings × 3 scenes × 3 seeds =
**72 runs**.

## Stage 3 — finalist confirmation and scaling

1. Add seeds 45 and 46 for every complementary Stage-2 combination and both of
   its single parents.
2. Select one equal-budget winner by mean paired ΔPSNR subject to the same
   no-regression condition.
3. At 512k, run the plain baseline, the equal-budget winner, and its better
   single parent on all three scenes with seeds 42/43/44:
   3 settings × 3 scenes × 3 seeds = **27 runs**.
4. Keep 1M/2M capacity results in a separate table. Never use them to claim a
   free feature gain.

The 512k check distinguishes a complementary interaction from a 256k-only
artifact. No further scaling stage is implied by this plan.
