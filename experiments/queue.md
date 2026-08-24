# Experiment Queue — Split-Cell / Thin-Surface Validation (v2)

Living queue. Source of truth for what to run next. Driven by
`specs/SPLIT-CELL-EXPERIMENT-PLAN-v2.md` and
`specs/SPLIT-CELL-EXECUTION-LOG.md`. Update status inline as runs
complete; mirror concise deltas to `.pi/team-updates.md`.

Conventions:
- Configs: `configs/best428_nointerp.yaml` (scalar baseline),
  `configs/best428_thinsurface.yaml` (thin K=4 + warm-start).
- Run command template:
  `micromamba run -n radfoam python train.py -c <cfg> --experiment_name <name> [--override ...]`
- Artifacts per run: `output/<name>/{config.yaml, model.pt, metrics.txt, events.out.tfevents.*}`.
- Metrics contract: `metrics.txt` keys listed in `train.py:1470-1612`
  (proj PSNR/SSIM/RMSE; Vol raw+IDW PSNR/SSIM/Sobel/Dice/CD/F1; thin-surface
  TB scalars `train/thin_surface_delta_loss`, `train/thin_surface_height_loss`).
- Compute target: `kw995` / `kw996` (RTX 6000 Ada) via Compute Manager `wh_` tools.
- Seed: one deterministic seed for P0/P1 smoke (per v2 plan); add 3-seed
  replication before publishing claims (P2).
- K=4 only until K-plumbing + gradcheck extended (v2 plan).

Status legend: [ ] queued · [~] running · [x] done · [!] failed/blocked.

---

## P0 — Engineering & correctness gates (BLOCKING)

These must land before any scientific claim. Tracked in
`specs/SPLIT-CELL-EXECUTION-LOG.md` Phase 0. Experiment Designer owns the
*test definitions*; Research Implementer owns the code.

**Current code state (execution log, `surface@503a380`):** P0-A/B/C/D are
formally green. P0-E executed on thin checkpoints (thin1a split PSNR 32.05;
delayed thin1b split PSNR 9.47), but scalar calibration remains required.
P0-F diagnostics remain pending. Delayed thin1b is a hard regression: it
reached test PSNR 30.46 at step 4999, then after start=6000 collapsed to
**−17.78** test PSNR / **18.09** VolRaw / .1275 Dice (split PSNR 9.47,
inside/outside means .187/.067, max density 7.45). It blocks chest and all
broad sweeps; only the B1-R rescue ladder is eligible.

- [x] **P0-A · Checkpoint round-trip** — save/load `density_delta`,
      `quaternions`, `texel_sites_2d`, `texel_heights`, `_thin_surface_active`,
      `_thin_K`, `_last_top_eigvec`. Verify reload reproduces `forward()`
      output bit-for-bit on a 2-cell scene.
  - Owner: Implementer · Deps: none · Artifacts: `test/test_checkpoint_thin.py`
    passing.
  - Why: `train.py` eval runs in-process, but reload/eval paths silently
    ignore thin-surface state today (execution-log risk register).

- [x] **P0-B · K=4 CUDA/config plumbing** — confirm `thin_surface_K` reaches
      `TraceSettings.thin_K` and kernel dispatch uses K=4. Guard K≠4 with a
      clear error until P2-K8 extends it.
  - Owner: Implementer · Deps: none · Artifacts: assert in
    `scene.py:initialize_thin_surface` + binding smoke.

- [x] **P0-C · `density_delta` grad shape/autograd contract** — verify
      `dL/d(density_delta)` shape (N,1), no broadcast silently zeroing grads;
      verify μ₊/μ₋ `max(...,0)` clamp doesn't kill gradient below zero.
  - Owner: Implementer · Deps: none · Artifacts: unit assertion.

- [x] **P0-D · Finite-difference gradient check** — 1-cell, 1-ray scene;
      central-difference vs `loss.backward()` for each of
      {density_base, density_delta, quaternions, texel_sites_2d,
      texel_heights}. fp32 rel-err < 1e-3. Probe grazing-ray branch
      (|n·d|<1e-3) separately.
  - Config source: hand-built in test (no YAML).
  - Command: `micromamba run -n radfoam python -m test.test_thin_surface_gradcheck`
  - Owner: Implementer writes, Experiment Designer reviews design.
  - Deps: P0-B, P0-C. Artifacts: passing test, console table of rel-errors.
  - **Hard gate: nothing in P1/P2 runs until this is green.**

- [~] **P0-E · Split-aware voxelization** — new evaluator: for each voxel
      sample, evaluate the signed side of the owning cell's internal surface
      and pick μ₊/μ₋ accordingly; optional supersampling per voxel. Mirror
      `vis_foam.voxelize_volumes` query/supersample pattern but respect the
      two-density split. Skip meshing (v2 plan).
  - Owner: Implementer · Deps: P0-B. Artifacts: `radfoam_model/voxelize_split.py`
    + a `vol_split_*` metric block appended to `metrics.txt`.
  - Why: scalar voxelization hides the exact contribution thin surfaces make.

- [ ] **P0-F · Diagnostics** — TB scalars for: active-surface fraction
      (cells with |h|₁ > gate_tau), |δ| p50/p95/max, ‖texel_heights‖₁ p95,
      quaternion-norm drift, normal coherence (avg |n·n_neighbor|). Plus a
      zero-init inertness check: with δ=0, h=0, output == scalar baseline to
      fp32 noise.
  - Owner: Implementer · Deps: P0-B. Artifacts: TB scalars + inertness test.

---

## Phase 1 — First runs (smoke + scientific signal)

Order is fixed by v2 plan: cube first, then chest. One seed.

### P1-CUBE — cube sanity via `old/test_cube.py`

- [~] **EXP-CUBE-1 · Scalar cube baseline** — runs tests 1a/1b/2a/2b from
      `old/test_cube.py` (existing harness, builds config in-memory, calls
      `train.py`). **Completed:** 1a manual: test PSNR 58.86, VolRaw PSNR
      89.52, Dice 1.0000, 171.7 s; 1b random/densified: test PSNR 40.94,
      VolRaw PSNR 36.07, Dice .9919, 175.0 s. Tests 2a/2b remain optional
      representation coverage, not a chest gate.
  - Config source: `old/test_cube.py:base_config()` (no thin fields).
  - Command: `cd old && micromamba run -n radfoam python test_cube.py`
    (or `--test 1a` to stage).
  - Artifacts: `output/cube_sanity/<test>/metrics.txt` + `config.yaml`.
  - Deps: none (independent of P0 — pure scalar). Run immediately to occupy
    a GPU while P0 engineering proceeds.
  - Pass criterion: 1a/2a (manual points) reconstruct to projection PSNR
    > 60 dB (perfect representability); 1b/2b (random+densify) converge to
    PSNR > 50 dB.

- [~] **EXP-CUBE-2 · Thin-surface cube** — thin 1a with `thin_surface_start=0`
      completed (K=4; test PSNR 10.38, VolRaw PSNR 41.80, Dice .9981, Sobel
      PSNR 39.29). This is a **stress-regression result**, not the staged test:
      it rejects "activation at iteration 0 is safe" but does not test the
      intended post-densification start=6000 configuration.
  - **Next controlled comparison: EXP-CUBE-2b delayed thin 1b.** Same random
    init/densification/seed/budget as scalar 1b; K=4; `thin_surface_start=6000`.
  - Config source: `old/test_cube.py --thin --test 1b --thin-start 6000`
    at `surface@503a380` (or the exact equivalent supported flags); archived
    emitted `config.yaml` is authoritative.
  - Command template: `micromamba run -n radfoam python old/test_cube.py --thin --test 1b --thin-start 6000`.
  - Artifacts: `output/cube_sanity/single_cube_random_thin_delayed/` containing
    config, checkpoint, metrics, TB event; split evaluator output under the
    same run or `output/splitcell_validation/`.
  - Deps: P0-E end-to-end evaluator + P0-F diagnostics. **This is the sole
    remaining cube gate before chest; do not run thin 2a/2b first.**

### P1-CHEST — clean comparison matrix on `0_chest_cone`

Dataset: `r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone`
(already set in both configs). 10k iters, 512k final points, ~1 hr/run
(baseline chest = 3.3 ks per `output/summary.csv`).

Four-cell matrix isolates boundary-alignment effects from thin-surface effects:

| ID | Config source | Overrides vs best428_nointerp | Purpose |
|----|---------------|-------------------------------|---------|
| **EXP-CH-1** | `configs/best428_nointerp.yaml` | — | Scalar baseline |
| **EXP-CH-2** | `configs/best428_nointerp.yaml` | `top_eig_align_weight=1e-2` | Baseline + boundary alignment (warm-start source) |
| **EXP-CH-3** | `configs/best428_thinsurface.yaml` | `top_eig_align_weight=0.0` | Thin K=4, **no** warm-start |
| **EXP-CH-4** | `configs/best428_thinsurface.yaml` | — | Thin K=4 **with** warm-start (default) |

- Command template (per cell):
  `micromamba run -n radfoam python train.py -c <cfg> --experiment_name chest_<id> <overrides>`
  e.g. EXP-CH-2:
  `... -c configs/best428_nointerp.yaml --experiment_name chest_CH2 --top_eig_align_weight 1e-2`
  (configargparse accepts CLI overrides of YAML keys.)
- Artifacts: `output/chest_<id>/{config.yaml, model.pt, metrics.txt, tb}`.
- Deps: P0-D (gradcheck), P0-E (split voxelization), P0-F (diagnostics) for
  EXP-CH-3/4. EXP-CH-1 can start as soon as a GPU is free.
- Seed: deterministic (single) for smoke; re-run 3 seeds for the winning
  config only (see P2-SEED).

**Primary metrics for the chest matrix:**
1. Projection: `test PSNR/SSIM/RMSE` (metrics.txt lines 1-6).
2. Volume: `Vol Raw PSNR`, `Vol Raw SSIM`, `Vol Raw Sobel PSNR` (edge
   sharpness — the hypothesis test for thin surfaces), `Vol Raw F1_1v`.
3. Split-aware volume (P0-E): `Vol Split PSNR/SSIM/Sobel` — append to
   metrics.txt.
4. Diagnostics (CH-3/4 only): active-surface fraction, |δ| p95, ‖h‖₁ p95,
   normal coherence — from TB.

**Decision rules (chest matrix):**
- **CONTINUE to P2** if EXP-CH-4 Vol Raw Sobel PSNR > EXP-CH-1 by ≥ 0.5 dB
  AND Vol Raw PSNR not worse by > 0.2 dB AND no NaN/Inf.
- **PIVOT to diag** if EXP-CH-4 regresses on Vol PSNR > 0.5 dB: inspect
  active-surface fraction (overfit?) and |δ| p95 (explosion?) before any
  further config sweep.
- **DROP warm-start** if EXP-CH-3 ≈ EXP-CH-4 within 0.2 dB Sobel: warm-start
  isn't paying for its complexity; queue EXP-AB-3 only at low priority.
- **DROP thin surface** if EXP-CH-4 Sobel ≤ EXP-CH-1: hypothesis falsified on
  chest; stop P2 expansion, write up negative result, try a thin-structure
  phantom (marschner_lobb) before declaring dead.

---

## Conditional batch scheduler (kw995-safe, no compromised controls)

Use this section operationally: submit only the current eligible batch, record
its manifest and summary, then unlock the named successor. This prevents a
failed thin implementation from consuming the queue while keeping spare
capacity productive with valid scalar controls.

### Batch B0 — formal gates + scalar-control lane

**Priority P0. Active now at `surface@503a380`; scalar 1a/1b and thin stress 1a have completed.**

| Lane | Jobs allowed in parallel | Dependencies / stop rule |
|---|---|---|
| Gate lane | P0-E end-to-end split-voxel sample/calibration and P0-F diagnostics/inertness check (P0-D formally passed) | Stop all delayed-thin jobs if P0-E fails scalar calibration or P0-F inertness fails. File a targeted bug, rerun only that gate. |
| Scalar lane | EXP-CUBE-1 **scalar only**, in cube order 1a → 1b → 2a → 2b | May overlap gate lane. Stop the remaining cube variants if a manual-point control (1a/2a) misses its criterion; this is a baseline/harness issue. |
| Chest control lane | EXP-CH-1 scalar baseline only, after scalar cube 1a is green | It may overlap cube 1b/2a/2b and P0 rerun. Do **not** submit CH-2 yet: preserve the declared cube→chest sequencing and wait for thin cube comparison before paying for boundary-alignment control. |

**B0 artifacts and summary:**
- P0 console logs under `output/splitcell_validation/p0/<commit>/`; write a
  one-line PASS/FAIL table with test command, GPU, and commit.
- Cube: `output/cube_sanity/<test>/config.yaml,metrics.txt`.
- Chest scalar: `output/chest_CH1/{config.yaml,metrics.txt,model.pt,tb}`.
- Create/append `output/splitcell_validation/manifest.csv` with columns:
  `batch,experiment,commit,host,gpu,seed,config_source,overrides,command,
  status,output_dir,start_utc,end_utc`.

### Batch B1 — controlled delayed-start thin cube gate

**Priority P0/P1. The start=0 thin 1a run is a valid negative *stress signal*
(10.38 vs scalar 58.86 test PSNR; a 48.48 dB projection regression) because it
completed, activated, and hence exercises real optimization/rendering. It
falsifies safe early activation and elevates activation timing to high
importance. It does *not* reject the intended representation/configuration:
start=0 changes the independent variable versus the staged design
(`thin_surface_start=6000` after densification), and the manual cube provides
no thin internal interface for the added DOF to model. Dice .9981 despite the
projection failure also proves Dice alone is not an adequate gate.

**Completed / failed:** EXP-CUBE-2b thin random/densified 1b, K=4,
`thin_surface_start=6000`, used the paired scalar 1b seed/schedule/budget.
It was stable before activation (test PSNR 30.46 at step 4999) and catastrophic
after activation (final −17.78). This rules out only a generic pre-activation
training failure; it does not identify which thin activation component caused
collapse. Do not spend compute on thin 2a/2b or any chest job.

**Reference scalar 1b (fixed):** test PSNR **40.94 dB**, VolRaw PSNR
**36.07 dB**, Dice **0.9919**. Record its scalar split-voxel evaluation too.

**Mandatory artifacts/diagnostics before classification:**
1. Checkpoint + `config.yaml`, `metrics.txt`, TB event, and final model;
   checkpoint reload reproduces a fixed evaluation batch.
2. `vol_split_*` outputs for **both** scalar 1b and delayed thin 1b. Scalar
   split evaluator must equal scalar raw volume values (`allclose` rtol=1e-5,
   atol=1e-6); derived scalar raw-vs-split metric deltas must be ≤0.01 dB for
   PSNR/Sobel PSNR and ≤1e-4 for SSIM. Otherwise P0-E is not calibrated and
   B1 is invalid, not a thin failure.
3. Thin diagnostics at activation and final: activation iteration=6000,
   active fraction, |delta| p50/p95/max, height-norm p50/p95/max,
   quaternion norm p50/p95, normal coherence, plus NaN/Inf scan. Also archive
   the final 100 steps of projection loss/PSNR to catch a post-activation drop.

**Exact B1 pass / go-to-chest criteria (all required):**
- Process exits 0; no NaN/Inf; P0-E scalar calibration passes; split metrics
  and all listed diagnostics are present.
- Test PSNR ≥ **40.44 dB** (no more than 0.50 dB below scalar 1b).
- Thin **split-aware** volume PSNR ≥ **35.57 dB** (no more than 0.50 dB below
  calibrated scalar split PSNR=36.07); split Dice ≥ **0.9869** (no more than
  0.005 below scalar); split Sobel PSNR ≥ scalar split Sobel PSNR − **0.50 dB**.
- Raw-volume safety check: VolRaw PSNR ≥ **35.07 dB** (no more than 1.00 dB
  below scalar) and Raw Dice ≥ **0.9819**. Raw is secondary because it does
  not represent the internal split, but a larger collapse is diagnostic.
- Activation is delayed: no thin parameter group before step 6000; diagnostics
  show finite, nonzero post-activation learning (active fraction >0 OR
  |delta| p95>1e-6), without clamp/gradient saturation evidence.

**Classification (actual): NO-GO / activation regression.** Test PSNR,
raw/split volume PSNR, and Dice all miss the gate catastrophically. Enter B1-R
below. No chest, phantom quality comparison, K=8, or broad hyperparameter
sweep is permitted until a rescue run re-passes every B1 criterion.

Append `output/splitcell_validation/cube_comparison.csv` with scalar/thin
projection PSNR, raw and split PSNR/SSIM/Sobel/Dice, runtime, NaN flag, and
all diagnostics above. The emitted config is the immutable source of truth.

### Batch B1-R — activation-regression rescue ladder (cube only)

**Priority P0/P1; sequential, not parallel.** Every rescue starts from the
failed delayed thin1b recipe and changes **exactly one experimental factor**.
Keep fixed: seed, cube 1b data/init, iterations, rays, `K=4`,
`thin_surface_start=6000`, scalar-density LR/scheduler, densification recipe,
and all non-thin loss weights. The default failed run is the reference, not a
new control. Do not combine rescues or tune based on a partial run.

#### R0 — required instrumentation / activation snapshot (no quality claim)

Before R1, add a state dump at t=5999 (pre-init), t=6000 after thin init but
before optimizer step, t=6001, t=6010, t=6100, and final. It is a prerequisite,
not a rescue treatment.

- Required fields per snapshot: fixed-ray loss and PSNR; base μ; δ; μ+/μ−
  min/p01/p50/p99/max; fraction hitting μ clamp; each thin parameter group's
  LR, grad norm, parameter norm, optimizer step and `exp_avg`/`exp_avg_sq`
  norm; heights; texel-site radius; quaternion norm; active fraction; cell
  count; number of cells created/pruned at 6000; parent/inheritance metadata.
- Artifact: `output/splitcell_validation/activation_trace_<commit>_1b.csv`
  plus serialized snapshot tensors/metadata. Missing any field invalidates the
  following diagnosis run; it is not evidence for a mechanism.

#### R1 — activation-continuity control

**Hypothesis isolated:** the forward/optimizer transition itself is
non-continuous, independently of learning dynamics.

- Single change vs failed thin1b: initialize/enable thin mode at 6000 but set
  all four new thin parameter-group LRs to **0** for the remainder. δ=h=0 must
  therefore remain zero; scalar parameters continue normally.
- Config source: failed thin1b emitted config + one `thin_surface_lr_scale=0`
  override/diagnostic patch; command follows `old/test_cube.py --thin --test
  1b --thin-start 6000` and archives config.
- Success: fixed-ray PSNR changes ≤0.10 dB from t=5999 to t=6001; loss ratio
  ≤1.02; final test PSNR ≥40.44; final VolRaw/split PSNR ≥35.57; scalar and
  split volumes remain calibrated; δ/h exactly zero within 1e-7.
- Fail: any discontinuity, nonzero δ/h, or final collapse identifies activation
  mode/kernel/optimizer-group insertion as the first bug. **Stop**; no other
  rescue is interpretable until fixed and R1 passes.

#### R2 — delta-only rescue (RAW PASS; split gate pending)

**Completed (user-reported newest result; execution log pending update):** K4
start=6000, global thin scale=1, delta scale=.01 (effective LR **5e-5**),
geometry scale=0. Test PSNR **43.74**, VolRaw PSNR **41.25**, Dice **.9976**,
Sobel PSNR **37.75**. R0 normal-thin LR collapsed; R1 all-thin-frozen was
stable. This localizes the catastrophic behavior to nonzero thin optimization
and demonstrates a safe *raw-metric* delta-only rescue. It does not yet prove
that the learned μ+/μ− split field is valid.

##### R2-EVAL — mandatory split-aware evaluation (no training run)

Run this before R3. It is required because VolRaw can evaluate only the base
field and can conceal unstable/unphysical side densities. Geometry learning
must not be enabled on a delta state whose split-aware field is already bad.

- Inputs/artifacts: `output/cube_sanity/single_cube_random_thin_R2/model.pt`,
  its emitted `config.yaml`, the same cube-1b GT/resolution/supersampling as
  scalar 1b, and scalar-1b split evaluation. Command template:
  `micromamba run -n radfoam python split_voxelize.py --model <R2/model.pt> --gt <same-cube-GT> --resolution 256 --supersample 4`.
  Use default hard-side evaluation; **no** `--blend_eps`.
- Required output: `volume_split.npy`, slices, and `r2_split_metrics.json/csv`
  appended to `output/splitcell_validation/rescue_summary.csv`; also record
  inside/outside means, μ+/μ− p01/p50/p99/max, and clamp fraction.
- Pass to R3 only if: split PSNR ≥**35.57**, split Dice ≥**.9869**, split Sobel
  PSNR ≥ scalar split Sobel −**.50 dB**, all metrics finite, clamp fraction
  <1%, and μ+/μ− p99 ≤2× preactivation base-μ p99. Scalar raw-vs-split
  calibration remains required (PSNR/Sobel delta ≤.01 dB; SSIM delta ≤1e-4).
- If R2-EVAL fails: **do not run R3**. Delta-only raw recovery is an evaluator/
  side-density diagnostic, not a representation rescue; audit δ distribution
  and clamp behavior first.

#### R3 — first safe geometry-learning threshold

**Eligible only after R2-EVAL passes. Hypothesis isolated:** whether any
controlled geometry learning, added to the proven delta-only state, destabilizes
ray-side assignment. This changes exactly one factor from R2: geometry group
scale **0 → 1e-3**.

- **Tag:** `R3_geom1e-3`.
- **Exact per-group scales:** global thin=`1.0`; density_delta=`0.01`
  (unchanged from R2; effective LR **5e-5**); quaternion=`0.001`, texel_sites
  =`0.001`, texel_heights=`0.001` (each effective LR **5e-6**, assuming the
  cube thin base LR 5e-3). Same scheduler shape and all other R2 settings.
- Command:
  `micromamba run -n radfoam python old/test_cube.py --test 1b --thin-surface --thin-start 6000 --run-tag R3_geom1e-3 --thin-lr-scale 1 --thin-delta-lr-scale 0.01 --thin-geometry-lr-scale 0.001`
- Artifact: `output/cube_sanity/single_cube_random_thin_R3_geom1e-3/` plus
  activation trace and split evaluation using the identical R2-EVAL protocol.
- **Early safety pass:** t5999→t6001 PSNR drop ≤.25 dB; t5999→t6100 drop
  ≤1.0 dB; no NaN/Inf; μ+/μ− p99 ≤2× t5999 base-μ p99; clamp fraction <1%;
  all three geometry groups have measurable movement (>1e-7 parameter change)
  but quaternion norm p99 remains within 1% of 1 and height p99 remains below
  one cell radius.
- **Full threshold pass:** all original B1 quality gates plus split-aware
  metrics: test ≥40.44, VolRaw ≥35.07, Raw Dice ≥.9819, split ≥35.57, split
  Dice ≥.9869, split Sobel deficit ≤.50 dB.
- **Decision:** full pass establishes 1e-3 as the first safe geometry scale;
  only then consider a single next decade step (1e-2). Early/full failure with
  R2-EVAL pass establishes geometry learning as unstable at 5e-6 effective LR:
  revert to geometry=0 and inspect geometry gradients/parameterization; do not
  advance to R4/chest.

#### R4 — weak-regularization control

**Eligible only if R1 passes and R2/R3 do not fully rescue. Hypothesis
isolated:** default penalties permit unsupported split amplitudes.

- Single change: apply a single global thin-regularization multiplier **10×**
  to both existing delta-L2 and height-L1 terms, leaving all LRs/schedulers and
  geometry trainable at failed-default values. Implement as one explicit
  `thin_surface_reg_scale=10` control; do not independently tune two weights.
- Artifact: `.../cube_1b_R4_reg10/` + trace.
- Early stability success: by step 6100, |δ| p95 and height-norm p95 are ≤1/3
  of failed thin1b's corresponding trace values, with no >1 dB PSNR decline.
  Full rescue: all B1 thresholds.
- Interpretation: recovery only under R4 implicates weak regularization;
  failure with bounded parameters indicates regularization magnitude is not
  the primary mechanism. Proceed R5.

#### R5 — densification-boundary / inheritance-order control

**Eligible only if R1 passes and R2–R4 do not fully rescue.** There should be
no *post*-activation densification (`densify_until=6000`), so inheritance is
not a plausible ongoing cause. The remaining test is whether activation shares
an iteration with the final densify/prune lifecycle.

- Single change: `thin_surface_start=6001`; all other failed thin1b settings
  unchanged. This separates thin initialization from the `densify_until=6000`
  boundary by one optimizer iteration without changing the final point budget.
- Artifact: `.../cube_1b_R5_start6001/` + trace including cell count and
  parent/new-cell fields at 5999–6001.
- Success: no post-start collapse and all B1 thresholds. A recovery here
  implicates lifecycle ordering/parameter inheritance at the final densify
  boundary, not generic late activation.
- Fail: if R5 also collapses with zero new/pruned cells at activation, deprioritize
  inheritance and escalate to a code-level CUDA/optimizer audit; stop quality
  experiments.

**R-ladder global stop/advance rules:**
- A run may identify a mechanism by **early stability recovery**, but may
  unlock chest only after **full B1 success** and a repeat of that winning
  rescue once (same seed/config, fresh process) to exclude transient failure.
- If R1 passes but R2–R5 all fail, stop tuning. The remaining hypothesis is a
  deeper backward/optimizer implementation fault despite finite-difference
  checks; require an Implementer audit using R0 traces and a minimal
  post-activation replay test.
- On any NaN/Inf, missing trace, uncalibrated scalar split evaluator, or changed
  non-target setting, mark the run invalid and rerun only the same treatment.
- Keep `output/splitcell_validation/rescue_summary.csv` with one row/run:
  commit, seed, exact single changed factor, t5999/t6001/t6100 PSNR/loss,
  final raw+split metrics, μ+/μ− stats, δ/height/geometry norms, cell counts,
  outcome {pass,early-recovery,fail,invalid}.

### Batch B2 — Milestone 4 chest delta-only four-way matrix

**Eligible now (P1):** Cube R2 passed split-aware evaluation (PSNR 43.46 vs
scalar 37.43, Dice .9960; inside/outside .9954/.0001). Geometry learning
failed at 1e-3 and 1e-4, so every chest thin arm is **delta-only**: delta scale
.01; quaternion/site/height scales 0. Do not launch any learned-geometry arm.

**Fixed controls for all four arms:** pull one identical `surface` commit;
record `git rev-parse HEAD`; `train.py:46-48` fixes torch/numpy seed=42 (no seed
CLI exists); retain each base config's identical chest budget—10,000 iters,
1M early + 4M late rays/batch, 64k→512k cells, densify 1000–6000, freeze 9500,
L1 loss, and dataset `r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone`.
Do not alter regularizers, sampling, point/density LRs, K, or evaluation
resolution. `--experiment_name` causes train.py to archive the resolved config.

**Boundary-alignment control convention:** BA is Stage-0-only in both BA arms:
`top_eig_align_weight=1e-2`, `top_eig_align_start=1000`,
`top_eig_align_until=6000`. The strict `< until` condition means it populates
`_last_top_eigvec` before thin activation but does not continue after it. This
is essential: with frozen heights, the height-gate cannot disable ongoing BA,
so leaving the config default `until=9500` would confound warm-start with
continued BA regularization.

| ID / exact output tag | Base config | Required CLI overrides |
|---|---|---|
| **EXP-CH-1** `splitcell_m4/chest_s42_scalar` | `configs/best428_nointerp.yaml` | `--top_eig_align_weight 0 --top_eig_align_start 1000 --top_eig_align_until 6000` |
| **EXP-CH-2** `splitcell_m4/chest_s42_scalar_ba` | `configs/best428_nointerp.yaml` | `--top_eig_align_weight 1e-2 --top_eig_align_start 1000 --top_eig_align_until 6000` |
| **EXP-CH-3** `splitcell_m4/chest_s42_thin_delta_nowarm` | `configs/best428_thinsurface.yaml` | `--thin_surface_start 6000 --thin_surface_K 4 --thin_surface_lr_scale 1 --thin_surface_delta_lr_scale 0.01 --thin_surface_quat_lr_scale 0 --thin_surface_sites_lr_scale 0 --thin_surface_heights_lr_scale 0 --top_eig_align_weight 0 --top_eig_align_start 1000 --top_eig_align_until 6000` |
| **EXP-CH-4** `splitcell_m4/chest_s42_thin_delta_ba` | `configs/best428_thinsurface.yaml` | `--thin_surface_start 6000 --thin_surface_K 4 --thin_surface_lr_scale 1 --thin_surface_delta_lr_scale 0.01 --thin_surface_quat_lr_scale 0 --thin_surface_sites_lr_scale 0 --thin_surface_heights_lr_scale 0 --top_eig_align_weight 1e-2 --top_eig_align_start 1000 --top_eig_align_until 6000` |

**Exact training commands** (all launched from repo root; do not launch yet):
```bash
micromamba run -n radfoam python train.py -c configs/best428_nointerp.yaml \
  --experiment_name splitcell_m4/chest_s42_scalar \
  --top_eig_align_weight 0 --top_eig_align_start 1000 --top_eig_align_until 6000

micromamba run -n radfoam python train.py -c configs/best428_nointerp.yaml \
  --experiment_name splitcell_m4/chest_s42_scalar_ba \
  --top_eig_align_weight 1e-2 --top_eig_align_start 1000 --top_eig_align_until 6000

micromamba run -n radfoam python train.py -c configs/best428_thinsurface.yaml \
  --experiment_name splitcell_m4/chest_s42_thin_delta_nowarm \
  --thin_surface_start 6000 --thin_surface_K 4 --thin_surface_lr_scale 1 \
  --thin_surface_delta_lr_scale 0.01 --thin_surface_quat_lr_scale 0 \
  --thin_surface_sites_lr_scale 0 --thin_surface_heights_lr_scale 0 \
  --top_eig_align_weight 0 --top_eig_align_start 1000 --top_eig_align_until 6000

micromamba run -n radfoam python train.py -c configs/best428_thinsurface.yaml \
  --experiment_name splitcell_m4/chest_s42_thin_delta_ba \
  --thin_surface_start 6000 --thin_surface_K 4 --thin_surface_lr_scale 1 \
  --thin_surface_delta_lr_scale 0.01 --thin_surface_quat_lr_scale 0 \
  --thin_surface_sites_lr_scale 0 --thin_surface_heights_lr_scale 0 \
  --top_eig_align_weight 1e-2 --top_eig_align_start 1000 --top_eig_align_until 6000
```

**Scientific interpretation:** CH1→CH2 estimates Stage-0 BA alone; CH3→CH4
estimates BA-derived warm-start plus its matched Stage-0 pretraining in the
frozen-geometry delta-only representation; CH1→CH3 tests delta-only split
without BA; CH2→CH4 tests adding delta-only split atop the same BA schedule.
No pair claims to isolate quaternion orientation independently of BA pretraining.

**Artifacts and post-training evaluation:** each arm must produce
`output/splitcell_m4/chest_s42_<tag>/{config.yaml,model.pt,metrics.txt,tb}`.
After all four finish, run the same hard-side evaluator (no `--blend_eps`) on
all four—scalar through the same code path—to avoid evaluator confounding:
```bash
for tag in chest_s42_scalar chest_s42_scalar_ba chest_s42_thin_delta_nowarm chest_s42_thin_delta_ba; do
  micromamba run -n radfoam python split_voxelize.py \
    --model output/splitcell_m4/$tag/model.pt \
    --gt r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone/vol_gt.npy \
    --resolution 256 --supersample 4 \
    --output output/splitcell_m4/$tag/split_eval
done
```
Collect diagnostics from the thin TB runs at 5999/6000/6001/6100/final:
implemented `thin/delta_abs_mean`, `thin/delta_abs_p95`,
`thin/delta_abs_max`, `thin/mu_plus_max`, `thin/mu_minus_max`,
`thin/height_l1_mean`, `thin/height_l1_max`, `thin/quat_norm_mean`,
`thin/quat_norm_max`, `thin/active_frac`, `thin/delta_nonzero_frac`, and
`thin/warm_start`; additionally archive per-group LR/grad norms and μ+/μ−
p01/p50/p99/max/clamp fraction in the activation trace. Geometry parameter-
change norms must remain zero (≤1e-7); CH4 must show `thin/warm_start=1` and
CH3 `thin/warm_start=0`. Also retain BA loss/cache evidence for CH2/CH4.

**B2 collection contract:** after all four complete, write one immutable
`output/splitcell_validation/chest_matrix_<commit>_seed<seed>.csv` with the
manifest fields plus: `test_psnr,test_ssim,test_rmse,vol_raw_psnr,
vol_raw_ssim,vol_raw_sobel_psnr,vol_raw_f1_1v,vol_split_psnr,
vol_split_ssim,vol_split_sobel_psnr,active_frac,delta_p50,delta_p95,
height_p95,normal_coherence,train_seconds,nan_inf_flag`. Missing split
columns on scalar runs must be blank/NA, never substituted with raw values.

**B2 stop rules:**
- Any NaN/Inf, missing checkpoint, or missing required thin diagnostics:
  invalidate only that thin cell, stop expansion, diagnose/re-run it.
- CH-3/CH-4 both worse than CH-1 by >0.5 dB raw PSNR: regression branch.
- CH-4 beat/win classification below uses both raw and split-aware metrics;
  split-aware improvement alone cannot override raw-PSNR regression.

### Batch B3-WIN — automatic follow-up after a positive chest result

**Priority P2. Unlock if CH-4 vs CH-1 has raw Sobel gain ≥0.5 dB, raw PSNR
loss ≤0.2 dB, no failures, and split-aware metrics corroborate rather than
contradict the direction.**

Run in this order, holding one GPU free for analysis/recovery where possible:
1. **AB-1 start sweep** {4000, 6000, 7000}, chest, 1 seed (3 jobs).
2. **AB-2 reg grid** 2×2 delta×height, chest, 1 seed (4 jobs).
3. Select the best two by raw Sobel subject to raw PSNR guard; launch
   **EXP-SEED** (two additional seeds each only if capacity permits; otherwise
   one winning config first).
4. Only after seed trend confirms: **AB-3** warm-start on/off on bonsai, then
   **AB-4** full 10-scene n75 breadth.

Stop the start/grid sweep early only for NaN/Inf or raw PSNR regression >1 dB;
do not rank partial runs. Do not launch AB-5/6/7 until a seeded winner exists.

### Batch B3-NEUTRAL — automatic follow-up after an inconclusive chest result

**Priority P2. Unlock if CH-4 raw Sobel change is within ±0.5 dB and raw PSNR
is within ±0.2 dB of CH-1.** The objective is diagnose signal, not broad sweep.

1. Run AB-3 warm-start on/off on `marschner_lobb_n75_clean` (thin-structure
   stress case), same one seed, plus scalar baseline (3 jobs).
2. Run a restricted regularization probe on chest: default, delta=1e-4,
   height=5e-5 (3 jobs; no full 2×2 yet).
3. If a thin-structure phantom improves Sobel ≥0.5 dB with PSNR guard,
   promote to B3-WIN starting with AB-1. Otherwise write a negative/inconclusive
   summary and stop broad scene expansion.

### Batch B3-REGRESSION — automatic follow-up after a negative chest result

**Priority P1 diagnostic only. Unlock if CH-4 fails the raw PSNR guard (>0.2 dB
loss) or Sobel is ≤ CH-1 −0.5 dB. Do not launch hyperparameter grids or suite.**

1. Inspect B2 diagnostics and checkpoint reload equality: active fraction,
   delta p95, height p95, normal coherence, thin-vs-scalar zero-init output.
2. Run exactly one controlled rerun selected by symptom:
   - excessive active fraction / heights → default with height weight ×10;
   - delta p95 explosion / clamp saturation → delta weight ×10;
   - inactive surfaces / near-zero delta → delta and height weight ÷10;
   - warm-start anomaly → CH-3 vs CH-4 short diagnostic on chest.
3. If controlled rerun still misses raw PSNR guard, close chest as a negative
   result and run only the marschner_lobb scalar/thin falsification comparison.
   No K=8, suite, Gaussian, gradient, scaling, or seed expansion.

### Batch B4 — breadth and mechanism mapping

**Priority P2/P3. Eligible only after B3-WIN and a seeded chest winner.**
AB-4 suite breadth → AB-5 Gaussian/gradient/thin → AB-6 co-training → AB-7
scaling. K=8 stays deferred until a separate K>4 plumbing + gradcheck gate.

## LC64-Air — staged representation/schedule diagnosis (planning only)

> **Approved v1 execution authority:** `specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v1.md`,
> `experiments/LC64-AIR-QUEUE-v1.md`,
> `experiments/LC64-AIR-SPLIT-DIAGNOSIS-MANIFEST-v1.yaml`, and
> `experiments/LC64-AIR-EVIDENCE-TEMPLATE-v1.csv` supersede any conflicting
> LC64 prose below. Reconciled binding details: A uses F1500; D is the matched
> {F0,F500,F1500} scalar/split ladder; E is q+height-only at absolute 2e-4/
> 5e-4; and a new `points_freeze_at` gate is required because existing
> `freeze_points` does not freeze coordinates.

**Motivation / scope.** Milestone 6 hard-side LC64 geometry learning improved
surface extraction but regressed prioritized volume/edge metrics (split−scalar:
PSNR −1.81 dB, Sobel −1.46 dB, 2D/3D SSIM ≈−.08). This plan diagnoses the
reported **air-region artifacts** without parameter-count breadth, densification,
or soft-side evaluation. It supersedes any LC64 32k promotion. **Do not execute
until the independent-mode engineering gate is green.**

### LC64 fixed protocol (every arm)

Create immutable per-arm YAMLs from a single `configs/lc64_air_base.yaml`; do
not rely on unrecorded CLI mutations. Fixed values:

```yaml
dataset: r2_gaussian
data_path: r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone
iterations: 10000
init_points: 64000
final_points: 64000
densify_from: 0
densify_until: 0
densify_factor: 1.0
redundancy_cap: 0.0
redundancy_cap_init: 0.0
redundancy_cap_final: 0.0
prune_variance_criterion: false
thin_surface_start: 0
thin_surface_K: 4
top_eig_align_weight: 0.0
# Geometry is frozen in stages A-C; no BA or geometry-learning confound.
thin_surface_quat_lr_scale: 0.0
thin_surface_sites_lr_scale: 0.0
thin_surface_heights_lr_scale: 0.0
```

Keep all other model/optimizer/ray settings identical to the frozen base YAML.
`train.py` currently fixes torch/numpy seed=42; Stage A-C use seed 42. Record
commit, resolved YAML hash, CUDA/PyTorch versions, worker/GPU, and the fixed
seed in `manifest.csv`. Run all arms through `split_voxelize.py --blend_eps 0
--resolution 256 --supersample 4`; scalar must use that same evaluator fallback.
No interpolation, meshing, IDW, or evaluator smoothing is allowed for ranking.

### Engineering gate E0 — fair independent-density implementation

`thin_surface_relative_delta: true` already implements bounded relative
mean+delta (`rho=thin_surface_delta_max_frac`). A true independent two-density
arm must be added before Stage A; raw additive delta is **not** a substitute.

- Required independent representation: two nonnegative density branches
  `mu_plus=softplus(raw_plus)`, `mu_minus=softplus(raw_minus)`, initialized
  with `raw_plus=raw_minus=base_raw` so the zero-split field equals scalar.
- Fair optimizer requirement: parameter count matches relative mode (2N
  density DOF); its mean coordinate follows the existing base-density LR
  schedule and its difference coordinate follows the *same swept split LR*
  as relative delta. This needs transformed/pair optimizer handling or an
  equivalent verified update; assigning one LR independently to both branches
  would confound representation with mean-learning rate.
- Required tests/artifacts: zero-init forward equality vs scalar (fixed rays
  and hard-side voxels, atol 1e-6); finite-difference gradients for both raw
  branches; checkpoint/reload; hard-side query; per-coordinate LR/grad logs.
- **Reject/hold:** any mismatch, missing optimizer-equivalence test, or
  checkpoint/query failure blocks all independent arms; relative-only controls
  may still run but cannot support a representation conclusion.

### Pre-registered outcome metrics and prune rules

At steps 500, 1500, 2500, 6000, and final, evaluate a fixed held-out ray set
and write a hard-side 256^3 volume. Define once from GT and reuse for every
arm: `air = gt <= 0.01 * p99(gt)`; `object = not air`; false-air-positive
threshold `0.05 * p99(gt)`. Report full/object/air PSNR and MAE, air p95/max
predicted density, air false-positive fraction, and a 1-voxel object-boundary
halo MAE. Also report full hard-side PSNR/SSIM/Sobel/Dice/CD/HD95/F1.

- **Safety kill at a scheduled checkpoint:** nonfinite value; air false-positive
  fraction >2× matched scalar; air MAE >1.5× matched scalar; or held-out
  projection PSNR >2 dB below matched scalar. Stop the arm, preserve its
  checkpoint/trace, label `pruned_safety`—never silently discard it.
- **Advance / representation win:** full hard-side PSNR and Sobel are each no
  worse than matched scalar by 0.30 dB, while air false-positive fraction or
  air MAE improves by ≥15%. Surface-only gains do not advance an arm.
- **Neutral:** all volume guards pass but air improvement <15%; retain as
  diagnostic, do not seed-replicate or expand its LR range.
- **Reject:** final failure of any volume guard, or any safety kill.

Required thin logs: delta mode/rho, μ+/μ− p01/p50/p99/max, clamp fraction,
delta/effective-delta p50/p95, per mean/difference and geometry group LR/grad
norms, active fraction, point displacement, and air-mask summary. Geometry
scales must log zero through Stage C.

### Stage A — scalar and zero-split validity controls (3 runs, seed 42)

All use normal point schedule `freeze_points=9500`, no densification, and
geometry frozen. These arms establish whether merely enabling each renderer at
iteration 0 creates air artifacts.

| ID | Only varying factor | Mode / split LR | Acceptance |
|---|---|---|---|
| A0 `LC64A_scalar_F9500` | no split | scalar | reference; completes finite |
| A1 `LC64A_relative0_F9500` | representation | relative, rho=.5, diff LR=0 | hard-side volumes and air metrics within 5% of A0; otherwise renderer/zero-init bug |
| A2 `LC64A_independent0_F9500` | representation | independent, diff LR=0 | same A0-equivalence guard; otherwise E0/independent bug |

**Gate A:** A1/A2 both pass equivalence before learned split LR is swept. If
only one passes, diagnose/fix that representation; do not compare it against
the other.

### Stage B — representation × split-LR screen (6 runs, seed 42)

Fixed: points `freeze_points=9500`, split starts iteration 0, geometry frozen,
rho=.5 for relative. The *only* swept factor inside each representation is
difference/split LR. Base thin LR is .005; scales below therefore map to an
absolute split LR of .00005, .00020, .00050. Point initial LR is .00020.

| IDs | Representation | `split_lr_scale` | Absolute split LR | Relation to point LR |
|---|---|---:|---:|---|
| B-R-01 / B-I-01 | relative / independent | .01 | 5e-5 | 0.25× |
| B-R-04 / B-I-04 | relative / independent | .04 | 2e-4 | 1× |
| B-R-10 / B-I-10 | relative / independent | .10 | 5e-4 | 2.5× |

For relative, map `split_lr_scale` to `thin_surface_delta_lr_scale`; for
independent, map it to its verified difference-coordinate scheduler. Generate
six explicit YAMLs; no arm changes parameterization, point schedule, rho,
geometry, densification, seed, or budget in addition to the listed factor.

**Stage-B pruning:** apply safety kill at scheduled checkpoints. Rank only
non-pruned arms by air false-positive reduction subject to volume guards. If
none is a representation win, reject the split-at-0 density hypothesis for
LC64 and stop—no early-freeze/geometry or LR=.40 expansion. If ≥1 wins, retain
at most one winner per representation for Stage C.

### Stage C — high-LR confirmation (0–2 runs, seed 42)

Only a Stage-B winner gets the pre-registered high setting:
`split_lr_scale=.40` → absolute split LR .002 (**10× point LR**). All else is
identical to its winning Stage-B arm. Tag `LC64C_<mode>_LR40_F9500`.

- Advance only if it satisfies the same representation-win guard and improves
  air metric by ≥5 percentage points over that representation's .10 arm.
- Otherwise reject LR=.40 as runaway/overfit and choose the lower-LR winner.

### Stage D — early point-freeze handoff (2–4 runs, seed 42)

Only the single best Stage-B/C representation+LR enters. This tests whether
point motion is causing air artifacts versus the split density carrying the
sub-cell correction; it does **not** change representation, split LR, rho,
geometry, no-densification protocol, or seed.

| Pair | Point schedule factor | Matched arms |
|---|---|---|
| D1 | normal `freeze_points=9500` vs early `freeze_points=1500` | scalar control + winning split |
| D2 (conditional) | early 1500 vs fixed `freeze_points=0` | scalar control + winning split |

D2 is eligible only if D1's winning split meets volume guards and improves its
air metric ≥15% over its **matched scalar F1500**. This prevents mistaking a
scalar early-freeze failure for a split benefit. Any D arm must pass the same
safety/representation gates against the scalar with the same freeze schedule.

### Stage E — optional sub-cell geometry handoff (2 runs maximum; separate)

Geometry is excluded from Stages A-D due LC64 R3/R3b failures. Only if Stage D
wins and a new geometry-stability unit test passes may it be tested. Compare
winner with geometry frozen vs geometry scale `1e-5` (quat/sites/heights all
same; pre-registered, substantially below prior failed 1e-4), with the same
point-freeze schedule. A geometry arm is rejected on any safety kill or if
hard-side PSNR/Sobel falls >0.30 dB; it must additionally improve the air
metric ≥15% over its geometry-frozen parent. No geometry LR sweep follows a
failure.

### Seeds, budget, and stopping

- Stages A-D: one deterministic seed (42) for screening. Stage A=3, B=6,
  C≤2, D=2–4, E≤2: **13–17** training runs maximum before replication.
- Replicate only the final winner and its matched scalar with seeds {43,44}
  (three seeds total). Require mean air improvement ≥15%, mean volume/Sobel
  guard, and the 95% CI of split−scalar air MAE below 0 before any new dataset
  or cell-count claim.
- Provisional allocation: cap each 64k/no-densify 10k-iteration job at 0.75
  GPU-hours; screening cap is 9.75–12.75 GPU-hours plus ≤3 GPU-hours for
  replication. First A0 wall time replaces this estimate for scheduling.
- Run Stage A, then B, then C/D/E strictly sequentially by gate; arms within
  an already-unlocked Stage B may use separate GPUs because their controls are
  immutable. Never backfill an arm after seeing results.

### Web-results artifact contract (required for every completed/pruned arm)

Publish a static, self-describing bundle under
`web-results/LC64-air-v1/<run_tag>/` and a top-level
`web-results/LC64-air-v1/{manifest.json,summary.csv,index.html}`. Each arm
bundle must contain or link by content hash to: resolved config, git commit,
seed/GPU/run ID, `metrics.txt`, hard-side split metric JSON, air-ROI JSON,
checkpoint path/hash, `volume_split_hard_ss4.npy`, side-map for split modes,
and fixed-window axial/coronal/sagittal slice PNGs plus an air-error/halo PNG.

`index.html` must present paired scalar-vs-split rows for the matched freeze
schedule and fixed color scales; it must label `pruned_safety` artifacts rather
than hide them. `manifest.json` must expose exact factor values, checkpoint
step, evaluator command (`--blend_eps 0 --resolution 256 --supersample 4`),
metric schema, and SHA256 hashes. Serve that directory through the worker web
tunnel only after files are complete; a missing image, side-map, config hash,
or evaluator command makes the arm non-reportable.

## P2 — Rolling ablation queue (keep compute busy after P1)

Fill GPUs only through the eligible conditional batch above. Each entry: 1
seed first, escalate to 3 seeds only for top-2 configs. Dispatch order is the
priority order.

- [ ] **EXP-AB-1 · `thin_surface_start` sweep** — {4000, 6000(default), 7000}
      on chest. Tests earlier (more fine-tune) vs later (post-freeze) activation.
  - Config: `configs/best428_thinsurface.yaml` + `--thin_surface_start <v>`.
  - Deps: P1-CHEST decision = CONTINUE. Artifacts: 3 runs.

- [ ] **EXP-AB-2 · Reg-weight 2×2 grid** — `thin_surface_delta_weight` ∈
      {1e-3, 1e-4} × `thin_surface_height_weight` ∈ {5e-4, 5e-5} on chest.
  - Config: `best428_thinsurface.yaml` + two CLI overrides. 4 runs.
  - Deps: P1-CHEST. Selects δ/height growth balance.

- [ ] **EXP-AB-3 · Warm-start on/off (isolated)** — `top_eig_align_weight` ∈
      {0, 1e-2} with thin surface ON, on chest + bonsai. Confirms/denies
      EXP-CH-3 vs CH-4 finding on a second scene.
  - Deps: P1-CHEST. 4 runs.

- [ ] **EXP-AB-4 · Scene breadth** — full synthetic n75 suite (10 scenes):
      EXP-CH-1 vs EXP-CH-4 only. Use `train_all.py` pattern.
  - Command: `micromamba run -n radfoam python train_all.py -c configs/best428_thinsurface.yaml --name suite_thin --worker W --of N`
    (and `-c configs/best428_nointerp.yaml --name suite_base`).
  - Deps: P1-CHEST CONTINUE. ~20 runs; parallelize across kw995/kw996.
  - Decision gate for "thin surface helps in general": mean Vol Raw Sobel
    PSNR > baseline, no scene regresses > 0.5 dB Vol PSNR.

- [ ] **EXP-AB-5 · Gaussian vs linear-gradient vs thin-surface** — 3-way on
      a vessel/thin-structure scene (bonsai or marschner_lobb_n75_clean).
      Maps where each mechanism belongs (fyi.md §9).
  - Configs: `best428_thinsurface.yaml`; `r2_gauss.yaml` (gaussian_start);
    `best428_nointerp.yaml` + `gradient_start 2000`.
  - Deps: P1-CHEST. 3 runs × 1 seed.

- [ ] **EXP-AB-6 · Co-train density_grad + thin surface** —
      `gradient_start=2000`, `thin_surface_start=6000`. Checks residual
      partitioning (smooth ramp vs sharp jump). fyi.md §8.
  - Deps: P1-CHEST. 2 scenes (chest, bonsai).

- [ ] **EXP-AB-7 · Scaling** — `final_points` ∈ {256k, 512k(default), 1M}
      with thin surface. Memory check (+14 floats/cell → ~28 MB @512k,
      ~56 MB @1M). fyi.md storage note.
  - Deps: P1-CHEST. 3 runs chest.

- [ ] **EXP-SEED · 3-seed replication of winning chest config** — only the
      single best config from the matrix. Required before any external claim.
  - Deps: P1-CHEST + at least one P2 sweep resolved. 2 extra runs.

- [ ] **EXP-AB-8 · K=8 (DEFERRED)** — only after K-plumbing + gradcheck
      extended for K≠4 (P0-B scope expansion). Vessel-heavy scene. fyi.md §6.
  - Deps: new Implementer task to extend K validation. Do not run before.

- [ ] **EXP-AB-9 · Post-hoc IDW σ-sweep on thin checkpoints** — does
      interpolation still help on top of split cells? Reuse
      `old/eval_sigma_sweep.py` pattern on `output/chest_CH4/model.pt`.
  - Deps: P1-CHEST done. No retraining.

---

## Success criteria (overall go/no-go)

1. **P0-D green** — passed at `surface@503a380`: all 5 param groups fp32
   finite-difference checks satisfy rel-err < 1e-3.
2. **P0-E/F chest prerequisite** — calibrated scalar raw==split voxel outputs,
   split metrics populated, and δ=0/h=0 inertness equals scalar baseline to
   fp32 noise; activity diagnostics must be present.
3. **EXP-CUBE-2b delayed thin 1b** — failed catastrophically (−17.78 test
   PSNR). The only path forward is B1-R: R0 trace; R1 continuity; then
   sequential one-factor R2 delta-LR, R3 geometry-freeze, R4 global-reg×10,
   R5 start=6001/lifecycle separation. A rescue must pass every B1 criterion
   **and** repeat in a fresh process before chest unlocks.
4. **EXP-CH-4** — Vol Raw Sobel PSNR > baseline by ≥ 0.5 dB, Vol Raw PSNR
   not worse by > 0.2 dB.
5. **EXP-AB-4** — mean Vol Raw Sobel PSNR > baseline across 10 scenes; no
   scene regresses > 0.5 dB Vol PSNR.
6. **Stability** — 0 NaN/Inf runs across all configs; |δ| bounded relative
   to `softplus(density)`.

## Likely failure modes (watch in TB / metrics.txt)

- Soft-Voronoi softmax adjoint → NaN on grazing rays (|n·d|<1e-3 fallback).
- δ explosion if `thin_surface_delta_weight` too low → μ₊/μ₋ clamp saturates.
- Height-field overfit → spurious surfaces in smooth regions (height_weight
  too low); watch active-surface fraction vs expected boundary fraction.
- Quaternion sign ambiguity → optimizer oscillation (track quat-norm drift).
- Warm-start silently no-op if `top_eig_align` didn't fire (P0-F asserts).
- Reload path ignores thin state (P0-A fixes; until then, in-process eval only).
- New densified cells get identity quaternions (fyi.md §10) — quantify %
  active post-densify.
- Single-iteration fixed-point degrades for steep |h/r|.
- Memory at 1M (EXP-AB-7).

## Variable importance (prior; update as P2 results land)

| Variable | Expected impact | Tested by |
|---|---|---|
| `thin_surface_start` | high | EXP-AB-1 |
| `thin_surface_delta_weight` | high | EXP-AB-2 |
| `thin_surface_height_weight` | medium | EXP-AB-2 |
| warm-start (`top_eig_align_weight`) | medium-high | EXP-CH-3 vs CH-4, EXP-AB-3 |
| `thin_surface_K` | low-medium (deferred) | EXP-AB-8 |
| co-train `gradient_start` | medium | EXP-AB-6 |
| `final_points` | low-medium | EXP-AB-7 |
