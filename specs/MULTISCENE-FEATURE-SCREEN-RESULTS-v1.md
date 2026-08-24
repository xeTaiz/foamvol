# Multi-scene feature screen v1: initial results

## Status

The initial screen completed all 213 non-smoke runs across chest, pepper, and
engine, with zero failures. Every run used the centre-registered 256-cubed SS4
volume evaluation path. The three smoke controls bring the evidence set to 216
completed rows.

This is a one-seed screen for each non-baseline arm. A `SCREEN_PASS` means the
measurement cleared the per-scene, five-seed baseline two-standard-deviation
rule on PSNR or Chamfer. It is a promotion criterion, not a replicated win.

The active TensorBoard tree was curated after completion. Its
`output/multiscene_feature_v1/summary.csv` now contains 88 data rows; the 128
completed rows moved during curation remain under
`output/multiscene_feature_v1_old/<scene>/<tag>/eval_vol.json`. Both locations
are evidence. The corrected interpretation and replicated follow-up matrix are
in [`MULTISCENE-FEATURE-FOLLOWUP-PLAN-v1.md`](MULTISCENE-FEATURE-FOLLOWUP-PLAN-v1.md).

| scene | baseline PSNR mean ± sd | screen passes |
|---|---:|---:|
| chest | 34.8978 ± 0.0214 dB | 20 |
| pepper | 35.0355 ± 0.0876 dB | 21 |
| engine | 34.8692 ± 0.0695 dB | 29 |

Total: 70 screen passes and 146 noise verdicts, including the three retained
smoke rows in the latter count.

## Equal-budget measurements

All entries below retain the 256k final-point budget; they are comparable as
executions of equal-budget configurations. The table preserves the raw
one-seed measurements. It does not imply that every named knob activated.
`delta_chamfer` is candidate minus baseline, so negative is favourable.

| scene | tag | family | delta PSNR (dB) | delta Chamfer |
|---|---|---|---:|---:|
| chest | REFG_prune_eps100 | REFG_GRID | +0.3909 | +0.088505 |
| chest | P_idw_cap03 | PRUNE_GRID | +0.3751 | +0.009050 |
| chest | P_cap02 | PRUNE_GRID | +0.3573 | +0.055159 |
| engine | TV_1e3 | TV | +0.5484 | -0.019817 |
| engine | P_idw_cap05 | PRUNE_GRID | +0.5129 | -0.017523 |
| engine | REFG_prune | REFG | +0.4865 | -0.017426 |
| pepper | REFG_prune_eps050 | REFG_GRID | +0.4513 | +0.015156 |
| pepper | P_idw | PRUNE | +0.4384 | +0.003319 |
| pepper | P_cap02 | PRUNE_GRID | +0.4318 | -0.012863 |

`P_cap02` is active variance-pruning cap evidence and passes in all three
scenes. The `P_idw*` rows require a different interpretation:
`prune_variance_criterion: false` combines with `redundancy_threshold: 0.0`, so
the IDW removable mask is empty and `redundancy_cap` cannot affect training.
`P_idw`, `P_idw_cap03`, `P_idw_cap05`, and `P_idw_cap10` are same-seed
repetitions of no redundancy pruning, not cap comparisons. In particular, the
old interpretation of `P_idw_cap05` as a cap winner is invalid.

The `REFG_prune_eps005/020/050/100` rows are also semantic duplicates of
`REFG_prune`: `ref_guided_eps` is read only by reference-guided densification,
while these arms set `ref_guided_densify: false`. They provide no evidence for
ranking epsilon values.

The complete same-seed groups include archived pepper `P_idw_cap03` at
35.1717 dB and archived chest `REFG_prune` at 34.7732 dB:

| semantic duplicate family | chest range / sd | pepper range / sd | engine range / sd |
|---|---:|---:|---:|
| no redundancy pruning (`P_idw*`) | 0.1595 / 0.0672 dB | 0.3022 / 0.1352 dB | 0.1170 / 0.0512 dB |
| reference-guided prune (`REFG_prune*`) | 0.5155 / 0.2014 dB | 0.1055 / 0.0385 dB | 0.1294 / 0.0509 dB |

This spread is an empirical runtime-noise warning: fixed scene, seed, and
effective configuration can differ enough to overturn small one-seed margins.
Raw rows are retained because that repeated-run evidence is useful.

## Capacity ladder: separate axis

`C512`, `C1M`, and `C2M` alter `final_points` to 512k, 1.024M, and 2.048M
respectively; they are capacity experiments, not feature experiments. They are
therefore only compared with one another here, not ranked beside equal-budget
arms.

| scene | C512 delta PSNR / Chamfer | C1M delta PSNR / Chamfer | C2M delta PSNR / Chamfer |
|---|---:|---:|---:|
| chest | +0.1559 / -0.345306 | -0.3088 / -0.503232 | -1.0557 / -0.381683 |
| engine | +0.4914 / -0.043525 | +0.9565 / -0.053399 | +0.8554 / -0.027076 |
| pepper | +0.3742 / +0.006532 | +0.5562 / +0.050181 | +0.2268 / +0.138911 |

## Interpretation and limit

This remains a one-seed feature screen, not a replicated multi-scene win.
Variance-based pruning with a 0.02–0.03 cap is the cleanest repeated
equal-budget signal. TV at `1e-3` is promising on pepper and engine but not
universal. Entropy-biased densification remains a geometry specialist.
Reference-guided pruning and actual nonzero-threshold IDW pruning require
targeted investigation. Capacity arms remain a separate budget class.

The follow-up protocol in
[`MULTISCENE-FEATURE-FOLLOWUP-PLAN-v1.md`](MULTISCENE-FEATURE-FOLLOWUP-PLAN-v1.md)
first measures same-seed reproducibility, then uses paired scene/seed blocks,
activation assertions, and joint PSNR/Chamfer promotion rules. Promote no
candidate from this screen alone.
