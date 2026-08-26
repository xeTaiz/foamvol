# RadFoam feature sweep and follow-up: concise handoff

## Scope and measurement contract

All headline values below use the corrected hard-volume evaluator: centre-registered 256³ voxelization, supersample 4, followed by `eval_vol.py`. Chamfer improvement is reported as baseline minus candidate, so positive is better. Capacity experiments are not treated as free feature gains.

The original chest revalidation contains 94 completed runs. The initial multi-scene screen contains 213 non-smoke runs across chest, pepper, and engine, plus three smoke controls. Follow-up Stage 0 completed 15 fixed-seed reproducibility runs; Stage 1 completed 180/180 replicated main-effect runs; Stage 2A completed 126/126 paired-combination runs. All Stage-1 and Stage-2A activation assertions passed.

## Fixed 256k baseline

`configs/fixed_final/256k.yaml` trains for 10,000 iterations from 64k to 256k requested points. Its active/default mechanism is variance-based redundancy pruning with `redundancy_cap: 0.05`, `prune_hops: 1`, `grad_smooth_hops: 1`, and the default densification mixture `gradient/idw/entropy = 0.4/0.3/0.3`, five entropy bins.

Chest five-seed baseline: **34.8725 ± 0.0655 dB PSNR**, **1.5388 ± 0.0759 Chamfer**, approximately 236k–238k achieved cells. The same-host four-run PSNR floor was tighter: 34.9009 ± 0.0188 dB.

Follow-up paired baselines on seeds 42–44 were:

| Scene | PSNR | Chamfer |
|---|---:|---:|
| chest | 34.8519 | 1.5104 |
| pepper | 35.0579 | 0.6011 |
| engine | 34.8893 | 0.4459 |

## Chest revalidation: robust findings

Eleven one-seed arms promoted; eight replicated beyond the precommitted two-sigma rule.

| Arm | PSNR mean ± sd | ΔPSNR | Chamfer mean ± sd | Chamfer improvement | Verdict |
|---|---:|---:|---:|---:|---|
| `P_cap03` | 35.1548 ± 0.0231 | +0.2823 | 1.5281 ± 0.0751 | +0.0107 | PSNR win |
| `P_idw` | 35.1807 ± 0.0750 | +0.3082 | 1.6090 ± 0.0268 | -0.0702 | PSNR win |
| `REFG_prune` | **35.1991 ± 0.0594** | **+0.3266** | 1.4892 ± 0.0166 | +0.0496 | PSNR winner at 256k |
| `D_bins3` | 34.9352 ± 0.0466 | +0.0628 | 1.2418 ± 0.0245 | +0.2970 | Chamfer win |
| `D_entonly` | 34.7355 ± 0.0884 | -0.1370 | 1.2892 ± 0.0809 | +0.2496 | Chamfer win, PSNR loss |
| `C512` | 34.8307 ± 0.2635 | -0.0418 | 1.1637 ± 0.0409 | +0.3751 | Capacity/Chamfer win |
| `C1M` | 34.5899 ± 0.0377 | -0.2826 | 1.0378 ± 0.0558 | +0.5010 | Capacity/Chamfer win, PSNR loss |
| `C2M` | 33.7415 ± 0.0288 | -1.1310 | 1.0437 ± 0.0416 | +0.4951 | Capacity/Chamfer win, PSNR loss |

`NV_1e4`, `S_he40`, and `TV_1e4` did not survive replication.

At 512k, matched three-seed `REFG_prune` reached 35.0318 ± 0.0396 dB versus 34.9141 ± 0.1355 for the plain baseline: nominal +0.1176 dB, below the matched 0.2710 dB two-sigma bar, with Chamfer worse by 0.0873. Therefore the REFG 256k gain is a low-budget artifact, not a demonstrated scalable win.

## Corrections from the initial three-scene screen

Two apparent grids were not real grids:

- `P_idw*` used `redundancy_threshold: 0`; the IDW removable mask was empty. These were repeated no-redundancy-pruning runs, not cap comparisons. `P_idw_cap05` is not a cap winner.
- `REFG_prune_eps*` changed `ref_guided_eps`, but that parameter is read only by reference-guided densification, which was disabled. These were semantic duplicates and do not rank epsilon.

Their same-seed duplicate spreads were large enough to reverse small margins: no-prune PSNR ranges were 0.1595/0.3022/0.1170 dB on chest/pepper/engine; REFG duplicate ranges were 0.5155/0.1055/0.1294 dB. This motivated replicated, paired scene/seed blocks.

Capacity scaling was scene-dependent: 512k improved Chamfer strongly on chest (-0.345 candidate-minus-baseline) and modestly on engine (-0.044), but not pepper (+0.007). At 1M/2M, chest lost PSNR while engine gained; capacity is a separate axis.

## Stage 1: replicated single mechanisms

The leading pruning settings form a tie band rather than a reliable strict order:

| Single mechanism | Mean paired ΔPSNR | Interpretation |
|---|---:|---|
| `REFG_C02_A20` | +0.393 dB | highest raw pruning mean; retained roughly 10k–14k extra cells |
| `NOPRUNE` | +0.366 dB | strong but capacity-confounded |
| `IDW020` | +0.350 dB | actual nonzero-threshold IDW; approximately capacity-neutral |
| `CAP030` | +0.341 dB | active variance pruning; modest extra cell count |
| `TV1e3` | +0.283 dB | strongest TV-only PSNR setting |
| `TV3e4` | +0.244 dB | joint PSNR/Chamfer setting |

`ENT60_B3` was retained as a geometry specialist. `TV3e4` and `ENT60_B3` were not selected as maximum-PSNR settings; they preserve or improve geometry better than the more aggressive PSNR configurations.

## Stage 2A: replicated two-way interactions

All eight pruning × TV combinations passed the precommitted PSNR-complementarity rule. Values are aggregate means across chest, pepper, and engine.

| Combination | ΔPSNR vs baseline | ΔPSNR vs better parent | Chamfer improvement vs better parent | Main caveat |
|---|---:|---:|---:|---|
| `NOPRUNE_TV1e3` | **+0.775 dB** | **+0.378 dB** | -0.075 | roughly 10k–14k extra cells |
| `IDW020_TV1e3` | +0.731 | +0.355 | -0.066 | strongest approximately capacity-neutral PSNR choice |
| `REFG_A20_TV1e3` | +0.700 | +0.278 | -0.087 | roughly 10k–14k extra cells |
| `IDW020_TV3e4` | +0.614 | +0.254 | -0.045 | moderate geometry loss |
| `REFG_A20_TV3e4` | +0.619 | +0.214 | -0.030 | capacity-confounded |
| `NOPRUNE_TV3e4` | +0.609 | +0.229 | -0.036 | capacity-confounded |
| `CAP030_TV1e3` | +0.582 | +0.203 | -0.018 | smaller geometry loss |
| `CAP030_TV3e4` | +0.532 | +0.174 | **+0.003** | cleanest balanced PSNR/geometry choice |

`CAP030_TV3e4` beat its better PSNR parent in every scene: +0.148 dB chest, +0.204 pepper, +0.170 engine. It was essentially neutral in aggregate Chamfer.

`TV3e4_ENT60_B3` was the geometry interaction: +0.214 dB PSNR over baseline, +0.106 dB over its better parent, and +0.030 Chamfer improvement beyond its better parent. Its Chamfer improvements were +0.032 chest, +0.013 pepper, and +0.057 engine.

Entropy generally did not combine with pruning. Parent-relative PSNR for `CAP030_ENT60_B3`, `NOPRUNE_ENT60_B3`, `REFG_A20_ENT60_B3`, and `IDW020_ENT60_B3` was approximately +0.049, +0.013, -0.059, and -0.087 dB respectively; none passed complementarity.

Current interpretations:

- Maximum raw PSNR: `NOPRUNE_TV1e3`, but capacity-confounded.
- Strongest capacity-neutral PSNR candidate: `IDW020_TV1e3`.
- Best balanced PSNR/geometry candidate: `CAP030_TV3e4`.
- Best geometry-specialist interaction: `TV3e4_ENT60_B3`.
- Motivated three-way test: `CAP030_TV3e4_ENT60_B3`; not measured yet.

## Feature state

### Enabled in the fixed baseline

- 256k requested final-point budget; 10k iterations.
- Variance-based redundancy pruning, cap 0.05, one-hop criterion.
- Gradient/IDW/entropy densification mixture 0.4/0.3/0.3 with five entropy bins.
- One-hop gradient smoothing.
- Raw-density L1 reconstruction objective.

### Deliberately varied in the current screen

- Variance cap 0.03.
- Actual IDW pruning: threshold 0.02, cap 0.02.
- No redundancy pruning.
- Reference-guided pruning: cap 0.02, edge alpha 20.
- Raw-density TV at `3e-4` and `1e-3`.
- Entropy-heavy densification: 0.2/0.2/0.6 with three bins.
- The eight complementary pruning × TV pairs, `TV3e4_ENT60_B3`, and `CAP030_TV3e4_ENT60_B3`.

### Definitely disabled in baseline and current candidates unless explicitly named above

- Direct reference-volume loss (`ref_volume_weight: 0`): tested variants did not promote.
- Reference-guided densification: off; epsilon sweeps were invalid semantic duplicates.
- Gradient-threshold densification: not carried forward after severe underfill and losses up to -3.86 dB/+1.29 Chamfer.
- Neighbor-variance and voxel-variance regularizers: weights zero; no replicated reason to enable.
- Eigen, Laplacian, CVT, variance-volume, and bilateral-filter training regularizers: not carried forward.
- TV area weighting, border-only TV, and TV annealing: off.
- High-error/targeted sampling: starts at -1, therefore off.
- Training-time interpolation and bilateral filtering: starts at -1, therefore off.
- Gaussian mode and joint Gaussian fine-tuning: starts at -1, therefore off.
- Linear-gradient mode: starts at -1, therefore off.
- Viewer/debug/save-volume paths: off.

These statements describe the current experiment contract, not a proof that every disabled idea can never work under another budget or implementation.

## Current full-dataset experiment

A seed-45, 75-view screen is running on all 15 R2 volumes and all 12 V100s: **15 volumes × 18 settings = 270 runs**. Settings are baseline; seven single mechanisms (`TV1e3`, `TV3e4`, `CAP030`, `IDW020`, `NOPRUNE`, `REFG_A20`, `ENT60_B3`); eight pruning × TV pairs; `TV3e4_ENT60_B3`; and the new `CAP030_TV3e4_ENT60_B3` triple.

Output: `output/multiscene_feature_followup_v1/stage3_75`. Revision: `599458b`. Same-view seed-45 baselines are mandatory references for the 45 REFG arms. A matching 25-view panel is planned after the 75-view result.
