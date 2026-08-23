# Multi-scene feature screen v1: initial results

## Status

The initial screen completed all 213 non-smoke runs across chest, pepper, and
engine. Every run used the centre-registered 256-cubed SS4 volume evaluation
path. There were 213 `DONE` markers and zero `FAILED` markers. The machine-readable
result table is `output/multiscene_feature_v1/summary.csv`.

This is a one-seed screen for each non-baseline arm. A `SCREEN_PASS` means the
measurement cleared the per-scene, five-seed baseline two-standard-deviation
rule on PSNR or Chamfer. It is a promotion criterion, not a replicated win.

| scene | baseline PSNR mean ± sd | screen passes |
|---|---:|---:|
| chest | 34.8978 ± 0.0214 dB | 20 |
| pepper | 35.0355 ± 0.0876 dB | 21 |
| engine | 34.8692 ± 0.0695 dB | 29 |

Total: 70 screen passes and 146 noise verdicts, including the three retained
smoke rows in the latter count.

## Equal-budget candidates

All entries below retain the 256k final-point budget; they are comparable as
feature/hyperparameter experiments. `delta_chamfer` is candidate minus baseline,
so negative is favourable.

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

The new local pruning grids (`P_cap02`, `P_idw_cap03`, `P_idw_cap05`) produce
screen passes in more than one scene. Whether that survives replication is not
established by this table.

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

## Limit

The screen mixes three data classes deliberately, but each arm has only one
feature seed. Promote no candidate based on these results alone. Stage-C
confirmation must compare three fresh candidate seeds with the corresponding
five-seed scene baseline before claiming a win.
