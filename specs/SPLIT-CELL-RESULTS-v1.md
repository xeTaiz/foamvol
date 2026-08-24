# Split-Cell Thin-Surface Results v1

**Status:** synthesized experimental record, 2026-07-19.  
**Branch lineage:** original thin surface → CUDA crossing-gradient fix `ec7d615` → warm-start/prune fix `095944f` → bounded relative delta `0c9e32e` → relative GPU gradcheck `ff27608` / test fix `5781ea9` → diagnostics fix `0cfe1ec`.

## What is verified

1. **Core CUDA correctness**
   - Zero-init thin forward, loss, base-density gradient, and point gradient now match scalar mode in crossing cells (point max difference `1.192e-07`).
   - `dp<0` non-crossing FD: density relative error `4.26e-05`; delta relative error `8.81e-05`.
   - Relative-delta GPU FD checks pass for activation continuity and crossing/non-crossing density/raw-delta paths (max tested FD relative error `6e-04`).

2. **Activation/lifecycle correctness**
   - Checkpoint thin-state persistence, K/shape handling, activation continuity, and stale warm-start cache after prune all have targeted tests.
   - Warm-start crash at iter6000 after standalone prune was fixed in `095944f`.

3. **Representation/optimization evidence**
   - Raw additive delta is unstable or harmful on chest even with frozen geometry and small LR.
   - Bounded relative delta is stable and mathematically safe, but did not improve chest reconstruction versus zero-learning control in tested settings.
   - Learned height/quaternion/site geometry is unstable on the cube rescue task at tested nonzero LRs; chest main matrix intentionally froze geometry.

## Decisive cube rescue results

| Run | Setup | Test PSNR | VolRaw PSNR | Dice | Decision |
|---|---|---:|---:|---:|---|
| scalar 1a | manual cube | 58.86 | 89.52 | 1.000 | control |
| thin 1a | K4, start0 | 10.38 | 41.80 | .998 | early activation unsafe |
| scalar 1b | random/densified cube | 40.94 | 36.07 | .992 | control |
| R0 | delta+geometry learned | -8.40 | — | .862 | fails |
| R1 | all thin LR=0 | 45.20 | 40.68 | .997 | activation safe |
| R2 | delta LR 5e-5, geometry frozen | 43.74 | 41.25 | .998 | stable; split PSNR 43.46 vs scalar 37.43 |
| R3 | geometry scale 1e-3 | 37.77 | 40.68 | .997 | geometry degrades |
| R3b | geometry scale 1e-4 | 42.63 | 38.94 | .996 | geometry still degrades |

## Decisive chest results

All split metrics use the same hard-side nearest-cell query field, 256³ volume, and 4³ supersampling.

| Arm | Parameterization | Test PSNR | Split PSNR | Split Sobel | Split Dice | Interpretation |
|---|---|---:|---:|---:|---:|---|
| CH1_clean | scalar | 49.26 | 35.01 | 17.35 | .857 | main baseline |
| CH2_clean | scalar + BA | 47.55 | 34.06 | 16.70 | .850 | BA worse |
| CH3_clean | raw delta .01, geometry frozen | 1.03 | 27.94 | 12.31 | .745 | catastrophic |
| CH4_clean | raw delta .01 + BA warm-start | -10.14 | 24.17 | 10.04 | .607 | worse |
| CH5 | raw delta LR=0 control | 48.79 | 35.01 | 17.35 | .857 | activation stable |
| CH6 | raw delta LR scale=1e-4 | 32.72 | 34.46 | 16.87 | .854 | harmful |
| CH7 | raw delta LR=1e-4, L2=.1 | 30.80 | 34.43 | 16.93 | .853 | harmful |
| CH8 | relative rho=.5, LR=.01 | 49.22 | 35.01 | 17.33 | .858 | neutral |
| CH9 | relative rho=.5, LR=0 | 49.06 | 35.06 | 17.36 | .858 | matched control |
| CH10 | relative rho=.5, LR=.1 | 48.94 | 34.55 | 16.45 | .856 | worse |
| CH11 | relative rho=.25, LR=.01 | 48.63 | 35.02 | 17.33 | .858 | neutral |

## Claims supported / not supported

### Supported
- A two-density Voronoi split can be implemented with differentiable CUDA ray tracing, correct zero-init activation behavior, and correct FD gradients.
- Bounded relative delta eliminates the observed raw-delta runaway and preserves chest reconstruction stability.
- The current learned split formulation has no demonstrated advantage over scalar cells on `0_chest_cone` under tested settings.

### Not supported
- No claim that split cells improve anatomical CT reconstruction.
- No claim that learned internal geometry helps; tested geometry learning is unstable.
- No paper-ready generalization, seed replication, or breadth result.

## Next hypothesis required for progress

Further tuning of raw/bounded delta LR and L2 is not justified by current evidence. A new representation/optimization hypothesis is needed before more broad compute, for example:
- surface activation sparsity/selection learned from edge evidence rather than all cells;
- likelihood/regularizer that resolves the line-integral identifiability of a two-sided split;
- a surface-only primitive allocation near high-confidence boundaries;
- a separate continuation/alternating optimizer for split parameters.

Any new hypothesis must first pass: single-cell FD, scalar-to-thin gradient equivalence, activation control, and cube rescue before chest/breadth.
