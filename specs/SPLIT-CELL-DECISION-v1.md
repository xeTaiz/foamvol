# Split-Cell Research Decision v1

## Decision
**Do not continue the current split-cell tuning/breadth campaign.**

The implementation is now substantially validated and the major correctness/lifecycle bugs were repaired, but the resulting representation has **no demonstrated advantage on the primary chest task**:

- Raw additive delta destabilizes or degrades chest projection and split-aware volume quality at all tested nonzero LR/regularization settings.
- Learned internal geometry is unstable on the cube rescue task and was intentionally frozen for chest.
- Bounded relative delta is gradient-correct and stable, but neutral or worse than the zero-learning/scalar-equivalent control across rho/LR ablations.
- No tested thin arm exceeds the scalar baseline in the required split-aware edge/volume metrics.

## What is complete
- CUDA forward/backward correctness, activation gradient equivalence, dp<0 sign case, checkpoint persistence, K plumbing, warm-start/prune lifecycle, and relative-mode FD coverage.
- Cube control/rescue ladder.
- Clean chest scalar/BA/raw-delta/relative-delta experiments with matched hard-side split evaluation.
- Durable records: `specs/SPLIT-CELL-EXECUTION-LOG.md` and `specs/SPLIT-CELL-RESULTS-v1.md`.

## Why further current sweeps are not justified
More seeds or scenes would measure a configuration already neutral/worse on the flagship chest test, while geometry is unstable. More LR/L2 tuning is unlikely to solve the fundamental two-sided line-integral identifiability problem exposed by the experiments.

## User decision required for a new research cycle
Choose or authorize one new hypothesis before reusing compute:

1. **Sparse/edge-gated split allocation:** activate split parameters only in data-supported boundary cells rather than all cells.
2. **Alternating/continuation optimization:** fit scalar foam, then optimize bounded split parameters with base density/points frozen, then controlled joint refinement.
3. **Surface primitive redesign:** represent a local surface primitive explicitly near boundaries rather than a free split in every Voronoi cell.
4. **Stop split-cell direction:** preserve correctness improvements but pivot to a different CT representation/prior.

Any selected hypothesis should follow the same gates: local FD → scalar/thin activation equivalence → controlled cube/phantom → chest control → breadth/replication.
