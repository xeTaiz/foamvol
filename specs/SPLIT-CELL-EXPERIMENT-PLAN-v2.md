# Split-Cell Thin-Surface Experiment Plan v2

## Locked User Decisions
- Start with cube using `old/test_cube.py`.
- Then run `r2_data` `0_chest_cone` as configured in the repository config.
- Implement split-aware voxelization first; skip meshing for now.
- Fix checkpoint save/load now.
- One deterministic seed is acceptable for initial smoke tests.
- Compute should target `kw995` and `kw996` RTX 6000 Ada machines using Compute Manager's available `wh_` tools.
- Initial experiments use K=4 only.
- Orchestrator should continue driving implementation, bug fixes, ablations, and progress tracking while user is away.

## Execution Phases

### Phase 0 — Engineering and Correctness Gates
Must complete before serious benchmark claims:
1. Checkpoint round-trip support for thin-surface parameters and metadata.
2. Verify/fix C++/CUDA config plumbing for K=4 and guard/disable unverified K values.
3. Verify/fix `density_delta` gradient shape/autograd contract.
4. Add finite-difference/controlled gradient tests for thin-surface forward/backward.
5. Add split-aware voxelization function similar to existing query/supersampling voxelization.
6. Add diagnostics for warm-start, active cells, delta/height norms, and zero-init inertness.

### Phase 1 — First Runs
Order:
1. Cube smoke test via `old/test_cube.py`.
2. `r2_data` `0_chest_cone` run.

Clean comparison set:
- Scalar baseline.
- Scalar baseline + boundary alignment.
- Thin-surface K=4 without boundary warm-start.
- Thin-surface K=4 with boundary warm-start.

Primary metrics:
- Projection PSNR/SSIM/RMSE.
- Split-aware voxel PSNR/SSIM where applicable.
- Thin-structure ROI/Sobel/edge metrics if available.
- Diagnostics: active-surface fraction, |delta| distribution, height norm, normal coherence/alignment.

### Phase 2 — Keep-Busy Ablation Queue
To be expanded by Experiment Designer after P1 results:
- `thin_surface_start` sweep.
- `thin_surface_delta_weight` × `thin_surface_height_weight` sweep.
- Warm-start on/off.
- Gaussian mode vs linear density gradient vs thin-surface.
- Additional scenes beyond chest.
- K=8 only after K=4 path is robust and K plumbing/gradcheck is extended.

## Progress Tracking
Use `specs/SPLIT-CELL-EXECUTION-LOG.md` as the concise catch-up file. Experts should also consider brief updates in `.pi/team-updates.md` when their work affects others.
