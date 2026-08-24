# Split-Cell Thin-Surface Experiment Plan v0

## Context
The repository has a new optional two-density thin-surface split-cell design for CT Voronoi cells. Each active cell learns `density_delta`, `quaternions`, `texel_sites_2d`, and `texel_heights`, letting a curved internal surface split each ray chord into two densities. The goal is to determine whether this improves thin-boundary reconstruction versus the prior single-density Voronoi-cell baseline.

Primary source documents reviewed by the team: `CLAUDE.md`, `fyi.md`, `configs/best428_thinsurface.yaml`, `configs/best428_nointerp.yaml`, `radfoam_model/scene.py`, `radfoam_model/render.py`, `train.py`, `src/tracing/pipeline.{h,cu}`, `torch_bindings/pipeline_bindings.cpp`, existing tests and analysis/visualization scripts.

## Current Evidence Summary
- Thin-surface implementation appears substantially wired end-to-end: CUDA forward/backward, C++ bindings, Python autograd, scene initialization, optimizer param groups, densification/pruning hooks, training activation, and regularization.
- No thin-surface experiment results were found.
- Existing tests are minimal and do not validate thin-surface gradients or checkpoint round trips.
- Existing evaluation and visualization mostly operate on scalar per-cell density/voxelization, so they can miss or discard learned internal surfaces.

## Blocking Issues Before Trusting Experiments
1. **Gradient correctness missing.** The handwritten CUDA backward through quaternion frame, soft-Voronoi height field, and fixed-point surface intersection must be finite-difference checked.
2. **Checkpoint save/load omits thin-surface state.** `model.pt` currently does not persist `density_delta`, `quaternions`, `texel_sites_2d`, `texel_heights`, or thin-surface active/K metadata. Post-training evaluation may silently become baseline.
3. **K propagation / K>4 risk.** Config exposes `thin_surface_K`, but one review found C++ settings may not propagate `thin_K`; backward has fixed-size arrays consistent with K<=8. Keep K=4 until verified.
4. **Potential gradient shape mismatch.** `density_delta` is `(N,1)` while bindings may allocate gradient as `(N,)`; verify/fix before gradcheck.
5. **Evaluation blind spots.** Voxel/mesh metrics and visualization likely query base scalar density only. Need projection-domain metrics plus split-aware diagnostics/voxel sampling before making volume/surface claims.
6. **Config confound.** `best428_thinsurface.yaml` enables `top_eig_align_weight=1e-2`; improvements may come from boundary alignment rather than split cells. Include a baseline + top-eig ablation.
7. **Reproducibility gaps.** Hardcoded seed, no mid-training resume, and `metrics.txt` lacks wall-clock time.

## Hypotheses
- H1: Thin-surface split cells improve reconstruction of sub-cell thin boundaries versus single-density Voronoi cells.
- H2: Gains will be most visible in edge/ROI/surface metrics, not necessarily global PSNR/SSIM.
- H3: Boundary-eigenvector warm-start improves stability and convergence versus identity/random orientation.
- H4: Delta and height regularization are necessary to avoid unsupported split surfaces or sparse-view overfitting.
- H5: Split cells complement rather than replace Gaussian bumps and linear density gradients; different methods win on different structure types.

## Phase 0 — Correctness Gates
### P0.1 Thin-surface finite-difference gradcheck
- Test minimal controlled rays/cells for gradients wrt base density, density_delta, quaternions, texel_sites_2d, texel_heights.
- Include crossing, non-crossing, grazing, surface-outside-chord, zero-height, nonzero-height, and delta clamp cases.
- Success: relative error <= 1e-3 where gradients are well-conditioned; documented tolerances near clamp/grazing discontinuities.

### P0.2 Checkpoint round-trip test
- Initialize nonzero thin-surface params, render fixed rays, save, reload, render again.
- Success: thin-surface params and metadata persist; projection outputs match within numerical tolerance.

### P0.3 K/config plumbing test
- Confirm `thin_surface_K` propagates from YAML/Python to C++ settings and gradients have correct shapes.
- Success: K=4 works; K=8 either passes shape/gradient sanity or is explicitly disabled with clear error.

### P0.4 Warm-start sanity test
- Confirm `_last_top_eigvec` is populated before thin-surface activation when boundary alignment is enabled.
- Success: log/report fraction of cells warm-started from eigvec vs identity fallback.

## Phase 1 — Minimal Scientific Signal
### P1.1 Controlled slab phantom
- Compare baseline vs baseline+top-eig vs thin-surface no-warm-start vs thin-surface warm-start.
- Use identical point count, rays, seed, iteration budget, and evaluation path.
- Metrics: global projection PSNR/SSIM, volume PSNR/SSIM if split-aware eval exists, slab ROI PSNR/MSE, Sobel PSNR/SSIM, edge localization/F1, active surface fraction.
- Success: thin-surface improves slab ROI and Sobel/edge metrics by meaningful margins without degrading global metrics beyond tolerance.

### P1.2 Debug phantoms
- Run on `shepp_logan_n75_clean`, `marschner_lobb_n75_clean`, and `nema_iec_n75_clean`.
- Compare baseline, baseline+top-eig, thin-surface K=4.
- Use 3 seeds if seed configurability is added; otherwise run one seed and clearly label results preliminary.

## Phase 2 — Ablations
- `thin_surface_start`: {4000, 6000, 7000}.
- `thin_surface_delta_weight`: {1e-3, 1e-4}; `thin_surface_height_weight`: {5e-4, 5e-5}.
- Warm-start on/off: `top_eig_align_weight` {0, 1e-2}.
- K: keep K=4 until plumbing verified; later compare K=4 vs K=8.
- Method comparison: baseline vs Gaussian mode vs linear density gradient vs thin-surface vs combinations.

## Phase 3 — Breadth Benchmarks
- Run best thin-surface config vs best scalar baseline on 10 synthetic R2 n75 scenes and selected `ct_org` cases.
- Primary metrics: mean and per-scene Vol/Sobel PSNR, projection PSNR/SSIM, F1_1v/F1_2v, mesh CD/HD95 where valid.
- Report regressions as well as wins; no scene should regress >0.5 dB without explanation.

## Diagnostics and Visualization Requirements
- Split-aware slice views: mu_bar, mu_plus, mu_minus, delta, active height norm, normal overlays, and Voronoi borders.
- Training dynamics: active-surface fraction, mean/percentiles of |delta|, height L1 norm, quaternion norm, normal coherence, warm-start alignment.
- Error maps: projection residuals and slice residuals, especially around thin-boundary ROI.
- For paper figures: controlled slab/vessel visual with internal surface overlay, ablation table, and failure-case plots.

## Compute Plan
- Correctness gates: minutes to short GPU jobs.
- Minimal P1 matrix: roughly 5-12 runs, estimated 2-4 GPU hours depending on final run budget.
- Full P2/P3 matrix: roughly 30-60 runs, estimated 14+ GPU hours serial; use worker split if available.
- Adopt job naming: `ts_<scene>_<factor>_<level>_seed<seed>` and summarize to CSV.

## Success Criteria
- P0 gates all pass before any claim of improvement.
- Controlled slab: ROI/Sobel/edge metrics improve over baseline and baseline+top-eig; global PSNR not worse by >0.2 dB.
- Debug phantoms: at least two of three show edge/thin-structure gains without severe global metric loss.
- Breadth: mean metrics improve or are neutral, and no systematic regressions on smooth-tissue cases.
- Diagnostics show learned surfaces are sparse and spatially correlated with true/reconstructed boundaries, not random high-error noise.

## Open Questions
- Which controlled thin phantom should be first: simple slab, cylindrical vessel wall, cortical shell, or all three?
- Should we restrict initial claims to projection-domain reconstruction until split-aware voxelization is implemented?
- How much implementation work should be done before first run: only P0 blockers, or also split-aware visualization/voxelization?
- What GPU budget and wall-clock deadline should constrain the first benchmark matrix?
