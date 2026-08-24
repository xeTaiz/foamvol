# Split-Cell Thin-Surface Experiment Plan v1

## Purpose
Plan the next validation and benchmark steps for the new two-density split-cell / thin-surface design in Radfoam CT. This updates `specs/SPLIT-CELL-EXPERIMENT-PLAN-v0.md` after specialist priming and Claude planning review.

## Key Repository Facts From Priming
- The feature is intended to split a Voronoi cell chord into two densities with an internal oriented curved surface.
- Implementation appears substantially present across CUDA/C++, Python autograd, scene params, densification/pruning, optimizer groups, training activation, and regularization.
- No trustworthy thin-surface experimental result exists yet.
- Existing tests and evaluation are not sufficient to validate the feature.

## Must-Fix / Must-Verify Before Trusting Any Benchmark
1. **Checkpoint persistence and reload behavior.** Save/load must include `density_delta`, `quaternions`, `texel_sites_2d`, `texel_heights`, `_thin_surface_active`, and `_thin_K`. Otherwise evaluation after reload silently becomes baseline.
2. **Gradient correctness.** Add finite-difference checks for base density, `density_delta`, quaternion orientation, texel sites, and texel heights. Include crossing, non-crossing, grazing, outside-chord, clamp, zero-height, and nonzero-height cases.
3. **Gradient shape and autograd contract.** Verify `density_delta` grad shape matches `(N,1)` or is reshaped safely; confirm all returned gradients match parameter shapes.
4. **K plumbing.** Confirm `thin_surface_K` propagates from config to C++/CUDA. Keep K=4 until K=8 is explicitly verified or guarded.
5. **Evaluation path validity.** Confirm the planned metric path actually uses thin-surface rendering/parameters. For early work, projection-domain evaluation may be safer than reload-based voxel/mesh metrics until split-aware voxelization exists.
6. **Warm-start sanity.** If using boundary-eigenvector warm-start, assert/log that `_last_top_eigvec` is populated and quantify fallback-to-identity cells.
7. **Feature inertness / learning-dynamics gate.** Because zero-initialized geometry has zero effect when `density_delta=0`, correctness tests can pass while the feature remains effectively inert. Add a small controlled optimization test showing `density_delta` becomes nonzero and then geometry params receive useful gradients/improve loss.

## Can Defer Until After P0/P1
- Full split-aware voxelizer and mesh metrics, if initial claims are limited to projection-domain and controlled visual diagnostics.
- K=8 or larger curvature tests.
- Mid-training resume support.
- Full seed infrastructure, though at least CLI seed support is strongly preferred before multi-seed claims.
- Full 10-15 scene benchmark suite.
- Paper-level polished figures.

## Clean Experimental Comparisons Needed
Avoid comparing `best428_nointerp.yaml` directly against `best428_thinsurface.yaml` as the only evidence, because the latter also enables boundary alignment. Minimum clean ablation:
1. Baseline scalar Voronoi.
2. Baseline + boundary alignment / top-eig regularizer.
3. Thin-surface without warm-start / no boundary alignment.
4. Thin-surface with boundary-alignment warm-start.

All runs should use the same cell budget, rays, training schedule, seed, and evaluation path unless the factor is intentionally varied.

## Phase P0 — Correctness and Plumbing Gates
- P0.1 checkpoint round-trip: initialize nonzero thin-surface state, render fixed rays, save/reload, render again, compare outputs and tensors.
- P0.2 CUDA finite-difference gradcheck on a tiny deterministic scene/ray set.
- P0.3 config/K/shape test for K=4; either pass or explicitly guard K=8.
- P0.4 warm-start instrumentation.
- P0.5 inertness/learning test: demonstrate a toy case where loss drives nonzero `density_delta` and then nonzero geometry gradients; verify regularization does not suppress all activation.

Success: all P0 gates pass before any benchmark is considered meaningful.

## Phase P1 — First Scientific Signal
### Controlled phantom
Use a simple known thin structure first: slab or shell/vessel wall thinner than median cell diameter.

Run the four clean comparisons above.

Metrics:
- Projection PSNR/SSIM/RMSE.
- ROI MSE/PSNR around the thin structure.
- Sobel/edge PSNR or edge F1.
- Active-surface diagnostics: fraction active, |delta| distribution, height norm distribution, normal alignment to known surface, learned split location error if available.

Success:
- Thin-surface beats both scalar baseline and baseline+boundary-alignment on ROI/edge metrics.
- Global metrics do not regress beyond a small tolerance, e.g. >0.2 dB PSNR loss.
- Diagnostics show surfaces activate near the true thin boundary rather than randomly.

### Debug phantoms
Next candidates: `shepp_logan_n75_clean`, `marschner_lobb_n75_clean`, `nema_iec_n75_clean`. Use one seed for smoke tests; use 3 seeds before making claims.

## Phase P2 — Ablation Sweeps
After P1 signal:
- `thin_surface_start`: {4000, 6000, 7000}.
- Regularization grid: `density_delta_weight` {1e-3, 1e-4} × `height_weight` {5e-4, 5e-5}.
- Warm-start on/off.
- Method comparison: scalar baseline, Gaussian mode, linear density gradient, thin-surface, and selected combinations.
- K=4 vs K=8 only after K plumbing/gradchecks pass.

## Phase P3 — Breadth Benchmarks
Run the best validated thin-surface setting against the best scalar baseline on a broader R2 synthetic suite and selected `ct_org` cases.

Primary reporting:
- Per-scene table, not only mean.
- Projection metrics and edge/ROI metrics.
- Volume/mesh metrics only if evaluation is confirmed split-aware or the limitation is explicitly stated.
- Regression audit: identify smooth scenes or cases where split cells hurt.

## Visualization / Diagnostics Needed Early
- Slice overlays of Voronoi borders plus learned internal surface/normal direction.
- Maps of `mu_bar`, `mu_plus`, `mu_minus`, `delta`, height norm, active cells.
- Histograms over training: |delta|, height norm, quaternion norm, normal coherence.
- Projection residuals and ROI residuals.
- Warm-start normal vs learned normal alignment.

## Compute Plan
- P0: short GPU jobs/tests.
- P1: roughly 5-12 runs; existing runs suggest ~20-30 min per 512k run, so several GPU hours.
- P2/P3: tens of runs; likely 14+ GPU hours serial, less with workers.

## User Decisions / Open Questions
1. Should first claims be projection-domain only until split-aware voxelization exists?
2. Should we fix checkpoint persistence before any run, or allow a temporary in-process eval path for the very first smoke test?
3. Which controlled phantom should be first: slab, shell/cortical wall, vessel cylinder, or all three?
4. Do we want seed configurability before P1, or is one deterministic seed acceptable for smoke testing only?
5. What GPU budget should bound P1 and P2?
6. Is K=8 in scope now, or should we explicitly lock initial experiments to K=4?
