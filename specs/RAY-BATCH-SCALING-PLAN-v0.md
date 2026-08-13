# Fixed-cell ray-batch scaling experiment — v0

Status: proposed / not launched

## Question
Does increasing rays per optimizer step improve gradient coverage and consistency enough to improve fixed-cell CT reconstruction, especially learned split-plane normals, when total sampled rays are held fixed and optimizer steps are reduced?

## Common controls
- Dataset: 75-view synthetic chest; seed 42; uniform ray sampling.
- Cell counts fixed independently at 64k and 128k (`init_points=final_points`, `densify_until=-1`).
- Total sampled-ray budget: 13B per end-to-end run, matching the existing `1M×9000 + 4M×1000` schedule.
- Keep that legacy ramp only in reference arms; high-batch arms use a constant batch size.
- L1 mean loss; same evaluation path: hard-side 256³/SS4 plus held-out projections.
- Scale all event times by sampled-ray exposure rather than raw optimizer iteration. Point hard-freeze and split activation at 1.5B rays; LR schedule endpoint at 13B rays.
- Scalar arm: ordinary flat density per Voronoi cell.
- Split arm: bounded relative delta (`rho=.5`), learned quaternion + K=4 heights, frozen 2D sites. Use matched scalar-equivalent warm phase and GT-independent boundary/geometric normal initialization before geometry unfreeze so the test is not dominated by the known large-angle basin problem.
- Report volume PSNR/SSIM, Sobel PSNR, Dice/F1/Chamfer/HD95, strict-air metrics, held-out projection metrics, hit-count distribution, zero-gradient/starvation fraction, and split geometry diagnostics.

## Stage 1: 64k screen
Eight end-to-end runs:

| Schedule | Rays/batch | Iterations | Total rays | LR multiplier | Scalar | Split |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 1M→4M at step 9000 | 10,000 | 13B | 1x | yes | yes |
| Large/same-LR | 4M | 3,250 | 13B | 1x | yes | yes |
| Large/adapted-LR | 4M | 3,250 | 13B | 2x | yes | yes |
| Very-large/adapted-LR | 8M | 1,625 | 13B | 3x | yes | yes |

LR multiplier applies to point, base-density, delta, quaternion, and height schedule endpoints. Gradient clipping remains unchanged. The 4M same-LR arm separates batch/step-count effects from LR adaptation; 4M 2x tests conservative sqrt-batch scaling; 8M 3x extends that rule.

At a small number of fixed ray-exposure checkpoints, compute two independent-batch gradient estimates and report per-cell gradient cosine/agreement, binned by hit count and cell radius. For split, compare normal-tangent/quaternion gradients; for scalar, density gradients. This is the direct test of the gradient-quality hypothesis.

## Stage 2: 128k confirmation
After Stage 1, select one high-batch schedule without looking only at projection PSNR. Run four arms:
- 128k scalar 1M reference
- 128k split 1M reference
- 128k scalar selected high-batch schedule
- 128k split selected high-batch schedule

Selection prioritizes hard-volume/Sobel quality, then surface and strict-air metrics, and requires improved gradient coverage/agreement. No broad 128k LR sweep.

## Stage 3: hit-confidence update ablation
Only at the selected 64k high-batch schedule, add one scalar and one split run with a per-cell confidence multiplier. Use CUDA forward hit counts and an EMA. Apply the multiplier to the actual cell-local Adam update, not only to raw gradients (Adam can cancel static gradient rescaling):

`confidence_i = clamp((ema_hits_i / median_positive_ema_hits)^0.5, 0.1, 1.0)`

Apply initially to cell attributes only (scalar/base density, delta, quaternion, heights); leave point updates unchanged to avoid a topology confound. Compare against the already-run no-confidence arm. Log multiplier quantiles and metric changes by hit-count/radius bins.

## Advancement / interpretation
- A larger batch is useful only if it improves hard-volume or Sobel quality, or gives a clear surface/air gain without >0.30 dB hard-volume/Sobel loss.
- Projection-only gains do not qualify.
- Treat fewer optimizer steps and altered Adam dynamics as explicit variables; do not interpret equal total sampled rays as equal optimization work.
- If 4M same-LR worsens but 4M adapted-LR recovers, infer an optimizer-step/LR issue, not that ray coverage is unhelpful.
- Do not launch Stage 2 or 3 until Stage 1 is complete.

## Runtime estimate
A direct 8M smoke test on RTX A6000 measured approximately 0.38 s/step scalar and 1.13 s/step split at 64k, using about 5 GB GPU memory. This predicts roughly 10 minutes scalar and 31 minutes split for 8M×1,625 before final diagnostics/evaluation. Existing full LC64 runs suggest 25–45 minutes for slower split/reference arms. With seven available A6000 GPUs:
- Stage 1 training + evaluations: about 1–1.5 hours wall time.
- Stage 2: about 0.5–1 hour.
- Stage 3: about 0.5 hour.
- Total gated compute after implementation: about 2–3 hours wall time; allow 4–6 hours end-to-end including implementation, tests, dispatch, evaluation, and artifact collection.
