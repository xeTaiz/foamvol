# Fixed-cell ray-batch scaling — results v1

## What was tested
- Synthetic 75-view chest, seed 42, L1 mean projection loss.
- Fixed 64k and 128k Voronoi cells; no densification/pruning.
- Scalar flat-density cells versus bounded relative split cells (`rho=.5`) with learned density delta, quaternion, and K=4 heights; 2D sites frozen.
- Point hard-freeze and split activation at 1.5B sampled rays.
- Each arm used exactly 13B sampled rays:
  - `ref`: 1M/batch for 9,000 steps then 4M/batch for 1,000 (10,000 steps), LR 1x.
  - `4m_lr1`: 4M × 3,250, LR 1x.
  - `4m_lr2`: 4M × 3,250, LR 2x.
  - `8m_lr3`: 8M × 1,625, LR 3x.
- Initial split LRs at 1x: delta `5e-4`, quaternion/heights `2e-4`; all base and split LRs scaled together for 2x/3x arms.
- Hard-side evaluation at 256³/SS4; strict-air ROI fixed from GT.

## Primary hard-side results

| Cells | Mode | Schedule | Volume PSNR | Sobel PSNR | SSIM3D | Dice | Air MAE | Air FPR | Proj PSNR |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|64k|scalar|ref|28.9811|27.0547|.8485|.9388|.0006988|.0011888|40.670|
|64k|scalar|4m_lr1|25.7769|24.7234|.7362|.9119|.0013416|.0111757|32.249|
|64k|scalar|4m_lr2|27.3090|25.8950|.7936|.9296|.0008901|.0035339|36.035|
|64k|scalar|8m_lr3|18.6485|18.0822|.4560|.8153|.0016228|.0160725|26.587|
|64k|split|ref|29.5999|27.6553|.8699|.9432|.0006469|.0007164|42.777|
|64k|split|4m_lr1|26.9109|25.6317|.7853|.9179|.0012443|.0101176|30.953|
|64k|split|4m_lr2|28.4744|26.8874|.8370|.9353|.0008102|.0026387|40.596|
|64k|split|8m_lr3|18.9797|18.2429|.4763|.8235|.0015294|.0163497|27.188|
|128k|scalar|ref|29.9818|27.9875|.8785|.9460|.0006022|.0006651|42.755|
|128k|scalar|4m_lr1|26.7050|25.4724|.7791|.9213|.0010660|.0076946|35.823|
|128k|scalar|4m_lr2|28.1746|26.7458|.8261|.9363|.0007532|.0023576|38.626|
|128k|scalar|8m_lr3|19.7162|18.7277|.4987|.8426|.0012803|.0129226|28.849|
|128k|split|ref|30.4918|28.5800|.8923|.9491|.0005721|.0004405|44.340|
|128k|split|4m_lr1|27.6865|26.3040|.8162|.9261|.0009771|.0070608|36.356|
|128k|split|4m_lr2|29.2235|27.7343|.8581|.9414|.0006932|.0016751|42.805|
|128k|split|8m_lr3|20.0139|18.8799|.5239|.8513|.0011934|.0122774|29.942|

## Surface metrics

| Cells | Mode | Schedule | Chamfer | HD95 | F1@1 | F1@2 |
|---:|---|---|---:|---:|---:|---:|
|64k|scalar|ref|5.379|39.12|.547|.693|
|64k|split|ref|4.603|36.37|.605|.747|
|64k|split|4m_lr1|4.499|32.12|.467|.690|
|64k|split|4m_lr2|5.503|41.98|.529|.707|
|128k|scalar|ref|4.054|32.47|.618|.754|
|128k|split|ref|4.069|32.19|.657|.784|
|128k|split|4m_lr1|3.888|30.63|.526|.743|
|128k|split|4m_lr2|3.800|31.31|.604|.766|

## Findings
1. Larger physical batches with proportionally fewer Adam updates did not improve end-to-end reconstruction at fixed sampled-ray budget. The reference schedule won every volume, Sobel, air-MAE, and air-FPR comparison.
2. At 4M, doubling LR recovered roughly half the loss from reducing optimizer steps, but remained 1.13–1.81 dB below the same-mode reference in volume PSNR and had 21–27% worse air MAE. The 8M/3x setting was catastrophic (about 9–11 dB below reference).
3. Split cells beat matched scalar cells in every schedule for volume and Sobel PSNR. On the reference schedule: +0.619/+0.601 dB at 64k and +0.510/+0.593 dB at 128k; air MAE improved 7.4%/5.0%, and FPR 39.7%/33.8%.
4. Split reference also improved surface F1 at both counts. Some 4M split arms achieved lower Chamfer/HD95 but sacrificed volume, edge, air, and usually F1 quality, so they do not advance.
5. Increasing 64k→128k improved the reference by about 0.9–1.0 dB volume and Sobel PSNR and reduced air errors.
6. Equal total rays produced little wall-time benefit: ray tracing dominated, so fewer optimizer steps did not materially shorten training.

## Important configuration caveat
The generator set `densify_from` to the number of steps corresponding to a 1B-ray point-LR warmup (1000/250/125 steps for ref/4M/8M). Historical LC64 configs used `densify_from: 0`, so the `ref` arms are not exact reproductions of the earlier LC64 baselines; for example, the new 64k scalar reference is 1.86 dB below the prior Stage-A scalar. This warmup was exposure-matched across the new matrix, so within-matrix batch comparisons remain interpretable, but absolute comparisons to prior LC64 studies are confounded. A corrected no-warmup replication is required before treating the split-reference gain as a definitive replacement for earlier results.

## Active split-cell audit and closeups
`experiments/analyze_split_cells.py` (`88516b9`) samples a 192³ grid, assigns each point to its exact nearest Voronoi owner, and checks whether the learned implicit field has both signs inside each sampled cell. A primary meaningful split additionally requires at least 8 finite samples, base density ≥5% of GT p99, absolute side difference ≥1% of GT p99, and relative side difference ≥10%.

| 64k split run | Surface crosses sampled cell | Meaningfully active | Fraction all cells |
|---|---:|---:|---:|
| ref | 51,282 | 12,432 | 19.4% |
| 4m_lr1 | 53,056 | 17,517 | 27.4% |
| 4m_lr2 | 53,097 | 15,771 | 24.6% |
| 8m_lr3 | 54,895 | 13,119 | 20.5% |

The reference conclusion is robust to stricter thresholds: 6,581 cells (10.3%) cross with relative contrast ≥20% and absolute difference ≥2% of GT p99; 1,961 (3.1%) cross with relative contrast ≥50% and absolute difference ≥5% of GT p99. Thus the split parameters are not universally dormant. They are difficult to identify in full-volume TensorBoard views because each 64k cell occupies only a few output pixels and no Voronoi/surface overlay is shown. Closeups render neighboring borders in white, the selected cell border in yellow, and the learned `s=0` surface in ultra-thin magenta. Artifacts are under `output/ray_batch_split_closeups/` locally and the browser gallery is served separately.

## Decision
- Do not promote 4M/8M fewer-step schedules.
- Do not launch hit-confidence update arms on these failed high-batch schedules.
- If continuing, rerun only the four reference scalar/split arms (64k/128k) and the most promising 4M/2x pair with `densify_from: 0`, or hold optimizer-step count fixed and increase total rays to isolate gradient quality from update count.
