# LC64 Air-Artifact Split-Cell Diagnosis — Execution Amendment v2

This amendment supersedes the E0 and Stage-D semantics in `specs/LC64-AIR-SPLIT-DIAGNOSIS-PLAN-v1.md`; all other gates remain unchanged.

## Resolved E0 comparison contract
The first independent-side treatment is **native raw-side Adam**, not a falsely coordinate-matched optimizer:

`mu_plus = softplus(raw_plus, beta=10) * activation_scale`

`mu_minus = softplus(raw_minus, beta=10) * activation_scale`

Initialize `raw_plus = raw_minus = legacy scalar raw density`, establishing exact scalar-equivalent rendered density. `raw_plus` and `raw_minus` are separate ordinary Adam groups with identical Adam settings and one common swept **raw split-parameter LR**. The legacy base density is not rendered or trained as a third density degree in independent mode.

Controlled: initialization, fixed-64k data/rays/seed, point-freeze schedule, loss, budget, hard-side evaluation, physical contrast penalty, and numeric LR values `{5e-5, 2e-4, 5e-4; conditional 2e-3}`.

Intentionally different: independent sides can alter physical mean and contrast through raw side parameters; bounded-relative mode retains base-mean optimization and bounded contrast. Do not claim matched mean/difference Adam schedules or pure-expressivity isolation. Report physical side mean and contrast.

A custom rotated-coordinate Adam is explicitly deferred to a later, separately planned ablation.

## E0 blocking acceptance tests
Before any Stage A run:
1. scalar/relative-zero/independent-zero agree on fixed-ray projections, loss, hard-side volume, and point gradients;
2. symmetric fixture preserves equal raw-side gradients under symmetric rays/loss;
3. GPU FD matrix covers raw plus/minus, crossing/noncrossing and both dp signs, asymmetric sides, and low-air negative raw values; raw-gradient chain rule is checked;
4. independent checkpoint representation discriminator/tensors validate, round-trip, reproduce query/projection, and reject mixed/malformed state;
5. no rendered third base-density degree exists in independent mode.

## True stationary-frame control
Add `points_hard_freeze_at` (default disabled) without changing legacy `freeze_points` behavior. At the configured boundary and before the intended frozen update: point optimizer LR is zero, primal points are non-trainable, and point optimizer state policy is explicit. Test the boundary convention at `T-1,T,T+1`: points are unchanged (≤1e-7), point LR is zero, and later triangulation cannot mutate them. Stage D uses `points_hard_freeze_at={0,500,1500}`; legacy `freeze_points` must not be interpreted as an abrupt freeze.

## Revised Stage B labels
Relative and independent arms use the same numeric values but are labelled **raw split-parameter LR**. All Stage B density-only arms keep q/heights/sites frozen. Stage E remains separately gated geometry learning.
