# Sweep-revalidate v1: results

## Status

The final analysis contains 94 completed runs (5 baseline + 50 screen + 33
Stage-C confirmations + 6 matched-host Stage-D runs), all scored through the
corrected evaluator. Eight Stage-C candidates clear the precommitted
two-standard-deviation rule. `REFG_prune` is the Stage-C PSNR winner, but
its 512k result does **not** retain a significant PSNR improvement over the
matched 512k baseline. This is evidence that **REFG_prune's** gain is a
low-budget artifact; `P_idw` and `P_cap03`, the other confirmed PSNR winners,
were not re-tested at 512k.

The earlier worker-harness outage is resolved. During it, the connector
reported `runtime: failed to create new OS thread (have 7 already; errno=11)
... fatal error: newosproc`; this blocked result retrieval despite completed
training. The recovered connection confirmed all 33 Stage-C markers,
including `S_he40_s44`, then retrieved every evaluation JSON.

## Stage A — noise floor

Five replicates of the unmodified `configs/fixed_final/256k.yaml` (`BASE_s42..46`),
differing only in `seed`, scored through `split_voxelize.py` (256³, supersample 4)
→ `eval_vol.py` → `air_metrics.py`.

| tag | psnr | chamfer | num_cells |
|---|---|---|---|
| BASE_s42 | 34.9064 | 1.6016 | 238053 |
| BASE_s43 | 34.8740 | 1.5693 | 237780 |
| BASE_s44 | 34.9178 | 1.4575 | 237416 |
| BASE_s45 | 34.9053 | 1.4570 | 237805 |
| BASE_s46 | 34.7589 | 1.6086 | 235741 |

`σ_psnr = 0.0655 dB`, `σ_chamfer = 0.0759` over all 5 replicates (mean psnr
34.8725, mean chamfer 1.5388). Both are well under the 0.30 dB stop-and-report
bound from the plan's Verification §5, so the screen is interpretable.

**Host-split caveat.** `BASE_s42..45` ran on `KW60996` (torch 2.13.0+cu126);
`BASE_s46` ran on `KW60995` (torch 2.7.1+cu126) because of a screening-time GPU
allocation decision (see Known caveats). The four same-host replicates alone
give `σ_psnr = 0.0188 dB` (mean 34.9009) — a 3.5× tighter floor than the
5-replicate figure. `BASE_s46` sits 0.14 dB below the KW60996 mean, consistent
with either a genuine outlier seed or a small cross-host/cross-torch-version
offset; five replicates cannot distinguish the two. The decision rule below
uses the full 5-replicate `σ = 0.0655 dB` because that is what the manifest
commits to. This is **not** conservative for every arm: 16 of the 50
Stage-B arms (`D_ca1`, `D_ca4`, `D_thr500`, `D_thr1000`, `D_thr1000_v5e3`,
`P_idw`, `P_cap03`, `P_cap10`, `P_hops2`, `S_tgt0`, `S_tgt30`, `S_he0`,
`S_he40`, `S_he_p2`, `BF_dens`, `BF_late`) ran on `KW60995`, not `KW60996`,
including three screening passes (`P_idw`, `P_cap03`, `S_he40`). If the
~0.14 dB `BASE_s46` gap is a real host/torch-version offset rather than
seed noise, it works *against* those three arms' favourable direction (PSNR
up), so their reported `Δpsnr` of +0.41, +0.27 and +0.06 dB are understated,
not inflated. The remaining 34 Stage-B arms plus all 5 `BASE_*` and all 33
Stage-C runs ran on `KW60996`.

REF_NPY (for `REF_*`/`REFG_*`/`INIT_ref`) = `output/sweep_revalidate/BASE_s42/volume_hard_ss4.npy`.

## Stage B — screen (50 arms, 1 replicate each, base 256k unless noted)

Full table (PSNR/chamfer/etc. from `eval_vol.json`, air columns from
`air_metrics.json`, `num_cells` from `metrics.txt`; host per arm noted above):

| tag | family | psnr | Δpsnr | chamfer | Δchamfer | num_cells | verdict |
|---|---|---|---|---|---|---|---|
| BF_dens | BF | 33.9741 | -0.8984 | 1.6648 | +0.1260 | 231041 | noise |
| BF_late | BF | 33.4060 | -1.4665 | 1.9408 | +0.4020 | 236505 | noise |
| C1M | CELLS | 34.6032 | -0.2693 | 1.0350 | -0.5038 | 988893 | **SIGNIFICANT** |
| C2M | CELLS | 33.8843 | -0.9882 | 1.0280 | -0.5108 | 1892884 | **SIGNIFICANT** |
| C512 | CELLS | 34.9661 | +0.0936 | 1.1218 | -0.4170 | 462204 | **SIGNIFICANT** |
| CVT_1e2 | CVT | 30.8535 | -4.0190 | 1.5518 | +0.0130 | 242861 | noise |
| CVT_1e3 | CVT | 34.2813 | -0.5912 | 1.3991 | -0.1397 | 242110 | noise |
| D_bins12 | DENS | 34.9109 | +0.0384 | 1.6928 | +0.1540 | 237586 | noise |
| D_bins3 | DENS | 34.9345 | +0.0620 | 1.2820 | -0.2568 | 237118 | **SIGNIFICANT** |
| D_ca1 | DENS | 34.8661 | -0.0063 | 1.5191 | -0.0197 | 236950 | noise |
| D_ca4 | DENS | 34.8801 | +0.0076 | 1.5473 | +0.0085 | 237282 | noise |
| D_ent0 | DENS | 34.8701 | -0.0023 | 1.7642 | +0.2254 | 236746 | noise |
| D_entonly | DENS | 34.6762 | -0.1963 | 1.2270 | -0.3118 | 239117 | **SIGNIFICANT** |
| D_gradheavy | DENS | 34.8944 | +0.0219 | 1.7549 | +0.2161 | 237575 | noise |
| D_thr1000 | DENS | 31.3666 | -3.5059 | 2.6083 | +1.0695 | 54865 | noise |
| D_thr1000_v5e3 | DENS | 31.0173 | -3.8552 | 2.8299 | +1.2911 | 52825 | noise |
| D_thr500 | DENS | 32.2473 | -2.6252 | 2.0039 | +0.4651 | 90961 | noise |
| EIG_1e2 | EIG | 33.6169 | -1.2556 | 1.4209 | -0.1179 | 239786 | noise |
| EIG_1e3 | EIG | 34.8108 | -0.0617 | 1.5395 | +0.0007 | 238527 | noise |
| GS_0 | GS | 34.8207 | -0.0517 | 1.5319 | -0.0069 | 236328 | noise |
| GS_2 | GS | 34.8792 | +0.0067 | 1.4951 | -0.0437 | 237352 | noise |
| INIT_ref | INIT | 34.8210 | -0.0515 | 1.6482 | +0.1094 | 244815 | noise |
| LAP_1e2 | LAP | 32.8191 | -2.0534 | 1.4350 | -0.1038 | 239384 | noise |
| LAP_1e3 | LAP | 34.6442 | -0.2283 | 1.5807 | +0.0419 | 240086 | noise |
| NV_1e3 | NV | 34.8281 | -0.0444 | 1.5857 | +0.0470 | 237292 | noise |
| NV_1e3_huber | NV | 34.8892 | +0.0167 | 1.5319 | -0.0069 | 237298 | noise |
| NV_1e3_median | NV | 34.8849 | +0.0124 | 1.7423 | +0.2035 | 237154 | noise |
| NV_1e3_sharp | NV | 34.8898 | +0.0174 | 1.5139 | -0.0249 | 237773 | noise |
| NV_1e4 | NV | 34.9100 | +0.0376 | 1.3788 | -0.1600 | 237382 | **SIGNIFICANT** |
| P_cap03 | PRUNE | 35.1474 | +0.2749 | 1.5558 | +0.0170 | 241815 | **SIGNIFICANT** |
| P_cap10 | PRUNE | 34.8951 | +0.0226 | 1.4718 | -0.0670 | 233156 | noise |
| P_hops2 | PRUNE | 34.7767 | -0.0957 | 1.4514 | -0.0874 | 250047 | noise |
| P_idw | PRUNE | 35.2861 | +0.4136 | 1.5405 | +0.0017 | 251328 | **SIGNIFICANT** |
| REF_1e2 | REF | 34.8796 | +0.0071 | 1.4706 | -0.0682 | 236888 | noise |
| REF_1e3 | REF | 34.8876 | +0.0151 | 1.6711 | +0.1323 | 237374 | noise |
| REF_1e3_noedge | REF | 34.9502 | +0.0777 | 1.4991 | -0.0397 | 237729 | noise |
| REFG_dens | REFG | 34.8620 | -0.0105 | 1.5643 | +0.0255 | 236797 | noise |
| REFG_prune | REFG | 35.1263 | +0.2538 | 1.5314 | -0.0074 | 247299 | **SIGNIFICANT** |
| S_he0 | SAMPLE | 34.8266 | -0.0459 | 1.5747 | +0.0359 | 237379 | noise |
| S_he40 | SAMPLE | 34.9349 | +0.0624 | 1.3727 | -0.1661 | 236639 | **SIGNIFICANT** |
| S_he_p2 | SAMPLE | 34.7447 | -0.1278 | 1.4248 | -0.1140 | 237028 | noise |
| S_tgt0 | SAMPLE | 34.8450 | -0.0275 | 1.4823 | -0.0565 | 236263 | noise |
| S_tgt30 | SAMPLE | 34.7554 | -0.1171 | 1.5745 | +0.0357 | 237800 | noise |
| TV_1e3 | TV | 34.9793 | +0.1068 | 1.8368 | +0.2980 | 237031 | noise |
| TV_1e4 | TV | 35.0053 | +0.1328 | 1.5264 | -0.0124 | 237547 | **SIGNIFICANT** |
| TV_1e4_area | TV | 34.9802 | +0.1077 | 1.4483 | -0.0905 | 237924 | noise |
| TV_1e4_border | TV | 34.6952 | -0.1773 | 1.5843 | +0.0455 | 235173 | noise |
| VV_1e2 | VV | 34.7851 | -0.0874 | 1.7246 | +0.1858 | 237491 | noise |
| VV_1e3 | VV | 34.8947 | +0.0222 | 1.4921 | -0.0467 | 237477 | noise |
| VV_1e4 | VV | 34.8507 | -0.0218 | 1.6252 | +0.0864 | 237286 | noise |

Reference rows (same evaluator, quoted from prior work, never differenced
against the numbers above per plan caveat):

| tag | psnr | ssim_3d | dice | chamfer | hausdorff_95 | f1_1v | f1_2v |
|---|---|---|---|---|---|---|---|
| R2-Gaussian | 35.8512 | 0.943398 | 0.889397 | 0.7283 | 4.1286 | 0.8838 | 0.9481 |
| SC256_ctrl | 34.8299 | 0.924534 | 0.849052 | 1.4388 | 12.7060 | 0.7880 | 0.8777 |

**Activation audit (Verification §6).** All 21 arms carrying a non-null
`assert_scalar` (`TV_1e4`, `TV_1e3`, `TV_1e4_area`, `TV_1e4_border`, `NV_1e4`,
`NV_1e3`, `NV_1e3_huber`, `NV_1e3_median`, `NV_1e3_sharp`, `VV_1e4`, `VV_1e3`,
`VV_1e2`, `EIG_1e3`, `EIG_1e2`, `LAP_1e3`, `LAP_1e2`, `CVT_1e3`, `CVT_1e2`,
`REF_1e3`, `REF_1e2`, `REF_1e3_noedge`) read `ACTIVE` — every regularizer that
was supposed to be on actually logged a nonzero loss. No arm needed
recording as untested-inactive except the deliberate `SMOKE_tv_inactive`
negative control.

**Cell-count achieved budgets (Verification §7).** `C512` reached 462204
(≈90.3% of its `final_points: 512000`, matching the prior 512k run's 462712),
`C1M` reached 988893 (≈96.6% of `1m.yaml`'s `final_points: 1024000`), `C2M`
reached 1892884 (≈92.4% of its overridden `final_points: 2048000`) — the
extended `densify_until: 7000` schedule was sufficient; no ray-batch
reduction fallback was needed.

## Stage-B screening passes (11 arms → promoted to Stage C)

`C1M`, `C2M`, `C512`, `D_bins3`, `D_entonly`, `NV_1e4`, `P_cap03`, `P_idw`,
`REFG_prune`, `S_he40`, `TV_1e4`.

Immediate caveats visible even before Stage C:

- **C1M and C2M pass on chamfer while losing on PSNR** (C1M -0.27 dB, C2M
  -0.99 dB vs the 256k baseline mean) — cell count buys geometric accuracy
  (chamfer, F1) at a PSNR cost at this ray/iteration budget held fixed across
  cell counts. `C512` is the only CELLS arm that also holds PSNR flat
  (+0.09 dB, within noise) while still passing on chamfer (-0.42, clearly
  outside 2σ). This matches the "cell-count scaling is real but budget the
  ray schedule with it" pattern flagged as a known caveat in
  `manifest.yaml`.
- **REF_1e3, REF_1e2, REF_1e3_noedge (the reference-volume loss, all three
  weight/edge-mask variants tested) do not pass**, but `REFG_prune`
  (reference-guided pruning, loss weight left at 0) does. The bug-corrupted
  family's first valid measurement is therefore a split verdict, not a
  blanket "it never worked": guiding *what gets pruned* with the reference
  field helped; adding it as a direct loss term did not, at any of the
  three configurations tested.
- **NV_1e4 passed on chamfer, not PSNR, and does not survive the tighter
  same-host floor.** Its `Δchamfer = -0.1600` clears the manifest's
  committed `2σ_chamfer = 0.1518` (5-replicate, mixed-host) bar; `Δpsnr =
  +0.0376` does not clear `2σ_psnr = 0.1310` on its own; the pass is on
  chamfer alone. NV_1e4 ran on `KW60996`, so re-measured against the
  4-replicate same-host floor (`BASE_s42-45` only, mean psnr 34.9009 dB,
  mean chamfer 1.5214, `σ_psnr = 0.0188`, `σ_chamfer = 0.0752`), NV_1e4 is
  `Δpsnr = +0.0092` (needs `> 0.0376`) and `Δchamfer = -0.1425` (needs
  `> 0.1504`) — it fails **both** axes under the tighter bar. Of the 11
  screening passes, NV_1e4 is the one whose pass is entirely an artefact of
  the wider, mixed-host noise estimate; its Stage-C confirmation is the
  most informative test in the batch.
- **D_thr500/1000/1000_v5e3 (gradient-threshold densification) are badly
  negative** on both axes (up to -3.86 dB, +1.29 chamfer) — these arms also
  reached far fewer cells (52825-90961 vs the ~237000 baseline) because
  `densify_grad_thresh` mode uses a fixed-interval schedule decoupled from
  `final_points`, so this screen conflates "wrong densification budget" with
  "descriptor doesn't work." Flagged, not resolved, by this sweep.

## Known caveats

- **Host split.** `BASE_s42-45`, 34 of 50 Stage-B arms, and all 33 Stage-C
  arms ran on `KW60996`; `BASE_s46` plus the other 16 Stage-B arms
  (enumerated above, including screening passes `P_idw`/`P_cap03`/`S_he40`)
  ran on `KW60995`, whose other 2 GPUs were free at Stage-B launch time
  (GPU0 there was pre-occupied by an invisible external tenant). Stage C
  was restricted entirely to `KW60996` once this was noticed, specifically
  to avoid compounding the possible host/torch-version confound into the
  3-replicate confirmation.
- **torch 2.13.0+cu126 (KW60996) vs 2.7.1+cu126 (KW60995).** Both print a
  compile-version mismatch warning against the CUDA extension (built for
  2.5.1) on every run, on both hosts; this is a pre-existing repo condition,
  not introduced by this sweep, and did not prevent any arm from completing.
- **REF_NPY/BASE_s42 circularity.** `ref_volume_path`/`init_volume_path` for
  the bug-corrupted family point at `BASE_s42`'s own SS4 reconstruction, not
  at `vol_r2.npy`, specifically so that a positive result here cannot be
  read as "regularizing toward R2." This makes `REF_*`/`REFG_*`/`INIT_ref`
  a self-consistency check (does the model's own converged reconstruction
  make a good guide signal) rather than an oracle-supervision check.
- **SMOKE_tv / SMOKE_tv_inactive are verification fixtures**, not screen
  data: 400-iteration truncated schedule, excluded from the decision-rule
  table by construction (`SMOKE` family), retained above only to document
  the guard's negative-control result (`INACTIVE`, retained with `DONE`,
  exactly the previously-missing check).
- **A KW60996 worker-harness restart during Stage B and Stage C.** Once
  during Stage B (killed 4 in-flight arms: `REFG_prune`, `TV_1e4_area`,
  `D_bins12`, `EIG_1e3`, all cleanly resumed from their per-arm `rm -rf`
  reset with no cross-contamination) and once late in Stage C (see Stage C
  section). Git state and `output/sweep_revalidate/` persisted across both
  restarts because `/code` is a durable volume; only in-flight training
  processes were lost.

## Stage C — confirmation (11 candidates × seeds 43/44/45)

Stage-A baseline: $34.8725 \pm 0.0655$ dB PSNR and
$1.5388 \pm 0.0759$ chamfer ($n=5$). The precommitted confirmation bars are
$0.1311$ dB PSNR or $0.1518$ chamfer improvement; chamfer deltas below are
`baseline mean − candidate mean`, so positive is favourable.

| arm | PSNR mean ± sd | ΔPSNR | chamfer mean ± sd | Δchamfer | Stage-C verdict |
|---|---:|---:|---:|---:|---|
| C1M | 34.5899 ± 0.0377 | -0.2826 | 1.0378 ± 0.0558 | +0.5010 | **win (chamfer)** |
| C2M | 33.7415 ± 0.0288 | -1.1310 | 1.0437 ± 0.0416 | +0.4951 | **win (chamfer)** |
| C512 | 34.8307 ± 0.2635 | -0.0418 | 1.1637 ± 0.0409 | +0.3751 | **win (chamfer)** |
| D_bins3 | 34.9352 ± 0.0466 | +0.0628 | 1.2418 ± 0.0245 | +0.2970 | **win (chamfer)** |
| D_entonly | 34.7355 ± 0.0884 | -0.1370 | 1.2892 ± 0.0809 | +0.2496 | **win (chamfer)** |
| NV_1e4 | 34.9317 ± 0.0214 | +0.0593 | 1.5100 ± 0.0801 | +0.0288 | noise |
| P_cap03 | 35.1548 ± 0.0231 | +0.2823 | 1.5281 ± 0.0751 | +0.0107 | **win (PSNR)** |
| P_idw | 35.1807 ± 0.0750 | +0.3082 | 1.6090 ± 0.0268 | -0.0702 | **win (PSNR)** |
| REFG_prune | **35.1991 ± 0.0594** | **+0.3266** | 1.4892 ± 0.0166 | +0.0496 | **win (PSNR; Stage-D winner)** |
| S_he40 | 34.6524 ± 0.3867 | -0.2201 | 1.5180 ± 0.0159 | +0.0208 | noise |
| TV_1e4 | 34.9854 ± 0.0551 | +0.1129 | 1.5111 ± 0.0234 | +0.0277 | noise |

Eight candidates replicate a result beyond the two-sigma bar. The five
chamfer wins are not interchangeable with a PSNR win: all three CELLS arms
and `D_entonly` lose PSNR, while `D_bins3` remains PSNR-neutral. `P_cap03`,
`P_idw`, and `REFG_prune` clear the PSNR bar; `REFG_prune` is the unique
highest mean-PSNR arm, so Stage D tests it rather than selecting on a
single-seed screen outcome.

`NV_1e4`, `S_he40`, and `TV_1e4` fail confirmation. In particular, the
borderline Stage-B `NV_1e4` chamfer result vanishes under replication, and
the `S_he40_s43` low outlier ($34.2081$ dB) makes the sample-family result
plainly non-robust.

## Stage D — 512k re-test

All reported Stage-D rows ran on `KW60996` (torch 2.13.0+cu126), so the
comparison is host- and runtime-matched. `D_BASE_s43..45` and
`D_REFG_prune_s43..45` retain the Stage-A `BASE_s42` SS4 reconstruction as
the fixed reference signal for the REFG arm.

| arm | PSNR mean ± sd | SSIM-3D mean ± sd | dice mean ± sd | chamfer mean ± sd | HD95 mean ± sd |
|---|---:|---:|---:|---:|---:|
| plain 512k baseline | 34.9141 ± 0.1355 | 0.925552 ± 0.002008 | 0.857264 ± 0.001629 | **1.1884 ± 0.0328** | **9.0044 ± 0.2138** |
| REFG_prune 512k | **35.0318 ± 0.0396** | **0.927563 ± 0.000211** | **0.859012 ± 0.000701** | 1.2757 ± 0.0910 | 10.5475 ± 1.1792 |

`REFG_prune` is nominally +0.1176 dB at 512k, but the matched baseline's
$2\sigma_\mathrm{PSNR}=0.2710$ dB, so it does not meet the same
precommitted significance rule that promoted it from Stage C. Chamfer is
also worse by 0.0873. **Verdict: REFG_prune's 256k PSNR win does not survive
the 512k re-test; for this arm, the evidence supports a low-budget artifact,
not a scalable improvement.** `P_idw` and `P_cap03` also cleared the Stage-C
PSNR bar (35.1807 and 35.1548 dB respectively) and remain the obvious
512k follow-ups; Stage D did not test them.

Two initial `D_REFG_prune_s44/s45` executions on `KW60995` completed after
a checksummed transfer of `BASE_s42`, but were excluded from the comparison:
they would have mixed torch 2.7.1+cu126 with the `KW60996` baseline. Both
were re-run with the same tags on `KW60996` before aggregation; only those
matched-host results are reported above.

## Legacy `summary.csv` cross-reference

`KW60898:/data_ibex_c2324/foamvol/output/` is reachable again and contains
historical `summary.csv` files for `sweep12`, `sweep13`, `sweep15`, and
`sweep16`. Their arm identifiers (for example `A5-sv05`, `C1-ss01`, and
`B4-sv-flat`) do not map to this manifest's independently named arms, and
the files omit the corresponding resolved configurations. No claimed
old-to-new pairing is therefore defensible. Their `vol_raw_psnr` fields
remain deliberately unquoted rather than fabricating a correspondence; in
any event they use the older, misregistered single-sample evaluator and
must never be differenced from the corrected values above.
