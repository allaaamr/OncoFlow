# Residual-change metrics: ΔSS, Δ-RMAE, ΔSS-L1

Implementation: `src/evaluation/change_metrics.py` (metric definitions,
eligibility guards, noise/drift diagnostics). Wired into per-case evaluation
in `scripts/full_test.py` (writes `change_metrics.csv` alongside the existing
`timepoint_metrics.csv`/`patient_metrics.csv`/`summary_metrics.csv`) and
aggregated across patients (and, optionally, across checkpoints) by
`scripts/aggregate_change_metrics.py`. Unit tests:
`tests/test_change_metrics.py`.

## Why not just use MSE/PSNR/SSIM?

Those are computed on the whole predicted follow-up volume `I1_hat` against
the ground truth `I1`. Almost every voxel is unchanged between two visits, so
they are dominated by static anatomy and are nearly maximised by the trivial
"copy the previous scan" predictor `I1_hat = I0` — a model that predicts
*nothing*. The clinically meaningful signal is exactly the part that
*changes*, so these metrics instead operate on change maps:

```
I0      = last visit  (model input)
I1      = ground-truth follow-up
I1_hat  = predicted follow-up

d_gt    = I1     - I0      # true change
d_pred  = I1_hat - I0      # predicted change
```

`d_gt - d_pred = I1 - I1_hat` — the residual change is exactly the
prediction error; `I0` cancels. ΔSS and Δ-RMAE are both normalised versions
of this same residual; they differ only in what they normalise by.

## ΔSS (Change Skill Score) — PRIMARY

```
ΔSS = 1 - Σ(d_gt - d_pred)² / Σ d_gt²        (sums over the region)
```

The denominator is the squared error the copy-baseline predictor would make.
ΔSS is skill *relative to doing nothing*:

| ΔSS | meaning |
|---|---|
| `1` | perfect prediction |
| `0` | exactly as good as copying `I0` forward (holds per patient, algebraically, regardless of how much change actually occurred) |
| `< 0` | worse than copying — the model's predicted change made things worse |

This is the standard forecast skill score against a persistence baseline —
equivalently the Nash–Sutcliffe efficiency (Nash & Sutcliffe, 1970) or an R²
against a persistence reference (Murphy, 1988). It is not a new construction.

### The r/m decomposition

```
m = ||d_pred|| / ||d_gt||                       # magnitude ratio
r = <d_pred, d_gt> / (||d_pred|| · ||d_gt||)     # pattern agreement (cosine similarity)

ΔSS = 2·r·m - m²  =  r² - (r - m)²               # exact identity, see tests/test_change_metrics.py
```

Read it as **skill = pattern quality − penalty for mis-scaled amplitude**:

- `ΔSS ≤ r²` always — you cannot beat your spatial pattern quality by
  rescaling.
- ΔSS is maximised at `m = r`, **not** `m = 1`. A model with imperfect
  pattern agreement (`r < 1`) is squared-error-optimal when it *under-shoots*
  amplitude. This is exactly why conditional-mean predictors blur — the
  metric is meant to expose that, not be surprised by it.
- `r` and `m` diagnose *different* failures:
  - low `r` → change predicted in the wrong places.
  - decent `r`, low `m` → right places, too timid (blurred/averaged-out
    change).

**Always report ΔSS with r and m.** ΔSS alone tells you there's a problem;
r/m tell you which one.

## Δ-RMAE (Residual-based Relative MAE) — SECONDARY

```
Δ-RMAE = Σ|d_gt - d_pred| / Σ 0.5·(|d_gt| + |d_pred|)      (sums over the region)
```

Range `[0, 2]`, lower is better. `0` = perfect, `2` = the copy baseline.

**Must be computed as a ratio of *sums*, never a per-voxel ratio that is then
averaged.** The per-voxel form divides by near-zero denominators in stable
tissue (registration jitter that never quite hits exact zero) and assigns
the *worst possible* penalty (2.0) to any voxel where the model correctly
predicts no change but `d_gt` is tiny nonzero noise. The resulting "average"
then mostly measures how much stable tissue the ROI contains, not model
skill. See `tests/test_change_metrics.py::test_delta_rmae_is_ratio_of_sums_not_per_voxel_average`
for a constructed counterexample.

### Why it's secondary, not primary

1. **Its own denominator contains the model's output** — a model partially
   controls its own normaliser. Predicting *uncorrelated noise* at roughly
   the right amplitude scores **≈1.41**, better than the copy baseline
   (2.0), despite containing zero useful information.
2. **It cannot tell a timid model from a wrong one.** A perfectly
   anti-correlated prediction (`d_pred = -d_gt` — shrinkage predicted where
   growth occurred) scores exactly **2.0**, identical to predicting nothing.
   ΔSS correctly separates these (`-3` vs `0`).
3. It doesn't decompose into pattern/magnitude components the way ΔSS does.

It's kept because it's bounded, robust to heavy-tailed residuals (it's an
L1 metric), and appears in prior work this repo is compared against.

## ΔSS-L1 — robustness row

```
ΔSS_L1 = 1 - Σ|d_gt - d_pred| / Σ|d_gt|
```

Same meaningful zero as ΔSS (copy baseline), but L1 instead of L2 — robust to
the heavy-tailed residuals misregistration produces at tissue boundaries. No
r/m decomposition. Use it as a sanity check: if ΔSS and ΔSS-L1 tell very
different stories, a handful of high-gradient outlier voxels are probably
driving the ΔSS number.

## Regions

Every metric above is computed independently over four regions per
(patient, timepoint):

| Region | Definition |
|---|---|
| `tumor` | Union of the baseline and follow-up tumor masks (same mask `full_test.py`'s MR_tumor uses — the two families of metrics can never silently disagree on what "tumor region" means) |
| `dilated` | `tumor` dilated by the pipeline's existing ROI dilation (`ROI_DILATION_ITERS`, ≈ a few mm — see `resolve_voxel_spacing_mm`) |
| `control` | Brain mask minus a 20mm dilation of the tumor union — **diagnostic**: ΔSS should be ≈0 here for every method. A method scoring well in the control zone is fitting noise or exploiting a normalisation artefact, not predicting real change. |
| `full` | The entire volume (no masking) |

## Eligibility guards (never impute a skipped value)

A region for a given case is **skipped** — every metric set to `NaN`, with a
`nan_reason` string — rather than silently treated as 0 or 1, when:

- it has fewer than 100 voxels, or
- `Σd_gt² / |R| < (3·σ)²`, where `σ` is a robust noise estimate (median
  absolute deviation, computed once per case from the `control` zone) — the
  region's true change is indistinguishable from noise.

This means the `control` and (often) `full` regions are frequently `NaN` by
design in cases with a small tumor and mostly-stable background — that's the
guard working as intended, not a bug.

## Reading scores against the noise ceiling, not against 1.0

```
ΔSS_max ≈ 1 - σ² / mean(d_gt²)     (per case, per region)
```

`σ` (the control-zone noise estimate) puts a ceiling below 1.0 on what any
model — including the oracle, if it were evaluated on noisy inputs instead of
exact ground truth — could plausibly achieve. Compare a model's ΔSS to this
ceiling, not to a flat 1.0: a ΔSS of 0.85 next to a ceiling of 0.87 is a
strong result; the same 0.85 next to a ceiling of 0.99 is not.

## Drift diagnostic

`drift = median(d_gt)` in the `control` zone. If intensity normalisation is
per-volume (`normalization_policy="whole_volume"` — see
`src/evaluation/metrics.py::resolve_data_range`), this is nonzero and can
correlate with how much the tumor changed, contaminating every region's
metrics. It's stored per case; `|drift| > σ` is flagged.

## Reference rows: copy baseline and oracle

`scripts/full_test.py` writes `method` = `"copy_baseline"` (`I1_hat = I0`)
and `"oracle"` (`I1_hat = I1`) rows into `change_metrics.csv` for every case,
computed algebraically (no extra model forward pass) alongside the actual
`"model"` rows. These are the reference points the results table needs:

- **copy_baseline** should show ΔSS = 0 exactly, Δ-RMAE = 2 exactly — often
  *alongside high SSIM/PSNR*. That contrast (high SSIM, zero skill) is the
  whole argument for using ΔSS at all.
- **oracle** should show ΔSS = 1, Δ-RMAE = 0, r = m = 1 exactly — a sanity
  check on the metric implementation itself, visible in every real run.

## Running it

```bash
# Per-case evaluation (writes change_metrics.csv alongside the existing CSVs)
python scripts/full_test.py --config configs/default.yaml --checkpoint <ckpt> \
    --patients-split patient_split.csv --split test --save-dir evaluations/my_run

# Aggregate across patients (and, optionally, across multiple runs/checkpoints)
python scripts/aggregate_change_metrics.py \
    --inputs evaluations/my_run/change_metrics.csv \
    --out-dir evaluations/my_run/change_metrics_aggregate
```

`aggregate_change_metrics.py` produces, per (method, region):

- **macro** aggregate (primary): each patient's own across-timepoint mean,
  then averaged across patients — every patient counts equally.
- **pooled** aggregate (secondary): raw numerators/denominators summed
  across every eligible patient-timepoint, then the ratio formed once —
  weights by how much true change occurred, robust to small-denominator
  instability in any one case. If macro and pooled disagree by more than
  `--disagreement-atol` (default 0.15 ΔSS), the script prints a warning:
  performance likely depends on how much change occurred in a given case.
- a **results table** (`results_table.csv`) with one row per method plus the
  `copy_baseline`/`oracle` reference rows, columns ΔSS/r/m/Δ-RMAE/ΔSS-L1
  alongside the existing SSIM/PSNR/MSE.
- a **scatter plot** of `||d_gt||` vs. `||d_pred||` (dilated region, one
  point per patient, identity line) — under-producing models collapse below
  the identity line.

## Unit tests

`tests/test_change_metrics.py` — 8 required checks plus extra coverage of the
eligibility/noise/control-mask helpers:

1. Identity: `ΔSS == 2rm - m²  ==  r² - (r-m)²`
2. Copy baseline (`I1_hat=I0`): `ΔSS==0`, `ΔSS_L1==0`, `Δ-RMAE==2`, `m==0`
3. Oracle (`I1_hat=I1`): `ΔSS==1`, `Δ-RMAE==0`, `r==1`, `m==1`
4. Half amplitude (`d_pred=0.5·d_gt`): `ΔSS==0.75`, `m==0.5`, `r==1`, `Δ-RMAE==2/3`
5. Anti-correlated (`d_pred=-d_gt`): `ΔSS==-3`, `Δ-RMAE==2` — documents that
   Δ-RMAE cannot distinguish "actively wrong" from "did nothing"
6. Uncorrelated noise, matched amplitude: `ΔSS≈-1`, `Δ-RMAE≈1.41` — documents
   Δ-RMAE rating useless noise as *better* than the copy baseline
7. Δ-RMAE must be a ratio of sums, not a per-voxel average (constructed
   counterexample)
8. Region invariance under padding with out-of-region voxels

Run with:

```bash
python -m pytest tests/test_change_metrics.py -v
```
