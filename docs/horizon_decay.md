# `horizon_decay` preservation mode

`region_aware_source.preservation_mode = "horizon_decay"` — a third
preservation-map mode alongside `hard_dilation` and `local_density`
(`src/utils/preservation_map.py`). It is a formula evaluated in the
dataloader/model's source-construction step, exactly like the other modes:
**no trainable parameters, no new loss term, no network.**

## Motivation

`hard_dilation` and `local_density` both build a preservation map `P(r)`
that depends only on space `r` — it is a **fixed map**, identical whether
the target visit is 30 days or 3 years away. `horizon_decay` is the only
mode that also depends on `dt`, the number of clinical days between the
most recent previous visit and the target visit.

## Formula

Reusing the same tumor-density field `R = K * M0` (`compute_local_tumor_density`,
a Gaussian/box-smoothed baseline mask) and `gamma` that `local_density` uses:

```
w(r)     = R(r) ** gamma                        # tumor weight in [0, 1]
P(r, dt) = (1 - w(r)) * exp(-lambda_bg * dt)
```

Two deliberately separable factors:

- **`(1 - w(r))` — spatial.** Zero in dense tumor, rising to 1 far from it.
  Unlike `local_density`, there is **no `p_min` floor** — `w(r) = 1` gives
  `P = 0` exactly, at every `dt`.
- **`exp(-lambda_bg * dt)` — temporal.** 1 at `dt = 0` (full trust), decaying
  towards 0 as the horizon grows. Distant healthy tissue is fully trusted at
  short horizons and only slowly loosens.

`P` is clamped to `[0, 1]` after computation for numerical safety (it is
already in range analytically for `dt >= 0`).

## The interpretable `lambda_bg` parameterization

A raw `lambda_bg` (~1e-4/day) is not something anyone can reason about
directly, so it is never set in config. Instead:

```yaml
region_aware_source:
  horizon_decay:
    gamma: 1.0
    bg_retention: 0.90              # trust remaining in distant healthy tissue...
    bg_retention_horizon_days: 365  # ...after this many days
```

```python
lambda_bg = -math.log(bg_retention) / bg_retention_horizon_days
```

(`src/utils/preservation_map.py::bg_retention_to_lambda`.) `bg_retention: 0.90`
at 365 days gives `lambda_bg ≈ 2.9e-4/day`, i.e. distant tissue keeps ~0.97 at
90 days, 0.90 at a year, 0.81 at two years. `TaGeDiff.__init__` prints the
derived `lambda_bg` and the implied retention at a handful of reference
horizons (90/365/730 days plus the configured horizon) whenever
`preservation_mode == "horizon_decay"`, so the operating point is visible in
every run log.

**Sensitivity check, not a tuning target:** run `bg_retention` ∈
`{0.95, 0.90, 0.80}` at a 365-day horizon as a robustness sweep.

## The two blends

`region_aware_source.blend` (`linear` | `vp`, default `linear`) controls how
any preservation map — from **any** of the four modes, not just
`horizon_decay` — turns into the flow-matching source state `z_0^RA`
(`src/utils/flow_matching.py::blend_source`):

```
linear (default, unchanged): z_0^RA = P * z_prev + (1 - P) * eps
vp     (new):                z_0^RA = P * z_prev + sqrt(1 - P**2) * eps
```

`eps` is the repo's existing standard-FM noise source
(`torch.randn_like(z_future)` in `ConditionalFlowMatching.sample_path`/
`construct_flow_path` — the exact same draw `flow_path="standard"` uses as
`x_0`), reused unmodified; `horizon_decay` does not introduce a new noise
term.

**Why `vp`:** under `linear`, assuming `z_prev` and `eps` have matched
(roughly unit) variance, the source variance is `P^2 + (1-P)^2` times the
data variance — a curve that dips to `0.5` at `P = 0.5`, a ~29% std deficit
(`sqrt(0.5) ≈ 0.707`) exactly in the peritumoral transition band, where
fidelity matters most. `vp` keeps the source variance constant (equal to the
data variance) for every `P` in `[0, 1]`, at the cost of no longer being a
literal convex combination of `z_prev` and `eps`.

Both blends agree **exactly** at the endpoints: `P = 0` gives noise
coefficient `1` under both (`1 - 0 = 1`, `sqrt(1 - 0^2) = 1`); `P = 1` gives
noise coefficient `0` under both. `blend` only changes behavior strictly
between the endpoints, and only the endpoints are exercised by the
pure-noise-equivalence and full-preservation tests below.

`1 - P**2` is clamped at `0` before the square root for numerical safety
(never an active branch, since every mode already produces `P` in `[0, 1]`).

## Design decision: `P = 0` in dense tumor at every horizon

`horizon_decay` deliberately gives up the naive boundary condition
`P(dt=0) = 1` inside dense tumor. Even at `dt = 0` — the previous scan taken
"now" — the tumor core still starts from pure noise. This is a **generative-
capacity allocation** decision, not a claim that tumor tissue is 0%
predictable at zero horizon: the model is given full freedom to regenerate
that region regardless of how recent the previous scan is, because tumor
morphology is exactly the thing the model exists to predict, and a nonzero
floor there would waste modeling capacity re-deriving the (irrelevant, since
non-tumor) previous image inside the lesion. This mirrors `hard_dilation`'s
`P = 0` inside the dilated tumor and is a stricter version of
`local_density`'s `p_min` floor (here, `p_min = 0` unconditionally).

This choice is harmless in practice: `dt = 0` never occurs in this repo's
data (every sample's horizon is a real gap between two distinct scan dates —
see `src/data/dataset.py`'s `target_day_rel = target_day_abs - last_input_day`,
which is `0` only if two visits share a day, filtered out by window
construction requiring `N >= 2` distinct sessions). The forgone boundary
condition is therefore never actually tested against real data.

## Where `dt` comes from

`dt` is the horizon **in days** from the most recent previous visit to the
target visit — independent of how many earlier visits are in the sliding
context window. This repo's dataset already computes exactly this quantity:
`src/data/dataset.py::PatientDataset.__getitem__` sets
`target_day_rel = target_day_abs - last_input_day`, where `last_input_day`
is `days_arr[input_indices[-1]]` — the last (most recent) visit in the
window, regardless of window length `k`. This becomes `batch["target_day"]`
after collation, and is the same value passed as `generate()`'s `target_day`
argument at inference. No new horizon-tracking logic was needed;
`horizon_decay` reuses this field directly, and `TaGeDiff._gather_last_valid`
(used for both `z_prev` and the baseline mask `M0`) already selects the same
most-recent-visit convention, so `dt`, `z_prev`, and `M0` are always mutually
consistent regardless of context length.

## What this mode does *not* do

- No trainable parameters, no calibration/NLL loss, no network — it is a
  closed-form function of `(M0, dt, gamma, lambda_bg)`, exactly like the
  other two modes are closed-form functions of `M0` alone.
- No treatment or genomics inputs.
- The `hard_dilation`/`local_density` modes and their `linear`-blend
  behavior are byte-identical to before this change (see
  `tests/test_flow_matching_region_source_strategies.py`'s backward-
  compatibility tests).
