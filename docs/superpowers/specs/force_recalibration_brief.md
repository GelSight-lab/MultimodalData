# Force estimation: what a 15 N recalibration should collect, and why

Update, 2026-09-16: the subsequent [model search](2026-09-16-force-model-search-results.md)
reached 0.305 N on the round holdout and 0.288 N on held-out positions across
six trained shapes using image/depth features and ridge regression. These
replace the earlier in-distribution expectations below. Direct PushT transfer
failed the output-distribution checks; actual React sensor calibration remains
necessary to establish its absolute force accuracy. The analysis below records
the earlier five-feature estimator.

Written 2026-09-16 for whoever runs the recalibration. Every number below was
measured on the existing calibration cache
(`force_recovery/feature_cache/glowtact_round_mm.json`, 477 presses), not
taken from a document.

## The target, restated

The ask was MAE 0.2–0.3 N with some extrapolation. Measured on its own
calibration objects, held out by press position, the production estimator is:

```
MAE 1.291 N      median relative error 32.8%      rho 0.720
```

0.2–0.3 N is roughly a quarter of that. Whether it is reachable depends on
where the error comes from, so that was measured first.

## Where the error is, measured

**It is multiplicative, not additive.** Error grows with force while relative
error stays flat:

| true force | n | MAE | relative |
|---|---|---|---|
| 0–1 N | 27 | 0.984 | 197 % |
| 1–2 N | 44 | 1.269 | 85 % |
| 2–3 N | 25 | 0.974 | 39 % |
| 3–4 N | 27 | 1.009 | 29 % |
| 4–5 N | 30 | 1.336 | 30 % |
| 5–6 N | 22 | 1.379 | 25 % |
| 6–7 N | 30 | 1.551 | 24 % |
| 7–8 N | 31 | 1.735 | 23 % |

A fixed 0.2–0.3 N is therefore the wrong shape of target: at 7 N it demands
4 % relative, at 0.5 N it is looser than the sensor's own repeatability.
**State the goal as a relative error** — 5 % would be excellent, 10 %
plausible — or as absolute error within a named force band.

## The three candidate causes, and what the data says

**1. Range. NOT the bottleneck.** The existing 477 presses are spread evenly
over 0.16–7.99 N (47–76 samples per 1 N bin) across 166 distinct press
positions. The error is produced in bins that are well populated. Extending to
15 N will remove the 7.87 N isotonic clip — 2.22 % of published samples sit
exactly on it — and that is worth doing, but it will not by itself move MAE in
the range already covered.

**2. Fitting. NOT the bottleneck.** Refitting in the log domain, which is the
matched loss for multiplicative noise, changes nothing:

```
production (5 features, linear)   MAE 1.291 N   rel 32.8 %   rho 0.720
same features, log domain         MAE 1.267 N   rel 33.1 %   rho 0.751
three features, log domain        MAE 1.354 N   rel 36.1 %   rho 0.680
```

**3. The features. THIS is the bottleneck.** No single feature correlates
above rho 0.72 with force:

```
vol 0.695   vol2 0.710   maxd 0.717   area 0.616   sqrt(area)*maxd 0.708
```

And the features are noisy at the source. On the two groups with ≥3 repeats at
the same position and the same force (±0.25 N), the within-group coefficient of
variation of `vol` is **13.7 %**. If force is roughly proportional to `vol`,
that is a ~14 % floor on relative error before any model is fitted — half of
the 32.8 % currently observed, from feature noise alone.

Only 2 such repeat groups exist, so 13.7 % is an estimate from very little
data. **Measuring this floor properly is the single most valuable thing the
recalibration can do**, because it decides whether 5 % is reachable at all.

## What to collect

1. **Repeats, above all.** At each of ~30 press positions, press to the same
   target force **5–10 times**. This is what turns the noise floor from an
   estimate into a measurement, and no amount of new single presses substitutes
   for it. The current set has 477 presses and effectively 2 repeat groups.

2. **Range to 15 N**, with the same even spread: ~40 presses per 1 N bin. Above
   8 N the gel may saturate geometrically (`maxd` stops growing); if it does,
   that is a finding worth recording, not a failure.

3. **Position coverage held.** 166 positions is good; keep ≥100, and keep the
   held-out-by-position protocol — a random split leaks, because presses at the
   same spot share their geometry.

4. **Record what the current cache omits**: gel temperature or session index,
   time since the sensor was seated, and which physical sensor. Drift between
   sessions is currently invisible and would show up as irreducible error.

5. **Extrapolation needs a second sensor.** `cross_dataset.py` states the
   finding plainly: ordering (rho) transfers across sensors, absolute newtons
   do not, and that is what per-sensor calibration means. If absolute-scale
   transfer is a requirement, the recalibration must cover **at least two
   physical sensors** so the between-sensor term can be separated from the
   within-sensor one. With one sensor it cannot be measured at all.

## What to expect

- Removing the 7.87 N clip: fixes the 2.22 % of samples that are floors rather
  than measurements. Real, bounded, not an MAE improvement in 0–8 N.
- A proper noise-floor measurement: tells you the achievable target. If the
  floor is ~14 %, then 32.8 % leaves roughly a factor of two to modelling, and
  5 % is out of reach without better features.
- Better features — more of the depth map than five scalars, or a small CNN on
  the reconstruction — is the only route to a large gain, and it needs the
  repeat data to be trainable and checkable.

## Protocol notes for whoever scores it

- Hold out **by press position**, never at random.
- Score rho on the **linear projection**, not the isotonic output: isotonic is
  fitted with `out_of_bounds="clip"`, so a target projecting outside the fitted
  range comes back constant and rho is nan.
- Compare arrays with `equal_nan=True`. An all-NaN column read as "different"
  once already made a byte-identical A/B look like a regression.
- Report relative error alongside MAE. MAE alone hides that the estimator is
  four times worse at 7 N than at 1 N in absolute terms, and equally good in
  relative ones.
