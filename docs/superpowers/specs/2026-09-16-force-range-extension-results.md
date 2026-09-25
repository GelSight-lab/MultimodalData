# Force Calibration v8: 0-15 N Extension

The user approved v7's contact sensitivity and requested a higher range because
predictions saturated. The previous isotonic endpoint was 7.79 N: raising an
output clipping constant alone would not change that endpoint.

## Method

- Preserve the exact v7 478-row low-force cache, gain field, linear score,
  isotonic knots, dI=4 contact mask and reference-noise contact weighting.
- Reconstruct 399 additional round calibration images with measured forces
  in (8, 15] N using the same `force_stages` and `feature_vector` as inference.
- Preserve all original held-out positions; hold out one third of previously
  unseen high-range positions. This leaves 274 high-load training samples and
  125 held-out samples. Of the training samples, 141 score above the old join
  and fit the tail. Samples below the join are not silently removed from tests.
- Fit an isotonic high-score tail, anchor it at the original final knot, and
  retain every original low-score knot. Interpolate continuously at the join;
  clip outside fitted support. No invented endpoint or extrapolation is used.
- The endpoint is now 14.99 N, supported by a measured training label. This is
  a bounded output range, not an OOD guarantee or proof of React force accuracy.

The original calibration cache SHA256 matches the v7 published verification
artifact. `fit(extend_range=False)` reproduces v7. Pipeline version is now 8;
historical reconstruction comparisons explicitly disable the range extension.

## Paired Calibration Results

Same held-out examples for both columns, split by press position:

| True force | n | v7 MAE (N) | v8 MAE (N) |
|---|---:|---:|---:|
| <1 N | 14 | 0.436 | 0.436 |
| 1-4 N | 67 | 1.083 | 1.083 |
| 4-8 N | 77 | 1.120 | 1.120 |
| 8-12 N | 69 | 3.613 | 2.791 |
| 12-15 N | 56 | 6.613 | 3.600 |
| All <=8 N | 158 | 1.044 | 1.044 |
| All >8 N | 125 | 4.957 | 3.153 |
| All | 283 | 2.772 | 1.975 |

All 158 original held-out outputs are exactly unchanged. However, 57 of 125
high-load test samples score below the old join and cannot benefit from this
extension. That limitation is retained rather than changing the user's approved
low-force mapping. This is not a 0.5 N-accuracy model. There are no exact zero
force labels; the near-zero contact gate remains an engineering rule.

An exploratory full-range refit changed the low-force mapping substantially
(<1 N MAE 1.100 N versus 0.436 N for the old fit, different position population).
It was not selected. The paired table above evaluates the anchored extension;
the exploratory result is not a paired improvement claim.

## Review Artifacts

[HF review page](https://yxma-react-force-recovery.static.hf.space/task-review-2026-09-16-range15/index.html)
and [dataset artifacts](https://huggingface.co/datasets/yxma/React/tree/main/data/force_calibration_review/2026-09-16-range15).
The original v7 page remains available from the new page.

Review outputs are isolated under
`/media/yxma/Disk1/twm/force_recovery/task_review_2026-09-16_range15`.
The same seven 30-second windows cover PushT, motherboard, rope and toy.
Contact masks, reference images and source-frame maps are checked against v7;
the plots and force-disc legend use 0-15 N. Historical v7 files are retained.

The video curves are dI8 historical baseline, v7 (8 N), and v8 (15 N).
Force weighting is applied after calibration: a tiny weighted v7 value can
originate from a saturated pre-weight score and therefore change in v8.
The low-curve preservation check accounts for this weighting. A thresholded
nonzero-force statistic is not a ground-truth contact recall statistic.

No full-episode force files or published release columns are overwritten.
React lacks measured force labels, so calibration MAEs are not React MAEs.

Across 7,282 fresh sensor frames in the selected windows:

| Task | Fresh frames | v7 ceiling frames | v8 ceiling frames | v8 peak (N) |
|---|---:|---:|---:|---:|
| PushT | 4,126 | 160 | 0 | 12.978 |
| Motherboard | 1,023 | 0 | 0 | 7.750 |
| Rope | 1,081 | 0 | 0 | 5.404 |
| Toy | 1,052 | 9 | 0 | 11.922 |

All 169 previously capped fresh frames now exceed 8 N. Every light-contact
proxy frame (intensity in [3,6)) retains its v7 nonzero decision. Two PushT
frames below intensity 3 cross the reporting threshold of 0.02 N, each changing
by less than 0.013 N; the contact mask and gate themselves are unchanged.
All checked positive-weight, below-join predictions remain exactly unchanged.

The uniform full-episode audit adds 1,344 sampled fresh sensor rows (some overlap
the selected windows). Its 30 old ceiling cases also no longer hit the new
ceiling; the highest audited prediction is 13.812 N on PushT. These are
behavior checks on seven episodes, not a whole-dataset performance guarantee.

The focused force/preview suite passes 75 tests. A 61-row production-writer
smoke test passes finite bounds, duplicate-frame holds and v8/15 N metadata
checks in an isolated output directory. A separate read of 56 named raw sensor
frames reproduces the cached predictions within 1e-8 N. Historical feature
search and reconstruction A/B scripts explicitly retain their <=8 N range.

Publication verification fully decodes all seven H.264/yuv420p videos: each
has 900 frames at 30 fps and a 30-second duration. Six are 1280x1004; toy is
1280x1024 because its unchanged missing-coordinate-frame warning needs another
line. Toy virtual targets remain omitted instead of guessing a pose frame.
Remote Playwright checks pass playback and seeking to 15 seconds for all seven
videos, CSV/JSON links, the disclosure control, and desktop/mobile layout without
page overflow or script errors. Screenshots and browser results are retained in
the local review directory. Formal release force columns remain untouched.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.react_calib build-tail
OPENBLAS_NUM_THREADS=1 python -m twm.force_recovery.react_calib range-report
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review render
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review publish
```

`calibration.json` includes paired predictions, force-band counts, split
description and calibration-cache hashes. `verification.json` records actual
named-frame recomputations and full video decode checks when publishing.
