# Four-task force calibration review, v7

## Scope and status

This is a review candidate, not a replacement of published React force data.
It improves the contact gate and fixes validation/reference defects. It does
not establish absolute force accuracy on React or solve cross-sensor OOD.

Review page:
https://yxma-react-force-recovery.static.hf.space/task-review-2026-09-16/index.html

Artifacts:
https://huggingface.co/datasets/yxma/React/tree/main/data/force_calibration_review/2026-09-16

Local artifacts are under
`/media/yxma/Disk1/twm/force_recovery/task_review_2026-09-16/`.
`plan.json` fixes the seven source windows. `cache/` contains the reconstructed
frames and estimates. `preview_forces/` contains explicitly review-only,
partially populated NPZs; these must not be promoted as full episodes.

## Algorithm changes

- Spatial gain fitting now uses training positions only. The previous gain fit
  included commanded depths from held-out positions. `legacy=True` is retained
  only to reproduce historical models for comparison.
- The hard 1 mm2 force cutoff becomes a continuous contact-evidence ramp. Its
  lower endpoint is the larger of 30 pixels' area and the episode reference
  noise area; its upper endpoint is `max(1 mm2, lower + 0.5 mm2)`.
- Reference noise is the 90th percentile of leave-one-out dI4 residual areas
  among the twelve reference frames. This assumes those frames are unloaded;
  it is not a learned contact probability or a known false-positive rate.
- Force estimation and the preview now select reference frames through the
  authoritative capture-time index map, rather than reconstructing a constant
  lag in the preview. Repeated tactile captures retain the same estimate.
- Nonfinite force features are rejected; zero positive contact depth returns
  zero without dividing by a zero centroid weight. Metadata records the noise
  floor, calibration ceiling and lack of React absolute-force validation.
- The renderer accepts an isolated force root and a diagnostic frame callback.
  Review videos retain its camera, wrist, OptiTrack and tactile layout, adding
  contact masks, arbitrary-unit reconstruction depth and three force traces.

Toy's uncurated release parquet has no declared pose coordinate frame and no
`episodes.jsonl`. Its preview therefore explicitly omits the DexForce virtual
target instead of guessing a coordinate transform. Raw OptiTrack poses,
camera projection, force overlays and tactile reconstruction remain present.
The omission is labelled in the video and webpage; release metadata is untouched.

The v7 force map still uses the 0-8 N round calibration. It has a fitted
ceiling of 7.79 N. It is deliberately not relabelled as a validated 0-15 N
model. The attempted full-range nonnegative geometry alternatives had roughly
2.3-2.6 N position-holdout MAE and were not promoted.

## Measured results

The corrected round-calibration position holdout has **MAE 1.043656 N** on
158 frames (320 training frames). The historical dI8 fit was 1.072246 N;
the dI4-only v6 fit was 1.037 N but retained the gain-fit leakage. These are
calibration-domain results, not React MAE. No substantial accuracy improvement
or 0.5 N target achievement is claimed.

Seven 30-second windows cover four PushT episodes and one episode each of
motherboard, rope and toy, both sensors. Selection maximizes weak-intensity
evidence and transitions without consulting force predictions; curated bad
intervals are avoided. This is targeted inspection, not a representative
random sample of all four tasks.

There are 12,600 sensor timeline rows, including 7,282 rows marked as newly
captured tactile frames. Another 96 fresh rows per sensor/episode are sampled
uniformly over each full episode (1,344 audit rows). These audit rows can
overlap the selected windows and are not independent additional episodes.

The table reports nonzero-force fractions (>0.02 N) among fresh frames with
published intensity in [3, 6). This is a weak-contact **proxy**, not contact
recall: visually unloaded references can themselves have intensity above 3.

| Task | Fresh window rows | Proxy rows | dI8 baseline | dI4 v6 | v7 candidate |
|---|---:|---:|---:|---:|---:|
| PushT | 4126 | 2300 | 46.61% | 84.96% | 76.78% |
| Motherboard | 1023 | 849 | 42.76% | 45.82% | 46.05% |
| Rope | 1081 | 1075 | 27.53% | 40.65% | 29.49% |
| Toy | 1052 | 908 | 29.41% | 37.78% | 62.78% |

For PushT's right sensor, the proxy fraction is 70.57% / 88.39% / 99.60%
(999 frames). On the left it is 28.21% / 82.32% / 59.26% (1301 frames).
Reference-noise gating suppresses many left-side detections; counting every
nonzero value as a successful contact would reward the noisy v6 gate.

All 1,124 fresh strong-proxy frames (intensity >=6) retain nonzero estimates.
There are zero duplicate-frame force mismatches. Estimates are finite and
range from 0 to 7.79 N in these windows. 160 PushT frames and 9 Toy frames
reach the fitted ceiling, so their true force cannot be resolved by this map.

Both historical baselines are recomputed using the same aligned raw images
and reference as v7. They are not copies of the existing published NPZs.

## Remaining failures and uncertainty

Three additional low-intensity frames per sensor/episode are excluded from
the reference image and noise fitting. Of these 42 probes, 10 still exceed
0.1 N. Their largest estimate per task is:

| Task | Probes above 0.1 N | Maximum estimate |
|---|---:|---:|
| PushT | 6 / 24 | 1.915 N |
| Motherboard | 1 / 6 | 0.132 N |
| Rope | 2 / 6 | 2.325 N |
| Toy | 1 / 6 | 2.064 N |

These probes are **not ground-truth zero-force samples**. The residuals could
include real light contact, reference drift or edge-reconstruction artifacts.
They prevent a claim that baseline force is reliably zero on all four tasks.

Contacts near the crop boundary can amplify weak image changes and produce
large depth/force estimates. The webpage reports their frequency and retains
the raw tactile image beside the reconstruction. Depth is labelled in arbitrary
units; there is no metric-depth ground truth or measured React force MAE.

A displayed zero below the estimated noise floor means an unresolved signal,
not a measured zero force. Bounding a model or suppressing weak signals is not
an OOD guarantee. The 8-15 N interval remains unvalidated, not silently clipped
and presented as solved. Actual sensor-specific loaded/unloaded calibration
and manual contact review are still needed before promoting new force labels.

## Reproduction

Run from the repository root with the existing calibration assets available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review render
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.task_review publish
```

Rendering is resumable per clip. Cache writes are atomic. The public CSVs
contain row indices, actual tactile source-frame indices and all three force
traces. `metrics.json` includes both window and full-episode audit summaries;
`verification.json` records raw-frame reproduction checks and source hashes.

The focused force/preview suite passes 68 tests. A 61-row production-writer
smoke test using real PushT images also passes, including finite estimates,
v7 metadata and repeated-frame invariants. These checks establish software
consistency, not force accuracy.

Publication verification completed: all seven MP4s decode through all 900
frames without FFmpeg errors. Re-reading 56 named raw tactile frames reproduces
the cached force and the preview NPZ values within 1e-6 N. All seven videos
and their CSVs are uploaded; the Space page uses the verified static-Space host.
