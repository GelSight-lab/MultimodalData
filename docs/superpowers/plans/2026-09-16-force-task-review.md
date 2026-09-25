# Force calibration task review

Implement the approved four-task review without overwriting published forces.

1. Fix held-out-position leakage in the spatial gain fit. Preserve an explicit
   legacy mode for reproducible comparisons, not production inference.
2. Replace the 1 mm2 hard contact cutoff with a continuous ramp from the
   minimum connected evidence. Allow episode reference noise to raise the
   floor. Keep the existing calibrated force range; do not relabel it 0-15 N.
3. Share timestamp-aligned reference frame selection with the preview renderer.
   Test low-contact continuity, invalid inputs, split isolation and references.
4. Evaluate a fixed set of seven 30-second windows (four PushT, one each
   motherboard/rope/toy), both sensors. Include untouched reference checks and
   sampled frames from additional episodes where practical. Contact-intensity
   bins are proxies, not ground truth or force-error measurements.
5. Reuse the canonical preview panel and projection renderer; append synchronized
   before/after traces and reconstruction tiles. Encode browser-compatible H.264.
6. Publish isolated review assets and metrics to yxma/React and a new page under
   the existing yxma/react-force-recovery static Space. Decode every video,
   inspect representative frames and verify remote assets before reporting.

Acceptance: finite bounded diagnostic estimates, reference-noise checks,
correct source-frame alignment, unchanged estimates on duplicate captures,
and reviewable evidence of limitations. No claim of OOD immunity, 0.5 N MAE
on React, validated 8-15 N forces, or exhaustive four-task validation.
