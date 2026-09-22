# Bounded Force Evaluation Plan

**Goal:** Evaluate 0-15 N estimators without treating bounded outputs or an OOD score as proof of transfer accuracy.

**Design:** Extend the six-shape feature cache using existing 8-15 N and low-force images. Compare nonnegative geometry regression, nearest-neighbor interpolation, and extra trees. Select using held-out shapes, not only held-out positions. Keep separate fitting, guard-calibration and test positions. Calibrate a nearest-neighbor support threshold and empirical error band only on calibration data. Unregistered sensor domains and unsupported inputs return NaN with an explicit reason.

**Files:** `bounded_force.py` owns the guarded estimator; `bounded_force_eval.py` owns data collection, selection and reports; `tests/test_bounded_force.py` checks range, invalid labels, unsupported inputs, and domain rejection.

- [x] Test and implement the inference contract. No negative or >15 N accepted estimates; rejected frames must not become zero-force labels.
- [x] Build a separate 0-15 N cache, preserving the previous 0-8 N cache and production pipeline.
- [x] Compare methods with shape-grouped selection and a held-out-position test. Report both unconditional error and accepted error/coverage.
- [x] Run nested leave-one-shape-out evaluation, then unlabeled PushT diagnostics. Domain rejection is a deployment restriction, not evidence of accurate PushT force estimation.
- [x] Save the experimental model, results and limitations. Do not promote without target-sensor force labels.

Bounded isotonic calibration of rich ridge scores was added after the initial
interpolation experiment. Its position-test diagnostics were not used for
selection. Both experiments are retained, and the selected model remains
experimental because transfer and the 0.5 N target are not solved.

Physical regression uses nonnegative coefficients on reconstructed deformation features. These have an arbitrary depth scale, so this is a physics-inspired baseline, not a calibrated finite-element force solver.
