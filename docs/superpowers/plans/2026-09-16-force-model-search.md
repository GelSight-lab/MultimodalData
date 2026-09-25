# Force Model Search Implementation Plan

> **For agentic workers:** Execute the tasks below in order and retain measured results.

**Goal:** Search for a force estimator with MAE <= 0.5 N on unseen press positions, without using test labels for model selection.

**Architecture:** Keep the deployed estimator intact while evaluating candidates in a separate reproducible experiment. Use existing reconstruction features, richer depth/contact descriptors, and reference-subtracted image descriptors. Fit all learned transforms inside training folds. Select by grouped training CV, evaluate the historical holdout once, and check generalization to other shapes before considering deployment.

**Tech Stack:** NumPy, SciPy, OpenCV, scikit-learn, existing React calibration pipeline.

## Tasks

- [x] Add tests for position-disjoint splits, exclusion of label/command metadata from features, and blank-frame handling.
- [x] Implement `twm/force_recovery/model_search.py` with deterministic feature extraction, bounded model search, grouped selection, JSON metrics and saved model.
- [x] Build feature caches from the original images with source filenames retained for auditing. Do not use filename metadata as inference features.
- [x] Compare linear, polynomial, kernel and tree models using four grouped folds on the 320 training samples. Score the chosen candidate on the 158 historical holdout samples only after selection.
- [x] Evaluate the selected method with nested position-grouped CV and on other labeled indenter families. Report force-band errors and position-bootstrap confidence intervals.
- [x] Save the candidate and report. Promote only if deployment compatibility and transfer evidence support it; otherwise retain as an experimental model and state the remaining gap. Candidates failed the PushT transfer check and remain experimental.

## Verification

Run `pytest tests/test_force_model_search.py tests/test_force_recalibration.py tests/test_force_frame_alignment.py tests/test_force_verify_stale.py -q`.
Reproduce the search with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m twm.force_recovery.model_search`.
Check that reports distinguish calibration MAE from unknown PushT absolute-force error.
