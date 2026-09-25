# Equivalent CPU Force Optimization

> **For agentic workers:** Use executing-plans task by task; request an independent review before handoff.

**Goal:** Reduce v8 CPU inference time without changing calibration, contact
sensitivity, alignment, geometry fields or the NPZ contract. User approved
this design on 2026-09-16. Acceptance: maximum force difference <=1e-6 N,
unchanged contact decisions, and unchanged geometry within numerical tolerance.

**Architecture:** Keep the existing NumPy/SciPy solvers. Cache only bounded,
read-only shape-dependent arrays, reuse contact masks, and make unused normal
maps optional. Skip gradient reconstruction only for an exactly empty contact
mask, not for a task-dependent or force-based threshold. Preserve full outputs.

**Tech Stack:** Python, NumPy, SciPy, OpenCV, h5py, pytest.

## Baseline

An untouched copy of the current dirty v8 package is retained at
`/tmp/react-force-speed-before-2uWyFb/force_recovery`. This includes earlier
uncommitted calibration work; git HEAD is not the correct baseline.
Initial profile: 128 fresh frames across four tasks/both sides, about 99 ms
per frame including H5 reads. Reconstruction accounted for about 77%.
Capture and preprocessing are active on this host: use interleaved repeated
comparisons and report medians, not a claim of globally optimal throughput.

## Tasks

- [x] Add `tests/test_force_speed.py` with equivalence and operation-count
  tests. Cover empty/weak/strong/edge contact, marker and markerless references,
  caller-provided masks, shape-cache immutability, and bounded cache size.
  Run `OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_force_speed.py -q`
  and observe failures before adding implementation.
- [x] In `poisson.py`, cache the unchanged Neumann denominator and full trend
  basis with `lru_cache(maxsize=4)`; set cached arrays read-only. Leave the
  contact-dependent least-squares fit and external Dirichlet solver unchanged.
- [x] In `calib_free.py`, reuse the already computed mask in `gradients`;
  add `include_normals=True` to preserve the diagnostic API. In
  `react_calib.force_stages`, request no normals. Replace RGB maximum reductions
  with an equivalent channel-wise maximum in both CF and LUT paths. Skip
  gradient and integration work only where an empty mask proves zero depth;
  retain the default diagnostic solver labels and all output fields.
- [x] Verify synthetic outputs against the frozen implementation. Add a
  reusable `benchmark_speed.py` CLI comparing both packages on four-task raw
  frames, with identical references/noise and alternating timing order. Include
  every output field in the equivalence check, and distinguish in-memory
  compute throughput from H5-inclusive timing. Never write production NPZs.
- [x] Compare a complete episode writer output from both implementations in
  isolated temporary destinations. Retain version 8 only if results meet the
  equivalence criteria. Run the force/preview/pipeline regression suites and
  the geometry unit tests, then request a read-only review.
- [x] Record measured results and limitations in a results spec and update
  `RUNBOOK.md` with commands, unchanged output semantics, worker/thread advice
  and the requirement to use the new code snapshot before launching workers.

No GPU port, downsampling, reduced precision, force-only schema, calibration
refit, stiffness change, full dataset reprocessing or publication is included.
Use the existing dirty worktree to preserve the approved v8 changes; do not
revert or commit unrelated work. The frozen package is the numerical oracle.

Completed: 184 paired four-task frames, 3886 full-writer output rows, 157
regression tests and the analytic Poisson edge checks passed. Force and geometry
differences were zero; all four full NPZs were array-equal in every field.
Repeated warm-H5-inclusive speedup was 1.77x. The permanent baseline and reports
are in `/media/yxma/Disk1/twm/force_recovery/speed_review_2026-09-16/`.
