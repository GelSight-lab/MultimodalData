# TWM Maintenance Completion Plan

> Use subagent-driven-development for bounded implementation and independent
> spec/quality reviews. Preserve the dirty main checkout; work on
> `refactor/twm-visualization` in `.worktrees/twm-visualization`.

**Goal:** Finish the code cleanup/debugging and documentation, including all
eight findings in `docs/twm-maintenance-audit.md`, the baseline failures, and
test reliability. Do not redefine completion as only the visualization phase.

**Architecture:** Keep source alignment, calibration authority and dataset schemas
unchanged. Fix lifecycle boundaries in their owning modules. Share pure rendering
and encoding where dependency boundaries permit; the published toolbox must still
work without the full recorder or model stack.

**Tech stack:** Python, pytest, NumPy, OpenCV/AV/FFmpeg, HDF5 and Parquet.

## 1. Recorder lifecycle safety

Files: `twm/recorder/{writer,capture}.py`, corresponding `tests/recorder` files.
- [x] Reproduce drain returning during flush with a threading.Event-controlled
  fake file; verify drain remains pending until flush is released.
- [x] Keep `_in_flight` nonzero across `_maybe_flush(f)` outside `_cv`; update
  counters and notify only after all file access for the batch finishes.
- [x] Reproduce persistent `writer.stats()` failure both before and after first
  snapshot. Fatal reporting must publish the original failure and set stop.
- [x] Use the previous stats/sensor snapshot or explicit initial fallback; do not
  call the failing stats provider again in `_publish_fatal`.
- [x] Run recorder regressions and independent spec then quality review.

## 2. Baseline checks and deterministic tests

Files: `twm/scripts/dataset_stats.py`, `twm/force_recovery/run_episode_mp4.py`,
`twm/calib_epoch.py`, relevant tests.
- [x] Reproduce the five known failures. Preserve/reuse the user's existing
  force-free statistics behavior without changing main checkout files.
- [x] Test empty tables, absent or partially present force columns and mixed
  coverage. Missing measurements must not become measured zero force.
- [x] Trace the task-list guard: distinguish a historical recovery scope from
  pipeline task defaults; use the authoritative task policy for defaults.
- [x] Make calibration tests follow the existing explicit-session/standing-epoch
  resolver policy; do not invent new extrinsics to make the test pass.
- [x] Separate scheduling-dependent capture checks from deterministic synthetic
  headless integration assertions without weakening production validation.

## 3. Preprocessing completion and encoding

Files: `twm/react_preprocess/{pipeline,complete,encode}.py`,
`twm/pipeline_stages.py`, preprocessing tests.
- [x] Add interrupted-sidecar-write and optional-wrist regression fixtures.
- [x] Replace parquet-only skip with complete required-artifact validation tied
  to the actual source modalities; retry incomplete outputs.
- [x] Reproduce encoder BrokenPipe and invalid frames; guarantee child reaping
  and original exception preservation through every close/failure path.
- [x] Verify successful normal preprocessing remains byte/schema compatible.

## 4. Toolbox readers and import boundaries

Files: `twm/react_toolbox/{io,__init__}.py`, toolbox tests.
- [x] Test empty, duplicate, unordered, missing and out-of-range frame requests
  through both decoder paths; define consistent explicit failures and cleanup.
- [x] Cache requested bounds outside loops and decode monotonically where
  possible. Preserve the existing sorted-unique request contract.
- [x] Verify independent toolbox imports with no parent TWM/model dependency.

## 5. Remaining visualization consolidation

Files: `twm/force_recovery/visualize.py`, pure helpers and visualization tests.
- [x] Characterize timestamped vs legacy tactile alignment using existing
  adapters; reproduce incorrect fixed shift before changing it.
- [x] Move pure image helpers out of heavyweight inference imports and route
  shared composition/export through the visualization module where applicable.
- [x] Replace history redraw work proportional to total history at every frame
  with bounded/incremental presentation; compare reference pixels where stable.
- [x] Remove redundant encoding and test early-exit resource cleanup.

## 6. Installed package and documentation

Files: `pyproject.toml`, packaging tests, `twm/README.md`, visualization guide,
`docs/twm-maintenance-audit.md`.
- [x] Build a wheel and reproduce missing subpackage imports outside checkout.
- [x] Correct package discovery/data inclusion; smoke-test installed CLI and
  pure modules without hardware imports or accidental checkout fallback.
- [x] Review remaining TWM test/verifier entrypoints, document optional hardware,
  datasets and research dependencies rather than silently ignoring them.
- [x] Run full applicable suite serially, pipeline guard, installed-wheel smoke,
  and benchmarks. Independently review completed changes.
- [x] Update audit items with concrete resolution/evidence and runnable commands.
  Mark goal complete only when all requirements are verified or actual external
  blockers are explicitly established; do not claim hardware tests were run.

For each bug: read the call path, write/reproduce RED, implement the smallest
root-cause fix, run GREEN plus affected regressions, review and commit scoped files.

## Final verification findings

The first full serial run finished with 1,195 passed, 3 failed and 1 skipped.
Independent quality review found one additional encoding compatibility regression.
These findings must be resolved before the final verification items above close:

- [x] Correct force-export provenance and operational documentation to match
  the existing sensor −Y normal, shared 2 N/mm controller stiffness, and virtual
  (not gel-compression) displacement. Test schema and sidecar declarations.
- [x] Preserve the force website clips' existing `yuv420p` output contract while
  retaining the batch preview writer's default; verify actual encoded format.
- [x] Fix source-frame dimensions at preprocessing encoder boundaries: the
  synthetic end-to-end chain has 48×64 images but writers declare 480×640.
  Retain strict block validation and verify output dimensions and frame counts.
- [x] Make the force-asset presence test establish its build prerequisite in
  an isolated fixture, rather than depend on this host's production release.
- [x] Repeat the complete serial suite and independent review after these fixes.

Final serial result on code commit `f6c3c29` (documentation-only follow-up
`078cc11`): **1,215 passed, 1 skipped, 11 warnings in 474.28 s**, exit 0.
The single skip is the opt-in real-time scheduling test, not a failed correctness
check. Independent final spec/quality review approved the fixes with no remaining
Critical/Important findings. Pipeline guard: **15 checks, 0 violations**;
fresh installed-wheel smoke: **1 passed in 2.78 s**. See the maintenance audit
for benchmark results, requirement coverage and operational exclusions.
