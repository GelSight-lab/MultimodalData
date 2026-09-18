# TWM maintenance audit — 2026-09-18

## Scope and visualization module

Inventory at base `039f6af`: 322 Python files / 77,509 lines in `twm`, with
148 test files under `tests` and 47 embedded tests/verifiers under `twm`.
This is a repository-wide structural inventory plus targeted source review,
not a claim that every historical research script received line-by-line review.

Reviewed the canonical renderer, viewer source adapters, recorder preview,
batch episode previews, capture/writer lifecycle, preprocessing pipeline/encoding/
tactile/contact, toolbox I/O, and selected force visualization/controller paths.

Delivered:

- `twm.visualization`: explicit RGB/BGR/depth/signed-height tile composition,
  ordered source-local overlays, missing modalities, layout metadata, and a
  compatibility adapter around existing projection geometry.
- Recorder, viewer, batch preview integrations; unified module CLI retaining
  existing viewer arguments. Standalone published toolbox remains independent.
- Batch preview generator and atomic verified streaming MP4 writer. No retained
  list of every full-resolution panel. Wrist slots/tone configuration resolved
  once per source instead of once per frame.
- Fixed viewer `r` reference-reset crash from stale pre-adapter variable names;
  now uses copies of the currently aligned tactile frames.
- Removed import-time `sys.modules` replacement and search-path mutation from
  viewer tests; parametrized alignment cases without dropping coverage. Recorder
  GUI tests now patch only GUI functions, avoiding module poisoning on later imports.
- Fixed batch labels for single-wrist rigs and tested HDF5 iteration with zero,
  left-only, right-only and two wrist streams.
- Removed this batch script's hard-coded checkout import path.

The memory improvement is algorithmic: for example, retaining 900 panels of
1280×528×3 bytes previously required 1.82 GB just for that list's pixels. The new
writer consumes frames incrementally; it does not promise constant total process
RSS (source arrays, models, and encoder buffers still exist). Preview rasterization
itself uses the same geometry and has similar runtime, not a claimed major speedup.

## Maintenance follow-up

The first audit's eight findings are tracked below. Changes are limited to code,
tests and documentation in the isolated worktree; main-checkout edits are preserved.

1. **Writer drain/flush race:** `_in_flight` and queued-byte accounting remain
   active until flush and file-size sampling finish. File I/O stays outside the
   condition lock. Event-controlled tests prove drain cannot return early.
2. **Recursive fatal reporting:** capture reuses the last writer/sensor snapshot
   or an explicitly unavailable initial snapshot instead of calling failing
   telemetry again. Controller finalization also tolerates persistent stats
   failures, marks the episode invalid, and closes its HDF5 handle.
3. **Preprocessing completion:** `react_preprocess.complete.is_complete()` checks
   the detection sidecar and requested videos/depth, using the actual source
   camera modalities. Without a source it retains strict seven-stream validation.
   A rebuild invalidates the old completion parquet first; sidecars finish before
   a temporary parquet is atomically published. Scheduler coverage and build
   prerequisites use the same checks, with its existing RGB-on/depth-off policy.
   Older outputs whose sidecars are newer than the parquet may be
   rebuilt once; metadata-only output cannot satisfy a later video/depth request.
4. **Frame-reader failures:** both AV and OpenCV return sorted unique requested
   indices as RGB, reject invalid indices, raise for unavailable frames, and close
   on failures. Empty selection returns `(0, 0, 0, 3)` without opening a decoder.
   Bounds are computed once and decoding proceeds monotonically. No fabricated
   black frames or silent dropped requests.
5. **Preprocess encoder cleanup:** frame block shape/dtype is validated before
   writing; stdin close failures still trigger process wait. Ordinary close/wait
   failures cannot replace an active body exception; cleanup-only failures carry
   output-path context and a chained cause.
6. **Force visualization:** canonical HDF5 alignment replaces unconditional
   legacy shifts. Clips use shared composition/export, a precomputed timeline
   and replayable reader lifetime. Estimator and force-target generation are
   outside this change.
7. **Import boundaries:** toolbox Arrow/video imports are lazy. Pure force image
   helpers move to `twm.visualization.force` with compatibility exports from
   `force_recovery.visualize`. The independently shipped toolbox has no dependency
   on the parent TWM package.
8. **Installed wheel:** package discovery includes subpackages and namespace
   command directories, plus calibration/config data. The smoke test extracts a
   freshly built wheel, runs imports/CLI help outside the checkout, and checks
   hardware drivers are not imported. This does not test a fresh full hardware
   installation; base robot dependencies still exist in project metadata.

### Test cleanup and baseline failures

- Statistics now retain scale/tactile information when force is absent, partial
  or nonfinite, while explicitly withholding force-derived summaries. Empty,
  zero-contact and fully saturated distributions render without invented values.
  Relevant force-free behavior from the user's dirty main checkout was preserved
  and extended on this branch without modifying that checkout.
- The task-list guard distinguishes historical recovery scopes from pipeline
  defaults. Calibration-session tests use the existing explicit-session/standing
  current-epoch policy; no extrinsics or resolver policy were changed.
- The host-scheduling-sensitive recorder test is marked `timing` and requires
  `TWM_TIMING_TESTS=1`. Deterministic headless content/schema/lifecycle tests and
  synthetic timing-threshold tests remain enabled. Production limits are unchanged.
- Imported `testset_root` helpers are no longer mistaken for tests. Historical
  truncation checks returning integer error counts are explicit command helpers,
  not falsely passing pytest tests; synthetic geometry has asserting regressions.

## Reproduce verification

From the worktree or checkout, with its analysis/test dependencies and FFmpeg:

```bash
python -m pytest -q
python -m twm.pipeline_guard
python -m pytest tests/test_installed_package.py -q
python -m twm.scripts.benchmark_visualization --iterations 100
TWM_TIMING_TESTS=1 python -m pytest tests/recorder/test_headless.py -m timing -q
```

Run the last command alone on an idle host. `pytest` collects `tests` and `twm`
using importlib mode to avoid duplicate-basename collisions. It does not run
every historical script's `main()` or every physical device integration.

## Verification notes

The baseline suite (`tests`, before implementation, maxfail=5) produced 919 passes
and five failures: an on-disk undeclared `pushT/2026-09-17` calibration session,
the task-list duplication guard, and three force-free dataset-statistics tests.
Unrelated user edits in the main checkout were not copied over or overwritten.
The results below describe the initial visualization phase; the maintenance
follow-up addresses those baseline failures and is being verified separately.
Initial refactor verification:

- Final focused integration selection: **102 passed**, including the four
  optional-wrist HDF5 cases added during review.
- Recorder directory: **152 passed**. Calibration/Z-up/force/alignment selection:
  **64 passed** (these selections overlap, so do not add their counts).
- Broader `tests` directory run: **1,075 passed, 6 failed**, 12 warnings in 438 s.
  It collected before the four optional-wrist tests were added. Five failures
  match the baseline and were independently reproduced. The sixth was the
  headless recording's real-time validator reporting two late ticks under load;
  the recorder-directory rerun passed. No capture timing thresholds were weakened.
- Pipeline guard: **15 checks, 0 violations**; diff whitespace and compilation
  checks passed. Independent spec and quality reviews found no remaining issues
  in the refactor after the GUI module-stub and single-wrist fixes.
- Synthetic 100-iteration benchmark: pixel equality passed; median legacy/shared
  panel rendering was approximately **6.68 / 6.54 ms**, p95 **8.08 / 8.34 ms** on
  this run. These are observations, not portable performance guarantees.

The branch remains unmerged. The entire repository is **not** claimed green;
embedded research verifiers under `twm` and physical hardware were not run as a
blanket suite.

No dataset publication, force/action regeneration, calibration rewrite, or live
hardware acquisition was performed by this refactor.
