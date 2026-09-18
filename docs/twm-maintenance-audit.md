# TWM maintenance audit — 2026-09-18

## Scope and delivered first phase

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

## Remaining findings, prioritized

These are follow-ups, **not changes made in this phase**. Keep separate regression
tests and scope each fix before changing capture/storage behavior.

1. **Writer drain/flush race** — `recorder/writer.py`: `_in_flight` is cleared and
   drain notified before `_maybe_flush`. `recorder/app.py` can then read/stamp/close
   the same HDF5 file. Keep in-flight through flush without holding the condition
   during I/O; use an event-controlled concurrency regression. Existing writer
   tests poll the flush and therefore do not prove drain waits for it.
2. **Fatal reporting can fail recursively** — `recorder/capture.py`:
   `_publish_fatal` calls `writer.stats()` even when stats itself caused the
   failure. Reuse the last snapshot (safe initial fallback) and ensure stop signals
   are set in a finally path.
3. **Incomplete preprocessing can look complete** — `preprocess/pipeline.py`:
   parquet existence gates skipping, but parquet is written before all sidecars.
   A failure after parquet can suppress a necessary retry. Define a completion
   receipt/artifact check that supports rigs with missing optional cameras;
   `preprocess/complete.py` is stronger but assumes seven streams.
4. **Inconsistent frame-reader failures** — `react_toolbox/io.py`: empty frame
   selections fail, AV can silently drop requested frames, OpenCV substitutes
   fixed-size black frames, and decoder cleanup needs stronger finally coverage.
   Cache the maximum requested index outside the decode loop; standardize missing
   frame semantics without altering the published toolbox's import independence.
5. **Preprocess encoder cleanup** — `preprocess/encode.py`: BrokenPipe from stdin
   close can prevent process wait. The new visualization writer handles this,
   but preprocessing has not been migrated. Add reaping/buffer-shape tests first.
6. **Force visualization cost/alignment** — `force_recovery/visualize.py` uses
   legacy tactile shifts for timestamped data, redraws growing history each frame
   (quadratic work), and encodes twice with incomplete finally cleanup. Do not
   change force target generation while consolidating presentation.
7. **Heavy import chains** — selected force/toolbox visualization paths import
   matplotlib, Arrow, decoders or inference dependencies just to use small image
   helpers. Move shared pure helpers carefully; do not introduce a parent-TWM
   dependency into the independently shipped toolbox.
8. **Wheel completeness** — `pyproject.toml` package declarations include `twm`
   but omit its subpackages. Add an installed-wheel smoke test outside the checkout
   before changing discovery (including optional hardware dependencies). Current
   instructions intentionally run from a checkout.

## Verification notes

The baseline suite (`tests`, before implementation, maxfail=5) produced 919 passes
and five failures: an on-disk undeclared `pushT/2026-09-17` calibration session,
the task-list duplication guard, and three force-free dataset-statistics tests.
Unrelated user edits in the main checkout were not copied over or overwritten.
Final refactor verification is recorded separately after integration.

No dataset publication, force/action regeneration, calibration rewrite, or live
hardware acquisition was performed by this refactor.
