# Unified TWM Visualization Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for bounded
> implementation tasks and independent spec/quality review. Preserve user edits.

**Goal:** A reusable multimodal compositor, compatible canonical previews, clearer
tests and one documented CLI for inspection/rendering.

**Architecture:** Keep existing projection mathematics authoritative. Introduce
explicit modality tiles and overlay composition independently from I/O. Adapt
legacy preview callers through a shared preview module; do not rewrite storage.

**Tech Stack:** Python, NumPy, OpenCV, pytest, existing HDF5/Parquet/video readers.

### Task 1: Compositor and compatibility adapter

Files: create `twm/visualization/{__init__,core,preview}.py` and
`tests/visualization/test_core.py`, `tests/visualization/test_preview.py`.

- [ ] Write failing tests for named BGR/RGB/depth tiles, missing modalities,
  stable layout, overlay order/clipping and source-array ownership:
  `np.testing.assert_array_equal(source, before)` after `renderer.render(...)`.
- [ ] Verify RED with `PYTHONPATH=. python -m pytest tests/visualization -q`.
- [ ] Implement `Tile`, `Renderer`, `RenderResult` using dataclasses, explicit
  input validation and ordered overlay callables. Overlays operate on tile copies.
- [ ] Implement `render_preview(*args, projection=None, **kwargs)` composing
  canonical `build_preview_panel` and `draw_projection_overlay` without changing
  image bytes when projection is omitted. Introduce a typed projection config.
- [ ] Test depth normalization with NaNs, invalid ranges and missing tile output;
  RGB sentinel `(255,0,0)` must render BGR `(0,0,255)`.
- [ ] Run new tests plus legacy visualization/recorder tests. Commit only task files.

### Task 2: Integrations and test hygiene

Files: `twm/recorder/app.py`, `twm/visualize.py`,
`twm/scripts/build_episode_previews.py`, `tests/test_visualize.py`,
`tests/visualization/test_integrations.py`.

- [ ] Characterize legacy vs shared `render_preview` bytes with synthetic panels.
- [ ] Preserve independent live base/overlay cadence; migrate live construction,
  playback and batch panel construction to `render_preview`.
- [ ] Remove import-time `sys.modules` replacement from viewer tests; import
  pure helpers directly from their owning module. Parametrize alignment cases.
- [ ] Verify each caller still supplies identical timestamps, calibration,
  masks and force arguments; retain compatibility exports.
- [ ] Run recorder/viewer/preview/force/Z-up regressions, then commit task files.

### Task 3: CLI, documentation, benchmark and audit

Files: `twm/visualization/__main__.py`, `twm/visualization/README.md`,
`twm/scripts/benchmark_visualization.py`, `tests/visualization/test_cli.py`,
`docs/twm-maintenance-audit.md`.

- [ ] Write CLI tests for `python -m twm.visualization --help` and backward
  compatible delegation to existing playback/export arguments without hardware.
- [ ] Add lazy CLI dispatch to viewer: import viewer only when executing playback.
- [ ] Document live/replay adapters, native tile overlays, explicit modality color
  conventions, Z-up/units/timestamps, missing channels and extension examples.
- [ ] Benchmark old/new preview equality and warm per-frame rendering with synthetic
  arrays using `time.perf_counter`, fixed iterations and no timing assertions.
- [ ] Record repository-wide inventory and targeted remaining cleanup instead of
  claiming all research/one-off scripts have been rewritten.
- [ ] Run `PYTHONPATH=. python -m twm.pipeline_guard` and complete applicable pytest
  suite; independently review spec compliance then code quality. Commit verified work.
