# Mocap Abnormal Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conservatively classify OptiTrack pose discontinuities, identify only safely interpolable spans, and generate a canonical-preview HTML review package for every unresolved span without overwriting released data.

**Architecture:** A pure NumPy detector converts each hand's 7-D pose stream into explainable event records using joint rotation, translation, return, inverse-jump, branch-support, and local smoothness evidence. A separate CLI scans release parquets, merges nearby per-transition findings into review events, renders short clips from the released camera/tactile videos with the existing preview panel, and writes `events.json`, CSV, and a self-contained HTML decision index. Repair remains opt-in and out of scope for the first review build.

**Tech Stack:** Python, NumPy, SciPy Rotation, PyArrow, OpenCV/FFmpeg, pytest, static HTML/JavaScript.

---

### Task 1: Explainable pose-event classification

**Files:**
- Create: `twm/react_preprocess/pose_anomaly.py`
- Create: `tests/test_pose_anomaly.py`

- [ ] **Step 1: Write failing synthetic tests**

Cover a one-frame return, a short contiguous return, A/B flicker with coupled translation, a real continuous large turn, a persistent branch change, an episode-edge jump, and a long known gap. Assert event bounds, `kind`, `repairable`, and evidence fields.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest -q tests/test_pose_anomaly.py`

Expected: collection fails because `twm.react_preprocess.pose_anomaly` does not exist.

- [ ] **Step 3: Implement the minimal detector**

Define immutable `PoseEvent` records and `detect_pose_events(pose, side, known_gaps=())`. Candidate transitions come from a physical discontinuity gate; classification then requires stronger temporal evidence. Returning spans must have two reliable anchors, a near-inverse exit jump, joint position/orientation return, and at most 15 interior frames to be repairable. Alternating branch observations use sign-insensitive quaternion distance plus translation support. One-sided and persistent changes remain review-only.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest -q tests/test_pose_anomaly.py`

Expected: all pose-anomaly tests pass.

### Task 2: Episode scanning and stable report schema

**Files:**
- Create: `twm/scripts/build_pose_review.py`
- Create: `tests/test_build_pose_review.py`

- [ ] **Step 1: Write failing scanner tests**

Use temporary parquets for both hands and a temporary `pose_gaps.json`. Verify source-frame preservation, per-hand events, merging of nearby transitions, deterministic IDs, unresolved-only filtering, and JSON/CSV serialization.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest -q tests/test_build_pose_review.py`

Expected: collection fails because `build_pose_review.py` does not exist.

- [ ] **Step 3: Implement scanning and dry-run CLI**

Add `scan_parquet`, `scan_tree`, `merge_review_events`, and CLI flags for release root, task, date, episode, output root, and `--render`. Default behavior only writes report metadata under a new review directory; it never edits parquet or release video files.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest -q tests/test_build_pose_review.py`

Expected: scanner tests pass.

### Task 3: Canonical preview clips

**Files:**
- Modify: `twm/scripts/build_pose_review.py`
- Modify: `tests/test_build_pose_review.py`

- [ ] **Step 1: Write failing renderer tests**

Create tiny synthetic MP4 streams and assert that rendering uses row-aligned released videos, tolerates absent wrist streams, includes exactly the requested context window, labels hand/event/frame, and produces a clip that decodes to the expected frame count.

- [ ] **Step 2: Run renderer tests and verify RED**

Run: `python -m pytest -q tests/test_build_pose_review.py -k render`

Expected: failure because the renderer is missing.

- [ ] **Step 3: Implement the renderer**

Decode `view_left`, `view_middle`, `view_right`, `tactile_left`, `tactile_right`, and optional wrist videos for the event window. Feed frames into the existing `build_preview_panel`, draw event state and raw pose metrics on each panel, encode through FFmpeg, and verify the resulting decoded frame count.

- [ ] **Step 4: Run renderer tests and verify GREEN**

Run: `python -m pytest -q tests/test_build_pose_review.py -k render`

Expected: renderer tests pass.

### Task 4: Interactive HTML decision index

**Files:**
- Modify: `twm/scripts/build_pose_review.py`
- Modify: `tests/test_build_pose_review.py`

- [ ] **Step 1: Write failing HTML tests**

Assert that the generated page links every unresolved clip, shows rotation/translation plots and evidence, offers `keep`, `repair`, and `invalidate` controls, persists choices in browser local storage, and exports a JSON decisions file.

- [ ] **Step 2: Run HTML tests and verify RED**

Run: `python -m pytest -q tests/test_build_pose_review.py -k html`

Expected: failure because HTML generation is missing.

- [ ] **Step 3: Implement self-contained HTML generation**

Embed event metadata and plot samples as JSON, render charts with SVG/JavaScript without external dependencies, store decisions by deterministic event ID, and add an export button.

- [ ] **Step 4: Run HTML tests and verify GREEN**

Run: `python -m pytest -q tests/test_build_pose_review.py -k html`

Expected: HTML tests pass.

### Task 5: Real-data audit and review build

**Files:**
- Create under review output only: `/media/yxma/Disk1/twm/review/mocap_motherboard_2026-09-17/`

- [ ] **Step 1: Run a metadata-only audit**

Run the CLI without `--render` on `release/motherboard`; compare event totals and representative frame ranges with the existing >30-degree inventory and `pose_gaps.json`.

- [ ] **Step 2: Inspect representative classifications**

Check at least one known repaired flicker, one short returning excursion, one long gap, one persistent branch change, and one plausible large real motion. Confirm the repaired 8544-8575 clip is not emitted as unresolved.

- [ ] **Step 3: Render all unresolved merged events**

Run the CLI with `--render`, one process only, so it does not compete aggressively with the active preprocessing scheduler. Verify every `events.json` clip path exists and decodes.

- [ ] **Step 4: Run complete regression verification**

Run: `python -m pytest -q tests/test_pose_anomaly.py tests/test_build_pose_review.py tests/test_pose_repair.py tests/test_pose_flicker.py tests/test_repair_release_poses.py tests/test_preview_clip_window.py tests/test_preview_gel_frame_choice.py`

Expected: all selected tests pass with zero failures.

