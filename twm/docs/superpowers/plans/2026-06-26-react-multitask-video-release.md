# React Multi-Task Video-Format Release — Implementation Plan

> **For agentic workers:** executed inline by the author session (autonomous run requested).

**Goal:** Convert the React dataset (motherboard + pushT) to a task-first, LeRobot-style video release (640×480 MP4 + parquet), with per-task camera extrinsics, and publish to HF with canonical preview videos.

**Architecture:** One `build_video_release.py` encodes each H5 episode → 5 MP4 (view_{L,M,R}, tactile_{L,R}, 640×480, CRF18) + 1 parquet (pose/scalar/timestamp). Reuse existing cam-align/p01/contact-scalar logic from `build_episodes_from_h5.py` and ffmpeg piping from `build_episode_previews.py`. Curation (bad_frames + segments) and previews reuse validated scripts. A final publisher assembles the task-first tree and pushes to HF.

**Tech Stack:** Python, h5py, numpy, ffmpeg (libx264 yuv444p), pyarrow (parquet), decord (decode/verify), huggingface_hub.

## Global Constraints
- Resolution: **640×480 original**, no resize for video streams.
- Codec: libx264, yuv444p, CRF 18, 30 fps, `+faststart`.
- Calibration: motherboard→May-12 (`result backup/`), pushT→June-26 (`calibration/result/`).
- 2026-05-19 poses keep applied offset (0.23, 0, 0.175).
- Frame i in MP4 == parquet row i == H5 frame (trim_offset + i).
- Layout: `data/<task>/{calibration,videos,meta,previews}` + per-task `bad_frames.json`/`segments.json`/`episodes.jsonl`; global `tasks.json`.
- Previews: existing canonical 1280×480 panel, first 30s @ 2x (unchanged style).
- Local staging root: `/media/yxma/Disk1/twm/release/`.

---

### Task 1: Video+parquet encoder (`scripts/build_video_release.py`)

**Files:** Create `scripts/build_video_release.py`. Reuse `to_thumb`? NO — full res, no crop/resize. Reuse `cam_align_poses`, `find_first_valid`, `contact_scalars`, p01 selection from `build_episodes_from_h5.py`.

**Produces:** per episode → `<stage>/videos/<date>/episode_NNN/{view_left,view_middle,view_right,tactile_left,tactile_right}.mp4` + `<stage>/meta/<date>/episode_NNN.parquet`.

- [ ] Encode each cam (cam0→right, cam1→left, cam2→middle) and gelsight L/R as 640×480 MP4 (no resize), BGR for cam (keep), RGB→BGR for gelsight before encoder. Stream from H5 in chunks, pipe raw bgr24 to ffmpeg.
- [ ] Cam-align OT poses (nearest-ts) for trimmed range; apply 2026-05-19 offset when task==motherboard and date==2026-05-19.
- [ ] Compute p01 reference (smoothed-intensity argmin) + contact scalars at FULL 640×480 resolution.
- [ ] Write parquet: frame_idx, timestamp, sensor_left_pose, sensor_right_pose, 6 tactile scalars, source_h5_frame.
- [ ] Post-encode assertion: each MP4 frame count == parquet rows == T.

### Task 2: Frame-alignment verification gate

- [ ] Decode MP4 frame i for one episode each task; compare to H5 frame (trim+i) resized→640 (it's already 640): mean abs diff < 8 (codec tolerance). Verify cam channel order (no R/B swap).
- [ ] Verify parquet pose row matches cam_align_poses output.
- [ ] BLOCK fan-out until this passes.

### Task 3: Fan-out encode all episodes

- [ ] motherboard: 32 episodes (05-10/05-11/05-19), pushT: 5 episodes (06-18). Multiprocess, bounded workers (disk I/O).

### Task 4: Per-task calibration + curation

- [ ] Copy May-12 extrinsics → `motherboard/calibration/`; June-26 → `pushT/calibration/` (+ gel→rigid both).
- [ ] Run `detect_bad_intervals.py` per task → `<task>/bad_frames.json` (needs per-episode .pt OR adapt to read parquet poses+scalars). Adapt detector to read parquet instead of .pt.
- [ ] Build `<task>/segments.json` (index → episode + frame_range), reusing find_clean_segments.
- [ ] Build `<task>/episodes.jsonl`.

### Task 5: Previews (canonical style, per-task calibration)

- [ ] Run `build_episode_previews.py` per task with task's calibration dir → `<task>/previews/<date>/episode_NNN.mp4`. First 30s @ 2x, existing panel layout + projection overlay.

### Task 6: Global metadata + docs

- [ ] `tasks.json`: per-task {dates, n_episodes, active_sensors, calibration_id, calibration_rmse, world_frame_offset}.
- [ ] Rewrite README (multi-task, video format, layout, how to load).
- [ ] `docs/schema.md` (parquet+video schema), `docs/calibration.md` (epoch→task map).

### Task 7: ReactVideoDataset loader (`examples/`)

- [ ] `examples/react_video_dataset.py`: decord MP4 decode + parquet align; window/segment sampling, skip_bad_frames, active_sensors. Smoke-test on one episode.

### Task 8: Publish to HF (task-first) + remove old

- [ ] Upload `data/<task>/...` + `tasks.json` + README + docs + examples.
- [ ] Delete old HF paths: `episodes/`, `segments/`, root `bad_frames.json`/`segments.json`/`freeze_intervals.json`, old `figures/episode_previews/` (replaced by `data/<task>/previews/`).
- [ ] Verify final tree.

## Self-Review
- Spec coverage: layout(T4,T8), video format(T1), calibration(T4), pushT(T1,T3), depth deferred(non-goal), loader(T7), previews(T5), verification(T2). ✓
- Open risk: detector currently reads `.pt`; T4 adapts it to parquet (or briefly also emit lightweight .pt per episode for detection). Decision: emit a tiny per-episode `_detect.pt` (poses+scalars only) during T1 to reuse detector unchanged — cheaper than rewrite.
