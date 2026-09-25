# React Multi-Task Dataset — Task-First, Video-Format Redesign

**Date:** 2026-06-26
**Status:** Design approved, pending spec review
**Owner:** yxma

## 1. Goal

Restructure the published `yxma/React` Hugging Face dataset to:

1. Support **multiple tasks** (currently `motherboard`; adding `pushT`; 4–5 expected) under a clean, self-contained, **task-first** layout.
2. Switch the publication format from raw `.pt` tensors to the **LeRobot-style video format** (per-camera MP4 + per-episode parquet), aligning with how the field (LeRobot, DROID, Open X-Embodiment) ships manipulation data.
3. **Embed camera extrinsics** per task, with the correct calibration epoch:
   - `motherboard` → **May-12** calibration
   - `pushT` → **June-26** calibration
4. **Process the new `pushT` task** (5 episodes, 2026-06-18) through the full curation pipeline.
5. Keep full **640×480 original resolution** for all RGB + GelSight streams (empirically only ~6.5 GB total at CRF 18 — smaller than the current 128×128 `.pt`).

Depth is **deferred** to a later upload (`<task>/depth/`); the source H5 retains depth as archive.

## 2. Background / Current State

- **Current HF layout (type-first, motherboard-only):**
  `episodes/motherboard/<date>/episode_*.pt` (32), `segments/motherboard/<date>/episode_*.segment_*.pt` (78), root-level `bad_frames.json` / `segments.json` / `tasks.json` / `freeze_intervals.json`, `figures/`, `docs/`, `examples/`, `metadata/`.
- **`.pt` schema:** `view_{left,middle,right}` + `tactile_{left,right}` (128×128 uint8), `sensor_{left,right}_pose` (T,7), `timestamps`, `tactile_*_{intensity,area,mixed}`, `_contact_meta`.
- **Source H5 (archive, on local disk):** motherboard 631 GB + pushT 403 GB ≈ 1.03 TB. 640×480 RGB×3 + depth×3 + GelSight×2 + OptiTrack. **Still present, not deleted.**
- **Calibration epochs (verified):**
  - March-19 (single cam) → for the deleted 2026-03-23 session.
  - **May-12** (`result backup/`, `*.json.may_bak`, `result.zip`): left+middle+right, 8 pts, RMSE 5.3 mm → **all surviving motherboard sessions (05-10/05-11/05-19)**.
  - **June-26** (current `calibration/result/`): left+middle+right, 11 pts, RMSE 0.48 px → **pushT** and future tasks.
- **Important fact:** camera extrinsics are NOT baked into stored tensors; they are used only for the projection overlay in previews. Stored poses are OptiTrack world-frame.
- **2026-05-19 world-frame offset** `(dx,dy,dz) = (0.23, 0, 0.175)` m already applied to that session's `sensor_*_pose`; must be preserved in the new parquet poses.

## 3. Target Layout (task-first)

```
React/
├── README.md                       # top-level dataset card (multi-task)
├── tasks.json                      # global registry: per-task calibration epoch, sensors, dates, counts
├── docs/                           # curation pipeline, schema, caveats, calibration chain
├── examples/                       # ReactVideoDataset loader + demos
└── data/
    ├── motherboard/
    │   ├── calibration/            # May-12 extrinsics
    │   │   ├── T_mocap_to_cam_left.json
    │   │   ├── T_mocap_to_cam_middle.json
    │   │   ├── T_mocap_to_cam_right.json
    │   │   ├── T_gel_to_rigid_left.json
    │   │   └── T_gel_to_rigid_right.json
    │   ├── videos/<date>/episode_NNN/
    │   │   ├── view_left.mp4   view_middle.mp4   view_right.mp4
    │   │   └── tactile_left.mp4   tactile_right.mp4
    │   ├── meta/<date>/episode_NNN.parquet
    │   ├── episodes.jsonl          # one row/episode: date, n_frames, duration_s, active_sensors, calibration_id, trim_offset, world_frame_offset
    │   ├── segments.json           # clean-segment index → (episode, frame_range)
    │   ├── bad_frames.json         # per-episode quality intervals
    │   └── previews/<date>/episode_NNN.mp4
    └── pushT/                       # identical structure; calibration = June-26
```

Each task is **self-contained** (calibration + data + curation + previews). Adding/removing a task is a single folder operation.

## 4. Publication Format (LeRobot-style)

### Video streams (per episode)
- 5 MP4 files: `view_{left,middle,right}.mp4`, `tactile_{left,right}.mp4`.
- **Resolution: 640×480 original** (no resize). BGR→encoder handled so decoded color matches H5 convention.
- Codec: **libx264, yuv444p, CRF 18** (high quality; chroma-preserving for tactile detail), `+faststart`.
- Frame rate: 30 fps. Frame `i` in every MP4 corresponds to row `i` in the parquet and source H5 frame `trim_offset + i`.

### Per-frame metadata (parquet, one row/frame)
Columns:
- `frame_idx` (int)
- `timestamp` (float64)
- `sensor_left_pose` (list[7] float32), `sensor_right_pose` (list[7] float32) — OptiTrack world frame, **05-19 offset already applied**
- `tactile_left_intensity/area/mixed`, `tactile_right_intensity/area/mixed` (float32) — computed on **full-resolution** GelSight vs p01 reference before any resize
- `source_h5_frame` (int) — = trim_offset + frame_idx

### Per-episode metadata (`episodes.jsonl`)
`{episode, date, n_frames, duration_s, active_sensors, calibration_id, trim_offset, world_frame_offset, gelsight_serials}`

### Calibration (`<task>/calibration/`)
- Copied from the correct epoch. Each `T_mocap_to_cam_*.json` keeps intrinsics + extrinsics + RMSE + `created_at`.
- `tasks.json` records `calibration_id` (e.g. `"may-12"`, `"june-26"`) and points each task to its folder.

### LeRobot compatibility
- Schema is "LeRobot-style": MP4-per-camera + parquet-per-episode + JSON meta, structurally aligned with LeRobot v2 (videos + episodes parquet + info/stats json).
- **Phase 1**: our own clean schema (loadable by the shipped `ReactVideoDataset`).
- **Phase 2 (later)**: add a full `info.json` + `stats.json` so the `lerobot` library can load it directly. Not in this milestone.

## 5. pushT Processing

- 5 episodes, 2026-06-18. Same H5 schema (3 cam + 2 GelSight + OptiTrack). GelSight serials `2DUPB53G` (left, swapped) + `2BKRDTAD` (right).
- Reuse `build_episodes_from_h5.py` logic for cam-aligned poses + p01 reference + contact scalars.
- Run `detect_bad_intervals.py` (validated 25/27 bit-identical vs published) → produce `pushT/bad_frames.json`.
- Slice clean segments → `pushT/segments.json`.
- **world-frame offset:** default NONE; only add if projection preview shows OT-origin drift (verify visually like 05-19).
- Large episodes (up to 26k frames) — encode fine; parquet stays small.

## 6. Dataloader (`examples/`)

- New `ReactVideoDataset`:
  - Random-access MP4 frame decode via `decord` (or `torchcodec`/`av` fallback).
  - Reads parquet for pose/scalar/timestamp aligned by `frame_idx`.
  - Supports window sampling, segment sampling, `skip_bad_frames`, `active_sensors` filtering, `which_sensors`.
  - Returns dict of `(T, C, H, W)` decoded frames + `(T, …)` metadata tensors.
- Keep a short note documenting the old `.pt` loader as **deprecated**.

## 7. Consistency Scope

- **motherboard is also converted to A1** (videos + parquet), replacing current `episodes/*.pt` + `segments/*.pt`.
- All 299k frames (motherboard 225,836 + pushT 73,302) re-encoded into the new format.
- Old HF paths removed in the same migration: `episodes/`, `segments/`, root `bad_frames.json` / `segments.json` / `freeze_intervals.json`. New global `tasks.json` retained/rewritten.

## 8. Verification (mandatory, not skippable)

1. **Frame alignment**: for ≥1 episode per task, assert decoded MP4 frame `i` ≈ H5 frame `trim_offset+i` (mean abs pixel diff < small ε accounting for codec), and parquet row `i` pose == cam-aligned H5 pose at that frame. Must pass on a single episode before fan-out.
2. **Color correctness**: decoded RGB matches H5 channel convention (no BGR/RGB swap regressions).
3. **05-19 offset present**: parquet poses for 2026-05-19 carry the (0.23,0,0.175) shift.
4. **Contact scalars**: spot-check parquet scalars vs recomputed-from-full-res values.
5. **Projection sanity**: previews rendered with the task's calibration; markers land on sensors (visual).
6. **Counts**: episodes.jsonl + segments.json totals match expected frame counts.

## 9. Timeline (max parallelism)

- Pipeline code + single-episode alignment verification: ~3 h (serial, gated).
- 36-episode encode (multiprocess, bounded by H5 read of ~600 GB needed streams): ~4–6 h background.
- Upload (~6.5 GB): <1 h.
- **Total: half-day to a day, mostly background.**

## 10. Non-Goals (this milestone)

- Depth in release (deferred to `<task>/depth/` later).
- Full `lerobot`-library direct loadability (Phase 2).
- Re-deriving higher-than-source resolution.
- Re-curation of failure modes beyond the validated `detect_bad_intervals.py` ruleset.

## 11. Risks

- **Frame drop on encode**: ffmpeg can drop/dup frames if input timing is off — mitigated by `-r 30` constant-rate raw input + post-encode frame-count assertion.
- **Decoder nondeterminism**: random-access seek in some decoders is off-by-one — pin decoder + verify alignment on real frames.
- **H5 read bottleneck / disk contention**: schedule encode workers to bound parallel H5 reads.
- **pushT OT-origin unknown**: may need its own world-frame offset; verify before publishing.
