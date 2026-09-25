---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - robotics
  - tactile
  - manipulation
  - multimodal
  - gelsight
  - realsense
  - motion-capture
  - world-model
  - human-collected
  - lerobot
pretty_name: React (Tactile-Visual Manipulation)
size_categories:
  - 100K<n<1M
configs:
  - config_name: motherboard
    data_files:
      - split: train
        path: data/motherboard/meta/**/*.parquet
  - config_name: pushT
    data_files:
      - split: train
        path: data/pushT/meta/**/*.parquet
  - config_name: all
    default: true
    data_files:
      - split: train
        path: data/**/meta/**/*.parquet
---

# React — Multi-Task Tactile-Visual Manipulation

Dense, contact-rich, synchronized multimodal interaction data collected from **human hands holding handheld GelSight tactile sensors** (no robot arm). Intended for **tactile-visual dynamics / world-model learning**.

> **133 min · 240 k frames @ 30 Hz · 3× RGB + 2× GelSight + OptiTrack · 2 tasks**

## Format — LeRobot-style video release

Each episode ships as **5 MP4 video streams** (640×480, H.264) + a **per-frame parquet** of poses and contact metrics. This matches how LeRobot / DROID / Open X-Embodiment ship manipulation data: tiny on disk (whole dataset ≈ 4.3 GB vs ~1 TB raw), random-access decodable, training-ready.

```
data/<task>/
├── calibration/                     # OptiTrack→camera extrinsics for this task
│   ├── T_mocap_to_cam_{left,middle,right}.json
│   ├── T_gel_to_rigid_{left,right}.json
│   └── calibration.json             # epoch, applies-to dates, RMSE, chain
├── videos/<date>/episode_NNN/
│   ├── view_left.mp4  view_middle.mp4  view_right.mp4    # 640×480 RGB
│   └── tactile_left.mp4  tactile_right.mp4               # 640×480 GelSight
├── meta/<date>/episode_NNN.parquet  # one row per frame (see below)
├── episodes.jsonl                   # one row per episode
├── segments.json                    # clean-segment index (no bad frames)
├── bad_frames.json                  # quality intervals per episode
└── previews/<date>/episode_NNN.mp4  # 1280×480 viewer-layout preview
```

### parquet columns (per frame, aligned to video frame `i`)
| Column | Type | Meaning |
|---|---|---|
| `frame_idx` / `frame_index` | int | 0…T-1, matches MP4 frame index |
| `episode` / `episode_index` | str / int | source episode key and its 0-based index within the task |
| `task` / `task_index` | str / int | task name and index (0=motherboard, 1=pushT) |
| `timestamp` | float64 | camera clock (s) |
| `sensor_left_pose`, `sensor_right_pose` | list[7] | OptiTrack world pose of each GelSight (xyz metres + quat **xyzw**, scalar-LAST — `scipy...Rotation.from_quat` takes it as-is) |
| `object_pose` | list[7] | OptiTrack world pose of the manipulated object (NaN where the object body was not tracked — e.g. all pushT) |
| `tactile_{L,R}_{intensity,area,mixed}` | float32 | contact metrics (computed at full 640×480) |
| `source_h5_frame` | int | index into the original recording |

**Decoded frames are RGB** (standard decoder convention) for all five RGB streams.

### depth (optional, `data/<task>/depth/`)
Per-camera depth is shipped as **lossless FFV1 16-bit video** (`gray16le`):
```
data/<task>/depth/<date>/episode_NNN/depth_{left,middle,right}.mkv
```
- uint16, **millimeters**; `0` = no return / invalid.
- Frame `i` aligns to the RGB video frame `i` and parquet row `i`.
- Decode with PyAV (`frame.to_ndarray()` → `(480, 640)` uint16). cv2 cannot read 16-bit video.
- Load via `ReactVideoDataset(..., load_depth=True)`.

## Tasks

`main` carries one week — 2026-09-10 onward. Earlier sessions live on
`old_data/`, and the 2026-09-09 session (different wrist cameras) ships
separately as `data/validation`.

| Task | Segments | Source recordings | Dates | Duration | Flagged frames |
|---|---|---|---|---|---|
| **motherboard** | 25 | 12 | 2026-09-11/12 | 93.9 min | 0.30 % |
| **pushT** | 68 | 23 | 2026-09-10/11/12/15 | 73.9 min | 0.04 % |
| **rope** | 53 | 16 | 2026-09-11/14/15 | 79.9 min | 0.15 % |

Segments and source recordings are reported separately on purpose: 25 segments
is not 25 independent recordings, and reading it that way overstates the
diversity by a factor of two.

See [`tasks.json`](tasks.json) for the machine-readable registry.

## Statistics

![wrist-camera era](assets/stats_wrist_era.png)

Scale, contact-force distribution, contact occupancy, and tactile validity per
task. The force distribution uses **unsaturated samples only**; the fraction at
the pipeline ceiling is annotated separately rather than mixed in, because a
clipped value is not a measurement. Raw numbers: [`assets/dataset_stats.json`](assets/dataset_stats.json).

![2026-09-09 session](assets/stats_arducam_session.png)

`data/validation` is drawn apart because it is not comparable: different wrist
cameras, and a session the rest of the release does not share.

## Calibration — read `up_axis`, and use the epoch the session declares

**The published poses are Z-up.** Every calibration file on `main` says so:

```json
{ "T_mocap_to_cam": [...], "up_axis": "z",
  "up_axis_note": "converted from the recorded Y-up by R_x(-90): (x,y,z)->(x,-z,y). ..." }
```

A file with no `up_axis` key is the Y-up original. **Pairing one with these
poses raises nothing**: the two forms of the same solve differ by 1.27 in
matrix norm, so the projected point stays inside the frame and the image looks
plausible. Tactile readings are unaffected. The view-frame action is wrong by
`R_x(90)`. Refuse a file that does not declare its axis rather than assuming
one — `twm.calibration_frame.require_zup` does exactly that.

Cameras are **recalibrated between sessions**, so a recording names its epoch
and nothing infers it from the date: pushT's 2026-06-18 belongs to the June-26
solve, measured eight days later. Every session on `main` declares the
**2026-09-09** epoch, published at `data/<task>/calibration/epoch_2026-09-09/`.

Camera extrinsics are used only for projection into a view; **stored poses are
OptiTrack world-frame** and do not depend on the calibration.

## Splits — held-out INTERVALS, not held-out episodes

`data/<task>/splits.json`. Each segment contributes a few held-out windows from
its middle; the rest of that segment trains. Holding out whole episodes would
spend the scarce resource — episodes, and with them scene layouts and lighting —
to buy an independence that a short-horizon world model does not need.

| Task | test | guard | train | intervals |
|---|---|---|---|---|
| motherboard | 11.9 % | 9.4 % | 78.8 % | 271 |
| pushT | 12.0 % | 9.5 % | 78.5 % | 133 |
| rope | 12.1 % | 9.6 % | 78.3 % | 109 |

**`guard` is the part that leaks if you ignore it.** A training window of span
S starting shortly BEFORE a held-out interval `[a, b]` still contains its
frames, so starts in `[a-(S-1), b]` must be rejected — not just `[a, b]`.
`guard_frames` is `max_train_window - 1` (63 frames at the published
`max_train_window` of 64) and it is RECORDED in the file. A loader using a
longer window must FAIL rather than silently leak: see `assert_window_fits`.
That failure mode leaves no trace in any metric until the numbers are
suspiciously good.

The guard frames are neither trained on nor tested on. That is what
independence costs; it is listed rather than folded into `train`.

An episode present in `episodes.jsonl` but absent from `splits.json` is read as
TRAIN by `ReactVideoDataset._split_filter`. Every published segment appears in
both.

## Downloading — depth is optional

The dataset splits into a **lightweight core** (RGB + tactile + poses, ~4.4 GB) and an **optional depth tree** (`data/<task>/depth/`, ~33 GB lossless). Depth lives in its own subtree so you can skip it entirely.

```python
from huggingface_hub import snapshot_download

# Core only — RGB + tactile + parquet, NO depth (~4.4 GB)
snapshot_download("yxma/React", repo_type="dataset",
                  ignore_patterns=["*/depth/*"])

# Everything including depth (~37 GB)
snapshot_download("yxma/React", repo_type="dataset")

# One task only
snapshot_download("yxma/React", repo_type="dataset",
                  allow_patterns=["data/motherboard/*"], ignore_patterns=["*/depth/*"])
```

Or use the helper: `python examples/download.py --no-depth` (see [`examples/download.py`](examples/download.py)).

The `ReactVideoDataset` loader **never touches depth unless you pass `load_depth=True`**, so depth-free training requires no depth download.

## Loading

```python
from examples.react_video_dataset import ReactVideoDataset

ds = ReactVideoDataset("data/motherboard", window_length=16, mode="segment")
sample = ds[0]
# sample["view_middle"]:       (16, 480, 640, 3) uint8 RGB
# sample["tactile_left"]:      (16, 480, 640, 3) uint8 RGB
# sample["sensor_left_pose"]:  (16, 7) float32
```
`mode="segment"` iterates clean spans (no bad frames by construction); `mode="window"` slides over whole episodes and skips `bad_frames.json` intervals. Backend: PyAV (install `decord` for faster random access).

## ⚠️ Known issue: tactile acquisition latency (~15 frames)

Recordings **up to and including 2026-06-18** have a GelSight-vs-camera capture
lag of **≈15 frames (~0.5 s)**: the tactile stream at index `i` was physically
captured ~15 frames *before* the camera/pose at the same index. Cause: a
recording-side `cv2.VideoCapture` V4L2 buffer that was never flushed
(throttled reads + no `BUFFERSIZE=1` + default pixel format). Fixed in the rig
on 2026-06-27; **future recordings will not have this lag**.

The streams are stored frame-aligned by tick index, so this lag is baked in but
**correctable**. The reference loader compensates at load time:

```python
ds = ReactVideoDataset("data/motherboard", tactile_latency=15)  # pairs view[i] with tactile[i+15]
```

`tactile_latency` shifts both the tactile videos and the tactile contact-scalar
columns; poses/views/depth are unchanged. Set `tactile_latency=0` for the raw
(uncompensated) data. The exact per-session value should be re-measured with
`camera_stream/measure_gelsight_latency.py`.

## Data quality
Per-task `bad_frames.json` flags `intensity_spikes`, `pose_teleports_{L,R}`,
`ot_loss_{L,R}` (OptiTrack track loss), `tactile_freeze_{L,R}` — a disconnected
sensor repeats its last frame rather than going blank, so every scalar detector
sees a perfectly steady signal — and `cam_corruption`, torn frames found by
decoding the published video, which no sidecar scalar can see.

Flagged fractions are in the task table above; `segments.json` already excludes
these spans.

## Notes
- **Depth** is available in the source recordings and will be added under `data/<task>/depth/` in a later upload.
- One pushT source recording (`episode_004`) was corrupt and excluded.
- The previous single-task `.pt` release (`episodes/`, `segments/`) is superseded by this video format.

## License
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
