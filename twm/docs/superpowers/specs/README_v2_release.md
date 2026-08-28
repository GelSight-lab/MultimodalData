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

| Task | Episodes | Dates | Duration | Clean segments | Calibration |
|---|---|---|---|---|---|
| **motherboard** | 32 | 2026-05-10/11/19 | 108 min | 76 (107 min) | **May-12** (RMSE ~5 mm) |
| **pushT** | 4 | 2026-06-18 | 25 min | 17 (25 min) | **June-26** (RMSE ~0.6 px) |

See [`tasks.json`](tasks.json) for the machine-readable registry (per-task dates, sensors, calibration epoch, world-frame offsets).

### Calibration epochs
Cameras were **recalibrated between tasks**. Each task points to the calibration valid for its recordings:
- `motherboard` → **May-12** extrinsics (`data/motherboard/calibration/`)
- `pushT` → **June-26** extrinsics (`data/pushT/calibration/`)

Camera extrinsics are used only for the projection overlay; **stored poses are OptiTrack world-frame** and independent of calibration. The 2026-05-19 motherboard session had a redefined world origin; an offset `(0.23, 0, 0.175) m` is already baked into its poses so all dates share one frame (recorded in `episodes.jsonl`).

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
Per-task `bad_frames.json` flags `intensity_spikes`, `pose_teleports_{L,R}`, `ot_loss_{L,R}` (OptiTrack track loss). Overall flagged: motherboard 0.90 %, pushT 0.67 %. `segments.json` already excludes them.

## Notes
- **Depth** is available in the source recordings and will be added under `data/<task>/depth/` in a later upload.
- One pushT source recording (`episode_004`) was corrupt and excluded.
- The previous single-task `.pt` release (`episodes/`, `segments/`) is superseded by this video format.

## License
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
