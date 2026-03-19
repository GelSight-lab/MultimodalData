# TWM Data Collection

Tools for collecting and reviewing multimodal data for the Tactile World Model (TWM) project.

## Hardware

| Sensor | Count | Details |
|--------|-------|---------|
| Intel RealSense D415 | 3 | Color (640×480 @ 30 Hz) + depth (640×480 @ 30 Hz) |
| GelSight Mini | 2 | Left + right tactile sensors, USB video (640×480 @ 30 Hz) |
| OptiTrack | 3 trackers | `motherboard`, `sensor_left`, `sensor_right` via VRPN/ROS |

Camera serials are set at the top of `data_collection.py` (`REALSENSE_SERIALS`, `GELSIGHT_SERIALS`).

---

## Collecting Data

```bash
python -m twm.data_collection --task <task_name>
```

`--task` is required and controls where data is saved:

```
data/<task_name>/<YYYY-MM-DD>/episode_000.h5
                              episode_001.h5
                              ...
```

### Controls

| Key | Action |
|-----|--------|
| `s` | Start a new episode |
| `e` | End episode and save |
| `r` | Reset GelSight diff reference to current frame |
| `q` | Quit (saves in-progress episode if recording) |

### Typical workflow

1. Start the script. The preview window opens; all sensors initialize.
2. Position the setup. Press `r` to set the GelSight diff reference (grey = no contact).
3. Press `s` to begin recording. The status bar turns red: `[REC ep_0000 | ...]`.
4. Perform the task.
5. Press `e` to end the episode. The script flushes buffered frames and OptiTrack poses to disk.
6. Repeat from step 2 for the next episode.
7. Press `q` to quit.

A log row is appended to `data/dataset_log.csv` after each episode.

---

## Visualizing Episodes

```bash
python -m twm.visualize path/to/episode_000.h5
python -m twm.visualize path/to/episode_000.h5 --fps 15   # override playback speed
```

### Controls

| Key | Action |
|-----|--------|
| `space` | Pause / resume |
| `→` / `d` | Next frame (while paused) |
| `←` / `a` | Previous frame (while paused) |
| `r` | Reset GelSight diff reference to current frame |
| `q` | Quit |

---

## HDF5 File Format

Each episode is one `.h5` file. Structure:

```
episode_NNN.h5
├── metadata/               (attrs: fps, task, created_at, realsense_serials, gelsight_serials)
├── timestamps              float64 [T]           — Unix time per frame
├── realsense/
│   ├── cam0/
│   │   ├── color           uint8  [T, 480, 640, 3]   — BGR
│   │   └── depth           uint16 [T, 480, 640]      — millimetres
│   ├── cam1/  (same)
│   └── cam2/  (same)
├── gelsight/
│   ├── left/
│   │   └── frames          uint8  [T, 480, 640, 3]   — raw RGB
│   └── right/  (same)
└── optitrack/
    ├── motherboard/
    │   ├── timestamps      float64 [N]           — Unix time per pose sample
    │   └── pose            float64 [N, 7]        — [x, y, z, qx, qy, qz, qw] (metres)
    ├── sensor_left/  (same)
    └── sensor_right/ (same)
```

**Notes:**
- `T` = number of camera frames (same across all camera streams within an episode).
- `N` = number of OptiTrack samples, recorded at the motion capture system rate (typically higher than camera FPS). Use `timestamps` to align with camera frames.
- All camera data is LZF-compressed, chunked per frame for fast random access.
- Depth values are in **millimetres** (uint16, range 0–65535).
- GelSight frames are raw; compute contact difference offline: `diff = frame - ref + 128` (clipped to uint8), where `ref` is a no-contact reference frame.

### Reading an episode

```python
import h5py
import numpy as np

with h5py.File("episode_000.h5", "r") as f:
    timestamps   = f["timestamps"][:]               # (T,)
    color_cam0   = f["realsense/cam0/color"][:]     # (T, 480, 640, 3)
    depth_cam0   = f["realsense/cam0/depth"][:]     # (T, 480, 640)
    gs_left      = f["gelsight/left/frames"][:]     # (T, 480, 640, 3)
    ot_poses     = f["optitrack/sensor_left/pose"][:] # (N, 7)
    ot_ts        = f["optitrack/sensor_left/timestamps"][:] # (N,)

    # Align OptiTrack to camera frame i
    i = 42
    cam_t = timestamps[i]
    nearest = np.argmin(np.abs(ot_ts - cam_t))
    pose_at_frame_i = ot_poses[nearest]  # [x, y, z, qx, qy, qz, qw]
```
