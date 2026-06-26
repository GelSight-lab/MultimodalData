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

## Prerequisites

### OptiTrack stream (required for pose data)

OptiTrack poses are streamed over VRPN. Before launching `data_collection.py`,
start the VRPN client in a separate terminal:

```bash
roslaunch vrpn_client_ros sample.launch
```

This connects to the Motive server and publishes pose topics for the
`motherboard`, `sensor_left`, and `sensor_right` rigid bodies. If the launch
isn't running, OptiTrack pose datasets in the saved HDF5 will be empty
(camera/GelSight recording still works).

> **Tip:** The Motive software on the OptiTrack PC must be open and tracking
> the rigid bodies for VRPN to broadcast poses.

---

## Collecting Data

```bash
python -m twm.data_collection --task <task_name>
```

`--task` is required and controls where data is saved. Episodes are written to:

```
/media/yxma/Disk1/twm/data/<task_name>/<YYYY-MM-DD>/episode_000.h5
                                                    episode_001.h5
                                                    ...
```

The root directory is set by `DATA_DIR` at the top of `data_collection.py`
(currently `/media/yxma/Disk1/twm/data`). The dataset log is written to
`<DATA_DIR>/dataset_log.csv`.

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

A log row is appended to `<DATA_DIR>/dataset_log.csv` after each episode.

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

## Camera Calibration

The live preview (`data_collection.py`) and the viewer (`visualize.py`) can
overlay each GelSight sensor's projected position onto the overhead RealSense
views. This requires a per-camera **OptiTrack → camera** extrinsic calibration
(`T_mocap_to_cam`). The full projection chain is:

```
P_gel_rigid  = gel_center_in_rigid_mm                (GelSight → rigid-body calib)
P_gel_mocap  = T_rigid_to_mocap @ P_gel_rigid        (live OptiTrack pose × 1000)
P_gel_cam    = T_mocap_to_cam   @ P_gel_mocap        (camera-view calib, below)
(u, v)       = project(K, P_gel_cam)                 (camera intrinsics K)
```

Calibration files live in `twm/calibration/result/`:

| File | What it maps |
|------|--------------|
| `T_mocap_to_cam_left.json`   | OptiTrack → **left** camera (serial `104122062574`, cam idx 1) |
| `T_mocap_to_cam_middle.json` | OptiTrack → **middle** camera (serial `217222066989`, cam idx 2) |
| `T_mocap_to_cam_right.json`  | OptiTrack → **right** camera (serial `143322063538`, cam idx 0) |
| `T_gel_to_rigid_left.json`   | GelSight surface center in the left rigid-body frame |
| `T_gel_to_rigid_right.json`  | GelSight surface center in the right rigid-body frame |

Each `T_mocap_to_cam_*.json` also stores that camera's intrinsics (`K`), RMSE,
and the raw point pairs used to solve it. The viewer/preview load these
automatically by matching each file's `camera_serial` to `REALSENSE_SERIALS`.

### When to recalibrate

- **A camera moved** (bumped, remounted, or re-aimed) → recalibrate that view.
- **The OptiTrack origin / ground plane was re-set** in Motive → recalibrate all views.
- The GelSight `T_gel_to_rigid_*` only needs redoing if the sensor's mounting on
  its rigid body changes.

### How it works

Click-to-calibrate using a reflective ball visible to **both** OptiTrack and the
camera. For each point you click the ball in the camera image and provide the
ball's OptiTrack position; with ≥4 (ideally ≥6–8) non-coplanar points a transform
is solved per camera and saved as JSON + NPY.

Two solvers are available (`--method`):

| Method | How | Notes |
|--------|-----|-------|
| `pnp` (default) | **2D↔3D**: clicked pixel + mocap point + factory color intrinsics, via `solvePnPRansac` + LM refine | **Depth-free** — avoids RealSense depth noise (the dominant error in `svd`) and minimizes pixel reprojection error directly. RANSAC auto-rejects bad clicks. Reports RMSE in **px**. |
| `svd` | **3D↔3D**: deproject the click with depth, then Arun SVD | Original method; sensitive to depth noise. Reports RMSE in **mm**. Kept for comparison. |

> Requires the OptiTrack VRPN stream running (see [Prerequisites](#prerequisites))
> so you can read each ball position from Motive.

**Auto-capture the mocap position** (skip typing) with `--mocap_body NAME`: the
ball's position is grabbed automatically from the live VRPN stream
`/vrpn_client_node/NAME/pose` when you advance a point. Requires the ball to be
tracked as a rigid body named `NAME` in Motive.

**Marker detection** — by default (`--source ir`) markers are detected in the
RealSense **infrared** stream (emitter on). OptiTrack markers are retroreflective
and built for IR: they appear as the brightest spots, the background is
suppressed, and — unlike RGB — the IR response is uniform across all three
cameras. Detection keys on **local contrast** (a white top-hat), not absolute
brightness or color, so it's robust to lighting gradients across the workspace.
The camera windows show the **RGB image** with each detected marker overlaid at
its mapped color pixel. Because the IR and color imagers are physically offset
(~few-cm baseline), each IR detection is transformed IR-pixel → color-pixel via
depth + factory IR→color extrinsics before being drawn/used — without this the
markers would sit ~15 px off in RGB. So detection is robust (IR) while display and
calibration stay in the color frame. Use `--source color` to detect directly in
RGB instead (far less reliable for these markers — verified on the rig: RGB found
2–5 of 8, IR is stable per camera).

> Detection is stable but each camera only sees the markers not occluded by the
> tracker bodies from its angle (e.g. one view may see all 8, another 5–7). The
> per-camera-optional-click workflow handles this — each camera just needs ≥4.
> Any marker the detector misses can still be hand-clicked (snap refines locally).

**Sub-pixel snapping** (on by default): detected markers are ringed in cyan; a
click **snaps** to the nearest one and refines to its intensity-weighted sub-pixel
center. Each window has a **`thresh` slider** (top-hat contrast), seeded per
camera (IR≈25 fixed; RGB adaptive). Drag it until markers are ringed; press **`m`**
to view the threshold mask, or **`-`/`=`** to nudge all sliders. Other knobs:
`--snap_radius`, `--blob_min_area`, `--blob_max_area`, `--no_snap`, `--blob_thresh`.

### Calibrate all cameras at once (recommended)

Place the ball once, click it in whichever camera views can see it, and type its
OptiTrack position **once** — shared across all those cameras. Each camera
accumulates its own points (≥4 each), so a ball out of frame for one camera just
doesn't contribute to it.

```bash
python -m twm.calibration.mocap_to_cam_multi --num_points 8                 # PnP (default)
python -m twm.calibration.mocap_to_cam_multi --num_points 10 \
    --mocap_body calib_ball                                                 # PnP + auto mocap
python -m twm.calibration.mocap_to_cam_multi --method svd                   # old depth-based solver
```

One live window opens per camera. Per-point controls (focus any window):

| Key | Action |
|-----|--------|
| left-click | Mark the ball in that camera view (re-click to move it) |
| `n` / `space` | Done with this point → enter its OptiTrack position once |
| `r` | Clear this point's clicks and re-click |
| `u` | Undo the last recorded point (all cameras) |
| `q` | Finish: solve + save with the points collected so far |

Solves and writes all three `T_mocap_to_cam_*.json` (+ `.npy`) at once, printing
per-camera RMSE and per-point residuals (✅ <5, ⚠️ <10, ❌ ≥10; in **px** for
`pnp`, **mm** for `svd`).

### Calibrate a single camera

```bash
python -m twm.calibration.mocap_to_cam --serial 217222066989 --num_points 6 \
    --output twm/calibration/result/T_mocap_to_cam_middle.json
```

Same click-then-type flow for one camera; controls: `y` accept · `r` redo ·
`q` quit. Use the output filename matching the camera's view from the table above.

### Using the overlay

Once calibrated, the projection overlay is **on by default**:

```bash
python -m twm.data_collection --task <task_name>   # press 'p' to toggle; --no_projection to start off
python -m twm.visualize path/to/episode_000.h5     # --no_projection to disable
```

A colored dot + XYZ axes is drawn on each calibrated camera view at the
projected GelSight surface center. Accuracy is bounded by the calibration RMSE
(~5 mm ≈ ~2 px at the working distance).

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
