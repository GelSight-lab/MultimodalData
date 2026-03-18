# TWM Data Collection Design
**Tactile World Model — Multimodal Data Collection Pipeline**
Date: 2026-03-17

## Overview

Standalone data collection pipeline for simultaneous recording from:
- 3× RealSense D415 cameras (RGB + Depth)
- 2× GelSight Mini tactile sensors
- OptiTrack motion capture system (sensor positions + object ground truth)

No robot arm involved. Sensors are hand-held or mounted on a rig. Recording is keyboard-triggered.

---

## New Files

```
camera_stream/
  realsense_stream.py       # RealsenseStream class
optitrack/
  __init__.py
  optitrack_stream.py       # OptitrackStream class
scripts/
  twm_data_collection.py    # main collection script
```

Existing `USBVideoStream` is used as-is for GelSight Mini — no changes needed.

---

## Components

### `RealsenseStream` (`camera_stream/realsense_stream.py`)

- Standalone class, does NOT inherit `BaseVideoStream`
- Identified by serial number (to distinguish 3 cameras)
- Starts a `pyrealsense2` pipeline in a background thread at **30 fps, 640×480**
- Keeps latest aligned color (uint8 BGR) and depth (uint16, mm) frames internally
- API:
  - `start()` / `stop()`
  - `get_color_frame()` — blocking wait, same pattern as `BaseVideoStream.get_frame()`
  - `get_depth_frame()` — blocking wait

### `OptitrackStream` (`optitrack/optitrack_stream.py`)

- Subscribes via `rospy` to 3 VRPN pose topics:
  - `/vrpn_client_node/motherboard/pose`
  - `/vrpn_client_node/sensor_left/pose`
  - `/vrpn_client_node/sensor_right/pose`
- Buffers all incoming poses in a `deque` (with timestamps) at full OptiTrack rate (~120 Hz)
- Spins rospy in a background thread
- API:
  - `start()` / `stop()`
  - `get_latest_pose(name)` → `(timestamp, [x, y, z, qx, qy, qz, qw])`
  - `flush_buffer(name)` → all buffered poses since last flush (used at episode end)

### GelSight Mini

- Uses existing `USBVideoStream` with serials:
  - Left: `2BGLKZNT` → `/dev/video14`
  - Right: `2BKRDTAD` → `/dev/video12`
- Resolution: 640×480, format: BGR, fps: 25 (native)

---

## Data Storage

### File organization

```
data/
  YYYY-MM-DD/
    episode_000.h5
    episode_001.h5
    ...
```

Episode number auto-increments by scanning existing files in the date folder.

### HDF5 structure per episode

```
episode_NNN.h5
├── metadata/              # attrs: date, fps, realsense_serials, gelsight_serials, notes
├── timestamps             # (N,) float64 — Unix timestamps, one per camera frame
├── realsense/
│   ├── cam0/
│   │   ├── color          # (N, 480, 640, 3) uint8, gzip compression
│   │   └── depth          # (N, 480, 640) uint16, gzip compression
│   ├── cam1/color, cam1/depth
│   └── cam2/color, cam2/depth
├── gelsight/
│   ├── left/frames        # (N, 480, 640, 3) uint8, gzip compression
│   └── right/frames       # (N, 480, 640, 3) uint8, gzip compression
└── optitrack/
    ├── timestamps          # (M,) float64 — full OptiTrack rate (~120 Hz), M ≥ N
    ├── motherboard/pose    # (M, 7) float64 [x, y, z, qx, qy, qz, qw]
    ├── sensor_left/pose    # (M, 7) float64
    └── sensor_right/pose   # (M, 7) float64
```

- Datasets are **resizable** (`maxshape=(None, ...)`), written frame-by-frame to keep memory usage flat
- OptiTrack stored at full native rate with its own timestamps; aligned to camera timestamps at training time via nearest-neighbor lookup
- Estimated size: ~1-3 GB/minute with gzip compression

---

## Collection Script (`scripts/twm_data_collection.py`)

### Startup
1. Initialize 3× `RealsenseStream`, 2× `USBVideoStream` (GelSight), 1× `OptitrackStream`
2. Wait for all streams to produce first frame
3. Open live preview window

### Live preview
Tiled OpenCV window:
- Row 1: 3 RealSense color feeds (640×480 each)
- Row 2: 2 GelSight feeds (640×480 each)
- Status bar overlay: `[IDLE]` or `[RECORDING ep_003 | 127 frames | 4.2s]`

### Keyboard controls
| Key | Action |
|-----|--------|
| `s` | Start new episode — creates `episode_NNN.h5`, begins writing |
| `e` | End episode — flushes OptiTrack buffer, closes HDF5, prints summary |
| `q` | Quit — stops all streams cleanly |

### Main loop (30 Hz)
At each tick:
1. Grab latest color + depth from 3× RealSense
2. Grab latest frame from 2× GelSight
3. If recording: append all frames + current timestamp to HDF5 datasets
4. Update preview window

OptiTrack runs in its own background thread at full rate (~120 Hz), buffering continuously. On episode end, the full buffer is flushed to HDF5.

---

## Sensor Serials

| Sensor | Serial / ID |
|--------|-------------|
| GelSight Left | `2BGLKZNT` |
| GelSight Right | `2BKRDTAD` |
| RealSense cam0 | TBD (find with `rs-enumerate-devices`) |
| RealSense cam1 | TBD |
| RealSense cam2 | TBD |

---

## Dependencies

- `pyrealsense2` — RealSense SDK Python bindings
- `rospy` — ROS Python client (for OptiTrack VRPN topics)
- `h5py` — HDF5 file writing
- `cv2` (OpenCV) — frame capture and preview
- `pyudev` — GelSight serial number lookup (existing)
