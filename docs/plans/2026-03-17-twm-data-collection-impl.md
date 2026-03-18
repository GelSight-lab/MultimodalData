# TWM Data Collection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multimodal data collection pipeline recording 3× RealSense D415 (RGB+Depth), 2× GelSight Mini, and OptiTrack pose data into per-episode HDF5 files, triggered by keyboard.

**Architecture:** Three new modules — `RealsenseStream` (wraps pyrealsense2 pipeline in background thread), `OptitrackStream` (rospy subscriber with pose buffer), and `scripts/twm_data_collection.py` (30 Hz main loop, keyboard control, HDF5 writer, OpenCV preview). No robot arm involved.

**Tech Stack:** `pyrealsense2`, `rospy` + `geometry_msgs`, `h5py`, `cv2`, `pyudev` (existing), Python threading

---

## Task 1: `RealsenseStream` class

**Files:**
- Create: `camera_stream/realsense_stream.py`
- Create: `tests/test_realsense_stream.py`

### Step 1: Write the failing test

Create `tests/test_realsense_stream.py`:

```python
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import sys

# Mock pyrealsense2 before importing
mock_rs = MagicMock()
sys.modules['pyrealsense2'] = mock_rs

from camera_stream.realsense_stream import RealsenseStream


class TestRealsenseStream(unittest.TestCase):

    def test_get_color_frame_returns_copy(self):
        """get_color_frame() returns a numpy array copy when a frame is available."""
        stream = RealsenseStream(serial="123456")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._color_frame = fake_frame
        stream._last_updated = __import__('time').time()

        result = stream.get_color_frame()
        self.assertEqual(result.shape, (480, 640, 3))
        # Ensure it's a copy, not the same object
        self.assertIsNot(result, fake_frame)

    def test_get_depth_frame_returns_copy(self):
        """get_depth_frame() returns a numpy array copy when a frame is available."""
        stream = RealsenseStream(serial="123456")
        fake_depth = np.zeros((480, 640), dtype=np.uint16)
        stream._depth_frame = fake_depth
        stream._last_updated = __import__('time').time()

        result = stream.get_depth_frame()
        self.assertEqual(result.shape, (480, 640))
        self.assertIsNot(result, fake_depth)

    def test_get_color_frame_times_out(self):
        """get_color_frame() raises TimeoutError if no frame arrives within timeout."""
        stream = RealsenseStream(serial="123456")
        stream._color_frame = None
        with self.assertRaises(TimeoutError):
            stream.get_color_frame(timeout=0.05)

    def test_get_depth_frame_times_out(self):
        """get_depth_frame() raises TimeoutError if no frame arrives within timeout."""
        stream = RealsenseStream(serial="123456")
        stream._depth_frame = None
        with self.assertRaises(TimeoutError):
            stream.get_depth_frame(timeout=0.05)


if __name__ == '__main__':
    unittest.main()
```

### Step 2: Run test to verify it fails

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_realsense_stream.py -v
```

Expected: `ModuleNotFoundError: No module named 'camera_stream.realsense_stream'`

### Step 3: Create `tests/` directory if it doesn't exist

```bash
mkdir -p tests
touch tests/__init__.py
```

### Step 4: Implement `RealsenseStream`

Create `camera_stream/realsense_stream.py`:

```python
import threading
import time
import numpy as np

import pyrealsense2 as rs


class RealsenseStream:
    """
    Threaded RealSense D415 stream providing aligned color (BGR uint8) and
    depth (uint16, millimetres) frames at a fixed fps.

    Usage:
        stream = RealsenseStream(serial="123456789012", fps=30)
        stream.start()
        color = stream.get_color_frame()   # (480, 640, 3) uint8
        depth = stream.get_depth_frame()   # (480, 640)    uint16
        stream.stop()
    """

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps

        self._color_frame = None
        self._depth_frame = None
        self._lock = threading.Lock()
        self._streaming = False
        self._last_updated = 0.0

    def start(self):
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._align = rs.align(rs.stream.color)
        self._pipeline.start(config)
        self._streaming = True
        threading.Thread(target=self._update, daemon=True).start()

    def stop(self):
        self._streaming = False
        if hasattr(self, '_pipeline'):
            self._pipeline.stop()

    def _update(self):
        while self._streaming:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self._align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if color and depth:
                    color_arr = np.asanyarray(color.get_data())
                    depth_arr = np.asanyarray(depth.get_data())
                    with self._lock:
                        self._color_frame = color_arr
                        self._depth_frame = depth_arr
                        self._last_updated = time.time()
            except Exception as e:
                print(f"[RealsenseStream {self.serial}] error: {e}")
                time.sleep(0.01)

    def get_color_frame(self, timeout: float = 2.0) -> np.ndarray:
        """Block until a color frame is available, then return a copy."""
        t0 = time.time()
        while self._color_frame is None:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"No color frame from RealSense serial={self.serial}")
            time.sleep(0.01)
        with self._lock:
            return self._color_frame.copy()

    def get_depth_frame(self, timeout: float = 2.0) -> np.ndarray:
        """Block until a depth frame is available, then return a copy."""
        t0 = time.time()
        while self._depth_frame is None:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"No depth frame from RealSense serial={self.serial}")
            time.sleep(0.01)
        with self._lock:
            return self._depth_frame.copy()
```

### Step 5: Run tests to verify they pass

```bash
python -m pytest tests/test_realsense_stream.py -v
```

Expected: 4 tests PASS.

### Step 6: Commit

```bash
git add camera_stream/realsense_stream.py tests/test_realsense_stream.py tests/__init__.py
git commit -m "feat: add RealsenseStream for D415 color+depth capture"
```

---

## Task 2: `OptitrackStream` class

**Files:**
- Create: `optitrack/__init__.py`
- Create: `optitrack/optitrack_stream.py`
- Create: `tests/test_optitrack_stream.py`

### Step 1: Write the failing test

Create `tests/test_optitrack_stream.py`:

```python
import unittest
from unittest.mock import MagicMock, patch
import sys
import time

# Mock rospy and geometry_msgs before importing
sys.modules['rospy'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()

from optitrack.optitrack_stream import OptitrackStream


class TestOptitrackStream(unittest.TestCase):

    def _make_pose_msg(self, x, y, z, qx, qy, qz, qw, stamp=0.0):
        msg = MagicMock()
        msg.header.stamp.to_sec.return_value = stamp
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def test_get_latest_pose_none_before_any_message(self):
        stream = OptitrackStream()
        self.assertIsNone(stream.get_latest_pose("motherboard"))

    def test_get_latest_pose_after_callback(self):
        stream = OptitrackStream()
        msg = self._make_pose_msg(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, stamp=100.0)
        cb = stream._make_callback("motherboard")
        cb(msg)

        result = stream.get_latest_pose("motherboard")
        self.assertIsNotNone(result)
        t, pose = result
        self.assertAlmostEqual(t, 100.0)
        self.assertEqual(pose, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    def test_flush_buffer_returns_all_and_clears(self):
        stream = OptitrackStream()
        cb = stream._make_callback("sensor_left")
        for i in range(5):
            msg = self._make_pose_msg(float(i), 0, 0, 0, 0, 0, 1.0, stamp=float(i))
            cb(msg)

        data = stream.flush_buffer("sensor_left")
        self.assertEqual(len(data), 5)
        # Buffer should be cleared
        self.assertEqual(len(stream.flush_buffer("sensor_left")), 0)

    def test_flush_buffer_returns_timestamps_and_poses(self):
        stream = OptitrackStream()
        cb = stream._make_callback("sensor_right")
        msg = self._make_pose_msg(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, stamp=42.0)
        cb(msg)

        data = stream.flush_buffer("sensor_right")
        t, pose = data[0]
        self.assertAlmostEqual(t, 42.0)
        self.assertEqual(len(pose), 7)


if __name__ == '__main__':
    unittest.main()
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_optitrack_stream.py -v
```

Expected: `ModuleNotFoundError: No module named 'optitrack'`

### Step 3: Implement `OptitrackStream`

Create `optitrack/__init__.py` (empty):
```python
```

Create `optitrack/optitrack_stream.py`:

```python
import threading
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped

TOPICS = {
    "motherboard": "/vrpn_client_node/motherboard/pose",
    "sensor_left":  "/vrpn_client_node/sensor_left/pose",
    "sensor_right": "/vrpn_client_node/sensor_right/pose",
}


class OptitrackStream:
    """
    Subscribes to three OptiTrack VRPN pose topics and buffers incoming poses.

    Usage:
        stream = OptitrackStream()
        stream.start()
        t, pose = stream.get_latest_pose("sensor_left")   # (float, [x,y,z,qx,qy,qz,qw])
        data = stream.flush_buffer("sensor_left")          # [(t, pose), ...]
        stream.stop()
    """

    def __init__(self, buffer_size: int = 50000):
        self._buffers = {name: deque(maxlen=buffer_size) for name in TOPICS}
        self._latest  = {name: None for name in TOPICS}
        self._lock = threading.Lock()
        self._running = False

    def _make_callback(self, name: str):
        def callback(msg: PoseStamped):
            t = msg.header.stamp.to_sec()
            pose = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
            with self._lock:
                self._buffers[name].append((t, pose))
                self._latest[name] = (t, pose)
        return callback

    def start(self):
        try:
            rospy.init_node("twm_data_collection", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass  # node already initialized
        for name, topic in TOPICS.items():
            rospy.Subscriber(topic, PoseStamped, self._make_callback(name))
        self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        self._spin_thread.start()
        self._running = True

    def stop(self):
        self._running = False

    def get_latest_pose(self, name: str):
        """Return (timestamp, [x,y,z,qx,qy,qz,qw]) or None if no data yet."""
        with self._lock:
            return self._latest[name]

    def flush_buffer(self, name: str):
        """Return all buffered (timestamp, pose) pairs and clear the buffer."""
        with self._lock:
            data = list(self._buffers[name])
            self._buffers[name].clear()
            return data
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_optitrack_stream.py -v
```

Expected: 4 tests PASS.

### Step 5: Commit

```bash
git add optitrack/__init__.py optitrack/optitrack_stream.py tests/test_optitrack_stream.py
git commit -m "feat: add OptitrackStream for VRPN pose buffering"
```

---

## Task 3: HDF5 episode writer helpers

These helpers will be used by the collection script. Write them as standalone functions in `scripts/twm_data_collection.py` (not a separate module — YAGNI).

**Files:**
- Create: `tests/test_hdf5_writer.py`
- Create: `scripts/twm_data_collection.py` (HDF5 helpers only, no main loop yet)

### Step 1: Write the failing test

Create `tests/test_hdf5_writer.py`:

```python
import unittest
import tempfile
import os
import numpy as np
import h5py

# We'll import the helpers directly from the script
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from twm_data_collection import create_episode_file, append_camera_frame, flush_optitrack_to_hdf5


class TestHDF5Writer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_create_episode_file_structure(self):
        """create_episode_file creates correct HDF5 group/dataset structure."""
        f, path = create_episode_file(
            date_dir=self.tmpdir,
            episode_num=0,
            realsense_serials=["AAA", "BBB", "CCC"],
            gelsight_serials=["2BGLKZNT", "2BKRDTAD"],
            fps=30,
        )
        f.close()

        with h5py.File(path, "r") as f:
            self.assertIn("timestamps", f)
            for i in range(3):
                self.assertIn(f"realsense/cam{i}/color", f)
                self.assertIn(f"realsense/cam{i}/depth", f)
            self.assertIn("gelsight/left/frames", f)
            self.assertIn("gelsight/right/frames", f)
            for name in ["motherboard", "sensor_left", "sensor_right"]:
                self.assertIn(f"optitrack/{name}/timestamps", f)
                self.assertIn(f"optitrack/{name}/pose", f)

    def test_append_camera_frame_grows_datasets(self):
        """append_camera_frame appends data and grows datasets by 1 each call."""
        f, path = create_episode_file(self.tmpdir, 1, ["A", "B", "C"], ["L", "R"], 30)

        color_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        depth_frames = [np.zeros((480, 640), dtype=np.uint16) for _ in range(3)]
        gs_frames    = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]

        append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp=1.0)
        append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp=2.0)

        self.assertEqual(f["timestamps"].shape[0], 2)
        self.assertEqual(f["realsense/cam0/color"].shape[0], 2)
        self.assertEqual(f["gelsight/left/frames"].shape[0], 2)
        f.close()

    def test_flush_optitrack_writes_poses(self):
        """flush_optitrack_to_hdf5 writes all buffered poses to HDF5."""
        f, path = create_episode_file(self.tmpdir, 2, ["A", "B", "C"], ["L", "R"], 30)

        optitrack_data = {
            "motherboard":  [(1.0, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
                             (1.1, [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])],
            "sensor_left":  [(1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
            "sensor_right": [],
        }

        flush_optitrack_to_hdf5(f, optitrack_data)

        self.assertEqual(f["optitrack/motherboard/pose"].shape, (2, 7))
        self.assertEqual(f["optitrack/sensor_left/pose"].shape, (1, 7))
        self.assertEqual(f["optitrack/sensor_right/pose"].shape[0], 0)
        f.close()

    def test_episode_filename_format(self):
        """create_episode_file produces correctly named file."""
        _, path = create_episode_file(self.tmpdir, 5, [], [], 30)
        self.assertTrue(path.endswith("episode_005.h5"))


if __name__ == '__main__':
    unittest.main()
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_hdf5_writer.py -v
```

Expected: `ModuleNotFoundError: No module named 'twm_data_collection'`

### Step 3: Implement HDF5 helpers in `scripts/twm_data_collection.py`

Create `scripts/twm_data_collection.py` with just the helpers for now:

```python
#!/usr/bin/env python3
"""
TWM Data Collection — Tactile World Model multimodal data collection.

Records 3x RealSense D415 (color+depth), 2x GelSight Mini, and OptiTrack
pose data into per-episode HDF5 files. Keyboard-triggered.

Controls:
  s — start new episode
  e — end episode
  q — quit
"""

import os
import time
import numpy as np
import h5py

# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_episode_file(date_dir, episode_num, realsense_serials, gelsight_serials, fps):
    """
    Create a new HDF5 episode file with resizable datasets.

    Returns (h5py.File, path_str). Caller is responsible for closing the file.
    """
    os.makedirs(date_dir, exist_ok=True)
    path = os.path.join(date_dir, f"episode_{episode_num:03d}.h5")
    f = h5py.File(path, "w")

    # metadata
    meta = f.create_group("metadata")
    meta.attrs["fps"] = fps
    meta.attrs["realsense_serials"] = realsense_serials
    meta.attrs["gelsight_serials"]  = gelsight_serials
    meta.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # camera timestamps (one per main-loop tick)
    f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)

    # RealSense color + depth
    for i in range(3):
        g = f.create_group(f"realsense/cam{i}")
        g.create_dataset("color", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                         dtype=np.uint8,  compression="gzip", compression_opts=4,
                         chunks=(1, 480, 640, 3))
        g.create_dataset("depth", shape=(0, 480, 640),    maxshape=(None, 480, 640),
                         dtype=np.uint16, compression="gzip", compression_opts=4,
                         chunks=(1, 480, 640))

    # GelSight
    for name in ["left", "right"]:
        g = f.create_group(f"gelsight/{name}")
        g.create_dataset("frames", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                         dtype=np.uint8, compression="gzip", compression_opts=4,
                         chunks=(1, 480, 640, 3))

    # OptiTrack — per-tracker timestamps + poses
    for name in ["motherboard", "sensor_left", "sensor_right"]:
        g = f.create_group(f"optitrack/{name}")
        g.create_dataset("timestamps", shape=(0,),    maxshape=(None,),    dtype=np.float64)
        g.create_dataset("pose",       shape=(0, 7),  maxshape=(None, 7),  dtype=np.float64)

    return f, path


def append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp):
    """
    Append one timestep of camera data to an open HDF5 file.

    Args:
        f:            open h5py.File
        color_frames: list of 3 numpy arrays (480, 640, 3) uint8
        depth_frames: list of 3 numpy arrays (480, 640) uint16
        gs_frames:    list of 2 numpy arrays (480, 640, 3) uint8  [left, right]
        timestamp:    float, Unix time
    """
    n = f["timestamps"].shape[0]
    f["timestamps"].resize(n + 1, axis=0)
    f["timestamps"][n] = timestamp

    for i, (color, depth) in enumerate(zip(color_frames, depth_frames)):
        ds_c = f[f"realsense/cam{i}/color"]
        ds_d = f[f"realsense/cam{i}/depth"]
        ds_c.resize(n + 1, axis=0);  ds_c[n] = color
        ds_d.resize(n + 1, axis=0);  ds_d[n] = depth

    for name, frame in zip(["left", "right"], gs_frames):
        ds = f[f"gelsight/{name}/frames"]
        ds.resize(n + 1, axis=0);  ds[n] = frame


def flush_optitrack_to_hdf5(f, optitrack_data):
    """
    Write buffered OptiTrack data to HDF5.

    Args:
        f:               open h5py.File
        optitrack_data:  dict mapping tracker name → list of (timestamp, pose) tuples
                         e.g. {"motherboard": [(t, [x,y,z,qx,qy,qz,qw]), ...], ...}
    """
    for name, data in optitrack_data.items():
        if len(data) == 0:
            continue
        timestamps = np.array([d[0] for d in data], dtype=np.float64)
        poses      = np.array([d[1] for d in data], dtype=np.float64)

        ds_t = f[f"optitrack/{name}/timestamps"]
        ds_p = f[f"optitrack/{name}/pose"]
        n = ds_t.shape[0]
        ds_t.resize(n + len(timestamps), axis=0);  ds_t[n:] = timestamps
        ds_p.resize(n + len(poses),      axis=0);  ds_p[n:] = poses


def next_episode_number(date_dir):
    """Return the next available episode number by scanning the date directory."""
    if not os.path.isdir(date_dir):
        return 0
    existing = [f for f in os.listdir(date_dir) if f.startswith("episode_") and f.endswith(".h5")]
    if not existing:
        return 0
    nums = [int(f.replace("episode_", "").replace(".h5", "")) for f in existing]
    return max(nums) + 1
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_hdf5_writer.py -v
```

Expected: 4 tests PASS.

### Step 5: Commit

```bash
git add scripts/twm_data_collection.py tests/test_hdf5_writer.py
git commit -m "feat: add HDF5 episode writer helpers for TWM collection"
```

---

## Task 4: Main collection loop

Complete `scripts/twm_data_collection.py` by adding the sensor initialization, live preview, and keyboard-controlled main loop.

**Files:**
- Modify: `scripts/twm_data_collection.py`

No unit tests for the main loop (hardware-dependent). Verified by running the script.

### Step 1: Append the main loop to `scripts/twm_data_collection.py`

Add the following after the existing HDF5 helpers:

```python
# ──────────────────────────────────────────────────────────────────────────────
# Sensor serials — update these to match your hardware
# ──────────────────────────────────────────────────────────────────────────────
REALSENSE_SERIALS = [
    "REPLACE_WITH_CAM0_SERIAL",   # find with: rs-enumerate-devices
    "REPLACE_WITH_CAM1_SERIAL",
    "REPLACE_WITH_CAM2_SERIAL",
]
GELSIGHT_SERIALS = {
    "left":  "2BGLKZNT",   # /dev/video14
    "right": "2BKRDTAD",   # /dev/video12
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FPS = 30


# ──────────────────────────────────────────────────────────────────────────────
# Preview helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_preview(color_frames, gs_frames, recording, frame_count, elapsed):
    """Build a tiled OpenCV preview image from all camera feeds."""
    import cv2

    thumb_w, thumb_h = 320, 240

    def thumb(img):
        return cv2.resize(img, (thumb_w, thumb_h))

    row1 = np.hstack([thumb(f) for f in color_frames])                     # 3 RealSense
    gs_row = [thumb(f) for f in gs_frames]
    # Pad row2 to same width as row1 (3 × thumb_w) with a black panel
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    row2 = np.hstack(gs_row + [blank])

    preview = np.vstack([row1, row2])

    # Status bar
    if recording:
        status = f"[RECORDING ep_{frame_count // 1:04d} | {frame_count} frames | {elapsed:.1f}s]"
        color = (0, 0, 220)
    else:
        status = "[IDLE]  s=start  e=end  q=quit"
        color = (0, 200, 0)

    cv2.putText(preview, status, (10, preview.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return preview


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import cv2
    from camera_stream.realsense_stream import RealsenseStream
    from camera_stream.usb_video_stream import USBVideoStream
    from optitrack.optitrack_stream import OptitrackStream

    date_str = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(DATA_DIR, date_str)

    # ── Init sensors ──────────────────────────────────────────────────────────
    print("Initializing RealSense cameras...")
    rs_streams = [RealsenseStream(serial=s, fps=FPS) for s in REALSENSE_SERIALS]
    for s in rs_streams:
        s.start()

    print("Initializing GelSight sensors...")
    gs_left  = USBVideoStream(serial=GELSIGHT_SERIALS["left"],  resolution=(640, 480))
    gs_right = USBVideoStream(serial=GELSIGHT_SERIALS["right"], resolution=(640, 480))
    gs_left.start()
    gs_right.start()

    print("Initializing OptiTrack...")
    optitrack = OptitrackStream()
    optitrack.start()

    print("Waiting for first frames from all sensors...")
    for s in rs_streams:
        s.get_color_frame()   # blocks until first frame arrives
    gs_left.get_frame()
    gs_right.get_frame()
    print("All sensors ready.\n")
    print("Controls:  s = start episode   e = end episode   q = quit\n")

    # ── State ─────────────────────────────────────────────────────────────────
    recording   = False
    h5_file     = None
    frame_count = 0
    episode_num = 0
    start_t     = 0.0
    tick_dt     = 1.0 / FPS

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            tick_start = time.time()

            # Grab frames
            color_frames = [s.get_color_frame() for s in rs_streams]
            depth_frames = [s.get_depth_frame() for s in rs_streams]
            gs_frames    = [gs_left.get_frame(), gs_right.get_frame()]
            t            = time.time()

            # Write if recording
            if recording and h5_file is not None:
                append_camera_frame(h5_file, color_frames, depth_frames, gs_frames, t)
                frame_count += 1

            # Preview
            elapsed = t - start_t if recording else 0.0
            preview = make_preview(color_frames, gs_frames, recording, frame_count, elapsed)
            cv2.imshow("TWM Data Collection", preview)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not recording:
                episode_num = next_episode_number(date_dir)
                h5_file, path = create_episode_file(
                    date_dir, episode_num, REALSENSE_SERIALS,
                    list(GELSIGHT_SERIALS.values()), FPS,
                )
                recording   = True
                frame_count = 0
                start_t     = time.time()
                print(f"\nRecording episode {episode_num:03d} → {path}")

            elif key == ord('e') and recording:
                recording = False
                # Flush OptiTrack buffer
                optitrack_data = {
                    name: optitrack.flush_buffer(name)
                    for name in ["motherboard", "sensor_left", "sensor_right"]
                }
                flush_optitrack_to_hdf5(h5_file, optitrack_data)
                h5_file.close()
                h5_file = None
                print(f"\nEpisode {episode_num:03d} saved — {frame_count} frames, "
                      f"{frame_count / FPS:.1f}s")

            elif key == ord('q'):
                if recording and h5_file is not None:
                    print("\nSaving in-progress episode before quit...")
                    recording = False
                    optitrack_data = {
                        name: optitrack.flush_buffer(name)
                        for name in ["motherboard", "sensor_left", "sensor_right"]
                    }
                    flush_optitrack_to_hdf5(h5_file, optitrack_data)
                    h5_file.close()
                break

            # Rate limiting
            sleep_t = tick_dt - (time.time() - tick_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        for s in rs_streams:
            s.stop()
        gs_left.stop()
        gs_right.stop()
        optitrack.stop()
        cv2.destroyAllWindows()
        print("All sensors stopped. Goodbye.")


if __name__ == "__main__":
    main()
```

### Step 2: Find RealSense serial numbers

Before running, find the serial numbers of your 3 cameras:

```bash
rs-enumerate-devices | grep Serial
```

Update `REALSENSE_SERIALS` in `scripts/twm_data_collection.py` with the three serial numbers.

### Step 3: Smoke test — verify sensors initialize

With all hardware connected and ROS/OptiTrack running:

```bash
cd /home/yxma/MultimodalData
python scripts/twm_data_collection.py
```

Expected:
- All 6 sensors initialize without error
- Preview window opens showing 3 color feeds (row 1) and 2 GelSight feeds (row 2)
- Status bar shows `[IDLE]`

### Step 4: Test a full episode

1. Press `s` — confirm terminal prints `Recording episode 000 → data/YYYY-MM-DD/episode_000.h5`
2. Wait ~5 seconds
3. Press `e` — confirm terminal prints episode saved with frame count
4. Verify the HDF5 file:

```bash
python -c "
import h5py, os, glob
files = glob.glob('data/**/*.h5', recursive=True)
f = h5py.File(files[-1], 'r')
print('timestamps shape:', f['timestamps'].shape)
print('cam0 color shape:', f['realsense/cam0/color'].shape)
print('cam0 depth shape:', f['realsense/cam0/depth'].shape)
print('gelsight left  :', f['gelsight/left/frames'].shape)
print('optitrack motherboard pose:', f['optitrack/motherboard/pose'].shape)
f.close()
"
```

Expected: all shapes have N > 0, cam0 color is `(N, 480, 640, 3)`.

### Step 5: Commit

```bash
git add scripts/twm_data_collection.py
git commit -m "feat: add TWM data collection main loop with keyboard control and live preview"
```

---

## Dependencies to install

If not already installed:

```bash
pip install pyrealsense2 h5py
# rospy comes from ROS (already available if OptiTrack topics are working)
```
