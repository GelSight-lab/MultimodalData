# Data Collection Robustness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix five correctness/robustness issues found in code review: RealSense SDK buffer aliasing, stale frame detection, OptiTrack pre-episode data, exception-safe HDF5 close, bounded writer queue, USBVideoStream thread exit, and visualizer robustness.

**Architecture:** Targeted fixes only — no structural changes. Each task is self-contained and independently testable. Tasks 1–2 affect data correctness; Tasks 3–5 affect reliability and robustness.

**Tech Stack:** Python threading, h5py, pyrealsense2 (mocked in tests), numpy, OpenCV

---

## Task 1: Fix RealSense SDK buffer aliasing and stale frame detection

**Files:**
- Modify: `camera_stream/realsense_stream.py`
- Modify: `tests/test_realsense_stream.py`

**Problem:**
1. `_update()` calls `np.asanyarray(color.get_data())` outside the lock, storing a *view* into the SDK's internal buffer. The SDK can reclaim that buffer on the next `wait_for_frames()` call, corrupting the stored frame. Fix: copy inside the lock.
2. `get_color_frame()` only waits until `_color_frame is not None`. After the first frame, it returns immediately even if the camera disconnected 10 seconds ago. Fix: check `_last_updated` freshness.

### Step 1: Add failing tests for stale frame detection

Add to `tests/test_realsense_stream.py` (inside the `TestRealsenseStream` class):

```python
def test_get_color_frame_raises_on_stale_frame(self):
    """get_color_frame() raises TimeoutError if the frame is too old."""
    stream = RealsenseStream(serial="123456")
    stream._color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stream._last_updated = __import__('time').time() - 10.0  # 10 seconds stale
    with self.assertRaises(TimeoutError):
        stream.get_color_frame(timeout=0.05)

def test_get_depth_frame_raises_on_stale_frame(self):
    """get_depth_frame() raises TimeoutError if the frame is too old."""
    stream = RealsenseStream(serial="123456")
    stream._depth_frame = np.zeros((480, 640), dtype=np.uint16)
    stream._last_updated = __import__('time').time() - 10.0
    with self.assertRaises(TimeoutError):
        stream.get_depth_frame(timeout=0.05)

def test_get_color_frame_fresh_succeeds(self):
    """get_color_frame() succeeds when _last_updated is recent."""
    stream = RealsenseStream(serial="123456")
    stream._color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stream._last_updated = __import__('time').time()  # just now
    result = stream.get_color_frame()
    self.assertEqual(result.shape, (480, 640, 3))
```

### Step 2: Run tests to verify they fail

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_realsense_stream.py -v
```

Expected: 2 new tests FAIL (TimeoutError not raised), 1 new test FAIL (old code returns without checking freshness).

### Step 3: Fix `_update()` — copy inside the lock

In `camera_stream/realsense_stream.py`, replace:

```python
if color and depth:
    color_arr = np.asanyarray(color.get_data())
    depth_arr = np.asanyarray(depth.get_data())
    with self._lock:
        self._color_frame = color_arr
        self._depth_frame = depth_arr
        self._last_updated = time.time()
```

With:

```python
if color and depth:
    with self._lock:
        self._color_frame = np.asanyarray(color.get_data()).copy()
        self._depth_frame = np.asanyarray(depth.get_data()).copy()
        self._last_updated = time.time()
```

### Step 4: Fix `get_color_frame()` and `get_depth_frame()` — staleness check

Replace both getters with:

```python
def get_color_frame(self, timeout: float = 2.0, max_age: float = 0.5) -> np.ndarray:
    """Block until a fresh color frame is available, then return a copy.

    Raises TimeoutError if no fresh frame arrives within `timeout` seconds.
    A frame is considered stale if it was captured more than `max_age` seconds ago.
    """
    t0 = time.time()
    while True:
        with self._lock:
            if self._color_frame is not None and time.time() - self._last_updated <= max_age:
                return self._color_frame.copy()
        if time.time() - t0 > timeout:
            age = time.time() - self._last_updated
            raise TimeoutError(
                f"No fresh color frame from RealSense serial={self.serial} "
                f"(last updated {age:.2f}s ago)"
            )
        time.sleep(0.01)

def get_depth_frame(self, timeout: float = 2.0, max_age: float = 0.5) -> np.ndarray:
    """Block until a fresh depth frame is available, then return a copy."""
    t0 = time.time()
    while True:
        with self._lock:
            if self._depth_frame is not None and time.time() - self._last_updated <= max_age:
                return self._depth_frame.copy()
        if time.time() - t0 > timeout:
            age = time.time() - self._last_updated
            raise TimeoutError(
                f"No fresh depth frame from RealSense serial={self.serial} "
                f"(last updated {age:.2f}s ago)"
            )
        time.sleep(0.01)
```

### Step 5: Run all tests

```bash
python -m pytest tests/test_realsense_stream.py -v
```

Expected: all 7 tests PASS (4 original + 3 new).

### Step 6: Commit

```bash
git add camera_stream/realsense_stream.py tests/test_realsense_stream.py
git commit -m "fix: copy RealSense frames inside lock and reject stale frames"
```

---

## Task 2: Fix episode lifecycle — OptiTrack pre-episode data and exception-safe HDF5 close

**Files:**
- Modify: `twm/data_collection.py`

**Problem 1:** On `s`, `h5_file` is created but OptiTrack buffers are not cleared. When `e` is pressed, `flush_buffer()` drains the entire deque including poses from before the episode started. Fix: drain (discard) OptiTrack buffers immediately when recording starts.

**Problem 2:** If a Python exception is raised during recording (e.g. a sensor error), the `finally` block stops sensors but does not flush the writer queue or close `h5_file`. This leaves an open/incomplete HDF5 file. Fix: handle the open episode in `finally`.

### Step 1: Clear OptiTrack buffers on episode start

In `twm/data_collection.py`, find the `if key == ord('s')` block. After `recording = True`, add:

```python
recording   = True
# Discard poses that arrived before this episode started
for name in ["motherboard", "sensor_left", "sensor_right"]:
    optitrack.flush_buffer(name)
frame_count = 0
```

The full block after the change:

```python
if key == ord('s') and not recording:
    episode_num = next_episode_number(date_dir)
    h5_file, path = create_episode_file(
        date_dir, episode_num, REALSENSE_SERIALS,
        list(GELSIGHT_SERIALS.values()), FPS,
        task_name=task_name,
    )
    recording   = True
    for name in ["motherboard", "sensor_left", "sensor_right"]:
        optitrack.flush_buffer(name)  # discard pre-episode poses
    frame_count = 0
    start_t     = time.time()
    print(f"\nRecording episode {episode_num:03d} → {path}")
```

### Step 2: Fix the `finally` block to close open episodes

Find the `finally:` block at the end of `main()`. Replace:

```python
finally:
    for s in rs_streams:
        s.stop()
    gs_left.stop()
    gs_right.stop()
    optitrack.stop()
    cv2.destroyAllWindows()
    print("All sensors stopped. Goodbye.")
```

With:

```python
finally:
    if recording and h5_file is not None:
        print("Unexpected exit during recording — saving episode...")
        try:
            writer.flush()
            optitrack_data = {
                name: optitrack.flush_buffer(name)
                for name in ["motherboard", "sensor_left", "sensor_right"]
            }
            flush_optitrack_to_hdf5(h5_file, optitrack_data)
            h5_file.close()
            print(f"Episode {episode_num:03d} saved ({frame_count} frames).")
        except Exception as save_err:
            print(f"Warning: could not save episode cleanly: {save_err}")
    for s in rs_streams:
        s.stop()
    gs_left.stop()
    gs_right.stop()
    optitrack.stop()
    cv2.destroyAllWindows()
    print("All sensors stopped. Goodbye.")
```

### Step 3: Run tests to verify nothing broke

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/ -v
```

Expected: all tests pass (no unit tests cover this path directly, but nothing should regress).

### Step 4: Commit

```bash
git add twm/data_collection.py
git commit -m "fix: clear OptiTrack buffer on episode start; close HDF5 in finally block"
```

---

## Task 3: Bound the HDF5 writer queue

**Files:**
- Modify: `twm/data_collection.py`

**Problem:** `queue.Queue()` is unbounded. If the writer thread falls behind (slow disk), the queue grows without limit. At ~6 MB per frame this can exhaust memory quickly. With `maxsize=150` (~5 seconds of buffer at 30 Hz), `queue.put()` will block naturally when full, slowing the main loop instead of OOMing.

### Step 1: Add a maxsize parameter to HDF5Writer

In `twm/data_collection.py`, find `class HDF5Writer` and change `__init__`:

```python
def __init__(self, maxsize: int = 150):
    self._queue = queue.Queue(maxsize=maxsize)
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
```

No other changes needed — `queue.Queue.put()` blocks by default when the queue is full.

### Step 2: Run tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

### Step 3: Commit

```bash
git add twm/data_collection.py
git commit -m "fix: bound HDF5 writer queue to 150 frames to prevent OOM on slow disk"
```

---

## Task 4: Fix USBVideoStream thread exit and assert

**Files:**
- Modify: `camera_stream/usb_video_stream.py`

**Problem 1:** The `update()` thread is started without `daemon=True` and loops `while True:` — it never exits. `stop()` sets `self.streaming = False` which pauses the loop but the thread stays alive, blocking Python from exiting cleanly.

**Problem 2:** `assert len(matching_devices) > 0` is used for runtime validation. Python `-O` strips assertions silently, so a missing device would produce no error. Should be a `RuntimeError`.

### Step 1: Make the thread daemon and exit cleanly on stop

In `camera_stream/usb_video_stream.py`, change `start()`:

```python
if create_thread:
    threading.Thread(target=self.update, args=(), daemon=True).start()
```

Change `update()` — replace `while True:` with `while self.streaming:` and add an inner check so the rate-limiting loop also exits promptly:

```python
def update(self):
    while self.streaming:
        while self.streaming and time.time() - self.last_updated < 1.0 / self.fps:
            time.sleep(0.001)
        if not self.streaming:
            break
        if not self.streaming:
            return
        grabbed, frame = None, None
        try:
            grabbed, frame = self.stream.read()
        except Exception as e:
            print(e)
            print("Error reading frame. Trying to ignore...")
            continue
        if grabbed:
            if self.resolution != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, self.resolution)
            if self.format == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = frame
            self.write_frame(frame)
            self.last_updated = time.time()
        else:
            time.sleep(0.01)
```

### Step 2: Replace assert with RuntimeError

In `parse_serial()`, replace:

```python
assert len(matching_devices) > 0, "No matching device found with serial: {}".format(serial)
```

With:

```python
if len(matching_devices) == 0:
    raise RuntimeError("No matching device found with serial: {}".format(serial))
```

### Step 3: Run tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

### Step 4: Commit

```bash
git add camera_stream/usb_video_stream.py
git commit -m "fix: USBVideoStream thread exits cleanly on stop; replace assert with RuntimeError"
```

---

## Task 5: Fix visualizer robustness

**Files:**
- Modify: `twm/visualize.py`
- Create: `tests/test_visualize.py`

**Problem 1:** `optitrack_at()` uses `np.argmin(np.abs(ts - camera_timestamp))` — O(N) linear scan over all timestamps. For a 10-minute episode at 120 Hz OptiTrack that's ~72,000 entries scanned per display frame per tracker.

**Problem 2:** `gs_ref = [f["gelsight/left/frames"][0], ...]` crashes with `IndexError` on partial episodes where GelSight data is empty (0 frames).

### Step 1: Write failing test for optitrack_at

Create `tests/test_visualize.py`:

```python
import unittest
import numpy as np
import sys
import os

# Mock h5py and cv2 so we can import without hardware
sys.modules['h5py'] = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock()
sys.modules['cv2'] = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from twm.visualize import optitrack_at


class TestOptitrackAt(unittest.TestCase):

    def _make_lookup(self, timestamps, poses):
        return {"tracker": (np.array(timestamps), np.array(poses))}

    def test_exact_match(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 2.0)
        self.assertAlmostEqual(result["tracker"][0], 2.0)

    def test_nearest_before(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 1.4)
        self.assertAlmostEqual(result["tracker"][0], 1.0)

    def test_nearest_after(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 1.6)
        self.assertAlmostEqual(result["tracker"][0], 2.0)

    def test_before_first(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 0.5)
        self.assertAlmostEqual(result["tracker"][0], 1.0)

    def test_after_last(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 5.0)
        self.assertAlmostEqual(result["tracker"][0], 3.0)

    def test_none_when_no_data(self):
        lookup = {"tracker": None}
        result = optitrack_at(lookup, 1.0)
        self.assertIsNone(result["tracker"])


if __name__ == '__main__':
    unittest.main()
```

### Step 2: Run to verify tests fail

```bash
python -m pytest tests/test_visualize.py -v
```

Expected: tests pass with argmin (argmin and searchsorted produce the same results for these cases), but this establishes the contract for the refactor.

### Step 3: Replace argmin with searchsorted in `optitrack_at`

In `twm/visualize.py`, replace:

```python
def optitrack_at(lookup, camera_timestamp):
    """Return nearest OptiTrack pose for each tracker at the given camera timestamp."""
    result = {}
    for name, data in lookup.items():
        if data is None:
            result[name] = None
            continue
        ts, poses = data
        idx = int(np.argmin(np.abs(ts - camera_timestamp)))
        result[name] = (float(ts[idx]), poses[idx].tolist())
    return result
```

With:

```python
def optitrack_at(lookup, camera_timestamp):
    """Return nearest OptiTrack pose for each tracker at the given camera timestamp."""
    result = {}
    for name, data in lookup.items():
        if data is None:
            result[name] = None
            continue
        ts, poses = data
        idx = int(np.searchsorted(ts, camera_timestamp))
        if idx == 0:
            pass  # clamp to first
        elif idx >= len(ts):
            idx = len(ts) - 1  # clamp to last
        elif abs(ts[idx - 1] - camera_timestamp) <= abs(ts[idx] - camera_timestamp):
            idx -= 1  # left neighbor is closer
        result[name] = (float(ts[idx]), poses[idx].tolist())
    return result
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_visualize.py -v
```

Expected: 6 tests PASS.

### Step 5: Guard against empty GelSight datasets

In `twm/visualize.py`, find the block that reads `gs_ref` (around line 127):

```python
# GelSight diff reference starts at frame 0
gs_ref = [
    f["gelsight/left/frames"][0].copy(),
    f["gelsight/right/frames"][0].copy(),
]
```

Replace with:

```python
# GelSight diff reference — use frame 0 if available, else blank grey
_blank_gs = np.full((480, 640, 3), 128, dtype=np.uint8)
gs_left_n  = int(f["gelsight/left/frames"].shape[0])
gs_right_n = int(f["gelsight/right/frames"].shape[0])
gs_ref = [
    f["gelsight/left/frames"][0].copy()  if gs_left_n  > 0 else _blank_gs.copy(),
    f["gelsight/right/frames"][0].copy() if gs_right_n > 0 else _blank_gs.copy(),
]
```

Then find where GelSight frames are read in the playback loop. Look for lines like:
```python
f["gelsight/left/frames"][i]
```
and replace with:
```python
f["gelsight/left/frames"][min(i, gs_left_n - 1)].copy()  if gs_left_n  > 0 else _blank_gs.copy()
f["gelsight/right/frames"][min(i, gs_right_n - 1)].copy() if gs_right_n > 0 else _blank_gs.copy()
```

### Step 6: Run full test suite

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

### Step 7: Commit

```bash
git add twm/visualize.py tests/test_visualize.py
git commit -m "fix: replace argmin with searchsorted in visualizer; guard empty GelSight datasets"
```
