# TWM Arducam Sensor-Camera Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record two USB-path-identified Arducam sensor cameras in the TWM HDF5 pipeline, preview and identify them before left/right mapping is known, and verify saved data on real hardware.

**Architecture:** A focused V4L2 stream class handles Arducam capture, while `twm.sensor_camera` owns configuration, topology resolution, mapping, and diagnostic commands. The existing capture loop and batch writer gain optional two-camera payloads, preserving old call signatures and file compatibility; the canonical preview gains an optional third row.

**Tech Stack:** Python 3, OpenCV/V4L2, pyudev, NumPy, h5py/hdf5plugin, unittest/pytest.

---

### Task 1: Stable topology configuration and resolution

**Files:**
- Create: `twm/sensor_camera.py`
- Create: `twm/config/arducam.json`
- Test: `tests/test_arducam.py`

- [x] **Step 1: Write failing configuration and resolution tests**

Test `load_config`, `validate_config`, `enumerate_capture_devices`, and
`resolve_slots` with injected inventories. Cover two unknown positions,
one-to-one left/right, duplicate positions, duplicate paths, missing paths,
duplicate factory serials, and non-capture metadata nodes. Representative
assertions:

```python
cfg = validate_config({"cameras": [
    {"slot": "cam0", "id_path": "usb-A", "position": "unknown"},
    {"slot": "cam1", "id_path": "usb-B", "position": "unknown"},
]})
resolved = resolve_slots(cfg, [
    VideoDevice("/dev/video6", "usb-A", "SN001", True),
    VideoDevice("/dev/video7", "usb-A", "SN001", False),
    VideoDevice("/dev/video10", "usb-B", "SN001", True),
])
assert [x.device for x in resolved] == ["/dev/video6", "/dev/video10"]
```

- [x] **Step 2: Run tests and confirm the missing-module failure**

Run: `python -m pytest tests/test_arducam.py -q`

Expected: collection fails because `twm.sensor_camera` does not exist.

- [x] **Step 3: Implement the configuration model and resolver**

Define immutable `CameraSlot` and `VideoDevice` dataclasses. Validate exactly
`cam0` and `cam1`, unique nonempty `id_path` values, common 640x480/30/MJPG
defaults, and positions either both `unknown` or exactly `{left, right}`.
Enumerate `video4linux` devices with pyudev and accept only devices whose
`ID_V4L_CAPABILITIES` contains `:capture:`. Resolve strictly by `ID_PATH` and
raise `ArducamConfigError` with the discovered inventory on zero or multiple
matches.

Create the current configuration:

```json
{
  "cameras": [
    {"slot": "cam0", "id_path": "pci-0000:00:14.0-usb-0:12.1:1.0", "position": "unknown", "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"},
    {"slot": "cam1", "id_path": "pci-0000:00:14.0-usb-0:12.3:1.0", "position": "unknown", "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"}
  ]
}
```

- [x] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_arducam.py -q`

Expected: all configuration/resolution tests pass.

- [x] **Step 5: Commit**

```bash
git add twm/sensor_camera.py twm/config/arducam.json tests/test_arducam.py
git commit -m "feat(twm): resolve Arducams by stable USB topology"
```

### Task 2: Dedicated Arducam V4L2 stream

**Files:**
- Create: `camera_stream/arducam_video_stream.py`
- Modify: `camera_stream/__init__.py`
- Test: `tests/test_arducam_stream.py`

- [x] **Step 1: Write failing stream tests with a fake VideoCapture**

Test V4L2 opening, requested FOURCC/size/FPS/buffer settings, first-frame
timeout, negotiated-shape rejection, timestamp publication, copied frame
access, idempotent stop, and capture-read failure messages. Inject a
`capture_factory` so no device is needed in unit tests.

```python
stream = ArducamVideoStream(slot, device="/dev/video6", capture_factory=factory)
stream.start()
frame, ts = stream.get_frame_with_timestamp(timeout=0.2)
assert frame.shape == (480, 640, 3)
assert np.isfinite(ts)
stream.stop()
```

- [x] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_arducam_stream.py -q`

Expected: import fails because `ArducamVideoStream` is absent.

- [x] **Step 3: Implement `ArducamVideoStream`**

Open the resolved device with `cv2.CAP_V4L2`; request MJPG, width, height,
FPS, and one buffer. Use a daemon thread that calls `read`, stamps successful
frames immediately, validates exact BGR shape, and publishes frame/timestamp
under a lock. `get_frame_with_timestamp(timeout)` waits on a condition and
raises `TimeoutError` instead of hanging. `stop()` clears the run event, joins
the thread, and releases capture.

- [x] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_arducam_stream.py -q`

Expected: all stream tests pass.

- [x] **Step 5: Commit**

```bash
git add camera_stream/arducam_video_stream.py camera_stream/__init__.py tests/test_arducam_stream.py
git commit -m "feat(twm): add timestamped Arducam V4L2 stream"
```

### Task 3: Backward-compatible HDF5 schema and batch writer

**Files:**
- Modify: `twm/data_collection.py`
- Modify: `tests/test_hdf5_writer.py`

- [x] **Step 1: Write failing schema and write tests**

Add tests proving old calls create no `arducam` group, configured calls create
both camera groups with exact arrays and attributes, batch appends preserve
known pixels and capture timestamps, missing timestamps fall back to tick
time, and Arducam-only diagnostic files can be written without fake legacy
datasets.

```python
f, _ = create_episode_file(tmp, 0, [], [], 30,
    arducam_config=config, include_legacy=False)
append_camera_frames_batch(f, [(None, None, None, 10.0, None,
    [cam0, cam1], [9.9, 9.95])])
assert f["arducam/cam1/timestamps"][0] == 9.95
```

- [x] **Step 2: Run the new tests and observe signature/schema failures**

Run: `python -m pytest tests/test_hdf5_writer.py -q`

Expected: failures for unsupported arguments and missing datasets.

- [x] **Step 3: Extend file creation, batch append, and writer enqueue**

Add optional `arducam_config=None` and `include_legacy=True` parameters to
`create_episode_file`. Add optional tuple elements 5/6 for Arducam frames and
timestamps to `append_camera_frames_batch`. Add optional
`arducam_frames=None, arducam_timestamps=None` to `HDF5Writer.enqueue`.
Create/write both streams only when configured, use existing BLOSC settings,
and preserve every existing positional call.

- [x] **Step 4: Run HDF5 and legacy regression tests**

Run: `python -m pytest tests/test_hdf5_writer.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add twm/data_collection.py tests/test_hdf5_writer.py
git commit -m "feat(twm): store sensor-camera frames and timestamps"
```

### Task 4: Capture-loop and recorder lifecycle integration

**Files:**
- Modify: `twm/data_collection.py`
- Create: `tests/test_twm_capture_loop.py`

- [x] **Step 1: Write failing capture-loop tests**

Use deterministic RealSense, GelSight, Arducam, OptiTrack, and writer doubles.
Verify the latest snapshot includes both sensor-camera frames, recording
enqueues both timestamps, start/stop preserves equal ticks, and construction
without Arducams retains legacy behavior.

- [x] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_twm_capture_loop.py -q`

Expected: constructor and enqueue expectations fail for absent Arducam inputs.

- [x] **Step 3: Integrate optional streams into `CaptureLoop` and `main`**

Load and resolve `twm/config/arducam.json`, start both streams before the
readiness gate, pass them to `CaptureLoop`, capture both timestamped frames on
each tick, include them in latest snapshots and writer enqueues, pass config
to episode creation, and stop both streams in `finally`. Add
`--no_arducam` as the explicit legacy escape hatch; configured missing cameras
otherwise fail before recording rather than producing black frames.

- [x] **Step 4: Run capture-loop and HDF5 regression tests**

Run: `python -m pytest tests/test_twm_capture_loop.py tests/test_hdf5_writer.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add twm/data_collection.py tests/test_twm_capture_loop.py
git commit -m "feat(twm): wire Arducams into recorder lifecycle"
```

### Task 5: Optional preview row

**Files:**
- Modify: `twm/viz.py`
- Modify: `tests/test_visualize.py`

- [x] **Step 1: Write failing preview tests**

Retain the exact 1280x480 shape without Arducams. With two frames and labels,
require 1280x720 output and verify that distinct test colors occupy the first
two 320x240 slots of row three.

- [x] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_visualize.py -q`

Expected: `build_preview_panel` rejects the new keyword arguments.

- [x] **Step 3: Add the optional sensor-camera row**

Add `arducam_frames=None` and `arducam_labels=None` keyword parameters. When
present, require two frames, resize each to 320x240, append two black tiles,
draw labels containing slot/path/position, and stack beneath the old panel.
Pass the current frames and labels from the recorder UI.

- [x] **Step 4: Run visualization and capture tests**

Run: `python -m pytest tests/test_visualize.py tests/test_twm_capture_loop.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add twm/viz.py twm/data_collection.py tests/test_visualize.py
git commit -m "feat(twm): preview both sensor-mounted cameras"
```

### Task 6: Identification and headless verification CLI

**Files:**
- Modify: `twm/sensor_camera.py`
- Modify: `tests/test_arducam.py`

- [x] **Step 1: Write failing mapping and verifier tests**

Test atomic mapping persistence, rejection of duplicate side assignments,
headless recording with deterministic streams, and validation failures for
empty, malformed, black, timestamp-invalid, or byte-identical streams.

- [x] **Step 2: Run tests and confirm missing command failures**

Run: `python -m pytest tests/test_arducam.py -q`

Expected: failures for absent identify/verify helpers.

- [x] **Step 3: Implement `python -m twm.sensor_camera identify`**

Show a side-by-side live preview labeled with slot and `ID_PATH`. Keys `0`
and `1` mark the focused/selected feed as left, automatically assigning the
other right; `u` restores unknown; `s` atomically saves the validated JSON;
`q` exits without writing.

- [x] **Step 4: Implement `python -m twm.sensor_camera verify`**

Resolve and start both real streams, create an Arducam-only HDF5 through
`create_episode_file`, enqueue timed samples through `HDF5Writer`, close and
reopen it, then validate identity, counts, shapes, finite monotonic timestamps,
timestamp span, variance, distinct streams, and distinct-frame cadence. Print
a JSON summary and exit nonzero if any invariant fails.

- [x] **Step 5: Run CLI unit tests**

Run: `python -m pytest tests/test_arducam.py -q`

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
git add twm/sensor_camera.py tests/test_arducam.py
git commit -m "feat(twm): add Arducam identification and recording verifier"
```

### Task 7: Documentation and complete verification

**Files:**
- Modify: `twm/README.md`
- Modify: `docs/superpowers/plans/2026-09-05-twm-arducam-integration.md`

- [x] **Step 1: Document configuration and operator commands**

Document duplicate serial behavior, stable path lookup, unknown-side
recording, `identify`, `verify`, normal recording, `--no_arducam`, HDF5 paths,
and the physical-mapping follow-up.

- [x] **Step 2: Run the focused automated suite**

Run:

```bash
python -m pytest tests/test_arducam.py tests/test_arducam_stream.py \
  tests/test_hdf5_writer.py tests/test_twm_capture_loop.py \
  tests/test_visualize.py -q
```

Expected: all tests pass.

- [x] **Step 3: Run the real two-camera verifier**

Run:

```bash
python -m twm.sensor_camera verify --duration 5 \
  --output /tmp/twm_arducam_verification.h5
```

Expected: exit 0 and JSON with `ok: true`, two distinct devices, positive
equal frame counts, valid shapes/variance/timestamps, distinct images, and
measured cadence near 30 FPS.

- [x] **Step 4: Independently inspect the saved HDF5**

Run a separate Python process that opens the file and prints every Arducam
dataset's shape, dtype, timestamp endpoints, intensity mean/variance, and
attributes. Confirm the evidence matches verifier output.

- [x] **Step 5: Run repository regression tests and syntax checks**

Run:

```bash
python -m compileall -q camera_stream twm tests
python -m pytest tests -q
git diff --check
```

Expected: compilation succeeds, tests pass, and `git diff --check` is silent.
If an unrelated baseline test requires unavailable hardware, record its exact
failure separately; no new focused test may fail.

- [x] **Step 6: Mark this plan complete and commit documentation**

Change every completed checkbox to `[x]`, then run:

```bash
git add twm/README.md docs/superpowers/plans/2026-09-05-twm-arducam-integration.md
git commit -m "docs(twm): document sensor-camera recording workflow"
```
