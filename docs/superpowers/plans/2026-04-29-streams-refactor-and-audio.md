# Streams refactor + contact-mic audio integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `camera_stream/`, `ft_sensor/`, `optitrack/`, `audio_stream/` into a single `streams/` package, fix three known bugs, and add 2-channel contact-microphone capture (1 kHz/mic) to `twm/data_collection.py`.

**Architecture:** Audio mirrors the OptiTrack pattern — a background producer thread per mic buffers `(timestamp, sample)` tuples, drained at episode boundaries and written under `audio/{left,right}/{timestamps,samples}` in HDF5. The `streams/` package gives every modality the same shape: `<modality>/base.py` + `<vendor>.py` + optional `<vendor>/` subfolder for vendor SDK / examples.

**Tech Stack:** Python 3.8+, h5py + hdf5plugin (BLOSC LZ4), pyserial, opencv-python, rospy (via mocks in tests), numpy.

**Spec:** `docs/superpowers/specs/2026-04-29-streams-refactor-and-audio-design.md`

---

## File structure (created/modified)

**Created:**
- `streams/__init__.py`
- `streams/base.py`
- `streams/camera/__init__.py`, `streams/camera/base.py`, `streams/camera/realsense.py`, `streams/camera/usb.py`, `streams/camera/digit.py`, `streams/camera/raspi.py` (moved from `camera_stream/`)
- `streams/optitrack/__init__.py`, `streams/optitrack/base.py` (NEW), `streams/optitrack/motive.py` (moved)
- `streams/ft_sensor/__init__.py`, `streams/ft_sensor/base.py`, `streams/ft_sensor/mms101.py`, `streams/ft_sensor/mms101/{example.py,mms101.py,README.md}` (moved)
- `streams/audio/__init__.py`, `streams/audio/base.py` (NEW), `streams/audio/contact_mic.py` (NEW), `streams/audio/contact_mic/viz.py` (moved), `streams/audio/contact_mic/README.md` (NEW)
- `tests/test_audio_stream.py` (NEW)
- `twm/episode_io.py` (extracted)
- `twm/preview.py` (extracted)
- `twm/sensors.py` (extracted)

**Modified:**
- `twm/data_collection.py` — imports, audio wiring, decomposition
- `twm/visualize.py` — import path for `make_preview`
- `twm/visualize_projection.py` — import paths
- `tests/test_hdf5_writer.py` — imports + new audio assertions
- `tests/test_realsense_stream.py` — imports
- `tests/test_optitrack_stream.py` — imports
- `probing_panda/scripts/example_gelsight_stream.py` — imports
- `probing_panda/scripts/find_gelsight_sensors.py` — print-string update
- `pyproject.toml` — package list

**Deleted (after moves):**
- `camera_stream/`, `optitrack/`, `ft_sensor/`, `audio_stream/`

---

## Task 1: Create `streams/` package skeleton

**Files:**
- Create: `streams/__init__.py`
- Create: `streams/base.py`

- [ ] **Step 1: Create empty package init**

`streams/__init__.py`:
```python
```
(Empty file — subpackages are imported explicitly by callers.)

- [ ] **Step 2: Create marker base class**

`streams/base.py`:
```python
"""Marker base class for all sensor stream types.

Concrete modalities (camera, audio, ft_sensor, optitrack) each define their
own richer base class with modality-specific reader API. This shared base
exists only for `start()`/`stop()` lifecycle and isinstance checks.
"""


class BaseStream:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 3: Verify Python can import the package**

```bash
cd /home/yxma/MultimodalData
python -c "from streams.base import BaseStream; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add streams/__init__.py streams/base.py
git commit -m "feat(streams): create empty package skeleton"
```

---

## Task 2: Move `camera_stream/` → `streams/camera/`

**Files:**
- Move/rename: `camera_stream/base_video_stream.py` → `streams/camera/base.py`
- Move/rename: `camera_stream/realsense_stream.py` → `streams/camera/realsense.py`
- Move/rename: `camera_stream/usb_video_stream.py` → `streams/camera/usb.py`
- Move/rename: `camera_stream/digit_video_stream.py` → `streams/camera/digit.py`
- Move/rename: `camera_stream/raspi_video_stream.py` → `streams/camera/raspi.py`
- Modify: `streams/camera/__init__.py` (created during move)
- Modify: `twm/data_collection.py:375-376`
- Modify: `tests/test_realsense_stream.py:10`
- Modify: `probing_panda/scripts/example_gelsight_stream.py:16`
- Modify: `probing_panda/scripts/find_gelsight_sensors.py:68`

- [ ] **Step 1: Create the new directory and move files with `git mv`**

```bash
cd /home/yxma/MultimodalData
mkdir -p streams/camera
git mv camera_stream/base_video_stream.py streams/camera/base.py
git mv camera_stream/realsense_stream.py streams/camera/realsense.py
git mv camera_stream/usb_video_stream.py streams/camera/usb.py
git mv camera_stream/digit_video_stream.py streams/camera/digit.py
git mv camera_stream/raspi_video_stream.py streams/camera/raspi.py
rm -rf camera_stream/__pycache__
git rm camera_stream/__init__.py
rmdir camera_stream
```

- [ ] **Step 2: Fix internal imports inside moved camera files**

Each moved file imports `from misc.utils import logging` (or similar). Verify these still work — `misc/` stays at repo root, so the imports are unchanged. Open each of the five files and confirm. If any uses `from camera_stream.base_video_stream import BaseVideoStream`, rewrite to `from streams.camera.base import BaseVideoStream`.

```bash
grep -n "camera_stream\|from \.base_video_stream\|from \.\." streams/camera/*.py || echo "no matches — clean"
```

If any matches found in the four concrete files (`realsense.py`, `usb.py`, `digit.py`, `raspi.py`), rewrite import lines to use the new paths.

- [ ] **Step 3: Write the new `streams/camera/__init__.py`**

`streams/camera/__init__.py`:
```python
from .base import BaseVideoStream
from .realsense import RealsenseStream
from .usb import USBVideoStream
from .digit import DigitVideoStream
from .raspi import RaspiVideoStream

__all__ = [
    "BaseVideoStream",
    "RealsenseStream",
    "USBVideoStream",
    "DigitVideoStream",
    "RaspiVideoStream",
]
```

No `try/except ImportError: print(...)` — these are in-package modules and a broken import is a real bug.

- [ ] **Step 4: Update callsite in `twm/data_collection.py`**

In `twm/data_collection.py` lines 375-376, change:
```python
    from camera_stream.realsense_stream import RealsenseStream
    from camera_stream.usb_video_stream import USBVideoStream
```
to:
```python
    from streams.camera import RealsenseStream, USBVideoStream
```

- [ ] **Step 5: Update callsite in `tests/test_realsense_stream.py:10`**

Change:
```python
from camera_stream.realsense_stream import RealsenseStream
```
to:
```python
from streams.camera import RealsenseStream
```

- [ ] **Step 6: Update callsite in `probing_panda/scripts/example_gelsight_stream.py:16`**

Change:
```python
from camera_stream.usb_video_stream import USBVideoStream
```
to:
```python
from streams.camera import USBVideoStream
```

- [ ] **Step 7: Update print string in `probing_panda/scripts/find_gelsight_sensors.py:68`**

Change:
```python
    print("from camera_stream.usb_video_stream import USBVideoStream")
```
to:
```python
    print("from streams.camera import USBVideoStream")
```

- [ ] **Step 8: Run camera test**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_realsense_stream.py -v
```
Expected: all existing tests pass.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor(streams): move camera_stream/ to streams/camera/"
```

---

## Task 3: Move `optitrack/` → `streams/optitrack/` + thin base ABC

**Files:**
- Create: `streams/optitrack/__init__.py`
- Create: `streams/optitrack/base.py`
- Move/rename: `optitrack/optitrack_stream.py` → `streams/optitrack/motive.py`
- Modify: `twm/data_collection.py:377`
- Modify: `tests/test_optitrack_stream.py:11`

- [ ] **Step 1: Move file with `git mv`**

```bash
cd /home/yxma/MultimodalData
mkdir -p streams/optitrack
git mv optitrack/optitrack_stream.py streams/optitrack/motive.py
rm -rf optitrack/__pycache__
git rm optitrack/__init__.py
rmdir optitrack
```

- [ ] **Step 2: Create `streams/optitrack/base.py`**

`streams/optitrack/base.py`:
```python
"""Abstract base for OptiTrack stream sources.

Concrete impl currently: `MotiveOptitrackStream` (NatNet via VRPN ROS topic).
Future impls (e.g. direct NatNet UDP) would subclass this.
"""

from streams.base import BaseStream


class BaseOptitrackStream(BaseStream):
    def get_latest_pose(self, name: str):
        """Return (timestamp, [x,y,z,qx,qy,qz,qw]) or None if no data yet."""
        raise NotImplementedError

    def flush_buffer(self, name: str):
        """Return all buffered (timestamp, pose) pairs and clear the buffer."""
        raise NotImplementedError
```

- [ ] **Step 3: Update `streams/optitrack/motive.py` to inherit from base**

Open `streams/optitrack/motive.py` and change the class declaration:

From:
```python
class OptitrackStream:
```
to:
```python
from streams.optitrack.base import BaseOptitrackStream


class OptitrackStream(BaseOptitrackStream):
```

(Add the import near the top, alongside the existing `import threading` / `from collections import deque` block.)

Class name stays `OptitrackStream` per spec §4.3.

- [ ] **Step 4: Write `streams/optitrack/__init__.py`**

`streams/optitrack/__init__.py`:
```python
from .base import BaseOptitrackStream
from .motive import OptitrackStream

__all__ = ["BaseOptitrackStream", "OptitrackStream"]
```

- [ ] **Step 5: Update callsite in `twm/data_collection.py:377`**

Change:
```python
    from optitrack.optitrack_stream import OptitrackStream
```
to:
```python
    from streams.optitrack import OptitrackStream
```

- [ ] **Step 6: Update callsite in `tests/test_optitrack_stream.py:11`**

Change:
```python
from optitrack.optitrack_stream import OptitrackStream
```
to:
```python
from streams.optitrack import OptitrackStream
```

The `sys.modules['rospy'] = MagicMock()` setup above the import does not need changes — it mocks by module name, which is unaffected by file relocation.

- [ ] **Step 7: Run optitrack tests**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_optitrack_stream.py -v
```
Expected: all existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(streams): move optitrack/ to streams/optitrack/ + add base ABC"
```

---

## Task 4: Move `ft_sensor/` → `streams/ft_sensor/` + fix wrong-name imports

**Files:**
- Move: `ft_sensor/base_ft_stream.py` → `streams/ft_sensor/base.py`
- Move: `ft_sensor/mms101_stream.py` → `streams/ft_sensor/mms101.py`
- Move: `ft_sensor/mms101/` → `streams/ft_sensor/mms101/` (vendor folder, verbatim)
- Modify: `streams/ft_sensor/__init__.py`

- [ ] **Step 1: Move with `git mv`**

```bash
cd /home/yxma/MultimodalData
mkdir -p streams/ft_sensor
git mv ft_sensor/base_ft_stream.py streams/ft_sensor/base.py
git mv ft_sensor/mms101_stream.py streams/ft_sensor/mms101.py
git mv ft_sensor/mms101 streams/ft_sensor/mms101
rm -rf ft_sensor/__pycache__
git rm ft_sensor/__init__.py
rmdir ft_sensor
```

- [ ] **Step 2: Fix the `from .base_ft_stream` reference in `mms101.py`**

`streams/ft_sensor/mms101.py` line 9 currently reads:
```python
from .base_ft_stream import BaseFTStream
```
Change to:
```python
from .base import BaseFTStream
```

- [ ] **Step 3: Write the corrected `streams/ft_sensor/__init__.py`**

`streams/ft_sensor/__init__.py`:
```python
from .base import BaseFTStream
from .mms101 import MMS101FTStream

__all__ = ["BaseFTStream", "MMS101FTStream"]
```

The previous `__init__.py` imported `Base_FT_Stream` and `MMS101_FT_Stream` (both wrong — actual classes are `BaseFTStream` and `MMS101FTStream`), wrapped in `try/except ImportError: print(...)`. Both the names and the silent-swallow are fixed.

- [ ] **Step 4: Verify the package imports cleanly**

```bash
cd /home/yxma/MultimodalData
python -c "from streams.ft_sensor import BaseFTStream, MMS101FTStream; print('ok')"
```
Expected: `ok` (this exercises the import surface; `MMS101FTStream` will not actually try to open the serial port until you instantiate it).

If this fails because `MMS101FTStream.__init__` tries to open a serial port at module-load time — it doesn't (look at `mms101.py:55` — `self.serialPortOpen()` runs in `__init__`, not at import). The import itself should succeed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(streams): move ft_sensor/ to streams/ft_sensor/ + fix __init__ imports"
```

---

## Task 5: Fix `MMS101FTStream` Timer + reader-loop bug

**Files:**
- Modify: `streams/ft_sensor/mms101.py:242-250`

- [ ] **Step 1: Read the current broken implementation**

Open `streams/ft_sensor/mms101.py`. Lines ~242-250 currently:
```python
    def start(self):
        self.initialize()

        def _update_ft(sensor):
            sensor.get_ft()
            sensor.medfilt_ft()

        self.thread = threading.Timer(0.001, target=_update_ft, args=[self])
        self.thread.start()
```

Bugs: `Timer.__init__` does not accept `target=`; `Timer` fires once after the delay then exits; the loop never runs.

- [ ] **Step 2: Replace with daemon-thread reader loop**

Replace lines 242-250 with:
```python
    def _reader_loop(self):
        while self._running:
            self.get_ft()
            self.medfilt_ft()

    def start(self):
        self.initialize()
        self._running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.serialPortClose()
```

- [ ] **Step 3: Initialize `_running = False` in `__init__`**

In `MMS101FTStream.__init__` (immediately after `super().__init__(...)`), add:
```python
        self._running = False
```

Place it adjacent to the existing `self.thread = None` style attributes, near the top of `__init__` after the `super()` call.

- [ ] **Step 4: Verify the module still imports**

```bash
cd /home/yxma/MultimodalData
python -c "from streams.ft_sensor import MMS101FTStream; print('ok')"
```
Expected: `ok`

(No bench test added this round — FT is not wired into `twm/data_collection.py` per spec §1, FT-ii. Bench verification is the next round's responsibility when FT is wired in.)

- [ ] **Step 5: Commit**

```bash
git add streams/ft_sensor/mms101.py
git commit -m "fix(ft_sensor): replace broken Timer with daemon reader thread"
```

---

## Task 6: Update `pyproject.toml` package list + clean stale dirs

**Files:**
- Modify: `pyproject.toml:21`

- [ ] **Step 1: Update package list**

In `pyproject.toml`, change:
```toml
[tool.setuptools]
packages = ["probing_panda", "camera_stream", "optitrack", "ft_sensor", "misc", "twm"]
```
to:
```toml
[tool.setuptools]
packages = [
    "probing_panda",
    "streams",
    "streams.camera",
    "streams.optitrack",
    "streams.ft_sensor",
    "streams.audio",
    "misc",
    "twm",
]
```

The `audio` subpackage is listed now even though Task 7 creates its content; the package declaration is forward-compatible and harmless.

- [ ] **Step 2: Verify editable install still resolves the new packages**

```bash
cd /home/yxma/MultimodalData
python -c "import streams; import streams.camera; import streams.optitrack; import streams.ft_sensor; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update pyproject.toml package list for streams/ refactor"
```

---

## Task 7: Implement `BaseAudioStream`

**Files:**
- Create: `streams/audio/__init__.py`
- Create: `streams/audio/base.py`

- [ ] **Step 1: Write `streams/audio/base.py`**

`streams/audio/base.py`:
```python
"""Abstract base for audio stream sources.

Concrete impl: `ContactMicStream` (Arduino USB-serial). Future impls
(USB sound card, I²S, network mic) subclass this with no caller-side changes.

Interface mirrors `OptitrackStream`'s shape: a background producer thread
buffers `(timestamp, sample)` tuples, drained at episode boundaries via
`flush_buffer()`.
"""

import collections
import threading

import numpy as np

from streams.base import BaseStream


class BaseAudioStream(BaseStream):
    def __init__(self, channel_name: str, sample_rate_hz: int, dtype=np.int16):
        self.channel_name = channel_name
        self.sample_rate_hz = sample_rate_hz
        self.dtype = dtype
        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def flush_buffer(self) -> list:
        """Atomically swap & return all buffered (t, sample) tuples."""
        with self._lock:
            out, self._buffer = list(self._buffer), collections.deque()
        return out

    def get_recent_samples(self, n: int) -> np.ndarray:
        """Non-destructive — for live preview level meter. Returns up to last n samples."""
        with self._lock:
            tail = list(self._buffer)[-n:]
        if not tail:
            return np.zeros(0, dtype=self.dtype)
        return np.fromiter((s for _, s in tail), dtype=self.dtype, count=len(tail))
```

- [ ] **Step 2: Stub `streams/audio/__init__.py` (will be filled in Task 8)**

`streams/audio/__init__.py`:
```python
from .base import BaseAudioStream

__all__ = ["BaseAudioStream"]
```

- [ ] **Step 3: Verify import**

```bash
cd /home/yxma/MultimodalData
python -c "from streams.audio import BaseAudioStream; b = BaseAudioStream('left', 1000); print(b.channel_name, b.sample_rate_hz)"
```
Expected: `left 1000`

- [ ] **Step 4: Commit**

```bash
git add streams/audio/__init__.py streams/audio/base.py
git commit -m "feat(audio): add BaseAudioStream"
```

---

## Task 8: Implement `ContactMicStream` with TDD

**Files:**
- Create: `tests/test_audio_stream.py`
- Create: `streams/audio/contact_mic.py`
- Modify: `streams/audio/__init__.py`

- [ ] **Step 1: Write the failing test**

`tests/test_audio_stream.py`:
```python
import threading
import time
import unittest

import serial

from streams.audio import ContactMicStream


def _spawn_writer(port_url: str, lines: list[str], delay_s: float = 0.001):
    """Open the writer end of a `loop://` serial url and pump lines into it."""
    writer = serial.serial_for_url(port_url, timeout=0)
    def run():
        for line in lines:
            writer.write((line + "\n").encode("utf-8"))
            time.sleep(delay_s)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return writer, t


class TestContactMicStream(unittest.TestCase):

    def test_buffers_parsed_int_samples(self):
        """ContactMicStream parses ints from the serial port and buffers them with timestamps."""
        url = "loop://"
        # When we instantiate ContactMicStream(device_path=url), it opens
        # the same `loop://` URL — both ends share an in-memory pipe.
        stream = ContactMicStream("left", url, sample_rate_hz=1000)
        stream.start()

        # Use stream's own port to write fake samples (loop:// is bidirectional)
        for value in [123, 456, 789]:
            stream._serial.write(f"{value}\n".encode("utf-8"))
        time.sleep(0.1)

        samples = stream.flush_buffer()
        stream.stop()

        values = [s for _, s in samples]
        self.assertIn(123, values)
        self.assertIn(456, values)
        self.assertIn(789, values)

    def test_skips_malformed_lines(self):
        """Non-integer lines are skipped without crashing."""
        stream = ContactMicStream("left", "loop://", sample_rate_hz=1000)
        stream.start()

        stream._serial.write(b"42\nNOT-AN-INT\n99\n")
        time.sleep(0.1)

        samples = stream.flush_buffer()
        stream.stop()

        values = [s for _, s in samples]
        self.assertIn(42, values)
        self.assertIn(99, values)
        self.assertNotIn("NOT-AN-INT", values)

    def test_stop_joins_reader_thread(self):
        """stop() cleanly joins the producer thread."""
        stream = ContactMicStream("left", "loop://", sample_rate_hz=1000)
        stream.start()
        time.sleep(0.05)
        stream.stop()
        self.assertFalse(stream.thread.is_alive())

    def test_get_recent_samples_returns_tail(self):
        """get_recent_samples returns the last N samples without consuming them."""
        stream = ContactMicStream("left", "loop://", sample_rate_hz=1000)
        stream.start()

        for value in [10, 20, 30, 40, 50]:
            stream._serial.write(f"{value}\n".encode("utf-8"))
        time.sleep(0.1)

        recent = stream.get_recent_samples(n=3)
        self.assertEqual(len(recent), 3)
        # Buffer should still contain all 5 — non-destructive
        all_samples = stream.flush_buffer()
        stream.stop()
        self.assertEqual(len(all_samples), 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_audio_stream.py -v
```
Expected: `ImportError: cannot import name 'ContactMicStream' from 'streams.audio'`

- [ ] **Step 3: Implement `streams/audio/contact_mic.py`**

`streams/audio/contact_mic.py`:
```python
"""Contact microphone stream — Arduino USB-serial source.

Each Arduino runs a tight `analogRead` loop and prints a single int per line
(0–1023 for a 10-bit ADC) at ~1 kHz. This module spawns a daemon thread that
reads as fast as possible, parses the int, timestamps it with `time.time()`,
and buffers `(t, sample)` tuples for later flush.

For first-time labeling, see `list_arduino_devices()`.
"""

import glob
import os
import threading
import time

import serial

from streams.audio.base import BaseAudioStream


class ContactMicStream(BaseAudioStream):
    """One contact microphone via Arduino-style USB-serial."""

    def __init__(
        self,
        channel_name: str,
        device_path: str,
        baud: int = 115200,
        sample_rate_hz: int = 1000,
    ):
        super().__init__(channel_name=channel_name, sample_rate_hz=sample_rate_hz)
        self.device_path = device_path
        self.baud = baud
        self._serial = None
        self.thread = None

    def start(self) -> None:
        self._serial = serial.serial_for_url(
            self.device_path, baudrate=self.baud, timeout=0.1
        )
        self._running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _reader_loop(self) -> None:
        backoff_s = 0.5
        while self._running:
            try:
                raw = self._serial.readline()
            except serial.SerialException as e:
                print(f"[ContactMicStream:{self.channel_name}] serial error: {e}; retrying in {backoff_s}s")
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 5.0)
                continue
            if not raw:
                continue
            backoff_s = 0.5
            try:
                value = int(raw.decode("utf-8", errors="ignore").strip())
            except (UnicodeDecodeError, ValueError):
                continue
            t = time.time()
            with self._lock:
                self._buffer.append((t, value))


def list_arduino_devices() -> list:
    """List stable USB-serial paths matching Arduino-style devices.

    Returns paths under `/dev/serial/by-id/` whose names contain 'Arduino'
    or 'ttyACM' or 'ttyUSB'. Use this for first-time per-board labeling.
    """
    candidates = []
    by_id = "/dev/serial/by-id"
    if os.path.isdir(by_id):
        for entry in sorted(os.listdir(by_id)):
            lower = entry.lower()
            if "arduino" in lower or "ftdi" in lower or "ch340" in lower:
                candidates.append(os.path.join(by_id, entry))
    if not candidates:
        # Fall back to the unstable enumeration so first-run users can see *something*
        candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return candidates
```

- [ ] **Step 4: Update `streams/audio/__init__.py` to export `ContactMicStream`**

`streams/audio/__init__.py`:
```python
from .base import BaseAudioStream
from .contact_mic import ContactMicStream, list_arduino_devices

__all__ = ["BaseAudioStream", "ContactMicStream", "list_arduino_devices"]
```

- [ ] **Step 5: Run tests to verify passing**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_audio_stream.py -v
```
Expected: 4 tests pass.

If `test_buffers_parsed_int_samples` fails because `loop://` writes don't propagate to reads on the same `Serial` instance, switch the test to use two ends:

```python
# Alternative pattern — replace stream._serial.write(...) blocks with:
stream._serial.write(b"123\n")  # `loop://` is one-end loopback; writes do echo to reads
```
The pyserial `loop://` URL is documented as bidirectional loopback on the *same* Serial — writes feed back into reads on the same instance. If your pyserial version differs, use `serial.serial_for_url("loop://", ...)` for both ends and pass the writer's `device_path` argument to a different fixture. Adjust as needed; the contract being tested is "ints arrive, malformed skipped, stop joins."

- [ ] **Step 6: Commit**

```bash
git add tests/test_audio_stream.py streams/audio/contact_mic.py streams/audio/__init__.py
git commit -m "feat(audio): add ContactMicStream with TDD coverage"
```

---

## Task 9: Move `audio_stream/viz.py` into the new module + add README

**Files:**
- Move: `audio_stream/viz.py` → `streams/audio/contact_mic/viz.py`
- Create: `streams/audio/contact_mic/__init__.py` (empty — `viz.py` is a script, not a module)
- Create: `streams/audio/contact_mic/README.md`

- [ ] **Step 1: Move viz.py with `git mv`**

```bash
cd /home/yxma/MultimodalData
mkdir -p streams/audio/contact_mic
git mv audio_stream/viz.py streams/audio/contact_mic/viz.py
rmdir audio_stream
```

The `audio_stream/` directory only contains `viz.py` and is now empty.

- [ ] **Step 2: Add empty `__init__.py` for the vendor subfolder**

`streams/audio/contact_mic/__init__.py`:
```python
```
(Empty — `viz.py` is a standalone script, not imported as a module. The empty `__init__.py` exists only so the folder doesn't shadow `streams/audio/contact_mic.py`.)

Wait — there's a name collision: `streams/audio/contact_mic.py` (module) and `streams/audio/contact_mic/` (folder). Python will prefer the package (folder) over the module if both exist. Resolution: rename the **folder** to `contact_mic_vendor/` to avoid ambiguity.

- [ ] **Step 3: Resolve the name collision (rename the vendor folder)**

```bash
cd /home/yxma/MultimodalData
git mv streams/audio/contact_mic streams/audio/contact_mic_vendor
```

- [ ] **Step 4: Write `streams/audio/contact_mic_vendor/README.md`**

`streams/audio/contact_mic_vendor/README.md`:
````markdown
# Contact microphone — vendor reference

This folder contains the original standalone visualization script
(`viz.py`) used to verify that an Arduino contact-mic rig is producing
data. It is **not** imported by the rest of the project; it exists as
a quick-bench tool.

## First-time setup

1. Flash an Arduino with a sketch that does:
   ```c
   void setup() { Serial.begin(115200); }
   void loop()  { Serial.println(analogRead(A0)); }
   ```
2. Plug it in. Find its stable path:
   ```bash
   python -c "from streams.audio import list_arduino_devices; print(list_arduino_devices())"
   ```
3. Edit `viz.py:6` `PORT = "..."` to that path, then run:
   ```bash
   python streams/audio/contact_mic_vendor/viz.py
   ```

## Wiring two mics for `twm/data_collection.py`

Once you have two Arduinos labeled (left and right), set the paths in
`twm/data_collection.py`:

```python
USB_SERIAL_DEVICES = {
    "audio_left":  "/dev/serial/by-id/usb-Arduino_..._<serial-A>-if00",
    "audio_right": "/dev/serial/by-id/usb-Arduino_..._<serial-B>-if00",
}
```

To figure out which serial maps to which physical mic: with both
plugged in, run `list_arduino_devices()`, unplug ONE, run it again,
note which entry disappeared. Label the remaining one. Plug back in.
````

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(audio): move viz.py to vendor subfolder + add labeling README"
```

---

## Task 10: Add audio HDF5 schema in `create_episode_file` (TDD)

**Files:**
- Modify: `tests/test_hdf5_writer.py`
- Modify: `twm/data_collection.py:27-70` (`create_episode_file` function)

- [ ] **Step 1: Add the failing test**

In `tests/test_hdf5_writer.py`, inside `class TestHDF5Writer`, add a new test method just after `test_create_episode_file_structure`:

```python
    def test_create_episode_file_includes_audio_groups(self):
        """create_episode_file creates audio/{left,right}/{timestamps,samples} datasets."""
        f, path = create_episode_file(
            date_dir=self.tmpdir,
            episode_num=10,
            realsense_serials=["AAA", "BBB", "CCC"],
            gelsight_serials=["L", "R"],
            fps=30,
        )
        f.close()

        with h5py.File(path, "r") as f:
            for name in ["left", "right"]:
                self.assertIn(f"audio/{name}/timestamps", f)
                self.assertIn(f"audio/{name}/samples", f)
                self.assertEqual(f[f"audio/{name}/timestamps"].dtype, np.float64)
                self.assertEqual(f[f"audio/{name}/samples"].dtype, np.int16)
                self.assertEqual(f[f"audio/{name}/timestamps"].shape, (0,))
            self.assertEqual(list(f["metadata"].attrs["audio_channels"]), ["left", "right"])
            self.assertEqual(int(f["metadata"].attrs["audio_sample_rate"]), 1000)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_hdf5_writer.py::TestHDF5Writer::test_create_episode_file_includes_audio_groups -v
```
Expected: FAIL with KeyError on `audio/left/timestamps`.

- [ ] **Step 3: Add audio groups to `create_episode_file`**

In `twm/data_collection.py`, inside `create_episode_file` (function defined ~line 27), add audio metadata + datasets just after the OptiTrack block (currently around line 65-68). The function ends with:

```python
    # OptiTrack — per-tracker timestamps + poses
    for name in ["motherboard", "sensor_left", "sensor_right"]:
        g = f.create_group(f"optitrack/{name}")
        g.create_dataset("timestamps", shape=(0,),    maxshape=(None,),    dtype=np.float64)
        g.create_dataset("pose",       shape=(0, 7),  maxshape=(None, 7),  dtype=np.float64)

    return f, path
```

Insert before `return f, path`:
```python
    # Audio — contact microphones (left, right) at ~1 kHz
    meta.attrs["audio_channels"]    = ["left", "right"]
    meta.attrs["audio_sample_rate"] = 1000
    for name in ["left", "right"]:
        g = f.create_group(f"audio/{name}")
        g.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
        g.create_dataset("samples",    shape=(0,), maxshape=(None,), dtype=np.int16)
```

- [ ] **Step 4: Run test to confirm passing**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_hdf5_writer.py -v
```
Expected: all tests pass (including the new one).

- [ ] **Step 5: Commit**

```bash
git add tests/test_hdf5_writer.py twm/data_collection.py
git commit -m "feat(twm): add audio/{left,right} HDF5 datasets to episode file"
```

---

## Task 11: Implement `flush_audio_to_hdf5` (TDD)

**Files:**
- Modify: `tests/test_hdf5_writer.py`
- Modify: `twm/data_collection.py` (add new helper)

- [ ] **Step 1: Add the failing test**

In `tests/test_hdf5_writer.py`, update the import on line 8 to also include `flush_audio_to_hdf5`:
```python
from twm.data_collection import (
    create_episode_file,
    append_camera_frame,
    flush_optitrack_to_hdf5,
    flush_audio_to_hdf5,
)
```

Add a new test method:
```python
    def test_flush_audio_writes_samples(self):
        """flush_audio_to_hdf5 writes (timestamp, sample) tuples to per-channel datasets."""
        f, path = create_episode_file(self.tmpdir, 11, ["A", "B", "C"], ["L", "R"], 30)

        audio_data = {
            "left":  [(1.0, 100), (1.001, 200), (1.002, 300)],
            "right": [(1.0, 400), (1.001, 500)],
        }
        flush_audio_to_hdf5(f, audio_data)
        f.close()

        with h5py.File(path, "r") as f:
            self.assertEqual(f["audio/left/timestamps"].shape, (3,))
            self.assertEqual(f["audio/left/samples"].shape, (3,))
            self.assertEqual(f["audio/right/timestamps"].shape, (2,))
            self.assertEqual(list(f["audio/left/samples"][:]), [100, 200, 300])
            self.assertEqual(list(f["audio/right/samples"][:]), [400, 500])

    def test_flush_audio_handles_empty_channel(self):
        """Empty channel lists are skipped without error."""
        f, path = create_episode_file(self.tmpdir, 12, ["A", "B", "C"], ["L", "R"], 30)
        flush_audio_to_hdf5(f, {"left": [], "right": [(1.0, 99)]})
        f.close()

        with h5py.File(path, "r") as f:
            self.assertEqual(f["audio/left/samples"].shape, (0,))
            self.assertEqual(f["audio/right/samples"].shape, (1,))
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_hdf5_writer.py -v
```
Expected: ImportError on `flush_audio_to_hdf5`.

- [ ] **Step 3: Implement `flush_audio_to_hdf5` in `twm/data_collection.py`**

In `twm/data_collection.py`, immediately after the `flush_optitrack_to_hdf5` function (around line 134), add:

```python
def flush_audio_to_hdf5(f, audio_data):
    """
    Write buffered audio samples to HDF5.

    Args:
        f:           open h5py.File
        audio_data:  dict mapping channel name → list of (timestamp, sample) tuples
                     e.g. {"left": [(t, sample), ...], "right": [...]}
    """
    for name, samples in audio_data.items():
        if not samples:
            continue
        ts   = np.array([s[0] for s in samples], dtype=np.float64)
        vals = np.array([s[1] for s in samples], dtype=np.int16)
        ds_t = f[f"audio/{name}/timestamps"]
        ds_s = f[f"audio/{name}/samples"]
        n = ds_t.shape[0]
        ds_t.resize(n + len(ts),   axis=0); ds_t[n:] = ts
        ds_s.resize(n + len(vals), axis=0); ds_s[n:] = vals
```

- [ ] **Step 4: Run tests to confirm passing**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_hdf5_writer.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_hdf5_writer.py twm/data_collection.py
git commit -m "feat(twm): add flush_audio_to_hdf5 helper"
```

---

## Task 12: Add `USB_SERIAL_DEVICES` config + wire `ContactMicStream` startup

**Files:**
- Modify: `twm/data_collection.py` (constants block + `main()` sensor init)

- [ ] **Step 1: Add the device config near other constants**

In `twm/data_collection.py`, find the constants block around line 264-274 (`REALSENSE_SERIALS`, `GELSIGHT_SERIALS`, `DATA_DIR`, `FPS`). Add directly below:

```python
# USB-serial device paths — use stable /dev/serial/by-id/ symlinks.
# Discover with: from streams.audio import list_arduino_devices; list_arduino_devices()
# See streams/audio/contact_mic_vendor/README.md for first-time labeling steps.
USB_SERIAL_DEVICES = {
    "audio_left":  "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_LEFT_PLACEHOLDER-if00",
    "audio_right": "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_RIGHT_PLACEHOLDER-if00",
}
AUDIO_SAMPLE_RATE_HZ = 1000
```

The `_PLACEHOLDER` strings are intentional — the user fills them in once they label their physical Arduinos. If the path doesn't resolve, the `_try_start_audio` fallback below produces a `_DummyAudioStream`.

- [ ] **Step 2: Add `_DummyAudioStream` and `_try_start_audio` helpers near `_DummyGelSight`**

In `twm/data_collection.py`, in the `main()` function around line 397-415 (where `_DummyGelSight` and `_try_start_gelsight` live as inner defs), add after `_try_start_gelsight`:

```python
    class _DummyAudioStream:
        def __init__(self, channel_name):
            self.channel_name = channel_name
        def start(self): pass
        def stop(self): pass
        def flush_buffer(self): return []
        def get_recent_samples(self, n): return np.zeros(0, dtype=np.int16)

    def _try_start_audio(channel_name, device_path):
        from streams.audio import ContactMicStream
        stream = ContactMicStream(channel_name, device_path, sample_rate_hz=AUDIO_SAMPLE_RATE_HZ)
        try:
            stream.start()
            return stream
        except Exception as e:
            print(f"  WARNING: Audio '{channel_name}' (path {device_path}) not available: {e}")
            print(f"           Continuing without it — samples will be empty.")
            return _DummyAudioStream(channel_name)
```

- [ ] **Step 3: Wire audio init in `main()` between OptiTrack init and the startup-frame wait**

In `main()`, after the `print("Initializing OptiTrack...")` block (around line 417-419) and before the `STARTUP_TIMEOUT = 15.0` line (~421), add:

```python
    print("Initializing contact microphones...")
    audio_streams = {
        "left":  _try_start_audio("left",  USB_SERIAL_DEVICES["audio_left"]),
        "right": _try_start_audio("right", USB_SERIAL_DEVICES["audio_right"]),
    }
```

- [ ] **Step 4: Add audio cleanup in the `finally:` block**

In `main()`, in the `finally:` block at the bottom (around line 561-580), add audio stream stops adjacent to the existing `gs_left.stop()` etc.:

```python
        for s in audio_streams.values():
            s.stop()
```

Place it after `gs_right.stop()` and before `optitrack.stop()`.

- [ ] **Step 5: Verify imports compile (no behavior test yet — that's manual)**

```bash
cd /home/yxma/MultimodalData
python -c "import twm.data_collection; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add twm/data_collection.py
git commit -m "feat(twm): wire ContactMicStream startup with dummy fallback"
```

---

## Task 13: Wire audio buffer drain at episode start/end/quit

**Files:**
- Modify: `twm/data_collection.py` (`main()` keypress handlers)

- [ ] **Step 1: Discard pre-episode audio on `'s'` key**

In `main()`, find the `if key == ord('s') and not recording:` block (around line 507-519). After the existing OptiTrack flush:
```python
                for name in ["motherboard", "sensor_left", "sensor_right"]:
                    optitrack.flush_buffer(name)  # discard pre-episode poses
```
add:
```python
                for s in audio_streams.values():
                    s.flush_buffer()  # discard pre-episode audio
```

- [ ] **Step 2: Drain + write audio on `'e'` key**

In `main()`, find the `elif key == ord('e') and recording:` block (around line 521-538). Just after the OptiTrack flush+write:
```python
                optitrack_data = {
                    name: optitrack.flush_buffer(name)
                    for name in ["motherboard", "sensor_left", "sensor_right"]
                }
                has_optitrack = any(len(v) > 0 for v in optitrack_data.values())
                flush_optitrack_to_hdf5(h5_file, optitrack_data)
```
add:
```python
                audio_data = {name: s.flush_buffer() for name, s in audio_streams.items()}
                has_audio = any(len(v) > 0 for v in audio_data.values())
                flush_audio_to_hdf5(h5_file, audio_data)
```
And update the `log_episode(...)` call below to pass `has_audio=has_audio` (the `log_episode` signature change comes in Task 14).

- [ ] **Step 3: Drain + write audio on `'q'` key (graceful quit while recording)**

In `main()`, find the `elif key == ord('q'):` block (~line 544-559). Inside the `if recording and h5_file is not None:` branch, after the `flush_optitrack_to_hdf5(...)` line, add the same audio block as Step 2:
```python
                    audio_data = {name: s.flush_buffer() for name, s in audio_streams.items()}
                    has_audio = any(len(v) > 0 for v in audio_data.values())
                    flush_audio_to_hdf5(h5_file, audio_data)
```
And similarly pass `has_audio=has_audio` to `log_episode`.

- [ ] **Step 4: Same in the `finally:` recovery block**

In `main()`'s `finally:` block (~line 561-575), inside `if recording and h5_file is not None:`, after `flush_optitrack_to_hdf5(h5_file, optitrack_data)`, add:
```python
                audio_data = {name: s.flush_buffer() for name, s in audio_streams.items()}
                flush_audio_to_hdf5(h5_file, audio_data)
```

- [ ] **Step 5: Verify import still compiles**

```bash
cd /home/yxma/MultimodalData
python -c "import twm.data_collection; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add twm/data_collection.py
git commit -m "feat(twm): drain audio buffers at episode start/end/quit"
```

---

## Task 14: Update `log_episode` for audio column + CSV migration

**Files:**
- Modify: `twm/data_collection.py:213-247` (`log_episode` function)

- [ ] **Step 1: Update `log_episode` signature and add audio column with migration**

Replace the current `log_episode` (around lines 213-247) with:

```python
def log_episode(data_dir, task_name, episode_num, h5_path, frame_count, fps,
                has_optitrack=True, has_audio=False, notes=""):
    """
    Append one row to data/dataset_log.csv.

    Creates the file with a header row if it doesn't exist. If the existing
    log was written before the `audio` column was introduced, this function
    migrates the file in place by reading existing rows, adding an empty
    `audio` column, and rewriting.
    """
    import csv

    log_path = os.path.join(data_dir, "dataset_log.csv")
    file_exists = os.path.isfile(log_path)

    duration_s = round(frame_count / fps, 2) if fps > 0 else 0
    size_mb    = round(os.path.getsize(h5_path) / 1e6, 1) if os.path.isfile(h5_path) else 0
    date_str   = time.strftime("%Y-%m-%d")
    saved_at   = time.strftime("%Y-%m-%dT%H:%M:%S")

    row = {
        "saved_at":   saved_at,
        "task":       task_name,
        "date":       date_str,
        "episode":    f"ep_{episode_num:03d}",
        "frames":     frame_count,
        "duration_s": duration_s,
        "size_mb":    size_mb,
        "optitrack":  "yes" if has_optitrack else "no",
        "audio":      "yes" if has_audio else "no",
        "path":       os.path.relpath(h5_path, data_dir),
        "notes":      notes,
    }
    fieldnames = list(row.keys())

    # Migrate existing CSV if it predates the `audio` column
    if file_exists:
        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
        if "audio" not in existing_fieldnames:
            print(f"Migrating {log_path}: adding 'audio' column to {len(existing_rows)} existing rows")
            for r in existing_rows:
                r["audio"] = ""
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    # Drop any keys not in new schema (defensive — should never happen)
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
            file_exists = True  # we just rewrote with header

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged → {log_path}")
```

- [ ] **Step 2: Verify `import twm.data_collection` still works**

```bash
cd /home/yxma/MultimodalData
python -c "import twm.data_collection; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Quick smoke check — log + migration round-trip**

```bash
cd /home/yxma/MultimodalData
python -c "
import os, tempfile, csv
from twm.data_collection import log_episode

# Pre-create a log file in the OLD format (no 'audio' column)
tmpdir = tempfile.mkdtemp()
log_path = os.path.join(tmpdir, 'dataset_log.csv')
with open(log_path, 'w') as f:
    f.write('saved_at,task,date,episode,frames,duration_s,size_mb,optitrack,path,notes\n')
    f.write('2026-01-01T00:00:00,old_task,2026-01-01,ep_000,100,3.3,12.0,yes,foo.h5,\n')

# Touch a fake h5 path so size_mb has something to read
fake_h5 = os.path.join(tmpdir, 'fake.h5')
open(fake_h5, 'wb').write(b'x'*1024)

log_episode(tmpdir, 'new_task', 1, fake_h5, 60, 30, has_optitrack=True, has_audio=True)

with open(log_path) as f:
    print(f.read())
"
```
Expected output: the migrated CSV has both rows; the old row's `audio` cell is empty; the new row's `audio` is `yes`.

- [ ] **Step 4: Commit**

```bash
git add twm/data_collection.py
git commit -m "feat(twm): log_episode adds audio column with backward-compat migration"
```

---

## Task 15: Implement `make_audio_panel` (TDD)

**Files:**
- Modify: `tests/test_visualize.py` (or create new test)
- Modify: `twm/data_collection.py` (`make_audio_panel` lives here for now; it moves to `twm/preview.py` in Task 18)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualize.py` (or a new `tests/test_preview.py` if test_visualize.py is purely for visualize.py — check first):

```bash
head -25 /home/yxma/MultimodalData/tests/test_visualize.py
```

If `test_visualize.py` is for the `twm.visualize` module specifically, create `tests/test_preview.py` instead:

`tests/test_preview.py`:
```python
import unittest
from unittest.mock import MagicMock

import numpy as np


class TestAudioPanel(unittest.TestCase):

    def test_make_audio_panel_dimensions(self):
        """make_audio_panel returns an image of requested w×h with 3 channels."""
        from twm.data_collection import make_audio_panel
        streams = {
            "left":  MagicMock(get_recent_samples=lambda n: np.zeros(0, dtype=np.int16)),
            "right": MagicMock(get_recent_samples=lambda n: np.zeros(0, dtype=np.int16)),
        }
        panel = make_audio_panel(streams, w=320, h=240)
        self.assertEqual(panel.shape, (240, 320, 3))
        self.assertEqual(panel.dtype, np.uint8)

    def test_make_audio_panel_handles_signal(self):
        """A non-empty buffer renders without crashing and the panel changes vs empty."""
        from twm.data_collection import make_audio_panel
        empty_streams = {
            "left":  MagicMock(get_recent_samples=lambda n: np.zeros(0, dtype=np.int16)),
            "right": MagicMock(get_recent_samples=lambda n: np.zeros(0, dtype=np.int16)),
        }
        loud_samples = np.full(200, 1023, dtype=np.int16)
        loud_streams = {
            "left":  MagicMock(get_recent_samples=lambda n: loud_samples),
            "right": MagicMock(get_recent_samples=lambda n: loud_samples),
        }
        empty_panel = make_audio_panel(empty_streams, w=320, h=240)
        loud_panel  = make_audio_panel(loud_streams, w=320, h=240)
        # Loud panel must differ from empty panel (the bars are filled)
        self.assertFalse(np.array_equal(empty_panel, loud_panel))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_preview.py -v
```
Expected: ImportError on `make_audio_panel`.

- [ ] **Step 3: Implement `make_audio_panel`**

In `twm/data_collection.py`, immediately after `make_optitrack_panel` (around line 318), add:

```python
def make_audio_panel(audio_streams, w=320, h=240):
    """Render a 2-bar VU-meter panel for left + right contact mics.

    Each bar shows peak / 512 (10-bit-ADC max). Colors:
      green  < 70% full
      amber >= 70%
      red   >= 95% (clip warning)
    """
    import cv2

    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, "Audio (peak / rms)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    bar_x0, bar_x1 = 8, w - 8
    bar_w = bar_x1 - bar_x0
    AUDIO_MID = 512  # 10-bit ADC zero level

    y = 50
    for channel in ["left", "right"]:
        stream = audio_streams.get(channel)
        cv2.putText(panel, channel, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        y += 14
        samples = None
        if stream is not None:
            try:
                samples = stream.get_recent_samples(200)
            except Exception:
                samples = None
        if samples is None or len(samples) == 0:
            cv2.putText(panel, "no signal", (8, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + 16), (60, 60, 60), 1)
        else:
            centered = samples.astype(np.int32) - AUDIO_MID
            peak = int(np.max(np.abs(centered)))
            rms  = float(np.sqrt(np.mean(centered ** 2)))
            frac = max(0.0, min(1.0, peak / 512.0))
            fill_x = bar_x0 + int(bar_w * frac)
            if   frac >= 0.95: color = (0,   0,   255)   # red
            elif frac >= 0.70: color = (0,   180, 255)   # amber
            else:              color = (0,   200, 0)     # green
            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + 16), (60, 60, 60), 1)
            cv2.rectangle(panel, (bar_x0, y), (fill_x, y + 16), color, -1)
            cv2.putText(panel, f"peak={peak:4d} rms={rms:6.1f}", (8, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += 60

    return panel
```

- [ ] **Step 4: Run test to confirm passing**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_preview.py -v
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_preview.py twm/data_collection.py
git commit -m "feat(twm): add make_audio_panel VU-meter helper"
```

---

## Task 16: Wire `make_audio_panel` into `make_preview`

**Files:**
- Modify: `twm/data_collection.py` (`make_preview` function ~line 321 + main loop ~line 474)

- [ ] **Step 1: Add `audio_streams` parameter to `make_preview`**

In `twm/data_collection.py`, change the `make_preview` signature (around line 321):

From:
```python
def make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, recording, frame_count, elapsed, buf=0, fps=0.0, task_name=""):
```
to:
```python
def make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, audio_streams, recording, frame_count, elapsed, buf=0, fps=0.0, task_name=""):
```

- [ ] **Step 2: Replace the `blank` panel with the audio panel**

In `make_preview`'s body, find:
```python
    blank = np.zeros((gs_h, rs_w, 3), dtype=np.uint8)
    row2 = np.hstack(gs_panels + [blank])
```
Replace with:
```python
    audio_panel = make_audio_panel(audio_streams, w=rs_w, h=gs_h)
    row2 = np.hstack(gs_panels + [audio_panel])
```

- [ ] **Step 3: Update the caller in `main()`**

Find the `make_preview(...)` call inside `main()` (around line 474). Change:
```python
            preview = make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, recording, frame_count, elapsed, writer.queue_size, fps, task_name=task_name)
```
to:
```python
            preview = make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, audio_streams, recording, frame_count, elapsed, writer.queue_size, fps, task_name=task_name)
```

- [ ] **Step 4: Update `twm/visualize.py` and `twm/visualize_projection.py` callers**

Both files import and use `make_preview`. Search:
```bash
grep -n "make_preview(" /home/yxma/MultimodalData/twm/visualize.py /home/yxma/MultimodalData/twm/visualize_projection.py
```

For each call site, the visualize tools don't have audio — pass an empty dict so the panel renders "no signal":
```python
make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, {}, recording=False, frame_count=..., elapsed=..., ...)
```

The exact call lines may differ — open each file and add `audio_streams={}` (kwarg form is clearest) at the appropriate position. If kwargs work, prefer:
```python
make_preview(..., audio_streams={}, ...)
```
to make the change minimal and unambiguous. Switch the new parameter to keyword-only by adding a `*` in the signature if you want enforcement, but plain positional addition is fine.

- [ ] **Step 5: Run existing tests to verify no regression**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_preview.py tests/test_visualize.py tests/test_hdf5_writer.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add twm/data_collection.py twm/visualize.py twm/visualize_projection.py
git commit -m "feat(twm): wire audio VU-meter into preview row 2"
```

---

## Task 17: Extract `twm/episode_io.py`

**Files:**
- Create: `twm/episode_io.py`
- Modify: `twm/data_collection.py` (move helpers out, leave thin re-export shim or update all callers)
- Modify: `tests/test_hdf5_writer.py` (import path)

- [ ] **Step 1: Create `twm/episode_io.py` and move HDF5 helpers**

Cut the following from `twm/data_collection.py` and paste into a new file `twm/episode_io.py`:
- `create_episode_file`
- `append_camera_frame`
- `append_camera_frames_batch`
- `flush_optitrack_to_hdf5`
- `flush_audio_to_hdf5`
- `HDF5Writer` class
- `log_episode`
- `next_episode_number`

`twm/episode_io.py` (header):
```python
"""Episode HDF5 I/O — file creation, batch appending, log writing."""

import collections
import csv
import os
import queue
import threading
import time

import h5py
import hdf5plugin
import numpy as np
```

- [ ] **Step 2: Add imports back into `twm/data_collection.py`**

At the top of `twm/data_collection.py`, after the existing imports, add:
```python
from twm.episode_io import (
    create_episode_file,
    append_camera_frame,
    append_camera_frames_batch,
    flush_optitrack_to_hdf5,
    flush_audio_to_hdf5,
    HDF5Writer,
    log_episode,
    next_episode_number,
)
```

These re-exports keep `twm.data_collection.create_episode_file` etc. resolvable at the same path for any caller that imports from `twm.data_collection` directly.

- [ ] **Step 3: Update `tests/test_hdf5_writer.py` import**

Change line 8 from:
```python
from twm.data_collection import (
    create_episode_file,
    append_camera_frame,
    flush_optitrack_to_hdf5,
    flush_audio_to_hdf5,
)
```
to:
```python
from twm.episode_io import (
    create_episode_file,
    append_camera_frame,
    flush_optitrack_to_hdf5,
    flush_audio_to_hdf5,
)
```

(Direct import is cleaner than going via the re-export.)

- [ ] **Step 4: Run tests**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_hdf5_writer.py tests/test_preview.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add twm/episode_io.py twm/data_collection.py tests/test_hdf5_writer.py
git commit -m "refactor(twm): extract episode_io.py from data_collection.py"
```

---

## Task 18: Extract `twm/preview.py`

**Files:**
- Create: `twm/preview.py`
- Modify: `twm/data_collection.py`
- Modify: `twm/visualize.py:27`
- Modify: `twm/visualize_projection.py:36`

- [ ] **Step 1: Create `twm/preview.py` and move preview helpers**

Cut `make_optitrack_panel`, `make_audio_panel`, `make_preview`, and `TRACKER_COLORS` from `twm/data_collection.py` and paste into `twm/preview.py`.

`twm/preview.py` (header):
```python
"""Preview tile rendering for TWM data collection."""

import cv2
import numpy as np


TRACKER_COLORS = {
    "motherboard":  (255, 200,   0),
    "sensor_left":  (  0, 255, 120),
    "sensor_right": (  0, 180, 255),
}


# ... (then make_optitrack_panel, make_audio_panel, make_preview)
```

- [ ] **Step 2: Re-export from `twm/data_collection.py`**

At the top of `twm/data_collection.py`, after the `from twm.episode_io import ...` block from Task 17, add:
```python
from twm.preview import make_preview, make_optitrack_panel, make_audio_panel
```

- [ ] **Step 3: Update `twm/visualize.py:27` to import directly**

Change:
```python
from twm.data_collection import make_preview
```
to:
```python
from twm.preview import make_preview
```

- [ ] **Step 4: Update `twm/visualize_projection.py:36`**

Change:
```python
from twm.data_collection import make_preview, REALSENSE_SERIALS
```
to:
```python
from twm.preview import make_preview
from twm.data_collection import REALSENSE_SERIALS
```

(`REALSENSE_SERIALS` stays in `data_collection.py` until Task 20 moves constants to top — it doesn't move out.)

- [ ] **Step 5: Update test_preview.py import**

Change `tests/test_preview.py` to import from the new location:
```python
from twm.preview import make_audio_panel
```
(Replace both occurrences.)

- [ ] **Step 6: Run tests**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/test_hdf5_writer.py tests/test_preview.py tests/test_visualize.py -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add twm/preview.py twm/data_collection.py twm/visualize.py twm/visualize_projection.py tests/test_preview.py
git commit -m "refactor(twm): extract preview.py from data_collection.py"
```

---

## Task 19: Extract `twm/sensors.py` with `init_sensors()` + dummy classes

**Files:**
- Create: `twm/sensors.py`
- Modify: `twm/data_collection.py`

- [ ] **Step 1: Create `twm/sensors.py`**

`twm/sensors.py`:
```python
"""Sensor lifecycle helpers for `twm/data_collection.py`.

Encapsulates dummy fallbacks (when a sensor is unplugged) and the
`init_sensors()` orchestration that returns a `Sensors` dataclass
with all live streams.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Dummy fallbacks — graceful degradation when a sensor is unplugged
# ──────────────────────────────────────────────────────────────────────────────

class DummyGelSight:
    def __init__(self):
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)
    def start(self): pass
    def stop(self): pass
    def get_frame(self): return self._frame


class DummyAudioStream:
    def __init__(self, channel_name: str):
        self.channel_name = channel_name
    def start(self): pass
    def stop(self): pass
    def flush_buffer(self): return []
    def get_recent_samples(self, n): return np.zeros(0, dtype=np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# Sensor bundle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Sensors:
    rs_streams: List
    gs_left: object
    gs_right: object
    optitrack: object
    audio_streams: Dict[str, object]


def _try_start_gelsight(side: str, serial: str):
    from streams.camera import USBVideoStream
    gs = USBVideoStream(serial=serial, resolution=(640, 480))
    try:
        gs.start()
        return gs
    except Exception as e:
        print(f"  WARNING: GelSight '{side}' (serial {serial}) not available: {e}")
        print(f"           Continuing without it — frames will be black.")
        return DummyGelSight()


def _try_start_audio(channel_name: str, device_path: str, sample_rate_hz: int):
    from streams.audio import ContactMicStream
    stream = ContactMicStream(channel_name, device_path, sample_rate_hz=sample_rate_hz)
    try:
        stream.start()
        return stream
    except Exception as e:
        print(f"  WARNING: Audio '{channel_name}' (path {device_path}) not available: {e}")
        print(f"           Continuing without it — samples will be empty.")
        return DummyAudioStream(channel_name)


def init_sensors(
    realsense_serials: List[str],
    gelsight_serials: Dict[str, str],
    usb_serial_devices: Dict[str, str],
    audio_sample_rate_hz: int,
    fps: int,
    rs_start_stagger_s: float,
    startup_timeout_s: float,
) -> Sensors:
    """Initialize all sensors, wait for first frames, return a Sensors bundle."""
    from streams.camera import RealsenseStream
    from streams.optitrack import OptitrackStream

    print("Initializing RealSense cameras...")
    rs_streams = [RealsenseStream(serial=s, fps=fps) for s in realsense_serials]
    for s in rs_streams:
        s.start()
        time.sleep(rs_start_stagger_s)

    print("Initializing GelSight sensors...")
    gs_left  = _try_start_gelsight("left",  gelsight_serials["left"])
    gs_right = _try_start_gelsight("right", gelsight_serials["right"])

    print("Initializing OptiTrack...")
    optitrack = OptitrackStream()
    optitrack.start()

    print("Initializing contact microphones...")
    audio_streams = {
        "left":  _try_start_audio("left",  usb_serial_devices["audio_left"],  audio_sample_rate_hz),
        "right": _try_start_audio("right", usb_serial_devices["audio_right"], audio_sample_rate_hz),
    }

    print("Waiting for first frames from all sensors...")
    for s in rs_streams:
        s.get_color_frame(timeout=startup_timeout_s)
    gs_left.get_frame()
    gs_right.get_frame()
    print("All sensors ready.\n")

    return Sensors(
        rs_streams=rs_streams,
        gs_left=gs_left,
        gs_right=gs_right,
        optitrack=optitrack,
        audio_streams=audio_streams,
    )
```

- [ ] **Step 2: Replace inline init in `twm/data_collection.py:main()`**

In `main()`, replace the entire init block (from `print("Initializing RealSense cameras...")` down through `print("All sensors ready.\n")`, plus the inline `_DummyGelSight` / `_DummyAudioStream` / `_try_start_*` defs) with:

```python
    sensors = init_sensors(
        realsense_serials=REALSENSE_SERIALS,
        gelsight_serials=GELSIGHT_SERIALS,
        usb_serial_devices=USB_SERIAL_DEVICES,
        audio_sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
        fps=FPS,
        rs_start_stagger_s=RS_START_STAGGER_S,
        startup_timeout_s=STARTUP_TIMEOUT,
    )
    rs_streams    = sensors.rs_streams
    gs_left       = sensors.gs_left
    gs_right      = sensors.gs_right
    optitrack     = sensors.optitrack
    audio_streams = sensors.audio_streams
    gs_ref = [gs_left.get_frame(), gs_right.get_frame()]
```

(`RS_START_STAGGER_S` and `STARTUP_TIMEOUT` are hoisted to top in Task 20 — for this task, define them inline near where you remove the magic numbers, or temporarily leave them as locals; Task 20 will tidy.)

- [ ] **Step 3: Add the import**

Near the top of `twm/data_collection.py`, with the other twm imports, add:
```python
from twm.sensors import init_sensors
```

- [ ] **Step 4: Verify import compiles**

```bash
cd /home/yxma/MultimodalData
python -c "import twm.data_collection; print('ok')"
python -c "from twm.sensors import init_sensors, Sensors, DummyGelSight, DummyAudioStream; print('ok')"
```
Expected: `ok` from both.

- [ ] **Step 5: Run tests**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/ -v --ignore=tests/test_realsense_stream.py --ignore=tests/test_optitrack_stream.py
```

(Skip the hardware-mocked tests for this round — they need their environment-mock fixtures which we already verified in Task 2-3.)

Expected: all run tests pass.

- [ ] **Step 6: Commit**

```bash
git add twm/sensors.py twm/data_collection.py
git commit -m "refactor(twm): extract sensors.py with init_sensors() + dummy fallbacks"
```

---

## Task 20: Hoist constants to `twm/data_collection.py` top

**Files:**
- Modify: `twm/data_collection.py`

- [ ] **Step 1: Add the missing constants near the existing constants block**

In `twm/data_collection.py`, in the constants block where `REALSENSE_SERIALS`, `GELSIGHT_SERIALS`, `DATA_DIR`, `FPS`, `USB_SERIAL_DEVICES`, `AUDIO_SAMPLE_RATE_HZ` already live, add:

```python
STARTUP_TIMEOUT       = 15.0  # seconds — cameras can be slow on first init
RS_START_STAGGER_S    = 0.5   # stagger RealSense starts to avoid USB bandwidth contention
TIMING_REPORT_INTERVAL = 60   # ticks between perf report lines
```

- [ ] **Step 2: Replace the inline `STARTUP_TIMEOUT = 15.0` and `TIMING_REPORT_INTERVAL = 60` definitions inside `main()` with reads of the module-level constants**

Search inside `main()` for:
```python
    STARTUP_TIMEOUT = 15.0
```
and
```python
    TIMING_REPORT_INTERVAL = 60
```

Delete these lines. Their references later in `main()` will resolve to the module-level constants.

- [ ] **Step 3: Verify import + a smoke test of `make_preview`**

```bash
cd /home/yxma/MultimodalData
python -c "import twm.data_collection; from twm.preview import make_preview; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Run all tests**

```bash
cd /home/yxma/MultimodalData
python -m pytest tests/ -v --ignore=tests/test_realsense_stream.py --ignore=tests/test_optitrack_stream.py
```
Expected: all pass.

(Optional: if pyrealsense2 + the rospy mocks are available in your dev env, run the full suite without `--ignore`. They were verified in Tasks 2-3.)

- [ ] **Step 5: Commit**

```bash
git add twm/data_collection.py
git commit -m "refactor(twm): hoist STARTUP_TIMEOUT and friends to module top"
```

---

## Task 21: Bench smoke test (manual — produces a checked-in test record)

**Files:**
- Create: `docs/superpowers/specs/2026-04-29-streams-refactor-smoke-test-log.md`

This task is **manual**. It cannot be run in CI; it requires the physical hardware (3 RealSense, 2 GelSight, 2 contact-mic Arduinos, OptiTrack rig). Document the result so future runs can compare.

- [ ] **Step 1: Set USB-serial paths**

```bash
cd /home/yxma/MultimodalData
python -c "from streams.audio import list_arduino_devices; print('\n'.join(list_arduino_devices()))"
```
Note the two Arduino paths. Edit `twm/data_collection.py` `USB_SERIAL_DEVICES` to use those paths (replacing the `_PLACEHOLDER` values).

To label "left" vs "right": with both plugged in, run the listing once. Unplug ONE Arduino. Run again. Whichever entry disappeared was the one you unplugged — label it physically with tape, then plug it back in.

- [ ] **Step 2: Start data collection**

```bash
cd /home/yxma/MultimodalData
python twm/data_collection.py --task smoke_test
```

Expected console:
- `Initializing RealSense cameras...`
- `Initializing GelSight sensors...`
- `Initializing OptiTrack...`
- `Initializing contact microphones...`
- `Waiting for first frames from all sensors...`
- `All sensors ready.`

The preview window opens. Bottom-right tile shows two VU-meter rows ("left" and "right") with bars and `peak=… rms=…` lines.

- [ ] **Step 3: Tap each contact mic**

Tap the left mic with a fingernail. The "left" bar should jump green → amber → maybe red. The "right" bar stays low. Repeat with the right mic.

- [ ] **Step 4: Record a 10-second episode**

Press `s`. The status bar turns red (`[REC ...]`). Wait ~10 seconds, then press `e`.

Console expected:
```
Episode 000 saved — 300 frames, 10.0s
Logged → /media/yxma/Disk1/twm/data/smoke_test/dataset_log.csv
```

- [ ] **Step 5: Inspect the HDF5**

```bash
cd /home/yxma/MultimodalData
python -c "
import h5py, glob
path = sorted(glob.glob('/media/yxma/Disk1/twm/data/smoke_test/*/episode_000.h5'))[-1]
with h5py.File(path, 'r') as f:
    print('keys:', list(f.keys()))
    print('audio/left/samples shape:', f['audio/left/samples'].shape)
    print('audio/right/samples shape:', f['audio/right/samples'].shape)
    print('audio sample dtype:', f['audio/left/samples'].dtype)
    print('first 5 left samples:', f['audio/left/samples'][:5])
    print('first 5 left timestamps:', f['audio/left/timestamps'][:5])
    print('audio_channels attr:', list(f['metadata'].attrs['audio_channels']))
"
```

Expected: `audio/left/samples.shape` ≈ `(10000,)` (give or take a few hundred for jitter). `dtype` is `int16`. Timestamps strictly increasing.

- [ ] **Step 6: Quit cleanly**

Press `q`. Console: `All sensors stopped. Goodbye.`

- [ ] **Step 7: Write smoke-test log**

Create `docs/superpowers/specs/2026-04-29-streams-refactor-smoke-test-log.md` with the actual observed values:

```markdown
# Smoke test — 2026-04-29 streams refactor + audio integration

| Metric                          | Observed                |
|---------------------------------|-------------------------|
| Console init lines              | (paste here)            |
| Episode 000 frame count          | 300                     |
| audio/left/samples shape        | (10142,)                |
| audio/right/samples shape       | (10118,)                |
| audio/{left,right} dtype        | int16                   |
| Timestamps monotonic?            | yes                     |
| VU-meter responded to mic taps  | yes (left, right)       |
| dataset_log.csv has audio col?  | yes                     |
| Quit clean (no exceptions)      | yes                     |
```

- [ ] **Step 8: Commit the smoke-test log**

```bash
git add docs/superpowers/specs/2026-04-29-streams-refactor-smoke-test-log.md
git commit -m "test: bench smoke-test record for streams + audio refactor"
```

---

## Self-Review

**Spec coverage:** every section of the design doc has at least one task —
- §2 module layout → Tasks 1-9
- §3 audio module → Tasks 7-9
- §4 refactor of existing modules → Tasks 2-4, 6
- §5 bugfixes + USB-serial collision → Tasks 4 (init), 5 (Timer), 12 (USB_SERIAL_DEVICES)
- §6 audio integration → Tasks 10-14
- §7 preview UX → Tasks 15-16
- §8 decomposition → Tasks 17-20
- §9 testing → Tasks 8, 10, 11, 15, 21
- §10 risks (mitigations baked into tasks 9 README, 14 migration, 5 bugfix scope)

**Placeholder scan:** code blocks contain real implementations or real test bodies. The only `_PLACEHOLDER` strings are in `USB_SERIAL_DEVICES` (intentional — user fills in once labeled).

**Type consistency:** `flush_buffer()` returns `list[(t, sample)]` in §3 base, in test code, in `flush_audio_to_hdf5`, and in the `'e'`/`'q'` handlers — consistent. `get_recent_samples(n)` returns `np.ndarray` of `dtype=int16` everywhere it's used. `BaseAudioStream` defines all four methods (`start`, `stop`, `flush_buffer`, `get_recent_samples`); `_DummyAudioStream` implements all four; the preview helper relies only on `get_recent_samples`.

---

## Execution

Plan saved to `docs/superpowers/plans/2026-04-29-streams-refactor-and-audio.md`.
