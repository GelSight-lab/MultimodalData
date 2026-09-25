# TWM Recorder Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `twm/data_collection.py` into a tested `twm/recorder/` package with a fail-fast, byte-bounded writer, preflight checks, and runtime health monitoring, while keeping every existing import of `twm.data_collection` working.

**Architecture:** `SensorRig.grab()` produces one frozen `Tick` per 30 Hz cycle; `CaptureLoop` submits it to `EpisodeWriter`, whose queue is bounded in bytes and which never drops (it raises). A `Recorder` controller on the GUI thread owns episode start/finalize and turns writer overload, faults, low disk, capture stalls and OptiTrack silence into a finalized, explicitly-invalid episode. HDF5 layout lives only in `schema.py`.

**Tech Stack:** Python 3.9, numpy, h5py + hdf5plugin (BLOSC LZ4), OpenCV for the preview, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-twm-recorder-library-design.md`

## Global Constraints

- Python 3.9 syntax only (`from __future__ import annotations` for `X | None` in annotations; no `match`).
- Work in the worktree `.worktrees/twm-arducam` on branch `refactor/twm-recorder` (branched from `feature/twm-arducam`). Run tests as `python -m pytest ...`.
- Every array shape stays `(480, 640, 3)` uint8 for color/GelSight/Arducam and `(480, 640)` uint16 for depth. HDF5 group names stay `timestamps`, `realsense/cam{i}/{color,depth}`, `gelsight/{left,right}/{frames,timestamps}`, `optitrack/{name}/{timestamps,pose}`, `arducam/cam{0,1}/{frames,timestamps}`.
- Never drop a tick silently. Overload raises `WriterOverloaded`.
- No hardware or cv2 imports at module import time anywhere in `twm/recorder/` except inside `app.run()` / `rig.default_drivers()`.
- Use `logging.getLogger("twm.recorder")`, never bare `print`, inside the package.
- Commit after each task with the message given in the task.

---

### Task 1: Package skeleton and configuration

**Files:**
- Create: `twm/recorder/__init__.py`
- Create: `twm/recorder/config.py`
- Test: `tests/recorder/__init__.py` (empty), `tests/recorder/test_config.py`

**Interfaces:**
- Produces: `RecorderConfig`, `WriterConfig`, `DiskConfig`, `parse_args(argv) -> RecorderConfig`, constants `REALSENSE_SERIALS`, `GELSIGHT_SERIALS`, `DATA_DIR`, `FPS`, `OT_TRACKERS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_config.py
from pathlib import Path

from twm.recorder.config import (DATA_DIR, FPS, RecorderConfig, WriterConfig,
                                 parse_args)


def test_parse_args_defaults_match_module_constants():
    cfg = parse_args(["--task", "pouring"])
    assert cfg.task == "pouring"
    assert cfg.data_dir == DATA_DIR
    assert cfg.fps == FPS
    assert cfg.active_sensors == ("sensor_left", "sensor_right")
    assert cfg.use_arducam is True
    assert cfg.show_projection is True
    assert cfg.writer == WriterConfig()
    assert cfg.disk.bandwidth_test_s == 2.0


def test_parse_args_maps_active_sensors_and_flags():
    cfg = parse_args(["--task", "t", "--active_sensors", "right",
                      "--no_projection", "--no_arducam", "--no_bandwidth_test",
                      "--data_dir", "/tmp/x", "--queue_seconds", "1.5",
                      "--min_free_gb", "7"])
    assert cfg.active_sensors == ("sensor_right",)
    assert cfg.show_projection is False
    assert cfg.use_arducam is False
    assert cfg.disk.bandwidth_test_s == 0.0
    assert cfg.data_dir == Path("/tmp/x")
    assert cfg.writer.queue_seconds == 1.5
    assert cfg.disk.min_free_gb == 7.0


def test_config_is_frozen_and_has_tick_dt():
    cfg = RecorderConfig(task="t", fps=25)
    assert abs(cfg.tick_dt - 0.04) < 1e-9
    try:
        cfg.fps = 10
    except Exception:
        return
    raise AssertionError("RecorderConfig must be frozen")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/__init__.py
"""TWM multimodal recorder as a library.

Modules
  config    RecorderConfig and the CLI that builds it
  frames    Tick — one synchronized multimodal sample
  schema    the HDF5 episode layout (create / append / finalize)
  writer    EpisodeWriter — byte-bounded, batched, fail-fast background writer
  rig       SensorRig — hardware startup, grab(), ordered shutdown
  capture   CaptureLoop — strict-rate capture thread
  preflight startup / episode-start checks
  monitor   health line for the preview
  episode   EpisodeStore (paths, numbering, CSV log) and EpisodeSummary
  app       Recorder controller, cv2 GUI loop, main()
"""
from twm.recorder.config import (DATA_DIR, FPS, GELSIGHT_SERIALS,  # noqa: F401
                                 OT_TRACKERS, REALSENSE_SERIALS, DiskConfig,
                                 RecorderConfig, WriterConfig, parse_args)
```

```python
# twm/recorder/config.py
"""Recorder configuration: one frozen dataclass tree, built from the CLI."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

REALSENSE_SERIALS: Tuple[str, ...] = (
    "143322063538",
    "104122062574",
    "217222066989",
)
GELSIGHT_SERIALS: Dict[str, str] = {
    # Left unit replaced 2026-08-07 (old 2DUPB53G failed USB enumeration).
    "left": "28YGZL6K",
    "right": "2BGLKZNT",
}
DATA_DIR = Path("/media/yxma/Disk1/twm/data")
FPS = 30
OT_TRACKERS: Tuple[str, ...] = ("motherboard", "sensor_left", "sensor_right")
ACTIVE_SENSOR_CHOICES: Dict[str, Tuple[str, ...]] = {
    "both": ("sensor_left", "sensor_right"),
    "left": ("sensor_left",),
    "right": ("sensor_right",),
}


@dataclass(frozen=True)
class WriterConfig:
    """Bounds and thresholds for EpisodeWriter (see writer.py)."""
    queue_seconds: float = 3.0         # capacity, in seconds of ticks
    batch_size: int = 10               # ticks per resize+write
    flush_interval_s: float = 10.0     # H5Fflush cadence (crash safety)
    warn_fraction: float = 0.25        # preview turns yellow above this
    overload_fraction: float = 0.5     # fail-fast if above this for ...
    overload_sustained_s: float = 3.0  # ... this long
    max_tick_gap_s: float = 0.5        # fail-fast if two ticks are further apart


@dataclass(frozen=True)
class DiskConfig:
    min_free_gb: float = 50.0          # refuse to start / auto-end below this
    bandwidth_test_s: float = 2.0      # startup self-test duration; 0 disables
    min_bandwidth_margin: float = 1.5  # writer must sustain fps × margin


@dataclass(frozen=True)
class RecorderConfig:
    task: str
    data_dir: Path = DATA_DIR
    fps: int = FPS
    realsense_serials: Tuple[str, ...] = REALSENSE_SERIALS
    gelsight_serials: Dict[str, str] = field(
        default_factory=lambda: dict(GELSIGHT_SERIALS))
    active_sensors: Tuple[str, ...] = ACTIVE_SENSOR_CHOICES["both"]
    use_arducam: bool = True
    arducam_config_path: Optional[Path] = None
    show_projection: bool = True
    startup_timeout_s: float = 15.0
    settle_s: float = 1.0
    warmup_drop_frames: int = 10
    ot_preflight_max_age_s: float = 2.0
    ot_watchdog_timeout_s: float = 10.0
    preview_fps: int = 15
    writer: WriterConfig = WriterConfig()
    disk: DiskConfig = DiskConfig()

    @property
    def tick_dt(self) -> float:
        return 1.0 / self.fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TWM multimodal data collection")
    p.add_argument("--task", required=True,
                   help="Task name; top-level folder under the data dir.")
    p.add_argument("--active_sensors", default="both",
                   choices=sorted(ACTIVE_SENSOR_CHOICES),
                   help="Which GelSight rigid bodies the OptiTrack watchdog "
                        "monitors.")
    p.add_argument("--no_projection", action="store_true",
                   help="Disable the GelSight→camera projection overlay.")
    p.add_argument("--no_arducam", action="store_true",
                   help="Run without the two sensor-mounted Arducams.")
    p.add_argument("--arducam_config", default=None,
                   help="Arducam JSON config (default twm/config/arducam.json).")
    p.add_argument("--data_dir", default=None,
                   help=f"Episode root (default {DATA_DIR}).")
    p.add_argument("--queue_seconds", type=float, default=None,
                   help="Writer queue capacity in seconds of ticks.")
    p.add_argument("--min_free_gb", type=float, default=None,
                   help="Refuse to record below this much free disk.")
    p.add_argument("--no_bandwidth_test", action="store_true",
                   help="Skip the startup write-throughput self-test.")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> RecorderConfig:
    a = build_parser().parse_args(argv)
    writer = WriterConfig()
    if a.queue_seconds is not None:
        writer = WriterConfig(**{**writer.__dict__, "queue_seconds": a.queue_seconds})
    disk = DiskConfig()
    disk_kw = dict(disk.__dict__)
    if a.min_free_gb is not None:
        disk_kw["min_free_gb"] = a.min_free_gb
    if a.no_bandwidth_test:
        disk_kw["bandwidth_test_s"] = 0.0
    disk = DiskConfig(**disk_kw)
    return RecorderConfig(
        task=a.task,
        data_dir=Path(a.data_dir) if a.data_dir else DATA_DIR,
        active_sensors=ACTIVE_SENSOR_CHOICES[a.active_sensors],
        use_arducam=not a.no_arducam,
        arducam_config_path=Path(a.arducam_config) if a.arducam_config else None,
        show_projection=not a.no_projection,
        writer=writer,
        disk=disk,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_config.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/__init__.py twm/recorder/config.py tests/recorder/
git commit -m "feat(recorder): package skeleton and frozen RecorderConfig"
```

---

### Task 2: `Tick` and the HDF5 schema module

**Files:**
- Create: `twm/recorder/frames.py`
- Create: `twm/recorder/schema.py`
- Test: `tests/recorder/test_frames.py`, `tests/recorder/test_schema.py`

**Interfaces:**
- Produces: `Tick(timestamp, color, depth, gelsight, gelsight_ts, arducam, arducam_ts, optitrack)`, `Tick.nbytes()`, `full_rig_tick_nbytes(n_realsense, n_gelsight, n_arducam)`, `synthetic_tick(timestamp, seed, n_realsense, n_gelsight, n_arducam)`; `create_episode_file(...)` (unchanged signature), `append_ticks(f, ticks)`, `append_optitrack(f, data)`, `count_optitrack_samples(f) -> int`, `write_episode_attrs(f, attrs)`, `tick_from_legacy(item) -> Tick`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_frames.py
import numpy as np
import pytest

from twm.recorder.frames import Tick, full_rig_tick_nbytes, synthetic_tick


def test_nbytes_counts_every_array_and_optitrack_samples():
    t = synthetic_tick(1.0, seed=0, n_realsense=3, n_gelsight=2, n_arducam=2)
    per_color = 480 * 640 * 3
    per_depth = 480 * 640 * 2
    assert t.nbytes() == 3 * per_color + 3 * per_depth + 2 * per_color + 2 * per_color
    assert full_rig_tick_nbytes(3, 2, 2) == t.nbytes()
    with_ot = Tick(1.0, optitrack={"motherboard": [(1.0, [0] * 7)] * 4})
    assert with_ot.nbytes() == 4 * Tick.OPTITRACK_SAMPLE_BYTES


def test_tick_rejects_mismatched_timestamp_lengths():
    frame = np.zeros((480, 640, 3), np.uint8)
    with pytest.raises(ValueError):
        Tick(1.0, gelsight=(frame, frame), gelsight_ts=(1.0,))
    with pytest.raises(ValueError):
        Tick(1.0, arducam=(frame,), arducam_ts=())


def test_synthetic_tick_is_deterministic_and_not_flat():
    a = synthetic_tick(5.0, seed=3)
    b = synthetic_tick(5.0, seed=3)
    np.testing.assert_array_equal(a.color[0], b.color[0])
    assert a.color[0].std() > 5
    assert a.gelsight_ts == (5.0, 5.0)
```

```python
# tests/recorder/test_schema.py
import numpy as np
import h5py
import pytest

from twm.recorder.frames import Tick, synthetic_tick
from twm.recorder.schema import (append_optitrack, append_ticks,
                                 count_optitrack_samples, create_episode_file,
                                 tick_from_legacy, write_episode_attrs)


@pytest.fixture
def episode(tmp_path):
    f, path = create_episode_file(str(tmp_path), 0, ["A", "B", "C"], ["L", "R"], 30,
                                  task_name="t")
    yield f, path
    if f.id.valid:
        f.close()


def test_append_ticks_writes_every_modality_and_optitrack(episode):
    f, _ = episode
    t0 = synthetic_tick(10.0, seed=0)
    t1 = Tick(11.0, color=t0.color, depth=t0.depth, gelsight=t0.gelsight,
              gelsight_ts=(10.9, 10.95),
              optitrack={"motherboard": [(10.5, [1, 2, 3, 0, 0, 0, 1])],
                         "sensor_left": [], "sensor_right": []})
    append_ticks(f, [t0, t1])

    np.testing.assert_allclose(f["timestamps"][:], [10.0, 11.0])
    assert f["realsense/cam2/color"].shape == (2, 480, 640, 3)
    assert f["realsense/cam2/depth"].shape == (2, 480, 640)
    np.testing.assert_array_equal(f["gelsight/right/frames"][1], t0.gelsight[1])
    np.testing.assert_allclose(f["gelsight/right/timestamps"][:], [10.0, 10.95])
    np.testing.assert_allclose(f["optitrack/motherboard/pose"][:],
                               [[1, 2, 3, 0, 0, 0, 1]])
    assert count_optitrack_samples(f) == 1


def test_append_ticks_skips_absent_groups(tmp_path):
    f, _ = create_episode_file(str(tmp_path), 1, [], [], 30, include_legacy=False)
    append_ticks(f, [Tick(1.0)])
    assert f["timestamps"].shape == (1,)
    assert "realsense" not in f
    f.close()


def test_append_ticks_with_empty_batch_is_noop(episode):
    f, _ = episode
    append_ticks(f, [])
    assert f["timestamps"].shape == (0,)


def test_tick_from_legacy_falls_back_to_tick_timestamp():
    frame = np.zeros((480, 640, 3), np.uint8)
    depth = np.zeros((480, 640), np.uint16)
    t = tick_from_legacy(([frame] * 3, [depth] * 3, [frame, frame], 7.0,
                          [None, 6.9]))
    assert t.gelsight_ts == (7.0, 6.9)
    assert t.arducam == ()
    t2 = tick_from_legacy((None, None, None, 7.0, None, [frame, frame], [None, None]))
    assert t2.color == () and t2.arducam_ts == (7.0, 7.0)


def test_write_episode_attrs_round_trips(episode):
    f, path = episode
    write_episode_attrs(f, {"valid": False, "invalid_reason": "overload",
                            "frame_count": 3, "max_tick_gap_s": 0.7})
    f.close()
    with h5py.File(path, "r") as g:
        m = g["metadata"].attrs
        assert bool(m["valid"]) is False
        assert m["invalid_reason"] == "overload"
        assert int(m["frame_count"]) == 3
        assert abs(float(m["max_tick_gap_s"]) - 0.7) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_frames.py tests/recorder/test_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder.frames'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/frames.py
"""Tick — one synchronized multimodal sample produced per capture cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

COLOR_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640)


@dataclass(frozen=True)
class Tick:
    """Everything captured in one 30 Hz cycle.

    Frames are the sensors' own copies (streams copy on read), so the
    writer may keep references without copying again. Per-sensor timestamps
    are the true capture time; the recorder tick time is `timestamp`.
    """
    OPTITRACK_SAMPLE_BYTES = 8 * 8  # 1 timestamp + 7 pose floats

    timestamp: float
    color: Tuple[np.ndarray, ...] = ()
    depth: Tuple[np.ndarray, ...] = ()
    gelsight: Tuple[np.ndarray, ...] = ()
    gelsight_ts: Tuple[float, ...] = ()
    arducam: Tuple[np.ndarray, ...] = ()
    arducam_ts: Tuple[float, ...] = ()
    optitrack: Mapping[str, Sequence[Tuple[float, Sequence[float]]]] = field(
        default_factory=dict)

    def __post_init__(self):
        if len(self.gelsight_ts) != len(self.gelsight):
            raise ValueError(f"{len(self.gelsight)} GelSight frames but "
                             f"{len(self.gelsight_ts)} timestamps")
        if len(self.arducam_ts) != len(self.arducam):
            raise ValueError(f"{len(self.arducam)} Arducam frames but "
                             f"{len(self.arducam_ts)} timestamps")

    def nbytes(self) -> int:
        arrays = (*self.color, *self.depth, *self.gelsight, *self.arducam)
        n = sum(int(a.nbytes) for a in arrays)
        n += self.OPTITRACK_SAMPLE_BYTES * sum(len(v) for v in self.optitrack.values())
        return n


def full_rig_tick_nbytes(n_realsense: int = 3, n_gelsight: int = 2,
                         n_arducam: int = 2) -> int:
    """Bytes of one tick for the given rig, without building one."""
    color = int(np.prod(COLOR_SHAPE))
    depth = int(np.prod(DEPTH_SHAPE)) * 2
    return n_realsense * (color + depth) + (n_gelsight + n_arducam) * color


def synthetic_tick(timestamp: float, seed: int = 0, n_realsense: int = 3,
                   n_gelsight: int = 2, n_arducam: int = 0) -> Tick:
    """A realistic-looking tick (gradient + noise) for benchmarks and tests.

    Pure noise defeats BLOSC and understates throughput; a flat frame
    overstates it. This sits in between, like a real scene.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:COLOR_SHAPE[0], 0:COLOR_SHAPE[1]]
    base = ((xx * 255 // COLOR_SHAPE[1]) + (yy * 255 // COLOR_SHAPE[0])) // 2

    def color_frame():
        noise = rng.integers(0, 24, COLOR_SHAPE, dtype=np.uint8)
        return (base[..., None].astype(np.uint8) + noise).astype(np.uint8)

    def depth_frame():
        noise = rng.integers(0, 40, DEPTH_SHAPE, dtype=np.uint16)
        return (base.astype(np.uint16) * 8 + noise).astype(np.uint16)

    return Tick(
        timestamp=timestamp,
        color=tuple(color_frame() for _ in range(n_realsense)),
        depth=tuple(depth_frame() for _ in range(n_realsense)),
        gelsight=tuple(color_frame() for _ in range(n_gelsight)),
        gelsight_ts=tuple(timestamp for _ in range(n_gelsight)),
        arducam=tuple(color_frame() for _ in range(n_arducam)),
        arducam_ts=tuple(timestamp for _ in range(n_arducam)),
    )
```

```python
# twm/recorder/schema.py
"""The HDF5 episode layout. Nothing else in the package names a dataset.

    metadata/                       attrs: fps, serials, task, created_at,
                                    and (at finalize) valid, ended_by, ...
    timestamps            float64 [T]      recorder tick time
    realsense/cam{i}/color   uint8 [T,480,640,3]
    realsense/cam{i}/depth  uint16 [T,480,640]
    gelsight/{left,right}/frames      uint8 [T,480,640,3]
    gelsight/{left,right}/timestamps  float64 [T]   capture time per frame
    arducam/cam{0,1}/frames, timestamps (only when configured)
    optitrack/{name}/timestamps float64 [N], pose float64 [N,7]
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, Mapping, Sequence, Tuple

import h5py
import hdf5plugin
import numpy as np

from twm.recorder.config import OT_TRACKERS
from twm.recorder.frames import COLOR_SHAPE, DEPTH_SHAPE, Tick

GELSIGHT_SIDES = ("left", "right")
ARDUCAM_SLOTS = ("cam0", "cam1")
N_REALSENSE = 3

# BLOSC LZ4 — benchmarked at ~100 fps of full-rig ticks vs 30 fps capture.
_BLOSC = dict(hdf5plugin.Blosc(cname="lz4", clevel=5,
                               shuffle=hdf5plugin.Blosc.SHUFFLE))


def _frame_dataset(group, name, frame_shape, dtype):
    return group.create_dataset(name, shape=(0, *frame_shape),
                                maxshape=(None, *frame_shape), dtype=dtype,
                                chunks=(1, *frame_shape), **_BLOSC)


def _scalar_dataset(group, name, width=None):
    shape = (0,) if width is None else (0, width)
    maxshape = (None,) if width is None else (None, width)
    return group.create_dataset(name, shape=shape, maxshape=maxshape,
                                dtype=np.float64)


def create_episode_file(date_dir, episode_num, realsense_serials,
                        gelsight_serials, fps, task_name="",
                        arducam_config=None, include_legacy=True):
    """Create `episode_NNN.h5` with empty resizable datasets.

    Returns (h5py.File, path). The caller closes the file.
    """
    os.makedirs(date_dir, exist_ok=True)
    path = os.path.join(date_dir, f"episode_{episode_num:03d}.h5")
    f = h5py.File(path, "w")

    meta = f.create_group("metadata")
    meta.attrs["fps"] = fps
    meta.attrs["realsense_serials"] = list(realsense_serials)
    meta.attrs["gelsight_serials"] = list(gelsight_serials)
    meta.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta.attrs["task"] = task_name
    if arducam_config:
        meta.attrs["arducam_config"] = json.dumps([{
            "slot": c.slot, "id_path": c.id_path, "position": c.position,
            "device_at_recording": getattr(c, "device", ""),
            "reported_serial": getattr(c, "reported_serial", ""),
            "width": c.width, "height": c.height, "fps": c.fps,
            "pixel_format": c.pixel_format,
        } for c in arducam_config], sort_keys=True)

    _scalar_dataset(f, "timestamps")

    if include_legacy:
        for i in range(N_REALSENSE):
            g = f.create_group(f"realsense/cam{i}")
            _frame_dataset(g, "color", COLOR_SHAPE, np.uint8)
            _frame_dataset(g, "depth", DEPTH_SHAPE, np.uint16)
        for side in GELSIGHT_SIDES:
            g = f.create_group(f"gelsight/{side}")
            _frame_dataset(g, "frames", COLOR_SHAPE, np.uint8)
            _scalar_dataset(g, "timestamps")
        for name in OT_TRACKERS:
            g = f.create_group(f"optitrack/{name}")
            _scalar_dataset(g, "timestamps")
            _scalar_dataset(g, "pose", width=7)

    if arducam_config:
        for c in arducam_config:
            g = f.create_group(f"arducam/{c.slot}")
            _frame_dataset(g, "frames", (c.height, c.width, 3), np.uint8)
            _scalar_dataset(g, "timestamps")
            g.attrs["usb_path"] = c.id_path
            g.attrs["device_at_recording"] = getattr(c, "device", "")
            g.attrs["reported_serial"] = getattr(c, "reported_serial", "")
            g.attrs["position"] = c.position
            g.attrs["width"] = c.width
            g.attrs["height"] = c.height
            g.attrs["fps"] = c.fps
            g.attrs["pixel_format"] = c.pixel_format
    return f, path


def _grow(ds, end: int):
    ds.resize(end, axis=0)
    return ds


def _write_frames(ds, start: int, frames: Iterable[np.ndarray]):
    # One chunk per frame; writing per frame avoids an np.stack copy of the
    # whole batch (80 MB per 10-tick batch) for no I/O benefit.
    for k, frame in enumerate(frames):
        ds[start + k] = frame


def append_ticks(f: h5py.File, ticks: Sequence[Tick]) -> None:
    """Append a batch: one resize per dataset, then one chunk write per frame."""
    if not ticks:
        return
    n = f["timestamps"].shape[0]
    end = n + len(ticks)
    _grow(f["timestamps"], end)[n:] = np.fromiter(
        (t.timestamp for t in ticks), dtype=np.float64, count=len(ticks))

    if "realsense" in f:
        for i in range(N_REALSENSE):
            _write_frames(_grow(f[f"realsense/cam{i}/color"], end), n,
                          (t.color[i] for t in ticks))
            _write_frames(_grow(f[f"realsense/cam{i}/depth"], end), n,
                          (t.depth[i] for t in ticks))
    if "gelsight" in f:
        for j, side in enumerate(GELSIGHT_SIDES):
            _write_frames(_grow(f[f"gelsight/{side}/frames"], end), n,
                          (t.gelsight[j] for t in ticks))
            _grow(f[f"gelsight/{side}/timestamps"], end)[n:] = [
                t.gelsight_ts[j] for t in ticks]
    if "arducam" in f:
        for j, slot in enumerate(ARDUCAM_SLOTS):
            _write_frames(_grow(f[f"arducam/{slot}/frames"], end), n,
                          (t.arducam[j] for t in ticks))
            _grow(f[f"arducam/{slot}/timestamps"], end)[n:] = [
                t.arducam_ts[j] for t in ticks]
    if "optitrack" in f:
        merged = {name: [] for name in OT_TRACKERS}
        for t in ticks:
            for name, samples in t.optitrack.items():
                merged.setdefault(name, []).extend(samples)
        append_optitrack(f, merged)


def append_optitrack(f: h5py.File, data: Mapping[str, Sequence]) -> None:
    """Append (timestamp, pose7) samples per tracker; empty lists are skipped."""
    for name, samples in data.items():
        if not samples or f"optitrack/{name}" not in f:
            continue
        ds_t = f[f"optitrack/{name}/timestamps"]
        ds_p = f[f"optitrack/{name}/pose"]
        n = ds_t.shape[0]
        end = n + len(samples)
        _grow(ds_t, end)[n:] = np.array([s[0] for s in samples], np.float64)
        _grow(ds_p, end)[n:] = np.array([s[1] for s in samples], np.float64)


def count_optitrack_samples(f: h5py.File) -> int:
    if "optitrack" not in f:
        return 0
    return sum(int(f[f"optitrack/{name}/timestamps"].shape[0])
               for name in f["optitrack"])


def write_episode_attrs(f: h5py.File, attrs: Mapping[str, Any]) -> None:
    """Record end-of-episode facts (validity, counts, diagnostics) in metadata."""
    meta = f["metadata"].attrs
    for key, value in attrs.items():
        meta[key] = value if value is not None else ""


def tick_from_legacy(item: Tuple) -> Tick:
    """Adapt the old positional tuple
    (color, depth, gs, timestamp[, gs_ts[, arducam[, arducam_ts]]]).
    Missing per-sensor timestamps fall back to the tick timestamp."""
    color, depth, gs, ts = item[0], item[1], item[2], float(item[3])
    gs_ts = item[4] if len(item) > 4 else None
    ard = item[5] if len(item) > 5 else None
    ard_ts = item[6] if len(item) > 6 else None

    def _ts(values, n):
        if not values:
            return tuple(ts for _ in range(n))
        return tuple(ts if v is None else float(v) for v in values)

    gs = tuple(gs or ())
    ard = tuple(ard or ())
    return Tick(timestamp=ts, color=tuple(color or ()), depth=tuple(depth or ()),
                gelsight=gs, gelsight_ts=_ts(gs_ts, len(gs)),
                arducam=ard, arducam_ts=_ts(ard_ts, len(ard)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_frames.py tests/recorder/test_schema.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/frames.py twm/recorder/schema.py tests/recorder/test_frames.py tests/recorder/test_schema.py
git commit -m "feat(recorder): Tick dataclass and single-owner HDF5 schema"
```

---

### Task 3: `EpisodeWriter` — byte-bounded, batched, fail-fast

**Files:**
- Create: `twm/recorder/writer.py`
- Test: `tests/recorder/test_writer.py`

**Interfaces:**
- Consumes: `Tick`, `append_ticks` (Task 2).
- Produces: `WriterOverloaded`, `WriterFault`, `WriterStats`, `EpisodeWriter(capacity_bytes, batch_size, flush_interval_s, overload_fraction, overload_sustained_s, min_free_gb, sink, clock, disk_usage)` with `submit(f, tick)`, `check() -> Optional[Tuple[str, str]]`, `stats() -> WriterStats`, `reset_episode_stats()`, `drain(timeout=None)`, `stop()`; `queue_capacity_bytes(queue_seconds, fps, tick_nbytes) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_writer.py
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from twm.recorder.frames import Tick
from twm.recorder.writer import (EpisodeWriter, WriterFault, WriterOverloaded,
                                 queue_capacity_bytes)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class BlockingSink:
    """Records batches; blocks every write until `release` is set."""
    def __init__(self):
        self.batches = []
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()

    def __call__(self, f, ticks):
        self.entered.set()
        self.release.wait(timeout=5)
        self.batches.append((f, list(ticks)))


class FakeFile:
    def __init__(self, path):
        self.filename = path
        self.flushes = 0
        open(path, "wb").close()

    def flush(self):
        self.flushes += 1


def small_tick(ts):
    return Tick(ts, color=(np.zeros((4, 4, 3), np.uint8),))  # 48 bytes


TICK_BYTES = small_tick(0).nbytes()


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.002)
    return False


def test_queue_capacity_is_seconds_times_fps_times_tick():
    assert queue_capacity_bytes(3.0, 30, 8_294_400) == 90 * 8_294_400


def test_submit_writes_in_batches_grouped_by_file():
    sink = BlockingSink()
    sink.release.clear()
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 100, batch_size=3, sink=sink)
    f0, f1, f2 = object(), object(), object()
    w.submit(f0, small_tick(-1))          # primer: the thread blocks inside the sink
    assert sink.entered.wait(2)
    for i in range(4):
        w.submit(f1, small_tick(i))
    for i in range(2):
        w.submit(f2, small_tick(10 + i))
    sink.release.set()
    w.drain()
    w.stop()
    assert [(f is f1, len(t)) for f, t in sink.batches] == [
        (False, 1), (True, 3), (True, 1), (False, 2)]
    assert [t.timestamp for _, ts in sink.batches for t in ts] == [-1, 0, 1, 2, 3, 10, 11]


def test_submit_raises_instead_of_dropping_when_full():
    sink = BlockingSink()
    sink.release.clear()
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 2, batch_size=1, sink=sink)
    f = object()
    w.submit(f, small_tick(0))
    w.submit(f, small_tick(1))
    with pytest.raises(WriterOverloaded) as info:
        w.submit(f, small_tick(2))
    assert "queue full" in str(info.value)
    sink.release.set()
    w.drain()
    w.submit(f, small_tick(3))          # accepts again once drained
    w.drain()
    w.stop()
    assert len([t for _, ts in sink.batches for t in ts]) == 3
    assert w.stats().peak_fraction == 1.0


def test_check_reports_sustained_overload_only_after_threshold():
    clock = FakeClock()
    sink = BlockingSink()
    sink.release.clear()
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 4, batch_size=1, sink=sink,
                      overload_fraction=0.5, overload_sustained_s=3.0, clock=clock)
    f = object()
    for i in range(3):                  # 75 % occupancy
        w.submit(f, small_tick(i))
    assert w.check() is None            # first sighting starts the timer
    clock.t += 2.0
    assert w.check() is None
    clock.t += 1.1
    kind, detail = w.check()
    assert kind == "overload" and "3.1s" in detail
    sink.release.set()
    w.drain()
    assert w.check() is None            # cleared once the queue is below threshold
    w.stop()


def test_sink_error_becomes_fault_and_never_deadlocks():
    def bad_sink(f, ticks):
        raise OSError(28, "No space left on device")
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, batch_size=2, sink=bad_sink)
    f = object()
    w.submit(f, small_tick(0))
    assert _wait(lambda: w.stats().fault is not None)
    with pytest.raises(WriterFault):
        w.submit(f, small_tick(1))
    with pytest.raises(WriterFault):
        w.drain()
    assert w.check()[0] == "writer_fault"
    assert "No space left" in w.check()[1]
    w.stop()
    assert w.stats().queue_items == 0


def test_periodic_flush_and_disk_sampling(tmp_path):
    clock = FakeClock()
    f = FakeFile(str(tmp_path / "ep.h5"))
    usage = lambda path: SimpleNamespace(free=123e9)
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, batch_size=1,
                      flush_interval_s=10.0, sink=lambda f, t: None,
                      clock=clock, disk_usage=usage, min_free_gb=50.0)
    w.submit(f, small_tick(0))
    w.drain()
    assert f.flushes == 0
    clock.t += 10.5
    w.submit(f, small_tick(1))
    w.drain()
    assert _wait(lambda: f.flushes == 1)
    s = w.stats()
    assert s.flushes == 1
    assert s.disk_free_gb == pytest.approx(123.0)
    assert w.check() is None
    w.stop()


def test_low_disk_reported_by_check(tmp_path):
    clock = FakeClock()
    f = FakeFile(str(tmp_path / "ep.h5"))
    usage = lambda path: SimpleNamespace(free=12e9)
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, batch_size=1,
                      flush_interval_s=0.0, sink=lambda f, t: None,
                      clock=clock, disk_usage=usage, min_free_gb=50.0)
    w.submit(f, small_tick(0))
    w.drain()
    assert _wait(lambda: w.stats().disk_free_gb is not None)
    kind, detail = w.check()
    assert kind == "disk_low" and "12.0 GB" in detail
    w.stop()


def test_stop_is_idempotent_and_stats_reset_per_episode():
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, sink=lambda f, t: None)
    w.submit(object(), small_tick(0))
    w.drain()
    assert w.stats().bytes_written == TICK_BYTES
    w.reset_episode_stats()
    assert w.stats().bytes_written == 0 and w.stats().peak_fraction == 0.0
    w.stop()
    w.stop()
    with pytest.raises(WriterFault):
        w.submit(object(), small_tick(1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_writer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder.writer'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/writer.py
"""EpisodeWriter — the only thread that touches HDF5 during recording.

Design
  * The queue is bounded in BYTES (queue + in-flight batch), sized from
    seconds-of-ticks, so memory is predictable (3 s ≈ 750 MB, not 2.5 GB).
  * It never drops. `submit` raises WriterOverloaded when the next tick
    would not fit; `check` reports sustained occupancy above a threshold
    before that happens, plus low disk and write faults. The capture loop
    turns any of those into a fail-fast end of the episode.
  * Ticks are written in batches (one resize per dataset) and the file is
    H5Fflush-ed every `flush_interval_s` so a crash loses seconds, not the
    episode (pushT/2026-06-18/episode_004.h5, 79 GB, was lost that way).
"""
from __future__ import annotations

import collections
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional, Sequence, Tuple

from twm.recorder.frames import Tick
from twm.recorder.schema import append_ticks

log = logging.getLogger("twm.recorder")


class WriterOverloaded(RuntimeError):
    """The next tick does not fit in the queue. Nothing was dropped."""


class WriterFault(RuntimeError):
    """A write failed (disk full, I/O error) or the writer is stopped."""


@dataclass(frozen=True)
class WriterStats:
    queue_items: int
    queue_bytes: int
    capacity_bytes: int
    peak_fraction: float
    bytes_written: int
    write_seconds: float
    last_batch_ms: float
    flushes: int
    file_mb_s: float
    disk_free_gb: Optional[float]
    overloaded_since: Optional[float]
    fault: Optional[str]

    @property
    def fraction(self) -> float:
        return self.queue_bytes / self.capacity_bytes if self.capacity_bytes else 0.0

    @property
    def mean_mb_s(self) -> float:
        """Raw (uncompressed) throughput of the sink."""
        return self.bytes_written / self.write_seconds / 1e6 if self.write_seconds else 0.0


def queue_capacity_bytes(queue_seconds: float, fps: float, tick_nbytes: int) -> int:
    return int(queue_seconds * fps * tick_nbytes)


class EpisodeWriter:
    def __init__(self, capacity_bytes: int, batch_size: int = 10,
                 flush_interval_s: float = 10.0, overload_fraction: float = 0.5,
                 overload_sustained_s: float = 3.0,
                 min_free_gb: Optional[float] = None,
                 sink: Callable[[Any, Sequence[Tick]], None] = append_ticks,
                 clock: Callable[[], float] = time.monotonic,
                 disk_usage: Callable[[str], Any] = shutil.disk_usage):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_s = flush_interval_s
        self.overload_fraction = overload_fraction
        self.overload_sustained_s = overload_sustained_s
        self.min_free_gb = min_free_gb
        self._sink = sink
        self._clock = clock
        self._disk_usage = disk_usage

        self._cv = threading.Condition()
        self._items: Deque[Tuple[Any, Tick]] = collections.deque()
        self._queue_bytes = 0          # queued + in-flight
        self._in_flight = 0
        self._peak_fraction = 0.0
        self._bytes_written = 0
        self._write_seconds = 0.0
        self._last_batch_ms = 0.0
        self._flushes = 0
        self._last_flush_t = clock()
        self._last_file_bytes = 0
        self._file_mb_s = 0.0
        self._disk_free_gb: Optional[float] = None
        self._overloaded_since: Optional[float] = None
        self._fault: Optional[str] = None
        self._stop_requested = False
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="EpisodeWriter",
                                        daemon=True)
        self._thread.start()

    # ── producer side ────────────────────────────────────────────────────────
    def submit(self, f, tick: Tick) -> None:
        """Enqueue one tick or raise. Never blocks, never drops."""
        nbytes = tick.nbytes()
        with self._cv:
            if self._fault is not None:
                raise WriterFault(self._fault)
            if self._stop_requested:
                raise WriterFault("writer stopped")
            if self._queue_bytes + nbytes > self.capacity_bytes:
                raise WriterOverloaded(
                    f"writer queue full ({len(self._items) + self._in_flight} ticks, "
                    f"{self._queue_bytes / 1e6:.0f} of {self.capacity_bytes / 1e6:.0f} MB)")
            self._items.append((f, tick))
            self._queue_bytes += nbytes
            self._peak_fraction = max(self._peak_fraction,
                                      self._queue_bytes / self.capacity_bytes)
            self._cv.notify_all()

    def check(self) -> Optional[Tuple[str, str]]:
        """Return (kind, detail) if the episode must end now, else None.

        kinds: "writer_fault", "disk_low", "overload".
        """
        with self._cv:
            if self._fault is not None:
                return "writer_fault", self._fault
            if (self.min_free_gb is not None and self._disk_free_gb is not None
                    and self._disk_free_gb < self.min_free_gb):
                return "disk_low", (f"{self._disk_free_gb:.1f} GB free on the "
                                    f"recording disk (< {self.min_free_gb:g} GB)")
            fraction = self._queue_bytes / self.capacity_bytes
            now = self._clock()
            if fraction > self.overload_fraction:
                if self._overloaded_since is None:
                    self._overloaded_since = now
                elif now - self._overloaded_since >= self.overload_sustained_s:
                    return "overload", (
                        f"writer queue above {self.overload_fraction:.0%} for "
                        f"{now - self._overloaded_since:.1f}s ({fraction:.0%} now)")
            else:
                self._overloaded_since = None
        return None

    def stats(self) -> WriterStats:
        with self._cv:
            return WriterStats(
                queue_items=len(self._items) + self._in_flight,
                queue_bytes=self._queue_bytes,
                capacity_bytes=self.capacity_bytes,
                peak_fraction=self._peak_fraction,
                bytes_written=self._bytes_written,
                write_seconds=self._write_seconds,
                last_batch_ms=self._last_batch_ms,
                flushes=self._flushes,
                file_mb_s=self._file_mb_s,
                disk_free_gb=self._disk_free_gb,
                overloaded_since=self._overloaded_since,
                fault=self._fault,
            )

    def reset_episode_stats(self) -> None:
        with self._cv:
            self._peak_fraction = 0.0
            self._bytes_written = 0
            self._write_seconds = 0.0
            self._last_batch_ms = 0.0
            self._overloaded_since = None
            self._last_file_bytes = 0
            self._file_mb_s = 0.0

    def drain(self, timeout: Optional[float] = None) -> None:
        """Block until every submitted tick is written. Raises WriterFault."""
        with self._cv:
            ok = self._cv.wait_for(
                lambda: (self._fault is not None
                         or (not self._items and self._in_flight == 0)),
                timeout=timeout)
            if self._fault is not None:
                raise WriterFault(self._fault)
            if not ok:
                raise WriterFault(f"drain timed out after {timeout}s with "
                                  f"{len(self._items)} ticks queued")

    def stop(self) -> None:
        """Write what is queued, then stop the thread. Idempotent."""
        with self._cv:
            if self._stopped:
                return
            self._stop_requested = True
            self._cv.notify_all()
        self._thread.join()
        with self._cv:
            self._stopped = True

    # ── writer thread ────────────────────────────────────────────────────────
    def _next_batch(self):
        with self._cv:
            self._cv.wait_for(lambda: self._items or self._stop_requested)
            if not self._items:
                return None, []
            f, first = self._items.popleft()
            batch = [first]
            while (self._items and len(batch) < self.batch_size
                   and self._items[0][0] is f):
                batch.append(self._items.popleft()[1])
            self._in_flight = len(batch)
            return f, batch

    def _run(self):
        while True:
            f, batch = self._next_batch()
            if not batch:
                return
            nbytes = sum(t.nbytes() for t in batch)
            t0 = self._clock()
            try:
                self._sink(f, batch)
            except Exception as exc:
                self._record_fault(f"{type(exc).__name__}: {exc}")
                continue
            dt = self._clock() - t0
            with self._cv:
                self._queue_bytes -= nbytes
                self._in_flight = 0
                self._bytes_written += nbytes
                self._write_seconds += dt
                self._last_batch_ms = dt * 1e3
                self._cv.notify_all()
            self._maybe_flush(f)

    def _record_fault(self, message: str):
        log.error("writer fault: %s — episode must end", message)
        with self._cv:
            self._fault = message
            self._items.clear()
            self._queue_bytes = 0
            self._in_flight = 0
            self._cv.notify_all()

    def _maybe_flush(self, f):
        now = self._clock()
        if now - self._last_flush_t < self.flush_interval_s:
            return
        elapsed = now - self._last_flush_t
        self._last_flush_t = now
        try:
            flush = getattr(f, "flush", None)
            if flush is not None:
                flush()
            filename = getattr(f, "filename", None)
            file_bytes = os.path.getsize(filename) if filename else 0
            free_gb = self._disk_usage(os.path.dirname(filename) or ".").free / 1e9 \
                if filename else None
            with self._cv:
                self._flushes += 1
                if self._last_file_bytes and elapsed > 0:
                    self._file_mb_s = (file_bytes - self._last_file_bytes) / elapsed / 1e6
                self._last_file_bytes = file_bytes
                self._disk_free_gb = free_gb
        except Exception as exc:  # a failed flush must not kill the recording
            log.warning("flush failed (%s); a crash now would lose the file", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_writer.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/writer.py tests/recorder/test_writer.py
git commit -m "feat(recorder): byte-bounded fail-fast EpisodeWriter with health checks"
```

---

### Task 4: `SensorRig` — hardware startup, grab(), ordered shutdown

**Files:**
- Create: `twm/recorder/rig.py`
- Test: `tests/recorder/test_rig.py`

**Interfaces:**
- Consumes: `RecorderConfig`, `Tick`, `COLOR_SHAPE`.
- Produces: `Drivers`, `default_drivers()`, `DummyGelSight`, `frame_with_timestamp(stream)`, `SensorRig(realsense, gelsight_left, gelsight_right, optitrack, arducam=(), arducam_config=())`, `SensorRig.open(config, drivers=None)`, `wait_ready(timeout_s, settle_s)`, `grab() -> Tick`, `latest_poses() -> dict`, `arducam_labels() -> list[str]`, `close()`, context manager.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_rig.py
import numpy as np
import pytest

from twm.recorder.config import RecorderConfig
from twm.recorder.frames import Tick
from twm.recorder.rig import Drivers, DummyGelSight, SensorRig


class Stream:
    def __init__(self, log, name, fail_start=False, value=1, ts=None):
        self.log, self.name, self.fail_start = log, name, fail_start
        self.color = np.full((480, 640, 3), value, np.uint8)
        self.depth = np.full((480, 640), value, np.uint16)
        self.ts = ts

    def start(self, **kwargs):
        self.log.append(f"start {self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed")

    def stop(self):
        self.log.append(f"stop {self.name}")

    def get_color_frame(self, **kw):
        return self.color

    def get_depth_frame(self, **kw):
        return self.depth

    def get_frame(self, **kw):
        return self.color

    def get_frame_with_timestamp(self, **kw):
        return self.color, self.ts


class Optitrack:
    def __init__(self, log):
        self.log = log
        self.buffers = {"motherboard": [(1.0, [0] * 7)], "sensor_left": [],
                        "sensor_right": [(2.0, [1] * 7)]}

    def start(self):
        self.log.append("start optitrack")

    def stop(self):
        self.log.append("stop optitrack")

    def get_latest_pose(self, name):
        return (3.0, [0] * 7) if name == "motherboard" else None

    def flush_buffer(self, name):
        data, self.buffers[name] = self.buffers[name], []
        return data


class Cam:
    def __init__(self, slot):
        self.slot, self.id_path, self.position = slot, f"usb-{slot}", "unknown"
        self.config, self.device = {}, f"/dev/{slot}"


def drivers(log, gelsight_fail=(), arducam_fail=False):
    return Drivers(
        realsense=lambda serial, fps: Stream(log, f"rs {serial}"),
        gelsight=lambda serial, resolution, name: Stream(
            log, f"gs {name}", fail_start=name in gelsight_fail, ts=42.0),
        optitrack=lambda: Optitrack(log),
        arducam=lambda config, device: Stream(log, f"ard {device}",
                                              fail_start=arducam_fail, ts=7.0),
        resolve_arducams=lambda path: [Cam("cam0"), Cam("cam1")],
        sleep=lambda s: None,
    )


def config(**kw):
    kw.setdefault("realsense_serials", ("A", "B"))
    kw.setdefault("gelsight_serials", {"left": "L", "right": "R"})
    return RecorderConfig(task="t", **kw)


def test_open_starts_in_order_and_close_stops_in_reverse():
    log = []
    rig = SensorRig.open(config(), drivers(log))
    assert log == ["start rs A", "start rs B", "start ard /dev/cam0",
                   "start ard /dev/cam1", "start gs left", "start gs right",
                   "start optitrack"]
    log.clear()
    rig.close()
    rig.close()
    assert log == ["stop optitrack", "stop gs right", "stop gs left",
                   "stop ard /dev/cam1", "stop ard /dev/cam0", "stop rs B",
                   "stop rs A"]


def test_open_failure_stops_everything_started_including_failed_one():
    log = []
    with pytest.raises(RuntimeError, match="cam0 failed"):
        SensorRig.open(config(), drivers(log, arducam_fail=True))
    assert log == ["start rs A", "start rs B", "start ard /dev/cam0",
                   "stop ard /dev/cam0", "stop rs B", "stop rs A"]


def test_keyboard_interrupt_during_open_also_cleans_up():
    log = []
    d = drivers(log)
    d = Drivers(**{**d.__dict__, "optitrack": lambda: (_ for _ in ()).throw(KeyboardInterrupt())})
    with pytest.raises(KeyboardInterrupt):
        SensorRig.open(config(use_arducam=False), d)
    assert log[-4:] == ["stop gs right", "stop gs left", "stop rs B", "stop rs A"]


def test_missing_gelsight_falls_back_to_dummy_and_is_stopped():
    log = []
    rig = SensorRig.open(config(use_arducam=False), drivers(log, gelsight_fail={"right"}))
    assert isinstance(rig.gelsight_right, DummyGelSight)
    assert "stop gs right" in log         # the failed stream was released
    tick = rig.grab()
    assert tick.gelsight[1].max() == 0
    assert tick.gelsight_ts == (42.0, tick.timestamp)   # dummy has no capture time


def test_grab_builds_tick_and_drains_optitrack():
    log = []
    rig = SensorRig.open(config(), drivers(log))
    tick = rig.grab()
    assert isinstance(tick, Tick)
    assert len(tick.color) == 2 and len(tick.depth) == 2
    assert tick.arducam_ts == (7.0, 7.0)
    assert tick.optitrack["motherboard"] == [(1.0, [0] * 7)]
    assert rig.grab().optitrack["motherboard"] == []     # drained
    assert rig.latest_poses()["motherboard"] == (3.0, [0] * 7)
    assert rig.arducam_labels() == ["cam0 usb-cam0 unknown", "cam1 usb-cam1 unknown"]


def test_wait_ready_failure_closes_rig():
    log = []
    d = drivers(log)

    class Slow(Stream):
        def get_color_frame(self, **kw):
            raise TimeoutError("no frame")

    d = Drivers(**{**d.__dict__, "realsense": lambda serial, fps: Slow(log, f"rs {serial}")})
    rig = SensorRig.open(config(use_arducam=False), d)
    with pytest.raises(TimeoutError):
        rig.wait_ready(timeout_s=0.01, settle_s=0.0)
    assert log[-1] == "stop rs A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_rig.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder.rig'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/rig.py
"""SensorRig — owns the hardware. Starts in order, stops in reverse, grabs Ticks.

Drivers are injected so the rig (and everything above it) is testable
without cameras. `default_drivers()` imports the real stream classes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from twm.recorder.config import OT_TRACKERS, RecorderConfig
from twm.recorder.frames import COLOR_SHAPE, Tick

log = logging.getLogger("twm.recorder")


class DummyGelSight:
    """Stands in for a GelSight that failed to start: black frames, no timestamp."""

    def __init__(self, side: str):
        self.side = side
        self._frame = np.zeros(COLOR_SHAPE, np.uint8)

    def start(self, **kwargs):
        pass

    def stop(self):
        pass

    def get_frame(self, **kwargs):
        return self._frame

    def get_frame_with_timestamp(self, **kwargs):
        return self._frame, None


def frame_with_timestamp(stream) -> Tuple[np.ndarray, Optional[float]]:
    """(frame, capture_ts) from any stream; ts is None if it has no clock."""
    fn = getattr(stream, "get_frame_with_timestamp", None)
    if fn is not None:
        return fn()
    return stream.get_frame(), None


@dataclass(frozen=True)
class Drivers:
    realsense: Callable[..., Any]          # (serial=, fps=) -> stream
    gelsight: Callable[..., Any]           # (serial=, resolution=, name=) -> stream
    optitrack: Callable[[], Any]
    arducam: Callable[..., Any]            # (config, device) -> stream
    resolve_arducams: Callable[[Optional[Path]], Sequence[Any]]
    sleep: Callable[[float], None] = time.sleep


def default_drivers() -> Drivers:
    from camera_stream.arducam_video_stream import ArducamVideoStream
    from camera_stream.realsense_stream import RealsenseStream
    from camera_stream.usb_video_stream import USBVideoStream
    from optitrack.optitrack_stream import OptitrackStream
    from twm.sensor_camera import load_config, resolve_slots

    def resolve(path: Optional[Path]):
        return resolve_slots(load_config(path) if path else load_config())

    return Drivers(realsense=RealsenseStream, gelsight=USBVideoStream,
                   optitrack=OptitrackStream, arducam=ArducamVideoStream,
                   resolve_arducams=resolve)


def _stop_all(started: List[Any]) -> None:
    """Best-effort reverse-order stop; one failure never skips the others."""
    while started:
        resource = started.pop()
        try:
            resource.stop()
        except Exception as exc:
            log.warning("could not stop %r: %s", resource, exc)


class SensorRig:
    def __init__(self, realsense: Sequence[Any], gelsight_left, gelsight_right,
                 optitrack, arducam: Sequence[Any] = (),
                 arducam_config: Sequence[Any] = (),
                 trackers: Sequence[str] = OT_TRACKERS,
                 clock: Callable[[], float] = time.time):
        self.realsense = list(realsense)
        self.gelsight_left = gelsight_left
        self.gelsight_right = gelsight_right
        self.optitrack = optitrack
        self.arducam = list(arducam)
        self.arducam_config = tuple(arducam_config)
        self.trackers = tuple(trackers)
        self._clock = clock
        self._started: List[Any] = [*self.realsense, *self.arducam,
                                    gelsight_left, gelsight_right, optitrack]

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def open(cls, config: RecorderConfig, drivers: Optional[Drivers] = None) -> "SensorRig":
        """Start every sensor. On any failure (including Ctrl-C) stop what
        was started, in reverse, and re-raise."""
        drivers = drivers or default_drivers()
        started: List[Any] = []

        def start(resource, **kwargs):
            started.append(resource)          # registered before start(): a
            resource.start(**kwargs)          # failing start() still gets stop()
            return resource

        try:
            realsense = []
            for serial in config.realsense_serials:
                log.info("starting RealSense %s", serial)
                realsense.append(start(drivers.realsense(serial=serial, fps=config.fps)))
                drivers.sleep(0.5)            # stagger: USB bandwidth contention

            arducam, arducam_config = [], ()
            if config.use_arducam:
                arducam_config = tuple(drivers.resolve_arducams(config.arducam_config_path))
                for cam in arducam_config:
                    log.info("starting Arducam %s at %s", cam.slot, cam.device)
                    arducam.append(start(drivers.arducam(cam.config, cam.device),
                                         timeout=config.startup_timeout_s))

            gelsight: Dict[str, Any] = {}
            for side, serial in config.gelsight_serials.items():
                stream = drivers.gelsight(serial=serial, resolution=(640, 480), name=side)
                try:
                    start(stream)
                except Exception as exc:
                    log.warning("GelSight %s (serial %s) unavailable: %s — "
                                "recording black frames for this side", side, serial, exc)
                    started.remove(stream)
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    stream = DummyGelSight(side)
                gelsight[side] = stream

            log.info("starting OptiTrack")
            optitrack = start(drivers.optitrack())
        except BaseException:
            _stop_all(started)
            raise

        rig = cls(realsense, gelsight["left"], gelsight["right"], optitrack,
                  arducam, arducam_config)
        rig._started = started
        return rig

    def wait_ready(self, timeout_s: float, settle_s: float,
                   sleep: Callable[[float], None] = time.sleep) -> None:
        """Block until every sensor has produced a frame, then pump frames
        for `settle_s` so auto-exposure converges before the reference grab.
        Closes the rig and re-raises on failure."""
        try:
            for s in self.realsense:
                s.get_color_frame(timeout=timeout_s)
            for s in self.arducam:
                s.get_frame_with_timestamp(timeout=timeout_s)
            self.gelsight_left.get_frame()
            self.gelsight_right.get_frame()
            end = self._clock() + settle_s
            while self._clock() < end:
                for s in self.realsense:
                    s.get_color_frame()
                for s in self.arducam:
                    s.get_frame()
                self.gelsight_left.get_frame()
                self.gelsight_right.get_frame()
                sleep(0.02)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        _stop_all(self._started)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── per-tick access ──────────────────────────────────────────────────────
    def grab(self) -> Tick:
        """Snapshot every sensor and drain the OptiTrack buffers into one Tick."""
        color = tuple(s.get_color_frame() for s in self.realsense)
        depth = tuple(s.get_depth_frame() for s in self.realsense)
        gs = [frame_with_timestamp(s) for s in (self.gelsight_left, self.gelsight_right)]
        ard = [s.get_frame_with_timestamp() for s in self.arducam]
        t = self._clock()
        flush = getattr(self.optitrack, "flush_buffer", None)
        ot = {name: flush(name) for name in self.trackers} if flush else {}
        return Tick(
            timestamp=t,
            color=color, depth=depth,
            gelsight=tuple(f for f, _ in gs),
            gelsight_ts=tuple(t if ts is None else float(ts) for _, ts in gs),
            arducam=tuple(f for f, _ in ard),
            arducam_ts=tuple(t if ts is None else float(ts) for _, ts in ard),
            optitrack=ot,
        )

    def latest_poses(self) -> Dict[str, Any]:
        return {name: self.optitrack.get_latest_pose(name) for name in self.trackers}

    def arducam_labels(self) -> List[str]:
        return [f"{c.slot} {c.id_path} {c.position}" for c in self.arducam_config]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_rig.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/rig.py tests/recorder/test_rig.py
git commit -m "feat(recorder): SensorRig with injected drivers and ordered cleanup"
```

---

### Task 5: `CaptureLoop` — strict-rate thread with fail-fast stop requests

**Files:**
- Create: `twm/recorder/capture.py`
- Test: `tests/recorder/test_capture.py`

**Interfaces:**
- Consumes: `SensorRig.grab()/latest_poses()`, `EpisodeWriter.submit/check/stats/reset_episode_stats`, `WriterOverloaded`, `WriterFault`.
- Produces: `StopRequest(kind, detail)`, `CaptureSnapshot`, `RecordingResult(h5_file, frame_count, max_gap_s, gap_count, stop_request)`, `CaptureLoop(rig, writer, fps, warmup_drop_frames, max_tick_gap_s, report_every, clock, sleep)` with `start()`, `stop()`, `latest()`, `request_reset_ref()`, `start_recording(h5_file)`, `stop_recording() -> Optional[RecordingResult]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_capture.py
import threading
import time

import numpy as np
import pytest

from twm.recorder.capture import CaptureLoop, StopRequest
from twm.recorder.frames import Tick
from twm.recorder.writer import EpisodeWriter


class FakeRig:
    """Ticks at wall-clock time; `jump` adds an artificial gap once."""
    def __init__(self):
        self.n = 0
        self.jump = 0.0
        self.fail = None
        self.frame = np.zeros((4, 4, 3), np.uint8)

    def grab(self):
        if self.fail:
            raise self.fail
        self.n += 1
        t = time.time() + self.jump
        self.jump = 0.0
        return Tick(t, gelsight=(self.frame + self.n, self.frame),
                    gelsight_ts=(t, t))

    def latest_poses(self):
        return {"motherboard": (time.time(), [0] * 7)}


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.002)
    raise AssertionError("timeout")


TICK = Tick(0.0, gelsight=(np.zeros((4, 4, 3), np.uint8),) * 2, gelsight_ts=(0.0, 0.0)).nbytes()


def make(rig=None, sink=None, capacity_ticks=100, warmup=0, max_gap=0.5):
    rig = rig or FakeRig()
    writer = EpisodeWriter(capacity_bytes=TICK * capacity_ticks, batch_size=1,
                           sink=sink or (lambda f, t: None))
    loop = CaptureLoop(rig, writer, fps=200, warmup_drop_frames=warmup,
                       max_tick_gap_s=max_gap)
    return rig, writer, loop


def test_snapshot_is_published_and_reference_can_be_reset():
    rig, writer, loop = make()
    loop.start()
    snap = _wait(loop.latest)
    first_ref = snap.gs_ref[0][0, 0, 0]
    loop.request_reset_ref()
    _wait(lambda: loop.latest().gs_ref[0][0, 0, 0] != first_ref)
    loop.stop()
    writer.stop()
    assert snap.recording is False and snap.writer.queue_items == 0


def test_warmup_frames_are_not_recorded():
    written = []
    rig, writer, loop = make(sink=lambda f, t: written.extend(t), warmup=5)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 3)
    result = loop.stop_recording()
    loop.stop()
    writer.drain()
    writer.stop()
    assert result.frame_count == len(written)
    assert result.stop_request is None
    assert rig.n >= result.frame_count + 5


def test_overload_ends_recording_without_dropping():
    gate = threading.Event()
    written = []

    def slow_sink(f, ticks):
        gate.wait(5)
        written.extend(ticks)

    rig, writer, loop = make(sink=slow_sink, capacity_ticks=3)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    snap = _wait(lambda: loop.latest() if loop.latest().stop_request else None)
    assert snap.stop_request.kind == "overload"
    assert snap.recording is False
    gate.set()
    result = loop.stop_recording()
    loop.stop()
    writer.drain()
    writer.stop()
    assert result.stop_request.kind == "overload"
    assert result.frame_count == len(written) == 3     # every accepted tick reached the sink


def test_tick_gap_ends_recording_as_capture_stall():
    rig, writer, loop = make(max_gap=0.2)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 2)
    rig.jump = 1.0
    snap = _wait(lambda: loop.latest() if loop.latest().stop_request else None)
    result = loop.stop_recording()
    loop.stop()
    writer.stop()
    assert snap.stop_request.kind == "capture_stall"
    assert result.max_gap_s >= 1.0 and result.gap_count == 1


def test_sensor_error_is_fatal_and_recording_can_still_finalize():
    rig, writer, loop = make()
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 1)
    rig.fail = TimeoutError("Arducam cam0 frame is stale")
    snap = _wait(lambda: loop.latest() if loop.latest().fatal_error else None)
    assert "cam0" in snap.fatal_error and snap.recording is True
    result = loop.stop_recording()
    loop.stop()
    writer.stop()
    assert result is not None and result.frame_count >= 1
    assert loop.stop_recording() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_capture.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder.capture'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/capture.py
"""CaptureLoop — grabs a Tick at a strict rate on its own thread.

The GUI thread reads `latest()` at whatever rate it manages; recording
cadence never depends on cv2. While recording, every accepted tick goes to
the writer; anything that would break the 30 Hz timeline (writer overload
or fault, low disk, a stall between ticks) becomes a StopRequest that the
controller turns into a finalized, explicitly-invalid episode.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from twm.recorder.frames import Tick
from twm.recorder.writer import WriterFault, WriterOverloaded, WriterStats

log = logging.getLogger("twm.recorder")


@dataclass(frozen=True)
class StopRequest:
    kind: str      # overload | writer_fault | disk_low | capture_stall
    detail: str


@dataclass(frozen=True)
class CaptureSnapshot:
    tick: Tick
    gs_ref: Tuple[np.ndarray, ...]
    ot_poses: Dict[str, Any]
    recording: bool
    frame_count: int
    elapsed: float
    fps_meas: float
    writer: WriterStats
    stop_request: Optional[StopRequest]
    fatal_error: Optional[str] = None


@dataclass(frozen=True)
class RecordingResult:
    h5_file: Any
    frame_count: int
    max_gap_s: float
    gap_count: int
    stop_request: Optional[StopRequest]


class CaptureLoop:
    def __init__(self, rig, writer, fps: int = 30, warmup_drop_frames: int = 10,
                 max_tick_gap_s: float = 0.5, report_every: int = 60,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.rig = rig
        self.writer = writer
        self.tick_dt = 1.0 / fps
        self.warmup_drop_frames = warmup_drop_frames
        self.max_tick_gap_s = max_tick_gap_s
        self.report_every = report_every
        self._clock = clock
        self._sleep = sleep

        self._lock = threading.Lock()
        self._latest: Optional[CaptureSnapshot] = None
        self._gs_ref: Optional[Tuple[np.ndarray, ...]] = None
        self._reset_ref = False
        self._recording = False
        self._h5_file = None
        self._frame_count = 0
        self._start_t = 0.0
        self._warmup_remaining = 0
        self._last_recorded_ts: Optional[float] = None
        self._gap_count = 0
        self._max_gap = 0.0
        self._stop_request: Optional[StopRequest] = None

        self._tick_times = collections.deque(maxlen=30)
        self._grab_s = 0.0
        self._ticks = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CaptureLoop", daemon=True)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── GUI-thread API (all non-blocking) ────────────────────────────────────
    def latest(self) -> Optional[CaptureSnapshot]:
        with self._lock:
            return self._latest

    def request_reset_ref(self) -> None:
        with self._lock:
            self._reset_ref = True

    def start_recording(self, h5_file) -> None:
        self.writer.reset_episode_stats()
        with self._lock:
            self._h5_file = h5_file
            self._recording = True
            self._frame_count = 0
            self._start_t = self._clock()
            self._warmup_remaining = self.warmup_drop_frames
            self._last_recorded_ts = None
            self._gap_count = 0
            self._max_gap = 0.0
            self._stop_request = None

    def stop_recording(self) -> Optional[RecordingResult]:
        """Stop feeding the writer and hand the open file back. Returns None
        if no episode is open. Does not wait for the writer."""
        with self._lock:
            if self._h5_file is None:
                return None
            result = RecordingResult(self._h5_file, self._frame_count, self._max_gap,
                                     self._gap_count, self._stop_request)
            self._recording = False
            self._h5_file = None
            self._stop_request = None
        return result

    # ── capture thread ───────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            t_start = self._clock()
            try:
                tick = self.rig.grab()
            except Exception as exc:
                self._publish_fatal(f"{type(exc).__name__}: {exc}")
                return
            self._grab_s += self._clock() - t_start

            with self._lock:
                recording, h5 = self._recording, self._h5_file
                if recording and self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                    recording = False
            stop_request = self._record(h5, tick) if recording else None
            if stop_request is not None:
                log.error("ending episode: %s — %s", stop_request.kind, stop_request.detail)

            self._tick_times.append(tick.timestamp)
            fps_meas = ((len(self._tick_times) - 1)
                        / (self._tick_times[-1] - self._tick_times[0])
                        if len(self._tick_times) >= 2
                        and self._tick_times[-1] > self._tick_times[0] else 0.0)
            ot_poses = self.rig.latest_poses()
            stats = self.writer.stats()

            with self._lock:
                if stop_request is not None:
                    self._recording = False
                    self._stop_request = stop_request
                if self._gs_ref is None or self._reset_ref:
                    self._gs_ref = tuple(g.copy() for g in tick.gelsight)
                    self._reset_ref = False
                self._latest = CaptureSnapshot(
                    tick=tick, gs_ref=self._gs_ref, ot_poses=ot_poses,
                    recording=self._recording, frame_count=self._frame_count,
                    elapsed=(tick.timestamp - self._start_t) if self._recording else 0.0,
                    fps_meas=fps_meas, writer=stats, stop_request=self._stop_request)

            self._ticks += 1
            if self._ticks % self.report_every == 0:
                log.info("[%s] fps=%.1f grab=%.1fms queue=%.0f%% write=%.0f MB/s",
                         "REC" if recording else "IDLE", fps_meas,
                         self._grab_s / self.report_every * 1e3,
                         stats.fraction * 100, stats.mean_mb_s)
                self._grab_s = 0.0

            remaining = self.tick_dt - (self._clock() - t_start)
            if remaining > 0:
                self._sleep(remaining)

    def _record(self, h5, tick: Tick) -> Optional[StopRequest]:
        if self._last_recorded_ts is not None:
            gap = tick.timestamp - self._last_recorded_ts
            self._max_gap = max(self._max_gap, gap)
            if gap > self.max_tick_gap_s:
                self._gap_count += 1
                return StopRequest("capture_stall",
                                   f"{gap:.2f}s between ticks (limit {self.max_tick_gap_s}s)")
        try:
            self.writer.submit(h5, tick)
        except WriterOverloaded as exc:
            return StopRequest("overload", str(exc))
        except WriterFault as exc:
            return StopRequest("writer_fault", str(exc))
        self._last_recorded_ts = tick.timestamp
        with self._lock:
            self._frame_count += 1
        health = self.writer.check()
        return StopRequest(*health) if health else None

    def _publish_fatal(self, message: str) -> None:
        log.error("capture stopped: %s", message)
        with self._lock:
            prev = self._latest
            self._latest = CaptureSnapshot(
                tick=prev.tick if prev else Tick(self._clock()),
                gs_ref=self._gs_ref or (), ot_poses=prev.ot_poses if prev else {},
                recording=self._recording, frame_count=self._frame_count,
                elapsed=(self._clock() - self._start_t) if self._recording else 0.0,
                fps_meas=0.0, writer=self.writer.stats(),
                stop_request=self._stop_request, fatal_error=message)
        self._stop.set()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_capture.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/capture.py tests/recorder/test_capture.py
git commit -m "feat(recorder): CaptureLoop with fail-fast stop requests"
```

---

### Task 6: Preflight checks

**Files:**
- Create: `twm/recorder/preflight.py`
- Test: `tests/recorder/test_preflight.py`

**Interfaces:**
- Consumes: `create_episode_file`, `append_ticks`, `synthetic_tick`, `full_rig_tick_nbytes`, `EpisodeWriter`, `queue_capacity_bytes`, `WriterStats`, `WriterConfig`, `RecorderConfig`.
- Produces: `CheckResult(name, ok, detail)`, `failures(results)`, `format_report(results)`, `stale_trackers(poses, active, max_age_s, now) -> List[str]`, `check_disk_free(path, min_free_gb, disk_usage)`, `check_optitrack_fresh(poses, active, max_age_s, now)`, `check_writer_idle(stats)`, `check_write_bandwidth(directory, fps, seconds, margin, writer_config, n_arducam=0, clock=..)`, `run_startup_preflight(config, disk_usage=..)`, `run_episode_preflight(config, poses, stats, now, disk_usage=..)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_preflight.py
from types import SimpleNamespace

import pytest

from twm.recorder.config import RecorderConfig, WriterConfig
from twm.recorder.preflight import (check_disk_free, check_optitrack_fresh,
                                    check_write_bandwidth, check_writer_idle,
                                    failures, format_report,
                                    run_episode_preflight, stale_trackers)
from twm.recorder.writer import WriterStats


def usage(free_gb):
    return lambda path: SimpleNamespace(free=free_gb * 1e9)


def stats(items=0, fault=None):
    return WriterStats(items, 0, 100, 0.0, 0, 0.0, 0.0, 0, 0.0, None, None, fault)


def test_disk_free_uses_nearest_existing_parent(tmp_path):
    missing = tmp_path / "task" / "2026-09-05"
    ok = check_disk_free(missing, 50.0, usage(239.0))
    assert ok.ok and "239.0 GB" in ok.detail
    bad = check_disk_free(missing, 50.0, usage(12.5))
    assert not bad.ok and "12.5 GB" in bad.detail and "50" in bad.detail


def test_optitrack_freshness_reports_missing_and_stale_bodies():
    poses = {"sensor_left": (100.0, [0] * 7), "sensor_right": None}
    r = check_optitrack_fresh(poses, ("sensor_left", "sensor_right"), 2.0, now=101.0)
    assert not r.ok and "sensor_right: no data yet" in r.detail
    r = check_optitrack_fresh(poses, ("sensor_left",), 2.0, now=103.5)
    assert not r.ok and "3.5s old" in r.detail
    assert check_optitrack_fresh(poses, ("sensor_left",), 2.0, now=101.0).ok
    assert stale_trackers(poses, ("sensor_left", "sensor_right"), 10.0, now=111.0) == \
        ["sensor_left silent 11.0s"]        # watchdog: None is not stale


def test_writer_idle_requires_empty_queue_and_no_fault():
    assert check_writer_idle(stats()).ok
    assert not check_writer_idle(stats(items=3)).ok
    assert "disk" in check_writer_idle(stats(fault="OSError: disk")).detail


def test_write_bandwidth_measures_real_writer(tmp_path):
    r = check_write_bandwidth(tmp_path, fps=30, seconds=0.3, margin=0.01,
                              writer_config=WriterConfig())
    assert r.ok, r.detail
    assert "ticks/s" in r.detail
    assert list(tmp_path.iterdir()) == []          # temp file removed
    skipped = check_write_bandwidth(tmp_path, 30, 0.0, 1.5, WriterConfig())
    assert skipped.ok and "skipped" in skipped.detail


def test_episode_preflight_collects_every_failure(tmp_path):
    cfg = RecorderConfig(task="t", data_dir=tmp_path)
    results = run_episode_preflight(cfg, {"sensor_left": None, "sensor_right": None},
                                    stats(items=1), now=0.0, disk_usage=usage(1.0))
    names = [r.name for r in failures(results)]
    assert names == ["disk_free", "optitrack_fresh", "writer_idle"]
    report = format_report(results)
    assert "FAIL disk_free" in report and "FAIL writer_idle" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_preflight.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder.preflight'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/preflight.py
"""Checks that must pass before the recorder starts and before each episode.

Each check is a pure function returning a CheckResult so the app can print
every failure at once instead of the first one it trips over.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from twm.recorder.config import RecorderConfig, WriterConfig
from twm.recorder.frames import full_rig_tick_nbytes, synthetic_tick
from twm.recorder.schema import append_ticks, create_episode_file
from twm.recorder.writer import (EpisodeWriter, WriterOverloaded, WriterStats,
                                 queue_capacity_bytes)

log = logging.getLogger("twm.recorder")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def failures(results: Sequence[CheckResult]) -> List[CheckResult]:
    return [r for r in results if not r.ok]


def format_report(results: Sequence[CheckResult]) -> str:
    return "\n".join(f"  {'ok  ' if r.ok else 'FAIL'} {r.name}: {r.detail}" for r in results)


def _existing_parent(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def check_disk_free(path, min_free_gb: float,
                    disk_usage: Callable[[str], Any] = shutil.disk_usage) -> CheckResult:
    free_gb = disk_usage(str(_existing_parent(Path(path)))).free / 1e9
    return CheckResult("disk_free", free_gb >= min_free_gb,
                       f"{free_gb:.1f} GB free (need {min_free_gb:g} GB)")


def stale_trackers(poses: Mapping[str, Any], active: Sequence[str],
                   max_age_s: float, now: float) -> List[str]:
    """Watchdog view: bodies whose last sample is older than max_age_s.
    A body with no sample at all is not stale (it may simply never have
    been in the volume)."""
    out = []
    for name in active:
        p = poses.get(name)
        age = now - p[0] if p is not None else 0.0
        if age > max_age_s:
            out.append(f"{name} silent {age:.1f}s")
    return out


def check_optitrack_fresh(poses: Mapping[str, Any], active: Sequence[str],
                          max_age_s: float, now: float) -> CheckResult:
    problems = []
    for name in active:
        p = poses.get(name)
        if p is None:
            problems.append(f"{name}: no data yet")
        elif now - p[0] > max_age_s:
            problems.append(f"{name}: last sample {now - p[0]:.1f}s old")
    return CheckResult("optitrack_fresh", not problems,
                       "; ".join(problems) if problems else
                       f"all of {', '.join(active)} fresh (< {max_age_s:g}s)")


def check_writer_idle(stats: WriterStats) -> CheckResult:
    if stats.fault:
        return CheckResult("writer_idle", False, f"writer fault: {stats.fault}")
    return CheckResult("writer_idle", stats.queue_items == 0,
                       f"{stats.queue_items} ticks still queued")


def check_write_bandwidth(directory, fps: int, seconds: float, margin: float,
                          writer_config: WriterConfig, n_arducam: int = 0,
                          clock: Callable[[], float] = time.monotonic,
                          sleep: Callable[[float], None] = time.sleep) -> CheckResult:
    """Push synthetic full-rig ticks through the real writer into a temp
    file in `directory` for `seconds`; require fps × margin ticks/s.

    The synthetic file uses the legacy schema; the requirement is scaled by
    the extra bytes Arducams add so the number still means "the real rig".
    """
    if seconds <= 0:
        return CheckResult("write_bandwidth", True, "skipped")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scale = full_rig_tick_nbytes(3, 2, n_arducam) / full_rig_tick_nbytes(3, 2, 0)
    required = fps * margin * scale
    ticks = [synthetic_tick(0.0, seed=k) for k in range(2)]
    with tempfile.TemporaryDirectory(dir=str(directory), prefix=".twm_bandwidth_") as tmp:
        f, path = create_episode_file(tmp, 0, [], [], fps, task_name="bandwidth_test")
        writer = EpisodeWriter(
            capacity_bytes=queue_capacity_bytes(1.0, fps, ticks[0].nbytes()),
            batch_size=writer_config.batch_size, flush_interval_s=float("inf"),
            sink=append_ticks)
        n = 0
        t0 = clock()
        try:
            while clock() - t0 < seconds:
                try:
                    writer.submit(f, ticks[n % 2])
                    n += 1
                except WriterOverloaded:
                    sleep(0.002)
            writer.drain()
            elapsed = clock() - t0
            f.flush()
            file_mb = Path(path).stat().st_size / 1e6
        finally:
            writer.stop()
            f.close()
    rate = n / elapsed if elapsed else 0.0
    return CheckResult(
        "write_bandwidth", rate >= required,
        f"{rate:.1f} ticks/s sustained, need {required:.1f} "
        f"({fps} fps × {margin:g} margin{f' × {scale:.2f} arducam' if n_arducam else ''}); "
        f"{file_mb / elapsed:.0f} MB/s to disk after compression")


def run_startup_preflight(config: RecorderConfig, n_arducam: int = 0,
                          disk_usage: Callable[[str], Any] = shutil.disk_usage
                          ) -> List[CheckResult]:
    return [
        check_disk_free(config.data_dir, config.disk.min_free_gb, disk_usage),
        check_write_bandwidth(config.data_dir, config.fps, config.disk.bandwidth_test_s,
                              config.disk.min_bandwidth_margin, config.writer,
                              n_arducam=n_arducam),
    ]


def run_episode_preflight(config: RecorderConfig, poses: Mapping[str, Any],
                          stats: WriterStats, now: float,
                          disk_usage: Callable[[str], Any] = shutil.disk_usage
                          ) -> List[CheckResult]:
    return [
        check_disk_free(config.data_dir, config.disk.min_free_gb, disk_usage),
        check_optitrack_fresh(poses, config.active_sensors,
                              config.ot_preflight_max_age_s, now),
        check_writer_idle(stats),
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_preflight.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/preflight.py tests/recorder/test_preflight.py
git commit -m "feat(recorder): startup and episode preflight checks"
```

---

### Task 7: `EpisodeStore`, `EpisodeSummary`, and the health line

**Files:**
- Create: `twm/recorder/episode.py`
- Create: `twm/recorder/monitor.py`
- Test: `tests/recorder/test_episode.py`, `tests/recorder/test_monitor.py`

**Interfaces:**
- Consumes: `create_episode_file`, `WriterStats`.
- Produces: `VALID_ENDINGS`, `EpisodeSummary(...)` with `duration_s`, `attrs()`, `notes()`, `describe()`; `next_episode_number(date_dir)`, `write_log_row(log_path, row)`, `EpisodeStore(data_dir, task, date=None)` with `date_dir`, `log_path`, `next_episode_number()`, `create(...)`, `log(summary)`; `health_level(stats, warn_fraction, min_free_gb)`, `minutes_remaining(stats)`, `health_line(stats, warn_fraction, min_free_gb) -> (text, level)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_episode.py
import csv
from pathlib import Path

import h5py

from twm.recorder.episode import (EpisodeStore, EpisodeSummary,
                                  next_episode_number)


def summary(**kw):
    base = dict(episode_num=3, path=Path("/x/episode_003.h5"), task="t",
                frame_count=90, fps=30, valid=True, ended_by="operator", reason="",
                max_tick_gap_s=0.04, gap_count=0, queue_peak_fraction=0.1,
                writer_mean_mb_s=200.0, has_optitrack=True)
    base.update(kw)
    return EpisodeSummary(**base)


def test_next_episode_number_scans_only_episode_files(tmp_path):
    assert next_episode_number(tmp_path / "missing") == 0
    (tmp_path / "episode_000.h5").touch()
    (tmp_path / "episode_007.h5").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / ".episode_002.h5.tmp").touch()
    assert next_episode_number(tmp_path) == 8


def test_summary_attrs_and_notes():
    ok = summary()
    assert ok.duration_s == 3.0
    assert ok.attrs()["valid"] is True and ok.attrs()["invalid_reason"] == ""
    assert ok.notes() == ""
    bad = summary(valid=False, ended_by="overload", reason="queue full")
    assert bad.attrs()["ended_by"] == "overload"
    assert bad.notes() == "INVALID: overload: queue full"
    assert "INVALID" in bad.describe()
    wd = summary(ended_by="watchdog", reason="sensor_left silent 10.2s")
    assert wd.notes() == "auto-ended: watchdog: sensor_left silent 10.2s"


def test_store_creates_files_in_task_date_dir_and_logs_rows(tmp_path):
    store = EpisodeStore(tmp_path, "pouring", date="2026-09-05")
    assert store.date_dir == tmp_path / "pouring" / "2026-09-05"
    assert store.next_episode_number() == 0
    f, path = store.create(0, ["A"], ["L", "R"], 30)
    f.close()
    assert path == store.date_dir / "episode_000.h5"
    assert store.next_episode_number() == 1
    with h5py.File(path, "r") as g:
        assert g["metadata"].attrs["task"] == "pouring"

    store.log(summary(path=path, task="pouring"))
    store.log(summary(path=path, task="pouring", valid=False, ended_by="disk_low",
                      reason="12 GB free"))
    rows = list(csv.DictReader(open(store.log_path)))
    assert [r["episode"] for r in rows] == ["ep_003", "ep_003"]
    assert rows[0]["notes"] == "" and rows[1]["notes"] == "INVALID: disk_low: 12 GB free"
    assert rows[0]["path"] == "pouring/2026-09-05/episode_000.h5"
    assert rows[0]["optitrack"] == "yes"
```

```python
# tests/recorder/test_monitor.py
from twm.recorder.monitor import health_level, health_line, minutes_remaining
from twm.recorder.writer import WriterStats


def stats(**kw):
    base = dict(queue_items=0, queue_bytes=0, capacity_bytes=1000, peak_fraction=0.0,
                bytes_written=0, write_seconds=0.0, last_batch_ms=0.0, flushes=0,
                file_mb_s=0.0, disk_free_gb=None, overloaded_since=None, fault=None)
    base.update(kw)
    return WriterStats(**base)


def test_levels():
    assert health_level(stats(), 0.25, 50.0) == "ok"
    assert health_level(stats(queue_bytes=300), 0.25, 50.0) == "warn"
    assert health_level(stats(disk_free_gb=20.0), 0.25, 50.0) == "fail"
    assert health_level(stats(fault="x"), 0.25, 50.0) == "fail"


def test_minutes_remaining_needs_a_rate():
    assert minutes_remaining(stats(disk_free_gb=60.0)) is None
    assert minutes_remaining(stats(disk_free_gb=60.0, file_mb_s=100.0)) == 10.0


def test_health_line_text():
    text, level = health_line(stats(queue_bytes=120, bytes_written=200e6,
                                    write_seconds=1.0, disk_free_gb=239.0,
                                    file_mb_s=100.0), 0.25, 50.0)
    assert text == "writer 12% | 200 MB/s | disk 239 GB (~40 min) | OK"
    assert level == "ok"
    text, _ = health_line(stats(fault="OSError: No space"), 0.25, 50.0)
    assert text.endswith("| FAULT: OSError: No space")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_episode.py tests/recorder/test_monitor.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/episode.py
"""Where episodes live on disk, how they are numbered, and how they are logged."""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from twm.recorder.schema import create_episode_file

VALID_ENDINGS = ("operator", "quit", "watchdog")
LOG_COLUMNS = ("saved_at", "task", "date", "episode", "frames", "duration_s",
               "size_mb", "optitrack", "path", "notes")


@dataclass(frozen=True)
class EpisodeSummary:
    episode_num: int
    path: Path
    task: str
    frame_count: int
    fps: int
    valid: bool
    ended_by: str           # operator | quit | watchdog | overload | writer_fault
                            # | disk_low | capture_stall | sensor_error
    reason: str
    max_tick_gap_s: float
    gap_count: int
    queue_peak_fraction: float
    writer_mean_mb_s: float
    has_optitrack: bool

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def attrs(self) -> Dict[str, Any]:
        """metadata attributes written at finalize (schema.write_episode_attrs)."""
        return {
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "valid": bool(self.valid),
            "invalid_reason": "" if self.valid else f"{self.ended_by}: {self.reason}",
            "ended_by": self.ended_by,
            "max_tick_gap_s": self.max_tick_gap_s,
            "gap_count": self.gap_count,
            "queue_peak_fraction": self.queue_peak_fraction,
            "writer_mean_mb_s": self.writer_mean_mb_s,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def notes(self) -> str:
        if not self.valid:
            return f"INVALID: {self.ended_by}: {self.reason}"
        if self.ended_by != "operator":
            return f"auto-ended: {self.ended_by}: {self.reason}".rstrip(": ")
        return ""

    def describe(self) -> str:
        text = (f"Episode {self.episode_num:03d} saved — {self.frame_count} frames, "
                f"{self.duration_s:.1f}s")
        if not self.valid:
            text += f" — INVALID ({self.ended_by}: {self.reason})"
        elif self.ended_by != "operator":
            text += f" ({self.notes()})"
        return text


def next_episode_number(date_dir) -> int:
    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        return 0
    nums = []
    for p in date_dir.iterdir():
        name = p.name
        if name.startswith("episode_") and name.endswith(".h5"):
            try:
                nums.append(int(name[len("episode_"):-len(".h5")]))
            except ValueError:
                continue
    return max(nums) + 1 if nums else 0


def write_log_row(log_path, row: Dict[str, Any]) -> None:
    """Append one CSV row, writing the header if the file is new."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LOG_COLUMNS))
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


class EpisodeStore:
    def __init__(self, data_dir, task: str, date: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.task = task
        self.date = date or time.strftime("%Y-%m-%d")

    @property
    def date_dir(self) -> Path:
        return self.data_dir / self.task / self.date

    @property
    def log_path(self) -> Path:
        return self.data_dir / "dataset_log.csv"

    def next_episode_number(self) -> int:
        return next_episode_number(self.date_dir)

    def create(self, episode_num: int, realsense_serials: Sequence[str],
               gelsight_serials: Sequence[str], fps: int,
               arducam_config=None) -> Tuple[Any, Path]:
        f, path = create_episode_file(str(self.date_dir), episode_num,
                                      list(realsense_serials), list(gelsight_serials),
                                      fps, task_name=self.task,
                                      arducam_config=arducam_config)
        return f, Path(path)

    def log(self, s: EpisodeSummary) -> Path:
        size_mb = round(os.path.getsize(s.path) / 1e6, 1) if Path(s.path).is_file() else 0
        write_log_row(self.log_path, {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": s.task,
            "date": self.date,
            "episode": f"ep_{s.episode_num:03d}",
            "frames": s.frame_count,
            "duration_s": round(s.duration_s, 2),
            "size_mb": size_mb,
            "optitrack": "yes" if s.has_optitrack else "no",
            "path": os.path.relpath(s.path, self.data_dir),
            "notes": s.notes(),
        })
        return self.log_path
```

```python
# twm/recorder/monitor.py
"""One status line for the preview and the log, derived from WriterStats."""
from __future__ import annotations

from typing import Optional, Tuple

from twm.recorder.writer import WriterStats


def health_level(stats: WriterStats, warn_fraction: float,
                 min_free_gb: Optional[float]) -> str:
    if stats.fault:
        return "fail"
    if (min_free_gb is not None and stats.disk_free_gb is not None
            and stats.disk_free_gb < min_free_gb):
        return "fail"
    if stats.fraction > warn_fraction or stats.overloaded_since is not None:
        return "warn"
    return "ok"


def minutes_remaining(stats: WriterStats) -> Optional[float]:
    if stats.disk_free_gb is None or stats.file_mb_s <= 0:
        return None
    return stats.disk_free_gb * 1e3 / stats.file_mb_s / 60.0


def health_line(stats: WriterStats, warn_fraction: float,
                min_free_gb: Optional[float]) -> Tuple[str, str]:
    level = health_level(stats, warn_fraction, min_free_gb)
    parts = [f"writer {stats.fraction:.0%}", f"{stats.mean_mb_s:.0f} MB/s"]
    if stats.disk_free_gb is not None:
        mins = minutes_remaining(stats)
        parts.append(f"disk {stats.disk_free_gb:.0f} GB"
                     + (f" (~{mins:.0f} min)" if mins is not None else ""))
    if stats.fault:
        parts.append(f"FAULT: {stats.fault}")
    else:
        parts.append({"ok": "OK", "warn": "WARN", "fail": "FAIL"}[level])
    return " | ".join(parts), level
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_episode.py tests/recorder/test_monitor.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/episode.py twm/recorder/monitor.py tests/recorder/test_episode.py tests/recorder/test_monitor.py
git commit -m "feat(recorder): EpisodeStore, EpisodeSummary and health line"
```

---

### Task 8: `Recorder` controller and the cv2 GUI

**Files:**
- Create: `twm/recorder/app.py`
- Test: `tests/recorder/test_recorder.py`

**Interfaces:**
- Consumes: everything above; `twm.viz.build_preview_panel`, `draw_projection_overlay`, `load_calibrations`, `CAM_CALIB_NAME`.
- Produces: `Recorder(config, rig, writer, capture, store, clock, disk_usage)` with `recording`, `last_summary`, `start_episode() -> List[CheckResult]`, `end_episode(ended_by="operator", reason="") -> Optional[EpisodeSummary]`, `poll(snapshot) -> Optional[EpisodeSummary]`, `close()`; `load_projection(config) -> Optional[dict]`; `run(config, drivers=None) -> int`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_recorder.py
import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from twm.recorder.app import Recorder
from twm.recorder.capture import CaptureLoop
from twm.recorder.config import RecorderConfig, WriterConfig
from twm.recorder.episode import EpisodeStore
from twm.recorder.frames import synthetic_tick
from twm.recorder.schema import append_ticks
from twm.recorder.writer import EpisodeWriter


class FakeRig:
    arducam_config = ()

    def __init__(self):
        self.poses = {"sensor_left": None, "sensor_right": None}
        self.base = synthetic_tick(0.0, seed=1)

    def grab(self):
        t = time.time()
        return type(self.base)(timestamp=t, color=self.base.color, depth=self.base.depth,
                               gelsight=self.base.gelsight, gelsight_ts=(t, t),
                               optitrack={"motherboard": [(t, [0] * 7)]})

    def latest_poses(self):
        return dict(self.poses)

    def fresh(self):
        now = time.time()
        self.poses = {k: (now, [0] * 7) for k in self.poses}

    def close(self):
        self.closed = True


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.005)
    raise AssertionError("timeout")


@pytest.fixture
def parts(tmp_path):
    cfg = RecorderConfig(task="t", data_dir=tmp_path, fps=60, warmup_drop_frames=0,
                         writer=WriterConfig(queue_seconds=0.5, batch_size=2,
                                             overload_sustained_s=0.2),
                         ot_watchdog_timeout_s=0.5)
    rig = FakeRig()
    gate = threading.Event()
    gate.set()

    def sink(f, ticks):
        gate.wait(5)
        append_ticks(f, ticks)

    tick_bytes = rig.grab().nbytes()
    writer = EpisodeWriter(capacity_bytes=tick_bytes * 4, batch_size=2,
                           overload_sustained_s=0.2, sink=sink)
    capture = CaptureLoop(rig, writer, fps=cfg.fps, warmup_drop_frames=0)
    store = EpisodeStore(tmp_path, "t", date="2026-09-05")
    rec = Recorder(cfg, rig, writer, capture, store,
                   disk_usage=lambda p: SimpleNamespace(free=500e9))
    capture.start()
    _wait(capture.latest)
    yield SimpleNamespace(cfg=cfg, rig=rig, writer=writer, capture=capture,
                          store=store, rec=rec, gate=gate)
    rec.close()


def test_start_refuses_when_optitrack_is_silent(parts):
    fails = parts.rec.start_episode()
    assert [f.name for f in fails] == ["optitrack_fresh"]
    assert parts.rec.recording is False


def test_operator_episode_is_valid_and_logged(parts):
    parts.rig.fresh()
    assert parts.rec.start_episode() == []
    assert parts.rec.recording
    _wait(lambda: parts.capture.latest().frame_count >= 4)
    s = parts.rec.end_episode()
    assert s.valid and s.ended_by == "operator" and s.frame_count >= 4
    with h5py.File(s.path, "r") as f:
        assert f["timestamps"].shape[0] == s.frame_count
        assert bool(f["metadata"].attrs["valid"]) is True
        assert f["optitrack/motherboard/pose"].shape[0] > 0
    assert "ep_000" in parts.store.log_path.read_text()
    assert parts.rec.start_episode() == []          # a second episode numbers 001
    assert parts.rec.end_episode().episode_num == 1


def test_overload_auto_ends_as_invalid_without_losing_accepted_ticks(parts):
    parts.rig.fresh()
    parts.gate.clear()
    parts.rec.start_episode()
    snap = _wait(lambda: parts.capture.latest() if parts.capture.latest().stop_request else None)
    assert snap.stop_request.kind == "overload"
    parts.gate.set()
    s = parts.rec.poll(snap)
    assert s is not None and s.valid is False and s.ended_by == "overload"
    with h5py.File(s.path, "r") as f:
        assert f["timestamps"].shape[0] == s.frame_count == 4
        assert f["metadata"].attrs["invalid_reason"].startswith("overload:")
    assert "INVALID: overload" in parts.store.log_path.read_text()
    assert parts.rec.recording is False
    assert parts.rec.poll(parts.capture.latest()) is None   # nothing left to end


def test_watchdog_auto_ends_but_keeps_episode_valid(parts):
    parts.rig.fresh()
    parts.rec.start_episode()
    _wait(lambda: parts.capture.latest().frame_count >= 2)
    parts.rig.poses = {k: (time.time() - 5.0, [0] * 7) for k in parts.rig.poses}
    snap = _wait(lambda: parts.capture.latest()
                 if parts.capture.latest().ot_poses["sensor_left"][0] < time.time() - 4 else None)
    s = parts.rec.poll(snap)
    assert s.valid and s.ended_by == "watchdog" and "silent" in s.reason


def test_close_finalizes_open_episode_as_quit(parts):
    parts.rig.fresh()
    parts.rec.start_episode()
    _wait(lambda: parts.capture.latest().frame_count >= 1)
    parts.rec.close()
    s = parts.rec.last_summary
    assert s.ended_by == "quit" and s.valid
    assert parts.rig.closed
    with h5py.File(s.path, "r") as f:
        assert f["metadata"].attrs["ended_by"] == "quit"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_recorder.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'twm.recorder.app'`

- [ ] **Step 3: Write the implementation**

```python
# twm/recorder/app.py
"""Recorder — the episode state machine — and the cv2 operator loop.

Recorder has no GUI dependency; run() wires hardware, writer, capture and
the preview window together. Keys: s start, e end, r reset GelSight
reference, p toggle projection overlay, q quit.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from twm.recorder.capture import CaptureLoop, CaptureSnapshot
from twm.recorder.config import RecorderConfig, parse_args
from twm.recorder.episode import VALID_ENDINGS, EpisodeStore, EpisodeSummary
from twm.recorder.frames import full_rig_tick_nbytes
from twm.recorder.monitor import health_line
from twm.recorder.preflight import (CheckResult, failures, format_report,
                                    run_episode_preflight, run_startup_preflight,
                                    stale_trackers)
from twm.recorder.rig import Drivers, SensorRig
from twm.recorder.schema import count_optitrack_samples, write_episode_attrs
from twm.recorder.writer import EpisodeWriter, WriterFault, queue_capacity_bytes

log = logging.getLogger("twm.recorder")


@dataclass
class OpenEpisode:
    num: int
    path: Path
    h5: Any


class Recorder:
    """Owns the open episode. Every transition happens on the caller's thread."""

    def __init__(self, config: RecorderConfig, rig, writer: EpisodeWriter,
                 capture: CaptureLoop, store: EpisodeStore,
                 clock: Callable[[], float] = time.time,
                 disk_usage: Callable[[str], Any] = shutil.disk_usage):
        self.config = config
        self.rig = rig
        self.writer = writer
        self.capture = capture
        self.store = store
        self._clock = clock
        self._disk_usage = disk_usage
        self._open: Optional[OpenEpisode] = None
        self.last_summary: Optional[EpisodeSummary] = None
        self._closed = False

    @property
    def recording(self) -> bool:
        return self._open is not None

    def start_episode(self) -> List[CheckResult]:
        """Run the episode preflight and start recording. Returns the failed
        checks; an empty list means recording started."""
        if self._open is not None:
            return []
        snap = self.capture.latest()
        poses = snap.ot_poses if snap else {}
        failed = failures(run_episode_preflight(self.config, poses, self.writer.stats(),
                                                self._clock(), self._disk_usage))
        if failed:
            log.error("cannot start episode:\n%s", format_report(failed))
            return failed
        num = self.store.next_episode_number()
        h5, path = self.store.create(num, self.config.realsense_serials,
                                     list(self.config.gelsight_serials.values()),
                                     self.config.fps,
                                     arducam_config=self.rig.arducam_config or None)
        self._open = OpenEpisode(num, path, h5)
        self.capture.start_recording(h5)
        log.info("recording episode %03d → %s", num, path)
        return []

    def end_episode(self, ended_by: str = "operator", reason: str = "") -> Optional[EpisodeSummary]:
        """Stop feeding, drain the writer, stamp validity, close, log."""
        if self._open is None:
            return None
        ep, self._open = self._open, None
        result = self.capture.stop_recording()
        frame_count = result.frame_count if result else 0
        if result and result.stop_request and ended_by in VALID_ENDINGS:
            ended_by, reason = result.stop_request.kind, result.stop_request.detail
        try:
            self.writer.drain()
        except WriterFault as exc:
            if ended_by in VALID_ENDINGS:
                ended_by, reason = "writer_fault", str(exc)
        stats = self.writer.stats()
        has_ot = False
        try:
            has_ot = count_optitrack_samples(ep.h5) > 0
        except Exception:
            pass
        summary = EpisodeSummary(
            episode_num=ep.num, path=ep.path, task=self.config.task,
            frame_count=frame_count, fps=self.config.fps,
            valid=ended_by in VALID_ENDINGS, ended_by=ended_by, reason=reason,
            max_tick_gap_s=result.max_gap_s if result else 0.0,
            gap_count=result.gap_count if result else 0,
            queue_peak_fraction=stats.peak_fraction,
            writer_mean_mb_s=stats.mean_mb_s, has_optitrack=has_ot)
        try:
            write_episode_attrs(ep.h5, summary.attrs())
        except Exception as exc:
            log.warning("could not write episode attrs: %s", exc)
        finally:
            try:
                ep.h5.close()
            except Exception as exc:
                log.warning("could not close %s cleanly: %s", ep.path, exc)
        self.store.log(summary)
        (log.error if not summary.valid else log.info)(summary.describe())
        self.last_summary = summary
        return summary

    def poll(self, snapshot: Optional[CaptureSnapshot]) -> Optional[EpisodeSummary]:
        """Call once per GUI tick. Ends the episode on capture events."""
        if snapshot is None or self._open is None:
            return None
        if snapshot.fatal_error:
            return self.end_episode("sensor_error", snapshot.fatal_error)
        if snapshot.stop_request:
            return self.end_episode(snapshot.stop_request.kind, snapshot.stop_request.detail)
        stale = stale_trackers(snapshot.ot_poses, self.config.active_sensors,
                               self.config.ot_watchdog_timeout_s, self._clock())
        if stale:
            return self.end_episode("watchdog", "OptiTrack " + ", ".join(stale))
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.end_episode("quit")
        except Exception:
            log.exception("could not finalize the open episode")
        self.capture.stop()
        self.writer.stop()
        self.rig.close()


# ── projection overlay (preview only) ────────────────────────────────────────

def load_projection(config: RecorderConfig) -> Optional[Dict[str, Any]]:
    """GelSight→camera overlay calibration, or None if unavailable."""
    if not config.show_projection:
        return None
    from twm.viz import CAM_CALIB_NAME, load_calibrations
    calib_dir = Path(__file__).resolve().parent.parent / "calibration" / "result"
    try:
        cam_calibs, gel_left, gel_right = load_calibrations(
            [calib_dir / CAM_CALIB_NAME[i] for i in range(3)],
            calib_dir / "T_gel_to_rigid_left.json",
            calib_dir / "T_gel_to_rigid_right.json")
    except Exception as exc:
        log.warning("projection overlay disabled (%s)", exc)
        return None
    cams = []
    for calib in cam_calibs:
        serial = calib["camera_serial"]
        if serial not in config.realsense_serials:
            log.warning("projection: camera %s not in realsense_serials, skipped", serial)
            continue
        cams.append({"index": config.realsense_serials.index(serial),
                     "T_mocap_to_cam": calib["T_mocap_to_cam"],
                     "intrinsics": calib["intrinsics"]})
    log.info("projection overlay: %d cameras calibrated (press p to toggle)", len(cams))
    return {"cams": cams, "gel_left": gel_left, "gel_right": gel_right} if cams else None


# ── operator loop ────────────────────────────────────────────────────────────

def run(config: RecorderConfig, drivers: Optional[Drivers] = None) -> int:
    n_arducam = 2 if config.use_arducam else 0
    log.info("task %s → %s", config.task, config.data_dir / config.task)
    results = run_startup_preflight(config, n_arducam=n_arducam)
    log.info("startup preflight:\n%s", format_report(results))
    if failures(results):
        log.error("startup preflight failed; fix the above and retry")
        return 2

    rig = SensorRig.open(config, drivers)
    rig.wait_ready(config.startup_timeout_s, config.settle_s)
    log.info("all sensors ready")

    tick_bytes = full_rig_tick_nbytes(len(rig.realsense), 2, len(rig.arducam))
    writer = EpisodeWriter(
        capacity_bytes=queue_capacity_bytes(config.writer.queue_seconds, config.fps, tick_bytes),
        batch_size=config.writer.batch_size,
        flush_interval_s=config.writer.flush_interval_s,
        overload_fraction=config.writer.overload_fraction,
        overload_sustained_s=config.writer.overload_sustained_s,
        min_free_gb=config.disk.min_free_gb)
    capture = CaptureLoop(rig, writer, fps=config.fps,
                          warmup_drop_frames=config.warmup_drop_frames,
                          max_tick_gap_s=config.writer.max_tick_gap_s)
    store = EpisodeStore(config.data_dir, config.task)
    recorder = Recorder(config, rig, writer, capture, store)
    try:
        capture.start()
        return _gui_loop(config, recorder, capture, rig, load_projection(config))
    finally:
        recorder.close()


def _gui_loop(config, recorder: Recorder, capture: CaptureLoop, rig,
              projection: Optional[Dict[str, Any]]) -> int:
    import cv2
    from twm.viz import build_preview_panel, draw_projection_overlay

    log.info("controls: s start | e end | r reset diff ref | p projection | q quit")
    arducam_labels = rig.arducam_labels() or None
    show_projection = projection is not None
    preview_dt, gui_dt = 1.0 / config.preview_fps, 1.0 / 30.0
    last_preview_t, panel = 0.0, None

    def draw_health(panel, snap):
        text, level = health_line(snap.writer, config.writer.warn_fraction,
                                  config.disk.min_free_gb)
        color = {"ok": (80, 200, 80), "warn": (0, 200, 255), "fail": (0, 0, 255)}[level]
        cv2.putText(panel, text, (8, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    while True:
        t0 = time.time()
        snap = capture.latest()
        if snap is None:
            time.sleep(0.005)
            continue
        recorder.poll(snap)
        if snap.fatal_error:
            log.error("capture stopped: %s", snap.fatal_error)
            return 1

        if panel is None or t0 - last_preview_t >= preview_dt:
            tick = snap.tick
            panel = build_preview_panel(
                list(tick.color), list(tick.gelsight), list(snap.gs_ref), snap.ot_poses,
                snap.recording, snap.frame_count, snap.elapsed,
                snap.writer.queue_items, snap.fps_meas, task_name=config.task,
                arducam_frames=list(tick.arducam) or None,
                arducam_labels=arducam_labels)
            if show_projection and projection:
                draw_projection_overlay(panel, snap.ot_poses, projection["cams"],
                                        projection["gel_left"], projection["gel_right"])
            draw_health(panel, snap)
            last_preview_t = t0
        cv2.imshow("TWM Data Collection", panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not recorder.recording:
            failed = recorder.start_episode()
            if failed:
                log.error("press s again once these pass:\n%s", format_report(failed))
        elif key == ord("e") and recorder.recording:
            recorder.end_episode("operator")
        elif key == ord("r"):
            capture.request_reset_ref()
            log.info("GelSight diff reference reset (queued)")
        elif key == ord("p"):
            if projection:
                show_projection = not show_projection
                panel = None
                log.info("projection overlay %s", "ON" if show_projection else "OFF")
            else:
                log.info("projection overlay unavailable (no calibration loaded)")
        elif key == ord("q"):
            if recorder.recording:
                recorder.end_episode("quit")
            cv2.destroyAllWindows()
            return 0

        remaining = gui_dt - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recorder/test_recorder.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add twm/recorder/app.py tests/recorder/test_recorder.py
git commit -m "feat(recorder): Recorder state machine and operator loop"
```

---

### Task 9: Compatibility facade and caller migration

**Files:**
- Rewrite: `twm/data_collection.py` (whole file replaced; target < 120 lines)
- Modify: `twm/sensor_camera.py:325-400` (`record_verification`)
- Modify: `twm/scripts/test_crash_leaves_readable_h5.py:26-60` (child script)
- Modify: `tests/test_hdf5_writer.py:140-146` (HDF5Writer stop test stays; uses adapter)
- Delete: `tests/test_twm_capture_loop.py` (superseded by `tests/recorder/test_capture.py`, `test_rig.py`, `test_recorder.py`)
- Create: `twm/recorder/__main__.py`
- Test: `tests/recorder/test_facade.py`

**Interfaces:**
- Produces (facade): `REALSENSE_SERIALS` (list), `GELSIGHT_SERIALS`, `DATA_DIR` (str), `FPS`, `create_episode_file`, `append_camera_frame`, `append_camera_frames_batch`, `flush_optitrack_to_hdf5`, `HDF5Writer` adapter, `log_episode`, `next_episode_number`, `make_preview`, `make_optitrack_panel`, `TRACKER_COLORS`, `load_calibrations`, `draw_projection_overlay`, `CAM_CALIB_NAME`, `main`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/recorder/test_facade.py
import numpy as np
import pytest

import twm.data_collection as dc
from twm.recorder.writer import WriterOverloaded


def test_facade_exports_legacy_names():
    assert isinstance(dc.REALSENSE_SERIALS, list) and len(dc.REALSENSE_SERIALS) == 3
    assert isinstance(dc.DATA_DIR, str)
    assert dc.FPS == 30
    for name in ("create_episode_file", "append_camera_frame", "append_camera_frames_batch",
                 "flush_optitrack_to_hdf5", "log_episode", "next_episode_number",
                 "make_preview", "make_optitrack_panel", "TRACKER_COLORS", "main"):
        assert callable(getattr(dc, name)) or name == "TRACKER_COLORS"


def test_hdf5writer_adapter_raises_instead_of_dropping(tmp_path):
    f, _ = dc.create_episode_file(str(tmp_path), 0, ["A", "B", "C"], ["L", "R"], 30)
    w = dc.HDF5Writer(maxsize=2, batch_size=1)
    color = [np.zeros((480, 640, 3), np.uint8)] * 3
    depth = [np.zeros((480, 640), np.uint16)] * 3
    gs = [np.zeros((480, 640, 3), np.uint8)] * 2
    for t in range(3):
        try:
            w.enqueue(f, color, depth, gs, float(t), gs_timestamps=[None, t - 0.01])
        except WriterOverloaded:
            break
    w.flush()
    w.stop()
    assert w.dropped_frames == 0
    assert f["timestamps"].shape[0] >= 1
    np.testing.assert_allclose(f["gelsight/left/timestamps"][0], 0.0)
    f.close()


def test_log_episode_writes_legacy_row(tmp_path):
    f, path = dc.create_episode_file(str(tmp_path / "t" / "d"), 4, [], [], 30)
    f.close()
    dc.log_episode(str(tmp_path), "t", 4, path, 60, 30, has_optitrack=False, notes="x")
    text = (tmp_path / "dataset_log.csv").read_text()
    assert "ep_004" in text and ",no," in text and text.strip().endswith(",x")
    assert dc.next_episode_number(str(tmp_path / "t" / "d")) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recorder/test_facade.py -q`
Expected: FAIL — `test_hdf5writer_adapter_raises_instead_of_dropping` fails because the old `HDF5Writer` drops silently (`dropped_frames` > 0 or timestamps mismatch); `test_facade_exports_legacy_names` fails on `isinstance(dc.DATA_DIR, str)` only if types changed — it may pass. At least one test must fail.

- [ ] **Step 3: Replace `twm/data_collection.py`**

```python
#!/usr/bin/env python3
"""TWM data collection — compatibility facade.

The recorder lives in `twm.recorder` (see its package docstring). This
module keeps the names older scripts import and forwards `main()`.

    python -m twm.data_collection --task <task_name>
    python -m twm.recorder --help
"""
from __future__ import annotations

import os
import time
from typing import Optional

from twm.recorder import config as _config
from twm.recorder.episode import next_episode_number, write_log_row  # noqa: F401
from twm.recorder.frames import full_rig_tick_nbytes
from twm.recorder.schema import (append_optitrack, append_ticks,  # noqa: F401
                                 create_episode_file, tick_from_legacy)
from twm.recorder.writer import EpisodeWriter, WriterOverloaded  # noqa: F401
from twm.viz import (CAM_CALIB_NAME, TRACKER_COLORS,  # noqa: F401
                     build_preview_panel as make_preview,
                     draw_projection_overlay, load_calibrations,
                     make_optitrack_panel)

REALSENSE_SERIALS = list(_config.REALSENSE_SERIALS)
GELSIGHT_SERIALS = dict(_config.GELSIGHT_SERIALS)
DATA_DIR = str(_config.DATA_DIR)
FPS = _config.FPS

flush_optitrack_to_hdf5 = append_optitrack


def append_camera_frames_batch(f, batch):
    """Legacy batch of (color, depth, gs, timestamp[, gs_ts[, arducam[, arducam_ts]]])."""
    append_ticks(f, [tick_from_legacy(item) for item in batch])


def append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp):
    append_camera_frames_batch(f, [(color_frames, depth_frames, gs_frames, timestamp)])


class HDF5Writer:
    """Legacy adapter over EpisodeWriter. Never drops: `enqueue` raises
    WriterOverloaded when `maxsize` ticks are already buffered."""

    FLUSH_INTERVAL_S = 10.0

    def __init__(self, maxsize: int = 90, batch_size: int = 10):
        self._writer = EpisodeWriter(
            capacity_bytes=maxsize * full_rig_tick_nbytes(),
            batch_size=batch_size, flush_interval_s=self.FLUSH_INTERVAL_S)

    def enqueue(self, f, color_frames, depth_frames, gs_frames, timestamp,
                gs_timestamps=None, arducam_frames=None, arducam_timestamps=None):
        self._writer.submit(f, tick_from_legacy(
            (color_frames, depth_frames, gs_frames, timestamp, gs_timestamps,
             arducam_frames, arducam_timestamps)))

    def flush(self):
        self._writer.drain()

    def stop(self):
        self._writer.stop()

    def stats(self):
        return self._writer.stats()

    @property
    def queue_size(self) -> int:
        return self._writer.stats().queue_items

    @property
    def dropped_frames(self) -> int:
        return 0


def log_episode(data_dir, task_name, episode_num, h5_path, frame_count, fps,
                has_optitrack=True, notes=""):
    """Append one row to <data_dir>/dataset_log.csv (legacy signature)."""
    write_log_row(os.path.join(data_dir, "dataset_log.csv"), {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task_name,
        "date": time.strftime("%Y-%m-%d"),
        "episode": f"ep_{episode_num:03d}",
        "frames": frame_count,
        "duration_s": round(frame_count / fps, 2) if fps else 0,
        "size_mb": round(os.path.getsize(h5_path) / 1e6, 1) if os.path.isfile(h5_path) else 0,
        "optitrack": "yes" if has_optitrack else "no",
        "path": os.path.relpath(h5_path, data_dir),
        "notes": notes,
    })


def main(argv: Optional[list] = None) -> int:
    from twm.recorder.app import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create `twm/recorder/__main__.py`**

```python
"""`python -m twm.recorder run --task X` or `python -m twm.recorder bench --dir D`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from twm.recorder.config import WriterConfig
from twm.recorder.preflight import check_write_bandwidth


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bench":
        p = argparse.ArgumentParser(prog="python -m twm.recorder bench",
                                    description="Measure sustained writer throughput "
                                                "into a directory using the real HDF5 path.")
        p.add_argument("--dir", required=True)
        p.add_argument("--seconds", type=float, default=5.0)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--margin", type=float, default=1.5)
        p.add_argument("--arducams", type=int, default=2)
        a = p.parse_args(argv[1:])
        r = check_write_bandwidth(Path(a.dir), a.fps, a.seconds, a.margin, WriterConfig(),
                                  n_arducam=a.arducams)
        print(("ok   " if r.ok else "FAIL ") + r.detail)
        return 0 if r.ok else 1
    if argv and argv[0] == "run":
        argv = argv[1:]
    from twm.recorder.app import main as run_main
    return run_main(argv)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Migrate `twm/sensor_camera.py::record_verification`**

Replace the import and the writer usage (lines 330 and 362-379) with:

```python
    from twm.recorder.frames import Tick
    from twm.recorder.schema import create_episode_file
    from twm.recorder.writer import EpisodeWriter, queue_capacity_bytes
```
and
```python
        writer = EpisodeWriter(
            capacity_bytes=queue_capacity_bytes(3.0, slots[0].fps,
                                                slots[0].width * slots[0].height * 3 * 2))
        deadline = time.monotonic() + duration
        tick_dt = 1.0 / slots[0].fps
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            samples = [stream.get_frame_with_timestamp(timeout=0.5)
                       for stream in streams]
            t = time.time()
            writer.submit(h5_file, Tick(
                timestamp=t,
                arducam=tuple(s[0] for s in samples),
                arducam_ts=tuple(t if s[1] is None else float(s[1]) for s in samples)))
            next_tick += tick_dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        writer.drain()
        writer.stop()
        dropped_frames = 0
        writer = None
```
Keep `report["dropped_frames"] = dropped_frames` and the `no_writer_drops` check (a WriterOverloaded now propagates as a failure instead of being counted).

- [ ] **Step 6: Migrate the crash test child**

In `twm/scripts/test_crash_leaves_readable_h5.py` replace the child's imports and writer lines:

```python
from twm.recorder.frames import Tick
from twm.recorder.writer import EpisodeWriter
...
w = EpisodeWriter(capacity_bytes=int(2e9), batch_size=4, flush_interval_s=interval)
...
while time.time() - t0 < 4.0:
    t = time.time()
    w.submit(f, Tick(t, color=tuple(colors), depth=tuple(depths),
                     gelsight=tuple(gels), gelsight_ts=(t, t)))
    n += 1
    time.sleep(0.05)
w.drain()
print(f"child: enqueued {{n}} frames, flushes={{w.stats().flushes}}", flush=True)
```
Remove the `HDF5Writer.FLUSH_INTERVAL_S = interval` line.

- [ ] **Step 7: Delete the superseded test and run everything**

```bash
git rm -q tests/test_twm_capture_loop.py
python -m pytest tests -q
python twm/scripts/test_crash_leaves_readable_h5.py
python -m twm.data_collection --help | head -3
wc -l twm/data_collection.py
```
Expected: all tests pass (old `tests/test_hdf5_writer.py` and `tests/test_arducam.py` included), the crash script prints both outcomes and exits 0, `--help` prints usage, facade under 120 lines.

- [ ] **Step 8: Commit**

```bash
git add -A twm/data_collection.py twm/recorder/__main__.py twm/sensor_camera.py twm/scripts/test_crash_leaves_readable_h5.py tests/recorder/test_facade.py
git commit -m "refactor(twm): data_collection becomes a facade over twm.recorder"
```

---

### Task 10: Documentation

**Files:**
- Modify: `twm/README.md` (Hardware note, "Collecting Data", add "Overload policy and preflight")
- Test: `python -m pytest twm/scripts/test_doc_references.py -q` if that script checks README paths (run it; fix any reference it flags).

- [ ] **Step 1: Update the README**

Replace the sentence `Camera serials are set at the top of data_collection.py (REALSENSE_SERIALS, GELSIGHT_SERIALS).` with:

```markdown
Camera serials, the data root, and every threshold live in
`twm/recorder/config.py` (`RecorderConfig`). Override at the command line:
`--data_dir`, `--queue_seconds`, `--min_free_gb`, `--no_bandwidth_test`.
```

After the "Typical workflow" list add:

```markdown
### What happens when the disk cannot keep up

The recorder never drops frames from the middle of an episode. The writer
queue is bounded in bytes (3 s of ticks ≈ 750 MB by default). If it stays
above 50 % for 3 s, fills up, hits a write error, or free disk falls below
50 GB, the current episode is finalized immediately, marked
`metadata.attrs["valid"] = False` with an `invalid_reason`, logged with
`notes = "INVALID: ..."`, and recording stops. The status line at the bottom
of the preview shows `writer <queue %> | <MB/s> | disk <GB> (~min left) | OK/WARN/FAIL`.

Before every episode the recorder checks free disk, OptiTrack freshness for
the active bodies, and that the writer is idle. At startup it also writes
two seconds of synthetic frames through the real pipeline and refuses to
run below 45 ticks/s (30 fps × 1.5). Run the same test by hand:

    python -m twm.recorder bench --dir /media/yxma/Disk1/twm/data --seconds 5

Episode metadata gained `valid`, `invalid_reason`, `ended_by`,
`max_tick_gap_s`, `gap_count`, `queue_peak_fraction`, `writer_mean_mb_s`.
OptiTrack poses are now written continuously with each batch rather than
only at episode end, so episodes longer than ~7 minutes keep every sample.

### Library layout

`twm/recorder/`: `config` → `rig` (hardware) → `capture` (30 Hz thread) →
`writer` (HDF5 thread) with `schema` (layout), `preflight`, `monitor`,
`episode` (paths + CSV log) and `app` (state machine + preview).
`twm/data_collection.py` is a thin compatibility facade.
```

- [ ] **Step 2: Run the doc reference check and the full suite**

Run: `python twm/scripts/test_doc_references.py; python -m pytest tests -q`
Expected: no missing references reported; all tests pass.

- [ ] **Step 3: Commit**

```bash
git add twm/README.md
git commit -m "docs(twm): describe the recorder library and its fail-fast policy"
```

---

### Task 11: Final verification

- [ ] **Step 1: Full suite, twice (threading flakiness shows on the second run)**

Run: `python -m pytest tests -q && python -m pytest tests/recorder -q -p no:cacheprovider`
Expected: all pass both times.

- [ ] **Step 2: Crash safety and bench**

Run:
```bash
python twm/scripts/test_crash_leaves_readable_h5.py
python -m twm.recorder bench --dir /tmp --seconds 2 --arducams 2
python -m twm.data_collection --help
python -c "import twm.data_collection as d, twm.visualize, twm.sensor_camera; print('imports ok')"
```
Expected: crash test passes; bench prints a ticks/s figure; help prints; imports ok.

- [ ] **Step 3: Sizes**

Run: `wc -l twm/data_collection.py twm/recorder/*.py`
Expected: facade < 120 lines; no module in `twm/recorder/` over ~400 lines.

- [ ] **Step 4: Report** the bench number, the test counts, and anything skipped.

## Self-Review

- Spec coverage: fail-fast (Task 3, 5, 8), byte-bounded queue (3), preflight (6, 8), runtime monitoring (3, 7, 8), OptiTrack per tick (4, 2), metadata (7, 8), backward compat (9), docs (10). Covered.
- Placeholder scan: none.
- Type consistency: `writer.check()` returns `Optional[Tuple[str, str]]` (Task 3) and is unpacked into `StopRequest(*health)` (Task 5); `RecordingResult.stop_request` consumed in Task 8; `WriterStats` positional order in `test_preflight.stats()` matches the dataclass field order in Task 3; `EpisodeStore.create(..., arducam_config=)` matches Task 8's call; `rig.arducam_config` is a tuple (Task 4) and `or None` in Task 8 keeps `create_episode_file`'s "falsy → no group" behavior.

---

### Task 12: Serial-keyed Arducam identification with a known left/right mapping

**Why:** The two Arducams now report distinct serials (`TWML0001` on the left
sensor, `TWMR0001` on the right; confirmed with `udevadm info` on
2026-09-05: `/dev/video10` `ID_SERIAL_SHORT=TWML0001`, `/dev/video6`
`ID_SERIAL_SHORT=TWMR0001`, both `ID_MODEL=Arducam-B0578-2.3MP-GS`, both
offering MJPG 640x480 at 30 fps). Selecting by serial survives re-plugging
into any USB port; the topology-path selector stays as a fallback for cameras
without a unique serial.

**Files:**
- Modify: `twm/sensor_camera.py` (`CameraSlot`, `ResolvedCamera`, `validate_config`, `resolve_slots`, `identify` label)
- Modify: `twm/config/arducam.json`
- Modify: `twm/recorder/schema.py` (`serial` attribute on `arducam/cam*` and in the metadata JSON)
- Modify: `twm/recorder/rig.py` (`arducam_labels` shows serial)
- Modify: `twm/README.md` (Hardware note, "Sensor-camera setup", HDF5 attrs)
- Test: `tests/test_arducam.py` (add tests), `tests/recorder/test_schema.py` (add one test)

**Interfaces:**
- Consumes: `create_episode_file` (Task 2), `SensorRig.arducam_labels` (Task 4).
- Produces: `CameraSlot(slot, id_path="", position="unknown", width, height, fps, pixel_format, serial="")` — at least one of `serial`/`id_path` non-empty; `ResolvedCamera(config, device, reported_serial, device_id_path="")` with properties `serial` (configured serial, else reported) and `id_path` (device path at resolve time, else configured); `resolve_slots` matches on `serial` when set, else on `id_path`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arducam.py`:

```python
def _raw_serial_config():
    return {
        "cameras": [
            {"slot": "cam0", "serial": "TWML0001", "position": "left"},
            {"slot": "cam1", "serial": "TWMR0001", "position": "right"},
        ]
    }


def test_validate_config_accepts_serial_keyed_cameras_without_paths():
    cam0, cam1 = validate_config(_raw_serial_config())
    assert (cam0.serial, cam0.position, cam0.id_path) == ("TWML0001", "left", "")
    assert (cam1.serial, cam1.position) == ("TWMR0001", "right")


@pytest.mark.parametrize("cameras, message", [
    ([{"slot": "cam0", "position": "left"}, {"slot": "cam1", "serial": "B", "position": "right"}],
     "serial or id_path"),
    ([{"slot": "cam0", "serial": "A"}, {"slot": "cam1", "serial": "A"}], "serial"),
])
def test_validate_config_rejects_missing_or_duplicate_serials(cameras, message):
    with pytest.raises(ArducamConfigError, match=message):
        validate_config({"cameras": cameras})


def test_resolve_slots_by_serial_ignores_port_and_metadata_nodes():
    slots = validate_config(_raw_serial_config())
    devices = [
        VideoDevice("/dev/video6", "usb-0:12.1", "TWMR0001", True),
        VideoDevice("/dev/video7", "usb-0:12.1", "TWMR0001", False),
        VideoDevice("/dev/video10", "usb-0:12.2", "TWML0001", True),
        VideoDevice("/dev/video11", "usb-0:12.2", "TWML0001", False),
        VideoDevice("/dev/video12", "usb-0:12.3", "2DUPB53G", True),
    ]
    cam0, cam1 = resolve_slots(slots, devices)
    assert (cam0.device, cam0.serial, cam0.position) == ("/dev/video10", "TWML0001", "left")
    assert (cam1.device, cam1.serial, cam1.position) == ("/dev/video6", "TWMR0001", "right")
    assert cam0.id_path == "usb-0:12.2"          # resolved from the device, not the config


def test_resolve_slots_reports_missing_serial_with_inventory():
    slots = validate_config(_raw_serial_config())
    devices = [VideoDevice("/dev/video6", "usb-0:12.1", "TWMR0001", True)]
    with pytest.raises(ArducamConfigError, match="TWML0001.*inventory"):
        resolve_slots(slots, devices)


def test_shipped_config_maps_left_and_right_serials():
    from twm.sensor_camera import DEFAULT_CONFIG_PATH, load_config
    cam0, cam1 = load_config(DEFAULT_CONFIG_PATH)
    assert (cam0.serial, cam0.position) == ("TWML0001", "left")
    assert (cam1.serial, cam1.position) == ("TWMR0001", "right")
    assert (cam0.width, cam0.height, cam0.fps, cam0.pixel_format) == (640, 480, 30, "MJPG")
```

Append to `tests/recorder/test_schema.py`:

```python
def test_arducam_groups_record_configured_serial(tmp_path):
    from twm.sensor_camera import CameraSlot, ResolvedCamera
    cams = (ResolvedCamera(CameraSlot("cam0", serial="TWML0001", position="left"),
                           "/dev/video10", "TWML0001", "usb-0:12.2"),
            ResolvedCamera(CameraSlot("cam1", serial="TWMR0001", position="right"),
                           "/dev/video6", "TWMR0001", "usb-0:12.1"))
    f, path = create_episode_file(str(tmp_path), 0, [], [], 30, arducam_config=cams,
                                  include_legacy=False)
    f.close()
    with h5py.File(path, "r") as g:
        assert g["arducam/cam0"].attrs["serial"] == "TWML0001"
        assert g["arducam/cam0"].attrs["usb_path"] == "usb-0:12.2"
        assert g["arducam/cam1"].attrs["position"] == "right"
        import json
        meta = json.loads(g["metadata"].attrs["arducam_config"])
        assert [m["serial"] for m in meta] == ["TWML0001", "TWMR0001"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_arducam.py tests/recorder/test_schema.py -q`
Expected: the new tests FAIL (`TypeError: __init__() got an unexpected keyword argument 'serial'`, `ArducamConfigError: each camera needs a nonempty id_path`); existing tests still pass.

- [ ] **Step 3: Implement**

In `twm/sensor_camera.py`:

```python
@dataclass(frozen=True)
class CameraSlot:
    slot: str
    id_path: str = ""
    position: str = "unknown"
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "MJPG"
    serial: str = ""          # preferred identity; id_path is the fallback


@dataclass(frozen=True)
class ResolvedCamera:
    config: CameraSlot
    device: str
    reported_serial: str
    device_id_path: str = ""

    @property
    def slot(self) -> str:
        return self.config.slot

    @property
    def serial(self) -> str:
        return self.config.serial or self.reported_serial

    @property
    def id_path(self) -> str:
        return self.device_id_path or self.config.id_path
    # position/width/height/fps/pixel_format properties unchanged
```

In `validate_config`, replace the `id_path` requirement and uniqueness check:

```python
        serial = str(entry.get("serial", "")).strip()
        id_path = str(entry.get("id_path", "")).strip()
        if not serial and not id_path:
            raise ArducamConfigError("each camera needs a nonempty serial or id_path")
        ...
        slots.append(CameraSlot(slot=slot, id_path=id_path, position=position,
                                width=..., height=..., fps=..., pixel_format=pixel_format,
                                serial=serial))
    ...
    serials = [s.serial for s in slots if s.serial]
    if len(serials) != len(set(serials)):
        raise ArducamConfigError("camera serial values must be unique")
    paths = [s.id_path for s in slots if s.id_path]
    if len(paths) != len(set(paths)):
        raise ArducamConfigError("camera id_path values must be unique")
```

In `resolve_slots`, match on the configured key:

```python
    for slot in slots:
        if slot.serial:
            key, wanted = "serial", slot.serial
            matches = [d for d in devices if d.is_capture and d.reported_serial == wanted]
        else:
            key, wanted = "path", slot.id_path
            matches = [d for d in devices if d.is_capture and d.id_path == wanted]
        if not matches:
            raise ArducamConfigError(
                f"{slot.slot} {key} {wanted!r} was not found; inventory: {_inventory(devices)}")
        if len(matches) > 1:
            raise ArducamConfigError(
                f"{slot.slot} {key} {wanted!r} matched multiple capture nodes; "
                f"inventory: {_inventory(devices)}")
        device = matches[0]
        resolved.append(ResolvedCamera(slot, device.device, device.reported_serial,
                                       device.id_path))
```

In `identify`, draw `f"{camera.serial}  {camera.id_path}"` on the second text line instead of only `camera.id_path`.

Replace `twm/config/arducam.json` with:

```json
{
  "cameras": [
    {
      "slot": "cam0",
      "serial": "TWML0001",
      "position": "left",
      "width": 640,
      "height": 480,
      "fps": 30,
      "pixel_format": "MJPG"
    },
    {
      "slot": "cam1",
      "serial": "TWMR0001",
      "position": "right",
      "width": 640,
      "height": 480,
      "fps": 30,
      "pixel_format": "MJPG"
    }
  ]
}
```

In `twm/recorder/schema.py`, inside `create_episode_file`: add `"serial": getattr(c, "serial", "")` to each metadata JSON entry, and `g.attrs["serial"] = getattr(c, "serial", "")` beside the other `arducam/<slot>` attributes.

In `twm/recorder/rig.py`: `arducam_labels` returns `f"{c.slot} {getattr(c, 'serial', '') or c.id_path} {c.position}"`.

In `twm/README.md`: replace the Hardware paragraph beginning "The two Arducams both report the factory serial `SN001`" with:

```markdown
The two Arducams are selected by USB serial in `config/arducam.json`
(`TWML0001` = left, `TWMR0001` = right), so they can be plugged into any port.
A camera without a unique serial can still be selected by its USB topology
path (`id_path`). Their HDF5 groups stay `arducam/cam0` (left) and
`arducam/cam1` (right); each group's `position` and `serial` attributes say
which is which.
```
and in the "Sensor-camera setup" section replace "inspect the current topology paths and update `config/arducam.json` if necessary" with "run the identification preview to confirm the serial → side mapping". Add `serial` to the attribute list in the HDF5 description.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_arducam.py tests/recorder/test_schema.py tests/test_hdf5_writer.py tests/recorder/test_rig.py tests/test_visualize.py -q`
Expected: all pass.

- [ ] **Step 5: Verify on the attached hardware**

Run: `python -m twm.sensor_camera verify --duration 3 --output /tmp/twm_arducam_verification.h5 --force`
Expected: JSON report with `"ok": true`; both `arducam/cam0` (TWML0001) and `arducam/cam1` (TWMR0001) present. If the cameras are not attached when this runs, report that instead of failing the task.

- [ ] **Step 6: Commit**

```bash
git add twm/sensor_camera.py twm/config/arducam.json twm/recorder/schema.py twm/recorder/rig.py twm/README.md tests/test_arducam.py tests/recorder/test_schema.py
git commit -m "feat(twm): select Arducams by serial with a known left/right mapping"
```
