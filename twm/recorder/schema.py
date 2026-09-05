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
            "serial": getattr(c, "serial", ""),
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
            g.attrs["serial"] = getattr(c, "serial", "")
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
