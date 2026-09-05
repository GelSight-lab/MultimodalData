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

import collections
import json
import os
import queue
import threading
import time
import numpy as np
import h5py
import hdf5plugin

def _gs_frame_ts(gs_stream):
    """Return (frame, capture_timestamp) for a GelSight stream, using the
    per-frame capture timestamp when available (USBVideoStream) and falling
    back to (frame, None) for streams/dummies without it."""
    fn = getattr(gs_stream, "get_frame_with_timestamp", None)
    if fn is not None:
        return fn()
    return gs_stream.get_frame(), None


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_episode_file(date_dir, episode_num, realsense_serials, gelsight_serials,
                        fps, task_name="", arducam_config=None,
                        include_legacy=True):
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
    meta.attrs["task"] = task_name

    if arducam_config:
        camera_meta = [{
            "slot": camera.slot,
            "id_path": camera.id_path,
            "position": camera.position,
            "device_at_recording": getattr(camera, "device", ""),
            "reported_serial": getattr(camera, "reported_serial", ""),
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "pixel_format": camera.pixel_format,
        } for camera in arducam_config]
        meta.attrs["arducam_config"] = json.dumps(camera_meta, sort_keys=True)

    # camera timestamps (one per main-loop tick)
    f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)

    # BLOSC LZ4 compression — benchmarked at ~100fps overhead vs 30fps capture rate.
    _blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

    if include_legacy:
        for i in range(3):
            g = f.create_group(f"realsense/cam{i}")
            g.create_dataset("color", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                             dtype=np.uint8,  chunks=(1, 480, 640, 3), **_blosc)
            g.create_dataset("depth", shape=(0, 480, 640),    maxshape=(None, 480, 640),
                             dtype=np.uint16, chunks=(1, 480, 640),    **_blosc)

    # GelSight — frames + per-sensor CAPTURE timestamps. The GelSight Mini
    # streams ~18.75 fps (hardware ceiling, < the 30 fps camera tick), so its
    # frames don't line up 1:1 with the main-loop `timestamps`. Each frame
    # carries the time it was actually captured (post-grab), letting downstream
    # code align tactile to cameras by nearest timestamp instead of by index.
    if include_legacy:
        for name in ["left", "right"]:
            g = f.create_group(f"gelsight/{name}")
            g.create_dataset("frames", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                             dtype=np.uint8, chunks=(1, 480, 640, 3), **_blosc)
            g.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)

    # OptiTrack — per-tracker timestamps + poses
    if include_legacy:
        for name in ["motherboard", "sensor_left", "sensor_right"]:
            g = f.create_group(f"optitrack/{name}")
            g.create_dataset("timestamps", shape=(0,),    maxshape=(None,),    dtype=np.float64)
            g.create_dataset("pose",       shape=(0, 7),  maxshape=(None, 7),  dtype=np.float64)

    if arducam_config:
        for camera in arducam_config:
            g = f.create_group(f"arducam/{camera.slot}")
            g.create_dataset(
                "frames",
                shape=(0, camera.height, camera.width, 3),
                maxshape=(None, camera.height, camera.width, 3),
                dtype=np.uint8,
                chunks=(1, camera.height, camera.width, 3),
                **_blosc,
            )
            g.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
            g.attrs["usb_path"] = camera.id_path
            g.attrs["device_at_recording"] = getattr(camera, "device", "")
            g.attrs["reported_serial"] = getattr(camera, "reported_serial", "")
            g.attrs["position"] = camera.position
            g.attrs["width"] = camera.width
            g.attrs["height"] = camera.height
            g.attrs["fps"] = camera.fps
            g.attrs["pixel_format"] = camera.pixel_format

    return f, path


def append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp):
    """Append one timestep — thin wrapper around the batch writer."""
    append_camera_frames_batch(f, [(color_frames, depth_frames, gs_frames, timestamp)])


def append_camera_frames_batch(f, batch):
    """
    Write a batch of frames to HDF5 in one resize+write per dataset.

    batch: list of (color_frames, depth_frames, gs_frames, timestamp,
                    gs_timestamps) tuples
           color_frames:  list of 3 arrays (480, 640, 3) uint8
           depth_frames:  list of 3 arrays (480, 640) uint16
           gs_frames:     list of 2 arrays (480, 640, 3) uint8
           timestamp:     float (main-loop tick time)
           gs_timestamps: list of 2 floats (per-sensor capture time) — optional
                          for backward compat; falls back to `timestamp`.

    One resize() call per dataset instead of one per frame — reduces HDF5
    b-tree metadata overhead by len(batch)×.
    """
    if not batch:
        return
    n  = f["timestamps"].shape[0]
    nb = len(batch)

    ts = np.array([b[3] for b in batch], dtype=np.float64)
    f["timestamps"].resize(n + nb, axis=0)
    f["timestamps"][n:] = ts

    if "realsense" in f:
        for i in range(3):
            color_batch = np.stack([b[0][i] for b in batch])   # (nb, 480, 640, 3)
            depth_batch = np.stack([b[1][i] for b in batch])   # (nb, 480, 640)
            ds_c = f[f"realsense/cam{i}/color"]
            ds_d = f[f"realsense/cam{i}/depth"]
            ds_c.resize(n + nb, axis=0);  ds_c[n:] = color_batch
            ds_d.resize(n + nb, axis=0);  ds_d[n:] = depth_batch

    if "gelsight" in f:
        for j, name in enumerate(["left", "right"]):
            gs_batch = np.stack([b[2][j] for b in batch])      # (nb, 480, 640, 3)
            ds = f[f"gelsight/{name}/frames"]
            ds.resize(n + nb, axis=0);  ds[n:] = gs_batch
            # per-sensor capture timestamps (fall back to tick time if absent)
            gts = np.array([(b[4][j] if len(b) > 4 and b[4] and b[4][j] is not None
                             else b[3]) for b in batch], dtype=np.float64)
            dst = f[f"gelsight/{name}/timestamps"]
            dst.resize(n + nb, axis=0);  dst[n:] = gts

    if "arducam" in f:
        for j, name in enumerate(["cam0", "cam1"]):
            frames = np.stack([b[5][j] for b in batch])
            ds = f[f"arducam/{name}/frames"]
            ds.resize(n + nb, axis=0);  ds[n:] = frames
            timestamps = np.array([
                (b[6][j] if len(b) > 6 and b[6] and b[6][j] is not None else b[3])
                for b in batch
            ], dtype=np.float64)
            dst = f[f"arducam/{name}/timestamps"]
            dst.resize(n + nb, axis=0);  dst[n:] = timestamps


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


class HDF5Writer:
    """
    Background thread that drains a queue of camera frames and writes them to HDF5
    in batches, keeping the main capture loop free of disk I/O.

    Frames are accumulated into batches of `batch_size` then written with a single
    resize+write per dataset, reducing HDF5 metadata overhead by batch_size×.
    enqueue() is non-blocking — frames are dropped (with a warning) if the queue
    is full, so the main loop never stalls.
    """

    # Seconds between H5Fflush calls. HDF5 keeps object headers, chunk B-trees
    # and the superblock's EOF field in a metadata cache and writes them at
    # close; raw chunks go straight through. So a recorder that dies without
    # closing leaves every byte of pixel data on disk and no way to reach it.
    # pushT/2026-06-18/episode_004.h5 is 79 GB in exactly that state: the root
    # object header is 24 zero bytes and the superblock still claims EOF 2048.
    # Recovering it took a scan for orphaned B-tree leaves and a chain of
    # inferences, and even then the timestamps were unrecoverable — only 2 of
    # 16 chunks had been evicted to disk, and reconstructing the rest by
    # interpolation misplaces frames by 15-24 (measured against episodes with
    # known timestamps; one episode stalls and it misses by 1431).
    #
    # A flush costs one metadata write against ~190 MB/s of pixels, and turns
    # "lose the entire recording" into "lose the last few seconds".
    FLUSH_INTERVAL_S = 10.0

    def __init__(self, maxsize: int = 300, batch_size: int = 10):
        self._queue      = queue.Queue(maxsize=maxsize)
        self._batch_size = batch_size
        self._dropped    = 0
        self._last_flush = time.time()
        self._flushes    = 0
        self._stopped    = False
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            # Block for first item in next batch
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            # Drain up to batch_size-1 more items without blocking
            batch = [item]
            while len(batch) < self._batch_size:
                try:
                    next_item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if next_item is None:
                    # Sentinel arrived mid-batch: write current batch, then exit
                    self._write_batch(batch)
                    for _ in batch:
                        self._queue.task_done()
                    self._queue.task_done()   # for the sentinel
                    return
                batch.append(next_item)

            self._write_batch(batch)
            for _ in batch:
                self._queue.task_done()

    def _write_batch(self, batch):
        f = batch[0][0]
        append_camera_frames_batch(f, [item[1:] for item in batch])
        self._maybe_flush(f)

    def _maybe_flush(self, f):
        """Push HDF5's metadata cache to disk on a timer.

        Without this the file is unopenable until close() — see
        FLUSH_INTERVAL_S. Failures are swallowed deliberately: a flush that
        cannot happen must not take the recording down with it, and the next
        one will try again.
        """
        now = time.time()
        if now - self._last_flush < self.FLUSH_INTERVAL_S:
            return
        self._last_flush = now
        try:
            f.flush()
            self._flushes += 1
        except Exception as e:                      # pragma: no cover
            print(f"[HDF5Writer] WARNING: flush failed ({e}); "
                  f"a crash now would leave the file unopenable")

    def enqueue(self, f, color_frames, depth_frames, gs_frames, timestamp,
                gs_timestamps=None, arducam_frames=None,
                arducam_timestamps=None):
        """Non-blocking. Drops frame (with warning) if queue is full.

        gs_timestamps: optional list of 2 per-sensor capture times.
        """
        try:
            self._queue.put_nowait((f, color_frames, depth_frames, gs_frames,
                                    timestamp, gs_timestamps, arducam_frames,
                                    arducam_timestamps))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 30 == 1:
                print(f"[HDF5Writer] WARNING: queue full, dropped {self._dropped} frames total")

    def flush(self):
        """Block until all queued frames are written to disk."""
        self._queue.join()

    def stop(self):
        if self._stopped:
            return
        self.flush()
        self._queue.put(None)
        self._thread.join()
        self._stopped = True

    @property
    def queue_size(self):
        return self._queue.qsize()

    @property
    def dropped_frames(self):
        return self._dropped


def log_episode(data_dir, task_name, episode_num, h5_path, frame_count, fps, has_optitrack=True, notes=""):
    """
    Append one row to data/dataset_log.csv recording metadata about a saved episode.
    Creates the file with a header row if it doesn't exist yet.
    """
    import csv

    log_path = os.path.join(data_dir, "dataset_log.csv")
    file_exists = os.path.isfile(log_path)

    duration_s   = round(frame_count / fps, 2) if fps > 0 else 0
    size_mb      = round(os.path.getsize(h5_path) / 1e6, 1) if os.path.isfile(h5_path) else 0
    date_str     = time.strftime("%Y-%m-%d")
    saved_at     = time.strftime("%Y-%m-%dT%H:%M:%S")

    row = {
        "saved_at":    saved_at,
        "task":        task_name,
        "date":        date_str,
        "episode":     f"ep_{episode_num:03d}",
        "frames":      frame_count,
        "duration_s":  duration_s,
        "size_mb":     size_mb,
        "optitrack":   "yes" if has_optitrack else "no",
        "path":        os.path.relpath(h5_path, data_dir),
        "notes":       notes,
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged → {log_path}")


def next_episode_number(date_dir):
    """Return the next available episode number by scanning the date directory."""
    if not os.path.isdir(date_dir):
        return 0
    existing = [f for f in os.listdir(date_dir) if f.startswith("episode_") and f.endswith(".h5")]
    if not existing:
        return 0
    nums = [int(f.replace("episode_", "").replace(".h5", "")) for f in existing]
    return max(nums) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Sensor serials — update these to match your hardware
# ──────────────────────────────────────────────────────────────────────────────
REALSENSE_SERIALS = [
    "143322063538",
    "104122062574",
    "217222066989",
]
GELSIGHT_SERIALS = {
    # BROKEN as of 2026-08-07: fails USB enumeration (error -71 / "Device not
    # responding to setup address") on every port tried (1-8.1, 1-11, 1-1) —
    # fault travels with the unit, so it's the cable/sensor, not the host
    # port. _try_start_gelsight() falls back to a black dummy frame for this
    # side; run data_collection.py with --active_sensors right until it's
    # replaced so the OT watchdog doesn't wait on it.
    "left":  "28YGZL6K",   # /dev/video0  (replacement unit, registered 2026-08-07)
    "right": "2BGLKZNT",   # /dev/video2  (replacement unit, registered 2026-08-07)
    # "left":  "2DUPB53G"  # old left sensor
    # "right": "2BKRDTAD"  # old right sensor
}
DATA_DIR = "/media/yxma/Disk1/twm/data"
FPS = 30


# ──────────────────────────────────────────────────────────────────────────────
# Preview helpers — single source of truth lives in `twm.viz`.
#
# The live recording UI uses the same panel + projection overlay as the GIF
# pipelines. `make_preview` / `make_optitrack_panel` / `TRACKER_COLORS` are
# kept here as thin re-exports for backward compatibility.
# ──────────────────────────────────────────────────────────────────────────────

from twm.viz import (  # noqa: E402
    TRACKER_COLORS,
    make_optitrack_panel,
    build_preview_panel as make_preview,
    load_calibrations,
    draw_projection_overlay,
    CAM_CALIB_NAME,
)


# ──────────────────────────────────────────────────────────────────────────────
# Capture loop — background thread, strict 30 fps
# ──────────────────────────────────────────────────────────────────────────────

class CaptureLoop:
    """Runs the sensor-grab + writer.enqueue tick on its own thread at strict
    target fps. The GUI thread reads `latest()` (most recent snapshot) at
    whatever rate it can manage — recording cadence no longer depends on
    cv2.imshow / cv2.waitKey latency.

    Thread-safety contract:
      * `latest()` returns a dict snapshot taken under a tiny lock; the frame
        arrays inside are the live shallow refs (the sensor streams already
        return copies, so they're safe to read).
      * `start_recording / stop_recording / request_reset_ref` are all
        non-blocking and take effect on the next capture tick.
    """

    OT_TRACKERS = ("motherboard", "sensor_left", "sensor_right")
    TIMING_REPORT_INTERVAL = 60
    # Drop the first N captured frames of every episode: the RealSense/GelSight
    # streams run continuously but auto-exposure / gain can still settle for a
    # few frames right after we begin writing, so frame 0 is often unreliable.
    WARMUP_DROP_FRAMES = 10

    def __init__(self, rs_streams, gs_left, gs_right, optitrack, writer, fps=30,
                 arducam_streams=None):
        self.rs_streams = rs_streams
        self.gs_left    = gs_left
        self.gs_right   = gs_right
        self.optitrack  = optitrack
        self.writer     = writer
        self.arducam_streams = (list(arducam_streams)
                                 if arducam_streams is not None else None)
        self.fps        = fps
        self.tick_dt    = 1.0 / fps

        self._lock = threading.Lock()
        self._latest = None
        self._gs_ref = [gs_left.get_frame(), gs_right.get_frame()]
        self._reset_ref_request = False
        self._recording = False
        self._h5_file = None
        self._frame_count = 0
        self._start_t = 0.0
        self._dropped_at_start = 0
        self._warmup_remaining = 0

        self._tick_times = collections.deque(maxlen=30)
        self._timing_accum = collections.defaultdict(float)
        self._timing_ticks = 0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _publish_fatal_error(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            latest = dict(self._latest or {})
            latest.update({
                "fatal_error": message,
                "recording": self._recording,
                "frame_count": self._frame_count,
                "elapsed": ((time.time() - self._start_t)
                            if self._recording else 0.0),
            })
            self._latest = latest
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            tick_start = time.time()

            t0 = time.time()
            try:
                color_frames = [s.get_color_frame() for s in self.rs_streams]
                depth_frames = [s.get_depth_frame() for s in self.rs_streams]
                # GelSight: grab frame + its true capture timestamp (the sensor
                # runs below the camera tick on some hardware).
                gs_l, ts_l = _gs_frame_ts(self.gs_left)
                gs_r, ts_r = _gs_frame_ts(self.gs_right)
                gs_frames = [gs_l, gs_r]
                gs_timestamps = [ts_l, ts_r]
                if self.arducam_streams is not None:
                    arducam_samples = [_gs_frame_ts(stream)
                                        for stream in self.arducam_streams]
                    arducam_frames = [sample[0] for sample in arducam_samples]
                    arducam_timestamps = [sample[1] for sample in arducam_samples]
                else:
                    arducam_frames = None
                    arducam_timestamps = None
                t = time.time()
            except Exception as exc:
                self._publish_fatal_error(exc)
                return
            self._timing_accum["grab"] += t - t0

            t0 = time.time()
            with self._lock:
                rec      = self._recording
                h5_file  = self._h5_file
                # Discard the first WARMUP_DROP_FRAMES frames of the episode so
                # camera warm-up artifacts never reach disk (frame_count stays 0
                # until the first frame we actually keep).
                if rec and self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                    rec = False
            if rec and h5_file is not None:
                self.writer.enqueue(
                    h5_file, color_frames, depth_frames, gs_frames, t,
                    gs_timestamps=gs_timestamps,
                    arducam_frames=arducam_frames,
                    arducam_timestamps=arducam_timestamps,
                )
                with self._lock:
                    self._frame_count += 1
            self._timing_accum["enqueue"] += time.time() - t0

            ot_poses = {name: self.optitrack.get_latest_pose(name)
                        for name in self.OT_TRACKERS}

            self._tick_times.append(t)
            fps_meas = (len(self._tick_times) / (self._tick_times[-1] - self._tick_times[0])
                        if len(self._tick_times) >= 2 else 0.0)

            with self._lock:
                if self._reset_ref_request:
                    self._gs_ref = [gs_frames[0].copy(), gs_frames[1].copy()]
                    self._reset_ref_request = False
                self._latest = {
                    "color_frames": color_frames,
                    "gs_frames":    gs_frames,
                    "gs_ref":       self._gs_ref,
                    "arducam_frames": arducam_frames,
                    "arducam_timestamps": arducam_timestamps,
                    "ot_poses":     ot_poses,
                    "timestamp":    t,
                    "frame_count":  self._frame_count,
                    "elapsed":      (t - self._start_t) if self._recording else 0.0,
                    "fps_meas":     fps_meas,
                    "recording":    self._recording,
                    "fatal_error":  None,
                }

            self._timing_ticks += 1
            if self._timing_ticks % self.TIMING_REPORT_INTERVAL == 0:
                n = self.TIMING_REPORT_INTERVAL
                state = "REC" if rec else "IDLE"
                print(f"[cap/{state}] fps={fps_meas:.1f} | "
                      f"grab={self._timing_accum['grab']/n*1000:.1f}ms | "
                      f"enqueue={self._timing_accum['enqueue']/n*1000:.1f}ms | "
                      f"q={self.writer.queue_size}")
                self._timing_accum.clear()

            sleep_t = self.tick_dt - (time.time() - tick_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── thread-safe API used by the GUI thread ────────────────────────────────
    def latest(self):
        with self._lock:
            return self._latest

    def request_reset_ref(self):
        with self._lock:
            self._reset_ref_request = True

    def start_recording(self, h5_file):
        with self._lock:
            self._h5_file = h5_file
            self._frame_count = 0
            self._start_t = time.time()
            self._recording = True
            self._dropped_at_start = self.writer.dropped_frames
            self._warmup_remaining = self.WARMUP_DROP_FRAMES

    def stop_recording(self):
        """Stop recording. Returns (h5_file, frame_count, dropped_this_episode)
        or None if we weren't recording."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            h5_file = self._h5_file
            frame_count = self._frame_count
            dropped = self.writer.dropped_frames - self._dropped_at_start
            self._h5_file = None
        # Flush outside the lock; the writer is thread-safe.
        self.writer.flush()
        return h5_file, frame_count, dropped


def _stop_resources(resources):
    """Best-effort reverse-order cleanup for a partially initialized rig."""
    for resource in reversed(resources):
        try:
            resource.stop()
        except Exception as exc:  # cleanup must continue for the other devices
            print(f"Warning: startup cleanup could not stop {resource!r}: {exc}")
    resources.clear()


def _startup_call(fn, resources, *args, **kwargs):
    """Run one startup operation and clean all registered devices on failure."""
    try:
        return fn(*args, **kwargs)
    except BaseException:
        _stop_resources(resources)
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _main(started_resources):
    import argparse
    import cv2
    from camera_stream.realsense_stream import RealsenseStream
    from camera_stream.usb_video_stream import USBVideoStream
    from camera_stream.arducam_video_stream import ArducamVideoStream
    from optitrack.optitrack_stream import OptitrackStream
    from twm.sensor_camera import load_config, resolve_slots

    parser = argparse.ArgumentParser(description="TWM multimodal data collection")
    parser.add_argument("--task", required=True, help="Task name (used as top-level folder, e.g. 'pouring', 'pick_place')")
    parser.add_argument("--active_sensors", default="both",
                        choices=["both", "left", "right"],
                        help="Which GelSight rigid bodies the OT watchdog monitors "
                             "(refuses to start an episode if any is silent; auto-ends "
                             "an in-progress episode after 10 s of silence).")
    parser.add_argument("--no_projection", action="store_true",
                        help="Disable the live GelSight→camera projection overlay "
                             "(dot + axes drawn on the RealSense thumbnails). Toggle "
                             "at runtime with the 'p' key.")
    parser.add_argument("--no_arducam", action="store_true",
                        help="Explicitly run the legacy recorder without the two "
                             "sensor-mounted Arducams.")
    parser.add_argument("--arducam_config", default=None,
                        help="Arducam JSON configuration (default: "
                             "twm/config/arducam.json).")
    args = parser.parse_args()
    task_name = args.task

    # OT watchdog config. Pre-flight is tight so the operator doesn't start
    # while OT is still warming up; recording-time timeout is looser to
    # tolerate brief marker occlusion without bailing.
    ACTIVE_SENSORS = {
        "both":  ["sensor_left", "sensor_right"],
        "left":  ["sensor_left"],
        "right": ["sensor_right"],
    }[args.active_sensors]
    OT_PREFLIGHT_MAX_AGE_S = 2.0
    OT_WATCHDOG_TIMEOUT_S = 10.0
    STARTUP_TIMEOUT = 15.0  # cameras can be slow on first initialization

    date_str = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(DATA_DIR, task_name, date_str)
    print(f"Task: {task_name}  |  Saving to: {date_dir}")

    # ── Init sensors ──────────────────────────────────────────────────────────
    print("Initializing RealSense cameras...")
    rs_streams = [_startup_call(RealsenseStream, started_resources,
                                serial=s, fps=FPS)
                  for s in REALSENSE_SERIALS]
    for s in rs_streams:
        started_resources.append(s)
        _startup_call(s.start, started_resources)
        _startup_call(time.sleep, started_resources, 0.5)

    arducam_config = ()
    arducam_streams = []
    if not args.no_arducam:
        print("Initializing sensor-mounted Arducam cameras...")
        slots = (_startup_call(load_config, started_resources,
                               args.arducam_config)
                 if args.arducam_config
                 else _startup_call(load_config, started_resources))
        arducam_config = _startup_call(resolve_slots, started_resources, slots)
        for camera in arducam_config:
            stream = _startup_call(
                ArducamVideoStream, started_resources,
                camera.config, camera.device,
            )
            arducam_streams.append(stream)
            started_resources.append(stream)
            _startup_call(stream.start, started_resources,
                          timeout=STARTUP_TIMEOUT)

    print("Initializing GelSight sensors...")

    class _DummyGelSight:
        def __init__(self):
            self._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        def start(self): pass
        def stop(self): pass
        def get_frame(self): return self._frame

    def _try_start_gelsight(side, serial):
        gs = USBVideoStream(serial=serial, resolution=(640, 480), name=side)
        try:
            gs.start()
            return gs
        except Exception as e:
            gs.stop()
            print(f"  WARNING: GelSight '{side}' (serial {serial}) not available: {e}")
            print(f"           Continuing without it — frames will be black.")
            return _DummyGelSight()

    gs_left = _startup_call(
        _try_start_gelsight, started_resources, "left", GELSIGHT_SERIALS["left"]
    )
    started_resources.append(gs_left)
    gs_right = _startup_call(
        _try_start_gelsight, started_resources, "right", GELSIGHT_SERIALS["right"]
    )
    started_resources.append(gs_right)

    print("Initializing OptiTrack...")
    optitrack = _startup_call(OptitrackStream, started_resources)
    started_resources.append(optitrack)
    _startup_call(optitrack.start, started_resources)

    print("Waiting for first frames from all sensors...")
    for s in rs_streams:
        _startup_call(s.get_color_frame, started_resources,
                      timeout=STARTUP_TIMEOUT)
    for s in arducam_streams:
        _startup_call(s.get_frame_with_timestamp, started_resources,
                      timeout=STARTUP_TIMEOUT)
    _startup_call(gs_left.get_frame, started_resources)
    _startup_call(gs_right.get_frame, started_resources)

    # Settle phase: getting the *first* frame is not enough — the USB GelSight
    # cameras need ~1-2 s of streaming for auto-exposure/white-balance to
    # converge (the right sensor settles slower than the left). Pump and
    # discard frames so the GelSight diff reference (grabbed in CaptureLoop)
    # and the first recorded frames all come from settled cameras.
    SETTLE_S = 1.0
    print(f"Letting cameras settle ({SETTLE_S:.1f}s)...")
    settle_end = time.time() + SETTLE_S
    while time.time() < settle_end:
        for s in rs_streams:
            _startup_call(s.get_color_frame, started_resources)
        _startup_call(gs_left.get_frame, started_resources)
        _startup_call(gs_right.get_frame, started_resources)
        for s in arducam_streams:
            _startup_call(s.get_frame, started_resources)
        _startup_call(time.sleep, started_resources, 0.02)
    print("All sensors ready.\n")
    print("Controls:  s = start episode   e = end episode   r = reset diff ref   "
          "p = toggle projection   q = quit\n")

    # ── Projection calibration ──────────────────────────────────────────────────
    # Project each GelSight surface center (+ body axes) onto the overhead
    # RealSense thumbnails, same math as the offline `visualize` replay. Purely
    # a preview overlay — does not touch what's written to disk.
    calib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "calibration", "result")
    project_cams = []
    gel_center_left = gel_center_right = None
    show_projection = not args.no_projection
    if show_projection:
        try:
            cam_calibs, gel_center_left, gel_center_right = load_calibrations(
                [os.path.join(calib_dir, CAM_CALIB_NAME[i]) for i in range(3)],
                os.path.join(calib_dir, "T_gel_to_rigid_left.json"),
                os.path.join(calib_dir, "T_gel_to_rigid_right.json"),
            )
            for calib in cam_calibs:
                serial = calib["camera_serial"]
                try:
                    c_idx = REALSENSE_SERIALS.index(serial)
                except ValueError:
                    print(f"  Projection: camera serial {serial} not in "
                          f"REALSENSE_SERIALS, skipping.")
                    continue
                project_cams.append({
                    "index": c_idx,
                    "T_mocap_to_cam": calib["T_mocap_to_cam"],
                    "intrinsics": calib["intrinsics"],
                })
            print(f"Projection overlay: ON ({len(project_cams)} cameras calibrated). "
                  "Press 'p' to toggle.\n")
        except BaseException as e:
            if not isinstance(e, Exception):
                _stop_resources(started_resources)
                raise
            print(f"Projection overlay: DISABLED (failed to load calibration: {e})\n")
            show_projection = False
            project_cams = []

    # ── State ─────────────────────────────────────────────────────────────────
    writer = _startup_call(HDF5Writer, started_resources)
    started_resources.append(writer)
    episode_num = 0
    path        = None
    capture = _startup_call(
        CaptureLoop, started_resources,
        rs_streams, gs_left, gs_right, optitrack, writer, fps=FPS,
        arducam_streams=(arducam_streams if arducam_streams else None),
    )
    started_resources.append(capture)
    _startup_call(capture.start, started_resources)
    arducam_labels = [
        f"{camera.slot} {camera.id_path} {camera.position}"
        for camera in arducam_config
    ]

    # GUI runs at a softer cadence than capture; PREVIEW_FPS controls how
    # often we rebuild the preview panel + push it to X11. Set conservatively
    # so even slow displays keep capture at the strict target FPS.
    PREVIEW_FPS = 15
    preview_dt  = 1.0 / PREVIEW_FPS
    gui_dt      = 1.0 / 30.0   # ceiling on how fast we burn CPU waking on waitKey
    last_preview_ts = 0.0
    last_panel = None

    def _end_episode(auto: bool = False):
        """Finalize the current recording: flush writer, flush OT, close H5,
        log row, print summary. Idempotent if not currently recording."""
        nonlocal path
        info = capture.stop_recording()
        if info is None:
            return
        h5_file, frame_count, dropped = info
        optitrack_data = {
            name: optitrack.flush_buffer(name)
            for name in ["motherboard", "sensor_left", "sensor_right"]
        }
        has_optitrack = any(len(v) > 0 for v in optitrack_data.values())
        flush_optitrack_to_hdf5(h5_file, optitrack_data)
        h5_file.close()
        log_episode(DATA_DIR, task_name, episode_num, path, frame_count, FPS,
                    has_optitrack=has_optitrack)
        tag = " (auto-ended)" if auto else ""
        drop_str = f", {dropped} frames DROPPED" if dropped else ""
        print(f"Episode {episode_num:03d} saved{tag} — {frame_count} frames, "
              f"{frame_count / FPS:.1f}s{drop_str}")

    # ── GUI loop ──────────────────────────────────────────────────────────────
    try:
        while True:
            gui_start = time.time()

            latest = capture.latest()
            if latest is None:
                time.sleep(0.005); continue
            if latest.get("fatal_error"):
                print(f"\n!!! Capture stopped: {latest['fatal_error']}")
                if latest.get("recording"):
                    _end_episode(auto=True)
                break

            # Preview gate: only rebuild + imshow at PREVIEW_FPS; otherwise
            # re-display the cached panel so cv2.waitKey still pumps events.
            if gui_start - last_preview_ts >= preview_dt or last_panel is None:
                last_panel = make_preview(
                    latest["color_frames"], latest["gs_frames"], latest["gs_ref"],
                    latest["ot_poses"], latest["recording"], latest["frame_count"],
                    latest["elapsed"], writer.queue_size, latest["fps_meas"],
                    task_name=task_name,
                    arducam_frames=latest["arducam_frames"],
                    arducam_labels=(arducam_labels if arducam_labels else None),
                )
                if show_projection and project_cams:
                    draw_projection_overlay(
                        last_panel, latest["ot_poses"], project_cams,
                        gel_center_left, gel_center_right,
                    )
                last_preview_ts = gui_start
            cv2.imshow("TWM Data Collection", last_panel)

            # OT watchdog (auto-end if a tracked body has gone silent).
            if latest["recording"]:
                now_wall = time.time()
                stale = []
                for tn in ACTIVE_SENSORS:
                    p = latest["ot_poses"].get(tn)
                    age = now_wall - (p[0] if p is not None else now_wall)
                    if age > OT_WATCHDOG_TIMEOUT_S:
                        stale.append(f"{tn} silent {age:.1f}s")
                if stale:
                    print(f"\n!!! OT watchdog: {', '.join(stale)}  →  auto-ending episode "
                          f"ep_{episode_num:03d}.")
                    _end_episode(auto=True)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not latest["recording"]:
                # OT pre-flight: refuse to start if any active body is silent.
                now_wall = time.time()
                stale = []
                for tn in ACTIVE_SENSORS:
                    p = latest["ot_poses"].get(tn)
                    if p is None:
                        stale.append(f"{tn}: no data yet")
                    else:
                        age = now_wall - p[0]
                        if age > OT_PREFLIGHT_MAX_AGE_S:
                            stale.append(f"{tn}: last sample {age:.1f}s old")
                if stale:
                    print("\n!!! Cannot start recording — OptiTrack pre-flight failed:")
                    for s in stale:
                        print(f"     - {s}")
                    print("   Bring sensors into the mocap volume + wait for live samples, "
                          "then press 's' again.\n")
                else:
                    episode_num = next_episode_number(date_dir)
                    h5_file, path = create_episode_file(
                        date_dir, episode_num, REALSENSE_SERIALS,
                        list(GELSIGHT_SERIALS.values()), FPS,
                        task_name=task_name,
                        arducam_config=arducam_config,
                    )
                    for name in ["motherboard", "sensor_left", "sensor_right"]:
                        optitrack.flush_buffer(name)
                    capture.start_recording(h5_file)
                    print(f"\nRecording episode {episode_num:03d} → {path}")

            elif key == ord('e') and latest["recording"]:
                _end_episode()

            elif key == ord('r'):
                capture.request_reset_ref()
                print("GelSight diff reference reset (queued).")

            elif key == ord('p'):
                if project_cams:
                    show_projection = not show_projection
                    print(f"Projection overlay: {'ON' if show_projection else 'OFF'}.")
                    last_panel = None  # force a rebuild so the overlay (dis)appears now
                else:
                    print("Projection overlay unavailable (no calibration loaded).")

            elif key == ord('q'):
                if latest["recording"]:
                    print("\nSaving in-progress episode before quit...")
                    _end_episode()
                writer.stop()
                break

            # Cap GUI tick so we don't burn CPU when nothing's changing.
            gui_sleep = gui_dt - (time.time() - gui_start)
            if gui_sleep > 0:
                time.sleep(gui_sleep)

    finally:
        # Always try to finalize a still-open recording on unexpected exit.
        try:
            _end_episode()
        except Exception as save_err:
            print(f"Warning: could not save episode cleanly: {save_err}")
        _stop_resources(started_resources)
        cv2.destroyAllWindows()
        print("All sensors stopped. Goodbye.")


def main():
    """Run the recorder with cleanup covering initialization through shutdown."""
    started_resources = []
    try:
        return _main(started_resources)
    finally:
        _stop_resources(started_resources)


if __name__ == "__main__":
    main()
