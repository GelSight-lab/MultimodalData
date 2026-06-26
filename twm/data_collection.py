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
import os
import queue
import threading
import time
import numpy as np
import h5py
import hdf5plugin

# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_episode_file(date_dir, episode_num, realsense_serials, gelsight_serials, fps, task_name=""):
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

    # camera timestamps (one per main-loop tick)
    f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)

    # BLOSC LZ4 compression — benchmarked at ~100fps overhead vs 30fps capture rate.
    _blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

    for i in range(3):
        g = f.create_group(f"realsense/cam{i}")
        g.create_dataset("color", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                         dtype=np.uint8,  chunks=(1, 480, 640, 3), **_blosc)
        g.create_dataset("depth", shape=(0, 480, 640),    maxshape=(None, 480, 640),
                         dtype=np.uint16, chunks=(1, 480, 640),    **_blosc)

    # GelSight
    for name in ["left", "right"]:
        g = f.create_group(f"gelsight/{name}")
        g.create_dataset("frames", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                         dtype=np.uint8, chunks=(1, 480, 640, 3), **_blosc)

    # OptiTrack — per-tracker timestamps + poses
    for name in ["motherboard", "sensor_left", "sensor_right"]:
        g = f.create_group(f"optitrack/{name}")
        g.create_dataset("timestamps", shape=(0,),    maxshape=(None,),    dtype=np.float64)
        g.create_dataset("pose",       shape=(0, 7),  maxshape=(None, 7),  dtype=np.float64)

    return f, path


def append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp):
    """Append one timestep — thin wrapper around the batch writer."""
    append_camera_frames_batch(f, [(color_frames, depth_frames, gs_frames, timestamp)])


def append_camera_frames_batch(f, batch):
    """
    Write a batch of frames to HDF5 in one resize+write per dataset.

    batch: list of (color_frames, depth_frames, gs_frames, timestamp) tuples
           color_frames: list of 3 arrays (480, 640, 3) uint8
           depth_frames: list of 3 arrays (480, 640) uint16
           gs_frames:    list of 2 arrays (480, 640, 3) uint8
           timestamp:    float

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

    for i in range(3):
        color_batch = np.stack([b[0][i] for b in batch])   # (nb, 480, 640, 3)
        depth_batch = np.stack([b[1][i] for b in batch])   # (nb, 480, 640)
        ds_c = f[f"realsense/cam{i}/color"]
        ds_d = f[f"realsense/cam{i}/depth"]
        ds_c.resize(n + nb, axis=0);  ds_c[n:] = color_batch
        ds_d.resize(n + nb, axis=0);  ds_d[n:] = depth_batch

    for j, name in enumerate(["left", "right"]):
        gs_batch = np.stack([b[2][j] for b in batch])      # (nb, 480, 640, 3)
        ds = f[f"gelsight/{name}/frames"]
        ds.resize(n + nb, axis=0);  ds[n:] = gs_batch


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

    def __init__(self, maxsize: int = 300, batch_size: int = 10):
        self._queue      = queue.Queue(maxsize=maxsize)
        self._batch_size = batch_size
        self._dropped    = 0
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

    def enqueue(self, f, color_frames, depth_frames, gs_frames, timestamp):
        """Non-blocking. Drops frame (with warning) if queue is full."""
        try:
            self._queue.put_nowait((f, color_frames, depth_frames, gs_frames, timestamp))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 30 == 1:
                print(f"[HDF5Writer] WARNING: queue full, dropped {self._dropped} frames total")

    def flush(self):
        """Block until all queued frames are written to disk."""
        self._queue.join()

    def stop(self):
        self.flush()
        self._queue.put(None)
        self._thread.join()

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
    # "left":  "2BGLKZNT",   # /dev/video14  (old left sensor)
    "left":  "2DUPB53G",   # /dev/video8
    # 'left': "2BKRDTAD"
    "right": "2BKRDTAD",   # /dev/video12
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

    def __init__(self, rs_streams, gs_left, gs_right, optitrack, writer, fps=30):
        self.rs_streams = rs_streams
        self.gs_left    = gs_left
        self.gs_right   = gs_right
        self.optitrack  = optitrack
        self.writer     = writer
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

    def _run(self):
        while not self._stop.is_set():
            tick_start = time.time()

            t0 = time.time()
            color_frames = [s.get_color_frame() for s in self.rs_streams]
            depth_frames = [s.get_depth_frame() for s in self.rs_streams]
            gs_frames    = [self.gs_left.get_frame(), self.gs_right.get_frame()]
            t            = time.time()
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
                self.writer.enqueue(h5_file, color_frames, depth_frames, gs_frames, t)
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
                    "ot_poses":     ot_poses,
                    "timestamp":    t,
                    "frame_count":  self._frame_count,
                    "elapsed":      (t - self._start_t) if self._recording else 0.0,
                    "fps_meas":     fps_meas,
                    "recording":    self._recording,
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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import cv2
    from camera_stream.realsense_stream import RealsenseStream
    from camera_stream.usb_video_stream import USBVideoStream
    from optitrack.optitrack_stream import OptitrackStream

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

    date_str = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(DATA_DIR, task_name, date_str)
    print(f"Task: {task_name}  |  Saving to: {date_dir}")

    # ── Init sensors ──────────────────────────────────────────────────────────
    print("Initializing RealSense cameras...")
    rs_streams = [RealsenseStream(serial=s, fps=FPS) for s in REALSENSE_SERIALS]
    for s in rs_streams:
        s.start()
        time.sleep(0.5)  # stagger starts to avoid USB bandwidth contention

    print("Initializing GelSight sensors...")

    class _DummyGelSight:
        def __init__(self):
            self._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        def start(self): pass
        def stop(self): pass
        def get_frame(self): return self._frame

    def _try_start_gelsight(side, serial):
        gs = USBVideoStream(serial=serial, resolution=(640, 480))
        try:
            gs.start()
            return gs
        except Exception as e:
            print(f"  WARNING: GelSight '{side}' (serial {serial}) not available: {e}")
            print(f"           Continuing without it — frames will be black.")
            return _DummyGelSight()

    gs_left  = _try_start_gelsight("left",  GELSIGHT_SERIALS["left"])
    gs_right = _try_start_gelsight("right", GELSIGHT_SERIALS["right"])

    print("Initializing OptiTrack...")
    optitrack = OptitrackStream()
    optitrack.start()

    STARTUP_TIMEOUT = 15.0  # seconds — cameras can be slow on first init
    print("Waiting for first frames from all sensors...")
    for s in rs_streams:
        s.get_color_frame(timeout=STARTUP_TIMEOUT)
    gs_left.get_frame()
    gs_right.get_frame()

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
            s.get_color_frame()
        gs_left.get_frame()
        gs_right.get_frame()
        time.sleep(0.02)
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
        except Exception as e:
            print(f"Projection overlay: DISABLED (failed to load calibration: {e})\n")
            show_projection = False
            project_cams = []

    # ── State ─────────────────────────────────────────────────────────────────
    writer      = HDF5Writer()
    episode_num = 0
    path        = None
    capture     = CaptureLoop(rs_streams, gs_left, gs_right, optitrack, writer, fps=FPS)
    capture.start()

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

            # Preview gate: only rebuild + imshow at PREVIEW_FPS; otherwise
            # re-display the cached panel so cv2.waitKey still pumps events.
            if gui_start - last_preview_ts >= preview_dt or last_panel is None:
                last_panel = make_preview(
                    latest["color_frames"], latest["gs_frames"], latest["gs_ref"],
                    latest["ot_poses"], latest["recording"], latest["frame_count"],
                    latest["elapsed"], writer.queue_size, latest["fps_meas"],
                    task_name=task_name,
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
        capture.stop()
        for s in rs_streams:
            s.stop()
        gs_left.stop()
        gs_right.stop()
        optitrack.stop()
        cv2.destroyAllWindows()
        print("All sensors stopped. Goodbye.")


if __name__ == "__main__":
    main()
