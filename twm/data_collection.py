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
import queue
import threading
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


class HDF5Writer:
    """
    Background thread that drains a queue of camera frames and writes them to HDF5,
    keeping the main capture loop free of disk I/O.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            f, color_frames, depth_frames, gs_frames, timestamp = item
            append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp)
            self._queue.task_done()

    def enqueue(self, f, color_frames, depth_frames, gs_frames, timestamp):
        """Non-blocking: copy frames and put on queue."""
        self._queue.put((
            f,
            [c.copy() for c in color_frames],
            [d.copy() for d in depth_frames],
            [g.copy() for g in gs_frames],
            timestamp,
        ))

    def flush(self):
        """Block until all queued frames are written."""
        self._queue.join()

    def stop(self):
        self.flush()
        self._queue.put(None)
        self._thread.join()

    @property
    def queue_size(self):
        return self._queue.qsize()


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
    "left":  "2BGLKZNT",   # /dev/video14
    "right": "2BKRDTAD",   # /dev/video12
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FPS = 30


# ──────────────────────────────────────────────────────────────────────────────
# Preview helpers
# ──────────────────────────────────────────────────────────────────────────────

TRACKER_COLORS = {
    "motherboard":  (255, 200,   0),
    "sensor_left":  (  0, 255, 120),
    "sensor_right": (  0, 180, 255),
}


def make_optitrack_panel(optitrack_poses, w=320, h=240):
    """Render a text panel showing live pose (x, y, z) for each tracker."""
    import cv2

    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, "OptiTrack", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    y = 48
    for name, color in TRACKER_COLORS.items():
        pose = optitrack_poses.get(name)
        cv2.putText(panel, name, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 18
        if pose is None:
            cv2.putText(panel, "  no data", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        else:
            _, xyz_quat = pose
            x_m, y_m, z_m = xyz_quat[:3]
            qx, qy, qz, qw = xyz_quat[3:]
            cv2.putText(panel, f"  x={x_m:+.3f} y={y_m:+.3f} z={z_m:+.3f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            y += 16
            cv2.putText(panel, f"  qx={qx:+.2f} qy={qy:+.2f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
            y += 16
            cv2.putText(panel, f"  qz={qz:+.2f} qw={qw:+.2f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        y += 24
    return panel


def make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, recording, frame_count, elapsed, buf=0):
    """Build a tiled OpenCV preview image from all camera feeds.

    Row 1: [cam0] [cam1] [cam2] [optitrack text]  — 4 × 320 = 1280px
    Row 2: [gs_l] [gs_l_diff] [gs_r] [gs_r_diff] [blank]  — 4×240 + 320 = 1280px
    """
    import cv2

    rs_w, rs_h = 320, 240
    gs_w, gs_h = 240, 240

    def rs_thumb(img):
        return cv2.resize(img, (rs_w, rs_h))

    def gs_thumb(img):
        return cv2.resize(img, (gs_w, gs_h))

    def diff_thumb(frame, ref):
        diff = np.clip(frame.astype(np.int16) - ref.astype(np.int16) + 128, 0, 255).astype(np.uint8)
        return cv2.resize(diff, (gs_w, gs_h))

    optitrack_panel = make_optitrack_panel(optitrack_poses, w=rs_w, h=rs_h)
    row1 = np.hstack([rs_thumb(f) for f in color_frames] + [optitrack_panel])

    gs_panels = []
    for frame, ref in zip(gs_frames, gs_ref):
        gs_panels.append(gs_thumb(frame))
        gs_panels.append(diff_thumb(frame, ref))
    blank = np.zeros((gs_h, rs_w, 3), dtype=np.uint8)
    row2 = np.hstack(gs_panels + [blank])

    preview = np.vstack([row1, row2])

    # Status bar
    if recording:
        status = f"[RECORDING ep_{frame_count:04d} | {frame_count} frames | {elapsed:.1f}s | buf={buf}]"
        color = (0, 0, 220)
    else:
        status = "[IDLE]  s=start  e=end  r=reset-ref  q=quit"
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
    gs_ref = [gs_left.get_frame(), gs_right.get_frame()]
    print("All sensors ready.\n")
    print("Controls:  s = start episode   e = end episode   r = reset diff ref   q = quit\n")

    # ── State ─────────────────────────────────────────────────────────────────
    writer      = HDF5Writer()
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

            # Write if recording (non-blocking)
            if recording and h5_file is not None:
                writer.enqueue(h5_file, color_frames, depth_frames, gs_frames, t)
                frame_count += 1

            # Preview
            elapsed = t - start_t if recording else 0.0
            optitrack_poses = {
                name: optitrack.get_latest_pose(name)
                for name in ["motherboard", "sensor_left", "sensor_right"]
            }
            preview = make_preview(color_frames, gs_frames, gs_ref, optitrack_poses, recording, frame_count, elapsed, writer.queue_size)
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
                print(f"\nFlushing {writer.queue_size} buffered frames...")
                writer.flush()
                # Flush OptiTrack buffer
                optitrack_data = {
                    name: optitrack.flush_buffer(name)
                    for name in ["motherboard", "sensor_left", "sensor_right"]
                }
                flush_optitrack_to_hdf5(h5_file, optitrack_data)
                h5_file.close()
                h5_file = None
                print(f"Episode {episode_num:03d} saved — {frame_count} frames, "
                      f"{frame_count / FPS:.1f}s")

            elif key == ord('r'):
                gs_ref = [gs_left.get_frame(), gs_right.get_frame()]
                print("GelSight diff reference reset.")

            elif key == ord('q'):
                if recording and h5_file is not None:
                    print("\nSaving in-progress episode before quit...")
                    recording = False
                    writer.flush()
                    optitrack_data = {
                        name: optitrack.flush_buffer(name)
                        for name in ["motherboard", "sensor_left", "sensor_right"]
                    }
                    flush_optitrack_to_hdf5(h5_file, optitrack_data)
                    h5_file.close()
                writer.stop()
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
