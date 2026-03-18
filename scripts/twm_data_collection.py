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
