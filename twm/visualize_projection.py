#!/usr/bin/env python3
"""
Visualize a TWM episode with GelSight contact centers projected onto the overhead camera.

Extends twm.visualize with OptiTrack→camera projection:
  - Reads GelSight-to-rigid-body calibrations (left/right)
  - Reads OptiTrack→camera extrinsic calibration
  - Projects GelSight surface centers onto the overhead RealSense view each frame

Usage:
    python -m twm.visualize_projection path/to/episode_000.h5 \
        --cam_calib  twm/calibration/result/T_mocap_to_cam.json \
        --gel_left   twm/calibration/result/T_gel_to_rigid_left.json \
        --gel_right  twm/calibration/result/T_gel_to_rigid_right.json

    # Optional: override which cam index is the overhead camera (default: auto from serial)
    python -m twm.visualize_projection ... --cam_index 2

Controls are the same as twm.visualize.
"""

import argparse
import collections
import json
import os
import sys
import time
import numpy as np
import h5py
import cv2
from scipy.spatial.transform import Rotation

from twm.data_collection import make_preview, REALSENSE_SERIALS


# ──────────────────────────────────────────────────────────────────────────────
# Reuse helpers from visualize.py
# ──────────────────────────────────────────────────────────────────────────────

SPEEDS = [1, 2, 5, 10]

ACTIONS = [
    ("SPACE",   "pause / resume"),
    ("→ / d",   "next frame"),
    ("← / a",   "prev frame"),
    ("1/2/3/4", "speed 1×/2×/5×/10×"),
    ("l",       "toggle loop"),
    ("r",       "reset diff reference"),
    ("q",       "quit"),
]


def make_action_menu(w=320, h=240, paused=False, loop=False):
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, "Controls", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.line(panel, (10, 32), (w - 10, 32), (80, 80, 80), 1)
    y = 52
    for key, desc in ACTIONS:
        highlight = (key == "SPACE")
        loop_key  = (key == "l")
        key_color  = (0, 220, 255) if highlight else (255, 180, 0) if loop_key else (140, 200, 140)
        desc_color = (220, 220, 220) if highlight else (160, 160, 160)
        cv2.putText(panel, f"[{key}]", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, key_color, 1)
        cv2.putText(panel, desc, (110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, desc_color, 1)
        y += 28
    state_text  = "|| PAUSED" if paused else "> PLAYING"
    state_color = (0, 140, 255) if paused else (0, 220, 80)
    cv2.putText(panel, state_text, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
    loop_text  = "LOOP ON" if loop else "LOOP OFF"
    loop_color = (255, 180, 0) if loop else (80, 80, 80)
    cv2.putText(panel, loop_text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, loop_color, 1)
    return panel


def load_optitrack(f):
    lookup = {}
    for name in ["motherboard", "sensor_left", "sensor_right"]:
        ts    = f[f"optitrack/{name}/timestamps"][:]
        poses = f[f"optitrack/{name}/pose"][:]
        lookup[name] = (ts, poses) if len(ts) > 0 else None
    return lookup


def optitrack_at(lookup, camera_timestamp):
    result = {}
    for name, data in lookup.items():
        if data is None:
            result[name] = None
            continue
        ts, poses = data
        idx = int(np.searchsorted(ts, camera_timestamp))
        if idx == 0:
            pass
        elif idx >= len(ts):
            idx = len(ts) - 1
        elif abs(ts[idx - 1] - camera_timestamp) <= abs(ts[idx] - camera_timestamp):
            idx -= 1
        result[name] = (float(ts[idx]), poses[idx].tolist())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Projection helpers
# ──────────────────────────────────────────────────────────────────────────────

def pose_to_matrix(position_mm, quaternion):
    """[x,y,z] (mm) + [qx,qy,qz,qw] → 4×4 homogeneous matrix."""
    q = np.array(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = np.array(position_mm, dtype=np.float64)
    return T


def project_gel_center(rigid_pose_7, gel_center_in_rigid_mm, T_mocap_to_cam, intrinsics):
    """
    Project a GelSight surface center onto the camera image.

    Args:
        rigid_pose_7:            [x,y,z, qx,qy,qz,qw] from OptiTrack (metres!)
        gel_center_in_rigid_mm:  (3,) offset in rigid body frame (mm)
        T_mocap_to_cam:          (4,4) mocap-mm → camera-mm
        intrinsics:              dict with fx, fy, ppx, ppy

    Returns:
        (u, v) pixel coordinates, or None if behind camera / no data.
    """
    if rigid_pose_7 is None:
        return None

    # OptiTrack pose: metres → mm
    pos_mm = np.array(rigid_pose_7[:3]) * 1000.0
    quat   = rigid_pose_7[3:]  # [qx, qy, qz, qw]

    T_rigid_to_mocap = pose_to_matrix(pos_mm, quat)

    # GelSight center in mocap frame (mm)
    P_gel_rigid_h = np.append(gel_center_in_rigid_mm, 1.0)
    P_gel_mocap = (T_rigid_to_mocap @ P_gel_rigid_h)[:3]

    # Mocap → camera frame (mm)
    P_gel_mocap_h = np.append(P_gel_mocap, 1.0)
    P_cam = (T_mocap_to_cam @ P_gel_mocap_h)[:3]

    # Must be in front of camera
    if P_cam[2] <= 0:
        return None

    # Project to pixel
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["ppx"], intrinsics["ppy"]
    u = fx * P_cam[0] / P_cam[2] + cx
    v = fy * P_cam[1] / P_cam[2] + cy
    return (u, v)


def load_calibrations(cam_calib_path, gel_left_path, gel_right_path):
    """Load all calibration JSON files. Returns (T_mocap_to_cam, intrinsics, cam_serial,
       gel_center_left_mm, gel_center_right_mm)."""
    with open(cam_calib_path, 'r') as fp:
        cam_calib = json.load(fp)
    T_mocap_to_cam = np.array(cam_calib["T_mocap_to_cam"], dtype=np.float64)
    intrinsics     = cam_calib["intrinsics"]
    cam_serial     = cam_calib["camera_serial"]

    with open(gel_left_path, 'r') as fp:
        gel_left = json.load(fp)
    gel_center_left = np.array(gel_left["gel_center_in_rigid_mm"], dtype=np.float64)

    with open(gel_right_path, 'r') as fp:
        gel_right = json.load(fp)
    gel_center_right = np.array(gel_right["gel_center_in_rigid_mm"], dtype=np.float64)

    return T_mocap_to_cam, intrinsics, cam_serial, gel_center_left, gel_center_right


# ──────────────────────────────────────────────────────────────────────────────
# Draw projected points on preview
# ──────────────────────────────────────────────────────────────────────────────

# Colors match TRACKER_COLORS in data_collection.py
GEL_COLORS = {
    "left":  (0, 255, 120),   # same as sensor_left
    "right": (0, 180, 255),   # same as sensor_right
}

RS_THUMB_W, RS_THUMB_H = 320, 240  # thumbnail size in preview
ORIG_W, ORIG_H = 640, 480          # original camera resolution


def draw_gel_projections(preview, cam_index, pixel_left, pixel_right):
    """
    Draw projected GelSight center dots on the correct camera thumbnail in the preview.

    Preview row 1 layout:  [cam0 320px] [cam1 320px] [cam2 320px] [optitrack 320px]
    """
    x_offset = cam_index * RS_THUMB_W
    y_offset = 0

    for label, pixel, color in [("L", pixel_left, GEL_COLORS["left"]),
                                 ("R", pixel_right, GEL_COLORS["right"])]:
        if pixel is None:
            continue
        # Scale from original resolution to thumbnail
        tx = int(pixel[0] * RS_THUMB_W / ORIG_W) + x_offset
        ty = int(pixel[1] * RS_THUMB_H / ORIG_H) + y_offset

        # Bounds check
        if 0 <= tx < preview.shape[1] and 0 <= ty < preview.shape[0]:
            cv2.circle(preview, (tx, ty), 6, color, -1)
            cv2.circle(preview, (tx, ty), 7, (255, 255, 255), 1)
            cv2.putText(preview, label, (tx + 10, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a TWM episode with GelSight center projection")
    parser.add_argument("file", help="Path to episode .h5 file")
    parser.add_argument("--fps", type=float, default=None,
                        help="Playback FPS (default: use recorded FPS from metadata)")
    parser.add_argument("--cam_calib", type=str, required=True,
                        help="Path to T_mocap_to_cam.json")
    parser.add_argument("--gel_left", type=str, required=True,
                        help="Path to T_gel_to_rigid_left.json")
    parser.add_argument("--gel_right", type=str, required=True,
                        help="Path to T_gel_to_rigid_right.json")
    parser.add_argument("--cam_index", type=int, default=None,
                        help="Override camera index in preview (0/1/2). "
                             "Default: auto-detect from serial in calibration file.")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    # ── Load calibrations ────────────────────────────────────────────────────
    T_mocap_to_cam, intrinsics, cam_serial, gel_center_left, gel_center_right = \
        load_calibrations(args.cam_calib, args.gel_left, args.gel_right)

    # Determine which cam index in the preview grid
    if args.cam_index is not None:
        cam_index = args.cam_index
    else:
        try:
            cam_index = REALSENSE_SERIALS.index(cam_serial)
        except ValueError:
            print(f"Warning: camera serial {cam_serial} not found in REALSENSE_SERIALS, defaulting to cam2")
            cam_index = 2

    print(f"Projection: overhead camera = cam{cam_index} (serial {cam_serial})")
    print(f"  GelSight left  center (rigid): [{gel_center_left[0]:.2f}, {gel_center_left[1]:.2f}, {gel_center_left[2]:.2f}] mm")
    print(f"  GelSight right center (rigid): [{gel_center_right[0]:.2f}, {gel_center_right[1]:.2f}, {gel_center_right[2]:.2f}] mm")
    print(f"  RMSE of mocap→cam calibration: {json.load(open(args.cam_calib))['rmse_mm']:.2f} mm")
    print()

    # ── Open HDF5 ────────────────────────────────────────────────────────────
    f = h5py.File(args.file, "r")
    n_frames = int(f["timestamps"].shape[0])
    if n_frames == 0:
        print("Episode has 0 frames.")
        f.close()
        sys.exit(1)

    fps          = args.fps or float(f["metadata"].attrs.get("fps", 30))
    fps_override = args.fps is not None
    task_name    = str(f["metadata"].attrs.get("task", ""))
    tick_dt      = 1.0 / fps
    timestamps   = f["timestamps"][:]
    optitrack    = load_optitrack(f)

    _blank_gs  = np.full((480, 640, 3), 128, dtype=np.uint8)
    gs_left_n  = int(f["gelsight/left/frames"].shape[0])
    gs_right_n = int(f["gelsight/right/frames"].shape[0])
    gs_ref = [
        f["gelsight/left/frames"][0].copy()  if gs_left_n  > 0 else _blank_gs.copy(),
        f["gelsight/right/frames"][0].copy() if gs_right_n > 0 else _blank_gs.copy(),
    ]

    paused     = False
    loop       = False
    speed      = 1
    frame_idx  = 0
    tick_times = collections.deque(maxlen=30)

    print(f"File:     {args.file}")
    print(f"Task:     {task_name or '(none)'}")
    print(f"Frames:   {n_frames}  |  FPS: {fps}  |  Duration: {n_frames / fps:.1f}s")
    print()
    print("Controls:  space=pause/resume  ←/a=prev  →/d=next  r=reset-ref  q=quit")
    print()

    try:
        while True:
            tick_start = time.time()
            frame_idx  = max(0, min(frame_idx, n_frames - 1))

            color_frames = [f[f"realsense/cam{i}/color"][frame_idx] for i in range(3)]
            gs_frames    = [
                f["gelsight/left/frames"][min(frame_idx, gs_left_n - 1)].copy()   if gs_left_n  > 0 else _blank_gs.copy(),
                f["gelsight/right/frames"][min(frame_idx, gs_right_n - 1)].copy() if gs_right_n > 0 else _blank_gs.copy(),
            ]
            cam_t           = float(timestamps[frame_idx])
            optitrack_poses = optitrack_at(optitrack, cam_t)
            elapsed         = cam_t - float(timestamps[0])

            preview = make_preview(
                color_frames, gs_frames, gs_ref,
                optitrack_poses,
                recording=False, frame_count=frame_idx, elapsed=elapsed,
                task_name=task_name,
            )

            # ── Project GelSight centers ─────────────────────────────────────
            sl_pose = optitrack_poses.get("sensor_left")
            sr_pose = optitrack_poses.get("sensor_right")

            pixel_left = project_gel_center(
                sl_pose[1] if sl_pose else None,
                gel_center_left, T_mocap_to_cam, intrinsics,
            )
            pixel_right = project_gel_center(
                sr_pose[1] if sr_pose else None,
                gel_center_right, T_mocap_to_cam, intrinsics,
            )

            draw_gel_projections(preview, cam_index, pixel_left, pixel_right)

            # ── Action menu + status bar ─────────────────────────────────────
            preview[240:480, 960:1280] = make_action_menu(w=320, h=240, paused=paused, loop=loop)

            tick_times.append(time.time())
            actual_fps = (len(tick_times) / (tick_times[-1] - tick_times[0])
                          if len(tick_times) >= 2 else 0.0)

            speed_str = f"{speed}x" if speed > 1 else "1x"
            status = (f"[{'PAUSED' if paused else 'PLAYING'}]  "
                      f"frame {frame_idx + 1}/{n_frames}  |  t={elapsed:.2f}s  |  {actual_fps:.1f}fps  |  {speed_str}")
            cv2.putText(preview, status, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("TWM Data Viewer + Projection", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key in (81, ord('a')):
                paused    = True
                frame_idx = max(0, frame_idx - 1)
            elif key in (83, ord('d')):
                paused    = True
                frame_idx = min(n_frames - 1, frame_idx + 1)
            elif key == ord('1'):
                speed = 1
            elif key == ord('2'):
                speed = 2
            elif key == ord('3'):
                speed = 5
            elif key == ord('4'):
                speed = 10
            elif key == ord('l'):
                loop = not loop
                print(f"Loop {'ON' if loop else 'OFF'}")
            elif key == ord('r'):
                gs_ref = [
                    f["gelsight/left/frames"][min(frame_idx, gs_left_n - 1)].copy()   if gs_left_n  > 0 else _blank_gs.copy(),
                    f["gelsight/right/frames"][min(frame_idx, gs_right_n - 1)].copy() if gs_right_n > 0 else _blank_gs.copy(),
                ]
                print(f"GelSight diff reference reset to frame {frame_idx + 1}")

            if not paused:
                frame_idx += 1
                if frame_idx >= n_frames:
                    if loop:
                        frame_idx = 0
                    else:
                        frame_idx = n_frames - 1
                        paused    = True
                        print("End of episode.")

            if not paused:
                if fps_override or frame_idx >= n_frames - 1:
                    target_dt = tick_dt
                else:
                    target_dt = float(timestamps[frame_idx] - timestamps[frame_idx - 1]) if frame_idx > 0 else tick_dt
                sleep_t = target_dt / speed - (time.time() - tick_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    finally:
        f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()