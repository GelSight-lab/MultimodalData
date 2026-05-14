#!/usr/bin/env python3
"""
Visualize a TWM episode with GelSight contact centers projected onto the overhead camera.

Extends twm.visualize with OptiTrack→camera projection:
  - Reads GelSight-to-rigid-body calibrations (left/right)
  - Reads OptiTrack→camera extrinsic calibration
  - Projects GelSight surface centers onto the overhead RealSense view each frame

Usage:
    python -m twm.visualize_projection path/to/episode_000.h5

    # Optional overrides:
    python -m twm.visualize_projection path/to/episode_000.h5 \
        --cam_calib  twm/calibration/result/T_mocap_to_cam_middle.json \
        --gel_left   twm/calibration/result/T_gel_to_rigid_left.json \
        --gel_right  twm/calibration/result/T_gel_to_rigid_right.json \
        --save_video output.mp4

Controls are the same as twm.visualize.
"""

import argparse
import collections
import glob
import json
import os
import sys
import threading
import time
import numpy as np
import h5py
import cv2
from scipy.spatial.transform import Rotation

from twm.data_collection import make_preview, REALSENSE_SERIALS


# ──────────────────────────────────────────────────────────────────────────────
# Frame prefetcher
# ──────────────────────────────────────────────────────────────────────────────

class FramePrefetcher:
    """Background thread that reads HDF5 frames ahead to hide disk latency."""

    def __init__(self, filepath, n_frames, gs_left_n, gs_right_n, buffer_size=10):
        self._filepath   = filepath
        self._n_frames   = n_frames
        self._gs_left_n  = gs_left_n
        self._gs_right_n = gs_right_n
        self._buf_size   = buffer_size
        self._cache      = {}
        self._lock       = threading.Lock()
        self._head       = 0
        self._stop       = False
        self._thread     = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _read_frame(self, f, idx):
        _blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        color = [f[f"realsense/cam{i}/color"][idx] for i in range(3)]
        gs = [
            f["gelsight/left/frames"][min(idx, self._gs_left_n - 1)].copy()
                if self._gs_left_n  > 0 else _blank.copy(),
            f["gelsight/right/frames"][min(idx, self._gs_right_n - 1)].copy()
                if self._gs_right_n > 0 else _blank.copy(),
        ]
        return color, gs

    def _worker(self):
        f = h5py.File(self._filepath, "r")
        try:
            while not self._stop:
                with self._lock:
                    head   = self._head
                    target = next(
                        (head + i for i in range(self._buf_size)
                         if head + i < self._n_frames and head + i not in self._cache),
                        None,
                    )
                if target is None:
                    time.sleep(0.005)
                    continue
                data = self._read_frame(f, target)
                with self._lock:
                    self._cache[target] = data
                    for k in [k for k in self._cache
                               if k < self._head or k >= self._head + self._buf_size]:
                        del self._cache[k]
        finally:
            f.close()

    def get(self, frame_idx, f_fallback):
        with self._lock:
            if abs(frame_idx - self._head) > self._buf_size:
                self._cache.clear()
            self._head = frame_idx
            data = self._cache.get(frame_idx)
        return data if data is not None else self._read_frame(f_fallback, frame_idx)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Reuse helpers from visualize.py
# ──────────────────────────────────────────────────────────────────────────────

SPEEDS = [1, 2, 5, 10, 25, 50]

ACTIONS = [
    ("SPACE",     "pause / resume"),
    ("-> / d",    "next frame"),
    ("<- / a",    "prev frame"),
    ("1/2/3/4/5/6", "speed 1x/2x/5x/10x/25x/50x"),
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


def project_gel_pose(rigid_pose_7, gel_center_in_rigid_mm, T_mocap_to_cam, intrinsics,
                     axis_len_mm=120.0):
    """
    Project GelSight center + rigid body coordinate axes into the camera image.

    Returns:
        center_pixel: (u, v) or None
        axis_pixels:  [(ux,vx), (uy,vy), (uz,vz)] — X/Y/Z axis tips, each may be None
    """
    if rigid_pose_7 is None:
        return None, None

    pos_mm = np.array(rigid_pose_7[:3]) * 1000.0
    quat   = rigid_pose_7[3:]
    T_rigid_to_mocap = pose_to_matrix(pos_mm, quat)
    R = T_rigid_to_mocap[:3, :3]

    P_gel_mocap = (T_rigid_to_mocap @ np.append(gel_center_in_rigid_mm, 1.0))[:3]

    axis_tips_mocap = [
        P_gel_mocap + R @ np.array([axis_len_mm, 0.0, 0.0]),
        P_gel_mocap + R @ np.array([0.0, axis_len_mm, 0.0]),
        P_gel_mocap + R @ np.array([0.0, 0.0, axis_len_mm]),
    ]

    def _project(P_mocap):
        P_cam = (T_mocap_to_cam @ np.append(P_mocap, 1.0))[:3]
        if P_cam[2] <= 0:
            return None
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["ppx"], intrinsics["ppy"]
        return (fx * P_cam[0] / P_cam[2] + cx,
                fy * P_cam[1] / P_cam[2] + cy)

    center_pixel = _project(P_gel_mocap)
    if center_pixel is None:
        return None, None

    return center_pixel, [_project(tip) for tip in axis_tips_mocap]


def load_calibrations(cam_calib_paths, gel_left_path, gel_right_path):
    """Load all calibration JSON files. 
    Returns:
      cam_calibs: list of dicts with keys: 'T_mocap_to_cam', 'intrinsics', 'cam_serial'
      gel_center_left: (3,) np array
      gel_center_right: (3,) np array
    """
    cam_calibs = []
    
    for path in cam_calib_paths:
        if not os.path.isfile(path):
            print(f"Notice: Calibration file not found: {path} (skipping)")
            continue
        with open(path, 'r') as fp:
            data = json.load(fp)
        cam_calibs.append({
            "T_mocap_to_cam": np.array(data["T_mocap_to_cam"], dtype=np.float64),
            "intrinsics": data["intrinsics"],
            "camera_serial": data["camera_serial"],
            "rmse_mm": data.get("rmse_mm", 0.0),
            "path": path
        })

    with open(gel_left_path, 'r') as fp:
        gel_left = json.load(fp)
    gel_center_left = np.array(gel_left["gel_center_in_rigid_mm"], dtype=np.float64)

    with open(gel_right_path, 'r') as fp:
        gel_right = json.load(fp)
    gel_center_right = np.array(gel_right["gel_center_in_rigid_mm"], dtype=np.float64)

    return cam_calibs, gel_center_left, gel_center_right


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
_AXIS_COLORS  = [(0, 0, 255), (0, 255, 0), (255, 128, 0)]
_AXIS_LABELS  = ["X", "Y", "Z"]


def draw_gel_projections(preview, cam_index, result_left, result_right):
    """
    Draw projected GelSight center dot + coordinate axes on the correct camera thumbnail.
    result_left/right: (center_pixel, axis_pixels) tuples from project_gel_pose.
    """
    x_offset = cam_index * RS_THUMB_W

    for label, result, dot_color in [("L", result_left,  GEL_COLORS["left"]),
                                      ("R", result_right, GEL_COLORS["right"])]:
        if result is None:
            continue
        center_pixel, axis_pixels = result
        if center_pixel is None:
            continue

        tx = int(center_pixel[0] * RS_THUMB_W / ORIG_W) + x_offset
        ty = int(center_pixel[1] * RS_THUMB_H / ORIG_H)
        if not (0 <= tx < preview.shape[1] and 0 <= ty < preview.shape[0]):
            continue

        # Draw the 3 axes first (behind the dot)
        for ax_pixel, ax_color, ax_label in zip(axis_pixels, _AXIS_COLORS, _AXIS_LABELS):
            if ax_pixel is None:
                continue
            atx = int(ax_pixel[0] * RS_THUMB_W / ORIG_W) + x_offset
            aty = int(ax_pixel[1] * RS_THUMB_H / ORIG_H)
            cv2.line(preview, (tx, ty), (atx, aty), ax_color, 2, cv2.LINE_AA)
            if 0 <= atx < preview.shape[1] and 0 <= aty < preview.shape[0]:
                cv2.putText(preview, ax_label, (atx + 3, aty - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, ax_color, 1, cv2.LINE_AA)

        # Center dot on top
        cv2.circle(preview, (tx, ty), 5, dot_color, -1, cv2.LINE_AA)
        cv2.circle(preview, (tx, ty), 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(preview, label, (tx + 8, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def process_episode(h5_path, out_video_path, args,
                    project_cams, gel_center_left, gel_center_right):
    """Play back (or export) a single episode. If out_video_path is set, runs headless."""
    if not os.path.isfile(h5_path):
        print(f"File not found: {h5_path}")
        return

    # ── Open HDF5 ────────────────────────────────────────────────────────────
    f = h5py.File(h5_path, "r")
    n_frames = int(f["timestamps"].shape[0])
    if n_frames == 0:
        print(f"Episode has 0 frames: {h5_path}")
        f.close()
        return

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

    prefetcher = FramePrefetcher(h5_path, n_frames, gs_left_n, gs_right_n)

    paused     = False
    loop       = False
    speed      = 1
    frame_idx  = 0
    tick_times = collections.deque(maxlen=30)

    WIN = "TWM Data Viewer + Projection"
    seek = {"requested": None}
    if out_video_path is None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", WIN, 0, max(1, n_frames - 1),
                           lambda pos: seek.__setitem__("requested", pos))

    print(f"File:     {h5_path}")
    print(f"Task:     {task_name or '(none)'}")
    print(f"Frames:   {n_frames}  |  FPS: {fps}  |  Duration: {n_frames / fps:.1f}s")
    print()
    if out_video_path is None:
        print("Controls:  space=pause/resume  ←/a=prev  →/d=next  r=reset-ref  q=quit")
        print()

    video_writer = None
    if out_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_video_path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps if fps else 30.0
        video_writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (1280, 480))
        print(f"Saving video to: {out_video_path}")

    try:
        while True:
            tick_start = time.time()

            if seek["requested"] is not None and seek["requested"] != frame_idx:
                frame_idx = seek["requested"]
            seek["requested"] = None

            frame_idx = max(0, min(frame_idx, n_frames - 1))

            color_frames, gs_frames = prefetcher.get(frame_idx, f)
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

            for pc in project_cams:
                result_left = project_gel_pose(
                    sl_pose[1] if sl_pose else None,
                    gel_center_left, pc["T_mocap_to_cam"], pc["intrinsics"],
                )
                result_right = project_gel_pose(
                    sr_pose[1] if sr_pose else None,
                    gel_center_right, pc["T_mocap_to_cam"], pc["intrinsics"],
                )
                draw_gel_projections(preview, pc["index"], result_left, result_right)

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

            if video_writer is not None:
                video_writer.write(preview)
                if frame_idx % 100 == 0:
                    print(f"  wrote frame {frame_idx + 1}/{n_frames}")
                if frame_idx >= n_frames - 1:
                    break
                key = 0xFF
            else:
                cv2.imshow(WIN, preview)
                if cv2.getTrackbarPos("Frame", WIN) != frame_idx:
                    cv2.setTrackbarPos("Frame", WIN, frame_idx)
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
            elif key == ord('5'):
                speed = 25
            elif key == ord('6'):
                speed = 50
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
                frame_idx += speed
                if frame_idx >= n_frames:
                    if loop:
                        frame_idx = frame_idx % n_frames
                    else:
                        frame_idx = n_frames - 1
                        paused    = True
                        print("End of episode.")

            if not paused and video_writer is None:
                if fps_override or frame_idx >= n_frames - 1 or frame_idx - speed < 0:
                    target_dt = tick_dt
                else:
                    target_dt = float(timestamps[frame_idx] - timestamps[frame_idx - speed]) / speed
                sleep_t = target_dt - (time.time() - tick_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    finally:
        prefetcher.stop()
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()
        f.close()
        if out_video_path is None:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a TWM episode with GelSight center projection")
    parser.add_argument("path", help="Path to episode .h5 file OR a directory of .h5 files")
    parser.add_argument("--fps", type=float, default=None,
                        help="Playback FPS (default: use recorded FPS from metadata)")
    parser.add_argument("--cam_calib", type=str, nargs='+',
                        default=[
                            "twm/calibration/result/T_mocap_to_cam_middle.json",
                            "twm/calibration/result/T_mocap_to_cam_left.json",
                            "twm/calibration/result/T_mocap_to_cam_right.json"
                        ],
                        help="Path(s) to T_mocap_to_cam_<name>.json (one per camera)")
    parser.add_argument("--gel_left", type=str,
                        default="twm/calibration/result/T_gel_to_rigid_left.json",
                        help="Path to T_gel_to_rigid_left.json")
    parser.add_argument("--gel_right", type=str,
                        default="twm/calibration/result/T_gel_to_rigid_right.json",
                        help="Path to T_gel_to_rigid_right.json")
    parser.add_argument("--save_video", type=str, default=None,
                        help="Single-file mode: path to output mp4.")
    parser.add_argument("--save_videos", action="store_true",
                        help="Directory mode: auto-save each episode as <stem>.mp4.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory mode: write mp4s here (default: next to each .h5).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Directory mode: re-render episodes whose mp4 already exists.")
    args = parser.parse_args()

    # ── Resolve input: file or directory ─────────────────────────────────────
    if os.path.isdir(args.path):
        h5_files = sorted(glob.glob(os.path.join(args.path, "*.h5")))
        if not h5_files:
            print(f"No .h5 files found in {args.path}")
            sys.exit(1)
        directory_mode = True
    elif os.path.isfile(args.path):
        h5_files = [args.path]
        directory_mode = False
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)

    # ── Load calibrations once ───────────────────────────────────────────────
    cam_calibs, gel_center_left, gel_center_right = \
        load_calibrations(args.cam_calib, args.gel_left, args.gel_right)

    project_cams = []
    for calib in cam_calibs:
        serial = calib["camera_serial"]
        try:
            c_idx = REALSENSE_SERIALS.index(serial)
        except ValueError:
            print(f"Warning: camera serial {serial} not found in REALSENSE_SERIALS, skipping.")
            continue
        project_cams.append({
            "index": c_idx,
            "T_mocap_to_cam": calib["T_mocap_to_cam"],
            "intrinsics": calib["intrinsics"],
            "serial": serial,
            "rmse": calib["rmse_mm"],
        })

    print(f"Loaded {len(project_cams)} camera calibrations for projection:")
    for pc in project_cams:
        print(f"  - index: {pc['index']} (serial {pc['serial']}) | RMSE: {pc['rmse']:.2f} mm")
    print(f"  GelSight left  center (rigid): [{gel_center_left[0]:.2f}, {gel_center_left[1]:.2f}, {gel_center_left[2]:.2f}] mm")
    print(f"  GelSight right center (rigid): [{gel_center_right[0]:.2f}, {gel_center_right[1]:.2f}, {gel_center_right[2]:.2f}] mm")
    print()

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if directory_mode:
        if not args.save_videos:
            print("Directory input requires --save_videos (batch export only).")
            sys.exit(1)
        out_dir = args.out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"Batch processing {len(h5_files)} episode(s) from {args.path}\n")
        for i, h5_path in enumerate(h5_files, 1):
            stem = os.path.splitext(os.path.basename(h5_path))[0]
            out_video = os.path.join(out_dir or os.path.dirname(h5_path), f"{stem}.mp4")
            if os.path.exists(out_video) and not args.overwrite:
                print(f"[{i}/{len(h5_files)}] SKIP (exists): {out_video}")
                continue
            print(f"[{i}/{len(h5_files)}] {h5_path} → {out_video}")
            process_episode(h5_path, out_video, args,
                            project_cams, gel_center_left, gel_center_right)
            print()
    else:
        process_episode(h5_files[0], args.save_video, args,
                        project_cams, gel_center_left, gel_center_right)


if __name__ == "__main__":
    main()