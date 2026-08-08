#!/usr/bin/env python3
"""
Visualize a TWM episode HDF5 file. By default the overlay draws the GelSight
contact centers projected onto each RealSense view using the calibrations in
the task's calibration epoch (see `calib_epoch`). Pass `--no_projection` to skip the overlay (and the
calibration loading).

Usage:
    # Interactive playback with projection overlay (default)
    python -m twm.visualize path/to/episode_000.h5

    # Same, but no overlay
    python -m twm.visualize path/to/episode_000.h5 --no_projection

    # Export every episode in a directory to mp4 (with overlay)
    python -m twm.visualize path/to/ --save_videos

    # Override calibration paths (rarely needed; default is the input's task
    # epoch resolved via calib_epoch)
    python -m twm.visualize path/to/episode_000.h5 \
        --cam_calib <epoch_dir>/T_mocap_to_cam_middle.json \
        --gel_left  <epoch_dir>/T_gel_to_rigid_left.json \
        --gel_right <epoch_dir>/T_gel_to_rigid_right.json \
        --save_video output.mp4

Controls:
    space       — pause / resume playback
    right / d   — next frame
    left  / a   — prev frame
    1..6        — speed 1x / 2x / 5x / 10x / 25x / 50x
    l           — toggle loop
    r           — reset GelSight diff reference
    q           — quit
"""

import argparse
import collections
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path
import numpy as np
import h5py
import cv2

# The calibration epoch is PER TASK (May-12 for motherboard, June-26 for
# pushT) and is resolved from the input path at parse time via calib_epoch —
# a constant here defaulted every task to June-26.

from twm.data_collection import make_preview, REALSENSE_SERIALS
from twm.viz import (
    project_gel_pose,
    load_calibrations,
    load_optitrack,
    optitrack_at,
    draw_projection_overlay,
)


# ──────────────────────────────────────────────────────────────────────────────
# Frame prefetcher
# ──────────────────────────────────────────────────────────────────────────────

class FramePrefetcher:
    """Background thread that reads HDF5 frames ahead to hide disk latency."""

    def __init__(self, filepath, n_frames, gs_left_n, gs_right_n, buffer_size=10,
                 tactile_latency=0):
        self._filepath   = filepath
        self._n_frames   = n_frames
        self._gs_left_n  = gs_left_n
        self._gs_right_n = gs_right_n
        self._buf_size   = buffer_size
        self._tac_lat    = int(tactile_latency)
        self._cache      = {}
        self._lock       = threading.Lock()
        self._head       = 0
        self._stop       = False
        self._thread     = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _read_frame(self, f, idx):
        _blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        color = [f[f"realsense/cam{i}/color"][idx] for i in range(3)]
        # Tactile-latency compensation: pull gelsight from idx+lat so that
        # delayed tactile data is shifted forward to match the vision frame.
        gs_idx = idx + self._tac_lat
        gs = [
            f["gelsight/left/frames"][max(0, min(gs_idx, self._gs_left_n - 1))].copy()
                if self._gs_left_n  > 0 else _blank.copy(),
            f["gelsight/right/frames"][max(0, min(gs_idx, self._gs_right_n - 1))].copy()
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


# load_optitrack, optitrack_at, project_gel_pose, load_calibrations, and
# draw_projection_overlay are imported from twm.viz (single source of truth).


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
    tac_lat    = int(getattr(args, "tactile_latency", 0))
    ref_idx_L  = max(0, min(tac_lat, gs_left_n  - 1)) if gs_left_n  > 0 else 0
    ref_idx_R  = max(0, min(tac_lat, gs_right_n - 1)) if gs_right_n > 0 else 0
    gs_ref = [
        f["gelsight/left/frames"][ref_idx_L].copy()  if gs_left_n  > 0 else _blank_gs.copy(),
        f["gelsight/right/frames"][ref_idx_R].copy() if gs_right_n > 0 else _blank_gs.copy(),
    ]
    if tac_lat != 0:
        print(f"Tactile latency compensation: gelsight pulled from h5_frame + {tac_lat}")

    prefetcher = FramePrefetcher(h5_path, n_frames, gs_left_n, gs_right_n,
                                 tactile_latency=tac_lat)

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

            # ── Project GelSight centers (single source of truth) ────────────
            draw_projection_overlay(
                preview, optitrack_poses, project_cams,
                gel_center_left, gel_center_right,
            )

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
                shifted = frame_idx + tac_lat
                gs_ref = [
                    f["gelsight/left/frames"][max(0, min(shifted, gs_left_n - 1))].copy()   if gs_left_n  > 0 else _blank_gs.copy(),
                    f["gelsight/right/frames"][max(0, min(shifted, gs_right_n - 1))].copy() if gs_right_n > 0 else _blank_gs.copy(),
                ]
                print(f"GelSight diff reference reset to frame {frame_idx + 1}"
                      + (f" (gs idx {shifted + 1}, +{tac_lat})" if tac_lat else ""))

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
        description="Visualize a TWM episode HDF5 file (with optional "
                    "GelSight-center projection overlay; on by default).")
    parser.add_argument("path", help="Path to episode .h5 file OR a directory of .h5 files")
    parser.add_argument("--fps", type=float, default=None,
                        help="Playback FPS (default: use recorded FPS from metadata)")
    parser.add_argument("--cam_calib", type=str, nargs='+', default=None,
                        help="Path(s) to T_mocap_to_cam_<name>.json (one per "
                             "camera); default: the input's task epoch via calib_epoch")
    parser.add_argument("--gel_left", type=str, default=None,
                        help="Path to T_gel_to_rigid_left.json (default: task epoch)")
    parser.add_argument("--gel_right", type=str, default=None,
                        help="Path to T_gel_to_rigid_right.json (default: task epoch)")
    parser.add_argument("--save_video", type=str, default=None,
                        help="Single-file mode: path to output mp4.")
    parser.add_argument("--save_videos", action="store_true",
                        help="Directory mode: auto-save each episode as <stem>.mp4.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory mode: write mp4s here (default: next to each .h5).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Directory mode: re-render episodes whose mp4 already exists.")
    parser.add_argument("--no_projection", action="store_true",
                        help="Skip the GelSight→camera projection overlay "
                             "(and the calibration loading). Default: overlay ON.")
    parser.add_argument("--tactile_latency", type=int, default=3,
                        help="Frames to advance gelsight reads (h5_frame + N) to "
                             "compensate for tactile capture lag. Default: 3.")
    args = parser.parse_args()

    if not args.no_projection and None in (args.gel_left, args.gel_right,
                                           args.cam_calib):
        from twm.calib_epoch import calib_dir_for_path
        cdir = calib_dir_for_path(args.path)   # raises rather than guessing
        print(f"calibration epoch: {cdir.name} (from input path)")
        if args.cam_calib is None:
            args.cam_calib = [str(cdir / f"T_mocap_to_cam_{n}.json")
                              for n in ("middle", "left", "right")]
        args.gel_left = args.gel_left or str(cdir / "T_gel_to_rigid_left.json")
        args.gel_right = args.gel_right or str(cdir / "T_gel_to_rigid_right.json")

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

    # ── Load calibrations (skipped under --no_projection) ────────────────────
    if args.no_projection:
        project_cams = []
        gel_center_left = None
        gel_center_right = None
        print("Projection overlay: OFF (--no_projection)\n")
    else:
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