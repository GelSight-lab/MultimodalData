#!/usr/bin/env python3
"""
Visualize a saved TWM data collection episode.

Usage:
    python -m twm.visualize path/to/episode_000.h5
    python -m twm.visualize path/to/episode_000.h5 --fps 15

Controls:
    space       — pause / resume playback
    right / d   — next frame  (works while paused)
    left  / a   — prev frame  (works while paused)
    r           — reset GelSight diff reference to current frame
    q           — quit
"""

import argparse
import collections
import os
import sys
import time
import numpy as np
import h5py
import cv2

from twm.data_collection import make_preview


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
# Action menu panel
# ──────────────────────────────────────────────────────────────────────────────

SPEEDS = [1, 2, 5, 10, 25, 50]   # available speed multipliers

ACTIONS = [
    ("SPACE",     "pause / resume"),
    ("→ / d",     "next frame"),
    ("← / a",     "prev frame"),
    ("1/2/3/4/5/6", "speed 1×/2×/5×/10×/25×/50×"),
    ("l",       "toggle loop"),
    ("r",       "reset diff reference"),
    ("q",       "quit"),
]


def make_action_menu(w=320, h=240, paused=False, loop=False):
    """Render a controls legend panel matching the blank slot in the preview grid."""
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

    # Playback state + loop indicator
    state_text  = "|| PAUSED" if paused else "> PLAYING"
    state_color = (0, 140, 255) if paused else (0, 220, 80)
    cv2.putText(panel, state_text, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

    loop_text  = "LOOP ON" if loop else "LOOP OFF"
    loop_color = (255, 180, 0) if loop else (80, 80, 80)
    cv2.putText(panel, loop_text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, loop_color, 1)

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# OptiTrack helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_optitrack(f):
    """Load all OptiTrack data from HDF5. Returns dict: name -> (timestamps, poses) or None."""
    lookup = {}
    for name in ["motherboard", "sensor_left", "sensor_right"]:
        ts    = f[f"optitrack/{name}/timestamps"][:]
        poses = f[f"optitrack/{name}/pose"][:]
        lookup[name] = (ts, poses) if len(ts) > 0 else None
    return lookup


def optitrack_at(lookup, camera_timestamp):
    """Return nearest OptiTrack pose for each tracker at the given camera timestamp."""
    result = {}
    for name, data in lookup.items():
        if data is None:
            result[name] = None
            continue
        ts, poses = data
        idx = int(np.searchsorted(ts, camera_timestamp))
        if idx == 0:
            pass  # clamp to first
        elif idx >= len(ts):
            idx = len(ts) - 1  # clamp to last
        elif abs(ts[idx - 1] - camera_timestamp) <= abs(ts[idx] - camera_timestamp):
            idx -= 1  # left neighbor is closer
        result[name] = (float(ts[idx]), poses[idx].tolist())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize a TWM episode HDF5 file")
    parser.add_argument("file", help="Path to episode .h5 file")
    parser.add_argument("--fps", type=float, default=None,
                        help="Playback FPS (default: use recorded FPS from metadata)")
    parser.add_argument("--save_video", type=str, default=None,
                        help="Path to save output video (e.g. output.mp4)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    f = h5py.File(args.file, "r")
    n_frames = int(f["timestamps"].shape[0])
    if n_frames == 0:
        print("Episode has 0 frames.")
        f.close()
        sys.exit(1)

    fps          = args.fps or float(f["metadata"].attrs.get("fps", 30))
    fps_override = args.fps is not None   # True = use fixed rate; False = use actual timestamps
    task_name    = str(f["metadata"].attrs.get("task", ""))
    tick_dt      = 1.0 / fps              # used only when fps_override is True
    timestamps = f["timestamps"][:]
    optitrack  = load_optitrack(f)

    # GelSight diff reference — use frame 0 if available, else blank grey
    _blank_gs  = np.full((480, 640, 3), 128, dtype=np.uint8)
    gs_left_n  = int(f["gelsight/left/frames"].shape[0])
    gs_right_n = int(f["gelsight/right/frames"].shape[0])
    gs_ref = [
        f["gelsight/left/frames"][0].copy()  if gs_left_n  > 0 else _blank_gs.copy(),
        f["gelsight/right/frames"][0].copy() if gs_right_n > 0 else _blank_gs.copy(),
    ]

    prefetcher = FramePrefetcher(args.file, n_frames, gs_left_n, gs_right_n)

    paused    = False
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

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps if fps else 30.0
        video_writer = cv2.VideoWriter(args.save_video, fourcc, out_fps, (1280, 480))
        print(f"Saving video to: {args.save_video}")

    try:
        while True:
            tick_start = time.time()
            frame_idx  = max(0, min(frame_idx, n_frames - 1))

            # Load this frame (from prefetch buffer or directly if cache miss)
            color_frames, gs_frames = prefetcher.get(frame_idx, f)
            cam_t           = float(timestamps[frame_idx])
            optitrack_poses = optitrack_at(optitrack, cam_t)
            elapsed         = cam_t - float(timestamps[0])

            # Reuse make_preview from twm.data_collection (recording=False → IDLE colours)
            preview = make_preview(
                color_frames, gs_frames, gs_ref,
                optitrack_poses,
                recording=False, frame_count=frame_idx, elapsed=elapsed,
                task_name=task_name,
            )

            # Replace the blank slot (bottom-right 320×240) with the action menu
            preview[240:480, 960:1280] = make_action_menu(w=320, h=240, paused=paused, loop=loop)

            tick_times.append(time.time())
            actual_fps = (len(tick_times) / (tick_times[-1] - tick_times[0])
                          if len(tick_times) >= 2 else 0.0)

            # Playback status bar (top of frame, different colour to collection status)
            speed_str = f"{speed}x" if speed > 1 else "1x"
            status = (f"[{'PAUSED' if paused else 'PLAYING'}]  "
                      f"frame {frame_idx + 1}/{n_frames}  |  t={elapsed:.2f}s  |  {actual_fps:.1f}fps  |  {speed_str}")
            cv2.putText(preview, status, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("TWM Data Viewer", preview)
            if video_writer is not None and not paused:
                video_writer.write(preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key in (81, ord('a')):   # left arrow or a
                paused    = True
                frame_idx = max(0, frame_idx - 1)
            elif key in (83, ord('d')):   # right arrow or d
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
                frame_idx += 1
                if frame_idx >= n_frames:
                    if loop:
                        frame_idx = 0
                    else:
                        frame_idx = n_frames - 1
                        paused    = True
                        print("End of episode.")

            # Rate-limit: use actual inter-frame interval from timestamps,
            # or fixed tick_dt if --fps was explicitly given.
            if not paused:
                if fps_override or frame_idx >= n_frames - 1:
                    target_dt = tick_dt
                else:
                    target_dt = float(timestamps[frame_idx] - timestamps[frame_idx - 1]) if frame_idx > 0 else tick_dt
                sleep_t = target_dt / speed - (time.time() - tick_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    finally:
        prefetcher.stop()
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()
        f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
