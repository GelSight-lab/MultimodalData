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
import os
import sys
import time
import numpy as np
import h5py
import cv2

from twm.data_collection import make_preview


# ──────────────────────────────────────────────────────────────────────────────
# Action menu panel
# ──────────────────────────────────────────────────────────────────────────────

ACTIONS = [
    ("SPACE",   "pause / resume"),
    ("→ / d",   "next frame"),
    ("← / a",   "prev frame"),
    ("r",       "reset diff reference"),
    ("q",       "quit"),
]


def make_action_menu(w=320, h=240, paused=False):
    """Render a controls legend panel matching the blank slot in the preview grid."""
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.putText(panel, "Controls", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.line(panel, (10, 32), (w - 10, 32), (80, 80, 80), 1)

    y = 58
    for key, desc in ACTIONS:
        # Highlight the relevant play/pause entry
        highlight = (key == "SPACE")
        key_color  = (0, 220, 255) if highlight else (140, 200, 140)
        desc_color = (220, 220, 220) if highlight else (160, 160, 160)

        cv2.putText(panel, f"[{key}]", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, key_color, 1)
        cv2.putText(panel, desc, (110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, desc_color, 1)
        y += 32

    # Playback state indicator
    state_text  = "|| PAUSED" if paused else "> PLAYING"
    state_color = (0, 140, 255) if paused else (0, 220, 80)
    cv2.putText(panel, state_text, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

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
        idx = int(np.argmin(np.abs(ts - camera_timestamp)))
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

    fps        = args.fps or float(f["metadata"].attrs.get("fps", 30))
    tick_dt    = 1.0 / fps
    timestamps = f["timestamps"][:]
    optitrack  = load_optitrack(f)

    # GelSight diff reference starts at frame 0
    gs_ref = [
        f["gelsight/left/frames"][0].copy(),
        f["gelsight/right/frames"][0].copy(),
    ]

    paused    = False
    frame_idx = 0

    print(f"File:     {args.file}")
    print(f"Frames:   {n_frames}  |  FPS: {fps}  |  Duration: {n_frames / fps:.1f}s")
    print()
    print("Controls:  space=pause/resume  ←/a=prev  →/d=next  r=reset-ref  q=quit")
    print()

    try:
        while True:
            tick_start = time.time()
            frame_idx  = max(0, min(frame_idx, n_frames - 1))

            # Load this frame from HDF5
            color_frames = [f[f"realsense/cam{i}/color"][frame_idx] for i in range(3)]
            gs_frames    = [
                f["gelsight/left/frames"][frame_idx],
                f["gelsight/right/frames"][frame_idx],
            ]
            cam_t           = float(timestamps[frame_idx])
            optitrack_poses = optitrack_at(optitrack, cam_t)
            elapsed         = cam_t - float(timestamps[0])

            # Reuse make_preview from twm_data_collection (recording=False → IDLE colours)
            preview = make_preview(
                color_frames, gs_frames, gs_ref,
                optitrack_poses,
                recording=False, frame_count=frame_idx, elapsed=elapsed,
            )

            # Replace the blank slot (bottom-right 320×240) with the action menu
            preview[240:480, 960:1280] = make_action_menu(w=320, h=240, paused=paused)

            # Playback status bar (top of frame, different colour to collection status)
            status = (f"[{'PAUSED' if paused else 'PLAYING'}]  "
                      f"frame {frame_idx + 1}/{n_frames}  |  t={elapsed:.2f}s")
            cv2.putText(preview, status, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("TWM Data Viewer", preview)
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
            elif key == ord('r'):
                gs_ref = [
                    f["gelsight/left/frames"][frame_idx].copy(),
                    f["gelsight/right/frames"][frame_idx].copy(),
                ]
                print(f"GelSight diff reference reset to frame {frame_idx + 1}")

            if not paused:
                frame_idx += 1
                if frame_idx >= n_frames:
                    frame_idx = n_frames - 1
                    paused    = True
                    print("End of episode.")

            # Rate-limit to playback FPS only when playing
            if not paused:
                sleep_t = tick_dt - (time.time() - tick_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

    finally:
        f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
