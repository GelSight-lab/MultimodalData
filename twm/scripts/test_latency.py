"""Measure tactile latency in a recording that has per-sensor GelSight
timestamps (new pipeline). Renders a 2-up comparison around the strongest
contact onset:

    TOP    = INDEX alignment      (cam[i] | gelsight[i])        — naive
    BOTTOM = TIMESTAMP alignment  (cam[i] | gelsight@nearest ts) — corrected

If the GelSight is tapped while in a camera's view, in TOP the deformation
lags the visual contact; in BOTTOM they coincide (timestamps align them).

Usage:
    python scripts/test_latency.py --h5 /media/.../episode_001.h5 --side left --cam 2
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa
import numpy as np

FPS = 30
PANEL = (640, 480)


def l2(a, b):
    return np.sqrt(((a.astype(np.float32) - b.astype(np.float32)) ** 2).sum(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--cam", type=int, default=2, help="RealSense cam idx the gel is visible in")
    ap.add_argument("--win", type=int, default=90, help="frames around onset")
    ap.add_argument("--out", default="/tmp/latency_test.mp4")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    cam_ts = f["timestamps"][:]
    T = len(cam_ts)
    gs = f[f"gelsight/{args.side}/frames"]
    has_gts = "timestamps" in f[f"gelsight/{args.side}"]
    gs_ts = f[f"gelsight/{args.side}/timestamps"][:] if has_gts else cam_ts.copy()
    print(f"frames={T} cam_fps={T/(cam_ts[-1]-cam_ts[0]):.1f} "
          f"per_sensor_ts={'YES' if has_gts else 'NO (old format)'}")

    # contact onset: tactile intensity vs a quiet reference
    ref = gs[min(10, T - 1)].astype(np.float32)
    inten = np.array([l2(gs[i], ref).mean() for i in range(T)])
    onset = int(inten.argmax())
    print(f"strongest contact: frame {onset} intensity={inten[onset]:.1f}")
    if inten[onset] < 12:
        print("WARNING: no strong contact event (max intensity < 12). "
              "Re-record while firmly tapping the gel IN a camera's view.")

    a = max(0, onset - args.win // 2)
    b = min(T, a + args.win)

    # nearest-timestamp gelsight frame for each cam frame
    def nearest_gs(t):
        return int(np.argmin(np.abs(gs_ts - t)))

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{PANEL[0]*2}x{PANEL[1]*2}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(args.out)], stdin=subprocess.PIPE)

    cam_ds = f[f"realsense/cam{args.cam}/color"]
    for i in range(a, b):
        cam = cam_ds[i]                                   # BGR
        gel_idx = gs[i]                                   # INDEX-aligned
        j = nearest_gs(cam_ts[i])
        gel_ts = gs[j]                                    # TIMESTAMP-aligned
        def lab(img, txt, color):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (PANEL[0], 22), (0, 0, 0), -1)
            cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            return img
        top = np.hstack([lab(cam, f"cam{args.cam} f{i}", (0,255,180)),
                         lab(gel_idx, f"INDEX gel[{i}]", (120,180,255))])
        bot = np.hstack([lab(cam, f"cam{args.cam} f{i}", (0,255,180)),
                         lab(gel_ts, f"TIMESTAMP gel[{j}] dt={ (gs_ts[j]-cam_ts[i])*1000:+.0f}ms", (0,255,180))])
        proc.stdin.write(np.vstack([top, bot]).tobytes())
    proc.stdin.close(); proc.wait()
    f.close()
    print(f"wrote {args.out}  (top=index-aligned, bottom=timestamp-aligned)")
    # numeric staleness
    stale = np.abs(cam_ts - gs_ts[:T] if len(gs_ts) >= T else cam_ts[:len(gs_ts)] - gs_ts).mean() if has_gts else 0
    print(f"mean index-alignment staleness = {stale*1000:.0f}ms = {stale*FPS:.1f} frames "
          f"(timestamp alignment removes this)")


if __name__ == "__main__":
    main()
