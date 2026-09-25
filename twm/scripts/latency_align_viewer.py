"""Interactive latency-alignment viewer for a raw recording H5.

Shows a camera next to both GelSight streams (raw + diff) and lets you shift
the tactile stream relative to the cameras in real time to find/verify the
alignment offset. Works with the new per-sensor `gelsight/<side>/timestamps`
(timestamp mode) or by frame index (index mode).

Panel (1600x480):
  [ cam | gsL raw | gsL diff | gsR raw | gsR diff ]

Controls:
  space            play / pause
  -> / d           next frame        <- / a   previous frame
  . / =            latency +1 frame   , / -    latency -1 frame
  t                toggle index  <->  timestamp alignment
  r                reset GelSight diff reference to current (compensated) frame
  [ / ]            change camera (0/1/2)
  q                quit

Latency: the GelSight shown at view-frame i is gelsight[i + latency] (index
mode) or the gelsight frame whose timestamp is nearest cam_ts[i] + latency*dt
(timestamp mode). Positive latency pulls tactile EARLIER to match the camera.

    python scripts/latency_align_viewer.py --h5 /media/.../episode_001.h5 --cam 2
"""
from __future__ import annotations

import argparse

import cv2
import h5py
import hdf5plugin  # noqa
import numpy as np

CW, CH = 320, 240          # per-cell size
FPS = 30


def l2(a, b):
    return np.sqrt(((a.astype(np.float32) - b.astype(np.float32)) ** 2).sum(2))


def diff_img(frame, ref):
    d = np.clip(frame.astype(np.int16) - ref.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    return d


class Aligner:
    def __init__(self, h5_path, cam=2):
        self.f = h5py.File(h5_path, "r")
        self.cam_ts = self.f["timestamps"][:]
        self.T = len(self.cam_ts)
        self.cam = cam
        self.has_gts = "timestamps" in self.f["gelsight/left"]
        self.gl = self.f["gelsight/left/frames"]
        self.gr = self.f["gelsight/right/frames"]
        self.gl_ts = self.f["gelsight/left/timestamps"][:] if self.has_gts else self.cam_ts
        self.gr_ts = self.f["gelsight/right/timestamps"][:] if self.has_gts else self.cam_ts
        self.refL = self.gl[min(5, self.T - 1)]
        self.refR = self.gr[min(5, self.T - 1)]
        self.mode_ts = self.has_gts        # default to timestamp alignment if available
        self.dt = float(np.median(np.diff(self.cam_ts))) if self.T > 1 else 1 / FPS

    def gel_index(self, side_ts, view_i, latency):
        """Resolve which gelsight frame to show for view frame `view_i`."""
        if self.mode_ts:
            target = self.cam_ts[view_i] + latency * self.dt
            return int(np.argmin(np.abs(side_ts - target)))
        return int(np.clip(view_i + latency, 0, len(side_ts) - 1))

    def panel(self, view_i, latency):
        cam = cv2.resize(self.f[f"realsense/cam{self.cam}/color"][view_i], (CW, CH))
        jl = self.gel_index(self.gl_ts, view_i, latency)
        jr = self.gel_index(self.gr_ts, view_i, latency)
        gL = self.gl[jl]; gR = self.gr[jr]
        cells = [
            cam,
            cv2.resize(gL, (CW, CH)),
            cv2.resize(diff_img(gL, self.refL), (CW, CH)),
            cv2.resize(gR, (CW, CH)),
            cv2.resize(diff_img(gR, self.refR), (CW, CH)),
        ]
        row = np.hstack(cells)
        mode = "TIMESTAMP" if self.mode_ts else "INDEX"
        dtl = (self.gl_ts[jl] - self.cam_ts[view_i]) * 1000
        bar = np.zeros((28, row.shape[1], 3), np.uint8)
        txt = (f"frame {view_i}/{self.T-1}  cam{self.cam}  "
               f"latency={latency:+d}f ({latency*self.dt*1000:+.0f}ms)  mode={mode}  "
               f"gsL_dt={dtl:+.0f}ms")
        cv2.putText(bar, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)
        labels = ["camera", "gsL raw", "gsL diff", "gsR raw", "gsR diff"]
        head = np.zeros((20, row.shape[1], 3), np.uint8)
        for k, lab in enumerate(labels):
            cv2.putText(head, lab, (k * CW + 6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        return np.vstack([head, row, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--latency", type=int, default=0, help="initial latency offset (frames)")
    args = ap.parse_args()

    al = Aligner(args.h5, cam=args.cam)
    print(f"T={al.T}  per_sensor_ts={'YES' if al.has_gts else 'NO'}  "
          f"start mode={'TIMESTAMP' if al.mode_ts else 'INDEX'}")
    WIN = "latency align"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", WIN, 0, max(1, al.T - 1), lambda v: None)
    cv2.createTrackbar("latency+30", WIN, 30 + args.latency, 60, lambda v: None)  # offset by +30

    i = 0; latency = args.latency; playing = False
    while True:
        # sync trackbars
        ti = cv2.getTrackbarPos("frame", WIN)
        if ti != i:
            i = ti; playing = False
        latency = cv2.getTrackbarPos("latency+30", WIN) - 30

        i = max(0, min(i, al.T - 1))
        cv2.imshow(WIN, al.panel(i, latency))
        if cv2.getTrackbarPos("frame", WIN) != i:
            cv2.setTrackbarPos("frame", WIN, i)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord(' '): playing = not playing
        elif k in (83, ord('d')): i = min(i + 1, al.T - 1); playing = False
        elif k in (81, ord('a')): i = max(i - 1, 0); playing = False
        elif k in (ord('.'), ord('=')): latency += 1; cv2.setTrackbarPos("latency+30", WIN, 30 + latency)
        elif k in (ord(','), ord('-')): latency -= 1; cv2.setTrackbarPos("latency+30", WIN, 30 + latency)
        elif k == ord('t'): al.mode_ts = not al.mode_ts
        elif k == ord('r'):
            al.refL = al.gl[al.gel_index(al.gl_ts, i, latency)]
            al.refR = al.gr[al.gel_index(al.gr_ts, i, latency)]
        elif k == ord('['): al.cam = max(0, al.cam - 1)
        elif k == ord(']'): al.cam = min(2, al.cam + 1)

        if playing:
            i = min(i + 1, al.T - 1)
            if i >= al.T - 1: playing = False
    cv2.destroyAllWindows()
    al.f.close()
    print(f"final latency offset = {latency:+d} frames "
          f"({latency*al.dt*1000:+.0f}ms), mode={'TIMESTAMP' if al.mode_ts else 'INDEX'}")


if __name__ == "__main__":
    main()
