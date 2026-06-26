#!/usr/bin/env python3
"""
Headless detection stability test for mocap_to_cam_multi.

Opens the calibration cameras, runs the top-hat marker detector for N frames,
and reports per-camera blob-count stability (how reliably it finds the markers).
Saves an annotated snapshot per camera so you can eyeball the detections.

Usage:
    python -m twm.calibration.test_detection --frames 120 --expect 8
"""

import argparse
import collections
import os

import cv2
import numpy as np

from twm.calibration.mocap_to_cam_multi import _Cam, adaptive_thresh
from twm.data_collection import REALSENSE_SERIALS


def main():
    ap = argparse.ArgumentParser(description="Headless marker-detection stability test.")
    ap.add_argument("--serials", nargs="*", default=REALSENSE_SERIALS)
    ap.add_argument("--source", choices=["ir", "color"], default="ir")
    ap.add_argument("--frames", type=int, default=120, help="Frames to sample.")
    ap.add_argument("--expect", type=int, default=8, help="Expected marker count.")
    ap.add_argument("--out_dir", type=str, default="/tmp/twm_detect_test",
                    help="Where to save annotated snapshots.")
    ap.add_argument("--thresh", type=int, default=None,
                    help="Override top-hat threshold (default: ir=25, color=adaptive).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Opening cameras (source={args.source})...")
    cams = [_Cam(i, s, source=args.source) for i, s in enumerate(args.serials)]

    # Warm up (auto-exposure settle), then seed threshold per camera.
    for _ in range(30):
        for c in cams:
            c.wait_frames()
    for c in cams:
        if args.thresh is not None:
            c.blob_thresh = args.thresh
        elif args.source == "ir":
            c.blob_thresh = 25
        else:
            c.blob_thresh = adaptive_thresh(c.det_gray)
        print(f"  cam{c.cam_idx} ({c.label}): thr={c.blob_thresh}  "
              f"contrast_max={int(c.tophat.max())}")

    counts = {c.cam_idx: collections.Counter() for c in cams}
    series = {c.cam_idx: [] for c in cams}
    last_frames = {}

    print(f"\nSampling {args.frames} frames...")
    for f in range(args.frames):
        for c in cams:
            c.wait_frames()
            n = len(c.mapped)   # markers actually placed on the RGB image
            counts[c.cam_idx][n] += 1
            series[c.cam_idx].append(n)
            last_frames[c.cam_idx] = (c.color.copy(), [m["color"] for m in c.mapped])

    print(f"\n{'='*68}")
    print(f"  Detection stability over {args.frames} frames (expect {args.expect})")
    print(f"{'='*68}")
    ok_all = True
    for c in cams:
        s = np.array(series[c.cam_idx])
        exact = int((s == args.expect).sum())
        atleast = int((s >= args.expect).sum())
        dist = dict(sorted(counts[c.cam_idx].items()))
        stable = exact / args.frames
        ok_all = ok_all and stable >= 0.95
        print(f"\n  cam{c.cam_idx} ({c.label}):")
        print(f"    count: mean={s.mean():.2f} std={s.std():.2f} "
              f"min={s.min()} max={s.max()}")
        print(f"    == {args.expect}: {exact}/{args.frames} ({100*stable:.1f}%)   "
              f">= {args.expect}: {atleast}/{args.frames} ({100*atleast/args.frames:.1f}%)")
        print(f"    count histogram (count: frames): {dist}")

        # Save annotated snapshot: markers drawn on the RGB image at their mapped
        # color pixels (this is what the calibration UI shows).
        img, pts = last_frames[c.cam_idx]
        for (cx, cy) in pts:
            cv2.circle(img, (int(round(cx)), int(round(cy))), 8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"cam{c.cam_idx} {c.label}: {len(pts)} markers on RGB (thr={c.blob_thresh})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        path = os.path.join(args.out_dir, f"cam{c.cam_idx}_{c.label}.png")
        cv2.imwrite(path, img)
        print(f"    snapshot: {path}")

    for c in cams:
        c.stop()

    print(f"\n{'='*68}")
    print(f"  VERDICT: {'STABLE ✅ (≥95% exact on all cams)' if ok_all else 'NOT fully stable ⚠️ — see per-cam numbers / snapshots'}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
