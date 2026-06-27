"""Measure GelSight (UVC) capture latency on the rig — quantifies the V4L2
buffer backlog that caused the ~15-frame tactile lag.

Method: after the camera has streamed for a moment (so its buffer fills),
time consecutive grab() calls. Buffered frames return *instantly*; the first
grab that has to WAIT for the camera marks the end of the backlog. The count
of instant grabs == how many frames stale read() would have been.

Run on the recording PC with a GelSight connected:
    python camera_stream/measure_gelsight_latency.py --serial 2BKRDTAD
    python camera_stream/measure_gelsight_latency.py --serial 2BKRDTAD --fixed

--fixed applies the same settings as the patched USBVideoStream
(V4L2 + MJPG + BUFFERSIZE=1) so you can compare before/after.
"""
import argparse
import time

import cv2
import pyudev


def find_video_id(serial):
    ctx = pyudev.Context()
    ids = []
    for d in ctx.list_devices(subsystem="video4linux"):
        if serial in (d.get("ID_SERIAL") or ""):
            ids.append(int(d.sys_number))
    if not ids:
        raise RuntimeError(f"no video device for serial {serial}")
    return sorted(ids)[0]


def backlog_frames(cap, settle_s=2.0, gap_ms=12.0):
    """Let the buffer fill for settle_s (no reads), then count how many grab()
    calls return faster than gap_ms (== buffered) before one blocks for a
    fresh camera frame. That count is the read() staleness in frames."""
    # prime: ensure streaming, then stop reading to let the queue fill
    cap.grab()
    time.sleep(settle_s)
    n = 0
    t_prev = time.time()
    while True:
        cap.grab()
        now = time.time()
        dt = (now - t_prev) * 1000.0
        t_prev = now
        if dt > gap_ms:          # this grab waited for the camera => backlog drained
            break
        n += 1
        if n > 120:              # safety
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--fixed", action="store_true",
                    help="Apply patched settings (V4L2+MJPG+BUFFERSIZE=1).")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    vid = find_video_id(args.serial)
    print(f"serial {args.serial} -> /dev/video{vid}  | mode = "
          f"{'FIXED (V4L2+MJPG+buffersize1)' if args.fixed else 'OLD (default)'}")

    if args.fixed:
        cap = cv2.VideoCapture(vid, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(vid)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_s = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"  negotiated FOURCC={fourcc_s}  size={int(cap.get(3))}x{int(cap.get(4))}  "
          f"buffersize={cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

    runs = [backlog_frames(cap) for _ in range(3)]
    print(f"  backlog (frames stale) over 3 trials: {runs}")
    avg = sum(runs) / len(runs)
    print(f"  => mean staleness ≈ {avg:.1f} frames = {avg/args.fps*1000:.0f} ms @ {args.fps:.0f}fps")
    cap.release()


if __name__ == "__main__":
    main()
