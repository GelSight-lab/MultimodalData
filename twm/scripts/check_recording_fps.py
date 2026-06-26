#!/usr/bin/env python3
"""Verify that a recorded episode actually hit 30 fps.

Reads the per-tick `timestamps` array from an H5 episode and reports the
mean, std, min, and max inter-frame interval. Recording is "healthy" when
mean ≈ 1/fps and the std is well below the inter-frame interval.

Usage:
    python -m twm.scripts.check_recording_fps <episode.h5>            # one file
    python -m twm.scripts.check_recording_fps <dir>                   # latest in dir
    python -m twm.scripts.check_recording_fps <ep.h5> --target-fps 30

Exit code 0 if recording is healthy (mean within 5% of target, std < 5ms),
1 otherwise — handy for CI / scripted post-checks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def report(path: Path, target_fps: float) -> bool:
    with h5py.File(str(path), "r") as f:
        ts = f["timestamps"][:]
        recorded_fps = float(f["metadata"].attrs.get("fps", target_fps))
    if ts.size < 2:
        print(f"{path}: only {ts.size} frame(s) — nothing to check.")
        return False

    target = target_fps if target_fps else recorded_fps
    dt = np.diff(ts)
    mean_dt = float(dt.mean())
    std_dt  = float(dt.std())
    min_dt  = float(dt.min())
    max_dt  = float(dt.max())
    measured_fps = 1.0 / mean_dt if mean_dt > 0 else 0.0
    expected_dt  = 1.0 / target if target > 0 else float("nan")
    drift_pct    = 100.0 * (mean_dt - expected_dt) / expected_dt if expected_dt > 0 else 0.0

    healthy = (abs(mean_dt - expected_dt) / expected_dt < 0.05) and (std_dt < 0.005)

    print(f"  file:      {path}")
    print(f"  frames:    {ts.size}")
    print(f"  duration:  {ts[-1] - ts[0]:.2f} s")
    print(f"  target:    {target:.1f} fps  ({expected_dt*1000:.1f} ms/tick)")
    print(f"  measured:  {measured_fps:.1f} fps  (mean dt = {mean_dt*1000:.1f} ms)")
    print(f"  std dt:    {std_dt*1000:.2f} ms")
    print(f"  min/max:   {min_dt*1000:.1f} / {max_dt*1000:.1f} ms")
    print(f"  drift:     {drift_pct:+.1f}%")
    print(f"  HEALTHY:   {'YES' if healthy else 'NO'}"
          + ("" if healthy else "  (need mean within 5% of target AND std < 5ms)"))
    return healthy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="H5 file OR directory (uses latest *.h5 within).")
    ap.add_argument("--target-fps", type=float, default=30.0,
                    help="Expected capture FPS (default 30; overridden by H5 metadata.fps if present).")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        h5s = sorted(p.glob("**/*.h5"), key=lambda x: x.stat().st_mtime)
        if not h5s:
            print(f"No .h5 files under {p}", file=sys.stderr); sys.exit(2)
        p = h5s[-1]
        print(f"(latest episode in dir: {p.name})")
    elif not p.is_file():
        print(f"Not found: {p}", file=sys.stderr); sys.exit(2)

    ok = report(p, args.target_fps)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
