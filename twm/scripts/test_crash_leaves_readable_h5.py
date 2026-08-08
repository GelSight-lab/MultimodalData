"""Prove the recorder's periodic flush survives a kill — by killing it.

A guard nobody has watched fail is not a guard. This kills a writer mid-
recording twice, once with HDF5Writer.FLUSH_INTERVAL_S left alone and once
with it set beyond the run's lifetime, and requires the two outcomes to
differ: flushed must open and hold frames, unflushed must not open at all.

The unflushed case is what pushT/2026-06-18/episode_004.h5 is — 79 GB whose
root object header is 24 zero bytes.

    python scripts/test_crash_leaves_readable_h5.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r'''
import os, sys, time
import numpy as np
sys.path.insert(0, {repo!r})
import h5py, hdf5plugin
from twm.data_collection import HDF5Writer, append_camera_frames_batch

path, interval = sys.argv[1], float(sys.argv[2])
HDF5Writer.FLUSH_INTERVAL_S = interval

f = h5py.File(path, "w")
opts = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)
for i in range(3):
    g = f.create_group(f"realsense/cam{{i}}")
    g.create_dataset("color", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                     chunks=(1, 480, 640, 3), dtype=np.uint8, **opts)
    g.create_dataset("depth", shape=(0, 480, 640), maxshape=(None, 480, 640),
                     chunks=(1, 480, 640), dtype=np.uint16, **opts)
for name in ("left", "right"):
    g = f.create_group(f"gelsight/{{name}}")
    g.create_dataset("frames", shape=(0, 480, 640, 3), maxshape=(None, 480, 640, 3),
                     chunks=(1, 480, 640, 3), dtype=np.uint8, **opts)
    g.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.float64)

w = HDF5Writer(batch_size=4)
rng = np.random.default_rng(0)
colors = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
depths = [rng.integers(0, 4000, (480, 640), dtype=np.uint16) for _ in range(3)]
gels   = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]

t0 = time.time()
n = 0
while time.time() - t0 < 4.0:
    w.enqueue(f, colors, depths, gels, time.time(), [time.time()] * 2)
    n += 1
    time.sleep(0.05)
w.flush()
print(f"child: enqueued {{n}} frames, flushes={{w._flushes}}", flush=True)
os._exit(9)          # the point of the test: no close(), no atexit, no cleanup
'''


def run_case(repo: Path, path: Path, interval: float) -> tuple[bool, int, str]:
    src = CHILD.format(repo=str(repo))
    r = subprocess.run([sys.executable, "-c", src, str(path), str(interval)],
                       capture_output=True, text=True)
    note = (r.stdout + r.stderr).strip().splitlines()
    note = note[-1] if note else ""
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        with h5py.File(path, "r") as f:
            return True, int(f["realsense/cam0/color"].shape[0]), note
    except Exception as e:
        return False, 0, f"{note} | open failed: {str(e).splitlines()[0]}"


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    # NOT /tmp: on this rig that is the root filesystem and it is full, so
    # both cases died with ENOSPC and "neither file opened" looked like a
    # pass for the unflushed case. A test that can fail for a reason unrelated
    # to what it measures has to say so — hence the ENOSPC surfaced in `note`.
    scratch = Path(os.environ.get("CRASHTEST_DIR", "/media/yxma/Disk1/twm"))
    tmp = Path(tempfile.mkdtemp(prefix="crashtest_", dir=str(scratch)))
    problems = []
    try:
        print("[1/2] killed WITHOUT a flush having happened "
              "(FLUSH_INTERVAL_S longer than the run)")
        ok_no, n_no, note_no = run_case(repo, tmp / "unflushed.h5", 3600.0)
        print(f"      opens={ok_no} frames={n_no}  {note_no}")

        print("[2/2] killed WITH periodic flush (FLUSH_INTERVAL_S = 1s)")
        ok_yes, n_yes, note_yes = run_case(repo, tmp / "flushed.h5", 1.0)
        print(f"      opens={ok_yes} frames={n_yes}  {note_yes}")

        if ok_no:
            problems.append(
                "the UNFLUSHED file opened — this test can no longer tell "
                "whether the flush is doing anything, so it is not evidence "
                "for anything. Find out what changed before trusting it.")
        if not ok_yes:
            problems.append("the FLUSHED file did not open — the flush does "
                            "not survive a kill")
        elif n_yes == 0:
            problems.append("the FLUSHED file opened but holds 0 frames")
    finally:
        for q in tmp.glob("*"):
            q.unlink()
        if tmp.exists():
            tmp.rmdir()

    for p in problems:
        print(f"  FAIL: {p}")
    print(f"crash test: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
