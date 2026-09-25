"""Adversarial tests for the auto-repair path — it must REFUSE, not build.

Automatic repair is a dangerous feature. Recovery gets back whatever HDF5 had
evicted from its metadata cache, which is reliably the pixels and rarely the
timestamps; a repair that quietly hands the pipeline a recording with no time
base produces an episode whose cross-modal alignment was invented. That is
strictly worse than the FAIL line it replaced.

So every check here reintroduces a defect and requires the gate to catch it.
A gate nobody has watched fail is not a gate.

    python scripts/test_repair_refuses.py
"""
from __future__ import annotations

import shutil
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                              # noqa: E402

from twm.react_preprocess import repair                         # noqa: E402
from twm.react_preprocess.config import H5_ROOTS                # noqa: E402
from twm.react_preprocess.h5io import discover                  # noqa: E402

SCRATCH = Path("/media/yxma/Disk1/twm")


def _make_episode(path: Path, frames: int = 4, *, timestamps: int | None = 4,
                  poses: int | None = 8) -> Path:
    """A miniature recording with the shape the pipeline expects."""
    import h5py
    import hdf5plugin

    opts = hdf5plugin.Blosc(cname="lz4", clevel=5,
                            shuffle=hdf5plugin.Blosc.SHUFFLE)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("metadata")
        g.attrs["task"] = "pushT"
        g.attrs["created_at"] = "2026-06-18T18:12:29"
        if timestamps is not None:
            f.create_dataset("timestamps",
                             data=np.arange(timestamps, dtype=np.float64))
        for name in repair.IMAGE_STREAMS:
            if name.endswith("depth"):
                shape, dtype = (frames, 8, 8), np.uint16
            else:
                shape, dtype = (frames, 8, 8, 3), np.uint8
            f.create_dataset(name, data=rng.integers(0, 255, shape, dtype=dtype),
                             chunks=(1,) + shape[1:], **opts)
        for body in ("sensor_left", "sensor_right"):
            n = poses if poses is not None else 0
            f.create_dataset(f"optitrack/{body}/timestamps",
                             data=np.arange(n, dtype=np.float64))
            f.create_dataset(f"optitrack/{body}/pose",
                             data=np.zeros((n, 7), np.float64))
    return path


def _break_like_a_kill(path: Path) -> Path:
    """Reproduce the recoverable signature exactly, on a file that opens.

    Zeroes the root object header and rewinds the superblock's EOF field and
    consistency flag to what they read at creation — which is what a recorder
    killed mid-run leaves behind.
    """
    with open(path, "r+b") as f:
        f.seek(20)
        f.write(struct.pack("<I", 1))          # consistency: open for write
        f.seek(40)
        f.write(struct.pack("<Q", 2048))       # EOF as at creation
        f.seek(96)
        f.write(b"\0" * 24)                    # root object header
    return path


def _break_unrecognisably(path: Path) -> Path:
    """Damage that is NOT the known signature — repair must not touch it."""
    with open(path, "r+b") as f:
        f.seek(96)
        f.write(b"\x07" * 24)                  # nonsense header, EOF intact
    return path


def main() -> int:
    problems: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="repairtest_", dir=str(SCRATCH)))
    try:
        # ── 1. a complete recording is eligible; each removal is refused ──
        good = _make_episode(tmp / "good.h5")
        ok, why = repair.release_eligibility(good)
        print(f"[1] complete recording      eligible={ok}  {why}")
        if not ok:
            problems.append(f"a complete recording was refused: {why}")

        cases = [
            ("no timestamps at all", dict(timestamps=None), ),
            ("timestamps shorter than the frames", dict(timestamps=2), ),
            ("no OptiTrack poses", dict(poses=0), ),
        ]
        for label, kw in cases:
            p = _make_episode(tmp / f"bad_{len(kw)}_{label[:6]}.h5", **kw)
            ok, why = repair.release_eligibility(p)
            print(f"    {label:36s} eligible={ok}  {why}")
            if ok:
                problems.append(f"a recording with {label} was allowed into "
                                f"the release")

        # ── 2. diagnose must separate the three states ──────────────────
        healthy = _make_episode(tmp / "healthy.h5")
        d = repair.diagnose(healthy)
        print(f"[2] healthy file            {d.signature} "
              f"repairable={d.repairable}")
        if d.repairable:
            problems.append("a healthy file was reported as repairable")

        killed = _break_like_a_kill(_make_episode(tmp / "killed.h5"))
        d = repair.diagnose(killed)
        print(f"    killed-mid-write        {d.signature} "
              f"repairable={d.repairable}")
        if d.signature != repair.METADATA_NEVER_FLUSHED or not d.repairable:
            problems.append(f"the recoverable signature was not recognised "
                            f"({d.signature})")

        weird = _break_unrecognisably(_make_episode(tmp / "weird.h5"))
        d = repair.diagnose(weird)
        print(f"    unfamiliar damage       {d.signature} "
              f"repairable={d.repairable}")
        if d.repairable:
            problems.append("damage that is not the known signature was "
                            "reported as repairable — recovery would guess")

        # ── 3. a recovered sibling must not publish as its own episode ──
        src = tmp / "episode_099.h5"
        _make_episode(src)
        shutil.copy(src, repair.recovered_path(src))
        found = discover("pushT", tmp)
        names = sorted(p.name for p in found)
        print(f"[3] discover() sees         {names}")
        if any(repair.is_recovered(p) for p in found):
            problems.append("discover() returned a .recovered.h5 — it would "
                            "publish as a separate episode")
        if repair.source_stem(repair.recovered_path(src)) != "episode_099":
            problems.append("source_stem let the .recovered suffix leak into "
                            "the published episode name")

        # ── 4. the real release must still contain exactly its episodes ──
        real = discover("pushT", H5_ROOTS["pushT"])
        print(f"[4] real pushT discover()   {len(real)} episodes: "
              f"{sorted(p.stem for p in real)}")
        if any(repair.is_recovered(p) for p in real):
            problems.append("the real pushT tree now yields a recovered file "
                            "as an episode")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for p in problems:
        print(f"  FAIL: {p}")
    print(f"repair gate: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
