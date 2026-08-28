"""Every episode says which world frame its poses are in, and it is checkable.

The 2026-05-19 session redefined the OptiTrack origin: its raw poses sit
(0.23, 0, 0.175) m from every other date. The release bakes the correction in,
so the published poses of all 32 motherboard episodes share one frame — but
that fact lives only in a free-text `note` inside `calibration.json`:

    "05-19 has a redefined OptiTrack world origin; offset (0.23,0,0.175)m
     baked into poses."

A pose array cannot say which frame it is in, and a consumer holding one
parquet file has nothing to check. I read that note's own offset backwards
earlier in this session — treating "release minus rawH5 = +0.23" as "the
release needs +0.23" — which is exactly the confusion a machine-readable
declaration removes.

Two layers, because a declaration can also be wrong:

  1  DECLARED. Each episode's parquet carries `twm.world_frame` and, for the
     dates that need it, the offset a RAW-H5 pose would need. One file, no
     lookup table, no date matching.

  2  CHECKABLE. It also carries a PROJECTION FINGERPRINT: the median pixel
     the gel centre projects to in each camera, computed at export time from
     the correct poses. Any consumer recomputes it from their own pose array;
     a mismatch means their frame is wrong, whatever the declaration says.

     The fingerprint catches any frame error, not only the known one.
     Measured on 05-19/episode_002, left sensor:

         missing world offset      159 - 223 px
         metres read as mm        1741 - 2782 px
         y/z axes swapped          185 - 227 px
         1 degree of yaw             1.7 - 3.2 px

     against a calibration rmse of 4.75 mm (about 3 px at this depth), so the
     tolerance below sits just above the noise the rig itself has.

    python scripts/test_world_frame_declared.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
TOL_PX = 6.0          # ~2x the rig's own reprojection noise


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import pyarrow.parquet as pq

    from twm.world_frame import (fingerprint, read_declaration,
                                 verify_fingerprint)

    # release_force/, not release/: the declaration is written by the
    # force exporter, and `release/` holds the pre-force parquet.
    rel = Path("/media/yxma/Disk1/twm/release_force/motherboard/meta")
    cases = [("2026-05-11", "episode_003"), ("2026-05-19", "episode_002")]

    # 1 — every episode declares its frame, in the file itself
    missing = []
    for date, ep in cases:
        d = read_declaration(rel / date / f"{ep}.parquet")
        if not d or "world_frame" not in d:
            missing.append(f"{date}/{ep}")
    check(not missing, "each episode declares its world frame",
          f"{len(cases) - len(missing)}/{len(cases)} declare it"
          + (f"; missing {missing}" if missing else ""))

    # 2 — and carries a fingerprint that the poses actually reproduce
    bad = []
    for date, ep in cases:
        p = rel / date / f"{ep}.parquet"
        d = read_declaration(p) or {}
        if "fingerprint" not in d:
            bad.append(f"{date}/{ep}: no fingerprint")
            continue
        t = pq.read_table(p).to_pydict()
        pose = np.asarray([x for x in t["sensor_left_pose"]], float)
        err = verify_fingerprint(pose, "left", "motherboard",
                                 d["fingerprint"]["left"], up_axis="z")
        if err > TOL_PX:
            bad.append(f"{date}/{ep}: {err:.1f} px")
    check(not bad, "the poses reproduce their own fingerprint",
          f"{len(cases) - len(bad)}/{len(cases)} within {TOL_PX} px"
          + (f"; {bad}" if bad else ""))

    # 3 — AND IT CATCHES THE REAL MISTAKE. Not "does it pass on good data" —
    # the offset applied backwards is the error that actually happened, twice.
    if not bad:
        date, ep = "2026-05-19", "episode_002"
        d = read_declaration(rel / date / f"{ep}.parquet")
        t = pq.read_table(rel / date / f"{ep}.parquet").to_pydict()
        pose = np.asarray([x for x in t["sensor_left_pose"]], float)
        raw = pose.copy()
        # the offset is declared in the RAW convention; these poses are the
        # release's, so rotate it before removing it -- otherwise this
        # negative control would "pass" for the wrong reason.
        from react_toolbox.frames import YUP_TO_ZUP as _M
        _o = np.asarray(d.get("raw_h5_offset_m") or [0, 0, 0], float)
        if (d.get("raw_h5_offset_up_axis") or "y") == "y":
            _o = np.asarray(_M, float) @ _o
        raw[:, :3] -= _o
        err = verify_fingerprint(raw, "left", "motherboard",
                                 d["fingerprint"]["left"], up_axis="z")
        check(err > 50.0, "a raw-H5 pose array is rejected",
              f"un-offset poses miss the fingerprint by {err:.1f} px "
              f"(tolerance {TOL_PX})")
    else:
        check(False, "a raw-H5 pose array is rejected", "not attempted")

    # 4 — the two published records of the same offset must agree. The
    #     parquet says what to add to a RAW H5 pose (Y-up); episodes.jsonl
    #     says where the release frame sits (Z-up). Rotate one into the other
    #     and they are the same vector, or one of them was half-converted.
    from react_toolbox.frames import YUP_TO_ZUP
    from twm.calib_epoch import world_offset_m
    dis = []
    for date, ep in cases:
        d = read_declaration(rel / date / f"{ep}.parquet") or {}
        off = d.get("raw_h5_offset_m")
        if off is None:
            continue
        ax = d.get("raw_h5_offset_up_axis") or "y"
        v = np.asarray(off, float)
        if ax == "y":
            v = np.asarray(YUP_TO_ZUP, float) @ v
        want = np.asarray(world_offset_m("motherboard", date, ep,
                                         up_axis="z"), float)
        if not np.allclose(v, want, atol=1e-9):
            dis.append(f"{date}/{ep}: parquet {ax}-up {off} -> "
                       f"{v.round(4).tolist()} vs episodes.jsonl "
                       f"{want.round(4).tolist()}")
    check(not dis, "both published records of the offset agree",
          f"{len(cases)} episodes; parquet raw_h5_offset_m rotates onto "
          f"episodes.jsonl world_frame_offset" if not dis else "; ".join(dis))

    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nworld frame declared: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
