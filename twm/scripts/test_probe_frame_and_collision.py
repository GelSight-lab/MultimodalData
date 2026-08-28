"""One sensor moves, the other stays, they never touch, and depth is visible.

Four properties the first version of the probe set did not have:

  1  THE WHOLE SENSOR FRAME IS DRAWN, not a dot. A dot shows position and
     hides orientation, which makes the six ROTATION probes unreadable — they
     move the marker barely at all. The React previews draw a triad for this
     reason and so does this.

  2  PERSPECTIVE IS REAL. The axis tips are placed in 3D at a fixed
     millimetre length and then projected, so a sensor nearer the camera draws
     a larger triad. Drawing a fixed PIXEL length would assert that two
     sensors at different depths are the same size, which is the one thing a
     projection is supposed to tell you.

  3  ONE SENSOR MOVES PER PROBE, and which one is random. Every probe in the
     first version moved the left sensor, so half the rig was never exercised
     and any left-specific defect would have been invisible.

  4  THEY DO NOT COLLIDE. Each gel carries a 0.12 m-diameter exclusion circle,
     so the two centres must stay at least 0.12 m apart for every step. In the
     real data the centres sit 0.225 m apart at the median and closer than
     0.12 m on 1.67% of frames, so this rejects a little and forbids the
     physically impossible.

    python scripts/test_probe_frame_and_collision.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import force_meta   # noqa: E402

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
COLLISION_M = 0.12


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import pyarrow.parquet as pq

    import react_toolbox as T
    from react_toolbox.synth_actions import COLLISION_DIAMETER_M, sample_probe
    from react_toolbox.frames import as_up_axis
    from twm.calib_epoch import calib_dir

    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = as_up_axis(T.load_calibration(stage), "z")  # release poses
    cam = cal["cams"]["middle"]

    check(abs(COLLISION_DIAMETER_M - COLLISION_M) < 1e-9,
          "the collision diameter is 0.12 m",
          f"COLLISION_DIAMETER_M = {COLLISION_DIAMETER_M}")

    # 1 / 2 — the frame projects with real perspective
    if not hasattr(T, "project_gel_frame"):
        check(False, "the whole sensor frame is projected", "no project_gel_frame")
        check(False, "a nearer sensor draws a bigger triad", "not attempted")
    else:
        # DEPTH IS MEASURED, NOT ASSUMED. My first version displaced along
        # world +z and called the result "further from the camera"; world +z
        # is not the camera's view direction here, so the "far" pose came out
        # NEARER and the check failed on its own premise. `project_gel_frame`
        # returns the camera-frame depth, so the two poses are ordered by what
        # it reports rather than by which way I guessed the axis points.
        base = np.array([0.40, 0.02, 0.20, 0.0, 0.0, 0.0, 1.0])
        cands = []
        for k in range(3):
            for sgn in (+1, -1):
                q = base.copy()
                q[k] += sgn * 0.35
                r = T.project_gel_frame(q, cal["gel_left"], cam, axis_len_mm=60.0)
                if r is not None:
                    cands.append((r["depth_mm"], q))
        cands.sort()
        near, far = cands[0][1], cands[-1][1]
        depths = (cands[0][0], cands[-1][0])
        spans = []
        for p in (near, far):
            r = T.project_gel_frame(p, cal["gel_left"], cam, axis_len_mm=60.0)
            if r is None:
                spans.append(None)
                continue
            c, tips = r["centre"], [t for t in r["tips"] if t is not None]
            spans.append(max(float(np.hypot(t[0] - c[0], t[1] - c[1]))
                             for t in tips) if tips else 0.0)
        ok = all(s is not None for s in spans)
        check(ok and len(spans[0:1]) == 1,
              "the whole sensor frame is projected",
              f"centre + {len(tips)} axis tips")
        check(ok and spans[0] > spans[1] * 1.2,
              "a nearer sensor draws a bigger triad",
              f"depth {depths[0]:.0f} mm -> span {spans[0]:.1f} px, "
              f"depth {depths[1]:.0f} mm -> span {spans[1]:.1f} px"
              if ok else "projection failed")

    # 3 / 4 — one sensor moves, chosen at random, and no collision
    P = (str(force_meta("motherboard"))+"/"
         "2026-05-11/episode_003.parquet")
    t = pq.read_table(P, columns=["sensor_left_pose", "sensor_right_pose"]).to_pydict()
    poses = {s: np.asarray([x for x in t[f"sensor_{s}_pose"]], float)
             for s in ("left", "right")}

    moving = set()
    viol = []
    for seed in range(12):
        try:
            r = sample_probe(poses, cal, seed=seed, view="middle")
        except (TypeError, ValueError) as exc:
            check(False, "one sensor moves per probe, chosen at random",
                  f"sample_probe: {type(exc).__name__}: {str(exc)[:70]}")
            check(False, "the two sensors never come within 0.12 m",
                  "not attempted")
            _report()
            return 1
        moving.add(r["moving_side"])
        for p in r["probes"]:
            d = p.get("min_separation_m")
            if d is not None and d < COLLISION_M - 1e-9:
                viol.append(f"{r['moving_side']}/{p['name']}: {d:.3f} m")
    check(moving == {"left", "right"},
          "one sensor moves per probe, chosen at random",
          f"over 12 seeds the moving side was {sorted(moving)}")
    check(not viol, "the two sensors never come within 0.12 m",
          f"{len(viol)} violating probe(s)" + (f"; {viol[:2]}" if viol else ""))

    _report()
    return 1 if sum(not ok for ok, _, _ in RESULTS) else 0


def _report() -> None:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    print(f"\nprobe frame + collision: {len(RESULTS)} checks, "
          f"{sum(not ok for ok, _, _ in RESULTS)} failing")


if __name__ == "__main__":
    raise SystemExit(main())
