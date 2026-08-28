"""The up-axis conversion moves poses and cameras as ONE piece.

A world-frame rotation applied to the poses but not to `T_mocap_to_cam` — or
with the inverse the wrong way round — moves every projection and raises
nothing. The numbers stay plausible. So the test is INVARIANCE: convert both,
and every projected pixel must be unchanged.

It also pins the direction. Two rotations take Y-up to Z-up with det +1; the
wrong one leaves the world upside down and no handedness check notices.

    python scripts/test_frames.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import force_meta   # noqa: E402

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from react_toolbox.calibration import load_calibration, project_gel_to_pixel
    from react_toolbox.frames import (UP_AXIS_RECORDED, YUP_TO_ZUP,
                                      convert_calibration, convert_poses, to_zup)
    from scipy.spatial.transform import Rotation
    from twm.calib_epoch import calib_dir

    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = load_calibration(stage)
    p = sorted(force_meta("motherboard").glob("*/*.parquet"))[0]
    t = pq.read_table(p, columns=["sensor_left_pose"]).to_pydict()
    P = np.asarray([x for x in t["sensor_left_pose"]], float)
    P = P[np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)][:400]

    # 1 — it is a rotation, and it sends UP to +z rather than -z
    det = float(np.linalg.det(YUP_TO_ZUP))
    up = YUP_TO_ZUP @ np.array([0.053, 0.997, 0.056])
    check(abs(det - 1) < 1e-12 and up[2] > 0.99,
          "the conversion is right-handed and sends up to +z",
          f"det {det:+.0f}; the measured table normal maps to "
          f"{np.round(up, 3).tolist()}")

    # 2 — THE INVARIANT. Convert BOTH and every pixel must be unchanged.
    zP, zcal = to_zup(P, cal)
    worst = 0.0
    n = 0
    for v in ("left", "middle", "right"):
        for a, b in zip(P[::13], zP[::13]):
            ua = project_gel_to_pixel(a, cal["gel_left"], cal["cams"][v])
            ub = project_gel_to_pixel(b, zcal["gel_left"], zcal["cams"][v])
            if ua is None or ub is None:
                continue
            n += 1
            worst = max(worst, float(np.hypot(ua[0]-ub[0], ua[1]-ub[1])))
    check(n > 50 and worst < 1e-9,
          "converting poses AND cameras leaves every projection identical",
          f"{n} projections across 3 views, worst movement {worst:.2e} px")

    # 3 — AND HALF-APPLYING IT BREAKS THINGS. If this passes silently, the
    #     invariance above proves nothing.
    half = 0.0
    for a, b in zip(P[::13], zP[::13]):
        ua = project_gel_to_pixel(a, cal["gel_left"], cal["cams"]["middle"])
        ub = project_gel_to_pixel(b, cal["gel_left"], cal["cams"]["middle"])  # OLD cal
        if ua is None or ub is None:
            continue
        half = max(half, float(np.hypot(ua[0]-ub[0], ua[1]-ub[1])))
    check(half > 50.0,
          "converting the poses alone moves the projection a lot",
          f"poses converted, calibration left alone: up to {half:.0f} px — "
          f"which is why the two are converted together or not at all")

    # 4 — round trip
    back = convert_poses(zP, to_zup=False)
    dp = float(np.abs(back[:, :3] - P[:, :3]).max())
    da = float(np.degrees((Rotation.from_quat(back[:, 3:7]).inv()
                           * Rotation.from_quat(P[:, 3:7])).magnitude()).max())
    check(dp < 1e-12 and da < 1e-9, "the conversion round-trips",
          f"worst {dp:.2e} m and {da:.2e} deg over {len(P)} poses")

    # 5 — the recorded convention is stated, and it is the one in the data
    up_axis = int(np.argmax(np.abs(np.array([0.053, 0.997, 0.056]))))
    check(UP_AXIS_RECORDED == "xyz"[up_axis],
          "the module declares the convention the data actually uses",
          f"UP_AXIS_RECORDED={UP_AXIS_RECORDED!r}; the measured table normal "
          f"is closest to +{'xyz'[up_axis]}")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    nf = sum(not x for x, _, _ in RESULTS)
    print(f"\nframes: {len(RESULTS)} checks, {nf} failing")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
