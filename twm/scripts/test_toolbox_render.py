"""The toolbox can draw its own projection, so a user can see it land.

A number is a weak debugging aid for a geometry problem. `project_gel_to_pixel`
returning (366, 188) tells you nothing about whether 366, 188 is on the sensor
— which is the only question that matters, and the one that would have caught
every projection defect this project has shipped: the wrong calibration epoch
(35-73 px), the gel centre defaulting to the rigid-body origin (21-36 px), the
missing world offset (155-223 px). All three are obvious in a picture and
invisible in a coordinate.

So the toolbox draws. The checks are about provenance, not prettiness:

  1  the marker is at `project_gel_to_pixel`'s own answer — ONE law. A second
     projection inside the renderer would drift from the first, and the drift
     would read as a calibration error rather than a drawing bug.
  2  a point behind the camera draws nothing, rather than a plausible marker
     at a wrapped-around coordinate.
  3  zero force draws no disc, so an unpressed sensor cannot look pressed.

    python scripts/test_toolbox_render.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

from react_toolbox.staging import staging_dir

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import shutil, tempfile
    import react_toolbox.calibration as C

    if not hasattr(C, "draw_projection") and not _has_render():
        check(False, "the toolbox can draw a projection",
              "no draw_projection in react_toolbox.viz or .calibration")
        _report()
        return 1
    from react_toolbox.viz import draw_projection

    from twm.calib_epoch import calib_dir
    stage = staging_dir()
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = C.load_calibration(stage)
    cam = cal["cams"]["middle"]
    gel = cal["gel_left"]
    pose = np.array([0.40, 0.02, 0.30, 0.0, 0.0, 0.0, 1.0])

    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_projection(frame, pose, gel, cam)
    check(out is not None and out.shape == frame.shape and out is not frame,
          "the toolbox can draw a projection",
          f"returned {None if out is None else out.shape}, "
          f"input untouched: {not np.array_equal(out, frame)}")

    # 1 — the marker must sit at the library's own projection
    uv = C.project_gel_to_pixel(pose, gel, cam)
    d = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2)
    ys, xs = np.nonzero(d > 20)
    if uv is None or len(xs) == 0:
        check(False, "the marker is at project_gel_to_pixel's answer",
              f"projection {'is None' if uv is None else 'drew nothing'}")
    else:
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        err = float(np.hypot(cx - uv[0], cy - uv[1]))
        check(err <= 6.0, "the marker is at project_gel_to_pixel's answer",
              f"ink centroid ({cx:.1f}, {cy:.1f}) vs projection "
              f"({uv[0]:.1f}, {uv[1]:.1f}), error {err:.1f} px")

    # 2 — behind the camera: nothing, not a wrapped coordinate
    behind = pose.copy()
    behind[:3] = [0.0, 0.0, -5.0]
    ob = draw_projection(frame, behind, gel, cam)
    check(np.array_equal(ob, frame), "a point behind the camera draws nothing",
          f"{int((np.abs(ob.astype(int)-frame.astype(int))).sum())} units of ink")

    # 3 — zero force draws no disc
    a = draw_projection(frame, pose, gel, cam, force_n=0.0)
    b = draw_projection(frame, pose, gel, cam)
    check(np.array_equal(a, b), "zero force draws no disc",
          f"force=0 differs from force=None by "
          f"{int(np.abs(a.astype(int)-b.astype(int)).sum())} units")

    _report()
    return 1 if sum(not ok for ok, _, _ in RESULTS) else 0


def _has_render() -> bool:
    try:
        from react_toolbox.viz import draw_projection  # noqa: F401
        return True
    except Exception:
        return False


def _report() -> None:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    print(f"\ntoolbox render: {len(RESULTS)} checks, "
          f"{sum(not ok for ok, _, _ in RESULTS)} failing")


if __name__ == "__main__":
    raise SystemExit(main())
