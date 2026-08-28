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

from react_paths import force_meta, release_root   # noqa: E402

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from react_toolbox.calibration import load_calibration, project_gel_to_pixel
    from react_toolbox.frames import (UP_AXIS_RECORDED, YUP_TO_ZUP, ZUP_TO_YUP,
                                      convert_calibration, convert_poses, to_zup)
    from react_toolbox.frames import require_up_axis
    from scipy.spatial.transform import Rotation
    from twm.calib_epoch import calib_dir

    # The calibration must come from the SAME tree as the poses below. It used
    # to come from calib_dir(), a separate tree that REACT_CALIB can redirect;
    # after the release was rotated to Z-up and that tree was not, this file
    # was silently testing a mismatched pair and still printing all-green.
    cal = load_calibration(release_root("motherboard"))
    require_up_axis(cal, where="the release")
    p = sorted(force_meta("motherboard").glob("*/*.parquet"))[0]
    t = pq.read_table(p, columns=["sensor_left_pose"]).to_pydict()
    P = np.asarray([x for x in t["sensor_left_pose"]], float)
    P = P[np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)][:400]

    # 0 — MEASURE the up axis from the data, do not restate it. This check
    #     used a literal normal from an earlier run and kept asserting the old
    #     convention after the release was converted — a constant that had
    #     stopped describing anything.
    tt = pq.read_table(p).to_pydict()
    O = np.asarray([x for x in tt["object_pose"]], float)
    S = np.asarray([x for x in tt["sensor_left_pose"]], float)
    F = np.asarray(tt.get("force_left_normal_n", np.zeros(len(S))), float)
    ok_ = ((F > 2) & np.isfinite(S).all(1) & np.isfinite(O).all(1)
           & (np.linalg.norm(O[:, 3:], axis=1) > .5))
    Ro = Rotation.from_quat(O[ok_, 3:7]).as_matrix()
    Rs = Rotation.from_quat(S[ok_, 3:7]).as_matrix()
    g = S[ok_, :3]*1000 + np.einsum("nij,j->ni", Rs, cal["gel_left"])
    Cb = np.einsum("nji,nj->ni", Ro, g - O[ok_, :3]*1000)
    nl = np.linalg.svd(Cb - Cb.mean(0))[2][2]
    nw = np.einsum("nij,j->ni", Ro, nl)
    k = int(np.argmax(np.abs(np.median(nw, axis=0))))
    nw *= np.sign(np.median(nw[:, k]))
    normal = np.median(nw, axis=0); normal /= np.linalg.norm(normal)
    measured = "xyz"[int(np.argmax(np.abs(normal)))]

    # 1 — the conversion is a rotation, and it sends UP to +z rather than -z
    det = float(np.linalg.det(YUP_TO_ZUP))
    yup_normal = normal if measured == "y" else ZUP_TO_YUP @ normal
    up = YUP_TO_ZUP @ yup_normal
    check(abs(det - 1) < 1e-12 and up[2] > 0.99,
          "the conversion is right-handed and sends up to +z",
          f"det {det:+.0f}; the table normal expressed Y-up "
          f"{np.round(yup_normal, 3).tolist()} maps to {np.round(up, 3).tolist()}")

    # 2 — THE INVARIANT. The release is a matched Z-up pair; rotate it back to
    #     the recorded Y-up, forward again with to_zup(), and every pixel must
    #     land where the release puts it.
    yP = convert_poses(P, to_zup=False)
    ycal = convert_calibration(cal, to_zup=False)
    zP, zcal = to_zup(yP, ycal)
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
    # This is exactly the bug it exists to catch: Z-up poses, Y-up calibration.
    half = 0.0
    for a in P[::13]:
        ua = project_gel_to_pixel(a, cal["gel_left"], cal["cams"]["middle"])
        ub = project_gel_to_pixel(a, ycal["gel_left"], ycal["cams"]["middle"])
        if ua is None or ub is None:
            continue
        half = max(half, float(np.hypot(ua[0]-ub[0], ua[1]-ub[1])))
    check(half > 50.0,
          "converting the poses alone moves the projection a lot",
          f"poses converted, calibration left alone: up to {half:.0f} px — "
          f"which is why the two are converted together or not at all")

    # 4 — round trip
    back = convert_poses(zP, to_zup=False)
    dp = float(np.abs(back[:, :3] - yP[:, :3]).max())
    da = float(np.degrees((Rotation.from_quat(back[:, 3:7]).inv()
                           * Rotation.from_quat(yP[:, 3:7])).magnitude()).max())
    check(dp < 1e-12 and da < 1e-9, "the conversion round-trips",
          f"worst {dp:.2e} m and {da:.2e} deg over {len(P)} poses")

    # 5 — the declared convention is the one the data actually has, measured
    check(UP_AXIS_RECORDED == measured and normal["xyz".index(measured)] > 0,
          "the declared convention is the one the data actually has",
          f"UP_AXIS_RECORDED={UP_AXIS_RECORDED!r}; measured table normal "
          f"{np.round(normal, 3).tolist()} -> +{measured}, "
          f"{np.degrees(np.arccos(abs(normal['xyz'.index(measured)]))):.1f} deg off")

    # 6 — the gizmo has to FIT. Its whole job is to be readable, and the axis
    # that points straight up is the one whose label runs off the top edge.
    from react_toolbox.viz import draw_world_gizmo
    frame = np.zeros((480, 640, 3), np.uint8)
    bad = []
    for name, c in cal["cams"].items():
        R = np.asarray(c["T_mocap_to_cam"], float)[:3, :3]
        for i, ax in enumerate("xyz"):
            d = R @ np.eye(3)[i]
            if float(np.hypot(d[0], d[1])) <= 0.12:
                continue                      # drawn as a dot at the origin
            ox = oy = 12 + 44 + 22            # draw_world_gizmo's tl origin
            lx = ox + float(d[0]) * (44 + 13)
            ly = oy + float(d[1]) * (44 + 13)
            # the label glyph reaches ~8 px above its anchor and ~6 below
            if not (10 <= lx <= 630 and 10 <= ly <= 474):
                bad.append(f"{name}.{ax} label at ({lx:.0f},{ly:.0f})")
    _ = draw_world_gizmo(frame, cal["cams"]["middle"], corner="tl",
                         title="world (z-up)")
    check(not bad, "every gizmo axis label lands inside the frame",
          "all 3 cameras, all in-plane axes fit"
          if not bad else "clipped: " + "; ".join(bad))

    # 7 — a calibration can be ASKED for a convention. Twelve scripts paired a
    #     Z-up pose source with a Y-up calibration because each one picked a
    #     directory and hoped. Asking removes the hope.
    from react_toolbox.frames import as_up_axis
    ycal2 = as_up_axis(cal, "y")
    zcal2 = as_up_axis(ycal2, "z")
    idem = as_up_axis(cal, "z")
    dz = max(float(np.abs(np.asarray(zcal2["cams"][v]["T_mocap_to_cam"]) -
                          np.asarray(cal["cams"][v]["T_mocap_to_cam"])).max())
             for v in cal["cams"])
    di = max(float(np.abs(np.asarray(idem["cams"][v]["T_mocap_to_cam"]) -
                          np.asarray(cal["cams"][v]["T_mocap_to_cam"])).max())
             for v in cal["cams"])
    dy = max(float(np.abs(np.asarray(ycal2["cams"][v]["T_mocap_to_cam"]) -
                          np.asarray(ycal["cams"][v]["T_mocap_to_cam"])).max())
             for v in cal["cams"])
    check(dz < 1e-12 and di < 1e-12 and dy < 1e-12
          and ycal2["up_axis"] == "y" and zcal2["up_axis"] == "z",
          "as_up_axis converts on demand and is a no-op when it already fits",
          f"z->y->z {dz:.1e}, already-z {di:.1e}, matches convert_calibration "
          f"{dy:.1e}; declarations {ycal2['up_axis']}/{zcal2['up_axis']}")

    # 8 — the raw-H5 offset. episodes.jsonl now stores it Z-up, but its whole
    #     documented purpose is to be ADDED to a pose read out of the source
    #     H5, which is Y-up. Handing the stored value straight to a raw
    #     consumer puts 175 mm on the wrong axis, twice.
    from twm.calib_epoch import world_offset_m
    oz = world_offset_m("motherboard", "2026-05-19", "episode_002", up_axis="z")
    oy = world_offset_m("motherboard", "2026-05-19", "episode_002", up_axis="y")
    check(np.allclose(oz, (0.23, -0.175, 0.0), atol=1e-9)
          and np.allclose(oy, (0.23, 0.0, 0.175), atol=1e-9),
          "world_offset_m answers in the convention the caller asks for",
          f"z-up {tuple(round(x, 4) for x in oz)}, "
          f"y-up {tuple(round(x, 4) for x in oy)}")

    # 9 — the raw-HDF5 viewers must not be handed the converted release.
    #     calib_dir() resolves $REACT_RELEASE BEFORE the repo's own Y-up tree,
    #     and that directory is now Z-up. Every interactive viewer reads Y-up
    #     poses straight out of the H5, so the moment a user exports
    #     REACT_RELEASE -- which react_paths documents as the normal way to
    #     point at the data -- they would silently view through a 200 px error.
    import os as _os
    from twm.calib_epoch import calib_dir as _cd
    _old = _os.environ.get("REACT_RELEASE")
    _os.environ["REACT_RELEASE"] = str(release_root())
    try:
        try:
            _cd("motherboard", up_axis="y")
            raised = ""
        except Exception as ex:
            raised = f"{type(ex).__name__}: {str(ex)[:60]}"
        got_z = _cd("motherboard", up_axis="z")
    finally:
        if _old is None:
            _os.environ.pop("REACT_RELEASE", None)
        else:
            _os.environ["REACT_RELEASE"] = _old
    check(raised.startswith("ValueError") and got_z.is_dir(),
          "a Y-up caller is refused the Z-up release calibration",
          f"asking for y-up raised [{raised}]; asking for z-up returned "
          f"{got_z.name}/")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    nf = sum(not x for x, _, _ in RESULTS)
    print(f"\nframes: {len(RESULTS)} checks, {nf} failing")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
