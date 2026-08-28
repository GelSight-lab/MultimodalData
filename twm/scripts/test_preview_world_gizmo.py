"""The preview's world-frame gizmo: right panel, right convention.

Two things can be wrong here and neither is visible in a video thumbnail.

It can land in the wrong panel. The 1280x480 preview is four tiles wide;
the middle camera is the SECOND one, x in [320, 640), and a gizmo drawn at
the frame's top-left corner sits on the LEFT camera instead -- still looks
deliberate, still labelled "world".

It can name the wrong axes. The renderer works entirely in the recorded
Y-up frame: raw HDF5 poses, the Y-up world offset, the Y-up extrinsics.
Every number a reader downloads is Z-up. Drawing the frame the renderer
happens to hold would put a video on the Hub whose axis labels contradict
the parquet beside it -- the same half-converted pairing that cost this
release a 153 px overlay error.

So: measure where the arrows point, and check the label against the
calibration rather than against another label.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
MID_X0, MID_X1, MID_Y1 = 320, 640, 240


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import importlib.util as il
    from react_toolbox.frames import as_up_axis, UP_AXIS_RECORDED
    from react_toolbox.staging import staging_dir
    from react_toolbox.calibration import load_calibration
    from twm.calib_epoch import calib_dir
    import shutil

    sp = il.spec_from_file_location(
        "_bep", Path(__file__).resolve().parent / "build_episode_previews.py")
    bep = il.module_from_spec(sp)
    sp.loader.exec_module(bep)

    st = staging_dir()
    shutil.copytree(calib_dir("motherboard"), st / "calibration")
    cal = load_calibration(st)
    cams = [{"index": 2, "T_mocap_to_cam": cal["cams"]["middle"]["T_mocap_to_cam"],
             "intrinsics": cal["cams"]["middle"]["intrinsics"]}]

    panel = np.zeros((480, 1280, 3), np.uint8)
    out = bep.draw_world_gizmo_on_panel(panel.copy(), cams, cal.get("up_axis"))

    # 1 — every drawn pixel is inside the MIDDLE camera tile
    ink = np.argwhere(out.any(2))
    inside = (ink[:, 1] >= MID_X0) & (ink[:, 1] < MID_X1) & (ink[:, 0] < MID_Y1)
    check(len(ink) > 200 and inside.all(),
          "the gizmo is drawn inside the middle camera tile",
          f"{len(ink)} pixels, x [{ink[:,1].min()}, {ink[:,1].max()}], "
          f"y [{ink[:,0].min()}, {ink[:,0].max()}]; the tile is "
          f"x [{MID_X0}, {MID_X1}) y [0, {MID_Y1})")

    # 2 — it names the PUBLISHED convention, not the one the renderer holds.
    #     The middle camera looks down at the table, so the vertical axis
    #     must point nearly at it: in-plane ~0.03, not 1.00.
    z = as_up_axis(cal, "z")["cams"]["middle"]["T_mocap_to_cam"][:3, :3]
    y = as_up_axis(cal, "y")["cams"]["middle"]["T_mocap_to_cam"][:3, :3]
    dz = z @ np.array([0, 0, 1.0])
    dy = y @ np.array([0, 0, 1.0])
    check(float(np.hypot(dz[0], dz[1])) < 0.2
          and float(np.hypot(dy[0], dy[1])) > 0.8,
          "the two conventions are distinguishable at all",
          f"world +z in-plane: {np.hypot(dz[0],dz[1]):.3f} Z-up vs "
          f"{np.hypot(dy[0],dy[1]):.3f} Y-up — a check that could not tell "
          f"them apart would prove nothing")

    # 3 — the drawn arrows match the Z-up extrinsics, measured off the image
    #     rather than read back from the same matrix that drew them.
    ox, oy = bep.GIZMO_ORIGIN
    tips = {}
    for i, ax in enumerate("xyz"):
        d = z @ np.eye(3)[i]
        if float(np.hypot(d[0], d[1])) <= 0.12:
            continue
        tips[ax] = (ox + d[0] * bep.GIZMO_SIZE, oy + d[1] * bep.GIZMO_SIZE)
    bad = []
    for ax, (tx, ty) in tips.items():
        # the arrow must have put ink within a few pixels of its predicted tip
        win = out[max(0, int(ty) - 4):int(ty) + 5, max(0, int(tx) - 4):int(tx) + 5]
        if not win.any():
            bad.append(f"{ax} tip ({tx:.0f},{ty:.0f}) is blank")
    check(len(tips) >= 2 and not bad,
          "each arrow reaches where the Z-up extrinsics put it",
          f"{len(tips)} in-plane axes checked at the pixel level"
          if not bad else "; ".join(bad))

    # 4 — and the vertical axis is drawn as a dot, not a stub arrow
    d = z @ np.array([0, 0, 1.0])
    check(float(np.hypot(d[0], d[1])) <= 0.12 and d[2] < 0,
          "world +z reads as pointing at the camera",
          f"in-plane {np.hypot(d[0],d[1]):.3f}, depth {d[2]:+.3f} "
          f"(negative = toward the viewer, drawn as a dot)")

    # 5 — the gizmo's axis colours must be the SAME as the sensor triads'
    #     in the same frame. The preview panel is BGR; react_toolbox.viz's
    #     palette is written for RGB images, so drawing it straight put a
    #     blue x and an orange z beside sensor triads whose X is red. Nothing
    #     in the position checks above can see that, and neither can grep.
    from twm.viz import AXIS_BGR
    wrong = []
    for i, ax in enumerate("xyz"):
        d = z @ np.eye(3)[i]
        if float(np.hypot(d[0], d[1])) <= 0.12:
            continue
        # sample along the shaft, away from the tip and the origin
        px = int(round(ox + d[0] * bep.GIZMO_SIZE * 0.6))
        py = int(round(oy + d[1] * bep.GIZMO_SIZE * 0.6))
        patch = out[py - 2:py + 3, px - 2:px + 3].reshape(-1, 3)
        lit = patch[patch.sum(1) > 120]
        if not len(lit):
            wrong.append(f"{ax}: nothing drawn at ({px},{py})")
            continue
        got = lit.mean(0)
        if int(np.argmax(got)) != int(np.argmax(AXIS_BGR[i])):
            wrong.append(f"{ax}: drawn BGR {got.round(0).astype(int).tolist()}, "
                         f"the triads use {list(AXIS_BGR[i])}")
    check(not wrong,
          "the gizmo uses the same axis colours as the sensor triads",
          "x/y/z agree with twm.viz.AXIS_BGR in the BGR panel"
          if not wrong else "; ".join(wrong))

    # 6 — it disturbs nothing else. Checked on the PANEL ARRAY, before
    #     encoding: H.264 is lossy and inter-predicted, so adding anything
    #     anywhere shifts a few hundred pixels all over the decoded frame.
    #     Diffing the videos and demanding a tight bounding box asks the
    #     codec for a guarantee it does not make -- I tried, and read 8% of
    #     the frame "changed" on two clips that are identical apart from
    #     this gizmo.
    import cv2
    pub = Path("/media/yxma/Disk1/twm/release/motherboard/previews/"
               "2026-05-10/episode_000.mp4")
    if not pub.exists():
        check(True, "the gizmo disturbs nothing outside its own box",
              "no rendered preview on this machine — skipped, not asserted")
    else:
        v = cv2.VideoCapture(str(pub))
        for _ in range(120):
            got, frame = v.read()
        v.release()
        before = frame.copy()
        after = bep.draw_world_gizmo_on_panel(frame.copy(), cams,
                                              cal.get("up_axis"))
        d = np.argwhere(np.abs(after.astype(np.int16)
                               - before.astype(np.int16)).max(2) > 0)
        okbox = (len(d) > 200 and d[:, 1].min() >= MID_X0
                 and d[:, 1].max() < MID_X1 and d[:, 0].max() < MID_Y1)
        check(okbox, "the gizmo disturbs nothing outside its own box",
              f"{len(d)} pixels changed, all in x "
              f"[{d[:,1].min()}, {d[:,1].max()}] y [{d[:,0].min()}, "
              f"{d[:,0].max()}] — the middle tile is x [{MID_X0}, {MID_X1})")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\npreview world gizmo: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
