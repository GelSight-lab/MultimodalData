"""The DexForce virtual target is drawn where the target pose actually projects.

The preview shows where the sensor IS. The force channel also carries where a
stiffness controller would have been COMMANDED to put it — `target_pose`,
observed pose advanced along the calibrated pressing direction by force / k.
Drawing both, joined, is the DexForce picture: the gap between them is the
force, in the same units and the same view as the motion.

The whole value of that picture is that the gap is trustworthy, so the checks
are about provenance, not about pixels being present:

  1  no force, no gap. When force is 0 the target IS the observed pose
     (asserted in the exporter), so the overlay must add nothing at all —
     a marker hovering over a sensor in free space would read as contact.
  2  the marker sits on the ray from the sensor to the target, at
     `TARGET_GAIN` times the true offset, projected by the SAME function that
     places the sensor dot. The gain exists because at true scale the gap is
     sub-pixel — measured p50 0.00 px, max 1.41 px on a real episode, against
     a force disc of radius 16.9 px — so the ring sat inside the disc and the
     picture said nothing. The exaggeration is applied to the world-space
     OFFSET before projection, so the ring stays on the pressing direction;
     scaling projected pixels would swing it off that line under an oblique
     camera.
  3  the gap grows with force, monotonically, at the shipped stiffness.

    python scripts/test_force_target_overlay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def _rig():
    """A camera looking down -z at a sensor 600 mm away, and one gel offset."""
    import viz
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 200.0]   # close enough that mm-scale gaps project
    intr = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}
    cam = [{"index": viz.DISPLAY_ORDER[0], "T_mocap_to_cam": T,
            "intrinsics": intr}]
    gel = np.array([0.0, 0.0, 0.0])
    return cam, gel


def _panel():
    import viz
    return np.zeros((viz.PANEL_H, viz.PANEL_W, 3), np.uint8)


def main() -> int:
    import viz
    from force_recovery.dexforce import STIFFNESS_N_PER_M

    k = STIFFNESS_N_PER_M / 1000.0
    cam, gel = _rig()
    # POSE7 POSITION IS IN METRES — `pose7_to_T` multiplies by 1000. My
    # first version wrote 10.0 meaning millimetres and put the sensor ten
    # metres away, where it projects to a plausible-looking pixel and
    # proves nothing.
    pose = np.array([0.010, -0.005, 0.0, 0.0, 0.0, 0.0, 1.0])   # m + xyzw
    poses = {"sensor_left": (0, pose), "sensor_right": None}

    if not hasattr(viz, "draw_projection_overlay"):
        check(False, "the overlay accepts a virtual target", "viz missing")
        return 1
    import inspect
    sig = inspect.signature(viz.draw_projection_overlay)
    if "targets_7" not in sig.parameters:
        check(False, "the overlay accepts a virtual target",
              "draw_projection_overlay has no `targets_7` parameter")
        check(False, "no force means no target marker", "not attempted")
        check(False, "the marker sits at the projected target", "not attempted")
        check_preview_frames()
        _report()
        return 1
    check(True, "the overlay accepts a virtual target",
          "draw_projection_overlay(targets_7=...)")

    def render(target=None, force=None):
        p = _panel()
        viz.draw_projection_overlay(
            p, poses, cam, gel, gel,
            forces_n={"left": force} if force is not None else None,
            targets_7={"left": target} if target is not None else None)
        return p

    # 1 — force 0: the target IS the pose, so nothing may be added.
    base = render()
    same = render(target=pose.copy(), force=0.0)
    d = int(np.abs(base.astype(int) - same.astype(int)).sum())
    check(d == 0, "no force means no target marker",
          f"target == pose adds {d} units of ink (must be 0)")

    # 2 — the marker must land on project_gel_pose(target), not anywhere else.
    #     Advance 3 mm along +z (this pose's own axis) => force 6 N at k=2.
    # ALONG +x, NOT +z. The test camera looks down +z, so a target displaced
    # along +z moves along the optical axis and projects almost on top of the
    # sensor — my first version measured a 1.0 px gap for every force and
    # called the renderer broken.
    tgt = pose.copy()
    tgt[0] += 0.003          # 3 mm lateral  => 6 N at k = 2 N/mm
    sen = viz.project_gel_pose(pose, gel, cam[0]["T_mocap_to_cam"],
                               cam[0]["intrinsics"])
    sx, sy = viz._scale_to_thumb(sen[0][0], sen[0][1])
    # BASE MUST CARRY THE SAME FORCE. Diffing against a force-free render
    # would put the force disc into "new ink" and the farthest such pixel is
    # on the disc rim, not on the target.
    base_f = render(force=6.0)
    got = render(target=tgt, force=6.0)
    diff = np.abs(got.astype(int) - base_f.astype(int)).sum(axis=2)
    ys, xs = np.nonzero(diff > 20)
    # where the marker MUST be: offset exaggerated in world mm, then projected
    shown = tgt.copy()
    shown[:3] = pose[:3] + viz.TARGET_GAIN * (tgt[:3] - pose[:3])
    exp = viz.project_gel_pose(shown, gel, cam[0]["T_mocap_to_cam"],
                               cam[0]["intrinsics"])
    if exp is None or len(xs) == 0:
        check(False, "the marker sits at the projected target",
              f"projection {'failed' if exp is None else 'drew nothing'}")
    else:
        ex, ey = viz._scale_to_thumb(exp[0][0], exp[0][1])
        # the new ink must reach the projected target; the connecting line
        # spans from the sensor dot, so check the FARTHEST new ink from the
        # sensor centre lands there
        far = int(np.argmax((xs - sx) ** 2 + (ys - sy) ** 2))
        err = float(np.hypot(xs[far] - ex, ys[far] - ey))
        check(err <= 8.0, "the marker sits at the projected target",
              f"farthest new ink at ({xs[far]}, {ys[far]}), "
              f"expected (gain {viz.TARGET_GAIN}x) at ({ex:.1f}, {ey:.1f}), "
              f"error {err:.1f} px")

    # 3 — the gap must grow with force, at the shipped k.
    # In NATIVE camera pixels: `_scale_to_thumb` rounds to the panel grid, and
    # three forces an integer apart can land on one thumbnail pixel.
    gaps = []
    sen_n = viz.project_gel_pose(pose, gel, cam[0]["T_mocap_to_cam"],
                                 cam[0]["intrinsics"])[0]
    for f_n in (1.0, 3.0, 6.0):
        t = pose.copy()
        t[0] += (f_n / k) / 1000.0
        pr = viz.project_gel_pose(t, gel, cam[0]["T_mocap_to_cam"],
                                  cam[0]["intrinsics"])[0]
        gaps.append(float(np.hypot(pr[0] - sen_n[0], pr[1] - sen_n[1])))
    check(gaps[0] < gaps[1] < gaps[2], "the gap grows with force",
          f"k={k} N/mm: 1 N -> {gaps[0]:.1f} px, 3 N -> {gaps[1]:.1f} px, "
          f"6 N -> {gaps[2]:.1f} px")

    check_axis_convention()
    check_press_arrow()
    check_preview_frames()
    _report()
    return 1 if sum(not ok for ok, _, _ in RESULTS) else 0


def check_axis_convention() -> None:
    """The pressing direction: default -y, with the dual-ball fit kept.

    The published `gel_axis_in_rigid` is normalize(gelball - refball): the
    line between two calibration ball centres, from three poses. It never
    measured the gel surface. It is the normal only if the fixture held both
    balls along the normal, and the evidence says it did not for the right
    sensor -- pressing hard on a level board, that axis sits 18.1 deg off the
    board normal while local -y sits 7.7 deg off. On the left the ordering
    reverses (7.1 vs 25.6), so the two sensors cannot both be right.

    A second sign of trouble in the same file: `depth_offset_mm` is -5.0 on
    the left (the ball centre backed off one ball radius to reach the gel
    surface) and 0.0 on the right (not backed off at all).

    Kinematics cannot settle it -- sum(R_i) has singular-value ratio 1.09 and
    1.04, so the axis is unidentifiable from motion alone.

    So: -y is the default, the fit stays reachable, and neither is silent.
    """
    import numpy as _np
    from force_recovery.dexforce import gel_axis as _ga

    try:
        d = _ga("motherboard", "left")
        b = _ga("motherboard", "left", source="dual_ball")
        y = _ga("motherboard", "left", source="body_y")
    except TypeError:
        check(False, "the pressing direction defaults to sensor local -y",
              "gel_axis() has no `source` argument")
        return
    ok = (_np.allclose(d, [0, -1, 0]) and _np.allclose(y, [0, -1, 0])
          and abs(_np.linalg.norm(b) - 1) < 1e-12
          and not _np.allclose(b, [0, -1, 0]))
    sep = float(_np.degrees(_np.arccos(abs(_np.clip(b @ y, -1, 1)))))
    check(ok, "the pressing direction defaults to sensor local -y",
          f"default == body_y == {_np.round(d, 3).tolist()}; the dual-ball "
          f"fit is still reachable at {_np.round(b, 3).tolist()}, {sep:.1f} "
          f"deg away")


def check_press_arrow() -> None:
    """The overlay must SHOW which way the force acts.

    GelSight Mini reports a normal force along the gel's own normal, in the
    SENSOR's local frame -- `gel_axis_in_rigid`, roughly local -y but not
    exactly any axis: (-0.17, -0.93, -0.32) on the left. A viewer reading a
    scalar "2.0 N" off the panel has no way to know that, and the natural
    guess -- world vertical -- is off by a median 7.7 deg on motherboard and
    23.3 deg on pushT.

    Checked against R(q) @ gel_axis projected into the image, NOT against a
    coordinate axis, and separately checked to be distinguishable from all
    three of them: an annotation that lands on top of an axis already drawn
    tells the viewer nothing new.
    """
    import numpy as _np
    from scipy.spatial.transform import Rotation as _R
    import viz as _v
    from force_recovery.dexforce import gel_axis as _ga

    cam = {"index": 2,
           "T_mocap_to_cam": _np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                        [0, 1, 0, 900.0], [0, 0, 0, 1]]),
           "intrinsics": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}}
    q = _R.from_euler("xyz", [12, -20, 33], degrees=True).as_quat()
    pose = _np.array([0.100, -0.060, 0.040, *q])      # metres, as everywhere
    gel = _np.array([-42.6, -36.6, -34.2])
    ax = _ga("motherboard", "left")

    if not hasattr(_v, "press_arrow_pixels"):
        check(False, "the panel shows the direction the force acts along",
              "viz.press_arrow_pixels does not exist")
        return
    panel = _np.zeros((480, 1280, 3), _np.uint8)
    before = panel.copy()
    _v.draw_projection_overlay(panel, {"sensor_left": (0, pose)},
                               [cam], gel, gel, forces_n={"left": 4.0},
                               press_axis={"left": ax, "right": ax})
    drew = _np.abs(panel.astype(int) - before.astype(int)).max(2) > 0
    got = _v.press_arrow_pixels(pose, gel, ax, cam)
    if got is None or not drew.any():
        check(False, "the panel shows the direction the force acts along",
              "nothing drawn, or the arrow does not project")
        return
    (u0, v0), (u1, v1) = got
    # press_arrow_pixels answers in FULL camera pixels (640x480). The panel
    # holds 320x240 thumbnails side by side, so the drawn ink is at
    # thumb(u,v) + the slot offset. Comparing the two spaces directly was the
    # first version of this check and it failed on correct drawing.
    from viz import _scale_to_thumb as _s2t, DISPLAY_POSITION, RS_THUMB_W
    tx, ty = _s2t(u1, v1)
    tx += DISPLAY_POSITION[cam["index"]] * RS_THUMB_W
    win = drew[max(0, ty - 5):ty + 6, max(0, tx - 5):tx + 6]
    d = _np.array([u1 - u0, v1 - v0], float)
    d /= (_np.linalg.norm(d) or 1)
    # The arrow must follow the ACTIVE axis. It used to be checked as ">8 deg
    # from every body axis", which was right while the axis was the dual-ball
    # fit and became wrong the moment -y became the default: the force really
    # does act along a body axis now, so that check failed on correct
    # drawing. What still has teeth is that the arrow TRACKS the parameter --
    # drawn with the dual-ball axis it must move by that axis's own 21.2 deg.
    alt = _ga("motherboard", "left", source="dual_ball")
    t2 = _v.press_arrow_pixels(pose, gel, alt, cam)
    moved = None
    if t2 is not None:
        a2 = _np.array([t2[1][0] - t2[0][0], t2[1][1] - t2[0][1]], float)
        a2 /= (_np.linalg.norm(a2) or 1)
        moved = float(_np.degrees(_np.arccos(abs(_np.clip(d @ a2, -1, 1)))))
    okarrow = bool(win.any())
    oktrack = moved is not None and moved > 3.0
    check(okarrow and oktrack,
          "the panel shows the direction the force acts along",
          f"ink reaches the projected R(q) @ gel_axis tip at panel "
          f"({tx}, {ty}); switching to the dual-ball axis swings the drawn "
          f"arrow {moved:.1f} deg, so it follows the axis rather than being "
          f"hardcoded"
          if okarrow and oktrack else
          f"tip ({tx}, {ty}): {'no ink' if not okarrow else 'ink ok'}; "
          f"dual-ball swing {moved}")


def check_preview_frames() -> None:
    """The preview draws the target from RELEASE poses over RAW-H5 frames.

    Everything else in build_episode_previews works in the recorded Y-up
    convention: it reads poses out of the HDF5, adds the Y-up world offset and
    projects with the Y-up extrinsics. The DexForce target alone comes from
    `_release_poses`, deliberately, so the drawn target is the published one.

    After the release was rotated to Z-up those two stopped being the same
    frame, and the target markers left the picture entirely -- magenta lines
    running off the bottom of every panel. The checks above did not see it:
    they exercise the drawing primitive on synthetic input, where both ends
    come from the same array.

    So compare the two SOURCES, not the drawing: same episode, same rows.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent))
    import numpy as _np
    try:
        import h5py
        import importlib.util as _il
        _sp = _il.spec_from_file_location(
            "_bep", _P(__file__).resolve().parent / "build_episode_previews.py")
        _b = _il.module_from_spec(_sp); _sp.loader.exec_module(_b)
        from twm.calib_epoch import world_offset_m
    except Exception as ex:
        check(False, "release poses and raw H5 poses are in one frame",
              f"could not load the preview module: {type(ex).__name__}")
        return
    # EVERY task. The first version checked motherboard only, and pushT --
    # which was still Y-up while this code assumed the release was Z-up --
    # had its DexForce target rotated into nonsense and drawn off-frame for
    # as long as that went unnoticed.
    from react_paths import release_root as _rr2
    for task in sorted(q.name for q in _rr2().iterdir()
                       if (q / "episodes.jsonl").exists()):
        _one_task(task)


def _one_task(task: str) -> None:
    import json
    import numpy as _np
    import h5py
    import importlib.util as _il
    from pathlib import Path as _P
    _sp = _il.spec_from_file_location(
        "_bep", _P(__file__).resolve().parent / "build_episode_previews.py")
    _b = _il.module_from_spec(_sp); _sp.loader.exec_module(_b)
    from twm.calib_epoch import world_offset_m
    from react_paths import release_root as _rr3
    ejs = (_rr3(task) / "episodes.jsonl").read_text().splitlines()
    key = json.loads([l for l in ejs if l.strip()][0])["episode"]
    date, ep = key.split("/")
    h5 = _P("/media/yxma/Disk1/twm/data") / task / date / f"{ep}.h5"
    rel = _b._release_poses(task, date, ep)
    if not h5.exists() or not rel:
        check(True, "release poses and raw H5 poses are in one frame",
              "raw HDF5 not on this machine — skipped, not asserted")
        return
    # Align the way the RENDERER does. `source_h5_frame` is a CAMERA frame
    # index; OptiTrack runs at its own rate and the release interpolates to
    # camera timestamps. Indexing the OptiTrack array by camera frame samples
    # a different instant and reads ~100 mm even on a date with no offset --
    # a floor that hides exactly the error this is looking for.
    import pyarrow.parquet as _pq
    from react_preprocess.config import STAGE_ROOT as _SR
    from viz import optitrack_at as _at
    from twm.visualize import load_optitrack as _lo
    pf = _SR / task / "meta" / date / f"{ep}.parquet"
    t = _pq.read_table(str(pf), columns=["source_h5_frame"]).to_pydict()
    idx = _np.asarray(t["source_h5_frame"], int)
    off = _np.asarray(world_offset_m(task, date, ep, up_axis="y"), float)
    with h5py.File(h5, "r") as f:
        lut = _lo(f)
        _b._apply_world_offset(lut, *off)
        cam_ts = f["timestamps"][:]
    R = rel["left"]
    e = []
    for k in range(0, min(len(idx), len(R)), 7):
        i = int(idx[k])
        if i >= len(cam_ts):
            continue
        q = _at(lut, float(cam_ts[i])).get("sensor_left")
        if q is None:
            continue
        e.append(float(_np.linalg.norm(
            _np.asarray(q[1][:3], float) - R[k, :3]) * 1000.0))
    e = _np.asarray([x for x in e if _np.isfinite(x)])
    d = float(_np.median(e)) if len(e) else float("inf")
    check(len(e) > 50 and d < 1.0,
          "the target's poses and the drawn poses are one frame",
          f"{len(e)} rows, median {d:.2f} mm, max "
          f"{(e.max() if len(e) else float('nan')):.2f} mm "
          f"(a Y-up/Z-up mix reads 515 mm and puts the target off-frame)")


def _report() -> None:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    print(f"\nforce target overlay: {len(RESULTS)} checks, "
          f"{sum(not ok for ok, _, _ in RESULTS)} failing")


if __name__ == "__main__":
    raise SystemExit(main())
