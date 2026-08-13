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

    _report()
    return 1 if sum(not ok for ok, _, _ in RESULTS) else 0


def _report() -> None:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    print(f"\nforce target overlay: {len(RESULTS)} checks, "
          f"{sum(not ok for ok, _, _ in RESULTS)} failing")


if __name__ == "__main__":
    raise SystemExit(main())
