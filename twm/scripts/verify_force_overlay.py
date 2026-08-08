"""Render the highest-force frames of one episode and MEASURE the overlay.

Written because the previous overlay verifier scanned only x in [0,480) — the
left tactile tiles — and reported "no overlay" for a clip whose dot was on the
right sensor. A verifier that inspects the wrong region is worse than none: it
reports success for a broken render.

So this one asserts three things it can actually measure:
  1. orange ink EXISTS, and lies in row 1 (the camera views), not row 2;
  2. its centroid sits within a few px of where `project_gel_pose` says the
     sensor is — the whole point of moving the dot;
  3. a sensor outside a camera's frustum draws NOTHING in that view, rather
     than a disc on whichever thumbnail its out-of-range coordinates land on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, "/home/yxma/MultimodalData")

from twm.force_overlay import (load_forces, radius_px,      # noqa: E402
                               row_for_h5_frame)
from twm.scripts.build_episode_previews import (            # noqa: E402
    FORCE_ROOT, _load_proj_calibs, _parquet_trim_and_rows)
from twm.tactile_align import gel_lag_frames                # noqa: E402
from twm.viz import (ROW2_Y, build_preview_panel,           # noqa: E402
                     draw_projection_overlay, load_optitrack, optitrack_at,
                     project_gel_pose, DISPLAY_POSITION, RS_THUMB_W, RS_THUMB_H,
                     _scale_to_thumb)

# Heavy artefacts live off-repo, like every other figure here.
OUT = Path("/media/yxma/Disk1/twm/figures/overlay_check")


def changed_mask(after, before):
    """Pixels the overlay actually altered.

    NOT a hue test. The first version classified "orange" ink and subtracted
    the pre-overlay orange, which silently erased every dot pixel that landed
    on the reddish motherboard — coverage read 22% where the render was in
    fact correct. The overlay is a blend, so what it changes is what it drew.
    """
    return (after.astype(int) - before.astype(int)).any(axis=2)


def main(h5_path: str, n: int = 4):
    h5 = Path(h5_path)
    task, date, ep = h5.parent.parent.name, h5.parent.name, h5.stem
    forces = load_forces(task, date, ep, FORCE_ROOT)
    if not forces:
        raise SystemExit(f"no force npz for {task}/{date}/{ep}")
    trim_pq, n_rows = _parquet_trim_and_rows(task, date, ep)
    cams, gel_L, gel_R = _load_proj_calibs(task)   # this task's epoch, verified
    OUT.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5), "r") as f:
        cam_ts = f["timestamps"][:]
        ot = load_optitrack(f)
        # Raw-H5 poses need the per-episode world offset (05-19 redefined the
        # origin); without it the differential checks still pass — they compare
        # ink to their own projection — while measuring the wrong place.
        from twm.calib_epoch import world_offset_m
        from twm.scripts.build_episode_previews import _apply_world_offset
        _apply_world_offset(ot, *world_offset_m(task, date, ep))
        lag = gel_lag_frames(f)
        n_gel = len(f["gelsight/left/frames"])
        gel_at = lambda i: min(int(i) + lag, n_gel - 1)      # noqa: E731

        # pick the frames with the LARGEST force, where the dot is biggest and
        # a misplacement is unmistakable
        side0 = max(forces, key=lambda s: np.nanmax(forces[s]))
        arr = forces[side0]
        rows = np.argsort(arr)[::-1]
        picks, seen = [], set()
        for r in rows:
            fr = int(r) + int(trim_pq) + 15
            if fr >= len(cam_ts) or any(abs(fr - p) < 60 for p in seen):
                continue
            picks.append(fr); seen.add(fr)
            if len(picks) >= n:
                break

        ref = int(picks[0])
        gs_ref = [f["gelsight/left/frames"][gel_at(ref)],
                  f["gelsight/right/frames"][gel_at(ref)]]
        print(f"{task}/{date}/{ep}  strongest side={side0}  "
              f"peak={np.nanmax(arr):.2f} N")

        for fr in picks:
            colors = [f[f"realsense/cam{c}/color"][fr] for c in range(3)]
            poses = optitrack_at(ot, float(cam_ts[fr]))
            panel = build_preview_panel(
                color_frames=colors,
                gs_frames=[f["gelsight/left/frames"][gel_at(fr)],
                           f["gelsight/right/frames"][gel_at(fr)]],
                gs_ref=gs_ref, optitrack_poses=poses,
                recording=False, frame_count=fr, elapsed=0.0,
                status_override="overlay check")
            row = row_for_h5_frame(fr, trim_pq, n_rows)
            ff = {s: float(a[row]) for s, a in forces.items() if row < len(a)}
            # Two renders, differing ONLY in the force argument. Diffing
            # against the bare panel instead measured the pose axes too --
            # they legitimately extend past a camera view into the tactile
            # row, so "ink in row 2" convicted the disc of the axes' habit.
            base = panel.copy()
            draw_projection_overlay(base, poses, cams, gel_L, gel_R)
            draw_projection_overlay(panel, poses, cams, gel_L, gel_R,
                                    forces_n=ff)

            # --- measure -------------------------------------------------
            new = changed_mask(panel, base)
            top = new[:ROW2_Y].sum()
            bot = new[ROW2_Y:].sum()
            # where does the projection say the sensor is?
            expect = []
            for pc in cams:
                slot = DISPLAY_POSITION.get(pc["index"])
                if slot is None:
                    continue
                for side, key, gel in (("left", "sensor_left", gel_L),
                                       ("right", "sensor_right", gel_R)):
                    pt = poses.get(key) if poses else None
                    if pt is None or ff.get(side, 0) <= 0.02:
                        continue
                    res = project_gel_pose(pt[1], gel, pc["T_mocap_to_cam"],
                                           pc["intrinsics"])
                    if res is None:
                        continue
                    ex, ey = _scale_to_thumb(res[0][0], res[0][1])
                    # In view of THIS camera? `project_gel_pose` returns
                    # coordinates outside the image when the sensor is out of
                    # frustum; adding the slot offset then lands them on the
                    # neighbouring thumbnail, which is how a disc first ended
                    # up somewhere no sensor was.
                    inview = 0 <= ex < RS_THUMB_W and 0 <= ey < RS_THUMB_H
                    expect.append((ex + slot * RS_THUMB_W, ey, side, inview))
            miss, skipped = [], 0
            for ex, ey, side, inview in expect:
                if not inview:
                    skipped += 1
                    r = int(radius_px(ff[side]))
                    win = new[max(ey - r, 0):ey + r + 1,
                              max(ex - r, 0):ex + r + 1]
                    assert win.sum() == 0, (
                        f"{side} is outside this camera's view at ({ex},{ey}) "
                        f"but ink was drawn there")
                    continue
                r = int(radius_px(ff[side]))
                win = new[max(ey - r, 0):ey + r + 1, max(ex - r, 0):ex + r + 1]
                cover = win.mean()
                assert cover > 0.5, (f"{side}@({ex},{ey}) is in view but the "
                                     f"disc covers only {cover:.0%} of it")
                miss.append(f"{side}@({ex},{ey}) cover={cover:.0%}")
            fname = OUT / f"{ep}_f{fr:05d}.png"
            cv2.imwrite(str(fname), panel)
            print(f"  frame {fr:5d}  F={ {k: round(v, 2) for k, v in ff.items()} }"
                  f"  ink row1={top:5d}px row2={bot}px  "
                  f"{' | '.join(miss)}"
                  + (f"  | {skipped} out-of-view correctly skipped"
                     if skipped else ""))
            assert bot == 0, (f"force disc leaked {bot}px into the tactile "
                              f"row -- it must stay inside its camera view")
            assert top > 0, "no force disc in the camera views"
    # The area-linear law is NOT checked here: total ink per frame also
    # depends on how many of the three cameras see the sensor, which varies
    # frame to frame, so a ratio across frames would be measuring visibility
    # and calling it geometry. `test_units` asserts the law directly on
    # `radius_px` and on rendered ink at fixed position.
    print(f"\nwrote {n} panels to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
