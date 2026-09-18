"""Generate per-episode preview MP4s in the canonical viewer layout.

Each preview shows the **first 30 seconds of usable data** (i.e. frames
after the OT-uninitialized trim prefix) played back at **2x speed**, so
each preview is 15 s long regardless of episode length (or shorter if the
recording itself is < 30 s). Compared to the previous "flipbook" recipe
that sampled 60 frames evenly across the whole episode and played at
50-120x speed, this gives users a real sense of the actual motion.

  - Source frames consumed: trim_offset .. trim_offset + 900 (or end of H5)
  - Output encoding: 60 fps (= 2x the 30 fps recording rate), yuv444p H.264 CRF 20
  - Output length: 15 s (or T/60 s if T < 900)
  - File size: roughly 10-25 MB per preview

Reuses `twm.viz.build_preview_panel` + `draw_projection_overlay` for the
exact same layout as the live viewer / `play_react_pt.py`.

Usage
-----
    python scripts/build_episode_previews.py --date 2026-05-19
    python scripts/build_episode_previews.py --date 2026-05-10 --clip_s 30 --speed 2

Output: <STAGE_ROOT>/<task>/previews/<date>/episode_NNN.mp4 — the same tree
the release pipeline renders and the uploader publishes from. This script
used to write to figures/episode_previews/ instead, so a full re-render
refreshed one tree while the uploader kept publishing the other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from functools import partial

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.data_collection import REALSENSE_SERIALS    # noqa
from twm.recorder.frames import decode_arducam  # noqa: E402
from twm.wrist_tone import apply_tone_curve, episode_wrist_gamma  # noqa: E402
from twm.visualization import (
    Projection, draw_preview_overlay, render_preview as build_preview_panel,
)
from twm.viz import (
    DISPLAY_ORDER,
    GS_THUMB_W,
    RS_THUMB_H,
    RS_THUMB_W,
    ROW2_Y,
    cam_aligned_pose,
    load_optitrack,
    optitrack_at,
    load_calibrations,
)


def _preview_root(task: str) -> Path:
    """Where previews live, asked of the release config rather than restated.

    One directory, because two directories of identically-named clips is how a
    re-render can appear to succeed and publish nothing.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from react_preprocess.config import STAGE_ROOT
    return STAGE_ROOT / task / "previews"


TASK = "motherboard"          # overridden by --task in main()
H5_ROOT  = Path("/media/yxma/Disk1/twm/data") / TASK
OUT_ROOT = _preview_root(TASK)
# Which extrinsics a task needs is NOT a constant here — cameras were
# recalibrated between tasks and this file pointed at one directory for all of
# them, publishing 36 motherboard previews with pushT's June-26 calibration.
# calib_epoch owns the mapping; see its docstring.

SOURCE_FPS       = 30.0       # recording frame rate
CLIP_S_DEFAULT   = 30.0       # seconds of usable data to sample (real time)
SPEED_DEFAULT    = 2.0        # playback speed (output_fps = SOURCE_FPS * speed)
EPISODES_ROOT    = Path("/media/yxma/Disk1/twm/processed/episodes") / TASK


def _apply_world_offset(ot_lookup, dx, dy, dz):
    """Apply (dx, dy, dz) m offset to every OT sample (in-place mutation
    of the `poses` array inside the (ts, poses) tuple).

    The offset is per EPISODE and comes from `calib_epoch.world_offset_m`,
    which reads the release's own episodes.jsonl. It used to come from
    --dx/--dy/--dz flags defaulting to zero, so the 2026-05-19 session — whose
    raw poses sit 0.23 m + 0.175 m from every other date — rendered with its
    sensor marker a quarter-metre off unless someone remembered the flags."""
    if dx == 0 and dy == 0 and dz == 0:
        return
    for name, data in ot_lookup.items():
        if data is None:
            continue
        ts, poses = data
        if poses.ndim == 2 and poses.shape[1] >= 3:
            poses[:, 0] += dx
            poses[:, 1] += dy
            poses[:, 2] += dz


# The world gizmo sits in the MIDDLE camera tile: the camera row is four
# tiles wide and the middle camera is the second, x in [320, 640). At the
# frame's own top-left it would sit on the LEFT camera and still look
# deliberate. Sized for a 320x240 tile, not for the 640x480 probe overlays.
# DERIVED, not restated: display position 1 is the middle tile, and
# DISPLAY_ORDER says which H5 camera index that tile shows.
MIDDLE_CAM_IDX = DISPLAY_ORDER[1]
GIZMO_SIZE = 26
GIZMO_MARGIN = 10
GIZMO_ORIGIN = (RS_THUMB_W + GIZMO_MARGIN + GIZMO_SIZE + 22,
                GIZMO_MARGIN + GIZMO_SIZE + 22)   # + the label reach


def draw_world_gizmo_on_panel(panel, project_cams, up_axis_of_source):
    """Draw the PUBLISHED world frame in the middle camera tile, in place.

    This renderer works entirely in the recorded Y-up frame, but every number
    a reader downloads is Z-up. Labelling the axes with the convention this
    process happens to hold would ship a video whose axis names contradict
    the parquet beside it. So the extrinsics are converted for the gizmo --
    and only for the gizmo; nothing else here moves.
    """
    from react_toolbox.frames import as_up_axis, UP_AXIS_RECORDED
    from react_toolbox.viz import draw_world_gizmo

    cam = next((c for c in project_cams if c["index"] == MIDDLE_CAM_IDX), None)
    if cam is None:
        return panel
    cal = as_up_axis({"up_axis": up_axis_of_source or "y",
                      "cams": {"middle": {
                          "T_mocap_to_cam": np.asarray(cam["T_mocap_to_cam"],
                                                       float),
                          "intrinsics": cam["intrinsics"]}}},
                     UP_AXIS_RECORDED)
    # This panel is BGR; react_toolbox.viz's axis palette is written for RGB
    # images. Drawn straight it gave a blue x and an orange z next to sensor
    # triads whose X is red -- two colour conventions in one frame, and no
    # position check or text search can see it. Swap in, swap back.
    tile = panel[:RS_THUMB_H, RS_THUMB_W:2 * RS_THUMB_W]
    drawn = draw_world_gizmo(tile[:, :, ::-1], cal["cams"]["middle"],
                             corner="tl", size=GIZMO_SIZE,
                             margin=GIZMO_MARGIN,
                             title=f"world ({UP_AXIS_RECORDED}-up)")
    tile[:] = drawn[:, :, ::-1]
    return panel


def _load_proj_calibs(task: str, date: str | None = None):
    """Extrinsics for the epoch THIS SESSION declares, verified on load.

    Keyed by (task, date), not by task: the rig was recalibrated between
    sessions, so two dates of the same task can need different extrinsics —
    2026-09-09 motherboard is the June-26 epoch, the May dates are May-12.
    """
    check_epoch(task, date)                 # refuses the wrong epoch loudly
    cdir = calib_dir(task, date=date)
    print(f"  calibration epoch {epoch_of(task, date)}  ({cdir.name})")
    up_axis = (json.loads((cdir / "T_mocap_to_cam_middle.json").read_text())
               .get("up_axis") or "y")
    cam_calib = [
        str(cdir / "T_mocap_to_cam_middle.json"),
        str(cdir / "T_mocap_to_cam_left.json"),
        str(cdir / "T_mocap_to_cam_right.json"),
    ]
    gel_L = str(cdir / "T_gel_to_rigid_left.json")
    gel_R = str(cdir / "T_gel_to_rigid_right.json")
    try:
        cam_calibs, gel_center_left, gel_center_right = load_calibrations(
            cam_calib, gel_L, gel_R)
        project_cams = []
        for c in cam_calibs:
            try:
                c_idx = REALSENSE_SERIALS.index(c["camera_serial"])
            except ValueError:
                continue
            project_cams.append({
                "index":          c_idx,
                "T_mocap_to_cam": c["T_mocap_to_cam"],
                "intrinsics":     c["intrinsics"],
                "serial":         c["camera_serial"],
                "rmse":           c.get("rmse_mm", 0.0),
            })
        return project_cams, gel_center_left, gel_center_right, up_axis
    except Exception as e:
        print(f"  WARN: calibration load failed ({e}); previews will lack projection overlay")
        return [], None, None, "y"


from twm.tactile_align import describe as gel_describe, gel_lag_frames
from twm.calib_epoch import (calib_dir, check_epoch, epoch_of,
                             world_offset_m, describe as calib_describe)
from twm.force_overlay import (draw_legend, load_forces, load_targets,
                               row_for_h5_frame)

FORCE_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")


def preview_reference(h5file, side: str, task: str, date: str, ep: str):
    """The no-contact gel this episode's diff tiles are drawn against.

    THE SAME POOL THE FORCE CHANNEL USES, imported rather than re-decided:
    `run_episode._reference_rows` takes the 15 lowest-intensity fresh rows at
    least a second apart, and the reference is the median of twelve of them.

    What was here was `sample_idx[0]` — whatever frame the recording happened
    to start on — with the comment "use the first sampled frame's gelsight as
    the static diff reference". On a recording that starts with the gel
    already pressed that is a press, and the whole episode's difference image
    is then referenced against it. Measured against an independent estimate of
    the free gel (the per-pixel median over the entire recording), as the
    fraction of the reference's own pixels already in contact:

        motherboard/2026-05-10     first frame     this pool
        episode_001                   0.21%          0.11%
        episode_002                   0.25%          0.12%
        episode_004                  12.98%          0.05%

    A single frame is also fragile where a median of twelve is not: one noisy
    frame moves it, and nothing downstream can tell.

    Falls back to the first frame ONLY if the release parquet is missing, and
    says so, because a preview quietly drawn against a different reference
    than the force channel is the defect this exists to remove.
    """
    import numpy as np

    from force_recovery.lut_calibration import crop
    from force_recovery.run_episode import STAGE_ROOT, reference_stack
    from twm.react_preprocess.h5io import open_episode
    from twm.tactile_align import gel_lag_frames

    frames = h5file[f"gelsight/{side}/frames"]
    n = len(frames)
    lag = gel_lag_frames(h5file)
    p = STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"
    if not p.exists():
        print(f"    ! {ep}: no release parquet — diff reference falls back "
              f"to the first frame", flush=True)
        return frames[min(lag, n - 1)]
    import pyarrow.parquet as pq
    t = pq.read_table(str(p))
    inten = np.asarray(t[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(t[f"tactile_{side}_is_new"].to_numpy())
    align = open_episode(Path(h5file.filename), task).align[side]
    stack = reference_stack(frames, align.index_map, inten, is_new)
    # uint8 back out: the panel builder differences raw frames, and `crop` is
    # applied downstream by whoever needs it — this must stay the same dtype
    # and shape as the `frames[i]` it replaces.
    assert crop is not None
    return np.median(stack, 0).astype(stack.dtype)


def _gel_source_frames(task, date, ep, out_root=None) -> dict:
    """Per-side: which raw H5 gel frame each row's force was computed from.

    Read from the force npz, NOT from a parquet. The npz holds
    `force_normal_n` and `source_frame` side by side, so the value drawn and
    the frame it came from travel in one file and cannot disagree. The uncut
    release parquet — the one the row mapping comes from — has no force
    columns at all, so it could not answer this anyway.
    """
    import numpy as _np
    root = Path(out_root) if out_root is not None else FORCE_ROOT
    out = {}
    for side in ("left", "right"):
        f = root / task / date / f"{ep}_{side}.npz"
        if not f.exists():
            continue
        with _np.load(f) as z:
            if "source_frame" in z:
                out[side] = _np.asarray(z["source_frame"], dtype=int)
    return out


def _parquet_trim_and_rows(task, date, ep, parquet=None):
    """Where this episode starts in the H5, and how many rows it has.

    THE source of both, for the clip window and for the force rows alike.
    They used to come from different places — the window from a
    processed/.pt sidecar, the rows from here — and the sidecar version answered 0
    for any episode whose .pt was missing. Zero is also the correct answer
    for every motherboard episode, so pushT (trim 5373-11617) played minutes
    of pre-roll and looked like frozen action rather than like a bug.

    Raises for an episode the release does not publish: a preview of
    unpublished data is a claim that it exists, which is the same defect the
    uploader's orphan gate refuses at the other end.
    """
    import numpy as _np
    import pyarrow.parquet as pq
    # `parquet` names the tree this episode was actually published from. It
    # defaults to the uncut release, which is where whole episodes live; cut
    # segments live in release_cut and have no entry there at all.
    f = (Path(parquet) if parquet is not None
         else Path("/media/yxma/Disk1/twm/release")/task/"meta"/date/f"{ep}.parquet")
    if not f.exists():
        raise FileNotFoundError(
            f"{task}/{date}/{ep}: no release parquet, so its trim offset is "
            f"unknown. Refusing to assume 0 — pushT episodes start as late as "
            f"frame 11617 and would render minutes of pre-roll as if it were "
            f"the episode.")
    t = pq.read_table(str(f), columns=["source_h5_frame"])
    return int(_np.asarray(t["source_h5_frame"].to_numpy())[0]), t.num_rows


class UndeclaredPoseFrame(ValueError):
    pass


def _release_poses(task: str, date: str, ep: str) -> dict:
    """Row-indexed observed sensor poses, from the release parquet.

    THE SAME SOURCE `export_force_columns` reads. The DexForce target is
    `observed + (F/k) n_hat`; if the renderer took `observed` from the raw
    OptiTrack stream while the published column took it from the parquet, the
    drawn target and the shipped target would differ by whatever the
    resampling does — a discrepancy that would look like a calibration error.
    """
    from react_preprocess.config import STAGE_ROOT as _SR
    import numpy as _np
    import pyarrow.parquet as pq
    f = _SR / task / "meta" / date / f"{ep}.parquet"
    if not f.exists():
        return {}
    cols = [c for c in ("sensor_left_pose", "sensor_right_pose")
            if c in pq.read_schema(str(f)).names]
    if not cols:
        return {}
    t = pq.read_table(str(f), columns=cols).to_pydict()
    out = {c.split("_")[1]: _np.asarray([x for x in t[c]], float) for c in cols}
    # ...but this renderer works in the RECORDED Y-up frame: it reads poses out
    # of the HDF5, adds the Y-up world offset and projects with the Y-up
    # extrinsics. Handing a Z-up release pose to the same drawing put the
    # DexForce target 515 mm away -- magenta lines off the bottom of every
    # panel.
    #
    # READ the convention, do not assume it -- but read it somewhere it is
    # actually written. This first assumed "the release is Z-up" and rotated
    # unconditionally, which was nonsense for pushT while pushT was still
    # Y-up. The obvious repair -- read the parquet's twm.world_frame -- was
    # ALSO wrong: this tree carries no such metadata (only the force export
    # does), so every pose read as "y" and the rotation stopped happening at
    # all. Both failures look like a working renderer.
    #
    # episodes.jsonl and the calibration beside these poses do carry it.
    from react_toolbox.frames import convert_poses as _cp
    got = None
    md = pq.read_schema(str(f)).metadata or {}
    if md.get(b"twm.world_frame"):
        got = json.loads(md[b"twm.world_frame"].decode()).get("up_axis")
    if got is None:
        ej = _SR / task / "episodes.jsonl"
        if ej.exists():
            for line in ej.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("episode", "").endswith(ep):
                    got = r.get("up_axis")
                    break
    # NOT the calibration file. Its `up_axis` describes the EXTRINSICS, not
    # these poses, and reading it as a proxy is what shipped for 2026-09-09:
    # curate wrote those rows without `up_axis`, the calibration said "z", and
    # a Y-up parquet was rotated as if it were Z-up. The sensor axes stayed
    # right (they come from the H5) and only the force-induced target moved,
    # which reads as a calibration error rather than a frame bug. There is no
    # third place this is written, so there is nothing left to fall back to.
    if got is None:
        raise UndeclaredPoseFrame(
            f"{task}/{date}/{ep}: nothing declares which up_axis the release "
            f"poses are in — neither the parquet's twm.world_frame metadata nor "
            f"the episodes.jsonl row. Guessing puts the DexForce target "
            f"hundreds of mm away while every other overlay stays correct. Add "
            f"up_axis to the episodes.jsonl row (react_preprocess writes poses "
            f"straight from the H5, so a freshly built episode is 'y').")
    if got == "y":
        return out                      # already the frame this renderer uses
    return {k: _cp(v, to_zup=False) for k, v in out.items()}


def preview_targets(task, date, episode, force_root, *, enabled=True):
    if not enabled:
        return {}
    return load_targets(task, date, episode, force_root,
                        _release_poses(task, date, episode))


def _flagged_intervals(task: str, date: str, ep: str) -> list[tuple[int, int, str]]:
    """Curation intervals for this episode, in episode-frame coords.

    The release catalogues its own defects (`bad_frames.json`); a preview that
    plays a flagged frame without saying so reads as dirty data, when it is a
    known, indexed sensor dropout excluded from the training segments.
    """
    import json
    p = Path("/media/yxma/Disk1/twm/release") / task / "bad_frames.json"
    if not p.exists():
        return []
    eps = json.loads(p.read_text()).get("episodes", {})
    out = []
    for k, spans in eps.get(f"{date}/{ep}", {}).items():
        if isinstance(spans, list):
            out += [(int(a), int(b), k) for a, b in spans]
    return sorted(out)


def gel_frame(source_frames: dict, side: str, row, h5_frame: int,
              n_gel: int) -> int:
    """The H5 gel frame to SHOW for a published row.

    The GelSight runs at ~17.8 Hz against a 29.8 Hz write tick, so the
    published tactile stream is a resample of the raw one and
    `force_<side>_source_frame` records which raw frame each row's scalars
    were actually computed from -- about N+2.

    Reading `frames[N]` instead put the force disc ~2 frames ahead of the tile
    it was drawn beside: measured on motherboard/2026-09-11/episode_000, force
    led the raw image by 2 frames (r=0.991 at -2 vs 0.980 at 0), while the
    PUBLISHED video and the published scalars agree at 0. The release was
    right and the preview was the odd one out -- the same shape of defect
    `tactile_align` was written to end, one stream further along.

    Per side: the two GelSights are independent streams with their own
    timestamps, and their columns differ.

    Falls back to the tick when there is no column (older releases) or no row
    (a tick outside the published episode) -- which is what this did before,
    so a missing column cannot silently shift anything.
    """
    col = source_frames.get(side)
    if col is None or row is None or not (0 <= int(row) < len(col)):
        return min(int(h5_frame), n_gel - 1)
    return min(int(col[int(row)]), n_gel - 1)


def clip_window(trim_offset: int, window_start: int | None,
                T_h5: int, n_frames_target: int) -> tuple[int, int]:
    """The H5 frames a preview plays.

    `window_start` is where the PUBLISHED episode begins. For a cut segment
    that is not the source recording's trim: `episode_003_seg01` may start
    9000 frames into `episode_003.h5`. Deriving it from the h5 path instead --
    the only name a segment shares with its siblings -- rendered all four
    segments of one episode as the identical clip, published under four names.

    None means "the whole episode is the publication", so the trim is the
    start, which is what every uncut episode wants.
    """
    start = trim_offset if window_start is None else int(window_start)
    return start, min(T_h5, start + n_frames_target)


def iter_preview_panels(h5_path: Path, clip_s: float,
                      project_cams, gel_center_left, gel_center_right,
                      dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      proj_up_axis: str = "y",
                      press_axes=None, window_start: int | None = None,
                      force_root: Path | None = None,
                      frame_transform=None, show_virtual_targets: bool = True):
    """Yield aligned BGR panels; source resources close with the generator."""
    force_root = FORCE_ROOT if force_root is None else Path(force_root)
    n_frames_target = int(round(clip_s * SOURCE_FPS))   # e.g. 30s * 30fps = 900
    task_name = h5_path.parent.parent.name
    date_name = h5_path.parent.name
    # Trim: the RELEASE parquet is the authority (same source the force rows
    # already use). `_get_trim_offset` reads a processed/.pt sidecar that
    # pushT never had, and "falls back to 0" — so pushT previews played the
    # H5 pre-roll: up to 6.5 min BEFORE the episode, sensors parked, action
    # frozen. A silent zero fallback is indistinguishable from the correct
    # answer on every recording that doesn't need one.
    # h5_path.stem is the SOURCE recording -- `episode_003` for every segment
    # cut out of it -- and that is the right key here: the force npz, the
    # release poses and `bad_frames.json` are all stored per recording and
    # indexed through this trim. What it must NOT decide is where the clip
    # starts; `window_start` carries that, because a segment's start is the
    # one thing its name has and its source's does not.
    trim_offset, n_rows = _parquet_trim_and_rows(task_name, date_name,
                                                h5_path.stem)
    trim_pq = trim_offset
    flagged = _flagged_intervals(task_name, date_name, h5_path.stem)
    with h5py.File(str(h5_path), "r") as f:
        cam_ts = f["timestamps"][:]
        T_h5 = len(cam_ts)
        # First 30s of the PUBLISHED episode (a segment starts part way in)
        start, end = clip_window(trim_offset, window_start, T_h5,
                                 n_frames_target)
        sample_idx = np.arange(start, end)

        # Pre-load OT once
        ot_lookup = load_optitrack(f)
        _apply_world_offset(ot_lookup, dx, dy, dz)

        gel_lag = gel_lag_frames(f)
        n_gel = len(f["gelsight/left/frames"])
        # Which raw gel frame each published row was computed from. `gel_lag`
        # still applies on legacy recordings, where the raw stream itself is
        # shifted; the two corrections are independent and compose.
        gel_src = _gel_source_frames(task_name, date_name, h5_path.stem, force_root)
        if gel_src:
            print(f"    gel frames follow force source_frame "
                  f"({'/'.join(sorted(gel_src))})", flush=True)
        gel_at = lambda side, r, i: gel_frame(  # noqa: E731
            gel_src, side, r, int(i) + gel_lag, n_gel)
        gs_ref_L = preview_reference(f, "left", task_name, date_name,
                                     h5_path.stem)
        gs_ref_R = preview_reference(f, "right", task_name, date_name,
                                     h5_path.stem)

        calib_note = calib_describe(task_name, date_name, h5_path.stem)
        forces = load_forces(task_name, date_name, h5_path.stem, force_root)
        # The DexForce virtual target, row-aligned with `forces` by
        # construction (same arrays, same indices) rather than by a second
        # lookup that has to agree with the first.
        targets = preview_targets(task_name, date_name, h5_path.stem, force_root,
                                  enabled=show_virtual_targets)

        wrist_slots = sorted(f["arducam"])[:2] if "arducam" in f else []
        wrist_labels = [{"cam0": "wrist left", "cam1": "wrist right"}.get(slot, slot)
                        for slot in wrist_slots] or None
        wrist_gamma = episode_wrist_gamma(f) if wrist_slots else 1.0
        overlay_errors = 0
        for f_idx_int in sample_idx:
            f_idx_int = int(f_idx_int)
            color_frames = [
                f[f"realsense/cam{cam_idx}/color"][f_idx_int]
                for cam_idx in range(3)
            ]
            gel_row = row_for_h5_frame(f_idx_int, trim_pq, n_rows)
            gs_L = f["gelsight/left/frames"][gel_at("left", gel_row, f_idx_int)]
            gs_R = f["gelsight/right/frames"][gel_at("right", gel_row, f_idx_int)]

            opt_poses = optitrack_at(ot_lookup, float(cam_ts[f_idx_int]))

            wrist = None
            if wrist_slots:
                # Same tick index as everything else, so the wrist row is in
                # step with the views and the tactile beside it.
                # The published curve, so the preview panel matches the
                # wrist videos the dataset ships. See `twm.wrist_tone`.
                wrist = [apply_tone_curve(
                             decode_arducam(f[f"arducam/{slot}/frames"][f_idx_int]), wrist_gamma)
                         for slot in wrist_slots]
            panel = build_preview_panel(
                color_frames=color_frames,
                gs_frames=[gs_L, gs_R],
                arducam_frames=wrist,
                arducam_labels=wrist_labels,
                gs_ref=[gs_ref_L, gs_ref_R],
                optitrack_poses=opt_poses,
                recording=False,
                frame_count=f_idx_int,
                elapsed=float(cam_ts[f_idx_int] - cam_ts[0]),
                fps=30.0,
                task_name=task_name,
                status_override=(
                    f"[{task_name}] {h5_path.parent.name}/{h5_path.stem}  "
                    + (f"gel+{gel_lag}f  " if gel_lag else "")
                    # Which calibration epoch and world offset produced the
                    # projection, in frame. A viewer who can see "calib
                    # 2026-05-12" can catch the wrong epoch; a viewer who
                    # cannot has to trust that nobody pointed the builder at
                    # the wrong directory, which is exactly what happened.
                    + f"{calib_note}  "
                    + f"H5 frame {f_idx_int}/{T_h5}  "
                    f"({float(cam_ts[f_idx_int] - cam_ts[0]):.1f}s)"
                ),
            )

            # If the release catalogues this frame as bad (bad_frames.json,
            # episode coords), say so ON the frame. The curation excludes
            # these spans from the training segments, but the preview plays
            # the raw stream — an unlabelled OT freeze or tactile spike here
            # reads as dirty data, and got reported as exactly that.
            ep_f = f_idx_int - trim_offset
            tags = sorted({k for a, b, k in flagged if a <= ep_f <= b})
            if tags:
                cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1),
                              (0, 0, 255), 2)
                cv2.putText(panel, "FLAGGED " + ",".join(tags)
                            + "  (excluded from training segments)",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255), 2)

            # Force for THIS frame, per side, resolved before drawing: the
            # disc is rendered inside draw_projection_overlay so it lands on
            # the same projected sensor centre as the pose axes.
            # ONE ROW LOOKUP FOR BOTH. The disc and the virtual target are two
            # views of the same row; resolving them separately is how the disc
            # ended up half a second from its own tactile tile.
            frame_forces, frame_targets = {}, {}
            if forces and trim_pq is not None:
                row = row_for_h5_frame(f_idx_int, trim_pq, n_rows)
                if row is not None:
                    frame_forces = {s: float(a[row]) for s, a in forces.items()
                                    if row < len(a)}
                    frame_targets = {s: a[row] for s, a in targets.items()
                                     if row < len(a)}

            if project_cams:
                try:
                    draw_preview_overlay(panel, opt_poses, Projection(
                        project_cams,
                        gel_center_left, gel_center_right,
                        forces_n=frame_forces or None,
                        targets_7=frame_targets or None,
                        press_axis=press_axes,
                    ))
                except Exception as e:
                    # Was `except Exception: pass`. A frame that failed to
                    # draw its overlay still encoded fine and looked
                    # deliberate — a clip could lose every disc and every
                    # axis without anyone learning why. Report once per
                    # episode; still don't abort, since one bad pose should
                    # not cost the whole preview.
                    if not overlay_errors:
                        print(f"  WARN: overlay failed at frame {f_idx_int}: "
                              f"{type(e).__name__}: {e}")
                    overlay_errors += 1
            elif frame_forces:
                # No calibration -> no sensor position to attach the disc to.
                # Say so rather than dropping the force silently.
                cv2.putText(panel, "force: no cam calibration", (8, 232),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 120, 255), 1,
                            cv2.LINE_AA)

            if project_cams:
                draw_world_gizmo_on_panel(panel, project_cams,
                                          proj_up_axis)

            if frame_forces:
                draw_legend(panel, 4 * GS_THUMB_W + 26, ROW2_Y + 96)

            if frame_transform is not None:
                panel = frame_transform(panel, f_idx_int)
            yield panel

    if overlay_errors:
        print(f"  WARN: {overlay_errors}/{len(sample_idx)} frames "
              f"rendered without the projection/force overlay")


def build_one_preview(h5_path: Path, out_mp4: Path,
                      clip_s: float, speed: float,
                      project_cams, gel_center_left, gel_center_right,
                      dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      proj_up_axis: str = "y", press_axes=None,
                      window_start: int | None = None, force_root: Path | None = None,
                      frame_transform=None, show_virtual_targets: bool = True) -> None:
    """Stream panels to a verified MP4, replaying source only on verification retry.

    ``frame_transform`` must be deterministic if supplied: a failed decode check
    starts a fresh source pass. Existing destination remains intact on failure.
    """
    from twm.visualization.export import write_video

    if not np.isfinite(clip_s) or clip_s <= 0:
        raise ValueError("clip_s must be finite and positive")
    frames = partial(
        iter_preview_panels, h5_path, clip_s, project_cams,
        gel_center_left, gel_center_right, dx=dx, dy=dy, dz=dz,
        proj_up_axis=proj_up_axis, press_axes=press_axes,
        window_start=window_start, force_root=force_root,
        frame_transform=frame_transform, show_virtual_targets=show_virtual_targets,
    )
    write_video(out_mp4, frames, fps=SOURCE_FPS * speed)


def _decoded_frame_count(mp4: Path) -> int:
    """Compatibility alias for the shared full-decode verification."""
    from twm.visualization.export import decoded_frame_count
    return decoded_frame_count(mp4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="motherboard",
                    help="task folder under data/ and episode_previews/")
    ap.add_argument("--date", required=True,
                    help="Date subfolder under H5_ROOT, e.g. 2026-05-19")
    ap.add_argument("--clip_s", type=float, default=CLIP_S_DEFAULT,
                    help=f"Seconds of usable (post-trim) data to sample (default {CLIP_S_DEFAULT})")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT,
                    help=f"Playback speed (default {SPEED_DEFAULT}x; output_fps = "
                         f"source_fps * speed)")
    ap.add_argument("--dx", type=float, default=0.0,
                    help="World-frame X offset (m) added to every OT pose before projection.")
    ap.add_argument("--dy", type=float, default=0.0,
                    help="World-frame Y offset (m) added to every OT pose before projection.")
    ap.add_argument("--dz", type=float, default=0.0,
                    help="World-frame Z offset (m) added to every OT pose before projection.")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="Optional list of episode_NNN stems to process. "
                         "Defaults to all episode_*.h5 under <date>/.")
    args = ap.parse_args()

    global TASK, H5_ROOT, OUT_ROOT, EPISODES_ROOT
    TASK = args.task
    H5_ROOT = Path("/media/yxma/Disk1/twm/data") / TASK
    OUT_ROOT = _preview_root(TASK)
    EPISODES_ROOT = Path("/media/yxma/Disk1/twm/processed/episodes") / TASK

    h5_dir = H5_ROOT / args.date
    if not h5_dir.is_dir():
        print(f"No such date: {h5_dir}", file=sys.stderr); sys.exit(1)
    h5_files = sorted(h5_dir.glob("episode_*.h5"))
    if args.episodes:
        wanted = set(args.episodes)
        h5_files = [p for p in h5_files if p.stem in wanted]
    if not h5_files:
        print(f"No episodes selected.", file=sys.stderr); sys.exit(1)

    print(f"Building previews for {len(h5_files)} episode(s) in {args.date} "
          f"(first {args.clip_s:.0f}s of post-trim data @ {args.speed:.1f}x speed -> "
          f"{args.clip_s / args.speed:.0f}s output)", flush=True)

    project_cams, glc, grc, proj_up_axis = _load_proj_calibs(args.task, args.date)
    # The direction GelSight Mini's normal force acts along, in the SENSOR's
    # own frame. Task-specific because each rig has its own gel calibration.
    from force_recovery.dexforce import gel_axis as _gax
    try:
        press_axes = {s: _gax(args.task, s) for s in ("left", "right")}
    except Exception as e:
        print(f"  WARN: no gel axis for {args.task} ({type(e).__name__}); "
              f"the force-direction arrow will be omitted")
        press_axes = None
    if project_cams:
        print(f"  projection overlay: ON ({len(project_cams)} cameras)", flush=True)

    out_dir = OUT_ROOT / args.date
    for h5 in h5_files:
        out_mp4 = out_dir / f"{h5.stem}.mp4"
        print(f"  {h5.stem}: ", end="", flush=True)
        # Per-episode world offset from the release's episodes.jsonl. The
        # --dx/--dy/--dz flags remain as a manual override for recordings the
        # release does not list; they are NOT the source of truth, because a
        # flag that defaults to zero silently mis-renders the one date that
        # needs it.
        try:
            dx, dy, dz = (args.dx, args.dy, args.dz)
            if (dx, dy, dz) == (0.0, 0.0, 0.0):
                dx, dy, dz = world_offset_m(args.task, args.date, h5.stem, up_axis="y")
        except KeyError as e:
            print(f"SKIP ({e})", flush=True)
            continue
        try:
            build_one_preview(h5, out_mp4, args.clip_s, args.speed,
                              project_cams, glc, grc,
                              dx=dx, dy=dy, dz=dz,
                              proj_up_axis=proj_up_axis,
                              press_axes=press_axes)
            print(f"OK  -> {out_mp4.relative_to(OUT_ROOT.parent.parent.parent)}  "
                  f"({out_mp4.stat().st_size / 1024:.0f} KB)", flush=True)
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
