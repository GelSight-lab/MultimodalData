"""THE definition of which calibration and which world frame a recording uses.

Why this module exists
----------------------
Two facts about the rig change between recordings, and both were hard-coded to
a single value by every consumer:

1. **Camera extrinsics were recalibrated between tasks.** The dataset says so
   itself (yxma/React README, "Calibration epochs"):
       motherboard -> May-12 extrinsics
       pushT       -> June-26 extrinsics
   The preview builder pointed at one directory, `calibration/result`, which
   holds the **June-26** files. pushT was right by luck; every motherboard
   preview was published with the wrong camera pose. Measured between the two
   epochs: |dT| = 53-64 mm, dR = 2.6-6.0 deg per camera, which lands the
   projected sensor marker 35-73 px from the sensor in a 640x480 view —
   visibly wrong, but shaped exactly like a slightly miscalibrated rig, not
   like a bug. The May-12 files were on disk the whole time — in a directory
   named `result backup`, a name that reads like a discardable copy rather
   than "the epoch the motherboard task requires".

2. **The 2026-05-19 session redefined the world origin.** Its poses are
   offset (0.23, 0, 0.175) m from every other date. The published release has
   that baked in; the **raw H5 does not**. Measured on episode_002:
       median(release - rawH5) = (+0.230, +0.000, +0.175) m   [2026-05-19]
       median(release - rawH5) = (+0.000, +0.000, +0.000) m   [2026-05-11]
   Anything reading poses out of the H5 must add it, and the preview builder
   applied 0 — so 05-19 carried BOTH errors at once, which is why it looked
   worst and got reported first.

The failure mode is the one `tactile_align` was written for, repeated: a fact
about the DATA lived in the CALLERS, so every new consumer had to rediscover
it and one silently did not. Same remedy — one module, imported everywhere,
reading the dataset's own declaration rather than restating it.

The offsets are NOT restated here: `episodes.jsonl` records
`world_frame_offset` per episode, and that file ships with the release. This
module reads it. A constant typed here would be a fourth copy of a number the
dataset already publishes.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
RELEASE = Path("/media/yxma/Disk1/twm/release")

# Which on-disk directory holds each task's epoch. The mapping is the thing
# worth naming; the directory names themselves are historical accidents.
# Verified equal to the published `data/<task>/calibration/` for both tasks.
CALIB_DIRS = {
    "motherboard": REPO / "calibration" / "epoch_2026-05-12",
    "pushT":       REPO / "calibration" / "epoch_2026-06-26",
}
EXPECTED_EPOCH = {"motherboard": "2026-05-12", "pushT": "2026-06-26"}

# The epochs themselves, named by the date each was MEASURED.
EPOCH_DIRS = {
    "2026-05-12": REPO / "calibration" / "epoch_2026-05-12",
    "2026-06-26": REPO / "calibration" / "epoch_2026-06-26",
    "2026-09-09": REPO / "calibration" / "epoch_2026-09-09",
}

# Which epoch each RECORDING SESSION belongs to. `task -> epoch` cannot answer
# this. The cameras were recalibrated between sessions, and a calibration is
# measured near its sessions rather than before them: the May recordings are
# covered by the May-12 measurement, pushT's 2026-06-18 by the June-26 one.
# Date order therefore does not determine the answer, so a session declares
# its epoch and nothing infers it.
#
# The 2026-09-09 motherboard session declares the epoch measured that same
# day at 07:41, by the operator's decision. Recorded on: the recorder's live
# overlay ran on the June-26 epoch during collection, and projecting each
# GelSight centre into all three views on a mid-episode frame puts June-26 on
# the sensors while 2026-09-09 sits 20-40 px away and May-12 30-60 px away.
# Both solves fit their own points equally well (0.5-0.7 px RMSE) and share an
# identical gel-to-rigid transform, so the two disagree only about where the
# cameras were. Re-check with the projection comparison if the previews ever
# look off.
CALIB_SESSIONS = {
    ("motherboard", "2026-05-10"): "2026-05-12",
    ("motherboard", "2026-05-11"): "2026-05-12",
    ("motherboard", "2026-05-19"): "2026-05-12",
    ("motherboard", "2026-09-09"): "2026-09-09",
    ("pushT",       "2026-06-18"): "2026-06-26",
    # The 2026-09-10 onward sessions -- the week `main` carries -- are covered
    # by the 2026-09-09 measurement, by the operator's decision: the cameras
    # were not moved between it and them. Declared here rather than inferred
    # from the date, because date order does not determine the answer (pushT's
    # 2026-06-18 belongs to the June-26 solve, measured eight days later).
    ("motherboard", "2026-09-11"): "2026-09-09",
    ("motherboard", "2026-09-12"): "2026-09-09",
    ("pushT",       "2026-09-10"): "2026-09-09",
    ("pushT",       "2026-09-11"): "2026-09-09",
    ("pushT",       "2026-09-12"): "2026-09-09",
}


def session_epoch(task: str, date: str) -> str:
    """The epoch a recording session declares.

    Raises for an undeclared session rather than falling back to the task
    default: the fallback is exactly how a session ships through another
    session's extrinsics with nobody noticing.
    """
    try:
        return CALIB_SESSIONS[(task, date)]
    except KeyError:
        known = sorted(d for t, d in CALIB_SESSIONS if t == task)
        raise KeyError(
            f"{task}: session {date!r} does not declare a calibration epoch. "
            f"Declared sessions: {known}. Add it to calib_epoch.CALIB_SESSIONS "
            f"— do not fall back to another session's extrinsics.") from None



def calib_dir(task: str, *, date: str | None = None,
              up_axis: str | None = None) -> Path:
    """Directory of camera extrinsics valid for `task` — and for `date`,
    when the caller knows which session it is reading (`CALIB_SESSIONS`).

    Looked up in this order, so the module works outside the repository it
    lives in:

        1. $REACT_CALIB                 explicit override
        2. $REACT_RELEASE/<task>/calibration   the PUBLISHED layout
        3. CALIB_DIRS[task]             this repo's working layout

    Until (2) existed, `calib_dir` resolved a path relative to this file, so a
    clean-room run of `build_probe_testset.py` died on
    "calibration dir for 'motherboard' missing: .../calibration/result backup"
    — a directory that exists on exactly one machine. The dataset publishes the
    same files under `data/<task>/calibration/`; nothing was missing except the
    lookup.

    Raises on an unknown task rather than falling back to another task's
    extrinsics: a wrong calibration does not look wrong, it looks like a
    slightly miscalibrated rig, which is how it shipped unnoticed once.
    """
    import os as _os

    def _ok(d: Path) -> Path:
        """Refuse a tree in the wrong convention instead of returning it.

        This resolves $REACT_RELEASE before the repo's own tree, and the
        release is now Z-up while every raw-HDF5 reader is Y-up. Returning a
        path cannot convert anything, so the only honest options are the right
        tree or an error -- not a 200 px picture that looks plausible.
        """
        if up_axis is None:
            return d
        if up_axis not in ("y", "z"):
            raise ValueError(f"up axis must be 'y' or 'z', got {up_axis!r}")
        f = d / "T_mocap_to_cam_middle.json"
        got = "y"
        if f.exists():
            got = json.loads(f.read_text()).get("up_axis") or "y"
        if got != up_axis:
            raise ValueError(
                f"{d} is a {got}-up calibration but the caller needs "
                f"{up_axis}-up. Raw HDF5 poses are Y-up as recorded; the "
                f"published release is Z-up. Point REACT_CALIB at a {up_axis}"
                f"-up tree, or convert with react_toolbox.frames.as_up_axis "
                f"after loading.")
        return d

    env = _os.environ.get("REACT_CALIB")
    if env and Path(env).is_dir():
        return _ok(Path(env))
    rel = _os.environ.get("REACT_RELEASE")
    if rel:
        cand = Path(rel) / task / "calibration"
        if cand.is_dir():
            return _ok(cand)
    if date is not None:
        d = EPOCH_DIRS[session_epoch(task, date)]
        if not d.is_dir():
            raise FileNotFoundError(
                f"calibration dir for the {session_epoch(task, date)} epoch "
                f"({task}/{date}) is missing: {d}")
        return _ok(d)
    try:
        d = CALIB_DIRS[task]
    except KeyError:
        raise KeyError(
            f"no calibration epoch declared for task {task!r}; add it to "
            f"calib_epoch.CALIB_DIRS — do not fall back to another task's "
            f"extrinsics") from None
    if not d.is_dir():
        raise FileNotFoundError(
            f"calibration dir for {task!r} missing: {d}. Set REACT_CALIB, or "
            f"REACT_RELEASE so that $REACT_RELEASE/{task}/calibration exists "
            f"(the dataset publishes it there).")
    return _ok(d)


_DATE_PART = re.compile(r"\d{4}-\d{2}-\d{2}")


def calib_dir_for_path(p: str | Path, *, up_axis: str = "y") -> Path:
    """Epoch dir inferred from any path containing a task-name component.

    Defaults to `up_axis="y"` because every caller is an interactive viewer
    reading poses straight out of the source HDF5, which is Y-up as recorded.
    If $REACT_RELEASE points calib_dir at the Z-up release, this raises rather
    than viewing through a 200 px error.

    For the interactive viewers, whose old default was `calibration/result`
    for every input — June-26 extrinsics under May recordings. No guess on
    failure: raises, listing the known tasks, so the caller passes explicit
    calibration paths instead of silently viewing through the wrong epoch.
    """
    all_parts = list(Path(p).parts) + list(Path(p).resolve().parts)
    parts = set(all_parts)
    hits = [t for t in CALIB_DIRS if t in parts]
    if len(hits) == 1:
        # A recording path carries its session date as a component
        # (<task>/<YYYY-MM-DD>/episode_NNN.h5). Use it: the epoch is per
        # session, and the task default is the wrong answer for any session
        # recorded after the rig was recalibrated.
        dates = [q for q in all_parts if _DATE_PART.fullmatch(q)]
        return calib_dir(hits[0], date=dates[0] if dates else None, up_axis=up_axis)
    raise KeyError(
        f"cannot infer task from {str(p)!r} (matches: {hits or 'none'}); pass "
        f"explicit --cam_calib/--gel_* paths. Known tasks: {sorted(CALIB_DIRS)}")


def epoch_of(task: str, date: str | None = None) -> str:
    """The calibration date actually on disk for `task` (from the files)."""
    p = calib_dir(task, date=date) / "T_mocap_to_cam_middle.json"
    created = json.loads(p.read_text()).get("created_at") or ""
    return created[:10]


def check_epoch(task: str, date: str | None = None) -> None:
    """Fail loudly if the directory does not hold the epoch it should.

    With a `date` it is the SESSION's declared epoch that must be there;
    without one, the task default.
    """
    got = epoch_of(task, date=date)
    want = session_epoch(task, date) if date is not None else EXPECTED_EPOCH.get(task)
    if want and got != want:
        where = f"{task}/{date}" if date is not None else task
        raise ValueError(
            f"{where}: calibration dir holds the {got} epoch, expected {want} "
            f"({calib_dir(task, date=date)}). Using another epoch's extrinsics "
            f"puts the projected sensor off the sensor.")


@lru_cache(maxsize=None)
def _episodes(task: str) -> dict:
    """`episodes.jsonl` keyed by its own `episode` field, `<date>/<episode>`."""
    p = RELEASE / task / "episodes.jsonl"
    if not p.exists():
        return {}
    return {r["episode"]: r for r in
            (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def release_episodes(task: str) -> set[str]:
    """`<date>/<episode>` keys the release actually publishes.

    The previews mirror the release; a preview of an episode the release does
    not contain is not "extra material", it is a claim that the episode
    exists. Four such clips (2026-03-23, 2026-05-15) sat on the dataset for
    months and were reported as "not updated" — they could never be updated,
    because there was nothing in the release to update them from.
    """
    return set(_episodes(task))


def world_offset_m(task: str, date: str, episode: str, *,
                   up_axis: str) -> tuple[float, float, float]:
    """Offset to ADD to poses to reach the release frame, in `up_axis`.

    `up_axis` is REQUIRED and has no default on purpose. The value is stored
    Z-up because the release is, but the documented use -- adding it to a pose
    read straight out of the source H5 -- is Y-up. A default would have been
    right for one caller and silently 175 mm wrong on the wrong axis for the
    other, which is how this class of bug got in.

    Read from the release's own `episodes.jsonl` (`world_frame_offset`), never
    restated. `episode` may be `episode_002` or `2026-05-19/episode_002`.

    An episode the release does not list RAISES. The first version returned
    (0, 0, 0) on a miss and its key was wrong — `episodes.jsonl` prefixes the
    date — so it silently reported "no shift" for the one date that has one.
    A default that is indistinguishable from the correct answer for every
    other recording is not a safe default; it is a silent failure.
    """
    key = episode if "/" in episode else f"{date}/{episode}"
    eps = _episodes(task)
    if not eps:
        return (0.0, 0.0, 0.0)                # no release tree; nothing to align to
    if key not in eps:
        raise KeyError(
            f"{task}: {key!r} is not in {RELEASE / task / 'episodes.jsonl'}, so "
            f"its world-frame offset is unknown. Refusing to assume zero — "
            f"2026-05-19 is offset (0.23, -0.175, 0) m Z-up and would "
            f"render wrong.")
    off = eps[key].get("world_frame_offset") or (0.0, 0.0, 0.0)
    stored = eps[key].get("up_axis") or "y"
    off = np.asarray([float(off[0]), float(off[1]), float(off[2])], float)
    if stored != up_axis:
        from .frames import YUP_TO_ZUP
        M = np.asarray(YUP_TO_ZUP, float)
        off = (M if up_axis == "z" else M.T) @ off
    return (float(off[0]), float(off[1]), float(off[2]))


def describe(task: str, date: str, episode: str) -> str:
    """One line for a status bar, so the applied correction is visible."""
    # a status line for the raw-H5 render paths, so: the Y-up convention
    dx, dy, dz = world_offset_m(task, date, episode, up_axis="y")
    # The session's epoch, not the task default: this line exists so a viewer
    # can catch a wrong epoch without trusting the pipeline, and a label that
    # names a different epoch than the one that drew the axes is worse than
    # no label at all.
    s = f"calib {epoch_of(task, date)}"
    if any((dx, dy, dz)):
        s += f" world+({dx:g},{dy:g},{dz:g})m"
    return s

# ── the world frame each session is in, as a FULL RIGID TRANSFORM ───────────
#
# WHAT THE RIG PHYSICALLY ALLOWS, which constrains this more than any fit.
#
# The OptiTrack ground plane is set with an L-bracket laid on the table, so the
# calibration defines +y as the normal of the SAME physical plane every time.
# Two calibrations of that rig can therefore differ only by
#
#     yaw          rotation about y (the plane normal)
#     in-plane     translation along x and z
#
# and NOT by a tilt about x or z. That is a property of the procedure, not an
# assumption about the data.
#
# A TILT I MEASURED, AND WHY IT WAS WRONG.
#
# Comparing the board's plane normal across sessions gave 3.38 deg for
# 2026-05-19, almost all of it about x — a rotation the procedure forbids. The
# measurement, not the constraint, was at fault: 2026-05-19's contact cloud is
# not planar. Split by in-plane position, its two halves give normals 6.35 deg
# apart, and one half agrees with the reference to 0.77 deg while the other is
# 6.61 deg off. Bootstrap jitter is 0.15 deg, so this is real spatial
# inhomogeneity — that session's contacts sample the board's relief unevenly —
# not sampling noise. The whole-cloud 3.38 deg was an average of two
# inconsistent halves. The reference sessions split to 0.56-1.52 deg, which is
# why the artefact showed up only here.
#
# So no tilt is applied. The reference date remains 2026-05-10; every other
# session declares the transform onto it, and future motherboard sessions must
# do the same rather than adding a fourth convention.
WORLD_REF_DATE = "2026-05-10"

WORLD_TRANSFORM = {
    # date: (rotation vector in DEGREES onto the reference frame,
    #        pivot in that session's own world mm, note)
    #
    # Empty: the only rotation the rig can produce is yaw about y, and the yaw
    # measured for 2026-05-19 (+2.41 deg) is not separable from the in-plane
    # translation by the evidence available — see WORLD_RESIDUAL. A number that
    # cannot be separated from another number is not a correction.
}

WORLD_RESIDUAL = {
    "2026-05-19": {
        "tilt_deg": 0.0,
        "tilt_note": "forbidden by the L-bracket procedure; the 3.38 deg once "
                     "measured here was an artefact of a non-planar contact "
                     "cloud (in-plane halves 6.35 deg apart, vs 0.56 deg on the "
                     "reference)",
        "yaw_deg": None, "yaw_applied": False,
        "yaw_note": "NOT MEASURED. Matching the tracked board's projected "
                    "outline against the board segmented in a camera gave "
                    "+2.41 deg with a 95% frame-bootstrap of [0.36, 3.15], but "
                    "that interval covered only frame-to-frame noise. Varying "
                    "the camera and the hull clipping moves 2026-05-19 over "
                    "-1.84 to +3.04 deg — and moves the REFERENCE date, which "
                    "is zero by construction, over -1.83 to +3.28 deg. The "
                    "method's systematic scatter is the size of the effect, so "
                    "it has no power and the number was withdrawn.",
        "in_plane_mm": None,
        "status": "the (230, 0, 175) mm translation the release already applies "
                  "is the only correction; how good it is remains unmeasured",
    },
}


def world_transform(task: str, date: str):
    """(R, t) taking `date`'s world frame onto WORLD_REF_DATE's, in mm.

    Currently the identity for every session: the only rotation the rig can
    produce is yaw about y, and that yaw is not yet separable from the in-plane
    translation. `world_residual` states what is known and what is not.
    """
    import numpy as _np
    from scipy.spatial.transform import Rotation as _R
    if task != "motherboard" or date not in WORLD_TRANSFORM:
        return _np.eye(3), _np.zeros(3)
    rv, pivot, _ = WORLD_TRANSFORM[date]
    R = _R.from_rotvec(_np.radians(_np.asarray(rv, float))).as_matrix()
    c = _np.asarray(pivot, float)
    return R, c - R @ c


def world_residual(task: str, date: str) -> dict:
    """What the transform does NOT fix, so a consumer can bound their error."""
    if task != "motherboard":
        return {}
    return dict(WORLD_RESIDUAL.get(date, {"tilt_deg": 0.0, "yaw_deg": 0.0,
                                          "yaw_applied": False,
                                          "in_plane_mm": None}))
