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
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent
RELEASE = Path("/media/yxma/Disk1/twm/release")

# Which on-disk directory holds each task's epoch. The mapping is the thing
# worth naming; the directory names themselves are historical accidents.
# Verified equal to the published `data/<task>/calibration/` for both tasks.
CALIB_DIRS = {
    "motherboard": REPO / "calibration" / "result backup",   # May-12 epoch
    "pushT":       REPO / "calibration" / "result",          # June-26 epoch
}
EXPECTED_EPOCH = {"motherboard": "2026-05-12", "pushT": "2026-06-26"}


def calib_dir(task: str) -> Path:
    """Directory of camera extrinsics valid for `task`.

    Raises on an unknown task rather than falling back to a default: a wrong
    calibration does not look wrong, it looks like a slightly miscalibrated
    rig, which is exactly how this shipped unnoticed.
    """
    try:
        d = CALIB_DIRS[task]
    except KeyError:
        raise KeyError(
            f"no calibration epoch declared for task {task!r}; add it to "
            f"calib_epoch.CALIB_DIRS — do not fall back to another task's "
            f"extrinsics") from None
    if not d.is_dir():
        raise FileNotFoundError(f"calibration dir for {task!r} missing: {d}")
    return d


def calib_dir_for_path(p: str | Path) -> Path:
    """Epoch dir inferred from any path containing a task-name component.

    For the interactive viewers, whose old default was `calibration/result`
    for every input — June-26 extrinsics under May recordings. No guess on
    failure: raises, listing the known tasks, so the caller passes explicit
    calibration paths instead of silently viewing through the wrong epoch.
    """
    parts = set(Path(p).parts) | set(Path(p).resolve().parts)
    hits = [t for t in CALIB_DIRS if t in parts]
    if len(hits) == 1:
        return calib_dir(hits[0])
    raise KeyError(
        f"cannot infer task from {str(p)!r} (matches: {hits or 'none'}); pass "
        f"explicit --cam_calib/--gel_* paths. Known tasks: {sorted(CALIB_DIRS)}")


def epoch_of(task: str) -> str:
    """The calibration date actually on disk for `task` (from the files)."""
    p = calib_dir(task) / "T_mocap_to_cam_middle.json"
    created = json.loads(p.read_text()).get("created_at") or ""
    return created[:10]


def check_epoch(task: str) -> None:
    """Fail loudly if the directory does not hold the epoch it should."""
    got, want = epoch_of(task), EXPECTED_EPOCH.get(task)
    if want and got != want:
        raise ValueError(
            f"{task}: calibration dir holds the {got} epoch, expected {want} "
            f"({calib_dir(task)}). Using another epoch's extrinsics puts the "
            f"projected sensor off the sensor.")


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


def world_offset_m(task: str, date: str, episode: str) -> tuple[float, float, float]:
    """Offset to ADD to raw-H5 OptiTrack poses to reach the release frame.

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
            f"2026-05-19 is offset (0.23, 0, 0.175) m and would render wrong.")
    off = eps[key].get("world_frame_offset") or (0.0, 0.0, 0.0)
    return (float(off[0]), float(off[1]), float(off[2]))


def describe(task: str, date: str, episode: str) -> str:
    """One line for a status bar, so the applied correction is visible."""
    dx, dy, dz = world_offset_m(task, date, episode)
    s = f"calib {epoch_of(task)}"
    if any((dx, dy, dz)):
        s += f" world+({dx:g},{dy:g},{dz:g})m"
    return s

# ── the world frame each session is in, as a FULL RIGID TRANSFORM ───────────
#
# WHY THIS REPLACES A BARE OFFSET.
#
# Re-running an OptiTrack calibration changes the world frame by a rigid
# transform, rotation included. The release corrected 2026-05-19 with a
# TRANSLATION ONLY, (230, 0, 175) mm, so its rotation went uncorrected — and
# every self-consistency check in this project is invariant to a shared world
# transform, which is why none of them caught it. A three-number offset cannot
# express the thing it is correcting.
#
# REFERENCE FRAME: 2026-05-10. Named, so "aligned" has a referent. Every other
# session declares the transform onto it, and future motherboard sessions must
# do the same rather than adding a fourth convention.
#
# HOW IT WAS MEASURED. The board lies on the same physical table every session,
# so the table normal in world coordinates is an invariant:
#
#     n_date = median over frames of  R_obj @ n_local
#     n_local = smallest singular vector of the contact cloud (force > 2 N)
#               expressed in the board's OWN frame
#
# The board frame matters: the board is picked up and tilted (median 3.3-4.2
# deg off its own median), so fitting a plane to WORLD contacts measures the
# board's average pose, not the table. Two attempts at that gave 3.24 and
# 3.76 deg between 05-10 and 05-11 — dates the board-frame method puts at
# 0.29 deg — and both were discarded.
#
#     05-11 vs 05-10   0.29 deg   <- the reproducibility floor
#     05-19 vs 05-10   3.38 deg
#
# stable across quiet-frame thresholds (3.41 / 3.49 / 3.44 / 3.24 / 2.96 deg
# for the quietest 100 / 50 / 25 / 10 / 5 %).
#
# WHAT IS NOT DETERMINED, AND IS THEREFORE NOT INVENTED HERE. A plane normal
# fixes two rotational degrees of freedom. Yaw about that normal, and the two
# in-plane translations, are invisible to it — verified: composing the fix with
# any spin about the normal still aligns the normals exactly. The height along
# the normal is measurable but marginal (+3.8 mm against a 1.5 mm floor), so it
# is recorded and NOT applied.
#
# PIVOT. The rotation is applied about the WORKSPACE CENTROID, not the mocap
# origin. Both are valid rigid transforms and they differ by a translation —
# the one degree of freedom that is undetermined. Pivoting where the data
# actually is leaves the in-plane position of the workspace unchanged, so the
# correction alters orientation without moving what was already right.
WORLD_REF_DATE = "2026-05-10"

WORLD_TRANSFORM = {
    # date: (rotation vector in DEGREES onto the reference frame,
    #        pivot in that session's own world mm, note)
    # Pivot is the MEDIAN CONTACT POSITION IN WORLD MILLIMETRES. My first
    # value was [363.1, 9.0, -362.2], lifted from a printout where those were
    # projections onto a rotated basis (e1, e2, n) rather than world xyz — the
    # z sign was even flipped. It moved the workspace 41 mm, which the
    # "does not move the workspace" check caught.
    "2026-05-19": ([3.376, -0.017, -0.118], [363.2, 22.8, 364.0],
                   "tilt only; yaw and in-plane translation undetermined"),
}
# Sessions absent from the table are already in the reference frame to within
# the 0.29 deg / 1.5 mm floor and get the identity.

WORLD_RESIDUAL = {
    "2026-05-19": {"tilt_deg": 0.29, "height_mm": 3.8,
                   "yaw_deg": None, "in_plane_mm": None},
}


def world_transform(task: str, date: str):
    """(R, t) taking `date`'s world frame onto WORLD_REF_DATE's, in mm.

    Returns numpy arrays; the identity for any session not in the table. Apply
    as `p_ref = R @ p_session + t`, positions in MILLIMETRES, and rotate the
    orientation by the same R.
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
    return dict(WORLD_RESIDUAL.get(date, {"tilt_deg": 0.29, "height_mm": 1.5,
                                          "yaw_deg": None, "in_plane_mm": None}))

