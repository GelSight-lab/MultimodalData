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
