"""Paths, constants and camera mapping for the React preprocessing pipeline.

Everything that used to be a magic number scattered across
``twm/scripts/build_*.py`` lives here.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── frame geometry / encoding ────────────────────────────────────────────────
W, H = 640, 480
FPS = 30.0
CRF = "18"
CHUNK = 128                 # frames per H5 read block (bounds memory)

# ── contact-scalar parameters (must match react_toolbox defaults) ────────────
TAU = 8.0                   # L2 threshold for the contact mask
P01_SMOOTH_WIN = 30         # smoothing window when picking the p01 reference

# ── stream naming ────────────────────────────────────────────────────────────
# H5 cam index -> published stream name (verified against calibration serials:
#   cam0 = 143322063538 -> right, cam1 = 104122062574 -> left,
#   cam2 = 217222066989 -> middle)
CAM_STREAM = {0: "view_right", 1: "view_left", 2: "view_middle"}
GEL_STREAM = {"left": "tactile_left", "right": "tactile_right"}
# The Arducam wrist cameras, by their H5 slot. Named by the side they are
# mounted on, like every other stream, so a reader never has to know that
# cam0 happens to be the left one.
WRIST_STREAM = {"cam0": "wrist_left", "cam1": "wrist_right"}
SIDES = ("left", "right")

# ── world-frame corrections already baked into published poses ───────────────
# The 2026-05-19 motherboard session redefined the OptiTrack origin.
WORLD_OFFSET = {("motherboard", "2026-05-19"): (0.23, 0.0, 0.175)}

# ── locations (overridable by env so the package works off the rig too) ──────
DATA_ROOT = Path(os.environ.get("REACT_DATA_ROOT", "/media/yxma/Disk1/twm/data"))
STAGE_ROOT = Path(os.environ.get("REACT_STAGE_ROOT", "/media/yxma/Disk1/twm/release"))

H5_ROOTS = {
    "motherboard": DATA_ROOT / "motherboard",
    "pushT": DATA_ROOT / "pushT",
    "rope": DATA_ROOT / "rope",
}

# Sessions that predate the multi-camera rig and are not published.
EXCLUDE_DATES = {"2026-03-23"}

# ── which task tracks the manipulated object ─────────────────────────────────
# `object_pose` is optional, and only the motherboard has a rigid body on it.
# The pushT block does not, and nothing else is planned to.
#
# Declared, because the alternative was a search: `_object_pose` tried the
# bodies (task, "object", "motherboard") in turn and took the first one the
# recording happened to contain. Motive was still emitting a `motherboard`
# body during the 2026-09-09 pushT session -- 3 stray samples over 8.9
# minutes, 157 mm apart, a marker cluster mistaken for the board -- so that
# episode shipped with `object_pose` 100% non-NaN, three garbage values
# nearest-neighboured across 13484 rows. Every other pushT episode is all
# NaN, which is the honest way to say "not tracked", and a reader has no way
# to tell the two apart from the column alone.
OBJECT_TRACKED_TASKS = frozenset({"motherboard"})

# The rigid body that IS the object, per task that has one.
OBJECT_BODY = {"motherboard": "motherboard"}

# ── sessions whose SOURCE recordings no longer exist ─────────────────────────
# Deleted 2026-09-09 to make room for re-collection: 1.19 TB of raw HDF5 across
# 38 episodes, by the operator's decision, knowing there was no other copy.
#
# What survives is the DERIVED release (`STAGE_ROOT`, and yxma/React on the
# Hub): H.264/H.265 video, parquet and previews. That is a lossy re-encode, so
# these sessions can still be READ and still be published, but they can never
# be re-processed -- no new encoding, no new overlay after a recalibration, and
# nothing that needs the original pixels.
#
# Declared here, not discovered from an empty directory, because "the files are
# gone" and "you asked for the wrong path" produce the same empty glob and want
# opposite responses from the caller.
# 2026-05-15 is deliberately absent: that directory was removed too, but it
# held no episodes, so it never shipped and there is nothing to explain.
RAW_DELETED = {
    ("motherboard", "2026-05-10"), ("motherboard", "2026-05-11"),
    ("motherboard", "2026-05-19"),
    ("pushT", "2026-06-18"),
}
RAW_DELETED_ON = "2026-09-09"


def deleted_note(task: str, date: str | None = None) -> str | None:
    """Why a source recording is missing, or None if it is not a deleted one.

    Only consulted once a lookup has already come back empty, so it answers
    "is this one of the deleted sessions", not "does this task have data" -- a
    task that lost one session and later gained another still resolves the new
    one normally and never reaches this. With no `date` it names every deleted
    session of the task.
    """
    dates = sorted(d for t, d in RAW_DELETED if t == task)
    if not dates:
        return None
    if date is not None and date not in dates:
        return None
    which = f"{task}/{date}" if date else f"{task} ({', '.join(dates)})"
    return (f"the source HDF5 for {which} was deleted on {RAW_DELETED_ON} to "
            f"make room for re-collection. The derived release survives in "
            f"{STAGE_ROOT} and in {HF_REPO}, but it is a lossy re-encode: it "
            f"can be read and republished, never re-processed.")

HF_REPO = "yxma/React"


def stage_dirs(task: str, date: str, episode: str):
    """Return (video_dir, meta_dir) for one episode in the staging tree."""
    root = STAGE_ROOT / task
    return root / "videos" / date / episode, root / "meta" / date
