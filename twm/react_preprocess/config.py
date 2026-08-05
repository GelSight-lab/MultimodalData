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
}

# Sessions that predate the multi-camera rig and are not published.
EXCLUDE_DATES = {"2026-03-23"}

HF_REPO = "yxma/React"


def stage_dirs(task: str, date: str, episode: str):
    """Return (video_dir, meta_dir) for one episode in the staging tree."""
    root = STAGE_ROOT / task
    return root / "videos" / date / episode, root / "meta" / date
