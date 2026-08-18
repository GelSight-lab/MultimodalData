"""Which world frame a pose array is in — read it, and check it.

The 2026-05-19 motherboard session redefined the OptiTrack origin: its raw
poses sit (0.23, 0, 0.175) m from every other date. The published poses have
the correction baked in, so all 32 episodes share one frame — but that fact
used to live in one free-text sentence inside `calibration.json`, which a
machine cannot read, which states a difference without a direction, and which
nothing can check.

Each episode's parquet now carries the declaration, and a FINGERPRINT: the
median pixel the gel centre projects to in each camera, computed from the
poses that ship. Recompute it from whatever poses you hold and compare.

    dec = read_world_frame("episode_002.parquet")
    err = verify_world_frame(my_poses, "left", "motherboard", dec)
    assert err < 6.0

Measured discriminating power on 2026-05-19/episode_002:

    missing world offset       222.9 px
    y/z axes swapped           227.2 px
    metres read as mm       1741-2782 px
    1 degree of yaw            1.7-3.2 px

against a calibration rmse of 4.75 mm — about 3 px at this depth. So it
catches every frame error this rig can plausibly suffer, including ones nobody
has thought of, which a check for one known offset cannot.
"""
from __future__ import annotations

import json

import numpy as np

VIEWS = ("left", "middle", "right")


def read_world_frame(parquet_path):
    """The declaration embedded in an episode's parquet, or None."""
    import pyarrow.parquet as pq
    md = pq.read_schema(str(parquet_path)).metadata or {}
    raw = md.get(b"twm.world_frame")
    return json.loads(raw.decode()) if raw is not None else None


def projection_fingerprint(pose7, gel_center_mm, cams) -> dict:
    """Median projected gel-centre pixel per camera.

    Median, not mean: a handful of tracking dropouts move a mean by tens of
    pixels and leave a median untouched, and a signature that drifts with the
    noise cannot be compared against a stored one.
    """
    from .calibration import project_gel_to_pixel

    p = np.asarray(pose7, float)
    ok = np.isfinite(p).all(1) & (np.linalg.norm(p[:, 3:], axis=1) > 0.5)
    p = p[ok]
    if len(p) < 10:
        raise ValueError(f"only {len(p)} valid poses — cannot fingerprint")
    out = {}
    for v in VIEWS:
        if v not in cams:
            continue
        uv = [project_gel_to_pixel(q, gel_center_mm, cams[v]) for q in p]
        uv = np.asarray([x for x in uv if x is not None], float)
        if len(uv) < 10:
            continue
        out[v] = [float(np.median(uv[:, 0])), float(np.median(uv[:, 1]))]
    return out


def verify_world_frame(pose7, side: str, task_root, declaration) -> float:
    """Worst per-camera pixel distance from the declared fingerprint.

    WORST, not mean: a frame error along one camera's optical axis is
    invisible to that camera and obvious to the others, so averaging would
    dilute exactly the evidence that matters.

    `task_root` is the directory holding `calibration/` — the same argument
    `load_calibration` takes.
    """
    from .calibration import load_calibration

    if not declaration or "fingerprint" not in declaration:
        raise ValueError("declaration has no fingerprint; this episode "
                         "predates the world-frame metadata")
    stored = declaration["fingerprint"].get(side)
    if not isinstance(stored, dict):
        raise ValueError(
            f"no fingerprint for side {side!r}. Pass the whole declaration; "
            f"this function selects the side. (The sides and the cameras "
            f"share the names left/right, so selecting by hand is easy to "
            f"get wrong — an earlier version of this check did, and returned "
            f"0.0 for every input as a result.)")
    cal = load_calibration(task_root)
    got = projection_fingerprint(pose7, cal[f"gel_{side}"], cal["cams"])
    common = [v for v in VIEWS if v in got and v in stored]
    if not common:
        raise ValueError("no camera in common between the fingerprint and "
                         "this calibration")
    return max(float(np.hypot(got[v][0] - stored[v][0],
                              got[v][1] - stored[v][1])) for v in common)
