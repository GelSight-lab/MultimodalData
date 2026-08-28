"""WHICH WORLD FRAME A POSE ARRAY IS IN — declared in the data, and checkable.

The problem
-----------
The 2026-05-19 session redefined the OptiTrack origin. Its raw poses sit
(0.23, 0, 0.175) m from every other date, the release bakes the correction in,
and that fact lives in one free-text sentence inside `calibration.json`:

    "05-19 has a redefined OptiTrack world origin; offset (0.23,0,0.175)m
     baked into poses."

Three things are wrong with a sentence:

* a machine cannot read it, so every consumer re-derives the rule
* it states a DIFFERENCE without a direction, and a difference does not carry
  one. I read it backwards myself — "release minus rawH5 = +0.23" became "the
  release needs +0.23" — and only caught it by checking where the tracked
  board actually sits on two dates (0.399 vs 0.376 m: a 5 cm difference in
  where it was put down, not the 0.288 m of an uncorrected frame)
* nothing can check it. A pose array that is silently in the wrong frame looks
  exactly like a pose array in the right one

What this module adds
---------------------
`declaration(...)` returns, per episode:

    world_frame       "common" — the frame all published poses share
    raw_h5_offset_m   what a RAW-H5 pose of this episode needs added
    fingerprint       the median pixel the gel centre projects to, per camera

The fingerprint is the part that cannot be got wrong by believing a label. It
is computed at export time from the poses that ship; a consumer recomputes it
from whatever pose array they hold and compares. Measured discriminating power
on 2026-05-19/episode_002, left sensor:

    missing world offset      159 - 223 px
    metres read as millimetres      1741 - 2782 px
    y/z axes swapped          185 - 227 px
    1 degree of yaw             1.7 - 3.2 px

against a calibration rmse of 4.75 mm — about 3 px at this depth. So it
catches every frame error this rig can plausibly suffer, including ones nobody
has thought of, which a hard-coded offset check cannot.

It is deliberately NOT a hash. A hash says "different"; a pixel distance says
"different by this much", and the two failure modes worth telling apart —
a wrong frame and a re-export with slightly different interpolation — differ
by two orders of magnitude here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
VIEWS = ("left", "middle", "right")
# The frame every published pose is in. A second value would mean the release
# stopped being self-consistent, which is a release bug, not a reader's problem.
COMMON = "common"


def _calib(task: str, up_axis: str):
    """The extrinsics in the convention the CALLER's poses are in.

    calib_dir() returns whatever tree the environment points it at, and after
    the release was rotated to Z-up that could be either convention -- so the
    directory is not the answer, the request is. Handed a Y-up calibration
    beside Z-up poses, this fingerprint reads 199.5 px from the truth.
    """
    from react_toolbox.frames import as_up_axis
    from twm.calib_epoch import calib_dir
    d = calib_dir(task)
    raw = {v: json.loads((d / f"T_mocap_to_cam_{v}.json").read_text())
           for v in VIEWS}
    declared = next(iter(raw.values())).get("up_axis") or "y"
    cal = as_up_axis({"up_axis": declared, "cams": {
        v: {"T_mocap_to_cam": np.asarray(c["T_mocap_to_cam"], float),
            "intrinsics": c["intrinsics"]} for v, c in raw.items()}}, up_axis)
    gel = {s: json.loads((d / f"T_gel_to_rigid_{s}.json").read_text())
           for s in ("left", "right")}
    return cal["cams"], gel


def fingerprint(pose7, side: str, task: str, *, up_axis: str) -> dict:
    """Median projected gel-centre pixel per camera. The frame's signature.

    Median, not mean: a handful of OptiTrack dropouts move a mean by tens of
    pixels and leave a median untouched, and a signature that drifts with the
    noise cannot be compared against a stored one.
    """
    from scipy.spatial.transform import Rotation

    cams, gel = _calib(task, up_axis)
    p = np.asarray(pose7, float)
    ok = np.isfinite(p).all(1) & (np.linalg.norm(p[:, 3:], axis=1) > 0.5)
    p = p[ok]
    if len(p) < 10:
        raise ValueError(f"only {len(p)} valid poses — cannot fingerprint")
    R = Rotation.from_quat(p[:, 3:7]).as_matrix()
    ctr = np.asarray(gel[side]["gel_center_in_rigid_mm"], float)
    q = p[:, :3] * 1000.0 + np.einsum("nij,j->ni", R, ctr)
    out = {}
    for v, c in cams.items():
        T = np.asarray(c["T_mocap_to_cam"], float)
        I = c["intrinsics"]
        X = q @ T[:3, :3].T + T[:3, 3]
        z = np.where(np.abs(X[:, 2]) < 1e-9, 1e-9, X[:, 2])
        out[v] = [float(np.median(I["fx"] * X[:, 0] / z + I["ppx"])),
                  float(np.median(I["fy"] * X[:, 1] / z + I["ppy"]))]
    return out


def verify_fingerprint(pose7, side: str, task: str, stored: dict, *,
                      up_axis: str) -> float:
    """Worst per-camera pixel distance between this pose array and `stored`.

    Returns the WORST, not the mean: a frame error that happens to be along
    one camera's optical axis is invisible to that camera and obvious to the
    others, so averaging would dilute exactly the evidence that matters.

    `stored` is THIS SIDE's {view: [u, v]} — not the whole per-side dict. The
    first version accepted either and picked with `side if side in stored`,
    which silently did the wrong thing because the sensor sides and the camera
    views share two names: `stored["left"]` is the left CAMERA's pixel, so
    `ref` became a 2-element list, every `v not in ref` was true, the loop
    skipped everything and the function returned 0.0 for ANY input. It passed
    a pose array with the world offset stripped out. Caught only by running it
    on that array and seeing an impossible zero.

    So: no guessing. A malformed `stored` raises.
    """
    got = fingerprint(pose7, side, task, up_axis=up_axis)
    if not isinstance(stored, dict) or not all(
            v in stored and len(stored[v]) == 2 for v in VIEWS):
        raise ValueError(
            f"stored fingerprint must be {{view: [u, v]}} for {VIEWS}; got "
            f"{stored!r}. Pass declaration['fingerprint'][side], not the "
            f"whole per-side dict — the two share the names 'left'/'right'.")
    return max(float(np.hypot(got[v][0] - stored[v][0],
                              got[v][1] - stored[v][1])) for v in VIEWS)


def read_declaration(parquet_path) -> dict | None:
    """The world-frame declaration carried by an episode's parquet, or None."""
    import pyarrow.parquet as pq
    md = pq.read_schema(str(parquet_path)).metadata or {}
    raw = md.get(b"twm.world_frame")
    if raw is None:
        return None
    return json.loads(raw.decode())


def build_declaration(task: str, date: str, ep: str, poses: dict) -> dict:
    """The declaration to embed, built from the release's own records.

    The offset is READ from `episodes.jsonl` via `calib_epoch.world_offset_m`,
    never retyped here: the dataset already publishes it, and a constant in
    this file would be the fifth copy of a number whose earlier copies are the
    reason this module exists.
    """
    from twm.calib_epoch import world_offset_m
    off = list(world_offset_m(task, date, ep, up_axis="y"))
    return {
        "world_frame": COMMON,
        "raw_h5_offset_m": off,
        "raw_h5_offset_up_axis": "y",
        "raw_h5_note": ("add this to a pose read straight out of the source "
                        "H5, which is Y-up as recorded; the published poses "
                        "already have it, expressed Z-up"),
        # the poses handed in here are the RELEASE's, which are Z-up
        "fingerprint": {s: fingerprint(poses[s], s, task, up_axis="z")
                        for s in ("left", "right") if s in poses},
        "fingerprint_note": ("median projected gel-centre pixel per camera; "
                             "recompute from your own poses and compare — "
                             "see twm.world_frame.verify_fingerprint"),
    }
