"""Per-task camera calibration: load OptiTrack->camera extrinsics/intrinsics
and project a GelSight sensor's pose into a camera image.

Calibration epoch is per task (motherboard=May-12, pushT=June-26); the files
live under data/<task>/calibration/. Convention matches twm.viz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# H5 cam_idx -> stream name (verified by serials)
CAM_NAME = {0: "right", 1: "left", 2: "middle"}


def load_calibration(task_root):
    """Load all camera calibrations + gel centers for a task.

    Returns dict: {cam_name: {"T_mocap_to_cam": (4,4), "intrinsics": {...},
                              "serial": str, "rmse": float}},
                  plus "gel_left"/"gel_right" center (3,) in rigid-body mm.
    """
    cdir = Path(task_root) / "calibration"
    out = {"cams": {}}
    for cam in ("left", "middle", "right"):
        p = cdir / f"T_mocap_to_cam_{cam}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out["cams"][cam] = {
            "T_mocap_to_cam": np.array(d["T_mocap_to_cam"], np.float64),
            "intrinsics": d["intrinsics"],
            "serial": d.get("camera_serial"),
            "rmse": d.get("rmse_mm", d.get("rmse_px")),
        }
    for side in ("left", "right"):
        p = cdir / f"T_gel_to_rigid_{side}.json"
        if p.exists():
            out[f"gel_{side}"] = _gel_center(p)
    return out


def _gel_center(path) -> np.ndarray:
    """The measured gel centre in rigid-body millimetres. RAISES if absent.

    This read three keys that do not exist in any published file —
    `T_gel_to_rigid`, `T`, `gel_center_mm` — and fell through to a default of
    `[0, 0, 0]`. The real key is `gel_center_in_rigid_mm`, and it is present in
    all four published files of both tasks.

    Zero is the worst possible default here because it is a VALID-LOOKING
    offset: it says "the gel centre is the rigid-body origin", so nothing
    downstream can tell it from a real answer. Every projection the toolbox
    produced was of the rigid body, off by the real offset — measured on
    motherboard/2026-05-11/episode_003, median over the episode: 35.8 px
    (left camera), 20.8 (middle), 28.0 (right), against a calibration rmse of
    4.75 mm ~ 3 px. Seven to twelve times the rig's own error, and still
    shaped like a slightly miscalibrated rig rather than like a bug.

    So: no fallback. A calibration file that cannot say where the gel is stops
    the caller, because a projection of the wrong point is worse than none.
    """
    d = json.loads(Path(path).read_text())
    for key in ("gel_center_in_rigid_mm", "gel_center_mm"):
        if key in d:
            return np.asarray(d[key], np.float64)
    T = d.get("T_gel_to_rigid", d.get("T"))
    if T is not None:
        T = np.asarray(T, np.float64)
        if T.shape == (4, 4):
            return T[:3, 3]
    raise KeyError(
        f"{path}: no gel centre. Looked for gel_center_in_rigid_mm, "
        f"gel_center_mm, T_gel_to_rigid, T. Refusing to assume [0,0,0] — "
        f"that is a valid-looking offset and would silently project the "
        f"rigid-body origin instead of the gel.")


def pose7_to_matrix(pose7):
    """[x,y,z, qx,qy,qz,qw] (m, scalar-last) -> 4x4 (mm translation)."""
    p = np.asarray(pose7, np.float64)
    x, y, z, qx, qy, qz, qw = p
    n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) + 1e-12
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x*1000, y*1000, z*1000]  # m->mm
    return T


def project_gel_to_pixel(sensor_pose7, gel_center_mm, cam_calib):
    """Project a GelSight center into a camera image. Returns (u, v) px or None
    if behind the camera. cam_calib is one entry from load_calibration()['cams'].
    """
    T_rigid = pose7_to_matrix(sensor_pose7)
    p_mocap = (T_rigid @ np.array([*gel_center_mm, 1.0]))[:3]
    p_cam = (cam_calib["T_mocap_to_cam"] @ np.array([*p_mocap, 1.0]))[:3]
    if p_cam[2] <= 0:
        return None
    K = cam_calib["intrinsics"]
    u = K["fx"] * p_cam[0] / p_cam[2] + K["ppx"]
    v = K["fy"] * p_cam[1] / p_cam[2] + K["ppy"]
    return float(u), float(v)
