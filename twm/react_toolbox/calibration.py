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
            d = json.loads(p.read_text())
            T = np.array(d.get("T_gel_to_rigid", d.get("T")), np.float64)
            out[f"gel_{side}"] = T[:3, 3] if T.shape == (4, 4) else np.array(d.get("gel_center_mm", [0, 0, 0]))
    return out


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
