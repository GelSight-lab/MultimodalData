"""Evaluate a world-model rollout against a synthetic probe's ground truth.

WHAT GROUND TRUTH MEANS HERE, AND WHAT IT DOES NOT

A probe is a commanded action sequence, not a recorded one. There is no
ground-truth future IMAGE — nobody performed this motion. What IS ground truth
is where the sensor WOULD BE if the action were executed exactly, which is a
pose sequence, and its projection into each camera. So a rollout is judged by
comparing the sensor the model draws against the pose the action commands, not
by pixel-matching a future frame that does not exist.

That is why the overlay matters: the comparison is spatial, and a scalar loss
against a nonexistent target would be meaningless.

THE OVERLAY'S OWN ERROR BAR

Projected ground truth is not exact. Measured on this rig:

    camera reprojection rmse        4.7 / 5.3 / 7.5 mm  ->  3.6-5.7 px at 800 mm
    gel centre in the rigid frame   <= ~5 mm            ->  ~3.8 px

so agreement inside roughly 6 px is at the noise floor and means "correct".
Sessions whose world frame is not pinned are excluded from the test set rather
than silently carrying a larger, unstated error — see the README.
"""
from __future__ import annotations

import numpy as np


def project_gt(poses7, gel_center_mm, cam_calib):
    """Ground-truth gel-centre pixels for a pose sequence. (T, 2), NaN if behind.

    ONE PROJECTION. This calls `calibration.project_gel_to_pixel`, the same
    function the dataset's own previews and the release fingerprint use, so an
    overlay drawn here cannot disagree with one drawn there.
    """
    from .calibration import project_gel_to_pixel

    out = np.full((len(np.atleast_2d(poses7)), 2), np.nan)
    for i, p in enumerate(np.atleast_2d(np.asarray(poses7, float))):
        uv = project_gel_to_pixel(p, gel_center_mm, cam_calib)
        if uv is not None:
            out[i] = uv
    return out


def overlay_gt(frame_rgb, poses7, gel_center_mm, cam_calib, *,
               held_pose7=None, held_gel_mm=None, every: int = 6,
               color=(255, 210, 63), label=None):
    """Draw the commanded trajectory on a frame. Returns a copy.

    Start is a filled dot, end a ring, the path a polyline sampled every
    `every` steps, and the sensor frame is drawn as a triad at both ends so
    ORIENTATION is visible — a dot cannot show a rotation probe, where the gel
    centre does not move at all.

    `held_pose7` draws the stationary hand dimmed, so a viewer can see the
    other sensor the rollout must also keep still.
    """
    import cv2

    from .viz import draw_sensor_frame

    out = np.ascontiguousarray(frame_rgb).copy()
    h, w = out.shape[:2]
    if held_pose7 is not None and held_gel_mm is not None:
        out = draw_sensor_frame(out, held_pose7, held_gel_mm, cam_calib,
                                stem=True, dim=True, label="held")
    # ORDER MATTERS. The triads go down first and the path markers on top:
    # drawn the other way round, the start triad's grey centre dot lands
    # exactly on the start marker and hides it — which is where the reader
    # looks to see where the motion begins.
    P = np.asarray(poses7, float)
    out = draw_sensor_frame(out, P[0], gel_center_mm, cam_calib,
                            stem=True, dim=True)
    out = draw_sensor_frame(out, P[-1], gel_center_mm, cam_calib,
                            stem=True, label=label)
    px = project_gt(P, gel_center_mm, cam_calib)
    pts = [(int(round(u)), int(round(v))) for u, v in px
           if np.isfinite(u) and 0 <= u < w and 0 <= v < h]
    for a, b in zip(pts[::every], pts[every::every]):
        cv2.line(out, a, b, color, 1, cv2.LINE_AA)
    if pts:
        cv2.circle(out, pts[0], 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pts[0], 4, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.circle(out, pts[-1], 6, color, 2, cv2.LINE_AA)
    return out


def rollout_error(pred_poses7, gt_poses7, gel_center_mm=None, cam_calib=None):
    """Per-step error of a rollout against the commanded ground truth.

    Returns a dict with, per step and summarised:
        pos_mm      Euclidean gel-centre error in world millimetres
        rot_deg     geodesic orientation error
        px          reprojection error, if a camera is given

    The pixel figure is what a reader of the overlay sees, and the millimetre
    figure is what the model actually got wrong; they differ by depth, so both
    are reported rather than one standing in for the other.
    """
    from scipy.spatial.transform import Rotation

    a = np.asarray(pred_poses7, float)
    b = np.asarray(gt_poses7, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if gel_center_mm is not None:
        Ra = Rotation.from_quat(a[:, 3:7]).as_matrix()
        Rb = Rotation.from_quat(b[:, 3:7]).as_matrix()
        c = np.asarray(gel_center_mm, float)
        pa = a[:, :3]*1000.0 + np.einsum("nij,j->ni", Ra, c)
        pb = b[:, :3]*1000.0 + np.einsum("nij,j->ni", Rb, c)
    else:
        pa, pb = a[:, :3]*1000.0, b[:, :3]*1000.0
    pos = np.linalg.norm(pa - pb, axis=1)
    qa, qb = Rotation.from_quat(a[:, 3:7]), Rotation.from_quat(b[:, 3:7])
    rot = np.degrees((qa.inv() * qb).magnitude())
    out = {"n_steps": int(n), "pos_mm": pos, "rot_deg": rot,
           "pos_mm_final": float(pos[-1]), "rot_deg_final": float(rot[-1]),
           "pos_mm_mean": float(pos.mean()), "rot_deg_mean": float(rot.mean())}
    if cam_calib is not None and gel_center_mm is not None:
        ua = project_gt(a, gel_center_mm, cam_calib)
        ub = project_gt(b, gel_center_mm, cam_calib)
        d = np.linalg.norm(ua - ub, axis=1)
        out["px"] = d
        out["px_final"] = float(d[-1])
        out["px_mean"] = float(np.nanmean(d))
    return out
