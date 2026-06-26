#!/usr/bin/env python3
"""
Calibrate OptiTrack → RealSense for ALL overhead cameras at once.

This is the multi-camera companion to `mocap_to_cam.py`. Because every camera
shares the same OptiTrack world frame, you place a reflective ball once, click
it in whichever camera views can see it, and provide the ball's OptiTrack
position *once* — that single mocap coordinate is shared across all cameras that
saw it. Each camera accumulates its own independent set of correspondences; a
camera only needs ≥4 points of its own, so a ball out of frame for one camera
simply doesn't contribute a point to that camera.

At the end, a separate transform `T_mocap_to_cam` is solved per camera and
written to the per-view JSON/NPY files the viewer expects
(`T_mocap_to_cam_{left,middle,right}.json`).

Solver (`--method`):
    pnp  (default)  2D↔3D: solvePnPRansac + refine, using only the clicked
                    pixel + the mocap point + the factory color intrinsics.
                    No depth — avoids the RealSense depth noise that dominates
                    the SVD method's error — and it directly minimizes pixel
                    reprojection error, the quantity the overlay cares about.
    svd             3D↔3D: deproject the click with depth, then Arun SVD.
                    Kept as a fallback / for A/B comparison.

Mocap input:
    By default you type the ball's OptiTrack x y z each point. With
    `--mocap_body NAME`, the position is grabbed automatically from the live
    VRPN stream `/vrpn_client_node/NAME/pose` (timestamp-fresh) when you
    advance — no typing, no transcription errors. Requires the VRPN client
    running and the ball tracked as a rigid body named NAME in Motive.

Usage:
    python -m twm.calibration.mocap_to_cam_multi --num_points 8
    python -m twm.calibration.mocap_to_cam_multi --num_points 10 --method pnp \
        --mocap_body calib_ball
    python -m twm.calibration.mocap_to_cam_multi --method svd          # old behavior

Marker auto-detection (Tier 3):
    By default (--source ir) markers are detected in the RealSense INFRARED stream
    (emitter on): retroreflective OptiTrack markers pop as the brightest spots and
    the IR response is uniform across cameras (RGB is not). Detection keys on local
    contrast (white top-hat), not absolute brightness or color. Each IR marker is
    mapped to its native COLOR pixel via depth + factory IR→color extrinsics, so
    the calibration stays in the color frame. Use --source color to detect in RGB.
    Detected markers are ringed in cyan; a left-click snaps to the nearest and
    refines to its sub-pixel center (--no_snap to disable). Each window gets a
    'thresh' slider (top-hat contrast); press 'm' to view the threshold mask.

Per-point controls (focus any camera window):
    left-click      mark the nearest marker in that view (re-click to move it)
    'thresh' slider per-window brightness threshold for marker detection
    [m]             toggle the threshold-mask debug view
    [-] / [=]       nudge all thresholds down / up (same as the sliders)
    [n] / [SPACE]   done with this point → capture/enter its OptiTrack position
    [r]             clear this point's clicks and re-click
    [u]             undo the LAST recorded point (all cameras)
    [q]             finish: solve + save with the points collected so far

All 3D coordinates are in millimetres. OptiTrack input/stream is scaled by
--mocap_scale (default 1000, i.e. metres → mm).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

# Reuse the single-camera helpers + the canonical serial/view mapping so the two
# tools never drift apart.
from twm.calibration.mocap_to_cam import (
    get_depth_at,
    deproject,
    rigid_transform_svd,
    parse_vector,
)
from twm.data_collection import REALSENSE_SERIALS
from twm.viz import CAM_CALIB_NAME


# View label per H5 cam index, derived from the canonical output filenames
# (e.g. "T_mocap_to_cam_right.json" -> "right").
def _view_label(cam_idx: int) -> str:
    name = CAM_CALIB_NAME.get(cam_idx, f"cam{cam_idx}")
    return name.replace("T_mocap_to_cam_", "").replace(".json", "")


# ── Sub-pixel marker detection (Tier 3) ─────────────────────────────────────────
#
# Detection keys on LOCAL CONTRAST, not absolute brightness or color. A white
# top-hat (image − morphological opening) removes the slowly-varying background,
# leaving only small bright spots relative to their surroundings. This is robust
# to per-camera exposure differences and to lighting gradients across the
# workspace — a global brightness threshold is not. The `thresh` value therefore
# operates on the top-hat *contrast* response (typically tens), not raw level.

# Structuring-element size for the top-hat. Must exceed the largest marker
# diameter so the opening fully erases markers (else the top-hat under-responds
# at the marker center). 25 px comfortably covers markers up to ~max_area=800.
TOPHAT_KSIZE = 25


def _tophat(gray):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KSIZE, TOPHAT_KSIZE))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)


def blobs_from_tophat(th, thresh=40, min_area=6, max_area=800,
                      min_circularity=0.55):
    """Find small, round blobs in a precomputed top-hat image `th`. Returns a
    list of (x, y) centroids. `thresh` is a top-hat contrast cutoff."""
    _, bw = cv2.threshold(th, thresh, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4.0 * np.pi * area / (per * per)
        if circ < min_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        out.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return out


def detect_bright_blobs(gray, thresh=40, min_area=6, max_area=800,
                        min_circularity=0.55):
    """Detect markers by local contrast (top-hat). Convenience wrapper that
    computes the top-hat then finds blobs."""
    return blobs_from_tophat(_tophat(gray), thresh, min_area, max_area, min_circularity)


def adaptive_thresh(gray, frac=0.5, lo=20, hi=200):
    """Per-camera initial top-hat contrast threshold. Scales with the strongest
    local contrast in the image (the markers), so it adapts to exposure and
    lighting without keying on absolute brightness."""
    if gray is None:
        return lo
    tmax = int(_tophat(gray).max())
    return int(np.clip(round(frac * tmax), lo, hi))


def refine_center(gray, x, y, win=9, rel=0.5):
    """Intensity-weighted centroid in a window around (x, y) → sub-pixel center.
    Uses a window-relative floor (rel·[max−min]) so it works for dim markers too;
    falls back to the input if the window is flat."""
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - win), min(w, xi + win + 1)
    y0, y1 = max(0, yi - win), min(h, yi + win + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    lo, hi = float(patch.min()), float(patch.max())
    if hi <= lo:
        return float(x), float(y)
    weight = patch - (lo + rel * (hi - lo))
    weight[weight < 0] = 0.0
    s = weight.sum()
    if s <= 0:
        return float(x), float(y)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((xs * weight).sum() / s), float((ys * weight).sum() / s)


def foreground_raw_depth(depth_img, u, v, win=8, margin_raw=60, min_count=3):
    """Nearest-cluster raw depth in a window around (u, v).

    Retroreflective markers usually have NO stereo depth (saturation → hole), so
    the depth must come from neighbours. A marker sits raised on a thin arm, so
    the neighbourhood mixes the arm/marker (foreground) with the table behind it
    (background) — and the foreground can be a small minority of the window. The
    median (or even a low percentile) is then biased toward the background → wrong
    Z → a parallax error of a few px in the mapped color pixel.

    Instead, take the nearest valid depth and the cluster within `margin_raw`
    units of it (the foreground surface). Require `min_count` supporting pixels so
    a single noisy near-pixel can't latch the estimate; otherwise fall back to a
    low percentile. (`margin_raw` is in raw depth units ≈ mm on the D415.)"""
    h, w = depth_img.shape
    xi, yi = int(round(u)), int(round(v))
    x0, x1 = max(0, xi - win), min(w, xi + win + 1)
    y0, y1 = max(0, yi - win), min(h, yi + win + 1)
    vals = depth_img[y0:y1, x0:x1].reshape(-1)
    vals = vals[vals > 0]
    if vals.size < min_count:
        return 0.0
    dmin = vals.min()
    cluster = vals[vals <= dmin + margin_raw]
    if cluster.size >= min_count:
        return float(np.median(cluster))
    return float(np.percentile(vals, 20))


def ir_pixel_to_color(u, v, depth_img, depth_scale, depth_intr, color_intr, extr,
                      win=8):
    """Map an IR/depth-frame pixel (u, v) to its native COLOR pixel + 3D point.

    On the D415 the depth image shares the left-IR (infrared 1) pixel grid, so a
    marker detected at (u, v) in IR has its depth at depth_img[v, u]. We take the
    foreground depth near the marker (see foreground_raw_depth), deproject with
    the depth intrinsics, transform into the color frame via the factory
    depth→color extrinsics, and project with the color intrinsics.

    Returns ((cu, cv) color pixel, P_color_mm 3-vec) or None if no valid depth."""
    raw = foreground_raw_depth(depth_img, u, v, win)
    if raw <= 0:
        return None
    d_m = raw * depth_scale  # metres (extrinsics translation is in metres)
    P = rs.rs2_deproject_pixel_to_point(depth_intr, [float(u), float(v)], d_m)
    Pc = rs.rs2_transform_point_to_point(extr, P)
    if Pc[2] <= 0:
        return None
    cu, cv = rs.rs2_project_point_to_pixel(color_intr, Pc)
    return (float(cu), float(cv)), [Pc[0] * 1000.0, Pc[1] * 1000.0, Pc[2] * 1000.0]


# ── Live OptiTrack reader (Tier 2: auto-grab mocap) ─────────────────────────────

class _MocapReader:
    """Subscribe to one VRPN PoseStamped topic and track the latest position.

    Lazily imports rospy so the manual-entry flow needs no ROS install.
    """

    def __init__(self, body: str):
        import rospy  # noqa: F401  (lazy)
        from geometry_msgs.msg import PoseStamped
        import threading

        self.body = body
        self.topic = f"/vrpn_client_node/{body}/pose"
        self._latest = None  # (t_sec, np.array([x,y,z]) in metres)
        self._lock = threading.Lock()

        try:
            rospy.init_node("twm_calib_multi", anonymous=True, disable_signals=True)
        except Exception:
            pass  # already initialized

        def _cb(msg):
            p = msg.pose.position
            with self._lock:
                self._latest = (msg.header.stamp.to_sec(),
                                np.array([p.x, p.y, p.z], np.float64))

        rospy.Subscriber(self.topic, PoseStamped, _cb)
        self._spin = threading.Thread(target=rospy.spin, daemon=True)
        self._spin.start()

    def latest(self, max_age_s: float = 0.5):
        """Return position (metres) if a fresh sample exists, else None."""
        with self._lock:
            data = self._latest
        if data is None:
            return None
        t, pos = data
        if time.time() - t > max_age_s:
            return None
        return pos


# ── Per-camera live capture ─────────────────────────────────────────────────────

class _Cam:
    """One RealSense camera: pipeline + aligned color/depth + intrinsics."""

    def __init__(self, cam_idx: int, serial: str, *, source="ir", snap=True,
                 snap_radius=25.0, blob_thresh=25, blob_min_area=5, blob_max_area=800,
                 blob_circ=0.5):
        self.cam_idx = cam_idx
        self.serial = serial
        self.label = _view_label(cam_idx)
        self.source = source                       # "ir" or "color"
        self.win = f"cam{cam_idx} ({self.label})  [{source}]  serial {serial}"

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if source == "ir":
            cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)  # depth → color grid (fallback clicks)

        color_prof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_prof = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = color_prof.get_intrinsics()          # K for solving (color)
        self.depth_intr = depth_prof.get_intrinsics()
        dsensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = dsensor.get_depth_scale()
        self.depth_to_mm = self.depth_scale * 1000.0
        # depth (== IR1 grid) → color extrinsics, for IR-pixel → color-pixel mapping.
        self.extr_d2c = depth_prof.get_extrinsics_to(color_prof)
        if source == "ir":
            try:
                dsensor.set_option(rs.option.emitter_enabled, 1)  # markers need IR light
            except Exception as e:
                print(f"  cam{cam_idx}: could not enable IR emitter: {e}")

        # Sub-pixel marker detection / click-to-snap config.
        self.snap = snap
        self.snap_radius = snap_radius
        self.blob_thresh = blob_thresh
        self.blob_min_area = blob_min_area
        self.blob_max_area = blob_max_area
        self.blob_circ = blob_circ

        self.color = None        # BGR color image — always the displayed image
        self.depth_native = None # depth on IR1 grid (for IR-pixel → color mapping)
        self.depth_aligned = None# depth on color grid (for fallback color-pixel clicks)
        self.det_gray = None     # grayscale image detection runs on (IR or color-gray)
        self.tophat = None       # local-contrast image used for detection + mask
        self.blobs = []          # raw detections in DETECTION coords (IR or color)
        self.mapped = []         # per-frame: [{"color":(cx,cy), "p":P3_mm or None}]

        # Accumulated correspondences for THIS camera (parallel lists).
        self.points_det = []    # (x, y) in the displayed image — for redraw
        self.points_px = []     # (x, y) COLOR pixel             — used by PnP
        self.points_cam = []    # (3,) color-frame mm or None    — used by SVD
        self.points_mocap = []  # (3,) mocap mm

        # Pending click for the current point.
        self.pending_raw = None   # (x, y) set by mouse callback (display coords)
        self.pending = None       # {"det","color","p_cam"}

    @property
    def K(self):
        i = self.intrinsics
        return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], np.float64)

    def wait_frames(self):
        frames = self.pipeline.wait_for_frames()
        self.color = np.asanyarray(frames.get_color_frame().get_data())
        self.depth_native = np.asanyarray(frames.get_depth_frame().get_data())
        self.depth_aligned = np.asanyarray(
            self.align.process(frames).get_depth_frame().get_data())

        if self.source == "ir":
            self.det_gray = np.asanyarray(frames.get_infrared_frame(1).get_data())
        else:
            self.det_gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        self.tophat = _tophat(self.det_gray)
        self.blobs = blobs_from_tophat(self.tophat, self.blob_thresh,
                                       self.blob_min_area, self.blob_max_area, self.blob_circ)

        # Map every detection into COLOR pixel coords so markers overlay on the RGB
        # image. IR detections are mapped through native depth + IR→color
        # extrinsics; color detections are already color pixels (3D via aligned
        # depth, best-effort). Markers without valid depth are dropped from
        # `mapped` (can't be placed on RGB).
        self.mapped = []
        for bx, by in self.blobs:
            if self.source == "ir":
                # Foreground depth near the marker (markers sit raised on thin arms
                # and often have depth holes; the nearest-cluster depth avoids the
                # background-table bias that shifts the mapped color pixel).
                res = ir_pixel_to_color(bx, by, self.depth_native, self.depth_scale,
                                        self.depth_intr, self.intrinsics, self.extr_d2c)
                if res is None:
                    continue
                (cx, cy), p = res
            else:
                raw_d = get_depth_at(self.depth_aligned, int(round(bx)), int(round(by)), 7)
                p = (deproject(self.intrinsics, bx, by, raw_d * self.depth_to_mm)
                     if raw_d > 0 else None)
                cx, cy = bx, by
            self.mapped.append({"color": (cx, cy), "p": p})

    def _nearest_mapped(self, x, y):
        """Nearest mapped marker (in COLOR coords) within snap_radius, or None."""
        best, best_d = None, self.snap_radius
        for m in self.mapped:
            cx, cy = m["color"]
            d = np.hypot(cx - x, cy - y)
            if d < best_d:
                best, best_d = m, d
        return best

    def resolve_pending(self, depth_patch: int):
        """Resolve a click (in COLOR/display coords) to a pending point. Snaps to
        the nearest mapped marker; otherwise falls back to the raw color pixel +
        aligned-depth 3D point."""
        if self.pending_raw is None:
            return
        rpx, rpy = self.pending_raw
        self.pending_raw = None

        m = self._nearest_mapped(rpx, rpy) if self.snap else None
        if m is not None:
            cx, cy = m["color"]
            self.pending = {"color": (cx, cy), "p_cam": m["p"]}
            pstr = (f"P=[{m['p'][0]:.0f},{m['p'][1]:.0f},{m['p'][2]:.0f}]"
                    if m["p"] is not None else "(no 3D)")
            print(f"   [{self.label}] snap → color=({cx:.1f},{cy:.1f}) {pstr}")
            return

        # Fallback: raw color click + aligned depth (works in both modes).
        raw_d = get_depth_at(self.depth_aligned, int(round(rpx)), int(round(rpy)), depth_patch)
        depth_mm = raw_d * self.depth_to_mm
        p_cam = deproject(self.intrinsics, rpx, rpy, depth_mm) if depth_mm > 0 else None
        self.pending = {"color": (float(rpx), float(rpy)), "p_cam": p_cam}
        extra = (f"P=[{p_cam[0]:.0f},{p_cam[1]:.0f},{p_cam[2]:.0f}]"
                 if p_cam is not None else "(no depth — OK for PnP)")
        print(f"   [{self.label}] raw  color=({rpx},{rpy}) {extra}")

    def render(self, show_mask=False):
        if show_mask:
            # Tuning view: show the DETECTION-space image (IR or color-gray) with
            # everything above threshold tinted red, plus raw detections — so the
            # 'thresh' slider/mask align with what the detector actually sees.
            disp = (cv2.cvtColor(self.det_gray, cv2.COLOR_GRAY2BGR)
                    if self.source == "ir" else self.color.copy())
            disp[self.tophat >= self.blob_thresh] = (0, 0, 255)
            for bx, by in self.blobs:
                cv2.circle(disp, (int(round(bx)), int(round(by))), 7, (255, 255, 0),
                           1, cv2.LINE_AA)
        else:
            # Normal view: RGB with markers overlaid at their mapped COLOR pixels.
            disp = self.color.copy()
            for m in self.mapped:
                cx, cy = m["color"]
                cv2.circle(disp, (int(round(cx)), int(round(cy))), 7, (255, 255, 0),
                           1, cv2.LINE_AA)
            for i, (px, py) in enumerate(self.points_px):
                ix, iy = int(round(px)), int(round(py))
                cv2.circle(disp, (ix, iy), 6, (0, 255, 0), 2)
                cv2.putText(disp, str(i + 1), (ix + 8, iy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if self.pending is not None:
                cx, cy = self.pending["color"]
                cv2.circle(disp, (int(round(cx)), int(round(cy))), 8, (0, 0, 255), 2,
                           cv2.LINE_AA)
        tmax = int(self.tophat.max()) if self.tophat is not None else 0
        view = "MASK" if show_mask else "RGB"
        status = "PENDING" if self.pending is not None else "click marker (or skip)"
        cv2.putText(disp, f"[{self.label}/{view}] pts={len(self.points_px)}  "
                    f"shown={len(self.mapped)}  thr={self.blob_thresh} ctr_max={tmax}  {status}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.imshow(self.win, disp)

    def commit_point(self, p_mocap_mm):
        """Attach the shared mocap coord to this camera's pending click."""
        if self.pending is None:
            return False
        self.points_det.append(self.pending["color"])
        self.points_px.append(self.pending["color"])
        self.points_cam.append(self.pending["p_cam"])
        self.points_mocap.append(np.asarray(p_mocap_mm, np.float64))
        self.pending = None
        return True

    def clear_pending(self):
        self.pending = None
        self.pending_raw = None

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ── Solvers ─────────────────────────────────────────────────────────────────────

def _solve_pnp(P_mocap, P_px, K):
    """2D↔3D. Returns (T 4x4, per_point_px_err, inlier_mask). Zero distortion to
    match the pinhole projection used downstream in twm.viz.

    Robust to RANSAC failure (e.g. near-coplanar markers): tries RANSAC/EPnP for
    outlier rejection, then falls back through SQPnP / iterative on all points.
    Raises only if every PnP variant fails."""
    obj = P_mocap.reshape(-1, 1, 3).astype(np.float64)
    img = P_px.reshape(-1, 1, 2).astype(np.float64)
    dist = np.zeros((5, 1))
    n = len(P_mocap)
    rvec = tvec = inl = None

    # 1) RANSAC + EPnP (rejects bad clicks). Looser threshold than before so a
    #    couple of noisy points don't prevent consensus.
    try:
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            obj, img, K, dist, reprojectionError=8.0, iterationsCount=300,
            flags=cv2.SOLVEPNP_EPNP)
        if ok and inliers is not None and len(inliers) >= 4:
            rvec, tvec, inl = rv, tv, inliers.flatten()
    except cv2.error:
        pass

    # 2) Fallbacks on all points (handle planar / few-point configs).
    if rvec is None:
        for flag in ("SOLVEPNP_SQPNP", "SOLVEPNP_ITERATIVE", "SOLVEPNP_EPNP"):
            if not hasattr(cv2, flag):
                continue
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=getattr(cv2, flag))
            except cv2.error:
                ok = False
            if ok:
                rvec, tvec, inl = rv, tv, np.arange(n)
                break

    if rvec is None:
        raise RuntimeError("all PnP variants failed")

    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)
    except cv2.error:
        pass
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    mask = np.zeros(n, bool)
    mask[inl] = True
    return T, err, mask


def _solve_svd(P_mocap, P_cam):
    """3D↔3D (Arun). Returns (T 4x4, per_point_mm_err)."""
    T = rigid_transform_svd(P_mocap, P_cam)  # P_cam ≈ T @ P_mocap
    err = np.array([np.linalg.norm((T @ np.append(P_mocap[i], 1.0))[:3] - P_cam[i])
                    for i in range(len(P_mocap))])
    return T, err


# ── Solve + save (per camera) ────────────────────────────────────────────────────

def _intr_dict(intr):
    return {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "width": intr.width, "height": intr.height}


def _K_from_intr(intr):
    return np.array([[intr["fx"], 0, intr["ppx"]],
                     [0, intr["fy"], intr["ppy"]], [0, 0, 1]], np.float64)


def _solve_from_points(P_mocap, P_px, P_cam, K, method):
    """Solve T from parallel point lists (P_cam entries may be None). Tries
    `method`; PnP falls back to SVD (3D-3D) if it fails or is short on points.
    Returns (T, info_dict) or raises RuntimeError if nothing works."""
    P_mocap = np.asarray(P_mocap, float)
    if method == "pnp":
        if len(P_px) >= 4:
            try:
                T, err, mask = _solve_pnp(P_mocap, np.asarray(P_px, float), K)
                rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
                return T, {"method": "pnp", "unit": "px", "errs": err.tolist(),
                           "rmse": rmse, "mask": mask.tolist(),
                           "extra": {"rmse_px": rmse, "num_inliers": int(mask.sum()),
                                     "reproj_err_px": err.tolist()}}
            except Exception as e:
                print(f"   ⚠ PnP failed ({e}); falling back to SVD (3D-3D).")
        else:
            print(f"   ⚠ only {len(P_px)} points (<4) for PnP; trying SVD.")

    idx = [i for i, pc in enumerate(P_cam) if pc is not None]
    if len(idx) < 4:
        raise RuntimeError(f"need ≥4 points (PnP px={len(P_px)}) or ≥4 with 3D "
                           f"(SVD have={len(idx)})")
    Pm = P_mocap[idx]
    Pc = np.asarray([P_cam[i] for i in idx], float)
    T, err = _solve_svd(Pm, Pc)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    used = "svd" if method == "svd" else "svd(fallback)"
    return T, {"method": used, "unit": "mm", "errs": err.tolist(), "rmse": rmse,
               "mask": [True] * len(idx),
               "extra": {"rmse_mm": rmse, "residuals_mm": err.tolist(), "svd_idx": idx}}


def _print_and_save(out_dir, out_name, serial, intr, P_mocap, P_px, P_cam, K,
                    mocap_scale, method, label):
    """Solve + save one camera. Always records the raw point arrays. Never raises:
    returns True if saved, False if the solve was impossible (raw points are still
    preserved in the separate dump)."""
    base = {
        "description": "T_mocap_to_cam: maps OptiTrack (mm) -> camera frame (mm)",
        "camera_serial": serial,
        "intrinsics": intr,
        "mocap_scale": mocap_scale,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "points_px": [list(map(float, p)) for p in P_px],
        "points_mocap_mm": [list(map(float, p)) for p in P_mocap],
        "points_cam_mm": [list(map(float, p)) if p is not None else None for p in P_cam],
    }
    try:
        T, info = _solve_from_points(P_mocap, P_px, P_cam, K, method)
    except Exception as e:
        print(f"\n[{label}] ❌ solve failed: {e}  — NOT saved (raw points kept in dump).")
        return False

    unit, errs, rmse, mask = info["unit"], info["errs"], info["rmse"], info["mask"]
    R, t = T[:3, :3], T[:3, 3]
    euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    print(f"\n{'='*60}")
    print(f"  [{label}]  T_mocap_to_cam  |  method={info['method']}  N={len(P_mocap)}")
    print(f"{'='*60}")
    for row in T:
        print(f"    [{row[0]:+10.6f} {row[1]:+10.6f} {row[2]:+10.6f} {row[3]:+10.4f}]")
    print(f"  Translation (mm): [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
    print(f"  Rotation (XYZ°):  [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]  "
          f"det(R)={np.linalg.det(R):.4f}")
    for i, e in enumerate(errs):
        mark = "✅" if e < 5.0 else ("⚠️" if e < 10.0 else "❌")
        outl = "" if mask[i] else "  (outlier)"
        print(f"    Point {i+1}: {e:.3f} {unit} {mark}{outl}")
    print(f"  RMSE: {rmse:.3f} {unit}  ({int(np.sum(mask))}/{len(errs)} inliers)")

    base["method"] = info["method"]
    base["num_points"] = int(len(P_mocap))
    base.update(info["extra"])
    base["T_mocap_to_cam"] = T.tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(out_dir) / out_name
    json_path.write_text(json.dumps(base, indent=2))
    np.save(json_path.with_suffix(".npy"), T)
    print(f"  ✅ Saved: {json_path}  (+ .npy)")
    return True


def solve_and_save(cam: _Cam, out_dir: Path, mocap_scale: float, method: str):
    return _print_and_save(out_dir, CAM_CALIB_NAME[cam.cam_idx], cam.serial,
                           _intr_dict(cam.intrinsics), cam.points_mocap, cam.points_px,
                           cam.points_cam, cam.K, mocap_scale, method, cam.label)


def dump_points(out_dir, cams, mocap_scale, method):
    """Write all cameras' raw correspondences to a timestamped JSON BEFORE solving,
    so a solver error can never lose the clicks. Returns the dump path."""
    data = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": method, "mocap_scale": mocap_scale, "cameras": [],
    }
    for cam in cams:
        data["cameras"].append({
            "cam_idx": cam.cam_idx, "serial": cam.serial, "label": cam.label,
            "out_name": CAM_CALIB_NAME[cam.cam_idx],
            "intrinsics": _intr_dict(cam.intrinsics),
            "points_px": [list(map(float, p)) for p in cam.points_px],
            "points_mocap_mm": [list(map(float, p)) for p in cam.points_mocap],
            "points_cam_mm": [list(map(float, p)) if p is not None else None
                              for p in cam.points_cam],
        })
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calib_points_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\n💾 Raw correspondences saved: {path}")
    return path


def reprocess(dump_or_json, out_dir, method):
    """Re-solve cameras from a saved dump (calib_points_*.json) or a single
    per-camera T_mocap_to_cam_*.json. Lets you change --method or recover after a
    crash without re-collecting points."""
    data = json.loads(Path(dump_or_json).read_text())
    if "cameras" in data:                      # a multi-camera dump
        scale = data.get("mocap_scale", 1000.0)
        cams = data["cameras"]
    else:                                      # a single per-camera result file
        scale = data.get("mocap_scale", 1000.0)
        serial = data["camera_serial"]
        out_name = next((n for i, n in CAM_CALIB_NAME.items()
                         if n in str(dump_or_json)), Path(dump_or_json).name)
        cams = [{"serial": serial, "label": out_name, "out_name": out_name,
                 "intrinsics": data["intrinsics"],
                 "points_px": data.get("points_px", []),
                 "points_mocap_mm": data.get("points_mocap_mm", []),
                 "points_cam_mm": data.get("points_cam_mm", [])}]
    print(f"Reprocessing {len(cams)} camera(s) from {dump_or_json}  (method={method})")
    for c in cams:
        P_cam = [np.array(p) if p is not None else None
                 for p in c.get("points_cam_mm", [])]
        _print_and_save(out_dir, c["out_name"], c["serial"], c["intrinsics"],
                        [np.array(p) for p in c["points_mocap_mm"]],
                        c.get("points_px", []), P_cam, _K_from_intr(c["intrinsics"]),
                        scale, method, c["label"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Calibrate OptiTrack → all RealSense cameras together.")
    ap.add_argument("--serials", nargs="*", default=REALSENSE_SERIALS,
                    help="RealSense serials in H5 cam-index order "
                         f"(default: {REALSENSE_SERIALS})")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["left", "middle", "right"],
                    help="Calibrate only these view(s) (e.g. --only left). Other "
                         "cameras' existing calibration files are left untouched.")
    ap.add_argument("--num_points", type=int, default=8,
                    help="Number of calibration points to collect (≥4 per camera).")
    ap.add_argument("--method", choices=["pnp", "svd"], default="pnp",
                    help="pnp (2D-3D, depth-free, default) or svd (3D-3D, depth-based).")
    ap.add_argument("--source", choices=["ir", "color"], default="ir",
                    help="Detection image: 'ir' (default; retroreflective markers "
                         "pop, mapped to the color pixel via depth+extrinsics) or "
                         "'color' (detect directly in RGB).")
    ap.add_argument("--mocap_body", type=str, default=None,
                    help="VRPN rigid-body name to auto-grab the ball position from "
                         "the live stream (e.g. 'calib_ball'). Omit to type x y z.")
    ap.add_argument("--mocap_scale", type=float, default=1000.0,
                    help="OptiTrack units → mm (default 1000, i.e. metres).")
    ap.add_argument("--out_dir", type=str,
                    default=str(Path(__file__).resolve().parent / "result"),
                    help="Directory for the per-view T_mocap_to_cam_*.json files.")
    ap.add_argument("--depth_patch", type=int, default=7,
                    help="Patch size for median depth reading (SVD only; default 7).")
    ap.add_argument("--no_snap", action="store_true",
                    help="Disable click-to-snap; use the raw click pixel as-is.")
    ap.add_argument("--snap_radius", type=float, default=25.0,
                    help="Max distance (px) to snap a click to a detected marker.")
    ap.add_argument("--blob_thresh", type=int, default=None,
                    help="Top-hat contrast threshold for detection. Default: seeded "
                         "per camera (ir≈25 fixed; color=adaptive). Tunable live.")
    ap.add_argument("--blob_min_area", type=int, default=5,
                    help="Min blob area (px²) for marker detection.")
    ap.add_argument("--blob_max_area", type=int, default=800,
                    help="Max blob area (px²) for marker detection.")
    ap.add_argument("--reprocess", type=str, default=None,
                    help="Re-solve from a saved calib_points_*.json dump (or a "
                         "single T_mocap_to_cam_*.json) without opening cameras. "
                         "Combine with --method to switch solver.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    # Offline re-solve path — no cameras needed.
    if args.reprocess:
        reprocess(args.reprocess, out_dir, args.method)
        return

    mocap = None
    if args.mocap_body:
        print(f"Auto-capture mocap from VRPN body '{args.mocap_body}' "
              f"(/vrpn_client_node/{args.mocap_body}/pose).")
        mocap = _MocapReader(args.mocap_body)

    print(f"Solver: {args.method.upper()}   Source: {args.source.upper()}   "
          f"Snap: {'OFF' if args.no_snap else f'ON (r={args.snap_radius:.0f}px)'}   Cameras:")
    cams = []
    for cam_idx, serial in enumerate(args.serials):
        if args.only and _view_label(cam_idx) not in args.only:
            continue
        print(f"  cam{cam_idx} ({_view_label(cam_idx)}): {serial}")
        cams.append(_Cam(cam_idx, serial, source=args.source, snap=not args.no_snap,
                         snap_radius=args.snap_radius,
                         blob_min_area=args.blob_min_area, blob_max_area=args.blob_max_area))

    def _make_cb(cam: _Cam):
        def _cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cam.pending_raw = (x, y)
        return _cb

    for i, cam in enumerate(cams):
        cv2.namedWindow(cam.win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(cam.win, 30 + i * 660, 40)
        cv2.setMouseCallback(cam.win, _make_cb(cam))

    show_mask = False

    print("\n⏳ Warming up cameras...")
    for _ in range(30):
        for cam in cams:
            cam.wait_frames()
            cam.render(show_mask)
        cv2.waitKey(33)

    # Seed each camera's detection threshold, then add a manual 'thresh' slider on
    # each window for hand-tuning. The slider is the source of truth; its callback
    # writes blob_thresh. IR response is uniform across cameras, so a low fixed
    # seed works; RGB exposure varies, so seed adaptively per camera.
    def _make_thr_cb(cam):
        def _cb(val):
            cam.blob_thresh = int(val)
        return _cb

    for cam in cams:
        if args.blob_thresh is not None:
            cam.blob_thresh = args.blob_thresh
        elif args.source == "ir":
            cam.blob_thresh = 25
        else:
            cam.blob_thresh = adaptive_thresh(cam.det_gray)
        cv2.createTrackbar("thresh", cam.win, cam.blob_thresh, 255, _make_thr_cb(cam))
    # One more frame so blobs recompute at the seeded threshold before we report.
    for cam in cams:
        cam.wait_frames()
        cam.render(show_mask)

    print("\nDetection check (per camera):")
    for cam in cams:
        tmax = int(cam.tophat.max()) if cam.tophat is not None else 0
        flag = "" if cam.blobs else "  ← no markers detected!"
        print(f"  cam{cam.cam_idx} ({cam.label}): blobs={len(cam.blobs)}  "
              f"contrast_max={tmax}  thr={cam.blob_thresh} (seed){flag}")
    if any(not c.blobs for c in cams):
        print("  Tip: drag the 'thresh' slider on that window (or 'm' to view the "
              "mask) until its markers appear.")

    print("\nDetected markers are ringed in cyan; a click snaps to the nearest one.")
    print("Per-point controls (focus any camera window):")
    print("  click=mark marker   [n]/SPACE=next point   [r]=clear clicks   "
          "[u]=undo last point   [m]=toggle mask   thresh slider per window   "
          "[q]=finish+save\n")

    def _get_mocap_mm(seers):
        """Return the shared mocap point (mm) for this round, or None to retry."""
        if mocap is not None:
            pos = mocap.latest()
            if pos is None:
                print(f"   ⚠ no fresh '{args.mocap_body}' pose from VRPN — "
                      f"bring the ball into the volume and press [n] again.")
                return None
            p = pos * args.mocap_scale
            print(f"   auto mocap: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] mm")
            return p
        while True:
            try:
                line = input("   OptiTrack position (x y z): ")
                return parse_vector(line)[:3] * args.mocap_scale
            except (ValueError, IndexError) as e:
                print(f"   ⚠ parse error: {e}. Try again.")

    collected = 0
    try:
        while collected < args.num_points:
            print(f"── Point {collected + 1}/{args.num_points} "
                  f"─ place ball, click in each view that sees it ─")
            for cam in cams:
                cam.clear_pending()

            advance = quit_now = skip_commit = False
            while not advance:
                for cam in cams:
                    cam.wait_frames()
                    cam.resolve_pending(args.depth_patch)
                    cam.render(show_mask)

                key = cv2.waitKey(20) & 0xFF
                if key in (ord("n"), ord(" ")):
                    if any(c.pending is not None for c in cams):
                        advance = True
                    else:
                        print("   (no camera has a click yet — click the ball or [q] to finish)")
                elif key == ord("m"):
                    show_mask = not show_mask
                    print(f"   threshold mask: {'ON' if show_mask else 'OFF'}")
                elif key in (ord("-"), ord("_")):
                    for cam in cams:  # nudge all sliders down (callback sets thresh)
                        cv2.setTrackbarPos("thresh", cam.win, max(0, cam.blob_thresh - 5))
                elif key in (ord("="), ord("+")):
                    for cam in cams:  # nudge all sliders up
                        cv2.setTrackbarPos("thresh", cam.win, min(255, cam.blob_thresh + 5))
                elif key == ord("r"):
                    for cam in cams:
                        cam.clear_pending()
                    print("   cleared this point's clicks.")
                elif key == ord("u"):
                    _undo_last_point(cams)
                    for cam in cams:
                        cam.clear_pending()
                    collected = max(0, collected - 1)
                    print(f"   undid last point (now {collected} recorded).")
                    skip_commit = advance = True
                elif key == ord("q"):
                    quit_now = advance = True

            if quit_now:
                print("Finishing early — solving with points collected so far.")
                break
            if skip_commit:
                continue

            seers = [c for c in cams if c.pending is not None]
            print(f"   {len(seers)} camera(s) saw it: {', '.join(c.label for c in seers)}")
            p_mocap = _get_mocap_mm(seers)
            if p_mocap is None:
                continue  # auto-capture stale; keep clicks, let user retry

            for cam in seers:
                cam.commit_point(p_mocap)
            collected += 1
            print(f"   ✅ Point {collected} recorded for "
                  f"{', '.join(c.label for c in seers)}.\n")
    finally:
        for cam in cams:
            cam.stop()
        cv2.destroyAllWindows()

    # Persist the raw correspondences BEFORE solving, so a solver error can never
    # lose the clicks (recover/retry later with --reprocess).
    dump_path = dump_points(out_dir, cams, args.mocap_scale, args.method)

    print("\n" + "#" * 60)
    print(f"# Solving per-camera transforms (method={args.method})")
    print("#" * 60)
    for cam in cams:
        try:
            solve_and_save(cam, out_dir, args.mocap_scale, args.method)
        except Exception as e:           # belt-and-suspenders: never abort the rest
            print(f"[{cam.label}] ❌ unexpected solve error: {e} (skipped; raw points "
                  f"are in {dump_path.name}).")

    print("\nDone. Per-camera point counts:")
    for cam in cams:
        print(f"  {cam.label}: {len(cam.points_px)} points")
    print(f"Raw points dump: {dump_path}")
    print(f"Re-solve anytime with:  python -m twm.calibration.mocap_to_cam_multi "
          f"--reprocess {dump_path}  [--method pnp|svd]")


def _undo_last_point(cams):
    """Remove the most recently recorded point from every camera that has it."""
    if not any(c.points_px for c in cams):
        return
    maxlen = max(len(c.points_px) for c in cams)
    for c in cams:
        if len(c.points_px) == maxlen and maxlen > 0:
            c.points_det.pop()
            c.points_px.pop()
            c.points_cam.pop()
            c.points_mocap.pop()


if __name__ == "__main__":
    main()
