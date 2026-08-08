"""Single source of truth for TWM visualization.

All TWM visual artifacts (the live recording UI, the replay viewer, the per-
episode preview GIFs, the dataloader demo GIFs, the freeze-inspection clips)
share the same panel builder and the same projection overlay. Consumers
import from this module instead of redefining their own layout helpers.

Layout in one place:
  * `build_preview_panel(color_frames, gs_frames, gs_ref, optitrack_poses, ...)`
    produces the canonical 1280×480 BGR panel that the live viewer shows.
    `color_frames` is keyed by H5 cam index (0/1/2 == right/left/middle per
    `REALSENSE_SERIALS` in `twm.data_collection`); this function reorders to
    the **left | middle | right** display order so downstream consumers see
    the same operator-intuitive ordering across viewer and GIF outputs.
  * `draw_projection_overlay` consumes that panel and draws the GelSight
    center + axes overlay on each camera thumbnail, with an optional red
    "FROZEN" ring on a specified sensor for failure-mode inspection.
  * `save_gif` is the adaptive-shrinking PIL encoder used by every GIF
    pipeline in the repo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from twm.force_overlay import draw_force_halo


# ──────────────────────────────────────────────────────────────────────────────
# Display configuration
# ──────────────────────────────────────────────────────────────────────────────

# H5 stores cameras in REALSENSE_SERIALS order:
#   cam0 = serial 143...538 (right)
#   cam1 = serial 104...574 (left)
#   cam2 = serial 217...989 (middle)
# We display them in spatial left → right order, which means cam1, cam2, cam0.
DISPLAY_ORDER: list[int] = [1, 2, 0]
DISPLAY_LABELS: list[str] = ["left cam", "middle cam", "right cam"]
# Inverse mapping: H5 cam_idx → its slot (0, 1, or 2) on the displayed panel.
DISPLAY_POSITION: dict[int, int] = {cam_idx: pos for pos, cam_idx in enumerate(DISPLAY_ORDER)}

# Default calibration FILENAMES for each H5 cam_idx (the directory is the
# task's epoch, from `calib_epoch`). Kept here so visualization tools don't have to
# repeat the mapping.
CAM_CALIB_NAME: dict[int, str] = {
    0: "T_mocap_to_cam_right.json",
    1: "T_mocap_to_cam_left.json",
    2: "T_mocap_to_cam_middle.json",
}

# Layout constants — match what the live recording UI used historically.
RS_THUMB_W, RS_THUMB_H = 320, 240   # one RealSense thumbnail
GS_THUMB_W, GS_THUMB_H = 240, 240   # one GelSight thumbnail (raw or diff)
ROW2_Y = RS_THUMB_H                 # top of row 2 (the tactile strip)
PANEL_W, PANEL_H = 1280, 480

# OptiTrack tracker colors (BGR, drawn into the BGR panel).
TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
    "motherboard":  (255, 200,   0),
    "sensor_left":  (  0, 255, 120),
    "sensor_right": (  0, 180, 255),
}

# Sensor-dot colors used by `draw_projection_overlay`. Wong palette
# (colorblind-safe). BGR for cv2.
GEL_DOT_BGR: dict[str, tuple[int, int, int]] = {
    "left":  (  0, 159, 230),   # Wong orange in BGR
    "right": (178, 114,   0),   # Wong blue in BGR
}
FROZEN_BGR: tuple[int, int, int] = (0, 0, 255)
AXIS_BGR = [(0, 0, 255), (0, 255, 0), (255, 128, 0)]
AXIS_LABELS = ["X", "Y", "Z"]


# ──────────────────────────────────────────────────────────────────────────────
# Calibration loading / pose math
# ──────────────────────────────────────────────────────────────────────────────

def pose7_to_T(pose_7) -> Optional[np.ndarray]:
    """7-vec (x,y,z in meters, qx,qy,qz,qw) → 4×4 pose matrix in millimeters.

    Returns None if the quaternion has zero norm (= no valid pose, e.g. an
    OptiTrack frame where the body was never tracked).
    """
    p = np.asarray(pose_7, np.float64)
    pos_mm = p[:3] * 1000.0
    q = p[3:]
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-9:
        return None
    q = q / n
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = pos_mm
    return T


def project_gel_pose(rigid_pose_7, gel_center_in_rigid_mm,
                     T_mocap_to_cam, intrinsics, axis_len_mm: float = 120.0):
    """Project a sensor's GelSight surface center + body axes into one camera.

    Returns (center_pixel, [x_tip, y_tip, z_tip]) in **native** camera pixel
    coords (e.g. (u, v) within 640×480), or None when off-image or pose is
    invalid. The downstream `draw_projection_overlay` scales these to the
    thumbnail's actual size on the panel.
    """
    if rigid_pose_7 is None:
        return None
    T_r2m = pose7_to_T(rigid_pose_7)
    if T_r2m is None:
        return None
    if np.allclose(rigid_pose_7, 0.0):
        return None
    P_gel = (T_r2m @ np.append(gel_center_in_rigid_mm, 1.0))[:3]
    R = T_r2m[:3, :3]
    tips_world = [
        P_gel + R @ np.array([axis_len_mm, 0, 0]),
        P_gel + R @ np.array([0, axis_len_mm, 0]),
        P_gel + R @ np.array([0, 0, axis_len_mm]),
    ]

    def _project(P_mocap_mm):
        P_cam = (T_mocap_to_cam @ np.append(P_mocap_mm, 1.0))[:3]
        if P_cam[2] <= 0:
            return None
        u = intrinsics["fx"] * P_cam[0] / P_cam[2] + intrinsics["ppx"]
        v = intrinsics["fy"] * P_cam[1] / P_cam[2] + intrinsics["ppy"]
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return float(u), float(v)

    center = _project(P_gel)
    if center is None:
        return None
    axes = [_project(t) for t in tips_world]
    return center, axes


def load_calibrations(cam_calib_paths: Iterable[str | Path],
                      gel_left_path: str | Path,
                      gel_right_path: str | Path):
    """Load camera + gel calibrations from JSON files.

    Returns (cam_calibs, gel_center_left, gel_center_right) where
    `cam_calibs` is a list of dicts with keys: T_mocap_to_cam, intrinsics,
    camera_serial, rmse_mm, path.
    """
    cam_calibs = []
    for path in cam_calib_paths:
        p = Path(path)
        if not p.is_file():
            print(f"Notice: Calibration file not found: {p} (skipping)")
            continue
        d = json.loads(p.read_text())
        cam_calibs.append({
            "T_mocap_to_cam": np.array(d["T_mocap_to_cam"], np.float64),
            "intrinsics":     d["intrinsics"],
            "camera_serial":  d["camera_serial"],
            "rmse_mm":        d.get("rmse_mm", 0.0),
            "path":           str(p),
        })
    gel_L = np.array(
        json.loads(Path(gel_left_path).read_text())["gel_center_in_rigid_mm"],
        np.float64,
    )
    gel_R = np.array(
        json.loads(Path(gel_right_path).read_text())["gel_center_in_rigid_mm"],
        np.float64,
    )
    return cam_calibs, gel_L, gel_R


# ──────────────────────────────────────────────────────────────────────────────
# OptiTrack helpers (per-frame interpolation)
# ──────────────────────────────────────────────────────────────────────────────

def load_optitrack(f) -> dict:
    """Read the full motherboard / sensor_left / sensor_right OT streams from
    an open h5py.File. Returns {name: (timestamps, poses)} or None per body
    when there's no data."""
    out = {}
    for name in ("motherboard", "sensor_left", "sensor_right"):
        ts = f[f"optitrack/{name}/timestamps"][:]
        poses = f[f"optitrack/{name}/pose"][:]
        out[name] = (ts, poses) if len(ts) > 0 else None
    return out


def optitrack_at(lookup: dict, camera_timestamp: float) -> dict:
    """Nearest-neighbor lookup of each tracker's pose at a camera timestamp.
    Returns {name: (matched_ts, [x,y,z,qx,qy,qz,qw])} or {name: None}.
    """
    res = {}
    for name, data in lookup.items():
        if data is None:
            res[name] = None
            continue
        ts, poses = data
        idx = int(np.searchsorted(ts, camera_timestamp))
        if idx >= len(ts):
            idx = len(ts) - 1
        elif idx > 0 and abs(ts[idx - 1] - camera_timestamp) <= abs(ts[idx] - camera_timestamp):
            idx -= 1
        res[name] = (float(ts[idx]), poses[idx].tolist())
    return res


def cam_aligned_pose(cam_ts: np.ndarray, opt_ts: np.ndarray, opt_pose: np.ndarray) -> np.ndarray:
    """Vectorized nearest-neighbor: produce per-camera-frame poses from an OT stream.
    Identical alignment to what `twm.contact_index` bakes into the `.pt` files."""
    if len(opt_ts) == 0:
        return np.zeros((len(cam_ts), opt_pose.shape[1]), opt_pose.dtype)
    idx = np.searchsorted(opt_ts, cam_ts)
    idx = np.clip(idx, 1, len(opt_ts) - 1)
    left = cam_ts - opt_ts[idx - 1]
    right = opt_ts[idx] - cam_ts
    pick = np.where(right < left, idx, idx - 1)
    return opt_pose[pick]


# ──────────────────────────────────────────────────────────────────────────────
# Panel builders
# ──────────────────────────────────────────────────────────────────────────────

def make_optitrack_panel(optitrack_poses, w: int = RS_THUMB_W, h: int = RS_THUMB_H) -> np.ndarray:
    """Render the right-most text panel showing live tracker pose (x,y,z + q)."""
    panel = np.zeros((h, w, 3), np.uint8)
    cv2.putText(panel, "OptiTrack", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y = 48
    for name, color in TRACKER_COLORS.items():
        pose = optitrack_poses.get(name) if optitrack_poses else None
        cv2.putText(panel, name, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 18
        if pose is None:
            cv2.putText(panel, "  no data", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        else:
            _, xyz_quat = pose
            x_m, y_m, z_m = xyz_quat[:3]
            qx, qy, qz, qw = xyz_quat[3:]
            cv2.putText(panel, f"  x={x_m:+.3f} y={y_m:+.3f} z={z_m:+.3f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            y += 16
            cv2.putText(panel, f"  qx={qx:+.2f} qy={qy:+.2f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
            y += 16
            cv2.putText(panel, f"  qz={qz:+.2f} qw={qw:+.2f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        y += 24
    return panel


def build_preview_panel(color_frames, gs_frames, gs_ref, optitrack_poses,
                         recording: bool, frame_count: int, elapsed: float,
                         buf: int = 0, fps: float = 0.0, task_name: str = "",
                         status_override: Optional[str] = None) -> np.ndarray:
    """The canonical 1280×480 BGR panel used everywhere.

    Layout:
      Row 1 (y=0..240):    [cam slot 0 | slot 1 | slot 2 | OptiTrack text]
      Row 2 (y=240..480):  [gs_L_raw | gs_L_diff | gs_R_raw | gs_R_diff | blank]

    `color_frames` is indexed by H5 cam_idx (0=right, 1=left, 2=middle as per
    `REALSENSE_SERIALS`). This function reorders to the spatial **left,
    middle, right** display order (`DISPLAY_ORDER`).

    Pass `status_override=str` to draw a custom string in the bottom status
    bar (used by GIF tooling); otherwise the standard REC/IDLE bar is drawn.
    """
    # INTER_NEAREST is ~3x faster than the default INTER_LINEAR on these
    # sizes; at thumbnail resolution the quality difference is invisible.
    def _rs_thumb(img):
        return cv2.resize(img, (RS_THUMB_W, RS_THUMB_H), interpolation=cv2.INTER_NEAREST)

    def _gs_thumb(img):
        return cv2.resize(img, (GS_THUMB_W, GS_THUMB_H), interpolation=cv2.INTER_NEAREST)

    def _diff_thumb(frame, ref):
        diff = np.clip(frame.astype(np.int16) - ref.astype(np.int16) + 128, 0, 255).astype(np.uint8)
        return cv2.resize(diff, (GS_THUMB_W, GS_THUMB_H), interpolation=cv2.INTER_NEAREST)

    # Row 1: display-ordered cams + OT text panel
    displayed_cams = [_rs_thumb(color_frames[i]) for i in DISPLAY_ORDER]
    ot_panel = make_optitrack_panel(optitrack_poses, w=RS_THUMB_W, h=RS_THUMB_H)
    row1 = np.hstack(displayed_cams + [ot_panel])

    # Row 2: 4 GelSight thumbs (L raw / L diff / R raw / R diff) + blank
    gs_panels = []
    for frame, ref in zip(gs_frames, gs_ref):
        gs_panels.append(_gs_thumb(frame))
        gs_panels.append(_diff_thumb(frame, ref))
    blank = np.zeros((GS_THUMB_H, RS_THUMB_W, 3), np.uint8)
    row2 = np.hstack(gs_panels + [blank])

    panel = np.vstack([row1, row2])

    # Bottom status bar
    if status_override is not None:
        status = status_override
        color = (200, 200, 200)
    else:
        task_prefix = f"[{task_name}]  " if task_name else ""
        if recording:
            status = f"{task_prefix}[REC ep_{frame_count:04d} | {frame_count} frames | {elapsed:.1f}s | buf={buf} | {fps:.1f}fps]"
            color = (0, 0, 220)
        else:
            status = f"{task_prefix}[IDLE]  s=start  e=end  r=reset-ref  q=quit  |  {fps:.1f}fps"
            color = (0, 200, 0)

    cv2.putText(panel, status, (10, panel.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Tile-label headers at the top of each cam thumb
    for slot, lab in enumerate(DISPLAY_LABELS):
        x = slot * RS_THUMB_W + 8
        cv2.putText(panel, lab, (x, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Projection overlay (cam thumbnails on row 1)
# ──────────────────────────────────────────────────────────────────────────────

def _scale_to_thumb(u: float, v: float,
                    src_w: int = 640, src_h: int = 480,
                    thumb_w: int = RS_THUMB_W, thumb_h: int = RS_THUMB_H):
    return int(u * thumb_w / src_w), int(v * thumb_h / src_h)


def draw_projection_overlay(panel: np.ndarray,
                             optitrack_poses: dict,
                             project_cams: list[dict],
                             gel_center_left: np.ndarray,
                             gel_center_right: np.ndarray,
                             *,
                             frozen_side: Optional[str] = None,
                             forces_n: Optional[dict] = None,
                             axis_len_mm: float = 120.0) -> None:
    """Draw GelSight projection (center dot + 3 axis tips) on each cam thumb.

    Rendered at 2x supersample then INTER_AREA-downsized so axis line widths
    are effectively 1.5 px (cv2 thickness is integer-only).

    `project_cams` is a list of {"index": cam_idx, "T_mocap_to_cam": ...,
    "intrinsics": ...}. `cam_idx` is the H5 index; this function uses
    `DISPLAY_POSITION` to place the dot on the correct displayed thumbnail.

    If `frozen_side ∈ {"left", "right"}`, a second red ring + label is drawn
    on top of that sensor's dot in every cam.

    `forces_n` = {"left": N, "right": N} adds the semi-transparent press-force
    disc, centred on the SAME projected point as that sensor's axes. It is
    drawn here rather than by the caller for exactly that reason: two
    independent projections of the sensor would eventually disagree, and the
    disagreement would look like a calibration error rather than a drawing
    bug. Drawn before the axes so the crisp pose marker stays readable
    through it.

    Modifies `panel` in place.
    """
    sl = optitrack_poses.get("sensor_left") if optitrack_poses else None
    sr = optitrack_poses.get("sensor_right") if optitrack_poses else None

    # 2x supersample buffer for sub-integer line widths.
    SCALE = 2
    H, W = panel.shape[:2]
    big = cv2.resize(panel, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)

    for pc in project_cams:
        cam_idx = pc["index"]
        slot = DISPLAY_POSITION.get(cam_idx)
        if slot is None:
            continue
        x_offset = slot * RS_THUMB_W

        for side, pose_tuple, gel in [("left", sl, gel_center_left),
                                       ("right", sr, gel_center_right)]:
            if pose_tuple is None:
                continue
            res = project_gel_pose(
                pose_tuple[1], gel, pc["T_mocap_to_cam"], pc["intrinsics"],
                axis_len_mm=axis_len_mm,
            )
            if res is None:
                continue
            center, axes = res
            cx, cy = _scale_to_thumb(center[0], center[1])
            cx += x_offset
            cx_b, cy_b = cx * SCALE, cy * SCALE
            # Press force, under the axes, at this same projected centre.
            if forces_n and forces_n.get(side) is not None:
                draw_force_halo(big, (cx, cy), float(forces_n[side]),
                                scale=SCALE,
                                bounds=(x_offset, 0,
                                        x_offset + RS_THUMB_W, RS_THUMB_H))
            # Axes (thickness 3 on 2x canvas = 1.5 effective px after downsize)
            for tip, ac, al in zip(axes, AXIS_BGR, AXIS_LABELS):
                if tip is None:
                    continue
                tx, ty = _scale_to_thumb(tip[0], tip[1])
                tx += x_offset
                tx_b, ty_b = tx * SCALE, ty * SCALE
                cv2.line(big, (cx_b, cy_b), (tx_b, ty_b), ac, 3, cv2.LINE_AA)
                if 0 <= tx < W and 0 <= ty < H:
                    cv2.putText(big, al, (tx_b + 6, ty_b - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.64, ac, 2, cv2.LINE_AA)
            # Dot (effective: inner r=2, outer r=3 -> doubled to 4 / 6 on 2x canvas)
            dot = GEL_DOT_BGR[side]
            cv2.circle(big, (cx_b, cy_b), 4, dot, -1, cv2.LINE_AA)
            cv2.circle(big, (cx_b, cy_b), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(big, side[0].upper(), (cx_b + 10, cy_b + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.64, dot, 2, cv2.LINE_AA)
            # FROZEN marker (effective r=6, stroke 1.5 -> 12 / 3 on 2x canvas)
            if frozen_side is not None and side == frozen_side:
                cv2.circle(big, (cx_b, cy_b), 12, FROZEN_BGR, 3, cv2.LINE_AA)

    panel[:] = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────────────────────────────────────
# GIF encoder (one source for the adaptive shrink loop)
# ──────────────────────────────────────────────────────────────────────────────

def save_gif(frames_rgb: list[np.ndarray], out_path: str | Path, *,
             fps: float = 15.0, max_kb: int = 4096,
             palette_colors: int = 256, min_palette: int = 128) -> int:
    """Encode an RGB frame list as a GIF, shrinking palette only as a last
    resort. Returns the on-disk size in bytes.

    The palette is built from a mosaic of evenly-spaced sample frames so the
    colors found late in the sequence aren't lost. Adaptive shrink: try
    `palette_colors` first; on overflow drop palette by 32 until `min_palette`.
    Spatial downscaling is the caller's responsibility (kept out of here so
    each artifact can choose its own aspect-preserving strategy).
    """
    out_path = Path(out_path)
    if not frames_rgb:
        raise ValueError("save_gif: empty frame list")

    pil = [Image.fromarray(f) for f in frames_rgb]
    pick = np.linspace(0, len(pil) - 1, min(16, len(pil))).astype(int)
    mosaic = Image.new("RGB", (pil[0].width * len(pick), pil[0].height))
    for i, k in enumerate(pick):
        mosaic.paste(pil[k], (i * pil[0].width, 0))

    colors = palette_colors
    while True:
        pal = mosaic.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
        q = [im.quantize(palette=pal, dither=Image.NONE) for im in pil]
        q[0].save(out_path, save_all=True, append_images=q[1:],
                  duration=int(round(1000 / fps)), loop=0,
                  optimize=True, disposal=0)
        size = out_path.stat().st_size
        if size <= max_kb * 1024 or colors <= min_palette:
            return size
        colors = max(min_palette, colors - 32)
