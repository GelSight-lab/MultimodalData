"""Visualization helpers — all return RGB uint8 (H, W, 3) images you can
imshow / save, no display side effects.
"""
from __future__ import annotations

import numpy as np

from .reference import difference, l2_diff
from .contact import contact_mask


def _colormap(gray01, name="viridis"):
    import cv2
    g = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    cmaps = {"viridis": cv2.COLORMAP_VIRIDIS, "jet": cv2.COLORMAP_JET,
             "magma": cv2.COLORMAP_MAGMA, "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)}
    bgr = cv2.applyColorMap(g, cmaps.get(name, cv2.COLORMAP_VIRIDIS))
    return bgr[..., ::-1].copy()   # BGR->RGB


def diff_heatmap(frame, reference, vmax=None, cmap="turbo"):
    """L2 difference-from-reference as a heatmap (contact intensity)."""
    d = l2_diff(frame, reference)
    vmax = vmax if vmax is not None else max(d.max(), 1e-6)
    return _colormap(d / vmax, cmap)


def contact_overlay(frame, reference, tau=8.0, color=(255, 0, 0), alpha=0.45):
    """Tint the contact region on top of the raw frame."""
    m = contact_mask(frame, reference, tau=tau)
    out = frame.copy().astype(np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def reference_compare(frame, reference):
    """Side-by-side [reference | frame | signed-diff] strip."""
    sd = difference(frame, reference, signed=True)
    return np.concatenate([reference, frame, sd], axis=1)


def depth_view(height_map, cmap="gray"):
    """Render a (H, W) height map as an RGB image.

    Default is **grayscale** (brighter = higher), the standard GelSight
    height-map convention (gsrobotics, GelSight Wedge, depth-recon papers).
    Pass cmap="turbo"/"jet"/"viridis" for a colormapped view instead.
    """
    h = height_map.astype(np.float32)
    rng = h.max() - h.min()
    norm = (h - h.min()) / (rng + 1e-6)
    if cmap in (None, "gray", "grey", "grayscale"):
        g = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)   # (H,W,3) gray RGB
    return _colormap(norm, cmap)


def height_to_pointcloud(height_map, stride=4, z_scale=1.0):
    """(H, W) height -> (N, 3) point cloud (x, y, z) for 3D rendering."""
    h = height_map[::stride, ::stride]
    ys, xs = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    return np.stack([xs.ravel(), ys.ravel(), (h * z_scale).ravel()], axis=1).astype(np.float32)


# ── camera-view projection, for checking your own geometry ──────────────────
# Force -> disc radius. Area is linear in force: human size judgement of a
# filled disc tracks its area, so radius ∝ F would exaggerate large forces
# roughly quadratically. Same law the published preview videos use.
F_FULL_N, R_MIN_PX, R_MAX_PX = 8.0, 3.0, 22.0
TARGET_GAIN = 40.0          # drawn gap exaggeration; see draw_projection


def force_radius_px(force_n):
    """THE force -> pixel-radius law, so a legend cannot drift from a disc."""
    f = max(float(force_n), 0.0)
    return R_MIN_PX + (R_MAX_PX - R_MIN_PX) * (min(f, F_FULL_N) / F_FULL_N) ** 0.5


def draw_projection(frame_rgb, sensor_pose7, gel_center_mm, cam_calib,
                    force_n=None, target_pose7=None, gain=TARGET_GAIN,
                    label=None):
    """Draw where the sensor projects into this camera view. Returns a copy.

    WHY THE TOOLBOX DRAWS AT ALL

    A coordinate is a weak debugging aid for a geometry problem. Knowing that
    `project_gel_to_pixel` returned (366, 188) says nothing about whether that
    is ON the sensor, and every projection defect this dataset has shipped was
    obvious in a picture and invisible in a number:

        wrong calibration epoch            35-73 px
        gel centre defaulted to the origin 21-36 px
        world offset not applied          155-223 px

    All three look like a slightly miscalibrated rig, which is exactly why
    they survived. Render one frame and they stop looking like that.

    Draws, when given: the gel centre (dot), the pressing normal (line), the
    press force (translucent disc, area linear in newtons) and the DexForce
    virtual target (ring joined to the dot).

    ONE PROJECTION. Every element goes through `calibration.project_gel_to_pixel`
    — the same call you use — so the picture cannot disagree with the number
    you got from the library.

    `gain` exaggerates the drawn target OFFSET only, and is printed on the
    image. At true scale it is invisible: force/k is millimetres while the
    view spans a metre, measured p50 0.00 px and max 1.41 px on a real
    episode, against a force disc of radius up to 22 px. An unlabelled
    exaggeration is a false statement about a distance.
    """
    import cv2

    from .calibration import project_gel_to_pixel

    out = np.ascontiguousarray(frame_rgb).copy()
    uv = project_gel_to_pixel(sensor_pose7, gel_center_mm, cam_calib)
    if uv is None:                      # behind the camera: draw nothing.
        return out                      # A wrapped coordinate is not a position.
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = out.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return out

    if force_n is not None and float(force_n) > 0.02:
        r = int(round(force_radius_px(force_n)))
        layer = out.copy()
        cv2.circle(layer, (u, v), r, (255, 120, 60), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.42, out, 0.58, 0, out)
        cv2.circle(out, (u, v), r, (255, 120, 60), 1, cv2.LINE_AA)
        cv2.putText(out, f"{float(force_n):.1f}N", (u + r + 4, v - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 60), 1,
                    cv2.LINE_AA)

    if target_pose7 is not None:
        t = np.asarray(target_pose7, np.float64).copy()
        o = np.asarray(sensor_pose7, np.float64)
        if not np.allclose(t[:3], o[:3]):        # force 0 -> target IS pose
            t[:3] = o[:3] + gain * (t[:3] - o[:3])   # exaggerate in WORLD mm,
            tuv = project_gel_to_pixel(t, gel_center_mm, cam_calib)  # then project
            if tuv is not None:
                tu, tv = int(round(tuv[0])), int(round(tuv[1]))
                cv2.line(out, (u, v), (tu, tv), (220, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(out, (tu, tv), 5, (220, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(out, f"target x{gain:g}", (tu + 7, tv + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 0, 255), 1,
                            cv2.LINE_AA)

    cv2.circle(out, (u, v), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(out, (u, v), 5, (0, 200, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(out, str(label), (u + 8, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1,
                    cv2.LINE_AA)
    return out
