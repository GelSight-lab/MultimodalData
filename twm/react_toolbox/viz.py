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


def draw_probe(frame_rgb, probe, gel_center_mm, cam_calib, color=None,
               every: int = 6, label=True):
    """Overlay one synthetic probe's FUTURE trajectory on the start frame.

    The whole evaluation protocol for these probes is "look at it": there is
    no ground-truth future image, so a human compares the model's rollout
    against where the sensor WOULD be. That comparison needs the ground truth
    drawn in the same pixels, which is this.

    Start is a filled dot, end is a ring, and the path between them is a
    polyline sampled every `every` steps — dense enough to show curvature
    under perspective, sparse enough that the workpiece stays visible.

    Steps whose projection leaves the image are DROPPED from the polyline
    rather than clamped to the border. A clamped point is a position the
    sensor never occupies, and a line drawn to it says the trajectory went
    somewhere it did not.
    """
    import cv2

    from .calibration import project_gel_to_pixel

    out = np.ascontiguousarray(frame_rgb).copy()
    h, w = out.shape[:2]
    col = color or ((0, 220, 255) if probe.get("kind") == "translation"
                    else (255, 100, 220))

    pts = []
    for i, q in enumerate(np.asarray(probe["poses"], float)):
        if i % every and i != len(probe["poses"]) - 1:
            continue
        uv = project_gel_to_pixel(q, gel_center_mm, cam_calib)
        if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
            continue                      # dropped, never clamped
        pts.append((int(round(uv[0])), int(round(uv[1]))))

    for a, b in zip(pts, pts[1:]):
        cv2.line(out, a, b, col, 1, cv2.LINE_AA)
    if pts:
        cv2.circle(out, pts[0], 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pts[0], 4, col, 1, cv2.LINE_AA)
        cv2.circle(out, pts[-1], 6, col, 2, cv2.LINE_AA)
        if label:
            amp = (f"{probe['amplitude_m']*100:.0f}cm"
                   if probe.get("kind") == "translation"
                   else f"{probe['amplitude_deg']:.0f}deg")
            cv2.putText(out, f"{probe['name']} {amp}",
                        (pts[-1][0] + 8, pts[-1][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    return out


def probe_contact_sheet(frame_rgb, probes, gel_center_mm, cam_calib,
                        cols: int = 4, tile=(320, 240)):
    """One tile per probe, plus the numbers a human needs to judge it.

    A single frame with all twelve overlaid was the first version and it does
    not work: every trajectory starts at the same pixel, so the labels stack
    into an unreadable knot and no probe can be assessed on its own. The point
    of this picture is per-probe comparison against a model rollout, and that
    needs one probe per tile.

    Each tile prints the amplitude, the horizon and the speed percentile,
    because "is this like the dataset" is the first question a reader asks and
    it should not require opening the metadata.
    """
    import cv2

    tw, th = tile
    n = len(probes)
    rows = int(np.ceil(n / cols))
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, p in enumerate(probes):
        img = draw_probe(frame_rgb, p, gel_center_mm, cam_calib, label=False)
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        amp = (f"{p['amplitude_m']*100:.0f}cm" if p["kind"] == "translation"
               else f"{p['amplitude_deg']:.0f}deg")
        tag = (f"{p['name']}  {amp}  {p['horizon_s']:.1f}s  "
               f"p{p['speed_percentile']:.0f}")
        col = (0, 220, 255) if p["kind"] == "translation" else (255, 100, 220)
        cv2.rectangle(img, (0, 0), (tw - 1, 14), (0, 0, 0), -1)
        cv2.putText(img, tag, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1,
                    cv2.LINE_AA)
        if not p.get("in_view", True):
            cv2.rectangle(img, (0, 0), (tw - 1, th - 1), (0, 0, 255), 2)
            cv2.putText(img, "LEAVES VIEW", (3, th - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1,
                        cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = img
    return sheet


AXIS_BGR_RGB = ((255, 60, 60), (60, 255, 60), (80, 160, 255))   # x, y, z


def draw_sensor_frame(frame_rgb, sensor_pose7, gel_center_mm, cam_calib,
                      axis_len_mm: float = 60.0, label=None, dim=False,
                      stem: bool = False):
    """Draw the sensor's gel centre and its three body axes, in perspective.

    The axis tips are placed in 3D and projected, so the triad shrinks with
    distance and foreshortens with orientation — the same convention the
    React dataset previews use. A dot would show position and hide
    orientation, which makes the six ROTATION probes unreadable: under a pure
    rotation about the gel centre the dot does not move at all.

    `dim` draws the held (non-moving) hand: same geometry, muted, so a viewer
    can see BOTH hands and judge the collision clearance that the sampler
    enforced numerically.

    `stem` draws a line back to the OptiTrack marker cluster. WHY IT EXISTS:
    the triad origin is the GEL CONTACT FACE, and the tool body extends 52 mm
    behind it — a median 21 px on this rig, p90 28 px. The whole camera
    calibration error budget is 4 px at 800 mm depth, so a viewer comparing
    the marker against the middle of the visible tool sees a 20 px gap that is
    geometry rather than error, and nothing on screen says which is which.
    Measured on the release, the two gel centres agree to 1.1 mm against the
    tracked board plane (95% CI [0.86, 1.39] over 21,337 contacts), so there
    is no 20 px error to find. The stem says where the tip is.
    """
    import cv2

    from .calibration import project_gel_frame, project_rigid_origin

    out = np.ascontiguousarray(frame_rgb).copy()
    r = project_gel_frame(sensor_pose7, gel_center_mm, cam_calib, axis_len_mm)
    if r is None:
        return out
    h, w = out.shape[:2]
    cx, cy = int(round(r["centre"][0])), int(round(r["centre"][1]))
    if not (0 <= cx < w and 0 <= cy < h):
        return out
    for tip, col in zip(r["tips"], AXIS_BGR_RGB):
        if tip is None:
            continue
        tx, ty = int(round(tip[0])), int(round(tip[1]))
        c = tuple(int(v * 0.45) for v in col) if dim else col
        cv2.line(out, (cx, cy), (tx, ty), c, 1 if dim else 2, cv2.LINE_AA)
    if stem:
        o = project_rigid_origin(sensor_pose7, cam_calib)
        if o is not None:
            ox, oy = int(round(o[0])), int(round(o[1]))
            if 0 <= ox < w and 0 <= oy < h:
                sc = (90, 90, 90) if dim else (170, 170, 170)
                cv2.line(out, (ox, oy), (cx, cy), sc, 1, cv2.LINE_AA)
                cv2.circle(out, (ox, oy), 2, sc, 1, cv2.LINE_AA)
    ring = (150, 150, 150) if dim else (255, 255, 255)
    cv2.circle(out, (cx, cy), 3, ring, -1, cv2.LINE_AA)
    if label:
        cv2.putText(out, str(label), (cx + 7, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, ring, 1, cv2.LINE_AA)
    return out


def draw_collision_circle(frame_rgb, sensor_pose7, gel_center_mm, cam_calib,
                          diameter_m: float = 0.12, color=(255, 200, 80)):
    """The exclusion circle around a gel, drawn at its own depth.

    Radius in PIXELS is derived from the radius in METRES at this gel's
    depth — fx * r / z — so the circle shrinks with distance like everything
    else. A fixed-pixel circle would claim a constant physical size at every
    depth, which is the opposite of what it is for.
    """
    import cv2

    from .calibration import project_gel_frame, project_rigid_origin

    out = np.ascontiguousarray(frame_rgb).copy()
    r = project_gel_frame(sensor_pose7, gel_center_mm, cam_calib)
    if r is None:
        return out
    K = cam_calib["intrinsics"]
    rad_px = float(K["fx"] * (diameter_m / 2.0 * 1000.0) / max(r["depth_mm"], 1e-6))
    cx, cy = int(round(r["centre"][0])), int(round(r["centre"][1]))
    h, w = out.shape[:2]
    if 0 <= cx < w and 0 <= cy < h and rad_px < max(w, h):
        cv2.circle(out, (cx, cy), int(round(rad_px)), color, 1, cv2.LINE_AA)
    return out

def draw_world_gizmo(frame_rgb, cam_calib, corner="tl", size=44, margin=12,
                     labels=("x", "y", "z"), title=None):
    """World-frame orientation gizmo in a corner. Returns a copy.

    Shows which way the WORLD axes point in this camera, the way a 3D viewport
    corner axis does. Directions come from the rotation of `T_mocap_to_cam`
    only — a gizmo is deliberately orthographic, because a perspective one
    would change as you moved it around the image and stop being a legend.

    THE PART A NAIVE VERSION GETS WRONG: these cameras look down at the table,
    so world +z (up) points almost AT the camera and its screen projection is
    nearly zero length. Drawn as a plain arrow it would read as "z does not
    exist". So each axis also carries its out-of-plane sign — a filled dot for
    pointing toward the viewer, a cross for away — the standard convention, and
    the arrow length is the in-plane component only.
    """
    import cv2

    out = np.ascontiguousarray(frame_rgb).copy()
    h, w = out.shape[:2]
    T = np.asarray(cam_calib["T_mocap_to_cam"], float)[:3, :3]
    # Inset by the FULL reach of the drawing, not by the arrow: the label sits
    # 13 px past the tip and its glyph another ~9. The axis pointing straight
    # up is the one that runs off the top edge, and that is the axis this
    # gizmo exists to show.
    reach = size + 22
    ox = margin + reach if "l" in corner else w - margin - reach
    oy = margin + reach if "t" in corner else h - margin - reach
    # title BELOW the disc: at the top corner there is no room above it and the
    # text clipped against the frame edge.
    if title:
        ty = oy + size + 22
        for c, th in (((0, 0, 0), 3), ((215, 215, 215), 1)):
            cv2.putText(out, title, (ox - size - 6, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, c, th, cv2.LINE_AA)
    # a faint disc so the gizmo reads against any background
    ov = out.copy()
    cv2.circle(ov, (ox, oy), size + 8, (18, 22, 34), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
    cv2.circle(out, (ox, oy), size + 8, (70, 80, 100), 1, cv2.LINE_AA)

    for i, col in enumerate(AXIS_BGR_RGB):
        e = np.zeros(3); e[i] = 1.0
        d = T @ e                        # world axis, in camera coordinates
        # camera x right, y down, z into the scene
        px, py, pz = float(d[0]), float(d[1]), float(d[2])
        inplane = float(np.hypot(px, py))
        tipx, tipy = int(round(ox + px * size)), int(round(oy + py * size))
        if inplane > 0.12:
            cv2.arrowedLine(out, (ox, oy), (tipx, tipy), col, 2, cv2.LINE_AA,
                            tipLength=0.28)
            lx = int(round(ox + px * (size + 13)))
            ly = int(round(oy + py * (size + 13)))
        else:
            lx, ly = ox + 14, oy - 14
        # out-of-plane sign: toward the viewer is -z in camera coordinates
        if abs(pz) > 0.55:
            if pz < 0:                    # toward the viewer
                cv2.circle(out, (ox, oy), 6, col, 2, cv2.LINE_AA)
                cv2.circle(out, (ox, oy), 2, col, -1, cv2.LINE_AA)
            else:                          # away from the viewer
                cv2.circle(out, (ox, oy), 6, col, 2, cv2.LINE_AA)
                r = 4
                cv2.line(out, (ox-r, oy-r), (ox+r, oy+r), col, 1, cv2.LINE_AA)
                cv2.line(out, (ox-r, oy+r), (ox+r, oy-r), col, 1, cv2.LINE_AA)
        for c, th in (((0, 0, 0), 3), (col, 1)):
            cv2.putText(out, labels[i], (lx - 4, ly + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, th, cv2.LINE_AA)
    return out

