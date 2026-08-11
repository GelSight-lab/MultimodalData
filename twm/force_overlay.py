"""Semi-transparent press-force disc for the episode preview videos.

It is drawn on the CAMERA views, centred on the sensor's projected position
(see the WHERE note below), by `viz.draw_projection_overlay`. This module owns
the force -> pixel law and the legend; `viz` owns the placement.

One mapping function, `radius_px`, is the single source of truth for how a
newton becomes a pixel. The drawn circle, the legend and any readout all call
it, so the picture cannot disagree with the number printed beside it.

DESIGN — area encodes force, not radius.
Human size judgement of a filled disc tracks its AREA, so a dot whose radius
were proportional to force would exaggerate large forces roughly quadratically.
`radius ∝ √F` makes the *area* linear in force, which is what a reader
actually reads off the screen.

FRAME ALIGNMENT — the part that is easy to get silently wrong, and did.
The force arrays are indexed by release-parquet row; the preview iterates raw
HDF5 frame indices. `trim = source_h5_frame[0]` comes from the parquet, not
from the preview's own `_get_trim_offset`, which reads a different field that
is often absent and then falls back to 0.

The mapping is `row = h5_frame - trim`, and NOT `- LEGACY_SHIFT` on top of it.
This subtracted the shift as well, on the reasoning that `run_episode` reads
frame `trim + row + LEGACY_SHIFT`. It does — but that +15 is how you reach the
GELSIGHT frame row r was computed from, and the gelsight stream lags the
cameras by exactly that much, so gelsight[trim + r + 15] is the tactile image
of CAMERA frame trim + r. The row a camera frame belongs to is unaffected.

The cost was the defect this docstring warned about, produced by the line that
was supposed to prevent it: in one panel the tactile tile showed gelsight
frame i+15 while the disc showed the force of gelsight frame i. Half a second
apart, on every published preview. Cross-correlating the displayed contact
signal against the displayed force peaked at -16 frames on three episodes of
motherboard/2026-05-10, and at 0 once the extra subtraction was removed
(`scripts/test_preview_alignment.py`).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from twm.tactile_align import LEGACY_SHIFT

# WHERE the dot goes: the camera views, at the sensor's own projected
# position — not on the tactile tiles. Two reasons the first version was wrong:
# the tile IS the tactile signal, so a disc over it hides the thing it
# annotates; and a dot pinned to a tile corner says how hard but never where,
# which is the half a reader cannot get from the number alone. `viz`
# draws it inside `draw_projection_overlay`, sharing that function's projected
# centre, so the dot and the pose axes cannot disagree about the sensor.
#
# Force -> radius. F_FULL is the force at which the dot reaches R_MAX; it is
# the top of the band the React episodes actually occupy, not a per-clip
# autoscale, so a dot means the same thing in every video.
F_FULL_N = 8.0
R_MIN_PX = 3.0
# Sized for a 320x240 camera thumbnail, not the old 240 px tactile tile: at
# full scale the disc spans 44 px, ~14% of the view's width. The previous
# 74 px radius covered the workpiece the video exists to show.
R_MAX_PX = 22.0
COLOR_BGR = (60, 120, 255)                        # orange, matches site accent
ALPHA = 0.42


def radius_px(force_n: float) -> float:
    """THE force -> pixel-radius law. Area is linear in force."""
    f = max(float(force_n), 0.0)
    return R_MIN_PX + (R_MAX_PX - R_MIN_PX) * np.sqrt(min(f, F_FULL_N) / F_FULL_N)


def row_for_h5_frame(h5_frame: int, trim_parquet: int, n_rows: int) -> int | None:
    """Release-parquet row annotating this CAMERA frame, or None if outside.

    No lag term: see the module docstring. The parameter is gone rather than
    defaulted to zero, so a caller that still believes a shift belongs here
    fails loudly instead of passing one that is silently ignored.
    """
    row = int(h5_frame) - int(trim_parquet)
    return row if 0 <= row < n_rows else None


def load_forces(task: str, date: str, episode: str,
                out_root: Path) -> dict[str, np.ndarray]:
    """Per-side force arrays for one episode, {} if not estimated yet."""
    out = {}
    for side in ("left", "right"):
        p = out_root / task / date / f"{episode}_{side}.npz"
        if p.exists():
            out[side] = np.asarray(np.load(p)["force_normal_n"], dtype=float)
    return out


def draw_force_halo(canvas: np.ndarray, xy: tuple[float, float],
                    force_n: float, *, scale: int = 1, label: bool = True,
                    bounds: tuple[int, int, int, int] | None = None) -> None:
    """Blend the force disc at a canvas pixel position, in place.

    `xy` is in FINAL panel pixels; `scale` is the supersample factor of
    `canvas` (viz renders the overlay on a 2x buffer, so radius and text
    scale with it — hard-coding 1x here would have drawn a half-size dot).

    `bounds` is the (x0, y0, x1, y1) rect the disc may occupy, in final panel
    pixels — the camera thumbnail it belongs to. It is REQUIRED in practice:
    `project_gel_pose` happily returns coordinates outside the 640x480 image
    when the sensor is out of a camera's frustum, and clipping merely to the
    canvas let one such projection paint a disc down in the tactile row. The
    render verifier caught it; the eye would not have, since it looks like a
    plausible dot on a plausible tile.

    A zero force draws nothing at all: an always-present dot would imply
    contact where there is none.
    """
    if not np.isfinite(force_n) or force_n <= 0.02:
        return
    H, W = canvas.shape[:2]
    bx0, by0, bx1, by1 = ((0, 0, W // scale, H // scale) if bounds is None
                          else bounds)
    # The CENTRE must lie in the view it annotates. A disc whose centre is
    # outside means the sensor is not visible in this camera; drawing a
    # clipped crescent at the border would assert a position that is wrong.
    if not (bx0 <= xy[0] < bx1 and by0 <= xy[1] < by1):
        return
    cx, cy = float(xy[0]) * scale, float(xy[1]) * scale
    r = radius_px(force_n) * scale
    c = (int(cx), int(cy))

    roi_x0 = max(int(cx - r - 3), bx0 * scale)
    roi_y0 = max(int(cy - r - 3), by0 * scale)
    roi_x1 = min(int(cx + r + 4), bx1 * scale)
    roi_y1 = min(int(cy + r + 4), by1 * scale)
    if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
        return
    roi = canvas[roi_y0:roi_y1, roi_x0:roi_x1]
    layer = roi.copy()
    cv2.circle(layer, (c[0] - roi_x0, c[1] - roi_y0), int(r), COLOR_BGR, -1,
               cv2.LINE_AA)
    cv2.addWeighted(layer, ALPHA, roi, 1 - ALPHA, 0, roi)
    cv2.circle(roi, (c[0] - roi_x0, c[1] - roi_y0), int(r), COLOR_BGR, scale,
               cv2.LINE_AA)
    if label:
        tx = min(int(cx + r + 4), (bx1 - 46) * scale)
        ty = max(int(cy - r - 4), (by0 + 12) * scale)
        for col, th in [((0, 0, 0), 3 * scale), (COLOR_BGR, scale)]:
            cv2.putText(canvas, f"{force_n:.1f}N", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, col, th,
                        cv2.LINE_AA)


def draw_legend(panel: np.ndarray, x: int, y: int,
                marks=(0.5, 2.0, 8.0)) -> None:
    """Scale key drawn with the SAME `radius_px`, so it cannot drift.

    Circles share a vertical CENTRE and labels share a BASELINE. Hanging each
    label off its own circle instead made the three sit at three heights —
    the eye reads that stagger as meaning something, when it only encodes the
    radius already shown by the circle.
    """
    cv2.putText(panel, "press force", (x, y - int(R_MAX_PX) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    baseline = int(y + R_MAX_PX + 14)
    cursor = x
    for f in marks:
        r = radius_px(f)
        cx, cy = int(cursor + r), int(y)
        layer = panel.copy()
        cv2.circle(layer, (cx, cy), int(r), COLOR_BGR, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, ALPHA, panel, 1 - ALPHA, 0, panel)
        cv2.circle(panel, (cx, cy), int(r), COLOR_BGR, 1, cv2.LINE_AA)
        label = f"{f:g}N"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(panel, label, (cx - tw // 2, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1,
                    cv2.LINE_AA)
        cursor += 2 * r + max(14, tw - 2 * r + 6)   # never let labels collide
