"""Semi-transparent force dot for the episode preview videos.

One mapping function, `radius_px`, is the single source of truth for how a
newton becomes a pixel. The drawn circle, the legend and any readout all call
it, so the picture cannot disagree with the number printed beside it.

DESIGN — area encodes force, not radius.
Human size judgement of a filled disc tracks its AREA, so a dot whose radius
were proportional to force would exaggerate large forces roughly quadratically.
`radius ∝ √F` makes the *area* linear in force, which is what a reader
actually reads off the screen.

FRAME ALIGNMENT — the part that is easy to get silently wrong.
The force arrays are indexed by release-parquet row; the preview iterates raw
HDF5 frame indices. `run_episode` reads frame `trim + row + LEGACY_SHIFT`
(LEGACY_SHIFT = 15) where `trim = source_h5_frame[0]` from the parquet, while
the preview's own `_get_trim_offset` reads a different field that is often
absent (it then falls back to 0). Using the preview's trim would put the force
~15 frames — half a second at 30 fps — away from the picture it labels, so
`row_for_h5_frame` below deliberately re-derives the mapping from the parquet.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Panel geometry (viz.build_preview_panel): row 2 is
# [gs_L_raw | gs_L_diff | gs_R_raw | gs_R_diff | blank], 240x240 each.
GS_THUMB = 240
ROW2_Y = 240
TILE_X = {"left": 0, "right": 2 * GS_THUMB}      # the RAW tile of each side

# Force -> radius. F_FULL is the force at which the dot reaches R_MAX; it is
# the top of the band the React episodes actually occupy (peaks ~2-8 N), not a
# per-clip autoscale, so a dot means the same thing in every video.
F_FULL_N = 8.0
R_MIN_PX = 4.0
R_MAX_PX = 74.0                                   # < GS_THUMB/2, stays in tile
COLOR_BGR = (60, 120, 255)                        # orange, matches site accent
ALPHA = 0.42


def radius_px(force_n: float) -> float:
    """THE force -> pixel-radius law. Area is linear in force."""
    f = max(float(force_n), 0.0)
    return R_MIN_PX + (R_MAX_PX - R_MIN_PX) * np.sqrt(min(f, F_FULL_N) / F_FULL_N)


def row_for_h5_frame(h5_frame: int, trim_parquet: int, n_rows: int,
                     legacy_shift: int = 15) -> int | None:
    """Release-parquet row that produced this HDF5 frame, or None if outside."""
    row = int(h5_frame) - int(trim_parquet) - int(legacy_shift)
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


def draw_force_dot(panel: np.ndarray, side: str, force_n: float,
                   centroid_xy: tuple[float, float] | None = None) -> None:
    """Blend one semi-transparent dot onto the raw GelSight tile, in place.

    `centroid_xy` is in tile pixels; without it the dot sits at the tile centre.
    A zero force draws nothing at all — an always-present dot would imply
    contact where there is none.
    """
    if not np.isfinite(force_n) or force_n <= 0.02:
        return
    x0 = TILE_X[side]
    cx, cy = centroid_xy if centroid_xy else (GS_THUMB / 2, GS_THUMB / 2)
    r = radius_px(force_n)
    cx = float(np.clip(cx, r, GS_THUMB - r))
    cy = float(np.clip(cy, r, GS_THUMB - r))
    c = (int(x0 + cx), int(ROW2_Y + cy))

    roi_x0, roi_y0 = int(c[0] - r - 3), int(c[1] - r - 3)
    roi_x1, roi_y1 = int(c[0] + r + 4), int(c[1] + r + 4)
    roi_x0, roi_y0 = max(roi_x0, 0), max(roi_y0, 0)
    roi_x1 = min(roi_x1, panel.shape[1])
    roi_y1 = min(roi_y1, panel.shape[0])
    if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
        return
    roi = panel[roi_y0:roi_y1, roi_x0:roi_x1]
    layer = roi.copy()
    cv2.circle(layer, (c[0] - roi_x0, c[1] - roi_y0), int(r), COLOR_BGR, -1,
               cv2.LINE_AA)
    cv2.addWeighted(layer, ALPHA, roi, 1 - ALPHA, 0, roi)
    cv2.circle(roi, (c[0] - roi_x0, c[1] - roi_y0), int(r), COLOR_BGR, 1,
               cv2.LINE_AA)
    # Readout at the TOP of the tile: the bottom strip is occupied by the
    # panel's own status bar, which covered it in the first render.
    cv2.putText(panel, f"{force_n:.1f} N", (x0 + 8, ROW2_Y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, f"{force_n:.1f} N", (x0 + 8, ROW2_Y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_BGR, 1, cv2.LINE_AA)


def draw_legend(panel: np.ndarray, x: int, y: int,
                marks=(0.5, 2.0, 8.0)) -> None:
    """Scale key drawn with the SAME `radius_px`, so it cannot drift."""
    cv2.putText(panel, "force", (x, y - int(R_MAX_PX) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    cursor = x
    for f in marks:
        r = radius_px(f)
        cx, cy = int(cursor + r), int(y)
        layer = panel.copy()
        cv2.circle(layer, (cx, cy), int(r), COLOR_BGR, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, ALPHA, panel, 1 - ALPHA, 0, panel)
        cv2.circle(panel, (cx, cy), int(r), COLOR_BGR, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{f:g}N", (int(cursor + r - 12), int(y + r + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1,
                    cv2.LINE_AA)
        cursor += 2 * r + 14
