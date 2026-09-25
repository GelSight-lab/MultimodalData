"""Contact detection for markerless GelSight.

Provides the classical, calibration-free contact pipeline:
  diff-from-reference (L2) -> threshold -> largest connected component -> mask,
plus the exact contact scalars the dataset ships (intensity/area/mixed) so
users can reproduce / recompute them.
"""
from __future__ import annotations

import numpy as np

from .reference import l2_diff

TAU = 8.0   # dataset default L2 threshold (uint8 scale)


def contact_metrics(frame, reference, tau: float = TAU):
    """Reproduce the dataset's per-frame contact scalars.

        d = ||frame - ref||_2  (per pixel, over RGB)
        intensity = mean(d)
        area      = mean(d > tau)
        mixed     = mean(d * (d > tau))

    Matches data/<task>/meta/*.parquet tactile_*_{intensity,area,mixed}.
    """
    d = l2_diff(frame, reference)
    above = d > tau
    return {
        "intensity": float(d.mean()),
        "area": float(above.mean()),
        "mixed": float((d * above).mean()),
    }


def contact_mask(frame, reference, tau: float = TAU, largest_only: bool = True,
                 min_area_px: int = 50):
    """Binary contact mask (H, W) bool.

    diff-from-reference L2 > tau, then (optionally) keep only the largest
    connected component and drop specks < min_area_px. No calibration needed.
    """
    import cv2
    d = l2_diff(frame, reference)
    mask = (d > tau).astype(np.uint8)
    if int(mask.sum()) < min_area_px:
        return np.zeros(mask.shape, bool)
    if not largest_only:
        return mask.astype(bool)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]              # skip background (label 0)
    keep = int(areas.argmax()) + 1
    out = lbl == keep
    return out if out.sum() >= min_area_px else np.zeros(mask.shape, bool)


def contact_centroid(mask):
    """(row, col) centroid of a contact mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())
