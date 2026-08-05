"""Contact metrics and reference-frame selection.

These reproduce the values shipped in the dataset parquet, and match
``react_toolbox.contact`` so producer and consumer agree by construction.
"""
from __future__ import annotations

import numpy as np

from .config import TAU


def l2_diff(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-pixel L2 distance from a reference frame.

    frames: (..., H, W, 3) uint8/float, reference: (H, W, 3). -> (..., H, W)
    """
    d = frames.astype(np.float32) - reference.astype(np.float32)
    return np.sqrt((d ** 2).sum(axis=-1))


def contact_metrics(frames: np.ndarray, reference: np.ndarray, tau: float = TAU):
    """(intensity, area, mixed) for a block of frames.

    intensity = mean L2 distance, area = fraction above `tau`,
    mixed = mean of the thresholded distance.
    """
    d = l2_diff(frames, reference)
    above = d > tau
    axes = (-2, -1)
    return (d.mean(axis=axes).astype(np.float32),
            above.mean(axis=axes).astype(np.float32),
            (d * above).mean(axis=axes).astype(np.float32))


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    w = max(1, min(win, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="same")


def pick_p01_reference(intensity: np.ndarray, win: int) -> int:
    """Index of the quietest (least-contact) frame — the no-contact reference.

    Smoothing first avoids latching onto a single noisy frame.
    """
    return int(smooth(intensity, win).argmin())


class NewFrameTracker:
    """Flags which frames are genuinely new tactile captures.

    A GelSight frame repeated across consecutive rows means the sensor did not
    update between those camera ticks — true of ~72 % of rows in legacy
    recordings (tactile really ran at ~8 fps while rows were written at 30 Hz),
    and of ~40 % in fixed recordings (18.75 fps sensor, 30 Hz rows).

    Detection is an exact frame comparison, so it is correct for both the
    legacy (duplicate pixels) and timestamped (repeated index) layouts.
    """

    def __init__(self):
        self._prev = None

    def update(self, block: np.ndarray) -> np.ndarray:
        """block: (n, H, W, 3) uint8 -> (n,) bool, True where the frame changed."""
        n = len(block)
        out = np.empty(n, dtype=bool)
        for i in range(n):
            prev = block[i - 1] if i > 0 else self._prev
            out[i] = True if prev is None else not np.array_equal(block[i], prev)
        self._prev = block[-1].copy() if n else self._prev
        return out


def duplication_stats(is_new: np.ndarray, fps: float = 30.0) -> dict:
    """Summarise how much of a tactile stream is duplicated."""
    n = len(is_new)
    if n == 0:
        return {"n_frames": 0, "n_unique": 0, "duplicate_ratio": 0.0,
                "effective_fps": 0.0, "max_repeat_run": 0}
    n_unique = int(is_new.sum())
    runs, cur = [], 0
    for flag in is_new:
        if flag:
            if cur:
                runs.append(cur)
            cur = 1
        else:
            cur += 1
    if cur:
        runs.append(cur)
    return {
        "n_frames": n,
        "n_unique": n_unique,
        "duplicate_ratio": float(1.0 - n_unique / n),
        "effective_fps": float(n_unique / (n / fps)) if n else 0.0,
        "max_repeat_run": int(max(runs)) if runs else 0,
    }
