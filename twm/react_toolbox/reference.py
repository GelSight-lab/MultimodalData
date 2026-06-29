"""Reference (background) frame handling + difference image for GelSight.

GelSight Mini is markerless, so background subtraction is clean (no markers
to occlude). Convention follows Sparsh (Meta, CoRL 2024): a single no-contact
reference frame + signed per-channel difference.

All frames are RGB uint8 (H, W, 3) — the convention returned by the dataset's
video decoder.
"""
from __future__ import annotations

import numpy as np


def get_reference(frames, mode: str = "p01", intensity=None, window: int = 30):
    """Pick a no-contact reference frame from a sequence.

    frames: (T, H, W, 3) uint8  OR a callable i->frame for lazy access with
            an explicit `intensity` array.
    mode:
        "first"        -> frames[0]
        "p01"          -> the 1st-percentile-intensity (quietest) frame; needs
                          `intensity` (T,) array of per-frame contact intensity,
                          else computes a cheap proxy = mean abs deviation from
                          the temporal median.
        "running_avg"  -> mean of the `window` lowest-intensity frames.
    Returns an (H, W, 3) uint8 reference.
    """
    arr = np.asarray(frames)
    T = arr.shape[0]
    if mode == "first":
        return arr[0].copy()

    if intensity is None:
        med = np.median(arr.reshape(T, -1).astype(np.float32), axis=0)
        intensity = np.abs(arr.reshape(T, -1).astype(np.float32) - med).mean(axis=1)
    intensity = np.asarray(intensity, np.float32)

    if mode == "p01":
        thr = np.percentile(intensity, 1)
        idx = int(np.where(intensity <= thr)[0][0]) if (intensity <= thr).any() else int(intensity.argmin())
        return arr[idx].copy()

    if mode == "running_avg":
        k = min(window, T)
        order = np.argsort(intensity)[:k]
        return arr[order].mean(axis=0).round().astype(np.uint8)

    raise ValueError(f"unknown reference mode: {mode}")


def difference(frame, reference, signed: bool = True):
    """Sparsh-style difference image.

    signed=True  -> (frame - ref)/255 + 0.5, clipped to [0,1], scaled to uint8.
                    Mid-gray = no change; preserves direction of deformation.
    signed=False -> |frame - ref| as uint8 (magnitude only).

    frame/reference: (H, W, 3) uint8. Returns (H, W, 3) uint8.
    """
    f = frame.astype(np.float32)
    r = reference.astype(np.float32)
    if signed:
        d = (f - r) / 255.0 + 0.5
        return (np.clip(d, 0, 1) * 255).astype(np.uint8)
    return np.clip(np.abs(f - r), 0, 255).astype(np.uint8)


def l2_diff(frame, reference):
    """Per-pixel L2 distance across RGB channels (H, W) float32.

    This is the quantity the dataset's contact scalars are built on:
    d[h,w] = ||frame[h,w,:] - ref[h,w,:]||_2.
    """
    return np.sqrt(((frame.astype(np.float32) - reference.astype(np.float32)) ** 2).sum(axis=2))
