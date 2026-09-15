"""Photometric correction for the wrist cameras.

The two wrist cameras record a much darker image than the three RealSense
views. Measured on the 2026-09-10/11 pushT sessions:

    stream        p50        p95        p99.5
    wrist USB     60-76      123-128    176-213
    RealSense     134-138    153        163-166

Note the p99.5 row: the wrist HIGHLIGHTS are brighter than the reference's, so
this is not an under-exposure with headroom to scale into. A linear gain would
either crush that specular tail (if chosen to lift the bulk) or do nothing at
all (0.77-0.95x if chosen to preserve it). The defect is the distribution — a
thin bright tail with the rest pressed into the shadows.

A power curve addresses exactly that: it lifts the midtones, holds 0 at 0 and
255 at 255, and clips nothing.

The exponent is ONE CONSTANT PER CAMERA GENERATION, not a per-episode fit. A
per-episode fit was tried first and is wrong: the wrist cameras ride on the
moving sensors, so what fills their view swings between a bright wood table and
a black PCB within a single episode, and an exponent fitted to that would
render the same object differently from clip to clip. The defect being
corrected belongs to the camera, so the correction has to be a property of the
camera.

Nor can the exponent be fitted by matching the wrist median to the RealSense
median: the wrist is a close-up and the reference is a wide overhead view, so
their medians differ because of FRAMING, not because of the cameras. The
constants below come from comparing the two cameras on the same surface — the
wood table both views contain — through its upper quantiles, which is where
that surface lives in both histograms:

    camera     wrist p75/p90    reference p75/p90    gamma from p75 / p90
    usb        106 / 120        146 / 152            1.57 / 1.45
    arducam    135 / 165        155 / 174            1.27 / 1.14

Why not fix it at the camera instead: the camera has no gain control and its
exposure already sits at the 30 fps ceiling (33.3 ms). `gamma` is at its 500
maximum, worth about +10 % — and measured on 2026-09-11, everything beyond that
buys brightness with frame rate, exactly linearly: exposure 666 (100 us) does
reach the reference's mean 108 / p95 154, at 15.1 fps instead of 30.1.

This is a declared change to published pixels. The exponent used for each
stream is written into the episode metadata; 1.0 means the stream was published
as recorded.
"""
from __future__ import annotations

import numpy as np

# The exponent is clamped into this range. Above the top a nearly-black view
# would have its sensor noise amplified into visible structure; below 1.0 the
# correction would DARKEN a stream, which throws information away rather than
# redistributing it (see `tone_gamma_for`).
GAMMA_LIMITS = (1.0, 3.0)
# The published exponent per (wrist-camera generation, task). `None` for the
# camera — a session recorded before the wrist cameras — publishes unchanged.
#
# The camera column is the measurement above. The task column is the operator's
# 2026-09-11 ruling, and it is deliberately content-based: the motherboard task
# fills the wrist view with a black PCB, and at the camera-derived exponent that
# subject sits at p50=15 with its traces unreadable. 2.2 makes the heatsink
# fins, the traces and the left-hand components legible. 2.4 is the operator's
# pick over 2.2 after seeing both. The cost, measured on that frame: empty
# histogram bins per channel 49 -> ~82, shadow temporal noise 20.8 -> ~27, and a
# violet cast as the lift exposes the blue channel's noise floor (which is why
# it stops here — at 3.0 the board reads as purple rather than black).
#
# This does NOT reintroduce the per-episode problem it replaced: the exponent is
# constant across a whole task, so no object is rendered differently from clip
# to clip. It is recorded per episode either way.
WRIST_TONE_GAMMA = {
    ("usb", "motherboard"): 2.4,
    ("usb", "pushT"): 1.5,
    ("arducam", "motherboard"): 1.8,
    ("arducam", "pushT"): 1.2,
}
# Used when a task has no entry: the camera-derived exponent, no task lift.
WRIST_TONE_GAMMA_DEFAULT = {"usb": 1.5, "arducam": 1.2}
# Medians closer than this need no correction; it is under the frame-to-frame
# variation of either stream.
MEDIAN_TOLERANCE = 2.0


def _quantile_grey(frames, quantile: float = 50.0) -> float:
    a = np.asarray(frames, np.float32)
    return float(np.percentile(a.mean(axis=-1), quantile))


def gamma_for_camera(kind, task: str | None = None) -> float:
    """The published exponent for a wrist-camera generation on a task."""
    if kind is None:
        return 1.0
    if (kind, task) in WRIST_TONE_GAMMA:
        return float(WRIST_TONE_GAMMA[(kind, task)])
    return float(WRIST_TONE_GAMMA_DEFAULT.get(kind, 1.0))


def tone_gamma_for(frames, reference_frames, quantile: float = 50.0) -> float:
    """Exponent that puts `frames`' quantile onto `reference_frames`' quantile.

    This is the tool the constants in ``WRIST_TONE_GAMMA`` were derived with,
    over pooled frames from a whole camera generation — not something to run
    per episode, for the reason in the module docstring.

    Solves ``255 * (m/255) ** (1/g) == r`` for g. Returns exactly 1.0 when the
    stream needs no lift, including when the reference is the darker of the
    two: the correction only ever brightens.
    """
    m, r = _quantile_grey(frames, quantile), _quantile_grey(reference_frames, quantile)
    if not (0.0 < m < 255.0) or not (0.0 < r < 255.0):
        return 1.0
    if r <= m + MEDIAN_TOLERANCE:
        return 1.0
    g = np.log(m / 255.0) / np.log(r / 255.0)
    lo, hi = GAMMA_LIMITS
    return float(np.clip(g, lo, hi))


def tone_curve_lut(gamma: float) -> np.ndarray:
    """256-entry uint8 lookup table for ``out = 255 * (in/255) ** (1/gamma)``."""
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.round(255.0 * x ** (1.0 / float(gamma))), 0, 255).astype(np.uint8)


def apply_tone_curve(frames, gamma: float):
    """Lift one frame or a block of them, unchanged at gamma 1.

    The curve drives the pixel's BRIGHTEST channel and every channel is then
    scaled by the same factor, so the R:G:B ratios — which are the hue and the
    saturation — come through untouched, and nothing can clip: the brightest
    channel lands exactly where the lookup table puts it, at most 255.

    Applying the curve to each channel independently instead, which is the
    obvious implementation, repaints the scene. Measured on a real wrist frame
    at gamma 2.4: hue moved 4 deg at the median and 18 deg at p95, and
    saturation fell by 20 points at the median and 78 at p5 — the wood table
    went pale and skin went grey. Grey pixels are unaffected by the choice, so
    the exponents derived from grey statistics keep their meaning.
    """
    a = np.asarray(frames)
    if abs(float(gamma) - 1.0) < 1e-9:
        return a
    peak = a.max(axis=-1)
    lifted = tone_curve_lut(gamma)[peak].astype(np.float32)
    scale = np.divide(lifted, peak, out=np.ones_like(lifted), where=peak > 0)
    return np.clip(a.astype(np.float32) * scale[..., None] + 0.5,
                   0, 255).astype(np.uint8)


def camera_kind_from_serials(serials) -> str | None:
    """Which wrist-camera generation reported these serials, or None if empty.

    Both generations report a serial and the USB pair reports ONE SHARED serial
    on both cameras, so the serial's value discriminates them where its
    presence cannot. The Arducam serials come from the shipped Arducam config
    so there is a single source for them.
    """
    serials = list(serials)
    if not serials:
        return None
    from twm.sensor_camera import ARDUCAM_CONFIG_PATH, load_config
    known = {c.serial for c in load_config(ARDUCAM_CONFIG_PATH)}
    return "arducam" if all(s in known for s in serials) else "usb"


def episode_wrist_gamma(f) -> float:
    """The exponent an open recording's wrist frames must be shown through.

    Everything needed is in the file's own metadata — the task and the wrist
    cameras' serials — so no caller has to be told which curve applies, and
    the recorder preview, the replay viewer, the dataset previews and the
    publish path cannot drift apart.
    """
    import json

    attrs = f["metadata"].attrs if "metadata" in f else {}
    raw = attrs.get("arducam_config", None)
    if not raw:
        return 1.0
    try:
        cams = json.loads(raw)
    except (TypeError, ValueError):
        return 1.0
    task = attrs.get("task", "")
    task = task.decode() if isinstance(task, bytes) else str(task)
    return gamma_for_camera(camera_kind_from_serials(
        [c.get("serial") for c in cams]), task)
