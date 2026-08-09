"""Calibration-free depth from a GelSight difference image.

WHY THIS EXISTS

React's depth panels show a deformation field that does not look like the
object pressing into the gel. Measured against the flat-gel constraint —
depth away from the contact patch must be zero, because the gel there is not
touched — the production reconstruction leaks 7.2% of its peak depth outside
the contact on React against 2.2% on the frames its lookup table was built
from.

TWO EXPLANATIONS I PUBLISHED HERE AND THEN HAD TO WITHDRAW

1. "It is cross-sensor transfer: a GlowTact table applied to a GelSight Mini."
   FALSE, and the pairing was backwards. `mini_cnc26/` (long misnamed
   `GelSight_force_final_14716` and referred to as GLOWTACT) is itself a
   **GelSight Mini** capture — it is the Mini arm of a two-sensor GlowTact
   study. The real mismatch is the other way round and is already measured in
   `glowtact_selfmade.py`: rest-gel hue is 169 deg for that Mini, 90/82 deg
   for React's pads, 62 deg for GlowTact's self-made pad. React's sensors are
   the self-made kind, and the table in production is the one furthest from
   them in colour.

2. "The gradient field carries a DC bias that Poisson integrates into a dome."
   FALSE. Removing the DC over the non-contact region changes the leak from
   0.0715 to 0.0713 over 24 strong presses. It is kept below because a flat
   region with a non-zero mean gradient is a measurement error either way,
   but it explains nothing.

WHAT THE MEASUREMENT ACTUALLY SAYS (24 strong presses, 4 episodes, both sides)

    reconstruction                  leak    leak p95   LUT coverage
    LUT, Mini CNC 2026 (shipped)   0.0721    0.1374        0.782
    LUT, GlowTact self-made pad    0.1037    0.2255        0.035
    LUT, Sparsh                    0.0210    0.1037        0.760
    calibration-free (this file)   0.0713    0.1351          --

So the calibration-free reconstruction is a TIE with the shipped LUT, not an
improvement — an earlier three-frame comparison said 2-3x better and was
small-sample noise, the third time that trap has been sprung in this project.
The finding worth acting on is the Sparsh table, 3.4x better on a criterion
that uses no labels at all. Whether any of this changes estimated FORCE is a
separate question and is settled on datasets that have ground-truth force,
not here.

The self-made table is worse despite the better hue match because it is
almost empty: 0.59% of its bins are filled, so 3.5% of React's contact pixels
land on an observed bin and the rest extrapolate.

THE ALTERNATIVE, AND WHY IT NEEDS NO CALIBRATION

A GelSight is lit by three coloured LEDs from three known directions. To first
order each channel's intensity change is the surface gradient projected on that
LED's azimuth:

    dI_k  ~  gx*cos(theta_k) + gy*sin(theta_k)

so (gx, gy) comes out of a 3x2 least-squares solve whose matrix is fixed by the
sensor's geometry. No table, no sphere presses, no per-sensor fitting. This is
what the GelSight Wedge driver does (`legacies/Wedge/src/gelsight/pose.py`,
`img2grad`), in its collapsed form:

    dx = dG / 255                       dy = (dR - dB) / 255
    dx = dx / sqrt(1 - dx^2) / 128      (sine -> tangent)

which is the same solve for LEDs at 90 / 210 / 330 degrees.

WHAT IS PHYSICS AND WHAT IS A CHOICE

Physics: the LED model above; the sine-to-tangent conversion; and that the gel
is flat outside contact, which is why the gradient's DC component is estimated
on the non-contact region and removed rather than left to integrate into a
dome. Removing it is not a tuned correction — a flat region whose mean gradient
is non-zero is a measurement error by definition.

A choice: which LED sits on which channel. That is sensor wiring, not a law,
so `LED_AZIMUTH_DEG` is a named constant and `azimuth_search` measures the
alternatives against the flat-gel constraint instead of asserting one.

Scale is NOT recovered. This returns depth in arbitrary units up to one global
factor; a force calibration fixes that factor exactly as it does for the LUT.
"""
from __future__ import annotations

import numpy as np

# Three LEDs, 120 degrees apart, in (R, G, B) channel order.
#
# Determined on SPHERE presses, not on React. A sphere pressed into the gel
# must reconstruct as a circular dome, so the second moment of the depth peak
# has axis ratio 1; anything else is the channel-to-LED map being wrong. Over
# 30 sphere presses from `mini_cnc26/round` (3-12 N):
#
#     (R,G,B) azimuth     axis ratio   flat-gel leak
#     (210, 330,  90)         1.266         0.0344     <- this one
#     ( 90, 210, 330)         1.328         0.0447
#     (210,  90, 330)         1.798         0.0602     <- the first guess here
#     (330,  90, 210)         1.848         0.0491
#
# The flat-gel leak alone did NOT settle this: searched on React frames it
# ranked six assignments within 1.7x of each other and reordered completely
# between a 3-frame and a 36-frame sample. A criterion that rewards a
# reconstruction for being small cannot tell a correct shape from a wrong one
# — the first-guess assignment split a connector into two blobs and turned a
# horizontal edge into a vertical bar while scoring the LOWEST leak of all.
# Known geometry is what settles it.
#
# 1.266 is not 1.0. Some of that is real (the indenter is not always pressed
# normal to the gel), and it is left visible rather than tuned away.
LED_AZIMUTH_DEG = (210.0, 330.0, 90.0)

# Contact test, shared with `stages()` so the two reconstructions are compared
# on identical pixels rather than on their own private masks.
VALID_DI = 8.0
DC_DILATE_PX = 25          # DC is estimated this far outside the contact


def led_matrix(azimuth_deg=LED_AZIMUTH_DEG) -> np.ndarray:
    """(3, 2) projection of surface gradient onto each LED azimuth."""
    t = np.deg2rad(np.asarray(azimuth_deg, np.float64))
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def contact_mask(dI: np.ndarray) -> np.ndarray:
    import cv2
    mag = cv2.GaussianBlur(np.abs(dI).max(2), (5, 5), 1.5)
    m = (mag > VALID_DI).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def gradients(dI: np.ndarray, azimuth_deg=LED_AZIMUTH_DEG,
              remove_dc: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Surface gradient from the signed RGB difference image.

    `dI` is signed on purpose. Taking |dI| throws away which side of the
    indenter a pixel is on, which is the only thing that distinguishes a bump
    from a dent, and it is also what the published difference images should
    show — a colour image, not a magnitude.
    """
    import cv2

    M = led_matrix(azimuth_deg)                       # (3, 2)
    A = np.linalg.pinv(M)                             # (2, 3)
    s = (dI.reshape(-1, 3) / 255.0) @ A.T             # (N, 2), sine-like
    s = s.reshape(dI.shape[0], dI.shape[1], 2)
    s = np.clip(s, -0.99, 0.99)
    g = s / np.sqrt(1.0 - s ** 2)                     # sine -> tangent

    if remove_dc:
        # The gel outside contact is flat, so its mean gradient is zero by
        # construction; whatever is measured there is illumination drift
        # between the frame and the reference. Left in, Poisson integrates it
        # into a global tilt — the dome.
        m = contact_mask(dI)
        far = cv2.dilate(m.astype(np.uint8),
                         np.ones((DC_DILATE_PX, DC_DILATE_PX), np.uint8)) == 0
        if far.sum() > 100:
            g = g - g[far].mean(axis=0)
    return g[..., 0], g[..., 1]


def reconstruct(img: np.ndarray, ref: np.ndarray,
                azimuth_deg=LED_AZIMUTH_DEG, scale: float = 1.0) -> dict:
    """Depth (arbitrary units x `scale`) with no per-sensor calibration."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson

    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = contact_mask(dI)
    gx, gy = gradients(dI, azimuth_deg)
    gx = np.where(valid, gx, 0.0)
    gy = np.where(valid, gy, 0.0)
    depth = fast_poisson(gx, gy)
    if valid.any() and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0) * scale
    return {"dI": dI, "valid": valid, "gx": gx, "gy": gy, "depth": d}


def flat_gel_leak(depth: np.ndarray, valid: np.ndarray) -> float:
    """Mean |depth| outside contact, over peak depth. Zero is correct.

    The one number that says whether a reconstruction is physically coherent
    without needing any label: the gel away from the indenter is flat.
    """
    if not valid.any() or depth.max() <= 0:
        return float("nan")
    return float(np.abs(depth[~valid]).mean() / depth.max())


def azimuth_search(frames, candidates=None) -> list[dict]:
    """Score channel-to-LED assignments on the flat-gel constraint.

    Which LED is on which channel is wiring, so it is measured rather than
    declared. `frames` is an iterable of (img, ref).
    """
    if candidates is None:
        base = (210.0, 90.0, 330.0)
        candidates = [base,
                      (90.0, 210.0, 330.0),
                      (330.0, 90.0, 210.0),
                      (90.0, 330.0, 210.0),
                      (210.0, 330.0, 90.0),
                      (330.0, 210.0, 90.0)]
    pairs = [(np.asarray(i, np.float32), np.asarray(r, np.float32))
             for i, r in frames]
    out = []
    for az in candidates:
        leaks = []
        for img, ref in pairs:
            r = reconstruct(img, ref, azimuth_deg=az)
            leaks.append(flat_gel_leak(r["depth"], r["valid"]))
        out.append({"azimuth_deg": az, "leak_mean": float(np.nanmean(leaks)),
                    "leak_p95": float(np.nanpercentile(leaks, 95))})
    return sorted(out, key=lambda r: r["leak_mean"])
