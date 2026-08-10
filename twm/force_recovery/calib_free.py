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
   FALSE, and the pairing was backwards. `cnc_mini_26/` (long misnamed
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
    LUT, cnc_mini_26 (shipped)   0.0721    0.1374        0.782
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

WHAT ACTUALLY IMPROVED IT, AND WHAT DID NOT (n = 468 / 390 / 390, force rho)

    change                                cnc_mini_26   FoTa cnc     FEATS
    boundary condition (see poisson.py)     +0.0380     +0.1479     0.0000
    normalise by the reference, not 255     +0.0391     +0.0770    +0.0235
    per-frame LED gain self-calibration     -0.0063     -0.0274    -0.3265

    calibration-free, now                     0.8379      0.3986     0.6389
    the LUT it is measured against            0.6143      0.3012     0.7577

Those are ALL presses, 84% of which run off the edge of the sensor because
both capture grids are bigger than the field of view. On presses imaged whole
(`visible_eval`), same protocol, same force range:

    calibration-free                          0.9950      0.9558     0.4700
    the LUT it is measured against            0.9909      0.8805     0.6327

The boundary condition was the reconstruction bug: the integrator pinned the
frame border to height zero, which is false for every contact reaching the
sensor edge — 409 of 468 and 294 of 390 frames here. FEATS is unchanged
because the rule leaves a marker gel on the clamped solver, having no flat gel
to anchor a free boundary on.

The LED gains were my idea and they lose everywhere. `channel_gains` estimates
a per-channel scale from the frame itself, on the reasoning that three LEDs
are not equally bright and an unmodelled gain rotates the recovered gradient.
The estimates come out wildly frame-dependent (0.36-1.81 across four presses
of the same probe), which is the tell: a physical LED gain is a constant, so
what the fit absorbs is scene-dependent model error — and putting that in the
reconstruction makes force worse, catastrophically so on the marker gel. Kept
behind `gains=False` with these numbers, because the idea is the obvious next
one to try and the next person deserves the result rather than the impulse.
"""
from __future__ import annotations

import numpy as np

# Three LEDs, 120 degrees apart, in (R, G, B) channel order.
#
# Determined on SPHERE presses, not on React. A sphere pressed into the gel
# must reconstruct as a circular dome, so the second moment of the depth peak
# has axis ratio 1; anything else is the channel-to-LED map being wrong. Over
# 30 sphere presses from `cnc_mini_26/round` (3-12 N):
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

# THE AZIMUTHS ARE FINE. SPARSH'S CHANNELS ARRIVED SWAPPED — DIAGNOSED, FIXED
#
# Found by scoring stage 1 alone: on Sparsh the LUT and the calibration-free
# depth of the same contact correlated at -0.13 inside the contact, against
# 0.69-0.83 everywhere else, and both rendered a round sphere press wrongly in
# ORTHOGONAL directions.
#
# First reading: "the LED azimuths are per-sensor and these are ours." A search
# over the six permutations put Sparsh's best at (90, 330, 210), axis ratio
# 1.50 against 3.33 for the value above, and that was recorded here as a
# different wiring. It is the same thing said less usefully: (90, 330, 210) IS
# (210, 330, 90) with R and B exchanged. The azimuths never differed; the
# CHANNELS did.
#
# Measured directly rather than searched. For a sphere the surface gradient
# points radially outward, so the dipole direction of each channel's dI is that
# channel's LED azimuth. 30 sphere presses per sensor:
#
#                    rest hue       R        G        B
#     our Mini        172.1 deg   259.2     5.1     51.1
#     Sparsh, as-is    42.1 deg    75.7     4.3    259.8
#     Sparsh, R<->B   197.9 deg   259.8     4.3     75.7
#
# Swapped, R and G land within 1 deg of ours. A different gel tint cannot do
# that; a channel-order difference does exactly it. Fixed in
# `debug_gallery.load_sparsh`, where the frames enter — NOT here, because this
# constant was never wrong.
#
# What the fix bought, 41 Sparsh sphere presses:
#
#                       sphere axis ratio   agreement with LUT   flat-gel leak
#     before                  3.62               -0.125              0.0272
#     after                   1.53               +0.920              0.0071
#
# And the point of having scored stage 1 separately: force rho on Sparsh was
# 0.909 BEFORE the fix, with the shape that wrong. After it, 0.894 for the
# calibration-free solve and 0.822 -> 0.894 for the LUT. A force number cannot
# see a geometry this broken.

# Does `reconstruct` return millimetres? NO, and the code now says so where a
# consumer can see it, because the docstring saying it was not enough.
#
# The solve produces a DIMENSIONLESS gradient (tan of the surface slope) and
# `fast_poisson` integrates it in PIXELS, so the result is a height in pixel
# units up to one unknown global factor. The LUT, by contrast, returns
# millimetres per pixel and integrates to millimetres.
#
# Presented as millimetres, the number reads 1.48-13.12 on a 4.25 mm gel, and
# the Open3D renderer — which applies a FIXED z exaggeration — drew those
# surfaces as vertical towers. That looked like a geometry failure and was
# reported as one. It is not: rescaling each surface to the LUT's peak turns
# every tower back into the right shape (a triangle indenter into a triangle,
# a two-hole stamp into a blob with two dimples), and per-pixel shape
# agreement with the LUT inside the contact is rho 0.75+.
#
# So: shape is recovered, scale is not. Consumers must normalise (figures) or
# absorb the factor in a fit (force — a linear least squares already does).
RETURNS_MILLIMETRES = False

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


def channel_gains(dI: np.ndarray, azimuth_deg=LED_AZIMUTH_DEG,
                  iters: int = 3) -> np.ndarray:
    """Per-channel LED gain, estimated FROM THE FRAME ITSELF. (3,), mean 1.

    The model `dI_k = gx cos(t_k) + gy sin(t_k)` assumes the three LEDs are
    equally bright and the three channels equally sensitive. They are not: a
    GelSight's LEDs are separate parts, and the camera's colour response is
    not flat. An unmodelled per-channel gain does not cancel — it tilts the
    3x2 solve toward the brightest channel, which rotates the recovered
    gradient direction.

    This stays calibration-FREE because nothing outside the frame is used: no
    rig, no sphere presses, no stored table. Alternate between solving for the
    gradient and rescaling each channel to best fit its own prediction, over
    the contact pixels only. The overall scale of the gains is unidentifiable
    (it trades against the gradient's own scale, which is already not
    recovered), so they are normalised to mean 1.
    """
    M = led_matrix(azimuth_deg)
    m = contact_mask(dI)
    if m.sum() < 200:
        return np.ones(3)
    d = dI[m] / 255.0                                   # (N, 3)
    a = np.ones(3)
    for _ in range(iters):
        g = (d / a) @ np.linalg.pinv(M).T               # (N, 2)
        pred = g @ M.T                                  # (N, 3)
        denom = (pred ** 2).sum(axis=0)
        a_new = np.where(denom > 1e-12, (d * pred).sum(axis=0) / denom, a)
        a_new = np.abs(a_new)
        if not np.all(a_new > 1e-6):
            return np.ones(3)
        a = a_new / a_new.mean()
    return a


def gradients(dI: np.ndarray, azimuth_deg=LED_AZIMUTH_DEG,
              remove_dc: bool = True, gains: bool = False,
              ref_img: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Surface gradient from the signed RGB difference image.

    `dI` is signed on purpose. Taking |dI| throws away which side of the
    indenter a pixel is on, which is the only thing that distinguishes a bump
    from a dent, and it is also what the published difference images should
    show — a colour image, not a magnitude.
    """
    import cv2

    M = led_matrix(azimuth_deg)                       # (3, 2)
    A = np.linalg.pinv(M)                             # (2, 3)
    a = channel_gains(dI, azimuth_deg) if gains else np.ones(3)
    # dI / I_ref, not dI / 255. The imaging model is
    #     I_k = albedo * G_k(p) * (n . l_k) + ambient
    # so the pixel gain G_k(p) — LED falloff across the pad, vignetting, and
    # the pad's own albedo — multiplies BOTH the frame and the reference, and
    # dividing by the reference cancels it. Differencing against a constant
    # 255 leaves it in, which makes the same slope read differently at
    # different places on the same pad. Worth +0.038 rho on cnc_mini_26 and
    # +0.019 on FoTa; it is the only one of eight candidate colour -> normal
    # changes that survived (see cf_variants).
    sig = (dI / np.maximum(np.asarray(ref_img, np.float64), 8.0)
           if ref_img is not None else dI / 255.0)
    s = (sig.reshape(-1, 3) / a) @ A.T                # (N, 2), sine-like
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


def normals(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Unit surface normals from the gradient field, (H, W, 3).

    n = (-gx, -gy, 1)/|.| — the quantity the photometric solve actually
    estimates. Depth is an INTEGRAL of this, so a normal map shows what was
    measured before any boundary condition or integration could distort it,
    which is the right place to look when a surface is suspected of being an
    artefact of the solver.
    """
    n = np.stack([-np.asarray(gx, np.float64), -np.asarray(gy, np.float64),
                  np.ones_like(gx, np.float64)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def normal_rgb(gx: np.ndarray, gy: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Normal map in the usual (n+1)/2 encoding, uint8 — flat gel is mauve.

    `gain` multiplies the GRADIENTS before the normal is formed, for display
    only. React's gradients are genuinely small (|grad z| p99 = 0.07-0.17), so
    at gain 1 the map is a flat mauve field and shows nothing. A caller that
    passes a gain must print it — `display_gain()` picks one from the data so
    the number is derived rather than dialled until it looks good.
    """
    return np.clip((normals(np.asarray(gx) * gain, np.asarray(gy) * gain)
                    + 1.0) * 127.5, 0, 255).astype(np.uint8)


def display_gain(gx: np.ndarray, gy: np.ndarray, target: float = 0.45) -> float:
    """Gain putting the 99th percentile gradient at `target` in the encoding."""
    p99 = float(np.percentile(np.hypot(gx, gy), 99))
    return 1.0 if p99 <= 1e-9 else max(1.0, round(target / p99, 1))


def reconstruct(img: np.ndarray, ref: np.ndarray,
                azimuth_deg=LED_AZIMUTH_DEG, scale: float = 1.0,
                normalize: bool = False, solver: str = "auto",
                gains: bool = False) -> dict:
    """Surface height with no per-sensor calibration — SHAPE, not millimetres.

    `normalize=True` divides by the frame's own peak, giving 0..1. That is the
    right choice for any FIGURE: a mesh renderer with a fixed z exaggeration
    turns an uncalibrated magnitude into a tower and makes correct geometry
    look broken. It is the wrong choice for feature extraction, where relative
    magnitude between frames carries the force signal.

    See RETURNS_MILLIMETRES.
    """
    from .poisson import integrate, poisson_dirichlet, poisson_neumann

    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = contact_mask(dI)
    gx, gy = gradients(dI, azimuth_deg, gains=gains, ref_img=ref)
    gx = np.where(valid, gx, 0.0)
    gy = np.where(valid, gy, 0.0)
    if solver == "auto":
        depth, used = integrate(gx, gy, valid, ref=ref)
    elif solver == "neumann":
        depth, used = poisson_neumann(gx, gy), "neumann"
    else:
        depth, used = poisson_dirichlet(gx, gy), "dirichlet"
    if valid.any() and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0) * scale
    if normalize:
        d = d / max(float(d.max()), 1e-12)
    from .poisson import contact_truncated
    return {"dI": dI, "valid": valid, "gx": gx, "gy": gy, "depth": d,
            "normals": normals(gx, gy), "solver": used,
            # True when the contact reaches the frame edge: the free boundary
            # has extrapolated a surface it could not see, and 14.5% of such
            # frames come out deeper than the gel is thick. See poisson.py.
            "truncated": contact_truncated(valid),
            "units": "relative (peak = 1)" if normalize
                     else "arbitrary (scale not recovered)"}


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
