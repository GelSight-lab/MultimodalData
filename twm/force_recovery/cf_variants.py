"""Candidate colour -> normal conversions, and what measuring them settled.

Every entry is a full image -> depth function so `scripts/calibfree_search`
can score them on ONE protocol: the same five features, the same per-group
half/half least squares, isotonic calibration, 5 seeds and within-group
shuffle control the published table uses. Nothing here is allowed to fit
anything on force — a variant may only change how an image becomes a surface.

THE SEARCH (cnc_mini_26, n = 468, pooled Spearman rho)

    variant                    rho      verdict
    dI/255 (the old step)    0.799      baseline
    ratio  (divide by ref)   0.837      ADOPTED, +0.038
    ratio + flat field       0.838      same thing, no extra
    ratio + soft mask        0.838      inside the noise
    ratio + noise mask       0.829      no
    ratio, no sine->tan      0.823      the tan step earns its place
    ratio + dilated mask     0.834      no
    ratio, no mask at all    0.825      no
    near-field LED dirs      0.845*     * only at R = 18 mm, a shallow peak
                                        picked on the target — rejected
    near-field dirs + mag    0.679-0.703 REFUTED, and it made the spatial
                                        variation worse, not better
    sRGB linearisation       0.770      REFUTED
    full 3-D photometric     0.748      REFUTED
    ratio + 3-D photometric  0.766      REFUTED
    per-frame LED gains      0.763      REFUTED (see calib_free.channel_gains)
    better reference frame   0.731->0.731  no change; the reference already
                                        agrees with a 24-frame median to
                                        0.8 grey levels

Only ONE of eleven candidates survived. `ratio` works because the imaging
model is I_k = albedo * G_k(p) * (n . l_k) + ambient: the per-pixel gain
G_k — LED falloff, vignetting, pad albedo — multiplies frame and reference
alike and cancels. Normalising by a constant 255 leaves it in, and then the
same slope reads differently at different places on the same pad.

CAN THIS REACH rho 0.95? NO, AND THE LIMIT IS NOT THE COLOUR CONVERSION

Measured on the same 468 presses:

    within ONE press position, rho(F, maxd)          1.000
    pooled across positions                          0.837
    + an ORACLE per-group quadratic position field
      fitted ON THE FORCE LABELS (not shippable)     0.899
    the CNC's own commanded indentation z, scored
      through the identical protocol                 0.967

The reconstruction is already perfectly monotone in force at a fixed
position; what remains is a spatial sensitivity that varies 4.9x (p10-p90)
across the pad, of which a quadratic in (x, y) explains 53%. Correcting it is
exactly the fitted gain field every pipeline here carries and which this work
was asked not to add — and even WITH that oracle correction the protocol
reaches 0.899, short of 0.95. The ceiling of 0.967 is what the ground truth
itself supports: force and commanded depth agree at rho 0.956-0.981 per
family, no better.

There is a second, harder limit underneath: 84% of these presses have their
contact CORE (depth > 0.2 of peak) running off the frame, so part of the
indentation is physically outside the field of view and no reconstruction can
recover it. The fully-in-view subset is 75 frames, 8-18 per group — too few
to score through this protocol, and it is left unscored rather than quoted.

So 0.95 is not reachable by improving the colour -> normal step, and the
honest ceiling for a fit-free reconstruction on this dataset is ~0.90.
"""
from __future__ import annotations

import numpy as np

from . import calib_free as CF
from .poisson import integrate

# Elevation of the three LEDs above the gel plane. Sensor geometry, not a fit
# on force — chosen by `elevation_by_sphere`, which asks that a sphere of
# known radius reconstruct as a sphere.
LED_ELEVATION_DEG = 20.0

SRGB_A = 0.055


def to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB byte values -> relative radiance. Standard curve, no parameters."""
    c = np.asarray(x, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + SRGB_A) / (1 + SRGB_A)) ** 2.4)


def _depth_from_grad(gx, gy, valid, ref):
    gx = np.where(valid, gx, 0.0)
    gy = np.where(valid, gy, 0.0)
    d, _ = integrate(gx, gy, valid, ref=ref)
    if valid.any() and np.median(d[valid]) < 0:
        d = -d
    return np.maximum(d, 0.0)


def _slope_from_signal(sig, valid, azimuth_deg=CF.LED_AZIMUTH_DEG,
                       tan: bool = True, clip: float = 0.99):
    """3-channel signal -> (gx, gy) by the linear LED projection + sine->tan."""
    M = CF.led_matrix(azimuth_deg)
    A = np.linalg.pinv(M)
    s = sig.reshape(-1, 3) @ A.T
    s = s.reshape(sig.shape[0], sig.shape[1], 2)
    s = np.clip(s, -clip, clip)
    g = s / np.sqrt(1.0 - s ** 2) if tan else s
    far = ~valid
    if far.sum() > 100:
        g = g - np.median(g[far], axis=0)
    return g[..., 0], g[..., 1]


# ------------------------------------------------------------------ variants
def base(img, ref):
    return CF.reconstruct(img, ref)["depth"]


def linear_rgb(img, ref):
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    sig = to_linear(img) - to_linear(ref)
    gx, gy = _slope_from_signal(sig, valid)
    return _depth_from_grad(gx, gy, valid, ref)


def ratio(img, ref):
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    r = np.asarray(ref, np.float64)
    sig = dI / np.maximum(r, 8.0)
    gx, gy = _slope_from_signal(sig, valid)
    return _depth_from_grad(gx, gy, valid, ref)


def ratio_linear(img, ref):
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    li, lr = to_linear(img), to_linear(ref)
    sig = (li - lr) / np.maximum(lr, 1e-3)
    gx, gy = _slope_from_signal(sig, valid)
    return _depth_from_grad(gx, gy, valid, ref)


def ratio_notan(img, ref):
    """Is the sine->tangent step earning its place on the ratio signal?

    On dI/255 the signal only reaches |s| ~ 0.12-0.38, where tan(s) ~ s and
    the step is nearly a no-op. On the ratio signal it reaches 0.30-0.71, so
    it becomes a real nonlinearity — which means it now has to be justified
    rather than inherited.
    """
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    sig = dI / np.maximum(np.asarray(ref, np.float64), 8.0)
    gx, gy = _slope_from_signal(sig, valid, tan=False)
    return _depth_from_grad(gx, gy, valid, ref)


def ratio_noise_mask(img, ref):
    """Contact mask from the frame's OWN noise, not a fixed grey level.

    VALID_DI = 8 is one number for every sensor and exposure. The robust
    spread of dI over the untouched gel measures the same thing per frame and
    costs no parameter beyond the multiplier, which is a detection threshold
    in sigmas, not a fit.
    """
    dI = img.astype(np.float32) - ref.astype(np.float32)
    a = np.abs(dI).max(axis=2)
    sigma = 1.4826 * np.median(np.abs(a - np.median(a)))
    import cv2
    thr = max(float(np.median(a) + 4.0 * sigma), 4.0)
    mag = cv2.GaussianBlur(a, (5, 5), 1.5)
    valid = cv2.morphologyEx((mag > thr).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    sig = dI / np.maximum(np.asarray(ref, np.float64), 8.0)
    gx, gy = _slope_from_signal(sig, valid)
    return _depth_from_grad(gx, gy, valid, ref)


def ratio_flatfield(img, ref):
    """Divide by the reference's SHAPE only (mean-normalised per channel).

    Separates the two things `ratio` does at once: cancelling the spatial
    falloff, and cancelling the per-channel absolute level. If the flat field
    alone recovers the gain, the channel levels were not the issue.
    """
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    r = np.asarray(ref, np.float64)
    shape = r / np.maximum(r.reshape(-1, 3).mean(axis=0), 1e-6)
    sig = dI / np.maximum(shape, 0.2) / 255.0
    gx, gy = _slope_from_signal(sig, valid)
    return _depth_from_grad(gx, gy, valid, ref)


def ratio_ps3d(img, ref):
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    sig = dI / np.maximum(np.asarray(ref, np.float64), 8.0)
    L = _light_dirs(CF.LED_AZIMUTH_DEG, LED_ELEVATION_DEG)
    b = (sig.reshape(-1, 3) @ np.linalg.inv(L).T).reshape(*sig.shape)
    n = b + np.array([0.0, 0.0, 1.0])
    nz = np.where(np.abs(n[..., 2]) < 1e-3, 1e-3, n[..., 2])
    gx, gy = -n[..., 0] / nz, -n[..., 1] / nz
    far = ~valid
    if far.sum() > 100:
        gx, gy = gx - np.median(gx[far]), gy - np.median(gy[far])
    return _depth_from_grad(gx, gy, valid, ref)


def _ratio_sig(img, ref):
    return (img.astype(np.float32) - ref.astype(np.float32)) / np.maximum(
        np.asarray(ref, np.float64), 8.0)


def ratio_nomask(img, ref):
    """Integrate every gradient, masking only for the anchor.

    Zeroing the gradient outside |dI| > 8 truncates the RIM of a weak contact,
    which is exactly where its slope lives. The three families that drag the
    pooled score down (round 0.742, quad_small 0.724, B 0.808 against quad
    0.982) are the small and shallow ones, so the mask is a suspect.
    """
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    gx, gy = _slope_from_signal(_ratio_sig(img, ref), valid)
    d, _ = integrate(gx, gy, valid, ref=ref)
    if valid.any() and np.median(d[valid]) < 0:
        d = -d
    return np.maximum(d, 0.0)


def ratio_dilate(img, ref, px: int = 9):
    """Keep the gradient inside a dilated mask — the rim, not just the core."""
    import cv2
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    keep = cv2.dilate(valid.astype(np.uint8),
                      np.ones((px, px), np.uint8)).astype(bool)
    gx, gy = _slope_from_signal(_ratio_sig(img, ref), valid)
    gx, gy = np.where(keep, gx, 0.0), np.where(keep, gy, 0.0)
    d, _ = integrate(gx, gy, valid, ref=ref)
    if valid.any() and np.median(d[valid]) < 0:
        d = -d
    return np.maximum(d, 0.0)


def ratio_soft(img, ref):
    """Weight the gradient by contact confidence instead of thresholding it.

    A hard mask is a decision taken per pixel with no confidence; a weak
    contact's rim sits right at the threshold, so half of it survives and half
    does not. The weight below is the same statistic the mask uses, passed
    through a smooth ramp rather than a step.
    """
    import cv2
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    mag = cv2.GaussianBlur(np.abs(dI).max(axis=2), (5, 5), 1.5)
    w = np.clip((mag - CF.VALID_DI * 0.5) / CF.VALID_DI, 0.0, 1.0)
    gx, gy = _slope_from_signal(_ratio_sig(img, ref), valid)
    d, _ = integrate(gx * w, gy * w, valid, ref=ref)
    if valid.any() and np.median(d[valid]) < 0:
        d = -d
    return np.maximum(d, 0.0)


# The pad is 13.2 x 9.9 mm and the LEDs sit at its rim, so "the light arrives
# from azimuth theta_k" is a FAR-FIELD approximation over a field of view
# comparable to the light's own distance. Under it, a press at the left of the
# pad and the same press at the right are told they were lit identically, and
# they were not — which is why the production pipelines all carry a fitted
# spatial gain field u(x, y) to patch it up afterwards.
#
# Treating each LED as a point at radius R on the rim gives the direction per
# PIXEL, from geometry, with no fit and no gain field. R -> infinity recovers
# the current model exactly, so this is a strict generalisation.
def _perpixel_dirs(shape, azimuth_deg, radius_mm):
    from .lut_calibration import MM_PER_PIXEL
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (x - (w - 1) / 2) * MM_PER_PIXEL
    y = (y - (h - 1) / 2) * MM_PER_PIXEL
    out = np.empty((h, w, 3, 2))
    for k, th in enumerate(np.deg2rad(np.asarray(azimuth_deg, np.float64))):
        lx, ly = radius_mm * np.cos(th), radius_mm * np.sin(th)
        dx, dy = x - lx, y - ly
        n = np.sqrt(dx * dx + dy * dy) + 1e-9
        out[..., k, 0], out[..., k, 1] = dx / n, dy / n
    return out


def _solve_perpixel(sig, M):
    """Per-pixel 3x2 least squares, closed form. sig (H,W,3), M (H,W,3,2)."""
    a = (M[..., 0] ** 2).sum(-1)
    b = (M[..., 0] * M[..., 1]).sum(-1)
    c = (M[..., 1] ** 2).sum(-1)
    r0 = (M[..., 0] * sig).sum(-1)
    r1 = (M[..., 1] * sig).sum(-1)
    det = a * c - b * b
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    return (c * r0 - b * r1) / det, (a * r1 - b * r0) / det


def ratio_nearfield(img, ref, radius_mm: float = 12.0):
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    sig = _ratio_sig(img, ref)
    M = _perpixel_dirs(sig.shape[:2], CF.LED_AZIMUTH_DEG, radius_mm)
    sx, sy = _solve_perpixel(sig, M)
    s = np.clip(np.stack([sx, sy], -1), -0.99, 0.99)
    g = s / np.sqrt(1.0 - s ** 2)
    far = ~valid
    if far.sum() > 100:
        g = g - np.median(g[far], axis=0)
    return _depth_from_grad(g[..., 0], g[..., 1], valid, ref)


def ratio_nearfield_full(img, ref, radius_mm: float = 12.0,
                         height_mm: float = 4.0):
    """Near-field LEDs, direction AND magnitude, from geometry alone.

    `ratio_nearfield` changed only the azimuth per pixel and bought almost
    nothing (+0.007). The magnitude term is the one that matters here. In the
    ratio formulation

        dI_k / I_ref,k  =  (dn . l_k) / (n0 . l_k),      n0 = (0,0,1)

    the denominator is sin(elevation of LED k AT THAT PIXEL). With a light at
    the rim of a 13 x 10 mm pad, that elevation is not constant across the
    field, so the same slope produces a different reading at different
    positions — which is precisely the "spatial gain field" that every
    pipeline here fits after the fact. Written out, the slope projection is

        ratio_k * sin(elev) / cos(elev)  =  ratio_k * tan(elev_k(p))

    which is geometry with two sensor constants and no fit on force.
    """
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    sig = _ratio_sig(img, ref)
    from .lut_calibration import MM_PER_PIXEL
    h, w = sig.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx = (xx - (w - 1) / 2) * MM_PER_PIXEL
    yy = (yy - (h - 1) / 2) * MM_PER_PIXEL
    M = np.empty((h, w, 3, 2))
    wgt = np.empty((h, w, 3))
    for k, th in enumerate(np.deg2rad(np.asarray(CF.LED_AZIMUTH_DEG, float))):
        dx = xx - radius_mm * np.cos(th)
        dy = yy - radius_mm * np.sin(th)
        r = np.sqrt(dx * dx + dy * dy) + 1e-9
        M[..., k, 0], M[..., k, 1] = dx / r, dy / r
        wgt[..., k] = height_mm / r            # tan(elevation) at this pixel
    sx, sy = _solve_perpixel(sig * wgt, M)
    s = np.clip(np.stack([sx, sy], -1), -0.99, 0.99)
    g = s / np.sqrt(1.0 - s ** 2)
    far = ~valid
    if far.sum() > 100:
        g = g - np.median(g[far], axis=0)
    return _depth_from_grad(g[..., 0], g[..., 1], valid, ref)


def _light_dirs(azimuth_deg, elev_deg):
    t = np.deg2rad(np.asarray(azimuth_deg, np.float64))
    p = np.deg2rad(float(elev_deg))
    return np.stack([np.cos(p) * np.cos(t), np.cos(p) * np.sin(t),
                     np.full(3, np.sin(p))], axis=1)          # (3, 3)


def ps3d(img, ref, elev_deg: float = LED_ELEVATION_DEG, linear: bool = True):
    """Lambertian photometric stereo: recover a unit normal, not a slope.

    Works on the DIFFERENCE image the same way the rest of the pipeline does,
    but inverts the full 3x3 light matrix instead of its xy projection, so the
    z component is solved rather than assumed. The gradient is then
    -n_x/n_z, -n_y/n_z, which replaces the sine->tangent approximation with
    the exact relation.
    """
    dI = img.astype(np.float32) - ref.astype(np.float32)
    valid = CF.contact_mask(dI)
    sig = ((to_linear(img) - to_linear(ref)) if linear
           else dI.astype(np.float64) / 255.0)
    L = _light_dirs(CF.LED_AZIMUTH_DEG, elev_deg)
    b = sig.reshape(-1, 3) @ np.linalg.inv(L).T
    b = b.reshape(sig.shape[0], sig.shape[1], 3)
    # A difference image gives the CHANGE of the normal; the unperturbed gel is
    # flat, so n = normalise(n0 + db) with n0 = (0, 0, 1).
    n = b + np.array([0.0, 0.0, 1.0])
    nz = np.where(np.abs(n[..., 2]) < 1e-3, 1e-3, n[..., 2])
    gx, gy = -n[..., 0] / nz, -n[..., 1] / nz
    far = ~valid
    if far.sum() > 100:
        gx = gx - np.median(gx[far])
        gy = gy - np.median(gy[far])
    return _depth_from_grad(gx, gy, valid, ref)


def ps3d_raw(img, ref):
    return ps3d(img, ref, linear=False)


VARIANTS = {
    "base": base,
    "ratio_notan": ratio_notan,
    "ratio_noise_mask": ratio_noise_mask,
    "ratio_flatfield": ratio_flatfield,
    "ratio_ps3d": ratio_ps3d,
    "ratio_nomask": ratio_nomask,
    "ratio_dilate": ratio_dilate,
    "ratio_soft": ratio_soft,
    "ratio_nearfield": ratio_nearfield,
    "ratio_nearfield_full": ratio_nearfield_full,
    "linear_rgb": linear_rgb,
    "ratio": ratio,
    "ratio_linear": ratio_linear,
    "ps3d_linear": ps3d,
    "ps3d_raw": ps3d_raw,
}
