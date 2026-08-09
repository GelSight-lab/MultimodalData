"""Integrate a gradient field into a height map — with the right boundary.

THE BUG THIS EXISTS TO FIX

Every reconstruction here (LUT and calibration-free) integrated gradients with
`fast_poisson`, the GelSight reference solver. It transforms with a discrete
SINE transform and writes its answer into `img[1:-1, 1:-1]` of a zero array,
which is a homogeneous DIRICHLET condition: the recovered height is pinned to
exactly 0 all the way around the frame.

That is correct only when the gel is flat at the border. It is false for any
contact that reaches the edge of the sensor — and on React those are common,
because the fixture presses parts against the side of the pad. The solver then
has to reconcile "this pixel is 2 mm deep" with "this pixel on the border is
0 mm", and it does it by bending the whole neighbourhood: a cliff at the frame
edge and a depressed skirt beside it. It looks like a reconstruction artefact
because it is one, and it is in the solver, not the sensor.

The physically right condition for integrating a gradient field is NEUMANN
(the normal derivative on the border is whatever the measured gradient says).
It leaves the height undetermined by one additive constant, which is exactly
the true ambiguity — nobody measured an absolute datum — and is fixed here by
subtracting the median of the non-contact gel, which IS known to be flat.

`poisson_neumann` is that solver, via the DCT (even reflection = zero normal
derivative). `poisson_dirichlet` wraps the original so the two can be compared
on identical inputs; `scripts/test_poisson_edge.py` measures both against
surfaces whose truth is known analytically.
"""
from __future__ import annotations

import numpy as np

__all__ = ["poisson_neumann", "poisson_dirichlet", "divergence", "integrate",
           "detrend_flat", "anchor_region", "free_boundary_ok"]


def divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """f = div(g), by the same one-sided differences the DST solver uses.

    Kept identical on purpose: if the two solvers disagree, the difference has
    to be the boundary condition and nothing else.
    """
    gxx = np.zeros_like(gx, dtype=np.float64)
    gyy = np.zeros_like(gy, dtype=np.float64)
    gxx[:, 1:] = gx[:, 1:] - gx[:, :-1]
    gyy[1:, :] = gy[1:, :] - gy[:-1, :]
    return gxx + gyy


def poisson_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Height from gradients under a zero-normal-derivative boundary.

    Returns a surface with zero mean; the caller anchors it (see module
    docstring). The DC term of the transform is unconstrained by construction
    — that is the additive constant, not a division by zero to be papered
    over, so it is set to zero explicitly rather than clamped.
    """
    try:
        from scipy.fft import dctn, idctn
    except ImportError:                                   # scipy < 1.4
        from scipy.fftpack import dctn, idctn

    f = divergence(np.asarray(gx, np.float64), np.asarray(gy, np.float64))
    m, n = f.shape
    fh = dctn(f, type=2, norm="ortho")
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    denom = (2 * np.cos(np.pi * x / n) - 2) + (2 * np.cos(np.pi * y / m) - 2)
    denom[0, 0] = 1.0
    zh = fh / denom
    zh[0, 0] = 0.0
    z = idctn(zh, type=2, norm="ortho")
    return z - z.mean()


# A free boundary needs somewhere to put the datum. `poisson_neumann` recovers
# height up to one constant, and that constant is fixed on gel that is KNOWN
# flat — the region well outside the contact. When there is no such region the
# free boundary has no anchor, and the honest answer is that the data does not
# support it.
#
# Measured, 40 frames per dataset, fraction of the frame more than 25 px from
# any contact:
#
#     cnc_mini_26   median 70.2%   min 42.5%
#     FoTa cnc      median 80.7%   min 68.3%
#     FEATS         median  0.1%   min  0.0%   (80% of frames under 2%)
#
# FEATS is a MARKER gel: the dot lattice makes |dI| > 8 fire almost everywhere,
# so nothing is left to anchor on and 79% of the frame border sits inside the
# "contact". That is exactly where the free boundary loses to the clamped one
# on force (rho 0.488 vs 0.615 calibration-free, 0.386 vs 0.758 through the
# LUT). The threshold sits in a gap three orders of magnitude wide, so it is a
# classification, not a tuned parameter.
ANCHOR_MIN_FRAC = 0.05
ANCHOR_DILATE_PX = 25


def anchor_region(valid: np.ndarray) -> np.ndarray:
    """Gel far enough from any contact to be treated as flat."""
    import cv2
    k = np.ones((ANCHOR_DILATE_PX, ANCHOR_DILATE_PX), np.uint8)
    return cv2.dilate(np.asarray(valid, np.uint8), k) == 0


def free_boundary_ok(ref: np.ndarray | None) -> bool:
    """Can this SENSOR support a free boundary? Decided from the reference.

    Deciding per frame was tried first and is wrong in a way worth recording.
    The anchor test below is a property of the contact, so on FEATS it said
    "no anchor" for 80% of frames and "anchor" for the other 20% — two
    different height conventions inside one dataset, which is a nuisance
    variable in every feature computed from them. It cost the LUT 0.044 rho on
    FEATS while helping nothing.

    The reference frame does not change within a dataset, so a decision taken
    from it cannot mix. A marker gel's dot lattice shifts under load, so
    |dI| > 8 fires across the whole pad and no flat gel is ever exposed —
    that is the real condition, and it is a property of the sensor.
    """
    if ref is None:
        return True
    from .marker_removal import marker_mask
    return marker_mask(ref) is None


def detrend_flat(z: np.ndarray, flat: np.ndarray, order: int = 1
                 ) -> np.ndarray:
    """Remove a low-order trend fitted on gel that is KNOWN flat.

    The clamped solver's hidden benefit was that pinning the border to zero
    also high-passes the surface, suppressing the slow illumination drift
    between a frame and its reference. A free boundary integrates that drift
    instead. Subtracting a plane (or quadratic) fitted over the non-contact
    region removes it explicitly — using the same physical statement the
    anchor already relies on, that this gel is flat — instead of getting it as
    a side effect of a boundary condition that is wrong about geometry.
    """
    ys, xs = np.nonzero(flat)
    if len(ys) < 200:
        return z - np.median(z[flat]) if flat.any() else z - np.median(z)
    x = xs / z.shape[1]
    y = ys / z.shape[0]
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, y * y, x * y]
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, z[flat], rcond=None)
    gy, gx_ = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    X = gx_ / z.shape[1]
    Y = gy / z.shape[0]
    full = [np.ones_like(X), X, Y]
    if order >= 2:
        full += [X * X, Y * Y, X * Y]
    return z - sum(ci * fi for ci, fi in zip(c, full))


# WHAT THE FREE BOUNDARY COSTS, AND WHERE
#
# Measured on 468 cnc_mini_26 presses, peak depth against the 4.25 mm gel:
#
#                              p50    p95     max    over gel thickness
#     Dirichlet (retired)     1.27   2.96    4.03          0.0%
#     Neumann + detrend       2.74   5.00    6.79         13.0%
#       contact imaged whole    --   2.79      --          0.0%   (n =  46)
#       contact truncated       --   5.15      --         14.5%   (n = 422)
#
# The free boundary produces depths the gel cannot physically have, and it does
# so ONLY where the contact runs off the frame — there it must extrapolate a
# surface it never saw. The clamped solver never exceeds the gel, but it buys
# that by asserting height 0 at the border, which is false for exactly those
# frames; it is not more correct, it is wrong in a direction that happens to
# look safe.
#
# Neither is fixed by clipping the output at 4.25 mm: that would hide an
# extrapolation behind a plausible number. The reconstruction reports the
# condition instead (`truncated` in calib_free.reconstruct), so a consumer can
# refuse the frame. On frames imaged whole the free boundary is both physical
# and better (rho 0.995 vs 0.991 through the LUT).
def contact_truncated(valid: np.ndarray) -> bool:
    """Does the contact reach the frame border? Then depth there is extrapolated."""
    v = np.asarray(valid, bool)
    return bool(v[0].any() or v[-1].any() or v[:, 0].any() or v[:, -1].any())


def integrate(gx: np.ndarray, gy: np.ndarray, valid: np.ndarray,
              ref: np.ndarray | None = None) -> tuple:
    """Gradients -> height, choosing the boundary condition FROM THE DATA.

    Returns (height, solver_name). Neumann when this sensor exposes flat gel
    AND this frame actually has some (the physically right condition — a
    contact reaching the frame edge means the border is genuinely not at zero,
    and clamping it there costs 15.4% of peak against an analytic surface);
    Dirichlet when it does not, because an unanchored free boundary is worse
    than an assumption you can at least name.
    """
    if not free_boundary_ok(ref):
        return poisson_dirichlet(gx, gy), "dirichlet-marker-gel"
    far = anchor_region(valid)
    if far.mean() < ANCHOR_MIN_FRAC:
        # Degenerate on a markerless gel — measured 0 of 40 frames on both
        # markerless datasets — but a full-frame contact has no datum either.
        return poisson_dirichlet(gx, gy), "dirichlet-no-anchor"
    # Quadratic, not a constant and not a plane. Measured on React's own
    # calibration (471 frames, held out by press position, identical split and
    # fitting code, only this line differing):
    #
    #     boundary / anchor                 rho     in view   clipped
    #     Dirichlet (retired)             0.739       0.933     0.660
    #     Neumann, median anchor          0.532       0.784     0.448
    #     Neumann + plane detrend         0.628       0.734     0.512
    #     Neumann + quadratic detrend     0.767       0.954     0.703
    #
    # The median-anchored free boundary LOST to the clamp, which is what sent
    # this back to the drawing board: pinning the border also high-passes the
    # surface, so the clamp was silently removing the illumination drift
    # between a frame and its reference. Removing that drift explicitly — on
    # gel the mask already says is flat — keeps the suppression and the correct
    # geometry, and beats the clamp on every subset including presses that
    # never touch the border.
    z = poisson_neumann(gx, gy)
    return detrend_flat(z, far, order=2), "neumann-detrended"


def poisson_dirichlet(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """The original GelSight DST solver, for comparison. Border is forced to 0."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson
    return fast_poisson(np.asarray(gx, np.float64), np.asarray(gy, np.float64))
