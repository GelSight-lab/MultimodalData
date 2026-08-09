"""A small imprint must still show its shape, not saturate into a black lump.

THE DEFECT

`gradient_shade` maps the surface's (dx + dy) into a grey band using the 2nd
and 98th percentiles. Those were taken over the WHOLE frame. A press covering
a few percent of the pad therefore had its band set by the noise of the flat
gel around it, and its own relief landed far outside that band: measured on a
quad_small press covering 6.3% of the frame, 46-48% of the contact's pixels
clipped to pure black or pure white. A square indenter rendered as a round
dark blob — the geometry was in the depth map and the shading destroyed it.

THE TEST

A synthetic square imprint occupying ~5% of the frame, which is the regime
that failed. Assert that few of its pixels saturate under the shipped
contact-scaled shading, AND that the retired whole-frame scaling still fails
the same check — a gate both arms pass is not a gate.

    python -m scripts.test_mesh_shading
"""
from __future__ import annotations

import numpy as np

from force_recovery.o3d_view import gradient_shade

MAX_CLIPPED = 0.10          # of contact pixels, shipped path
MIN_RETIRED_CLIPPED = 0.25  # the defect must still be visible to this test


def _square_imprint(h=240, w=320, side=56):
    """A flat-topped square press near the top edge, on flat gel."""
    z = np.zeros((h, w), np.float32)
    y0, x0 = 8, 150
    z[y0:y0 + side, x0:x0 + side] = 1.4
    import cv2
    z = cv2.GaussianBlur(z, (9, 9), 3)
    on = z > 0.05 * z.max()
    return z, on


def _clipped_fraction(shade: np.ndarray, on: np.ndarray) -> float:
    """Fraction of contact pixels pinned at either end of the grey band."""
    v = shade[on]
    lo, hi = v.min(), v.max()
    at_end = (np.abs(v - lo) < 1e-9) | (np.abs(v - hi) < 1e-9)
    return float(at_end.mean())


def _whole_frame_shade(z: np.ndarray, contrast: float = 0.30) -> np.ndarray:
    """The retired scaling, reproduced here so the test can compare against it."""
    dy, dx = np.gradient(z.astype(np.float64))
    shade = dx + dy
    lo, hi = np.percentile(shade, [2, 98])
    shade = (np.clip((shade - lo) / (hi - lo), 0, 1) if hi - lo > 1e-9
             else np.full_like(shade, 0.5))
    return 0.5 + (shade - 0.5) * 2.0 * contrast + 0.12


def main() -> int:
    z, on = _square_imprint()
    print(f"  synthetic square imprint covers {on.mean()*100:.1f}% of the frame")
    shipped = _clipped_fraction(gradient_shade(z), on)
    retired = _clipped_fraction(_whole_frame_shade(z), on)
    print(f"    contact-scaled (shipped) : {shipped*100:5.1f}% of contact "
          f"pixels saturated")
    print(f"    whole-frame   (retired)  : {retired*100:5.1f}%")
    bad = []
    if shipped > MAX_CLIPPED:
        bad.append(f"the shipped shading saturates {shipped*100:.0f}% of the "
                   f"imprint (> {MAX_CLIPPED*100:.0f}%) — small presses will "
                   f"render as featureless lumps again")
    if retired < MIN_RETIRED_CLIPPED:
        bad.append(f"the retired scaling only saturates {retired*100:.0f}% "
                   f"here, so this test can no longer see the defect it "
                   f"guards and a pass means nothing")
    for b in bad:
        print(f"  FAIL: {b}")
    print(f"mesh-shading: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
