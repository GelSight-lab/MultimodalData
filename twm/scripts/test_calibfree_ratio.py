"""The calibration-free solve is invariant to illumination, and fits nothing.

THE PROPERTY, AND THE FIRST VERSION OF THIS TEST THAT DID NOT MEASURE IT

The colour -> normal step divides by the REFERENCE image rather than by 255,
because the imaging model is

    I_k = albedo * G_k(p) * (n . l_k) + ambient

so the per-pixel gain G_k — LED falloff across the pad, vignetting, the pad's
own albedo — multiplies frame and reference alike and cancels in the ratio.
Worth +0.038 rho on cnc_mini_26.

My first version asserted that a vignette leaves the recovered SHAPE alone,
by rank correlation over the contact. Both normalisations passed it (0.9998
vs 0.9990): a smooth gain across a small contact barely reorders anything, so
the statistic could not see the defect it was written for. What the gain
actually changes is MAGNITUDE, and magnitude is what carries force across
frames:

    same press, illumination scaled 0.5-1.3x    peak depth spread
    reference-normalised (shipped)                       x1.03
    dI / 255 (retired)                                   x2.80

So the test asserts invariance of the peak, and asserts that the retired path
still FAILS it — a gate that both arms pass is not a gate.

    python -m scripts.test_calibfree_ratio
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from force_recovery import calib_free as CF
from force_recovery.poisson import integrate

ROOT = Path(__file__).resolve().parents[1]
RECON = ("calib_free.py", "poisson.py", "cf_variants.py")
MAX_SPREAD = 1.10          # shipped path, over a 2.6x illumination range
MIN_RETIRED_SPREAD = 2.0   # the defect must still be visible to this test


def _synthetic(h=240, w=320):
    """A press and its reference, exactly consistent with the LED model."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    bump = 26.0 * np.exp(-(((x - 150) ** 2 + (y - 120) ** 2) / (2 * 40.0 ** 2)))
    gx, gy = np.gradient(bump, axis=1), np.gradient(bump, axis=0)
    M = CF.led_matrix()
    ref = np.full((h, w, 3), 150.0)
    img = ref + np.stack([gx * M[k, 0] + gy * M[k, 1]
                          for k in range(3)], -1) * 255
    return img, ref, bump


def _peak(img, ref, use_ref: bool) -> float:
    dI = img.astype(np.float32) - ref.astype(np.float32)
    v = CF.contact_mask(dI)
    gx, gy = CF.gradients(dI, ref_img=(ref if use_ref else None))
    d, _ = integrate(np.where(v, gx, 0.0), np.where(v, gy, 0.0), v, ref=ref)
    if v.any() and np.median(d[v]) < 0:
        d = -d
    return float(np.percentile(np.maximum(d, 0.0), 99.8))


def spread(use_ref: bool) -> float:
    img, ref, _ = _synthetic()
    h, w = img.shape[:2]
    _, x = np.mgrid[0:h, 0:w].astype(np.float64)
    peaks = []
    for g in (0.5, 0.7, 1.0, 1.3):
        v = (g * (0.6 + 0.4 * (x / w)))[..., None]
        peaks.append(_peak(img * v, ref * v, use_ref))
    a = np.array(peaks)
    return float(a.max() / max(a.min(), 1e-9))


def main() -> int:
    bad = []
    shipped, retired = spread(True), spread(False)
    print(f"  peak-depth spread over a 2.6x illumination range")
    print(f"    reference-normalised (shipped) : x{shipped:.2f}")
    print(f"    dI/255 (retired)               : x{retired:.2f}")
    if shipped > MAX_SPREAD:
        bad.append(f"illumination changes the recovered depth by "
                   f"x{shipped:.2f} (> {MAX_SPREAD}) — the reference "
                   f"normalisation is gone or broken")
    if retired < MIN_RETIRED_SPREAD:
        bad.append(f"the retired normalisation only spreads x{retired:.2f} — "
                   f"this test can no longer see the defect it guards, so a "
                   f"pass means nothing")

    for name in RECON:
        src = ROOT / "force_recovery" / name
        if not src.exists():
            continue
        for i, line in enumerate(src.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or '"""' in s:
                continue
            if ("force" in s and any(k in s for k in
                                     ("lstsq", "polyfit", "curve_fit",
                                      "IsotonicRegression"))):
                bad.append(f"{name}:{i}: the reconstruction is fitting on "
                           f"force — that is a calibration, not a solve")
    print(f"  reconstruction free of force fits: "
          f"{'yes' if not any('fitting on' in b for b in bad) else 'NO'}")
    for b in bad:
        print(f"  FAIL: {b}")
    print(f"calibfree-ratio: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
