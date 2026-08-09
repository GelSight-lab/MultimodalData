"""Calibration-free depth recovers SHAPE, not millimetres — assert both halves.

`calib_free`'s docstring says "Scale is NOT recovered. This returns depth in
arbitrary units up to one global factor." `reconstruct` then returned that
number as if it were millimetres, and every consumer believed it: the pilot
figure showed peaks of 13.12 mm on a 4.25 mm gel, and the Open3D meshes — which
apply a FIXED z exaggeration — rendered those surfaces as vertical towers. The
geometry was never wrong; a scale-free quantity was being drawn as if it had a
scale.

Two assertions, because fixing one without the other hides the problem:

  SHAPE   normalised calibration-free depth agrees with the LUT's, per pixel,
          inside the contact. If this fails the reconstruction is broken.
  SCALE   raw calibration-free depth does NOT agree in magnitude, and the
          module must not present it as millimetres. If this ever starts
          passing, either the scale was calibrated (fine — update the test) or
          someone quietly multiplied by a constant they could not justify.

    python scripts/test_calibfree_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.force_recovery import calib_free as CF          # noqa: E402
from twm.force_recovery.debug_gallery import (load_glowtact,  # noqa: E402
                                              stages)

GEL_MM = 4.25
N = 12
SHAPE_RHO_MIN = 0.75


def main() -> int:
    rows, get = load_glowtact()
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.permutation(len(rows))[:N]]

    shape_rho, raw_ratio, over_gel = [], [], 0
    for fr in sel:
        img, ref = get(fr)
        lut = stages(img, ref)["depth"]
        r = CF.reconstruct(img, ref)
        cf, v = r["depth"], r["valid"]
        if v.sum() < 200 or lut.max() <= 0 or cf.max() <= 0:
            continue
        a, b = lut[v], cf[v]
        shape_rho.append(float(np.corrcoef(a / a.max(), b / b.max())[0, 1]))
        raw_ratio.append(float(cf.max() / lut.max()))
        over_gel += int(cf.max() > GEL_MM)

    rho = float(np.median(shape_rho))
    ratio = float(np.median(raw_ratio))
    print(f"[scale] {len(shape_rho)} presses")
    print(f"  shape agreement (normalised, per-pixel rho)   median {rho:.3f}")
    print(f"  raw magnitude ratio calib-free / LUT          median {ratio:.2f}")
    print(f"  raw peaks above the {GEL_MM} mm gel thickness      "
          f"{over_gel}/{len(shape_rho)}")

    problems = []
    if rho < SHAPE_RHO_MIN:
        problems.append(f"shape agreement {rho:.3f} < {SHAPE_RHO_MIN} — the "
                        f"calibration-free reconstruction is not recovering "
                        f"the same surface, which is a real defect and not a "
                        f"units question")
    if CF.RETURNS_MILLIMETRES:
        problems.append("calib_free claims to return millimetres. It solves a "
                        "dimensionless gradient and integrates it in pixels; "
                        "the number has no millimetre meaning until a scale is "
                        "calibrated. Set RETURNS_MILLIMETRES only when one is.")
    if not CF.RETURNS_MILLIMETRES and over_gel == 0 and ratio < 1.2:
        problems.append(f"raw magnitude now tracks the LUT (ratio {ratio:.2f}, "
                        f"0 peaks past the gel) while still declaring itself "
                        f"scale-free — if a scale was calibrated, say so; if a "
                        f"constant was slipped in, justify it")

    for p in problems:
        print(f"  FAIL: {p}")
    print(f"calibfree scale: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
