"""No reconstruction may report the gel compressed by more than it is thick.

WHY THE BOUND EXISTS

The free-boundary integrator is right about geometry and unbounded on contacts
the sensor only sees part of: the visible flank says the surface is descending
and where it stops is outside the image, so the height ramps with nothing to
stop it. Measured over 40 cnc_mini_26 presses, peak depth by anchoring:

    constant anchor   p50 3.61   p95 9.26   45% over the gel
    plane detrend     p50 2.66   p95 6.54   35%
    quadratic         p50 2.48   p95 5.25   18%
    Dirichlet         p50 1.28   p95 2.37    0%   (bounded by being wrong)

Better detrending only mitigates. On contacts imaged WHOLE the bound never
binds — 0 of 46 sampled frames reach it — so this is not a cosmetic clamp on
good data, it is a physical ceiling on frames whose depth is not identifiable.
Force rho over 300 presses: 0.5348 unbounded -> 0.5622 bounded (0.5225 if the
frames are dropped instead, so the bounded value carries more than nothing).

This asserts the ceiling holds, that it does NOT bind on fully imaged
contacts, and that the thickness has one definition.

    python -m scripts.test_gel_bound
"""
from __future__ import annotations

import numpy as np

from force_recovery import debug_gallery as dg
from force_recovery.debug_gallery import stages
from force_recovery.lut_calibration import GEL_THICKNESS_MM
from force_recovery.visible_eval import in_fov, visible

N = 40


def main() -> int:
    bad = []
    rows, get = dg.load_glowtact()
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.permutation(len(rows))[:N]]

    peaks, whole_peaks, whole = [], [], 0
    for fr in sel:
        img, ref = get(fr)
        st = stages(img, ref)
        pk = float(st["depth"].max())
        peaks.append(pk)
        if pk > GEL_THICKNESS_MM + 1e-9:
            bad.append(f"{fr.get('group','?')}: depth {pk:.3f} mm exceeds the "
                       f"{GEL_THICKNESS_MM} mm gel — the physical ceiling is "
                       f"not being applied")
        if in_fov(fr) and visible(img, ref):
            whole += 1
            whole_peaks.append(pk)
    print(f"  {len(sel)} presses: peak depth max {max(peaks):.3f} mm, "
          f"ceiling {GEL_THICKNESS_MM} mm")
    if whole_peaks:
        print(f"  of which {whole} imaged whole: peak max "
              f"{max(whole_peaks):.3f} mm")
        if max(whole_peaks) >= GEL_THICKNESS_MM - 1e-9:
            bad.append(f"the ceiling BINDS on a fully imaged contact "
                       f"({max(whole_peaks):.3f} mm) — it is meant to catch "
                       f"unidentifiable depths, not to clip good data")
    else:
        print("  no fully imaged contact in the sample — that half not checked")

    # one definition of the thickness
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    hits = []
    for p in sorted((root / "force_recovery").glob("*.py")):
        if p.name == "lut_calibration.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("GEL_THICKNESS_MM") and "=" in line \
                    and "import" not in line:
                hits.append(f"{p.name}:{i}")
    if hits:
        bad.append(f"gel thickness re-declared at {', '.join(hits)} — import "
                   f"it from lut_calibration")
    print(f"  extra declarations of the thickness: {len(hits)}")

    for b in bad:
        print(f"  FAIL: {b}")
    print(f"gel-bound: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
