"""Should React's force channel be computed from the calibration-free recon?

WHAT THIS DECIDES

`react_calib` is the step that turns a reconstruction into newtons: it fits on
GlowTact `round` presses of known load, holds out by PRESS POSITION (frames of
one press are near-duplicates, so a random split would score its own training
data), and calibrates with a least squares plus isotonic regression.

That least squares absorbs any global scale factor, so the calibration-free
solve not returning millimetres is irrelevant here — the arbitrary factor is
exactly what the fit determines. The only question is which reconstruction
carries more force signal, and it is answered by swapping the input and
changing nothing else.

MATCHED, or it decides nothing. Both feature sets are built over the SAME
frames, the intersection where both pass the >= 30 px depth filter is kept,
and both go through `react_calib.fit` itself — the production fitting code,
not a copy — with the identical split.

    python -m scripts.react_calib_recon_ab
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from force_recovery import calib_free as CF
from force_recovery import react_calib as RC
from force_recovery.debug_gallery import stages
from force_recovery.lut_calibration import GLOWTACT, MM_PER_PIXEL, PAT, crop

ARMS = ("lut", "calibfree")


def _feats(depth: np.ndarray, absolute_floor: bool):
    """react_calib's cache row, from a depth map. Same block for both arms."""
    d = np.clip(np.asarray(depth, np.float64), 0, None)
    m = d > 0.05 if absolute_floor else d > 0.05 * max(d.max(), 1e-12)
    if m.sum() < 30:
        return None
    px = MM_PER_PIXEL ** 2
    yy, xx = np.nonzero(m)
    w = d[m]
    out = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
           "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px),
           "cx": float((xx * w).sum() / w.sum()),
           "cy": float((yy * w).sum() / w.sum())}
    out["h1"] = np.sqrt(out["area"]) * out["maxd"]
    return out


# The LUT is in millimetres, so its floor is the absolute 0.05 mm the
# production path uses. The calibration-free solve has no millimetre scale, so
# the same absolute number would mean something different for it; its floor is
# the same fraction of its own peak. This is the ONLY asymmetry, it is forced,
# and it is the one `calibfree_eval` already documents.


def main() -> int:
    ref = crop(np.asarray(Image.open(GLOWTACT / "round" / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    rows = {a: [] for a in ARMS}
    kept = dropped = 0
    files = sorted((GLOWTACT / "round").glob("*.jpg"))
    for p in files:
        m = PAT.search(p.name)
        if not m or not (0.15 < float(m["f"]) <= RC.F_MAX_N):
            continue
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        a = _feats(stages(img, ref)["depth"], absolute_floor=True)
        b = _feats(CF.reconstruct(img, ref)["depth"], absolute_floor=False)
        if a is None or b is None:
            dropped += 1
            continue
        kept += 1
        meta = {"f": float(m["f"]), "x": float(m["x"]), "y": float(m["y"]),
                "z": -float(m["z"])}
        rows["lut"].append({**a, **meta})
        rows["calibfree"].append({**b, **meta})
        if kept % 100 == 0:
            print(f"  {kept} matched frames", flush=True)

    print(f"\nmatched {kept} frames ({dropped} dropped by one arm or the "
          f"other)\n")
    orig = RC.CACHE
    try:
        for arm in ARMS:
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as fh:
                json.dump(rows[arm], fh)
                RC.CACHE = Path(fh.name)
            print(f"== {arm}")
            _, h = RC.fit(report=True, holdout=True)
            inview = h["clip"] <= 0
            for label, msk in (("fully in view", inview), ("clipped", ~inview)):
                if msk.sum() >= 12:
                    r = spearmanr(h["pred"][msk], h["f"][msk]).statistic
                    print(f"     {label:14s} n={msk.sum():3d}  rho={r:.3f}  "
                          f"MAE={np.abs(h['pred'][msk]-h['f'][msk]).mean():.3f} N")
    finally:
        RC.CACHE = orig
    print("\nSame frames, same split, same fitting code — only the "
          "reconstruction differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
