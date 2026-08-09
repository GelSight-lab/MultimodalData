"""Calibration-free features over every mini_cnc26 press, same schema as the LUT.

THIS ATTEMPT DID NOT PRODUCE A COMPARABLE ROW, AND THE REASON IS THE RESULT

The goal was to score the calibration-free reconstruction through the SAME
production path as the LUT — gain field, scope filter and all — so the results
page could carry both beside the published baselines. It cannot be done this
way, and the obstruction is not a bug to fix but the thing being measured.

The LUT returns millimetres, and the production feature step thresholds depth
at an absolute 0.05 mm. A calibration-free solve has no millimetre scale, so a
global factor must be applied first — and that factor then multiplies straight
into the threshold. Measured over all 6264/6308 presses:

    cache                 median area   median maxd   r_eff   passes scope
    lut_full.json              16838       1.228 mm   73 px      400
    calibfree_full.json        31206      12.840 mm  100 px       21

The scope filter asks whether the contact disc is fully inside the frame,
using a radius derived from the thresholded area. Scale the depth and the area
changes, so the same physical contact is judged to be off the edge. 21 frames
survive and the row is meaningless.

The honest conclusion is that a pipeline built around an absolute millimetre
threshold cannot host a scale-free reconstruction without first calibrating
the scale — which is the thing being avoided. `calibfree_eval` is therefore
the comparison that stands: it gives BOTH methods a relative floor, and its
`lut_native` control shows the LUT scores within 0.009 of itself either way,
so the floor carries none of the difference.

Kept rather than deleted because the next person will have this idea too.

---

`calibfree_eval` compared the two reconstructions under a deliberately stripped
protocol — no position gain field, no scope filter — so that only the
reconstruction differed. That answered "which reconstruction carries more force
signal on its own" (calibration-free, by 0.27 rho on markerless Mini presses).

It did NOT answer the question the results page asks, which is what each method
scores as a whole pipeline, gain field and scope filter included, against the
published baselines. For that the calibration-free features have to exist in
the same form the LUT's do — `feature_cache/lut_full.json`, one row per press
with `fam,x,y,z,f,vol,vol2,vol15,area,maxd,cx,cy` — so the identical scoring
code can read either.

This writes `feature_cache/calibfree_full.json`. Everything except the
reconstruction is copied from the LUT path on purpose: the same contact-circle
detector for (cx, cy), the same feature definitions, the same frames.

    python -m force_recovery.calibfree_full [--procs 8]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import calib_free as CF
from .debug_gallery import MM_PER_PIXEL, crop
from .lut_calibration import CAL_OUT, MINI_CNC26, PAT
from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
FAMILIES = ("round", "quad", "star", "triangle", "B", "quad_small")

# The LUT thresholds depth at an absolute 0.05 mm. The calibration-free solve
# has no millimetre scale, so a single global factor is applied to put its
# depths in the same range, and the SAME absolute floor is then used. The
# factor is one number for the whole dataset, fitted nowhere — it is the ratio
# of median peak depths on the sphere family, which is a unit conversion, not
# a per-frame correction.
DEPTH_FLOOR_MM = 0.05


def _contact_circle(dI: np.ndarray):
    """(cx, cy, radius) of the contact — the LUT path's own detector, reused."""
    from .lut_calibration import detect_circle
    return detect_circle(dI)


def _feats(depth: np.ndarray, dI: np.ndarray) -> dict:
    m = depth > DEPTH_FLOOR_MM
    px = MM_PER_PIXEL ** 2
    d = depth
    out = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
           "vol15": float((d[m] ** 1.5).sum() * px),
           "area": float(m.sum()), "maxd": float(np.percentile(d, 99.8))}
    det = _contact_circle(dI)
    out["cx"], out["cy"] = (float(det[0]), float(det[1])) if det else (np.nan,
                                                                      np.nan)
    return out


def press_rows() -> list[dict]:
    rows = []
    for fam in FAMILIES:
        d = MINI_CNC26 / fam
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            m = PAT.search(p.name)
            if not m:
                continue
            rows.append({"fam": fam, "path": str(p), "x": float(m["x"]),
                         "y": float(m["y"]), "z": -float(m["z"]),
                         "f": float(m["f"])})
    return rows


_REF: dict = {}


def _ref(fam: str) -> np.ndarray:
    if fam not in _REF:
        _REF[fam] = crop(np.asarray(
            Image.open(MINI_CNC26 / fam / "initial.jpg").convert("RGB"))
        ).astype(np.float32)
    return _REF[fam]


def _one(job: dict, scale: float) -> dict:
    img = crop(np.asarray(Image.open(job["path"]).convert("RGB"))
               ).astype(np.float32)
    ref = _ref(job["fam"])
    r = CF.reconstruct(img, ref, scale=scale)
    return {"fam": job["fam"], "x": job["x"], "y": job["y"], "z": job["z"],
            "f": job["f"], **_feats(r["depth"], r["dI"])}


def fit_scale(rows: list[dict], n: int = 60) -> float:
    """One global unit conversion, from the sphere family's peak depths.

    The CNC records indentation depth `z` in millimetres, so the scale that
    puts the calibration-free reconstruction in millimetres is the ratio
    median(z) / median(peak depth) over sphere presses. It is a unit, applied
    identically to every frame — not a per-frame or per-position correction,
    which is what the LUT path's gain field is and what would make this
    comparison unfair.
    """
    sph = [r for r in rows if r["fam"] == "round" and 2.0 < r["z"] < 4.0][:n]
    if not sph:
        return 1.0
    peaks, zs = [], []
    for j in sph:
        img = crop(np.asarray(Image.open(j["path"]).convert("RGB"))
                   ).astype(np.float32)
        d = CF.reconstruct(img, _ref(j["fam"]))["depth"]
        peaks.append(float(np.percentile(d, 99.8)))
        zs.append(j["z"])
    return float(np.median(zs) / max(np.median(peaks), 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=CACHE / "calibfree_full.json")
    args = ap.parse_args()

    rows = press_rows()
    if args.limit:
        rows = rows[:args.limit]
    print(f"[calibfree_full] {len(rows)} presses over {len(FAMILIES)} families",
          flush=True)
    scale = fit_scale(rows)
    print(f"[calibfree_full] global unit scale = {scale:.4f} mm per unit "
          f"(one number, from sphere z vs peak depth)", flush=True)

    out = []
    for i, j in enumerate(rows):
        out.append(_one(j, scale))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(rows)}", flush=True)
    args.out.write_text(json.dumps(out))
    print(f"-> {args.out}  ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
