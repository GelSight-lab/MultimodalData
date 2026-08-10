"""LUT vs calibration-free force, same frames, same protocol, same split.

The question this answers is narrow and was asked directly: on React, can the
reconstruction run WITHOUT a per-sensor lookup table? React has no ground-truth
force, so the answer has to come from the datasets that do.

Both reconstructions are run over identical frames and reduced to the identical
five features (`vol, vol2, maxd, area, sqrt(area)*maxd`), then scored by the
protocol `force_eval_all` already uses — per-group half/half least squares,
isotonic calibration on the fit half, pooled Spearman rho, five seeds — beside
a within-group label shuffle. Only the reconstruction differs, so the
difference in rho is attributable to it and to nothing else.

What is NOT being asked: which reconstruction produces a prettier depth map.
That was asked earlier with a label-free proxy (depth outside the contact must
be zero) and the proxy picked an LED map that split a connector into two blobs.
Force labels are the arbiter here.

    python -m force_recovery.calibfree_eval [--datasets cnc_mini_26 cnc feats]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import calib_free as CF
from .debug_gallery import MM_PER_PIXEL, stages
from .force_eval_all import CACHE, evaluate

# Depth is thresholded at a fixed height before features are taken. The LUT
# returns millimetres; the calibration-free solve returns the same quantity up
# to one global factor, so the threshold has to scale with the data or it
# means something different for each. Taken as a fraction of that frame's own
# peak, which is scale-free and identical for both.
#
# THIS IS NOT THE DEPLOYED FEATURE LAW, and its rho is not comparable to the
# results table. This module is an ABLATION: six arms differing only in the
# reconstruction, held to one threshold so a difference in rho is attributable
# to the reconstruction rather than to the threshold. The deployed estimator
# (`react_calib.feature_vector`) decides contact on the difference image
# instead — a relative floor cannot express "no contact", since every frame
# has a peak. Keeping the uniform floor here is what makes the arms
# comparable to EACH OTHER; do not lift these numbers into the headline.
DEPTH_FLOOR_FRAC = 0.05

# Every arm scored on identical frames. `*_dirichlet` / `lut_neumann` exist so
# the boundary condition can be attributed on its own.
METHODS = ("lut", "lut_native", "calibfree", "calibfree_dirichlet",
           "lut_neumann", "calibfree_gains")


def _feats_from_depth(d: np.ndarray) -> dict:
    m = d > DEPTH_FLOOR_FRAC * max(d.max(), 1e-9)
    px = MM_PER_PIXEL ** 2
    f = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
         "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px)}
    f["h1"] = np.sqrt(f["area"]) * f["maxd"]
    return f


def features(img: np.ndarray, ref: np.ndarray, method: str) -> dict:
    """Features for one frame under one reconstruction.

    `lut_native` is the control that keeps this comparison honest. The LUT's
    own feature step thresholds depth at an ABSOLUTE 0.05 mm; the
    calibration-free solve has no millimetre scale, so it can only use a
    relative floor. Changing the reconstruction and the threshold at once
    would make any difference in rho unattributable — so the LUT is scored
    both ways and the gap between `lut` and `lut_native` says how much of any
    difference is the threshold rather than the reconstruction.
    """
    if method == "lut":
        return _feats_from_depth(stages(img, ref)["depth"])
    if method == "lut_native":
        return dict(stages(img, ref)["feats"])
    if method == "calibfree":
        return _feats_from_depth(CF.reconstruct(img, ref)["depth"])
    if method == "calibfree_gains":
        # Per-frame LED gain self-calibration (calib_free.channel_gains).
        return _feats_from_depth(
            CF.reconstruct(img, ref, gains=True)["depth"])
    if method == "calibfree_dirichlet":
        # The retired boundary condition, kept as an A/B arm. Only the solver
        # differs from `calibfree`, so any rho difference is the boundary and
        # nothing else. See force_recovery.poisson: the DST solver pins the
        # frame border to zero, which is false whenever a contact reaches it.
        return _feats_from_depth(
            CF.reconstruct(img, ref, solver="dirichlet")["depth"])
    if method == "lut_neumann":
        from .poisson import poisson_neumann
        st = stages(img, ref)
        gx = np.where(st["valid"], st["gx"], 0.0)
        gy = np.where(st["valid"], st["gy"], 0.0)
        d = poisson_neumann(gx, gy)
        if st["valid"].any() and np.median(d[st["valid"]]) < 0:
            d = -d
        if (~st["valid"]).sum() > 100:
            d = d - np.median(d[~st["valid"]])
        return _feats_from_depth(np.maximum(d, 0.0))
    raise KeyError(method)


def _load(name: str):
    """(iterable of (img, ref, force, group), label) for one GT dataset."""
    from .debug_gallery import load_cnc, load_feats, load_glowtact

    if name == "cnc_mini_26":
        rows, get = load_glowtact()
        return rows, get, "cnc_mini_26 (markerless, 0-20 N)"
    if name == "cnc":
        # FoTa's press grid spans x 0-20 mm, y 0-16 mm; the Mini sees about
        # 13.2 x 9.9 mm. So most of the grid is pressed OUTSIDE the field of
        # view, and this row was labelled "in view" while scoring all of it:
        # 83 of 390 frames actually qualified, and the other 307 are presses
        # the sensor can only see the edge of, or nothing of. `force_eval_all`
        # has always applied this filter, so the two published tables were
        # scoring different populations under the same name.
        #
        #     all frames     rho 0.277   (ceiling 0.814)   34% of ceiling
        #     in view        rho 0.588   (ceiling 0.908)   65% of ceiling
        #
        # The filter is the SAME expression force_eval_all uses. Sampling is
        # widened first so that filtering does not just shrink n.
        import os
        old_n = os.environ.get("CNC_N")
        os.environ["CNC_N"] = "3000"
        try:
            rows, get = load_cnc()
        finally:
            if old_n is None:
                os.environ.pop("CNC_N", None)
            else:
                os.environ["CNC_N"] = old_n
        rows = [r for r in rows
                if 5 < r["x"] < 13 and 4 < r["y"] < 12][:390]
        return rows, get, "FoTa cnc_Mini (markerless, in view)"
    if name == "feats":
        rows, get = load_feats()
        return rows, get, "FEATS (marker gel)"
    raise KeyError(name)


def build_cache(name: str, limit: int = 0) -> dict:
    rows, get, label = _load(name)
    if limit:
        rows = rows[:limit]
    out = {"dataset": name, "label": label, "rows": []}
    for i, fr in enumerate(rows):
        img, ref = get(fr)
        rec = {"f": float(fr["f"]), "group": str(fr["group"])}
        for meth in METHODS:
            rec[meth] = features(img, ref, meth)
        # Does the contact reach the frame border? That is the population the
        # boundary condition can possibly affect; averaged over a dataset of
        # centred CNC presses a real edge effect would be diluted to nothing.
        v = CF.contact_mask(img.astype("float32") - ref.astype("float32"))
        rec["edge"] = bool(v[0].any() or v[-1].any()
                           or v[:, 0].any() or v[:, -1].any())
        out["rows"].append(rec)
        if (i + 1) % 50 == 0:
            print(f"    {name}: {i+1}/{len(rows)}", flush=True)
    return out


def score(cache: dict, method: str, seeds: int = 5,
          subset: str = "all") -> dict:
    rows = cache["rows"]
    if subset == "edge":
        rows = [r for r in rows if r.get("edge")]
    elif subset == "interior":
        rows = [r for r in rows if not r.get("edge")]
    if len(rows) < 24:
        return {"rho": float("nan"), "mae": float("nan"),
                "shuffle_rho": float("nan"), "n": len(rows)}
    X = np.array([[r[method][k] for k in ("vol", "vol2", "maxd", "area", "h1")]
                  for r in rows], float)
    f = np.array([r["f"] for r in rows], float)
    g = np.array([r["group"] for r in rows])
    return evaluate(X, f, g, seeds=seeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["cnc_mini_26", "cnc", "feats"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=CACHE / "calibfree_vs_lut.json")
    args = ap.parse_args()

    table = []
    for name in args.datasets:
        print(f"== {name}", flush=True)
        cache = build_cache(name, args.limit)
        row = {"dataset": name, "label": cache["label"],
               "n": len(cache["rows"])}
        for meth in METHODS:
            row[meth] = score(cache, meth)
        row["n_edge"] = sum(bool(r.get("edge")) for r in cache["rows"])
        row["edge_subset"] = {m: score(cache, m, subset="edge")
                              for m in ("calibfree", "calibfree_dirichlet",
                                        "lut", "lut_neumann")}
        table.append(row)
        print(f"   n={row['n']}  LUT(rel)={row['lut']['rho']:.4f}  "
              f"LUT(native)={row['lut_native']['rho']:.4f}  "
              f"calib-free={row['calibfree']['rho']:.4f}", flush=True)

    args.out.write_text(json.dumps(table, indent=1))
    print()
    hdr = (f"{'dataset':38s}{'n':>5}{'LUT rel':>9}{'LUT nat':>9}"
           f"{'calib-free':>11}{'CF MAE':>9}{'shuffle':>9}")
    print(hdr)
    for r in table:
        print(f"{r['label']:38.36s}{r['n']:>5}"
              f"{r['lut']['rho']:>9.4f}{r['lut_native']['rho']:>9.4f}"
              f"{r['calibfree']['rho']:>11.4f}"
              f"{r['calibfree']['mae']:>9.3f}"
              f"{r['calibfree']['shuffle_rho']:>9.3f}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
