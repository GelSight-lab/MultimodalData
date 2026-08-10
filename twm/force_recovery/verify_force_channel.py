"""Is the published force channel good? Compare it against an independent one.

The React dataset ships 480,080 estimated newtons produced by ONE
reconstruction — a colour lookup table built on `cnc_mini_26`, a GelSight Mini
whose rest-gel hue sits 79-87 degrees from React's pads. Ground-truth force
does not exist for React, so "is the channel good" cannot be answered directly.

What CAN be answered: does a second, independent estimator agree with it. The
calibration-free reconstruction shares no calibration, no lookup table and no
sphere presses with the shipped one — only the frames, the feature definitions
and the newton fit. Where two estimators built on different assumptions agree,
the number is probably about the contact; where they disagree, at least one is
wrong and the frames can be looked at.

This is a weaker claim than a load cell and it is the strongest one available.
It is reported as agreement, never as accuracy.

VERDICT (1,500 React frames, 6 episodes, both sensors)

    calibration-free, held out by press position on React's own
      newton calibration (sphere family, 0-8 N)      rho 0.310  MAE 1.68 N
    LUT (react_calib), same split, same frames       rho 0.739  MAE 1.23 N

    agreement between the two on React frames        rho 0.857
    mean |difference|                                0.72 N
    p95 |difference|                                 3.39 N

**The shipped channel stays.** This reverses the direction an earlier
comparison pointed: `calibfree_eval` had calibration-free ahead by 0.27 rho,
but that ran over six indenter families at 0-20 N with no position gain field.
React's newton scale is fitted on the SPHERE family at 0-8 N with the gain
field and the clipping correction, and in that domain the lookup table is more
than twice as good. A method that wins on one scope and loses on the deployed
scope has not won.

So there is no evidence to justify recomputing 480,080 published values, and
the agreement at rho 0.857 between two estimators sharing no calibration is
consistent with both measuring contact. It is NOT a certification of the
absolute scale — p95 disagreement is 3.39 N against a 7.29 N ceiling, which is
the same caveat the dataset README already carries: reliable for how hard
relative to other frames, not a load cell.

    python -m force_recovery.verify_force_channel [--frames N] [--rebuild]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import calib_free as CF
from . import react_calib as RC
from .debug_gallery import MM_PER_PIXEL, crop, stages
from .lut_calibration import CNC_MINI_26, PAT
from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
CF_CACHE = CACHE / "react_calib_calibfree.json"

EPISODES = [
    ("motherboard", "2026-05-10", "episode_000", "left"),
    ("motherboard", "2026-05-10", "episode_000", "right"),
    ("motherboard", "2026-05-11", "episode_005", "left"),
    ("motherboard", "2026-05-19", "episode_000", "left"),
    ("pushT", "2026-06-18", "episode_001", "right"),
    ("pushT", "2026-06-18", "episode_002", "left"),
]


def _feats_calibfree(img, ref) -> dict:
    """Same five features, same units, from the calibration-free depth.

    The scale is fixed later by the newton fit, exactly as it is for the LUT —
    a linear least squares absorbs one global factor, so a scale-free
    reconstruction costs nothing here that a table-based one does not.
    """
    d = CF.reconstruct(img, ref)["depth"]
    m = d > 0.05 * max(d.max(), 1e-9)
    px = MM_PER_PIXEL ** 2
    f = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
         "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px)}
    f["h1"] = np.sqrt(f["area"]) * f["maxd"]
    return f, d, m


def build_calibfree_cache() -> None:
    """React's newton fit, on calibration-free features. Same frames as the LUT."""
    from PIL import Image

    ref = crop(np.asarray(Image.open(CNC_MINI_26 / "round" / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    rows = []
    files = sorted((CNC_MINI_26 / "round").glob("*.jpg"))
    for i, p in enumerate(files):
        m = PAT.search(p.name)
        if not m:
            continue
        fN = float(m["f"])
        if not (0.15 < fN <= RC.F_MAX_N):
            continue
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        feats, d, mm = _feats_calibfree(img, ref)
        if mm.sum() < 30:
            continue
        yy, xx = np.nonzero(mm)
        w = d[mm]
        rows.append({**feats, "f": fN, "x": float(m["x"]), "y": float(m["y"]),
                     "z": -float(m["z"]),
                     "cx": float((xx * w).sum() / w.sum()),
                     "cy": float((yy * w).sum() / w.sum())})
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(files)} -> {len(rows)} kept", flush=True)
    CF_CACHE.write_text(json.dumps(rows))
    print(f"{len(rows)} frames -> {CF_CACHE}")


def fit_calibfree():
    """RC.fit's procedure, on the calibration-free cache — same code path."""
    orig = RC.CACHE
    RC.CACHE = CF_CACHE
    try:
        return RC.fit(report=True)
    finally:
        RC.CACHE = orig


def compare(n_frames: int = 400) -> dict:
    """Per-frame LUT vs calibration-free newtons on React."""
    from . import showcase as S

    # BOTH arms through react_calib's own fitting code, differing only in the
    # reconstruction. This module used to carry a private copy of the fit so it
    # could have a second arm; once the force channel moved to the
    # calibration-free solve that copy WAS the same arm twice, and the "LUT"
    # arm was feeding stages() features into a calibration-free-fitted model.
    # The runtime check in react_calib.predict caught it.
    for r in ("calibfree", "lut"):
        if not RC.cache_for(r).exists():
            print(f"[verify] building the {r} calibration cache", flush=True)
            RC.build_cache(r)
    pred_cf = RC.fit(report=False, recon="calibfree")
    pred_lut = RC.fit(report=False, recon="lut")

    out = []
    for task, date, ep, side in EPISODES:
        z, is_new, inten, pose, frame, ref, h5 = S._react_context(
            task, date, ep, side)
        rows = np.flatnonzero(is_new)
        if len(rows) > n_frames:
            rows = rows[np.linspace(0, len(rows) - 1, n_frames).astype(int)]
        for r in rows:
            img = crop(frame(int(r))).astype(np.float32)
            a = float(pred_lut(RC.force_stages(img, ref, recon="lut")))
            b = float(pred_cf(RC.force_stages(img, ref, recon="calibfree")))
            out.append({"ep": f"{task}/{date}/{ep}/{side}", "row": int(r),
                        "lut_n": a, "cf_n": b})
        print(f"  {task}/{ep}/{side}: {len(rows)} frames", flush=True)
    return {"frames": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", type=Path, default=CACHE / "force_agreement.json")
    args = ap.parse_args()

    if args.rebuild or not CF_CACHE.exists():
        print("[verify] building the calibration-free newton cache", flush=True)
        build_calibfree_cache()

    res = compare(args.frames)
    fr = res["frames"]
    a = np.array([r["lut_n"] for r in fr])
    b = np.array([r["cf_n"] for r in fr])
    from scipy.stats import spearmanr, pearsonr
    res["n"] = len(fr)
    res["spearman"] = float(spearmanr(a, b).statistic)
    res["pearson"] = float(pearsonr(a, b)[0])
    res["mad_n"] = float(np.abs(a - b).mean())
    res["p95_abs_diff_n"] = float(np.percentile(np.abs(a - b), 95))
    res["lut_range"] = [float(a.min()), float(a.max())]
    res["cf_range"] = [float(b.min()), float(b.max())]
    worst = np.argsort(-np.abs(a - b))[:12]
    res["worst"] = [fr[i] for i in worst]
    args.out.write_text(json.dumps(res, indent=1))

    print()
    print(f"frames compared      {res['n']}")
    print(f"Spearman rho         {res['spearman']:.4f}")
    print(f"Pearson r            {res['pearson']:.4f}")
    print(f"mean |difference|    {res['mad_n']:.3f} N")
    print(f"p95 |difference|     {res['p95_abs_diff_n']:.3f} N")
    print(f"LUT range            {res['lut_range'][0]:.2f} - "
          f"{res['lut_range'][1]:.2f} N")
    print(f"calib-free range     {res['cf_range'][0]:.2f} - "
          f"{res['cf_range'][1]:.2f} N")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
