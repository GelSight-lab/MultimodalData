"""Search the calibration-free colour -> normal step, on force, one protocol.

The rules this search obeys, because breaking either would make a higher
number meaningless:

  * NO new force-fitting factor. Every variant is scored on the SAME five
    features (vol, vol2, maxd, area, h1) through `force_eval_all.evaluate` —
    the same per-group half/half least squares, isotonic calibration, 5 seeds
    and within-group shuffle control the published table uses. A variant may
    only change how an image becomes a surface.
  * NO per-sensor calibration. Anything a variant needs must come from the
    frame, the reference, or fixed sensor geometry. Fitting a constant on the
    force labels would be a lookup table with extra steps.

Frames are decoded once and held, so a variant costs only its reconstruction.

    python -m scripts.calibfree_search [--datasets cnc_mini_26 cnc] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sys as _sys
from pathlib import Path as _Path
# repo root, so `force_recovery` / `twm` / `react_toolbox` import however
# this file is invoked. Six scripts lacked this and failed at import; all
# six sat in validate_all's "slow" skip list, so nothing ran them.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


from force_recovery import debug_gallery as dg
from force_recovery.force_eval_all import evaluate
from force_recovery.run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "calibfree_search.json"
FEATURES = ("vol", "vol2", "maxd", "area", "h1")
DEPTH_FLOOR_FRAC = 0.05           # same relative floor as calibfree_eval

LOADERS = {"cnc_mini_26": "load_glowtact", "cnc": "load_cnc",
           "feats": "load_feats"}


def load(name: str, limit: int = 0):
    rows, get = getattr(dg, LOADERS[name])()
    if limit:
        rows = rows[:limit]
    frames = []
    for fr in rows:
        img, ref = get(fr)
        frames.append((np.asarray(img, np.float32), np.asarray(ref, np.float32),
                       float(fr["f"]), str(fr["group"])))
    return frames


def feats_from_depth(d: np.ndarray) -> list[float]:
    from force_recovery.debug_gallery import MM_PER_PIXEL
    m = d > DEPTH_FLOOR_FRAC * max(d.max(), 1e-9)
    px = MM_PER_PIXEL ** 2
    vol = float(d[m].sum() * px)
    vol2 = float((d[m] ** 2).sum() * px)
    maxd = float(np.percentile(d, 99.8))
    area = float(m.sum() * px)
    return [vol, vol2, maxd, area, float(np.sqrt(area) * maxd)]


def score(frames, recon, seeds: int = 5) -> dict:
    X, f, g = [], [], []
    for img, ref, force, grp in frames:
        X.append(feats_from_depth(recon(img, ref)))
        f.append(force)
        g.append(grp)
    return evaluate(np.array(X), np.array(f), np.array(g), seeds=seeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["cnc_mini_26"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*")
    args = ap.parse_args()

    from force_recovery import cf_variants as V

    table = {}
    for ds in args.datasets:
        print(f"== {ds}", flush=True)
        frames = load(ds, args.limit)
        print(f"   {len(frames)} frames held in memory", flush=True)
        names = args.variants or list(V.VARIANTS)
        table[ds] = {}
        for name in names:
            r = score(frames, V.VARIANTS[name])
            table[ds][name] = r
            pg = "  ".join(f"{k}={v:.3f}"
                           for k, v in sorted(r["per_group_rho"].items()))
            print(f"   {name:20s} rho {r['rho']:.4f} (sd {r['rho_sd']:.3f})  "
                  f"MAE {r['mae']:.2f}  shuf {r['shuffle_rho']:+.3f}  | {pg}",
                  flush=True)
    OUT.write_text(json.dumps(table, indent=1))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
