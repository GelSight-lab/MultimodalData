"""Stage 1 on its own: is the DEPTH any good, with no force labels at all?

Force estimation here is two steps and they fail differently:

    1  image -> depth        no ground truth exists for this
    2  depth -> newtons      ground truth exists, and is what every rho reports

A rho only ever scores the pair. A reconstruction that is geometrically wrong
but monotone in contact size still gets a high rho, so stage 2 cannot be used
to certify stage 1. This scores stage 1 alone.

WITHOUT LABELS, WHAT IS THERE TO MEASURE

Four things, each a physical statement that must hold whatever the force was:

  flat-gel leak      mean |depth| off-contact over peak depth. The gel away
                     from the indenter is not touched, so this is 0 for a
                     coherent surface. `calib_free.flat_gel_leak`.
  over the gel       fraction of frames whose peak exceeds the 4.25 mm
                     elastomer. Physically impossible; only meaningful for the
                     LUT, which is the one in millimetres.
  truncated          fraction whose contact reaches the frame border. Not a
                     defect of the method — a fact about the capture — but it
                     bounds what any reconstruction can know, so it is
                     reported beside the rest.
  agreement          per-pixel Spearman between the LUT and the
                     calibration-free depth INSIDE the contact. Two methods
                     sharing no calibration and no lookup table agreeing on
                     the shape is evidence neither is inventing it; it is not
                     evidence that either is right.

None of these can replace looking at the panels, which is why the site shows
both. They catch the failures an eye misses across hundreds of frames.

    python -m force_recovery.depth_eval [--per-dataset 120]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "depth_eval.json"
DATASETS = (("cnc_mini_26", "GelSight Mini CNC"),
            ("cnc", "FoTa cnc_Mini"),
            ("feats", "FEATS (marker)"),
            ("sparsh", "Sparsh / Meta"),
            ("faf", "FeelAnyForce"))


def evaluate_one(name: str, cap: int) -> dict:
    from scipy.stats import spearmanr

    from . import calib_free as CF
    from .debug_gallery import stages
    from .force_recon_matrix import _rows
    from .lut_calibration import GEL_THICKNESS_MM
    from .poisson import contact_truncated

    rows, get = _rows(name)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))][:cap]
    leak_lut, leak_cf, over, trunc, agree, peaks = [], [], [], [], [], []
    for fr in rows:
        img, ref = get(fr)
        st = stages(img, ref)
        r = CF.reconstruct(img, ref)
        v = r["valid"]
        if not v.any():
            continue
        leak_lut.append(CF.flat_gel_leak(st["depth"], st["valid"]))
        leak_cf.append(CF.flat_gel_leak(r["depth"], v))
        peaks.append(float(st["depth"].max()))
        over.append(float(st["depth"].max()) >= GEL_THICKNESS_MM - 1e-9)
        trunc.append(bool(r["truncated"]))
        core = st["depth"] > 0.2 * max(st["depth"].max(), 1e-9)
        if core.sum() > 200:
            agree.append(spearmanr(st["depth"][core], r["depth"][core]).statistic)
    f = lambda a: float(np.nanmedian(a)) if len(a) else float("nan")  # noqa: E731
    return {"dataset": name, "n": len(trunc),
            "leak_lut": f(leak_lut), "leak_calibfree": f(leak_cf),
            "peak_lut_mm": f(peaks),
            "over_gel_frac": float(np.mean(over)) if over else float("nan"),
            "truncated_frac": float(np.mean(trunc)) if trunc else float("nan"),
            "shape_agreement": f(agree)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=120)
    args = ap.parse_args()

    table = []
    for name, label in DATASETS:
        try:
            r = evaluate_one(name, args.per_dataset)
        except Exception as exc:                               # noqa: BLE001
            print(f"  {name}: UNAVAILABLE — {exc}", flush=True)
            continue
        r["label"] = label
        table.append(r)
        print(f"  {name:12s} n={r['n']:4d}  leak LUT {r['leak_lut']:.3f} / "
              f"CF {r['leak_calibfree']:.3f}  peak {r['peak_lut_mm']:.2f} mm  "
              f"over-gel {r['over_gel_frac']*100:.0f}%  "
              f"truncated {r['truncated_frac']*100:.0f}%  "
              f"agree {r['shape_agreement']:.3f}", flush=True)
    OUT.write_text(json.dumps(table, indent=1))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
