"""Does relaxing the truncation test cost anything? Ask the score.

`visible()` rejected a frame if any pixel of the contact core touched any
border. That is too strict — most of the frames it threw away were clipped by
a hair — but "too strict" is a claim about the score, not about frame counts.
Loosening ANY validity rule recovers frames; the question is whether the
recovered frames still support the ranking, or whether they are noise that the
old rule was correctly excluding.

So: re-score every dataset at a range of thresholds, on the SAME evaluation
protocol the results table uses, and keep the largest threshold that does not
lower rho. Both arms are re-scored, because a filter change moves the
population under both and a one-armed comparison would confuse the two.

Three of the five datasets — feats, sparsh, faf — ship no press coordinates,
so `in_fov` returns True for every frame there and this test is their ONLY
truncation gate. They are the ones that constrain the answer.

    python scripts/truncation_threshold_sweep.py [--datasets ...] [--n 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from force_recovery import calib_free as CF                     # noqa: E402
from force_recovery import visible_eval as VE                   # noqa: E402
from force_recovery.debug_gallery import stages                 # noqa: E402
from force_recovery.force_eval_all import evaluate              # noqa: E402
from force_recovery.force_recon_matrix import _feats, _rows     # noqa: E402
from force_recovery.run_episode import OUT_ROOT                 # noqa: E402

OUT = OUT_ROOT / "feature_cache" / "truncation_sweep.json"
CANDIDATES = [0.0, 0.3, 0.6, 1.0, 1.5, float("inf")]
DATASETS = ["cnc_mini_26", "cnc", "feats", "sparsh", "faf"]


def scan(name: str, n: int):
    """Every frame's chord ratio, force, group — measured ONCE per dataset.

    The ratio does not depend on the threshold, so scoring six thresholds does
    not mean decoding six times. Reconstruction is the expensive part and it is
    also threshold-independent, so both feature vectors are cached here too.
    """
    rows, get = _rows(name)
    out = []
    for fr in rows[:n]:
        if not VE.in_fov(fr):
            continue
        img, ref = get(fr)
        r = VE.edge_chord_ratio(img, ref)
        if not np.isfinite(r):
            continue                     # no contact — excluded at every threshold
        out.append({
            "ratio": float(r), "f": float(fr["f"]), "g": str(fr["group"]),
            "cf": _feats(CF.reconstruct(img, ref), False),
            "lut": _feats(stages(img, ref), True),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    table = []
    for ds in args.datasets:
        pool = scan(ds, args.n)
        ratios = np.array([p["ratio"] for p in pool])
        print(f"\n== {ds}  {len(pool)} frames in field of view, "
              f"chord ratio p50 {np.median(ratios):.3f} "
              f"p90 {np.percentile(ratios, 90):.3f}")
        print(f"   {'threshold':>9}  {'kept':>12}  {'groups':>6}  "
              f"{'calibfree':>9}  {'LUT':>7}")
        rows = []
        for t in CANDIDATES:
            keep = [p for p in pool if p["ratio"] <= t]
            g = np.array([p["g"] for p in keep])
            if len(keep) < 40 or len(set(g)) < 2:
                print(f"   {t:9.2f}  {len(keep):5d} too few")
                continue
            f = np.array([p["f"] for p in keep])
            r = {"threshold": t, "n": len(keep), "groups": len(set(g))}
            for arm in ("cf", "lut"):
                X = np.array([p[arm] for p in keep])
                e = evaluate(X, f, g)
                r[arm] = {"rho": e["rho"], "rho_sd": e["rho_sd"],
                          "shuffle": e["shuffle_rho"]}
            rows.append(r)
            print(f"   {t:9.2f}  {len(keep):5d} ({len(keep)/len(pool):4.0%})  "
                  f"{r['groups']:6d}  {r['cf']['rho']:9.4f}  "
                  f"{r['lut']['rho']:7.4f}")
        table.append({"dataset": ds, "pool": len(pool), "rows": rows})

    # THE GATE. Live threshold must not cost rho against the strict rule on any
    # dataset, beyond the seed spread the protocol already reports.
    print(f"\n{'':16s}{'strict (0.0)':>22s}{'live':>22s}")
    bad = 0
    for t in table:
        by = {r["threshold"]: r for r in t["rows"]}
        s, live = by.get(0.0), by.get(VE.EDGE_CHORD_RATIO)
        if not s or not live:
            print(f"  {t['dataset']:14s}  NOT COMPARABLE (a threshold was skipped)")
            bad += 1
            continue
        for arm in ("cf", "lut"):
            d = live[arm]["rho"] - s[arm]["rho"]
            tol = 2 * max(s[arm]["rho_sd"], live[arm]["rho_sd"])
            ok = d >= -tol
            bad += not ok
            print(f"  {t['dataset']:10s} {arm:4s}  {s[arm]['rho']:10.4f}"
                  f"{live[arm]['rho']:22.4f}   {d:+.4f} "
                  f"(tol {tol:.4f}) {'ok' if ok else 'COSTS RHO'}")
    OUT.write_text(json.dumps(
        {"live_threshold": VE.EDGE_CHORD_RATIO, "candidates": CANDIDATES,
         "n_per_dataset": args.n, "table": table}, indent=1))
    print(f"\nthreshold {VE.EDGE_CHORD_RATIO}: {bad} arm(s) cost rho -> {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    from force_recovery.artifact_lock import one_writer
    with one_writer(OUT):
        raise SystemExit(main())
