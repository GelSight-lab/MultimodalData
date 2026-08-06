"""Method optimization on the cached features — fit on train, judge on val.

Answers three questions with one table:
  * which single physical feature tracks force best (volume vs max depth
    vs area — the user's question, settled empirically);
  * how much a 1-2 parameter model on top of physics buys (power law,
    tiny linear combination) — still model-based, no images memorized;
  * what edge filtering is worth.

Every variant is fitted on the train split only and reported on val
(pooled ρ / r, per-probe median ρ, MAE in newtons after the fitted scale).
FEATS' own U-net on the same frames is the NN baseline.
"""
from __future__ import annotations

import json

import numpy as np

from .feature_cache import CACHE

EDGE_MM = 3.0          # contact centroid closer than this to the pad border


def _load(split: str):
    d = json.loads((CACHE / f"cnc_mini_{split}.json").read_text())
    rows = [r for r in d["rows"] if np.isfinite(r["vol"])]
    return rows


def _arr(rows, key):
    return np.array([r[key] for r in rows], float)


def _metrics(pred, truth, rows):
    from scipy.stats import pearsonr, spearmanr

    per_probe = []
    probes = np.array([r["probe"] for r in rows])
    for p in np.unique(probes):
        m = probes == p
        if m.sum() >= 8:
            per_probe.append(float(spearmanr(pred[m], truth[m]).statistic))
    return {
        "rho": float(spearmanr(pred, truth).statistic),
        "r": float(pearsonr(pred, truth).statistic),
        "rho_probe_med": float(np.median(per_probe)) if per_probe else None,
        "mae_n": float(np.abs(pred - truth).mean()),
        "n": len(pred),
    }


def _fit_scale(x, y):
    good = x > 0
    return float(np.median(y[good] / np.maximum(x[good], 1e-9))) \
        if good.any() else 0.0


def _fit_power(x, y):
    good = (x > 1e-3) & (y > 1e-2)
    lx, ly = np.log(x[good]), np.log(y[good])
    b, a = np.polyfit(lx, ly, 1)
    return float(np.exp(a)), float(b)


def _fit_linear(feats_tr, y_tr, keys):
    A = np.column_stack([feats_tr[k] for k in keys] + [np.ones(len(y_tr))])
    w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return w


def run(edge_filter_train: bool = True) -> dict:
    tr_rows, va_rows = _load("train"), _load("val")

    def prep(rows, edge_filter):
        if edge_filter:
            rows = [r for r in rows
                    if np.isfinite(r["border_mm"]) and r["border_mm"] > EDGE_MM]
        f = {k: _arr(rows, k) for k in
             ("vol", "vol_soft", "vol15", "area", "maxd", "force_true",
              "feats_pred_n")}
        return rows, f

    results = {}
    for tag, edge in (("all", False), ("edge_filtered", True)):
        tr, ftr = prep(tr_rows, edge and edge_filter_train)
        va, fva = prep(va_rows, edge)
        y_tr, y_va = ftr["force_true"], fva["force_true"]

        variants = {}
        # single features, median-scale fit
        for key in ("vol", "vol_soft", "vol15", "area", "maxd"):
            s = _fit_scale(ftr[key], y_tr)
            variants[f"{key} (scale)"] = _metrics(fva[key] * s, y_va, va)
        # power law on volume
        a, b = _fit_power(ftr["vol_soft"], y_tr)
        variants["vol_soft power a*x^b"] = _metrics(
            a * np.maximum(fva["vol_soft"], 1e-9) ** b, y_va, va)
        # tiny linear combo
        keys = ("vol_soft", "area", "maxd")
        w = _fit_linear(ftr, y_tr, keys)
        pred = sum(w[i] * fva[k] for i, k in enumerate(keys)) + w[-1]
        variants["linear[vol_soft,area,maxd]"] = _metrics(pred, y_va, va)
        # NN baseline on identical frames
        variants["FEATS U-net (markerless input)"] = _metrics(
            fva["feats_pred_n"], y_va, va)

        results[tag] = {"n_train": len(tr), "n_val": len(va),
                        "variants": variants}

    (CACHE / "optimize_report.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    res = run()
    for tag, block in res.items():
        print(f"=== {tag} (train {block['n_train']}, val {block['n_val']}) ===")
        for name, m in block["variants"].items():
            print(f"  {name:34s} rho={m['rho']:+.3f} r={m['r']:+.3f} "
                  f"probe_med={m['rho_probe_med']} mae={m['mae_n']:.2f}N")
