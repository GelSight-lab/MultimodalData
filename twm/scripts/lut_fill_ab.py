"""Can the sparse LUT be replaced by something that generalises?

THE PROBLEM, MEASURED

The table is (90, 90, 90, 2) = 729k bins holding 1.26M sphere-press samples:

    observed bins                        26.6%   (median 3 samples each)
    observed bins with < 10 samples      86.6%
    pixels landing on an UNOBSERVED bin  16.3%   (nearest-neighbour filled)
    pixels landing on a bin with < 10    57.0%

So over half of every reconstruction is served by bins fitted from fewer than
ten presses, and one pixel in six by a bin with no data at all. That is the
LUT's real defect — not a wrong value somewhere, a table that was never
filled.

WHAT IS COMPARED

Each arm maps a difference image to a gradient field. Everything after that —
the mask, the integration, the features, the protocol — is identical, so any
difference in force rho is the colour-to-gradient map and nothing else.

    nearest    the shipped table, unobserved bins nearest-filled
    linear     g = M @ dI, no intercept, fitted on the observed bins.
               Two 3-vectors. Enforces g(0) = 0 by construction, which the
               table violates (it returns 0.083 mm/px for dI = 0).
    quadratic  the same with second-order terms
    mlp        a small network, dI -> g, trained on the observed bins

The parametric arms are evaluated PER PIXEL, not per bin, so they also drop
the table's quantisation.

RESULTS (force rho, presses scored IN VIEW and imaged whole, same mask,
integrator, features and protocol for every arm)

    arm                 cnc_mini_26   FoTa cnc   FEATS (marker)
                          n = 404      n = 359      n = 200
    nearest (shipped)      0.9907      0.8979       0.7511
    linear                 0.9971      0.8763       0.5682
    quadratic              0.9944      0.9431       0.6974
    mlp                    0.9893      0.9219       0.6959

    agreement with the table on its own observed bins (weighted median
    |error|, mm/px):   linear 0.0149   quadratic 0.0109   mlp 0.0040

THE FIT-QUALITY ORDERING IS THE REVERSE OF THE USEFULNESS ORDERING

The MLP reproduces the table best and is never the best reconstruction. That
is what a table whose median bin holds 3 samples looks like: its fine
structure is noise, and capacity spent reproducing it is capacity spent
learning noise. Quadratic — six terms per output — is the best overall,
worth +0.045 rho on the foreign sensor over the shipped nearest-fill.

A CORRECTION TO MY OWN FIRST RUN

Measured on an UNFILTERED sample first, this read linear 0.8140 against
nearest 0.4877 and looked like a decisive win for dropping the table. That
population is dominated by contacts truncated by the frame edge, where depth
is not identifiable at all. Scored in view the same comparison is 0.9971 vs
0.9907. The effect was real but four times smaller than the first number, and
the first number was measuring the truncation, not the table.

FEATS is the one row where the raw table still wins. It is the marker gel; a
smooth colour-to-gradient map has no way to represent the dot lattice.

    python -m scripts.lut_fill_ab [--frames 200]
"""
from __future__ import annotations

import argparse

import numpy as np
import sys as _sys
from pathlib import Path as _Path
# repo root, so `force_recovery` / `twm` / `react_toolbox` import however
# this file is invoked. Six scripts lacked this and failed at import; all
# six sat in validate_all's "slow" skip list, so nothing ran them.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


from force_recovery import debug_gallery as dg
from force_recovery.debug_gallery import CNT, LUT
from force_recovery.force_eval_all import evaluate
from force_recovery.lut_calibration import BINS, DI_RANGE, MM_PER_PIXEL
from force_recovery.poisson import integrate

MIN_COUNT = 5          # bins with fewer samples are noise, not data


def training_set(min_count: int = MIN_COUNT):
    """Observed bin centres and their gradients. X (N,3) in grey levels."""
    obs = CNT >= min_count
    idx = np.array(np.nonzero(obs)).T.astype(np.float64)
    X = idx / (BINS - 1) * 2 * DI_RANGE - DI_RANGE
    y = LUT[obs]
    w = CNT[obs].astype(np.float64)
    return X, y, w


def _poly(X, order):
    c = [X]
    if order >= 2:
        c.append(X ** 2)
        c.append(np.stack([X[:, 0] * X[:, 1], X[:, 0] * X[:, 2],
                           X[:, 1] * X[:, 2]], axis=1))
    return np.concatenate(c, axis=1)


def fit_models(min_count: int = MIN_COUNT):
    X, y, w = training_set(min_count)
    out = {}
    sw = np.sqrt(w)[:, None]
    for name, order in (("linear", 1), ("quadratic", 2)):
        A = _poly(X, order)
        coef, *_ = np.linalg.lstsq(A * sw, y * sw, rcond=None)
        out[name] = ("poly", order, coef)
    try:
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=60,
                         early_stopping=True, random_state=0)
        keep = w >= 20              # train on the best-supported bins
        m.fit(X[keep] / DI_RANGE, y[keep])
        out["mlp"] = ("mlp", None, m)
        print(f"  mlp trained on {keep.sum()} bins (count >= 20)", flush=True)
    except Exception as exc:                                # noqa: BLE001
        print(f"  mlp unavailable: {exc}")
    print(f"  fitted on {len(X)} bins with count >= {min_count} "
          f"({w.sum():,.0f} samples)", flush=True)
    return out


def grad_of(model, dI):
    kind, order, obj = model
    flat = dI.reshape(-1, 3).astype(np.float64)
    if kind == "poly":
        g = _poly(flat, order) @ obj
    else:
        g = obj.predict(flat / DI_RANGE)
    return g.reshape(dI.shape[0], dI.shape[1], 2)


def grad_table(dI):
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    return LUT[q[..., 0], q[..., 1], q[..., 2]].copy()


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    models = fit_models()
    X, y, w = training_set()
    print("\n  agreement with the table on ITS OWN observed bins "
          "(count-weighted median |error|, mm/px):")
    for nm, m in models.items():
        pred = grad_of(m, X.reshape(1, -1, 3))[0]
        err = np.abs(pred - y).mean(axis=1)
        o = np.argsort(err)
        cw = np.cumsum(w[o]) / w.sum()
        print(f"    {nm:10s} {err[o][np.searchsorted(cw, 0.5)]:.4f}")

    rows, get = dg.load_glowtact()
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.permutation(len(rows))[:args.frames]]
    frames = [(*get(fr), fr["f"], fr["group"]) for fr in sel]
    print(f"\n  force rho over {len(frames)} cnc_mini_26 presses "
          f"(same mask, integrator, features, protocol):")

    def score(gfun):
        Xf, f, g = [], [], []
        for img, ref, force, grp in frames:
            dI = img - ref
            gg = gfun(dI)
            mag = cv2.GaussianBlur(np.abs(dI).max(2), (5, 5), 1.5)
            v = cv2.morphologyEx((mag > 8.0).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8)).astype(bool)
            gg = np.where(v[..., None], gg, 0.0)
            d, _ = integrate(gg[..., 0], gg[..., 1], v, ref=ref)
            if np.median(d[v]) < 0:
                d = -d
            d = np.clip(d, 0, None)
            m = d > 0.05 * max(d.max(), 1e-12)
            px = MM_PER_PIXEL ** 2
            area = float(m.sum() * px)
            maxd = float(np.percentile(d, 99.8))
            Xf.append([float(d[m].sum() * px), float((d[m] ** 2).sum() * px),
                       maxd, area, np.sqrt(area) * maxd])
            f.append(force)
            g.append(grp)
        return evaluate(np.array(Xf), np.array(f), np.array(g))

    r = score(grad_table)
    print(f"    {'nearest (shipped)':22s} rho {r['rho']:.4f}  "
          f"shuffle {r['shuffle_rho']:+.3f}", flush=True)
    for nm, m in models.items():
        r = score(lambda d, m=m: grad_of(m, d))
        print(f"    {nm:22s} rho {r['rho']:.4f}  "
              f"shuffle {r['shuffle_rho']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
