"""0-15 N force model search on the v8 evaluation split.

Why this exists
---------------
v8 (`react_calib.fit`) scores 1.975 N MAE on its 283 held-out round presses.
Most of that error comes from high-load presses whose frozen v7 linear score
stays in the low part of the isotonic curve, where the appended tail cannot
reach them. v8 is structurally unable to fix that, because it keeps the v7
features and weights on purpose.

This script refits a force model over the whole 0-15 N range, using the
759-d observation features from `model_search.observation_features`, on the
SAME position split as v8: the same 283 test samples and the same held-out
positions. It never changes deployment.

Rules this follows (see evidence-driven-optimization):
  * model selection uses only GroupKFold over the TRAINING positions;
  * the held-out 283 are scored after selection and are never used to choose;
  * the v8 number is recomputed here on the same rows, not quoted.

Run:
    OPENBLAS_NUM_THREADS=8 python -m twm.force_recovery.fullrange_search
"""
from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np

from . import react_calib as RC

ROOT = RC.CACHE.parent
FULL_CACHE = ROOT / "bounded_force_0_15_2026-09-16" / "round_v1_di4.joblib"
OUT = ROOT / "fullrange_search_2026-09-24"


def _pos_key(rows):
    return (np.round([r["x"] for r in rows], 1) * 1000
            + np.round([r["y"] for r in rows], 1))


def _sig(r):
    # The v8 caches store z negated relative to the file name.
    return (round(r["x"], 2), round(r["y"], 2), round(abs(r["z"]), 2), round(r["f"], 2))


def v8_split():
    """Return (held-out position keys, set of v8 held-out sample signatures)."""
    base = json.loads(RC.CACHE.read_text())
    tail = json.loads(RC.TAIL_CACHE.read_text())
    bk = _pos_key(base)
    uniq = np.unique(bk)
    hold = set(uniq[np.random.default_rng(0).permutation(len(uniq))[:max(len(uniq) // 3, 1)]])
    tk = _pos_key(tail)
    ht = RC.tail_holdout_mask(tk, bk, hold)
    test_pos = set(hold) | set(tk[ht])
    v8_test = {_sig(r) for r, k in zip(base, bk) if k in hold} | {
        _sig(r) for r, m in zip(tail, ht) if m}
    v8_all = {_sig(r) for r in base} | {_sig(r) for r in tail}
    return test_pos, v8_test, v8_all


def load():
    """Rows, feature dict, and masks: train, test (== v8's 283)."""
    rows, data = joblib.load(FULL_CACHE)
    test_pos, v8_test, v8_all = v8_split()
    key = _pos_key(rows)
    sigs = [_sig(r) for r in rows]
    if max(Counter(sigs).values()) != 1:
        raise ValueError("Duplicate samples in the full-range cache")
    in_test_pos = np.isin(key, list(test_pos))
    test = np.array([s in v8_test for s in sigs])
    if test.sum() != len(v8_test) or not np.all(in_test_pos[test]):
        raise AssertionError("Cache does not reproduce the v8 held-out set")
    # Samples at held-out positions that v8 did not score (the <=0.15 N
    # presses) are dropped entirely: training on them would leak the position.
    train = ~in_test_pos
    return rows, data, key, train, test


def v8_predictions(rows, test):
    """v8 predictions for the same 283 rows, aligned to `rows[test]`."""
    _model, held = RC.fit(report=False, holdout=True)
    base = json.loads(RC.CACHE.read_text())
    tail = json.loads(RC.TAIL_CACHE.read_text())
    test_pos, v8_test, _ = v8_split()
    bk = _pos_key(base)
    order = [_sig(r) for r, k in zip(base, bk) if k in test_pos and _sig(r) in v8_test]
    tk = _pos_key(tail)
    ht = RC.tail_holdout_mask(tk, bk, {k for k in test_pos if k in set(bk)})
    order += [_sig(r) for r, m in zip(tail, ht) if m]
    lookup = dict(zip(order, held["pred"]))
    flook = dict(zip(order, held["f"]))
    sig = [_sig(rows[i]) for i in np.flatnonzero(test)]
    pred = np.array([lookup[s] for s in sig])
    np.testing.assert_allclose([flook[s] for s in sig], [rows[i]["f"] for i in np.flatnonzero(test)])
    return pred


def bands(y, p):
    out = []
    for lo, hi in ((0, 1), (1, 4), (4, 8), (8, 12), (12, 15.01)):
        m = (y >= lo) & (y < hi)
        out.append({"range_n": [lo, hi], "n": int(m.sum()),
                    "mae_n": float(np.abs(p[m] - y[m]).mean()) if m.any() else None})
    return out


def position_bootstrap(y, p, groups, seed=42, n=2000):
    err = np.abs(p - y)
    u = np.unique(groups)
    s = np.array([err[groups == g].sum() for g in u])
    c = np.array([(groups == g).sum() for g in u])
    d = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    b = s[d].sum(1) / c[d].sum(1)
    return np.percentile(b, [2.5, 97.5]).tolist()


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
from sklearn.base import BaseEstimator, RegressorMixin, clone  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402


class KRR(BaseEstimator, RegressorMixin):
    """RBF kernel ridge (sklearn 1.1's KernelRidge breaks on SciPy >= 1.14)."""

    def __init__(self, alpha=1e-2, gamma=1e-3):
        self.alpha, self.gamma = alpha, gamma

    def _k(self, a, b):
        from scipy.spatial.distance import cdist
        return np.exp(-self.gamma * cdist(a, b, "sqeuclidean"))

    def fit(self, x, y):
        import scipy.linalg as sl
        self.x_ = np.asarray(x, float)
        self.mean_ = float(np.mean(y))
        k = self._k(self.x_, self.x_)
        k[np.diag_indices_from(k)] += self.alpha
        self.coef_ = sl.solve(k, np.asarray(y, float) - self.mean_, assume_a="pos")
        return self

    def predict(self, x):
        return self._k(np.asarray(x, float), self.x_) @ self.coef_ + self.mean_


def _sqrt_target(est):
    return TransformedTargetRegressor(est, func=np.sqrt, inverse_func=np.square)


def candidates(dims):
    out = []
    for feat, d in dims.items():
        for a in (0.1, 1, 10, 100):
            out.append((f"ridge_a{a}", feat, make_pipeline(StandardScaler(), Ridge(alpha=a, solver="svd"))))
            out.append((f"ridge_a{a}_sqrt", feat, _sqrt_target(make_pipeline(StandardScaler(), Ridge(alpha=a, solver="svd")))))
        for a in (1e-3, 1e-2, 1e-1):
            for g in (0.03, 0.1, 0.3, 1):
                k = lambda: KRR(alpha=a, gamma=g / d)  # noqa: E731
                out.append((f"krr_a{a}_g{g}", feat, make_pipeline(StandardScaler(), k())))
                out.append((f"krr_a{a}_g{g}_sqrt", feat, _sqrt_target(make_pipeline(StandardScaler(), k()))))
        for c in (10, 100, 1000):
            for g in (0.03, 0.1, 0.3):
                out.append((f"svr_C{c}_g{g}", feat, make_pipeline(
                    StandardScaler(), SVR(C=c, gamma=g / d, epsilon=0.1))))
        out.append(("hgb", feat, HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1, random_state=0)))
        out.append(("extra", feat, ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.5, random_state=0, n_jobs=8)))
    return out


def cv_predict(est, x, y, groups, folds=5):
    pred = np.empty(len(y))
    for tr, va in GroupKFold(folds).split(x, y, groups):
        pred[va] = clone(est).fit(x[tr], y[tr]).predict(x[va])
    return np.clip(pred, 0, RC.F_MAX_N)


def search(feature_sets=("basic", "geometry", "image", "combined")):
    rows, data, key, train, test = load()
    y = np.array([r["f"] for r in rows])
    dims = {k: data[k].shape[1] for k in feature_sets}
    results = []
    for i, (name, feat, est) in enumerate(candidates(dims)):
        p = cv_predict(est, data[feat][train], y[train], key[train])
        results.append({"name": name, "feature": feat, "cv_mae_n": float(np.abs(p - y[train]).mean())})
        if (i + 1) % 20 == 0:
            best = min(results, key=lambda r: r["cv_mae_n"])
            print(f"{i+1}: best {best}", flush=True)
    results.sort(key=lambda r: r["cv_mae_n"])
    return results


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = search()
    (OUT / "cv_search.json").write_text(json.dumps(res, indent=1))
    for r in res[:25]:
        print(r)


def v8_from_basic(basic, noise_area_mm2=0.0, _cache={}):
    """v8 prediction from the cached `basic` row (vol, vol2, maxd, area, h1, cx, cy).

    `predict` needs a depth map only for the depth-weighted centroid, so a
    four-pixel bilinear stand-in at (cx, cy) reproduces it exactly. Checked
    against the 283 held-out v8 values in `check_v8_from_basic`.
    """
    if "p" not in _cache:
        _cache["p"] = RC.fit(report=False)
    vol, vol2, maxd, area, h1, cx, cy = map(float, basic)
    d = np.zeros((240, 320))
    x0, y0 = int(np.floor(cx)), int(np.floor(cy))
    fx, fy = cx - x0, cy - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        d[y0 + dy, x0 + dx] = w + 1e-12
    m = np.zeros(d.shape, bool)
    m[y0:y0 + 6, x0:x0 + 6] = True
    st = {"recon": RC.FORCE_RECONSTRUCTION, "depth": d, "contact": m,
          "feats": {"vol": vol, "vol2": vol2, "maxd": maxd, "area": area, "h1": h1}}
    return _cache["p"](st, noise_area_mm2=noise_area_mm2)


def check_v8_from_basic():
    rows, data, key, train, test = load()
    a = np.array([v8_from_basic(b) for b in data["basic"][test]])
    b = v8_predictions(rows, test)
    print("v8_from_basic max |diff| on the 283:", np.abs(a - b).max())
    return np.abs(a - b).max()


# Column layout of `model_search.observation_features` (verified by length).
GEO_BASIC = slice(0, 7)          # vol, vol2, maxd, area, h1, cx, cy
GEO_CXCY = [5, 6]
GEO_STATS = slice(7, 27)          # 9 depth percentiles, 3 globals, 4 moments, 4 borders
GEO_MAP = slice(27, 75)           # 8x6 depth map
IMG_BLOCK = 342                   # per array: 144 map + 144 |map| + 54 percentiles


def feature_groups(data):
    g, im = data["geometry"], data["image"]
    assert g.shape[1] == 75 and im.shape[1] == 2 * IMG_BLOCK
    img_pct = np.hstack([im[:, o + 288:o + IMG_BLOCK] for o in (0, IMG_BLOCK)])
    img_map = np.hstack([im[:, o:o + 288] for o in (0, IMG_BLOCK)])
    basic_nopos = np.delete(g[:, GEO_BASIC], GEO_CXCY, axis=1)
    return {"basic_nopos": basic_nopos, "cxcy": g[:, GEO_CXCY], "geo_stats": g[:, GEO_STATS],
            "geo_map": g[:, GEO_MAP], "img_pct": img_pct, "img_map": img_map}


def spatial_blocks(rows, k=6):
    from sklearn.cluster import KMeans
    xy = np.array([[r["x"], r["y"]] for r in rows])
    return KMeans(k, random_state=0, n_init=10).fit_predict(xy)


# --------------------------------------------------------------------------
# Final protocol: select on TRAINING rows only, score the 283 once.
# --------------------------------------------------------------------------
SHAPES = ("quad", "star", "triangle", "B", "quad_small")
FEATURE_SETS = {
    "inv133": ("basic_nopos", "geo_stats", "img_pct"),
    "inv135": ("basic_nopos", "geo_stats", "img_pct", "cxcy"),
    "geo25": ("basic_nopos", "geo_stats"),
    "all759": ("basic_nopos", "geo_stats", "img_pct", "geo_map", "img_map", "cxcy"),
}


def stack(data, feature_set):
    g = feature_groups(data)
    return np.hstack([g[k] for k in FEATURE_SETS[feature_set]])


def make_model(d, alpha, gamma, target):
    from sklearn.compose import TransformedTargetRegressor as TTR
    base = make_pipeline(StandardScaler(), KRR(alpha=alpha, gamma=gamma / d))
    if target == "sqrt":
        return TTR(base, func=np.sqrt, inverse_func=np.square, check_inverse=False)
    return base


def load_shapes():
    out = {}
    for s in SHAPES:
        r, d = joblib.load(FULL_CACHE.with_name(f"{s}_v1_di4.joblib"))
        out[s] = (r, d)
    return out


def final(k_blocks=6):
    from sklearn.cluster import KMeans
    from scipy.stats import spearmanr
    rows, data, key, train, test = load()
    y = np.array([r["f"] for r in rows])
    shapes = load_shapes()
    xy = np.array([[r["x"], r["y"]] for r in rows])
    km = KMeans(k_blocks, random_state=0, n_init=10).fit(xy[train])
    blk_round = km.predict(xy)
    blk_shape = {s: km.predict(np.array([[r["x"], r["y"]] for r in shapes[s][0]])) for s in SHAPES}
    ys = {s: np.array([r["f"] for r in shapes[s][0]]) for s in SHAPES}

    grid = [(fs, a, g, t, ms) for fs in FEATURE_SETS for a in (1e-4, 1e-3, 1e-2)
            for g in (0.02, 0.05, 0.1, 0.3, 1.0) for t in ("sqrt", "none") for ms in (False, True)]
    sel = []
    Xr = {fs: stack(data, fs) for fs in FEATURE_SETS}
    Xs = {fs: {s: stack(shapes[s][1], fs) for s in SHAPES} for fs in FEATURE_SETS}
    tri = np.flatnonzero(train)
    for i, (fs, a, g, t, ms) in enumerate(grid):
        d = Xr[fs].shape[1]
        pred = np.empty(len(tri))
        for b in range(k_blocks):
            va = blk_round[tri] == b
            X = [Xr[fs][tri[~va]]]
            Y = [y[tri[~va]]]
            if ms:
                for s in SHAPES:
                    keep = blk_shape[s] != b
                    X.append(Xs[fs][s][keep])
                    Y.append(ys[s][keep])
            m = make_model(d, a, g, t).fit(np.vstack(X), np.concatenate(Y))
            pred[va] = np.clip(m.predict(Xr[fs][tri[va]]), 0, RC.F_MAX_N)
        sel.append({"features": fs, "alpha": a, "gamma": g, "target": t, "multishape": ms,
                    "train_block_cv_mae_n": float(np.abs(pred - y[tri]).mean())})
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(grid)} best {min(sel, key=lambda r: r['train_block_cv_mae_n'])}", flush=True)
    sel.sort(key=lambda r: r["train_block_cv_mae_n"])
    best = sel[0]
    fs = best["features"]
    d = Xr[fs].shape[1]

    def fit_on(extra_rows_mask=None, include_shapes=best["multishape"], exclude=None):
        X, Y = [Xr[fs][train]], [y[train]]
        if include_shapes:
            for s in SHAPES:
                if s != exclude:
                    X.append(Xs[fs][s])
                    Y.append(ys[s])
        return make_model(d, best["alpha"], best["gamma"], best["target"]).fit(np.vstack(X), np.concatenate(Y))

    model = fit_on()
    p = np.clip(model.predict(Xr[fs][test]), 0, RC.F_MAX_N)
    v8 = v8_predictions(rows, test)
    yt = y[test]

    def summary(q):
        return {"mae_n": float(np.abs(q - yt).mean()), "rmse_n": float(np.sqrt(np.mean((q - yt) ** 2))),
                "rho": float(spearmanr(q, yt).statistic),
                "mae_position_bootstrap_ci95_n": position_bootstrap(yt, q, key[test]),
                "bands": bands(yt, q)}

    # paired bootstrap over held-out positions for the improvement
    e_new, e_v8 = np.abs(p - yt), np.abs(v8 - yt)
    u = np.unique(key[test])
    idx = [np.flatnonzero(key[test] == k) for k in u]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(4000):
        pick = np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])
        diffs.append(e_v8[pick].mean() - e_new[pick].mean())
    # the 57 presses v8 could not reach: >8 N but score below the join
    _m, held = RC.fit(report=False, holdout=True)
    join = _m.tail_info["join_score"]
    lookup = dict(zip(zip(np.round(held["f"], 4), np.round(held["pred"], 4)), held["score"]))
    score = np.array([lookup[(round(f, 4), round(q, 4))] for f, q in zip(yt, v8)])
    stuck = (yt > 8) & (score <= join)

    # leave-one-shape-out with the chosen config (all round rows + other shapes)
    loso = {}
    for h in SHAPES:
        X = [Xr[fs]] + [Xs[fs][s] for s in SHAPES if s != h]
        Y = [y] + [ys[s] for s in SHAPES if s != h]
        m = make_model(d, best["alpha"], best["gamma"], best["target"]).fit(np.vstack(X), np.concatenate(Y))
        q = np.clip(m.predict(Xs[fs][h]), 0, RC.F_MAX_N)
        v = np.array([v8_from_basic(b) for b in shapes[h][1]["basic"]])
        loso[h] = {"n": int(len(q)), "new_mae_n": float(np.abs(q - ys[h]).mean()),
                   "v8_mae_n": float(np.abs(v - ys[h]).mean())}

    report = {
        "scope": "GelSight Mini CNC presses (cnc_mini_26), 0-15 N. NOT validated on React sensors.",
        "selection": f"{k_blocks}-block spatial CV on the {int(train.sum())} v8 training rows only; "
                     "held-out rows never used for selection",
        "chosen": best, "selection_table": sel,
        "heldout_283": {"new": summary(p), "v8": summary(v8),
                        "improvement_mae_n": float(e_v8.mean() - e_new.mean()),
                        "improvement_ci95_n": np.percentile(diffs, [2.5, 97.5]).tolist(),
                        "v8_unreachable_group": {"n": int(stuck.sum()),
                                                 "v8_mae_n": float(e_v8[stuck].mean()),
                                                 "new_mae_n": float(e_new[stuck].mean())}},
        "leave_one_shape_out": loso,
        "test_predictions": [{"x": rows[i]["x"], "y": rows[i]["y"], "z": rows[i]["z"], "f": rows[i]["f"],
                              "new_n": float(a), "v8_n": float(b)}
                             for i, a, b in zip(np.flatnonzero(test), p, v8)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_report.json").write_text(json.dumps(report, indent=1))
    # Evaluated model (train split only) and a candidate refit on every labelled row.
    joblib.dump({"model": model, "features": fs, "config": best, "trained_on": "v8 train split (+shapes)"},
                OUT / "heldout_model.joblib")
    Xall = [Xr[fs]] + ([Xs[fs][s] for s in SHAPES] if best["multishape"] else [])
    Yall = [y] + ([ys[s] for s in SHAPES] if best["multishape"] else [])
    joblib.dump({"model": make_model(d, best["alpha"], best["gamma"], best["target"]).fit(np.vstack(Xall), np.concatenate(Yall)),
                 "features": fs, "config": best, "trained_on": "all round + shapes"}, OUT / "candidate.joblib")
    print(json.dumps({k: v for k, v in report.items() if k not in ("selection_table", "test_predictions")}, indent=1))
    return report
