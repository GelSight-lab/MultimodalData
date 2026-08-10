"""Would non-negative weights change React's force channel? Measured, not argued.

The cross-dataset matrix has one row that transfers NEGATIVELY: FeelAnyForce's
least-squares weights rank every other sensor backwards. The cause is not the
reconstruction but the fit — the five features are collinear (condition number
200-2800), so least squares can cancel large opposite-sign terms, and the
cancellation only holds at the feature ratios it was fitted on. Constraining
the weights to be non-negative removes the negative transfer completely.

That raises an obvious question about the DEPLOYED channel, whose weights
cancel the same way (+12.3 N of positive terms against -10.1 N of negative
ones, netting 2.2 N). This module answers it before any reprocessing, on the
two axes that matter:

  rig, held out by press position, many seeds
        Is either fit actually better where the labels are? A single split
        said NNLS won by 0.021 rho. Fifteen seeds say the split noise is
        +-0.035, i.e. that split proved nothing.

  React episode frames, the two weight vectors side by side
        React is only a transfer case if its contacts fall outside the rig's
        fitted range. If they do not, the choice cannot matter, and the
        cheapest way to know is to run both and look at the disagreement.

Writes react_weight_ab.json. The site quotes it; nothing here is typed into
HTML.

    python -m force_recovery.react_weight_ab [--frames 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "react_weight_ab.json"
SEEDS = 15
EPISODE = ("motherboard", "2026-05-10", "episode_000", "left")


def _design():
    """The production design matrix, gain field and all, from react_calib."""
    from . import react_calib as RC

    rows, a = RC._load(RC.FORCE_RECONSTRUCTION)
    f, cx, cy, z = a("f"), a("cx"), a("cy"), a("z")
    X0 = np.column_stack([a(k) for k in RC.FEATURES])
    phi = RC._basis(cx / 100, cy / 100)
    gw, *_ = np.linalg.lstsq(np.hstack([phi * z[:, None], -phi]), a("maxd"),
                             rcond=None)
    gain = gw[:6]
    u = 1.0 / np.clip(phi @ gain, 0.15, 3.0)
    X = RC._with_clip(np.column_stack([X0[:, 0] * u, X0[:, 1] * u ** 2,
                                       X0[:, 2] * u, X0[:, 3], X0[:, 4] * u]),
                      a("area"), cx, cy)
    key = np.round(a("x"), 1) * 1000 + np.round(a("y"), 1)
    return X, f, key, gain


def _models(X, f):
    from scipy.optimize import nnls
    from sklearn.isotonic import IsotonicRegression
    out = {}
    for tag, w in (("ols", np.linalg.lstsq(X, f, rcond=None)[0]),
                   ("nnls", nnls(X, f)[0])):
        out[tag] = (w, IsotonicRegression(out_of_bounds="clip").fit(X @ w, f))
    return out


def rig_seeds(X, f, key) -> dict:
    """Held out by press position — neighbouring frames of one press are
    near-duplicates, so a random split would score its own training data."""
    from scipy.stats import spearmanr
    uniq = np.unique(key)
    r = {"ols": [], "nnls": []}
    m = {"ols": [], "nnls": []}
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        hold = set(uniq[rng.permutation(len(uniq))[:max(len(uniq) // 3, 1)]])
        te = np.array([k in hold for k in key])
        tr = ~te
        if te.sum() < 20 or tr.sum() < 40:
            continue
        for tag, (w, iso) in _models(X[tr], f[tr]).items():
            p = iso.predict(X[te] @ w)
            r[tag].append(float(spearmanr(p, f[te]).statistic))
            m[tag].append(float(np.abs(p - f[te]).mean()))
    d = np.array(r["nnls"]) - np.array(r["ols"])
    return {"seeds": len(d),
            "rho_median": {k: float(np.median(v)) for k, v in r.items()},
            "rho_sd": {k: float(np.std(v)) for k, v in r.items()},
            "mae_median": {k: float(np.median(v)) for k, v in m.items()},
            "paired_diff_median": float(np.median(d)),
            "paired_diff_sd": float(np.std(d)),
            "nnls_better_seeds": int((d > 0).sum())}


def episode_agreement(X, f, gain, frames: int) -> dict:
    """Both weight vectors over real React contacts, through `force_stages`."""
    import h5py
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr

    from . import react_calib as RC
    from .lut_calibration import crop
    from .run_episode import (DATA_ROOT, LEGACY_SHIFT, STAGE_ROOT,
                              _reference_rows)

    mdl = _models(X, f)
    lo = {k: float((X @ w).min()) for k, (w, _) in mdl.items()}
    hi = {k: float((X @ w).max()) for k, (w, _) in mdl.items()}

    def u_at(px, py):
        return 1.0 / np.clip(RC._basis(np.atleast_1d(px / 100),
                                       np.atleast_1d(py / 100)) @ gain,
                             0.15, 3.0)

    task, date, ep, side = EPISODE
    tb = pq.read_table(str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"))
    inten = np.asarray(tb[f"tactile_{side}_intensity"].to_numpy())
    isnew = np.asarray(tb[f"tactile_{side}_is_new"].to_numpy())
    trim = int(np.asarray(tb["source_h5_frame"].to_numpy())[0])
    ref_rows = _reference_rows(inten, isnew)
    sel = np.linspace(0, len(inten) - 1, frames).astype(int)

    pred = {"ols": [], "nnls": []}
    proj, contact = [], []
    with h5py.File(str(DATA_ROOT / task / date / f"{ep}.h5"), "r") as h:
        fr = h[f"gelsight/{side}/frames"]
        n = len(fr)
        ref = np.median(np.stack(
            [crop(fr[min(trim + int(r) + LEGACY_SHIFT, n - 1)]).astype(
                np.float32) for r in ref_rows[:12]]), 0)
        for r in sel:
            st = RC.force_stages(
                crop(fr[min(trim + int(r) + LEGACY_SHIFT, n - 1)]), ref)
            ft, d = st["feats"], st["depth"]
            mm = st.get("contact")
            if mm is None:
                mm = d > 0.05
            if ft["area"] < 1.0 or mm.sum() < 30:
                for k in pred:
                    pred[k].append(0.0)
                contact.append(False)
                continue
            yy, xx = np.nonzero(mm)
            ww = d[mm]
            px = float((xx * ww).sum() / ww.sum())
            py = float((yy * ww).sum() / ww.sum())
            uu = float(u_at(px, py)[0])
            v = RC._with_clip(
                np.array([[ft["vol"] * uu, ft["vol2"] * uu ** 2,
                           ft["maxd"] * uu, ft["area"], ft["h1"] * uu]]),
                np.array([ft["area"]]), np.array([px]), np.array([py]))
            for k, (w, iso) in mdl.items():
                pred[k].append(float(iso.predict(v @ w)[0]))
            proj.append(float((v @ mdl["ols"][0])[0]))
            contact.append(True)
    c = np.array(contact)
    o, nn = np.array(pred["ols"]), np.array(pred["nnls"])
    pr = np.array(proj)
    return {"episode": "/".join(EPISODE), "frames": int(len(c)),
            "contact_frames": int(c.sum()),
            "rho": float(spearmanr(o[c], nn[c]).statistic),
            "median_abs_diff_n": float(np.median(np.abs(o - nn)[c])),
            "max_abs_diff_n": float(np.abs(o - nn)[c].max()),
            "outside_rig_range_frac": float(((pr < lo["ols"]) |
                                             (pr > hi["ols"])).mean())}


def main() -> int:
    from .artifact_lock import one_writer
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    with one_writer(OUT):
        X, f, key, gain = _design()
        mdl = _models(X, f)
        contrib = {k: (w * X.std(0)).tolist() for k, (w, _) in mdl.items()}
        rep = {"n_rig": int(len(f)), "n_columns": int(X.shape[1]),
               "condition_number": float(np.linalg.cond(X)),
               "standardised_contribution": contrib,
               "ols_positive_sum": float(sum(c for c in contrib["ols"]
                                             if c > 0)),
               "ols_negative_sum": float(sum(c for c in contrib["ols"]
                                             if c < 0)),
               "rig": rig_seeds(X, f, key),
               "react": episode_agreement(X, f, gain, args.frames)}
        OUT.write_text(json.dumps(rep, indent=1))
    g = rep["rig"]
    r = rep["react"]
    print(f"  标定台 n={rep['n_rig']}, 条件数 {rep['condition_number']:.0f}; "
          f"OLS 正项 {rep['ols_positive_sum']:+.2f} N / 负项 "
          f"{rep['ols_negative_sum']:+.2f} N")
    print(f"  {g['seeds']} 个种子: OLS rho {g['rho_median']['ols']:.3f} vs "
          f"NNLS {g['rho_median']['nnls']:.3f}; 配对差 "
          f"{g['paired_diff_median']:+.3f} ± {g['paired_diff_sd']:.3f} "
          f"(NNLS 更好 {g['nnls_better_seeds']}/{g['seeds']})")
    print(f"  React {r['contact_frames']}/{r['frames']} 接触帧: 两者 rho "
          f"{r['rho']:.4f}, |差| 中位 {r['median_abs_diff_n']:.3f} N, "
          f"区间外 {r['outside_rig_range_frac']*100:.1f}%")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
