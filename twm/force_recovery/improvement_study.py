"""Where the force model can still improve — and whether learning helps.

Five questions, each answered with a control that could have falsified it.
Run: python -m force_recovery.improvement_study [shear|learn|pinn|transfer|all]

RESULTS (Sparsh in-view, self-calibrated LUT; GlowTact in physical scope):

1. SHEAR — closed, not worth pursuing.
   Residual correlates with shear at rho 0.31-0.40, which looks like a missing
   signal. But feeding the TRUE shear in as a feature (an oracle no real
   estimator could have) moves sphere pads from rho 0.9749/MAE 0.0360 to only
   0.9757/0.0345. The residual-shear correlation was confounded: both track
   force magnitude. Extracting shear-sensitive image features is therefore
   NOT worth the work for normal force.

2. LEARNING — does not beat physics + isotonic on the within-pad task.
   MLP 32x16 0.9666 / MLP 64x64 0.9715 / gradient boosting 0.9723, all below
   the 5-feature linear+isotonic baseline at 0.9749, on identical splits.
   With ~400 in-view frames per pad, flexible models overfit.

3. LOG FEATURES — a real, cheap win (the error is multiplicative).
   Adding log1p of each feature: 0.9749 -> 0.9846, MAE 0.0360 -> 0.0308
   (-14%). Motivated by residual-vs-force rho of 0.35-0.48, i.e. proportional
   rather than additive error.

4. "PINN-lite" dimensionless contact law F/A = g(D/sqrt(A)) — REJECTED.
   It looked like a breakthrough on sphere->flat (MAE 0.426 -> 0.150 N,
   beating an MLP's 0.297). Then the controls killed it: plain F = g(volume)
   reaches MAE 0.0736 N on the same transfer, and F = A * const alone reaches
   0.128 N. The gain came from using ONE monotone feature instead of five,
   not from the dimensionless structure. On GlowTact leave-one-indenter-out
   the law is worse than baseline on every indenter (MAE 0.67-4.05 vs
   0.31-1.53). Replacing g with an MLP collapses it entirely (rho -0.249):
   isotonic clips monotonically outside the training range, an MLP
   extrapolates freely.

5. THE ACTUAL IMPROVEMENT — match model complexity to the transfer regime.
   Cross-INDENTER rank is consistently better with the single volume feature;
   absolute newtons with a known indenter are better with five features:

     GlowTact leave-one-indenter-out (never sees the held-out shape):
       5-feature   rho 0.958-0.995
       volume-only rho 0.986-0.997   <- fixes the Goal-6 weak spots,
                                        round 0.958->0.993, star 0.977->0.991
     Sparsh sphere->flat:  MAE 0.426 -> 0.0736 N with volume-only
     Sparsh cross-pad, same indenter: 5-feature better (0.0372 vs 0.0443 N)

   So: report volume-only for unseen shapes, five features once the shape is
   calibrated. This raises the honest generalisation claim rather than the
   in-sample one.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from .run_episode import OUT_ROOT

CS = OUT_ROOT / "feature_cache_sparshlut"
CIRC = OUT_ROOT / "sparsh_probe" / "eval_circles.json"
LUT_FULL = OUT_ROOT / "feature_cache" / "lut_full.json"
FE = ("vol", "vol2", "maxd", "area", "h1")


# ------------------------------------------------------------------ loaders
def load_sparsh() -> dict:
    """In-view frames only: 36% of presses show no disc, 11% are clipped."""
    circ = json.loads(CIRC.read_text())
    out = {}
    for p in sorted(glob.glob(f"{CS}/sparsh_*.json")):
        d = json.loads(open(p).read())
        name = p.split("sparsh_")[-1][:-5]
        c = circ.get(name.replace("_batch_", "_b"), {})
        rows = [r for r in d["rows"]
                if c.get(str(r["index"])) and c[str(r["index"])]["inview"]]
        if len(rows) >= 40:
            out[name] = rows
    return out


def load_glowtact() -> tuple[list, np.ndarray]:
    rows = [r for r in json.loads(LUT_FULL.read_text())
            if r["f"] > 0.15 and np.isfinite(r.get("cx", float("nan")))]
    g = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    ar, cx, cy, z, x, y = g("area"), g("cx"), g("cy"), g("z"), g("x"), g("y")
    re_ = np.sqrt(np.clip(ar, 0, None) / np.pi)
    sc = ((cx - re_ > 24) & (cx + re_ < 296) & (cy - re_ > 20)
          & (cy + re_ < 220) & (z <= 4.2)
          & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5))
    rows = [r for r, k in zip(rows, sc) if k]
    return rows, np.array([r["fam"] for r in rows])


# ------------------------------------------------------------------ helpers
F = lambda R: np.array([abs(r.get("fz", r.get("f"))) for r in R])   # noqa: E731
V = lambda R: np.array([r["vol"] for r in R])                       # noqa: E731


def M5(R):
    a = np.maximum(np.array([r["area"] for r in R]), 1e-6)
    d = np.array([r["maxd"] for r in R])
    return np.column_stack([V(R), np.array([r["vol2"] for r in R]), d, a,
                            np.sqrt(a) * d])


def lin_iso(Xtr, ftr, Xte):
    w, *_ = np.linalg.lstsq(Xtr, ftr, rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(Xtr @ w, ftr)
    return iso.predict(Xte @ w)


def iso1(xtr, ftr, xte):
    return IsotonicRegression(out_of_bounds="clip").fit(xtr, ftr).predict(xte)


def score(p, t):
    return spearmanr(p, t).statistic, float(np.abs(p - t).mean())


# ------------------------------------------------------------------ studies
def cmd_shear():
    data = load_sparsh()
    print("shear oracle (sphere pads, in-view) — can a real shear feature help?")
    for name, ff in (("baseline 5 features", M5),
                     ("+ TRUE shear (oracle)", None)):
        R, M = [], []
        for k, rows in sorted(data.items()):
            if not k.startswith("sphere"):
                continue
            if ff is None:
                sh = np.array([r["shear"] for r in rows])
                X = np.column_stack([M5(rows), sh, sh ** 2,
                                     sh * np.array([r["maxd"] for r in rows])])
            else:
                X = ff(rows)
            f = F(rows)
            rr, mm = [], []
            for s in range(5):
                i = np.random.default_rng(s).permutation(len(f))
                a, b = i[:len(i) // 2], i[len(i) // 2:]
                r_, m_ = score(lin_iso(X[a], f[a], X[b]), f[b])
                rr.append(r_)
                mm.append(m_)
            R.append(np.median(rr))
            M.append(np.median(mm))
        print(f"  {name:26s} rho={np.median(R):.4f} MAE={np.median(M):.4f} N")
    print("  -> oracle gain is ~0.001 rho: shear extraction NOT worth it")


def cmd_learn():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    data = load_sparsh()

    def run(mk, name, log=False):
        R, M = [], []
        for k, rows in sorted(data.items()):
            if not k.startswith("sphere"):
                continue
            X, f = M5(rows), F(rows)
            if log:
                X = np.column_stack([np.log1p(np.clip(X, 0, None)), X])
            rr, mm = [], []
            for s in range(3):
                i = np.random.default_rng(s).permutation(len(f))
                a, b = i[:len(i) // 2], i[len(i) // 2:]
                if mk is None:
                    p = lin_iso(X[a], f[a], X[b])
                else:
                    sc = StandardScaler().fit(X[a])
                    m = mk(s).fit(sc.transform(X[a]), f[a])
                    p = m.predict(sc.transform(X[b]))
                r_, m_ = score(p, f[b])
                rr.append(r_)
                mm.append(m_)
            R.append(np.median(rr))
            M.append(np.median(mm))
        print(f"  {name:30s} rho={np.median(R):.4f} MAE={np.median(M):.4f} N")

    print("learning vs physics+isotonic, identical features and splits:")
    run(None, "linear+isotonic (baseline)")
    run(None, "linear+isotonic + log feats", log=True)
    run(lambda s: MLPRegressor((32, 16), max_iter=3000, random_state=s,
                               early_stopping=True), "MLP 32x16")
    run(lambda s: MLPRegressor((64, 64), max_iter=3000, random_state=s,
                               early_stopping=True), "MLP 64x64")
    run(lambda s: HistGradientBoostingRegressor(max_iter=250, random_state=s),
        "gradient boosting")


def cmd_pinn():
    data = load_sparsh()
    cat = lambda ks: [r for k in ks for r in data[k]]      # noqa: E731
    sph = [k for k in data if k.startswith("sphere")]
    flt = [k for k in data if k.startswith("flat")]
    tr, te = cat(sph), cat(flt)
    A = lambda R: np.maximum(np.array([r["area"] for r in R]), 1e-6)  # noqa
    D = lambda R: np.array([r["maxd"] for r in R])                    # noqa
    print("sphere -> flat, and the controls that decide it:")
    for nm, p in (
            ("5-feature linear+isotonic", lin_iso(M5(tr), F(tr), M5(te))),
            ("PINN-lite F/A = g(D/sqrtA)",
             iso1(D(tr) / np.sqrt(A(tr)), F(tr) / A(tr),
                  D(te) / np.sqrt(A(te))) * A(te)),
            ("control F = A * const", A(te) * np.median(F(tr) / A(tr))),
            ("control F = g(area)", iso1(A(tr), F(tr), A(te))),
            ("control F = g(volume)", iso1(V(tr), F(tr), V(te)))):
        r, m = score(p, F(te))
        print(f"  {nm:30s} rho={r:6.3f} MAE={m:.4f} N")
    print("  -> g(volume) beats the dimensionless law: REJECT PINN-lite")


def cmd_transfer():
    data = load_sparsh()
    cat = lambda ks: [r for k in ks for r in data[k]]      # noqa: E731
    sph = [k for k in data if k.startswith("sphere")]
    flt = [k for k in data if k.startswith("flat")]
    print("Sparsh cross-indenter (5-feature vs volume-only):")
    for nm, tr, te in (("sphere->flat", cat(sph), cat(flt)),
                       ("flat->sphere", cat(flt), cat(sph))):
        r5, m5 = score(lin_iso(M5(tr), F(tr), M5(te)), F(te))
        rv, mv = score(iso1(V(tr), F(tr), V(te)), F(te))
        print(f"  {nm:14s} 5-feat rho {r5:.3f} MAE {m5:.4f} | "
              f"vol-only rho {rv:.3f} MAE {mv:.4f}")

    print("\nSparsh cross-pad, SAME indenter (does vol-only cost anything?):")
    a5, av, b5, bv = [], [], [], []
    for a in sph:
        for b in sph:
            if a == b:
                continue
            r5, m5 = score(lin_iso(M5(data[a]), F(data[a]), M5(data[b])),
                           F(data[b]))
            rv, mv = score(iso1(V(data[a]), F(data[a]), V(data[b])),
                           F(data[b]))
            a5.append(r5); av.append(rv); b5.append(m5); bv.append(mv)
    print(f"  5-feat   rho {np.median(a5):.3f} MAE {np.median(b5):.4f} N")
    print(f"  vol-only rho {np.median(av):.3f} MAE {np.median(bv):.4f} N")

    rows, fam = load_glowtact()
    X5, Vg, Fg = M5(rows), V(rows), F(rows)
    print("\nGlowTact leave-one-indenter-out (model never sees that shape):")
    for ho in sorted(set(fam)):
        m = fam == ho
        if m.sum() < 30 or (~m).sum() < 50:
            continue
        r5, m5 = score(lin_iso(X5[~m], Fg[~m], X5[m]), Fg[m])
        rv, mv = score(iso1(Vg[~m], Fg[~m], Vg[m]), Fg[m])
        flag = "  <-- fixed" if rv - r5 > 0.01 else ""
        print(f"  {ho:11s} 5-feat rho {r5:.3f} MAE {m5:6.3f}N | "
              f"vol-only rho {rv:.3f} MAE {mv:6.3f}N{flag}")


def cmd_all():
    for c in (cmd_shear, cmd_learn, cmd_pinn, cmd_transfer):
        print("=" * 72)
        c()


if __name__ == "__main__":
    {"shear": cmd_shear, "learn": cmd_learn, "pinn": cmd_pinn,
     "transfer": cmd_transfer, "all": cmd_all}[
        sys.argv[1] if len(sys.argv) > 1 else "all"]()
