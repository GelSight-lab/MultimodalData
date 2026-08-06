"""Re-run force estimation on every dataset, each beside its shuffle control.

One protocol everywhere, so the four numbers are comparable: within each
group (indenter family / probe / capture group / gel pad) the frames are
split half fit, half eval; a 5-feature least-squares fit is calibrated by
isotonic regression on the fit half; predictions are pooled across groups and
scored by Spearman rho and MAE. 5 seeds, median reported with the spread.

Beside every row, the SAME protocol with the force labels permuted WITHIN
each group. A within-group shuffle is the control that matters here: it keeps
the group structure and the marginal force distribution and destroys only the
frame-to-force pairing, so it measures what the protocol scores when the
features carry nothing. (A global shuffle would be too easy to beat — on
FeelAnyForce the pooled rho survived global shuffling at 0.442 vs 0.455,
which is how we caught that its frame join had never been demonstrated.)

Why this run exists: the marker-inpainting step was integrated into the
DEPTH / 3D path (`marker_removal.stages_depth`). The force path
(`debug_gallery.stages`) was deliberately left untouched, because marker
removal does not help force — so every number here must be unchanged, and
`spotcheck` proves the reconstruction itself is bit-identical by recomputing
cached features from the raw frames.

Run:
  python -m force_recovery.force_eval_all            # table + force_matrix.json
  python -m force_recovery.force_eval_all spotcheck  # regression check only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
CACHE_SP = OUT_ROOT / "feature_cache_sparshlut"
DG = OUT_ROOT / "site_assets" / "debug_gallery"
SEEDS = 5
OUT_JSON = CACHE / "force_matrix.json"


# ------------------------------------------------------------------ protocol
def _fit_apply(Xf, ff, Xe):
    w, *_ = np.linalg.lstsq(Xf, ff, rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(Xf @ w, ff)
    return iso.predict(Xe @ w)


def _one_seed(X, f, groups, seed, shuffle=False):
    """Per-group half/half + isotonic -> pooled (truth, pred) for the eval half."""
    pred = np.full(len(f), np.nan)
    fuse = f.copy()
    for gi, g in enumerate(sorted(set(groups))):
        idx = np.where(groups == g)[0]
        if shuffle:
            # permute labels WITHIN the group, so the group's force
            # distribution is untouched and only the pairing dies
            fuse[idx] = f[idx][np.random.default_rng(9000 + 13 * gi + seed)
                               .permutation(len(idx))]
        # deterministic across processes (str hash is salted in python3)
        idx = idx[np.random.default_rng(1000 * seed + 7 * gi)
                  .permutation(len(idx))]
        h = len(idx) // 2
        fi, ei = idx[:h], idx[h:]
        if len(fi) < 8:
            continue
        pred[ei] = _fit_apply(X[fi], fuse[fi], X[ei])
    ok = np.isfinite(pred)
    return fuse[ok], pred[ok], groups[ok]


def evaluate(X, f, groups, seeds=SEEDS) -> dict:
    """Pooled rho/MAE (median of `seeds`), per-group rho range, shuffle control."""
    X = np.asarray(X, float)
    f = np.asarray(f, float)
    groups = np.asarray(groups)
    rr, mm, sh, per = [], [], [], {g: [] for g in sorted(set(groups))}
    n_eval = 0
    for s in range(seeds):
        t, p, g = _one_seed(X, f, groups, s)
        rr.append(float(spearmanr(p, t).statistic))
        mm.append(float(np.abs(p - t).mean()))
        n_eval = len(t)
        for gg in per:
            m = g == gg
            if m.sum() >= 8:
                per[gg].append(float(spearmanr(p[m], t[m]).statistic))
        ts, ps, _ = _one_seed(X, f, groups, s, shuffle=True)
        sh.append(float(spearmanr(ps, ts).statistic))
    pg = {k: float(np.median(v)) for k, v in per.items() if v}
    return {"rho": float(np.median(rr)), "mae": float(np.median(mm)),
            "rho_min": float(np.min(rr)), "rho_max": float(np.max(rr)),
            "rho_sd": float(np.std(rr)), "shuffle_rho": float(np.median(sh)),
            "n_eval": int(n_eval), "n_groups": len(set(groups)),
            "per_group_rho": pg}


def _basis(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


# ------------------------------------------------------------------ datasets
def ds_glowtact() -> dict:
    """GlowTact: 6 indenter families, 0-20 N, sphere-supervised gain field.

    Scope is physical, not cherry-picked: contact fully inside the frame and
    the gel not bottomed out (z <= 4.2 mm).
    """
    rows = json.loads((CACHE / "lut_full.json").read_text())
    rows = [r for r in rows if r["f"] > 0.15
            and np.isfinite(r.get("cx", np.nan))]
    a = lambda k: np.array([r[k] for r in rows])            # noqa: E731
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    V, V2, A, D, cx, cy = (a("vol"), a("vol2"), a("area"), a("maxd"),
                           a("cx"), a("cy"))
    grp = np.array([r["fam"] for r in rows])
    m = (grp == "round") & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5)
    PHI = _basis(x[m], y[m])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[m][:, None], -PHI]), D[m],
                            rcond=None)
    u = 1.0 / np.clip(_basis(x, y) @ w[:6], 0.15, 3.0)
    X = np.column_stack([V * u, V2 * u ** 2, D * u,
                         np.sqrt(np.clip(A, 0, None)) * D * u, A])
    r_eff = np.sqrt(np.clip(A, 0, None) / np.pi)
    sc = ((cx - r_eff > 24) & (cx + r_eff < 296) & (cy - r_eff > 20)
          & (cy + r_eff < 220) & (z <= 4.2)
          & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5))
    return evaluate(X[sc], f[sc], grp[sc])


def ds_cnc() -> dict:
    """FoTa cnc_Mini, strictly in view (the press grid is bigger than the FOV)."""
    rows = json.loads((DG / "features_cnc_full.json").read_text())
    a = lambda k: np.array([r[k] for r in rows])            # noqa: E731
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    grp = np.array([r["group"] for r in rows])
    inner = (x > 5) & (x < 13) & (y > 4) & (y < 12)
    PHI = _basis(x[inner], y[inner])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[inner][:, None], -PHI]),
                            a("maxd")[inner], rcond=None)
    u = 1.0 / np.clip(_basis(x, y) @ w[:6], 0.15, 3.0)
    X = np.column_stack([a("vol") * u, a("vol2") * u ** 2, a("maxd") * u,
                         a("area"),
                         np.sqrt(np.clip(a("area"), 0, None)) * a("maxd") * u])
    return evaluate(X[inner], f[inner], grp[inner])


FE = ("vol", "vol2", "maxd", "area", "h1")


def ds_feats(inpaint: bool = False) -> dict:
    """FEATS, recomputed from the images — the actual regression check.

    `inpaint=True` runs the same frames through the DEPTH-path marker
    removal and fits force on those features. It is reported so the site can
    show the cost of the step it did NOT adopt for force, on identical
    frames and identical splits.

    `debug_gallery.RNG` is module-level and every loader draws from it, so
    the frame sample depends on which datasets ran earlier in the process.
    Reseeding here makes this evaluation order-independent, makes the two
    passes here a genuinely paired comparison, and reproduces the baseline
    the marker study published (rho 0.7747 / MAE 5.03 N).
    """
    from . import debug_gallery as dg
    from .debug_gallery import load_feats, stages
    from .marker_removal import marker_mask, stages_depth

    dg.RNG = np.random.default_rng(0)
    frames, get = load_feats()
    _, ref = get(frames[0])
    mask = marker_mask(ref)
    rows = []
    for j, fr in enumerate(frames):
        img, r = get(fr)
        st = (stages_depth(img, r, mask=mask) if inpaint else stages(img, r))
        rows.append({"group": fr["group"], "f": fr["f"], **st["feats"]})
        if (j + 1) % 130 == 0:
            print(f"    feats{' (inpainted)' if inpaint else ''}: "
                  f"{j+1}/{len(frames)}", flush=True)
    # published so results_page scores the identical frames, not a second
    # random sample of the same parquet
    if not inpaint:
        (CACHE / "feats_rows.json").write_text(json.dumps(rows))
    X = np.array([[r[k] for k in FE] for r in rows])
    f = np.array([r["f"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    return evaluate(X, f, grp)


def ds_sparsh() -> dict:
    """Sparsh (Meta), Sparsh-native LUT, in-view frames, per-gel-pad."""
    from .sparsh_figure import _inview_mask, _load, _xy

    data = _load("sparsh")
    Xs, fs, gs = [], [], []
    for name, rows in data.items():
        m = _inview_mask(name, rows)
        if m.sum() < 40:
            continue
        X, f = _xy([r for r, k in zip(rows, m) if k])
        Xs.append(X)
        fs.append(f)
        gs += [name] * len(f)
    return evaluate(np.vstack(Xs), np.concatenate(fs), np.array(gs))


# ------------------------------------------------------------------ regression
def spotcheck(n: int = 40) -> dict:
    """Recompute cached cnc features from the raw frames — bit-for-bit.

    `debug_gallery.stages` is the single force path for every dataset, so if
    it reproduces a cached feature exactly on real frames it is unchanged
    everywhere. This is the check that the marker work did not leak into
    force.
    """
    import os

    os.environ.setdefault("CNC_N", "390")
    from .debug_gallery import load_cnc, stages

    cached = {}
    for r in json.loads((DG / "features_cnc_full.json").read_text()):
        cached[(r["group"], round(r["f"], 4), round(r["x"], 4),
                round(r["y"], 4), round(r["z"], 4))] = r
    frames, get = load_cnc()
    worst, hits = 0.0, 0
    for fr in frames:
        key = (fr["group"], round(fr["f"], 4), round(fr["x"], 4),
               round(fr["y"], 4), round(fr["z"], 4))
        if key not in cached:
            continue
        img, ref = get(fr)
        got = stages(img, ref)["feats"]
        for k in FE:
            worst = max(worst, abs(got[k] - cached[key][k]))
        hits += 1
        if hits >= n:
            break
    print(f"spotcheck: {hits} cnc frames recomputed from raw, "
          f"max |cached - fresh| over {FE} = {worst:.3e}")
    assert hits >= 10, "no cached frames matched — cache/loader drift"
    assert worst < 1e-9, f"stages() changed: max feature delta {worst}"
    return {"n_frames": hits, "max_abs_delta": float(worst)}


# ------------------------------------------------------------------ report
ROWS = [
    ("GlowTact (markerless, 0-20 N)", ds_glowtact),
    ("FoTa cnc_Mini (markerless, in view)", ds_cnc),
    ("FEATS (marker gel)", ds_feats),
    ("Sparsh / Meta (markerless, Sparsh LUT, in view)", ds_sparsh),
]


def main() -> dict:
    out = {"protocol": "per-group half/half + isotonic, 5 seeds, median; "
                       "control = force labels permuted within each group",
           "spotcheck": spotcheck(),
           "datasets": {}}
    for name, fn in ROWS:
        print(f"== {name}", flush=True)
        out["datasets"][name] = fn()
    print("\n-- marker-inpainted FEATS features (NOT adopted for force)",
          flush=True)
    out["datasets"]["FEATS (marker gel, dots inpainted — rejected for force)"] \
        = ds_feats(inpaint=True)

    print(f"\n{'dataset':52s} {'n':>6s} {'rho':>7s} {'[min,max]':>15s} "
          f"{'MAE [N]':>9s} {'shuffle':>8s}")
    for k, v in out["datasets"].items():
        print(f"{k:52s} {v['n_eval']:6d} {v['rho']:7.4f} "
              f"[{v['rho_min']:.3f},{v['rho_max']:.3f}]".ljust(85)
              + f"{v['mae']:9.3f} {v['shuffle_rho']:8.3f}")
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT_JSON}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "spotcheck":
        spotcheck()
    else:
        main()
