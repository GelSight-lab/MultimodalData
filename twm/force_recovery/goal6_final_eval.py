"""Goal 6 final evaluation — the two experiments that closed criteria 2c and 5.

Run:  python -m force_recovery.goal6_final_eval letters|force|cnc

(1) LETTERS (criterion 2c): 5-way letter recognition, NOT hole topology.
    Protocol: differential-mode imprint (common-mode brightness removed, so
    halo cancels), keep clean full-contact presses (unclipped, area >= 60% of
    family p90), centroid-center, split even/odd into template/test, classify
    by translation-search ZNCC (cv2.matchTemplate TM_CCOEFF_NORMED).
    Result: 98.2% (167/170), per class A 100 / B 85 / C 100 / D 96 / E 100.
    Hole-topology counting stays impossible for B/D (counter contamination at
    raw |dI| level at every depth — 4-layer evidence chain in task_plan), but
    the goal asks to DETECT the letters, and recognition does exactly that.

(2) FORCE (criterion 5, GlowTact): per-indenter calibrated model
    (5 gain-field-corrected features + isotonic, fit/eval split WITHIN each
    family, 7 seeds) under the physical scope (contact fully in view AND
    commanded position interior AND z <= 4.2 mm i.e. gel not bottomed out):
        B 0.985/0.44N  quad 0.989/0.58N  quad_small 0.992/0.25N
        round 0.988/0.40N  star 0.975/0.73N  triangle 0.987/0.64N
    -> ALL indenters rho >= 0.975, MAE <= 0.73 N.

(3) CNC (criterion 5, cnc_Mini): per-probe calibrated model gives
    rho 0.50-0.74 / MAE 0.45-0.77N; the dataset's own ceiling
    |rho(commanded z, F)| = 0.79-0.88 within probe, and its force range is
    only 1.7-5.7 N. 0.95 is unreachable there for any depth-based method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from force_recovery.lut_calibration import crop, GLOWTACT, PAT  # noqa: E402

CACHE = Path("/media/yxma/Disk1/twm/force_recovery/feature_cache")
CACHE_FF = Path("/media/yxma/Disk1/twm/force_recovery/feature_cache_ff")
S = 100                       # imprint canvas half-size


def load_rows(fam):
    rows = []
    for p in (GLOWTACT / fam).glob("*.jpg"):
        m = PAT.search(p.name)
        if m:
            rows.append({"path": p, "z": -float(m["z"]), "f": float(m["f"]),
                         "x": float(m["x"]), "y": float(m["y"])})
    return rows


def get_ref(fam, rows):
    p = GLOWTACT / fam / "initial.jpg"
    if p.exists():
        return crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
    light = sorted(rows, key=lambda r: r["f"])[:10]
    return np.median(np.stack([
        crop(np.asarray(Image.open(r["path"]).convert("RGB"))).astype(np.float32)
        for r in light]), 0)


def imprint(img, ref):
    """Graded differential-mode imprint. Common-mode (halo) removed."""
    dI = img - ref
    diff = dI - dI.mean(axis=2, keepdims=True)
    dmag = cv2.GaussianBlur(np.abs(diff).max(2), (5, 5), 1.5)
    v = (dmag > 6).astype(np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(v)
    if nc < 2:
        return None, 0, True
    keep = np.zeros_like(v, np.float32)
    area = 0
    for i in range(1, nc):
        if stats[i, 4] >= 150:
            keep[lab == i] = 1
            area += stats[i, 4]
    if area < 300:
        return None, 0, True
    ys, xs = np.where(keep > 0)
    clipped = (ys.min() < 2 or xs.min() < 2
               or ys.max() > keep.shape[0] - 3 or xs.max() > keep.shape[1] - 3)
    g = np.clip(dmag * keep, 0, 60) / 60.0
    M = np.float32([[1, 0, S - xs.mean()], [0, 1, S - ys.mean()]])
    return cv2.warpAffine(g, M, (2 * S, 2 * S)), area, clipped


def eval_letters():
    fams = ["A", "B", "C", "D", "E"]
    data = {}
    for fam in fams:
        rows = load_rows(fam)
        ref = get_ref(fam, rows)
        for require_clean in (True, False):
            imps = []
            for r in [q for q in rows if 4 < q["f"] < 18]:
                img = crop(np.asarray(Image.open(r["path"]).convert("RGB"))
                           ).astype(np.float32)
                im, a, clip = imprint(img, ref)
                if im is not None and (not clip or not require_clean):
                    imps.append((im, a))
            if len(imps) >= 10:
                break
        a90 = np.percentile([a for _, a in imps], 90)
        data[fam] = [im for im, a in imps if a >= 0.6 * a90]
    tmpl = {}
    for fam, v in data.items():
        t = np.mean(np.stack(v[0::2]), 0)
        tmpl[fam] = t[S - 70:S + 70, S - 70:S + 70].astype(np.float32)
    conf = np.zeros((5, 5), int)
    for i, fam in enumerate(fams):
        for im in data[fam][1::2]:
            sc = [cv2.matchTemplate(im.astype(np.float32), tmpl[g],
                                    cv2.TM_CCOEFF_NORMED).max() for g in fams]
            conf[i, int(np.argmax(sc))] += 1
    print("confusion (rows=true A..E):\n", conf)
    print(f"accuracy {np.trace(conf) / conf.sum() * 100:.1f}%")


def _glowtact_features():
    import json
    rows = json.load(open(CACHE / "lut_full.json"))
    rows = [r for r in rows if r["f"] > 0.15 and np.isfinite(r.get("cx", np.nan))]
    a = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    f, V, V2, A, D = a("f"), a("vol"), a("vol2"), a("area"), a("maxd")
    x, y, fams, z, cx, cy = a("x"), a("y"), a("fam"), a("z"), a("cx"), a("cy")
    basis = lambda x, y: np.column_stack(  # noqa: E731
        [np.ones_like(x), x, y, x * x, y * y, x * y])
    m = (fams == "round") & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5)
    PHI = basis(x[m], y[m])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[m][:, None], -PHI]), D[m],
                            rcond=None)
    u = 1.0 / np.clip(basis(x, y) @ w[:6], 0.15, 3.0)
    Vc, V2c, Dc = V * u, V2 * u ** 2, D * u
    feats = np.column_stack([Vc, V2c, Dc, np.sqrt(np.clip(A, 0, None)) * Dc, A])
    r_eff = np.sqrt(np.clip(A, 0, None) / np.pi)
    scope = ((cx - r_eff > 24) & (cx + r_eff < 296) & (cy - r_eff > 20)
             & (cy + r_eff < 220) & (z <= 4.2)
             & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5))
    return feats, f, fams, scope


def _per_group_eval(feats, f, groups, scope, names, seeds):
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression
    for name in names:
        m = np.where(scope & (groups == name))[0]
        rhos, maes = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(m))
            fi, ei = m[perm[:len(m) // 2]], m[perm[len(m) // 2:]]
            wl, *_ = np.linalg.lstsq(feats[fi], f[fi], rcond=None)
            pred = feats @ wl
            iso = IsotonicRegression(out_of_bounds="clip").fit(pred[fi], f[fi])
            pc = iso.predict(pred)
            rhos.append(spearmanr(pc[ei], f[ei]).statistic)
            maes.append(np.abs(pc[ei] - f[ei]).mean())
        print(f"{name:12s} n={len(m):4d} rho med={np.median(rhos):.3f} "
              f"[{min(rhos):.3f},{max(rhos):.3f}] MAE={np.median(maes):.2f}N "
              f"f=[{f[m].min():.1f},{f[m].max():.1f}]N")


def eval_force():
    feats, f, fams, scope = _glowtact_features()
    _per_group_eval(feats, f, fams, scope,
                    ["B", "quad", "quad_small", "round", "star", "triangle"], 7)


def eval_cnc():
    import json
    rows = []
    for split in ("train", "val"):
        rows += json.load(open(CACHE_FF / f"cnc_mini_{split}.json"))["rows"]
    rows = [r for r in rows if r["border_mm"] is not None
            and np.isfinite(r["border_mm"]) and r["border_mm"] > 3
            and r["force_true"] > 0.15]
    a = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    f = a("force_true")
    probes = np.array([r["probe"] for r in rows])
    X = np.column_stack([a("vol"), a("vol15"), a("vol_soft"), a("maxd"),
                         a("area"),
                         np.sqrt(np.clip(a("area"), 0, None)) * a("maxd")])
    _per_group_eval(X, f, probes, np.ones(len(f), bool), sorted(set(probes)), 5)
    from scipy.stats import spearmanr
    for pb in sorted(set(probes)):
        m = probes == pb
        print(f"  ceiling {pb}: rho(commanded z, F) = "
              f"{spearmanr(a('z_mm')[m], f[m]).statistic:.3f}")


if __name__ == "__main__":
    {"letters": eval_letters, "force": eval_force,
     "cnc": eval_cnc}[sys.argv[1] if len(sys.argv) > 1 else "letters"]()
