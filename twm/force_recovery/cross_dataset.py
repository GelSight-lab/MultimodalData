"""Fit on one dataset, predict on every other. A transfer matrix.

WHAT IS FITTED

One model per dataset, which is the granularity settled on after measuring
that per-group fitting buys nothing (within 0.02 everywhere, and the single
model was never worse). The model is the production one:

    w   = lstsq(X, f)                       5 features -> newtons, linear
    iso = IsotonicRegression().fit(X @ w, f)     monotone calibration

with X = (vol, vol2, maxd, area, sqrt(area)*maxd) from the calibration-free
reconstruction — the one the force channel now uses.

TWO NUMBERS PER CELL, BECAUSE THEY ANSWER DIFFERENT QUESTIONS

Spearman rho is rank-based and the isotonic step is monotone, so the isotonic
step CANNOT change rho — a cell's rho is decided by the linear weights alone.
It asks: does the feature-to-force ORDERING transfer to another sensor?

MAE asks the other half: does the absolute newton scale transfer? It will not,
wherever two datasets cover different force ranges, and that is not a defect
of the reconstruction — it is what a per-dataset calibration means.

Reporting only rho would hide the second question; reporting only MAE would
punish a transfer that is perfectly ordered but differently scaled.

THE DIAGONAL IS HELD OUT. Fitting and scoring a dataset on the same frames
would put a number on the diagonal that no off-diagonal cell could be compared
against. On the diagonal the frames are split half/half within each group, as
the published protocol does; off the diagonal the whole source dataset fits and
the whole target dataset is scored.

EVERY CELL NEEDS ITS FLOOR

The five features are collinear and all monotone in contact size, and rho only
cares about ordering — so on an easy target almost ANY weight direction ranks
the forces correctly. Measured with 400 random weight vectors per target,
|rho|:

    target        random median   p90    max    own diagonal
    cnc_mini_26       0.884      0.981  0.992      0.992
    FoTa cnc          0.782      0.937  0.961      0.971
    Sparsh            0.711      0.908  0.938      0.936
    FEATS             0.514      0.666  0.690      0.513
    FeelAnyForce      0.460      0.620  0.694      0.661

That reframes the matrix. FEATS' model scoring 0.912 on cnc_mini_26 is barely
above what a RANDOM direction gets there (0.884) — the cell is a statement
about how easy cnc_mini_26 is to rank, not about FEATS. And on its own data,
FEATS' fit (0.513) is indistinguishable from a random projection (0.514).
Only FoTa cnc has a diagonal above its random MAXIMUM (0.971 vs 0.961), i.e.
is the one dataset where fitting the weights demonstrably beats guessing them.

The baseline is drawn under each column for exactly this reason.

    python -m force_recovery.cross_dataset [--per-dataset 600]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "cross_dataset.json"
FEAT_CACHE = OUT_ROOT / "feature_cache" / "cross_dataset_features.npz"
SEEDS = 5

DATASETS = (("cnc_mini_26", "GelSight Mini CNC"),
            ("cnc", "FoTa cnc_Mini"),
            ("feats", "FEATS (marker)"),
            ("sparsh", "Sparsh / Meta"),
            ("faf", "FeelAnyForce"))


def _fit(X, f):
    from sklearn.isotonic import IsotonicRegression
    w, *_ = np.linalg.lstsq(X, f, rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X @ w, f)
    return w, iso


def _apply(model, X):
    w, iso = model
    return iso.predict(X @ w)


def gather(name: str, cap: int, whole_only: bool = True):
    """Features/force/group for one dataset, sampled across the WHOLE pool."""
    from . import calib_free as CF
    from .force_recon_matrix import _feats, _rows
    from .visible_eval import in_fov, visible

    rows, get = _rows(name)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))]   # sample everywhere
    X, f, g = [], [], []
    for fr in rows:
        if len(f) >= cap:
            break
        img, ref = get(fr)
        if whole_only and not (in_fov(fr) and visible(img, ref)):
            continue
        X.append(_feats(CF.reconstruct(img, ref), False))
        f.append(float(fr["f"]))
        g.append(str(fr["group"]))
    return np.array(X), np.array(f), np.array(g)


def main() -> int:
    """One writer at a time — see `artifact_lock` for why."""
    from .artifact_lock import one_writer
    with one_writer(OUT):
        return _main()


def _main() -> int:
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=600)
    ap.add_argument("--datasets", nargs="*", default=[d for d, _ in DATASETS])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    # Features are the expensive part and do not change between re-scorings;
    # cached so the matrix can be recomputed in seconds.
    cache = {}
    if FEAT_CACHE.exists() and not args.refresh:
        z = np.load(FEAT_CACHE, allow_pickle=True)
        cache = {k: z[k] for k in z.files}
    data = {}
    for name, label in DATASETS:
        if name not in args.datasets:
            continue
        try:
            if f"{name}_X" in cache:
                X, f, g = (cache[f"{name}_X"], cache[f"{name}_f"],
                           cache[f"{name}_g"])
            else:
                X, f, g = gather(name, args.per_dataset)
                cache[f"{name}_X"], cache[f"{name}_f"] = X, f
                cache[f"{name}_g"] = g
        except Exception as exc:                               # noqa: BLE001
            print(f"  {name}: UNAVAILABLE — {exc}", flush=True)
            continue
        if len(f) < 60:
            print(f"  {name}: only {len(f)} frames — skipped", flush=True)
            continue
        data[name] = (X, f, g, label)
        print(f"  {name:12s} n={len(f):5d}  力 {f.min():6.2f}-{f.max():6.2f} N  "
              f"组 {len(set(g))}", flush=True)

    if cache:
        np.savez_compressed(FEAT_CACHE, **cache)
    names = [n for n, _ in DATASETS if n in data]
    rho = {a: {} for a in names}
    mae = {a: {} for a in names}
    for src in names:
        Xs, fs, gs, _ = data[src]
        for tgt in names:
            Xt, ft, gt, _ = data[tgt]
            if src == tgt:
                # Held out within group, and MEDIAN OVER SEEDS like the
                # published protocol. A single split put FEATS at 0.389 where
                # five seeds give more — the diagonal was noisier than every
                # off-diagonal cell it is meant to be compared against.
                rr, mm = [], []
                for s in range(SEEDS):
                    tr = np.zeros(len(ft), bool)
                    for gi, q in enumerate(sorted(set(gt))):
                        idx = np.where(gt == q)[0]
                        idx = idx[np.random.default_rng(1000 * s + 7 * gi)
                                  .permutation(len(idx))]
                        tr[idx[:len(idx) // 2]] = True
                    m = _fit(Xt[tr], ft[tr])
                    p, truth = _apply(m, Xt[~tr]), ft[~tr]
                    rr.append(float(spearmanr(p, truth).statistic))
                    mm.append(float(np.abs(p - truth).mean()))
                rho[src][tgt] = float(np.median(rr))
                mae[src][tgt] = float(np.median(mm))
                continue
            m = _fit(Xs, fs)
            p, truth = _apply(m, Xt), ft
            rho[src][tgt] = float(spearmanr(p, truth).statistic)
            mae[src][tgt] = float(np.abs(p - truth).mean())

    OUT.write_text(json.dumps(
        {"rho": rho, "mae": mae,
         "n": {k: int(len(v[1])) for k, v in data.items()},
         "force_range": {k: [float(v[1].min()), float(v[1].max())]
                         for k, v in data.items()}}, indent=1))

    hdr = "拟合\\评估   " + "".join(f"{n[:11]:>13s}" for n in names)
    for title, tab, fmt in (("Spearman ρ(秩,等距回归不影响它)", rho, "{:>13.3f}"),
                            ("MAE [N](绝对标度是否迁移)", mae, "{:>13.2f}")):
        print(f"\n{title}")
        print(hdr)
        for src in names:
            line = f"{src[:11]:12s}"
            for tgt in names:
                line += fmt.format(tab[src][tgt])
            print(line)
    # ---- summary: who teaches well, who is learnable
    print("\n每行的离对角均值(该数据集拟合出的模型迁移到别处的能力):")
    for src in names:
        off = [rho[src][t2] for t2 in names if t2 != src]
        print(f"  {src:12s} {np.mean(off):.3f}   (自身对角 {rho[src][src]:.3f})")
    print("每列的离对角均值(该数据集有多容易被别人的模型预测):")
    for tgt in names:
        off = [rho[s2][tgt] for s2 in names if s2 != tgt]
        print(f"  {tgt:12s} {np.mean(off):.3f}   (自身对角 {rho[tgt][tgt]:.3f})")

    # A cell needs its floor, exactly as a rho needs its shuffle. Here the
    # floor is a RANDOM weight vector: the five features are collinear and all
    # monotone in contact size, so on an easy target almost any direction
    # ranks the forces correctly and a high off-diagonal cell says more about
    # the target than about the source model.
    base = {}
    rngb = np.random.default_rng(0)
    for tgt in names:
        Xt, ft, _, _ = data[tgt]
        Xs = (Xt - Xt.mean(0)) / np.maximum(Xt.std(0), 1e-12)
        r = np.array([abs(spearmanr(Xs @ rngb.normal(size=Xs.shape[1]),
                                    ft).statistic) for _ in range(400)])
        base[tgt] = {"median": float(np.median(r)),
                     "p90": float(np.percentile(r, 90)), "max": float(r.max())}
    print("\n随机权重基线(400 次随机方向的 |rho|),每个目标数据集:")
    for tgt in names:
        b = base[tgt]
        print(f"  {tgt:12s} 中位 {b['median']:.3f}  p90 {b['p90']:.3f}  "
              f"最大 {b['max']:.3f}   自身对角 {rho[tgt][tgt]:.3f}")
    OUT.write_text(json.dumps(
        {"rho": rho, "mae": mae, "random_baseline": base,
         "n": {k: int(len(v[1])) for k, v in data.items()},
         "force_range": {k: [float(v[1].min()), float(v[1].max())]
                         for k, v in data.items()}}, indent=1))

    _figure(rho, mae, names, data, base)
    print(f"\n-> {OUT}")
    return 0


def _figure(rho, mae, names, data, base=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .cjk_font import use_cjk
    lbl = {n: l for n, (_, _, _, l) in data.items()}
    use_cjk(list(lbl.values()) + ["拟合数据集", "评估数据集", "秩相关",
                                  "绝对误差", "对角线为留出", "帧"])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    for a, (tab, title, cmap, fmt) in zip(ax, (
            (rho, "Spearman ρ — 排序是否迁移", "viridis", "{:.3f}"),
            (mae, "MAE [N] — 绝对标度是否迁移", "magma_r", "{:.2f}"))):
        M = np.array([[tab[s][t] for t in names] for s in names])
        im = a.imshow(M, cmap=cmap)
        a.set_xticks(range(len(names)))
        a.set_yticks(range(len(names)))
        a.set_xticklabels([f"{n}\n(n={len(data[n][1])})" for n in names],
                          fontsize=8.5)
        a.set_yticklabels(names, fontsize=9)
        a.set_xlabel("评估数据集")
        a.set_ylabel("拟合数据集")
        a.set_title(title, fontsize=11)
        for i in range(len(names)):
            for j in range(len(names)):
                v = M[i, j]
                rel = (v - M.min()) / max(M.max() - M.min(), 1e-9)
                a.text(j, i, fmt.format(v), ha="center", va="center",
                       fontsize=9,
                       color="white" if (rel < 0.45) ^ (cmap == "magma_r")
                       else "black",
                       fontweight="bold" if i == j else "normal")
        fig.colorbar(im, ax=a, shrink=0.82)
        if base is not None and tab is rho:
            a.set_xticklabels(
                [f"{n}\n(n={len(data[n][1])})\n随机基线 {base[n]['median']:.2f}"
                 for n in names], fontsize=8)
    fig.suptitle("跨数据集迁移矩阵 · 每个数据集单独拟合(一个线性最小二乘 + 等距回归)\n"
                 "对角线为组内半半留出、5 seeds 中位;非对角为整源拟合、整目标评估\n"
                 "每列下方的「随机基线」是随机权重方向能拿到的 |ρ| 中位 —— 单元格必须减去它来读",
                 fontsize=10)
    out = OUT_ROOT / "site2" / "assets" / "cross_dataset.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
