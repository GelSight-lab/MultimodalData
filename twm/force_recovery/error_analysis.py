"""Where the force estimate goes wrong, frame by frame.

A rho says how well the ordering holds; it cannot say WHICH frames break it or
what they look like. This takes the same held-out predictions the tables are
built from, ranks them by absolute error, and draws the ten worst and five best
of each dataset through every stage of the pipeline:

    raw · difference dI · gx · gy · depth (LUT) · depth (calibration-free)

so a failure can be attributed to the image, the gradient solve, or the
integration rather than to "the method".

Errors are RELATIVE (|pred - true| / true force range), because the datasets
span 0.08-1.06 N and 0-34 N and an absolute newton means different things in
each. The best five are drawn as a control: if the good and the bad frames look
alike, the error is not coming from the reconstruction.

WHAT IT FOUND — AND ONE CLAIM RETRACTED

The figures answer the attribution question cleanly: the ten worst frames of
each dataset reconstruct as well as the five best. Same crisp gradient dipoles,
same compact depth blob, no ramping, no truncation. Whatever the residual error
is, it is in the depth→force mapping, not in image→gradient→depth.

I first read that residual as `IsotonicRegression(out_of_bounds="clip")`
refusing to extrapolate above its fit half's largest value, and wrote here that
"47% of the worst decile sit at their group's ceiling". THAT NUMBER WAS WRONG.
It came from an n=201 sampling produced while two runs of this module were
writing OUT_JSON at once (see `main`). On the clean single-writer run:

    dataset       n    pinned at the ceiling    ... of the worst decile
    cnc_mini_26  303       5  (1.7%)                   10%
    cnc          266      10  (3.8%)                    4%
    feats         98       1  (1.0%)                    0%
    sparsh       205      10  (4.9%)                   19%
    faf          300      15  (5.0%)                    7%

So clipping explains a small minority of the largest errors, not half of them.
Nor is any other single mechanism responsible: the worst decile sits mildly
nearer its group's force extremes (0.29 against 0.19 of the range, Spearman
+0.05 to +0.24 — weak, and absent entirely on cnc), and its sign is not
consistently a bias (43-74% underestimates depending on the dataset).

The honest statement is the negative one, which is also the useful one: a
better reconstruction will not move these frames.

    xvfb-run -a python -m force_recovery.error_analysis [--per-dataset 400]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT_DIR = OUT_ROOT / "site2" / "assets"
OUT_JSON = OUT_ROOT / "feature_cache" / "error_analysis.json"
N_WORST, N_BEST = 10, 5

DATASETS = (("cnc_mini_26", "GelSight Mini CNC"),
            ("cnc", "FoTa cnc_Mini"),
            ("feats", "FEATS (marker)"),
            ("sparsh", "Sparsh / Meta"),
            ("faf", "FeelAnyForce"))


def collect(name: str, cap: int):
    """Held-out predictions plus the frames they came from.

    The split and the model come from `force_eval_all.heldout_pred` — the same
    call the tables use. An earlier version of this re-implemented that loop
    here and matched it only because the seeding happens to coincide at seed 0.
    """
    from . import calib_free as CF
    from .force_eval_all import heldout_pred
    from .force_recon_matrix import _feats, _rows
    from .visible_eval import in_fov, visible

    rows, get = _rows(name)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    keep, X, f, g = [], [], [], []
    for fr in rows:
        if len(f) >= cap:
            break
        img, ref = get(fr)
        if not (in_fov(fr) and visible(img, ref)):
            continue
        X.append(_feats(CF.reconstruct(img, ref)["depth"], False))
        f.append(float(fr["f"]))
        g.append(str(fr["group"]))
        keep.append(fr)
    X, f, g = np.array(X), np.array(f), np.array(g)
    pred, _labels = heldout_pred(X, f, g, seed=0)
    ok = np.isfinite(pred)

    # The figure and the caption must describe the SAME frame. Recompute the
    # features of a sample of the kept frames straight from their images and
    # require them to match the row that produced the prediction; a mismatch
    # means an indexing slip between `keep` and `X`, which would put a number
    # under the wrong picture and look entirely plausible.
    idx_ok = np.flatnonzero(ok)
    for j in idx_ok[:: max(len(idx_ok) // 8, 1)][:8]:
        img, ref = get(keep[j])
        again = np.array(_feats(CF.reconstruct(img, ref)["depth"], False))
        if not np.allclose(again, X[j], rtol=1e-9, atol=1e-12):
            raise AssertionError(
                f"{name}: frame {j} does not reproduce its own features — the "
                f"picture and the number would refer to different frames")

    span = max(f.max() - f.min(), 1e-9)
    rel = np.abs(pred - f) / span
    return ([keep[i] for i in idx_ok], get,
            f[ok], pred[ok], g[ok], rel[ok], span)


def figure(name: str, label: str, cap: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import calib_free as CF
    from . import eval_panel as EP
    from .cjk_font import use_cjk
    from .debug_gallery import stages

    frames, get, truth, pred, grp, rel, span = collect(name, cap)
    if len(frames) < N_WORST + N_BEST:
        return {"dataset": name, "available": False,
                "reason": f"only {len(frames)} held-out frames"}
    order = np.argsort(-rel)
    pick = list(order[:N_WORST]) + list(order[::-1][:N_BEST])
    heads = ["原图", "差分 dI(彩色)", "gx", "gy", "LUT 深度", "免标定深度"]
    use_cjk(heads + ["误差最大", "误差最小", "真值", "预测", "相对误差"])

    fig, ax = plt.subplots(len(pick), 6, figsize=(19.5, 2.75 * len(pick)),
                           constrained_layout=True)
    for r, i in enumerate(pick):
        fr = frames[i]
        img, ref = get(fr)
        cf = CF.reconstruct(img, ref)
        lut = stages(img, ref)["depth"]
        gm = max(np.abs(cf["gx"]).max(), np.abs(cf["gy"]).max(), 1e-9) * 0.35
        cells = [(np.clip(img, 0, 255).astype(np.uint8), None, None),
                 (EP.diff_rgb(img, ref), None, None),
                 (cf["gx"], "coolwarm", (-gm, gm)),
                 (cf["gy"], "coolwarm", (-gm, gm)),
                 (lut, "inferno", None),
                 (cf["depth"] / max(cf["depth"].max(), 1e-12), "inferno", None)]
        for a, (d, cm, lim) in zip(ax[r], cells):
            a.imshow(d, cmap=cm, **({} if lim is None
                                    else {"vmin": lim[0], "vmax": lim[1]}))
            a.axis("off")
        tag = "误差最大" if r < N_WORST else "误差最小"
        ax[r, 0].text(0.03, 0.96,
                      f"{tag} #{(r if r < N_WORST else r - N_WORST) + 1}\n"
                      f"{grp[i]}\n真值 {truth[i]:.2f} N  预测 {pred[i]:.2f} N\n"
                      f"相对误差 {rel[i]*100:.1f}%",
                      transform=ax[r, 0].transAxes, va="top", fontsize=8.5,
                      bbox=dict(facecolor="#ffe9e9" if r < N_WORST else "#e9ffe9",
                                alpha=0.85, edgecolor="none"))
    for a, h in zip(ax[0], heads):
        a.set_title(h, fontsize=10)
    fig.suptitle(f"{label} — 误差分析 · 上 {N_WORST} 行是相对误差最大的帧,"
                 f"下 {N_BEST} 行是最小的(对照)\n"
                 f"相对误差 = |预测−真值| / 该数据集力量程({span:.2f} N)",
                 fontsize=11)
    out = OUT_DIR / f"errors_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print(f"  {name:12s} n={len(frames)}  最大相对误差 {rel.max()*100:.1f}%  "
          f"中位 {np.median(rel)*100:.1f}%  -> {out.name}", flush=True)
    return {"dataset": name, "label": label, "available": True,
            "n_heldout": int(len(frames)), "force_span": float(span),
            "rel_err_median": float(np.median(rel)),
            "rel_err_p90": float(np.percentile(rel, 90)),
            "rel_err_max": float(rel.max()), "asset": out.name,
            "worst_groups": [str(grp[i]) for i in order[:N_WORST]]}


def main() -> int:
    """One writer at a time.

    Two runs of this module were once in flight together — a leftover
    background job and a new one — and both wrote OUT_JSON. The artifact then
    held a mix of two samplings (n=201 in the file against n=303 in the log)
    and the printed lines interleaved, which is how it was noticed. A lock is
    cheaper than discovering that from a log.
    """
    import os
    lock = OUT_JSON.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"another run holds {lock} (pid "
                         f"{lock.read_text().strip()}) — wait, or remove it")
    try:
        return _main()
    finally:
        lock.unlink(missing_ok=True)


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=400)
    ap.add_argument("--datasets", nargs="*", default=[d for d, _ in DATASETS])
    args = ap.parse_args()
    out = []
    for name, label in DATASETS:
        if name not in args.datasets:
            continue
        try:
            out.append(figure(name, label, args.per_dataset))
        except Exception as exc:                               # noqa: BLE001
            print(f"  {name}: UNAVAILABLE — {exc}", flush=True)
            out.append({"dataset": name, "available": False,
                        "reason": str(exc)})
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
