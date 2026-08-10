"""Predicted force against ground truth, both reconstructions, side by side.

A rho is a summary; this is the thing itself. One row per dataset, two panels
in the row — LUT and calibration-free — on SHARED axes, so the comparison a
reader makes by eye is between the methods and not between the axis ranges.

Every point is a held-out prediction: within each group the frames are split
half fit / half evaluate, a five-feature least squares is calibrated by
isotonic regression on the fit half, and only the evaluate half is plotted.
Seed 0 of the same five the tables report.

Each panel carries its rho, its MAE, its WITHIN-GROUP SHUFFLE control and the
MARGIN between them.

The shuffle is an ABSOLUTE rho, not a change in rho. It is the score this
protocol returns when the force labels are permuted inside each group — the
features carrying nothing — and the first version of this figure printed it as
"+0.226", which reads as an increase. It is a floor, and rho minus that floor
is what the reconstruction is worth. FeelAnyForce is excluded because its
floor is 0.82 against a rho of 0.95: a margin of 0.13, and a scatter that
would look convincing.

    xvfb-run -a python -m force_recovery.pred_vs_gt [--per-group 40]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT_PNG = OUT_ROOT / "site2" / "assets" / "pred_vs_gt.png"
OUT_JSON = OUT_ROOT / "feature_cache" / "pred_vs_gt.json"

# FeelAnyForce is deliberately absent: its within-group shuffle control reads
# +0.63, so its scatter would look convincing and mean nothing.
PLOT = (("cnc_mini_26", "cnc_mini_26 — GelSight Mini, CNC 压头"),
        ("cnc", "FoTa cnc_Mini"),
        ("sparsh", "Sparsh / Meta"),
        ("feats", "FEATS(marker 胶)"))

ARMS = (("lut", "LUT 标定重建"), ("calibfree", "免标定重建"))


def collect(name: str, per_group: int):
    """Held-out (truth, pred) for both arms on the fully imaged presses."""
    import collections

    from scipy.stats import spearmanr

    from . import calib_free as CF
    from .debug_gallery import stages
    from .force_eval_all import _one_seed, evaluate
    from .force_recon_matrix import _feats, _rows
    from .visible_eval import in_fov, visible

    rows, get = _rows(name)
    X = {a: [] for a, _ in ARMS}
    f, g = [], []
    seen = collections.Counter()
    for fr in rows:
        q = str(fr["group"])
        if seen[q] >= per_group:
            continue
        img, ref = get(fr)
        if not (in_fov(fr) and visible(img, ref)):
            continue
        seen[q] += 1
        X["lut"].append(_feats(stages(img, ref)["depth"], True))
        X["calibfree"].append(_feats(CF.reconstruct(img, ref)["depth"], False))
        f.append(float(fr["f"]))
        g.append(q)
    f, g = np.array(f), np.array(g)
    out = {"n": int(len(f)), "groups": int(len(set(g)))}
    for arm, _ in ARMS:
        Xa = np.array(X[arm])
        t, p, _gg = _one_seed(Xa, f, g, 0)
        r = evaluate(Xa, f, g)
        out[arm] = {"truth": t.tolist(), "pred": p.tolist(),
                    "rho": r["rho"], "mae": r["mae"],
                    "shuffle": r["shuffle_rho"]}
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .cjk_font import use_cjk

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=40)
    args = ap.parse_args()

    labels = [lbl for _, lbl in PLOT] + [lbl for _, lbl in ARMS]
    use_cjk(labels + ["真值力", "预测力", "打乱标签后", "净额", "帧", "组"])

    data = {}
    for name, label in PLOT:
        try:
            data[name] = collect(name, args.per_group)
            d = data[name]
            print(f"  {name:12s} n={d['n']:4d}  "
                  + "  ".join(f"{a}: rho {d[a]['rho']:.4f} shuf "
                              f"{d[a]['shuffle']:+.3f}" for a, _ in ARMS),
                  flush=True)
        except Exception as exc:                               # noqa: BLE001
            print(f"  {name}: UNAVAILABLE — {exc}", flush=True)

    rows = [(n, l) for n, l in PLOT if n in data]
    fig, ax = plt.subplots(len(rows), 2, figsize=(9.2, 4.2 * len(rows)),
                           constrained_layout=True)
    ax = np.atleast_2d(ax)
    for i, (name, label) in enumerate(rows):
        d = data[name]
        hi = max(max(d[a]["truth"] + d[a]["pred"]) for a, _ in ARMS)
        lo = 0.0
        for j, (arm, arm_label) in enumerate(ARMS):
            a = ax[i, j]
            t = np.array(d[arm]["truth"])
            p = np.array(d[arm]["pred"])
            a.plot([lo, hi], [lo, hi], color="#888", lw=1, ls="--", zorder=1)
            a.scatter(t, p, s=9, alpha=0.45, edgecolors="none",
                      color="#1f77b4" if arm == "lut" else "#d62728", zorder=2)
            a.set_xlim(lo, hi)
            a.set_ylim(lo, hi)
            a.set_aspect("equal")
            a.set_xlabel("真值力 [N]")
            if j == 0:
                a.set_ylabel("预测力 [N]")
            a.set_title(f"{label}\n{arm_label}", fontsize=10)
            # NOT "+0.226". The shuffle is an ABSOLUTE rho — the score this
            # protocol returns when the labels are permuted within each group
            # — and printing it with a leading + made it read as a change in
            # rho, which is what a reader asked. Shown as a value, with the
            # margin that is the thing actually being claimed.
            a.text(0.04, 0.96,
                   f"ρ = {d[arm]['rho']:.4f}\nMAE = {d[arm]['mae']:.3f} N\n"
                   f"打乱标签后 ρ = {d[arm]['shuffle']:.3f}\n"
                   f"净额 = {d[arm]['rho'] - d[arm]['shuffle']:.3f}\n"
                   f"n = {len(t)} 帧 / {d['groups']} 组",
                   transform=a.transAxes, va="top", fontsize=8.5,
                   bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    fig.suptitle("留出预测 vs 真值力 · 完整成像的按压 · 虚线为 y = x\n"
                 "同一行共享坐标轴,两栏只差重建方法 · "
                 "「打乱标签后 ρ」是特征无信息时该协议自身的得分,净额才是重建的贡献",
                 fontsize=10.5)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    OUT_JSON.write_text(json.dumps(
        {k: {a: {kk: vv for kk, vv in v[a].items() if kk not in
                 ("truth", "pred")} for a, _ in ARMS} | {"n": v["n"]}
         for k, v in data.items()}, indent=1))
    print(f"\n-> {OUT_PNG}  ({OUT_PNG.stat().st_size/1e6:.1f} MB)")
    print(f"-> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
