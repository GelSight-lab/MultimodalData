"""Sparsh result figure: per-sensor sphere self-calibration on a foreign gel.

Three panels, all from the cached Sparsh features:

  1. predicted vs ground truth, per-batch calibrated (the honest headline)
  2. the within-batch shuffle control beside it (a global shuffle is
     meaningless here; on FeelAnyForce the pooled rho survived global
     shuffling at 0.442 vs 0.455, which is how we caught that its frame join
     was never demonstrated)
  3. cross-batch transfer: fit on one pad, apply unchanged to another

The story this figure has to carry, after the self-calibration experiment:
the GlowTact table is simply the WRONG map for this sensor (its gradients sit
93.3 deg from the analytic sphere gradient, chance being 90), and rebuilding
the table from Sparsh's own sphere presses fixes it (4.5 deg, single dome,
residual 0.179 -> 0.0545 mm). Force follows: in-view rho 0.878 -> 0.968.
So this is "calibrate once per sensor", not zero-shot transfer.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
CACHE_SP = OUT_ROOT / "feature_cache_sparshlut"
PROBE = OUT_ROOT / "sparsh_probe"
SITE_ASSETS = OUT_ROOT / "site" / "assets"
ORANGE, PURPLE, GREY = "#d95f02", "#7570b3", "#888888"
FEATS = ("vol", "vol2", "maxd", "area", "h1")


def _load(which: str = "sparsh"):
    """which='sparsh' -> Sparsh-native LUT features; 'glowtact' -> the old ones."""
    root = CACHE_SP if which == "sparsh" else CACHE
    out = {}
    for p in sorted(root.glob("sparsh_*.json")):
        if p.name == "sparsh_metrics.json":
            continue
        d = json.loads(p.read_text())
        rows = d["rows"] if isinstance(d, dict) else d
        out[p.stem.replace("sparsh_", "")] = rows
    return out


def _inview_mask(name, rows):
    """Contact-disc visibility, the filter that decides everything here.

    36% of Sparsh presses show no detectable disc and 11% are clipped by the
    crop; the clipped subset carries the HIGHEST median force yet scores worse,
    so this is a visibility effect, not force-range filtering.
    """
    circ = json.loads((PROBE / "eval_circles.json").read_text())
    key = name.replace("_batch_", "_b")
    c = circ.get(key, {})
    return np.array([bool(c.get(str(r["index"])) and c[str(r["index"])]["inview"])
                     for r in rows])


def _xy(rows):
    X = np.column_stack([[r[k] for r in rows] for k in FEATS])
    return X, np.array([abs(r["fz"]) for r in rows])


def _fit_apply(Xf, ff, Xe):
    w, *_ = np.linalg.lstsq(Xf, ff, rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(Xf @ w, ff)
    return iso.predict(Xe @ w)


def _calibrated(rows, seeds=5):
    """Half/half within-batch, returns (truth, pred) pooled over seeds."""
    X, f = _xy(rows)
    T, P = [], []
    for s in range(seeds):
        idx = np.random.default_rng(s).permutation(len(f))
        fi, ei = idx[:len(idx) // 2], idx[len(idx) // 2:]
        P.append(_fit_apply(X[fi], f[fi], X[ei]))
        T.append(f[ei])
    return np.concatenate(T), np.concatenate(P)


def build() -> Path:
    """Panel 1: GlowTact vs Sparsh-native LUT on identical in-view frames.
    Panel 2: the within-pad shuffle control. Panel 3: cross-pad transfer."""
    res = {}
    for which in ("glowtact", "sparsh"):
        data = _load(which)
        T, P, ST, SP = [], [], [], []
        for name, rows in data.items():
            m = _inview_mask(name, rows)
            if m.sum() < 40:
                continue
            rws = [r for r, k in zip(rows, m) if k]
            t, p = _calibrated(rws)
            T.append(t)
            P.append(p)
            X, f = _xy(rws)
            fs = np.random.default_rng(11).permutation(f)
            idx = np.random.default_rng(0).permutation(len(f))
            fi, ei = idx[:len(idx) // 2], idx[len(idx) // 2:]
            ST.append(fs[ei])
            SP.append(_fit_apply(X[fi], fs[fi], X[ei]))
        res[which] = dict(t=np.concatenate(T), p=np.concatenate(P),
                          st=np.concatenate(ST), sp=np.concatenate(SP))

    def rm(d, a="t", b="p"):
        return (spearmanr(d[b], d[a]).statistic,
                float(np.abs(d[b] - d[a]).mean()))

    rg, mg = rm(res["glowtact"])
    rs, ms = rm(res["sparsh"])
    rsh = spearmanr(res["sparsh"]["sp"], res["sparsh"]["st"]).statistic

    # cross-pad transfer, Sparsh-native LUT, in-view sphere pads
    data = _load("sparsh")
    sph = sorted(k for k in data if k.startswith("sphere"))
    sub = {}
    for k in sph:
        m = _inview_mask(k, data[k])
        sub[k] = _xy([r for r, kk in zip(data[k], m) if kk])
    M = np.zeros((len(sph), len(sph)))
    for i, a in enumerate(sph):
        for j, b in enumerate(sph):
            M[i, j] = spearmanr(_fit_apply(*sub[a], sub[b][0]),
                                sub[b][1]).statistic

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for a, (d, c, ti) in zip(ax[:2], [
            (res["glowtact"], GREY,
             f"GlowTact table (wrong sensor)\nρ = {rg:.3f}   MAE {mg:.3f} N"),
            (res["sparsh"], ORANGE,
             f"self-calibrated on Sparsh spheres\nρ = {rs:.3f}   "
             f"MAE {ms:.3f} N")]):
        a.scatter(d["t"], d["p"], s=6, alpha=.35, color=c)
        lim = float(d["t"].max()) * 1.05
        a.plot([0, lim], [0, lim], "k--", lw=.8, alpha=.5)
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_xlabel("ground truth |Fz| [N]")
        a.set_title(ti, fontsize=9)
        a.grid(alpha=.18, lw=.6)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    ax[0].set_ylabel("predicted [N]")
    ax[1].text(.04, .96, f"shuffle control ρ = {rsh:.2f}", fontsize=8,
               color="#666", va="top", transform=ax[1].transAxes)

    im = ax[2].imshow(M, cmap="viridis", vmin=0.90, vmax=1.0)
    lbl = [s.replace("sphere_batch_", "pad ") for s in sph]
    ax[2].set_xticks(range(len(sph)))
    ax[2].set_yticks(range(len(sph)))
    ax[2].set_xticklabels([l[-1] for l in lbl], fontsize=8)
    ax[2].set_yticklabels([l[-1] for l in lbl], fontsize=8)
    ax[2].set_xlabel("applied to pad")
    ax[2].set_ylabel("fitted on pad")
    ax[2].set_title("cross-pad transfer (ρ)\none table, six gel pads",
                    fontsize=9)
    for i in range(len(sph)):
        for j in range(len(sph)):
            ax[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       fontsize=6.5, color="w" if M[i, j] < 0.96 else "k")
    fig.colorbar(im, ax=ax[2], fraction=.046)
    fig.suptitle("Sparsh (Meta): rebuilding the table from the sensor's own "
                 "sphere presses — in-view frames", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    out = SITE_ASSETS / "results_sparsh.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    stats = {"rho_glowtact_lut": float(rg), "mae_glowtact_lut": mg,
             "rho_sparsh_lut": float(rs), "mae_sparsh_lut": ms,
             "rho_within_pad_shuffle": float(rsh),
             "n_inview": int(len(res["sparsh"]["t"])),
             "transfer_min": float(M.min()), "transfer_max": float(M.max())}
    (CACHE / "sparsh_metrics.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return out


if __name__ == "__main__":
    print(build())
