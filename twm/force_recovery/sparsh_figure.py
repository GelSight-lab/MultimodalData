"""Sparsh result figure: what a FOREIGN sensor costs the physics pipeline.

Three panels, all from the cached Sparsh features:

  1. predicted vs ground truth, per-batch calibrated (the honest headline)
  2. the within-batch shuffle control beside it (a global shuffle is
     meaningless here; on FeelAnyForce the pooled rho survived global
     shuffling at 0.442 vs 0.455, which is how we caught that its frame join
     was never demonstrated)
  3. cross-batch transfer: fit on one pad, apply unchanged to another

The caption this figure has to carry: on Sparsh the GlowTact-calibrated LUT
does NOT reconstruct valid geometry (a sphere press comes out bilobed with a
central dip), so these correlations track dI magnitude, not depth. They are a
transfer LIMIT, not a reconstruction result.
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
SITE_ASSETS = OUT_ROOT / "site" / "assets"
ORANGE, PURPLE, GREY = "#d95f02", "#7570b3", "#888888"
FEATS = ("vol", "vol2", "maxd", "area", "h1")


def _load():
    out = {}
    for p in sorted(CACHE.glob("sparsh_*.json")):
        d = json.loads(p.read_text())
        rows = d["rows"] if isinstance(d, dict) else d
        out[p.stem.replace("sparsh_", "")] = rows
    return out


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
    data = _load()
    t_all, p_all, shuf = [], [], []
    for name, rows in data.items():
        t, p = _calibrated(rows)
        t_all.append(t)
        p_all.append(p)
        # within-batch shuffle: destroys the frame-level link, KEEPS the
        # batch's force range — the only control that can fail honestly
        X, f = _xy(rows)
        fs = np.random.default_rng(11).permutation(f)
        idx = np.random.default_rng(0).permutation(len(f))
        fi, ei = idx[:len(idx) // 2], idx[len(idx) // 2:]
        shuf.append((fs[ei], _fit_apply(X[fi], fs[fi], X[ei])))
    t_all = np.concatenate(t_all)
    p_all = np.concatenate(p_all)
    st = np.concatenate([a for a, _ in shuf])
    sp = np.concatenate([b for _, b in shuf])

    rho = spearmanr(p_all, t_all).statistic
    mae = float(np.abs(p_all - t_all).mean())
    rho_s = spearmanr(sp, st).statistic

    # cross-batch transfer among the sphere pads
    sph = sorted(k for k in data if k.startswith("sphere"))
    M = np.zeros((len(sph), len(sph)))
    for i, a in enumerate(sph):
        Xa, fa = _xy(data[a])
        for j, b in enumerate(sph):
            Xb, fb = _xy(data[b])
            M[i, j] = spearmanr(_fit_apply(Xa, fa, Xb), fb).statistic

    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for a, (t, p, c, ti) in zip(ax, [
            (t_all, p_all, ORANGE, f"per-pad calibrated\nρ = {rho:.2f}  "
                                   f"MAE {mae:.3f} N"),
            (st, sp, GREY, f"control: labels shuffled\nwithin pad  "
                           f"ρ = {rho_s:.2f}")]):
        a.scatter(t, p, s=7, alpha=.45, color=c)
        lim = max(float(t.max()), 1e-3) * 1.05
        a.plot([0, lim], [0, lim], "k--", lw=.8, alpha=.5)
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_xlabel("ground truth |Fz| [N]")
        a.set_title(ti, fontsize=9)
        a.grid(alpha=.18, lw=.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("predicted [N]")

    im = ax[2].imshow(M, cmap="viridis", vmin=0.3, vmax=0.65)
    ax[2].set_xticks(range(len(sph)))
    ax[2].set_yticks(range(len(sph)))
    lbl = [s.replace("sphere_batch_", "b") for s in sph]
    ax[2].set_xticklabels(lbl, fontsize=8)
    ax[2].set_yticklabels(lbl, fontsize=8)
    ax[2].set_xlabel("applied to pad")
    ax[2].set_ylabel("fitted on pad")
    ax[2].set_title("cross-pad transfer (ρ)\ndiagonal = same pad", fontsize=9)
    for i in range(len(sph)):
        for j in range(len(sph)):
            ax[2].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       fontsize=6.5,
                       color="w" if M[i, j] < 0.55 else "k")
    fig.colorbar(im, ax=ax[2], fraction=.046)
    fig.suptitle("Sparsh (Meta) — a foreign GelSight: correlations survive, "
                 "geometry does not", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    out = SITE_ASSETS / "results_sparsh.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    stats = {"rho": float(rho), "mae": mae, "rho_within_pad_shuffle":
             float(rho_s), "n": int(len(t_all)),
             "transfer_off_diag_min": float(M[~np.eye(len(sph), dtype=bool)].min()),
             "transfer_off_diag_max": float(M[~np.eye(len(sph), dtype=bool)].max()),
             "transfer_diag_min": float(np.diag(M).min()),
             "transfer_diag_max": float(np.diag(M).max())}
    (CACHE / "sparsh_metrics.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return out


if __name__ == "__main__":
    print(build())
