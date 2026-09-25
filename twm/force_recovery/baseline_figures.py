"""Predicted-vs-truth scatters that were missing: Sparsh three-way, FeelAnyForce.

Two gaps the numbers had but the pictures did not:

* `results_sparsh.png` compared our OWN two tables (GlowTact vs self-calibrated)
  and showed no baseline at all, even after the three-way comparison landed on
  the page — including the fact that FeelAnyForce beats us there.
* FeelAnyForce as a DATASET had a row and a section but no scatter, so a reader
  could not see that its 14 admissible captures look different from the 28
  without a contact-free reference.

Both figures are drawn from the cached predictions, never from re-typed
numbers, and each panel prints the same rho the results table quotes.

Run: python -m force_recovery.baseline_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
SITE = OUT_ROOT / "site" / "assets"
ORANGE, GREY, PURPLE, GREEN = "#d95f02", "#8a8a8a", "#7570b3", "#1b9e77"


def _panel(ax, t, p, colour, title, note=None, unit="N"):
    t, p = np.asarray(t, float), np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    ax.scatter(t, p, s=6, alpha=.35, color=colour, linewidths=0)
    lim = max(float(t.max()), float(np.percentile(p, 99.5)), 1e-3) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=.8, alpha=.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"ground truth [{unit}]", fontsize=9)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=.18, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rho = spearmanr(p, t).statistic
    mae = float(np.abs(p - t).mean())
    ax.text(.04, .96, f"ρ = {rho:.3f}", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=colour, va="top")
    ax.text(.04, .855, f"MAE {mae:.3f} {unit}", transform=ax.transAxes,
            fontsize=8.5, color="#666", va="top")
    if note:
        ax.text(.04, .78, note, transform=ax.transAxes, fontsize=8,
                color="#666", va="top")
    return rho, mae


def sparsh_three_way():
    """Ours vs FeelAnyForce vs FEATS on the identical in-view frames."""
    d = json.loads((CACHE / "sparsh_baselines_scatter.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.0))
    _panel(ax[0], d["truth"], d["ours"], ORANGE,
           "ours — physics\nper-pad calibrated")
    _panel(ax[1], d["truth"], d["anyforce"], GREEN,
           "FeelAnyForce\nno fitting, reads newtons", "zero labels")
    _panel(ax[2], d["truth"], d["feats"], PURPLE,
           "FEATS U-net\nno fitting", "random control ρ = 0.265")
    ax[0].set_ylabel("predicted [N]", fontsize=9)
    fig.suptitle("Sparsh (GelSight Mini, markerless): the one dataset where a "
                 "trained network beats us — identical frames",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = SITE / "results_sparsh_baselines.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def faf_dataset():
    """FeelAnyForce as a dataset: the 14 usable captures vs the 28 without."""
    d = json.loads((CACHE / "faf_scatter.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.0))
    _panel(ax[0], d["tierA_truth"], d["tierA_pred"], ORANGE,
           "14 captures WITH a contact-free reference",
           "within-capture shuffle ρ = 0.338")
    _panel(ax[1], d["tierB_truth"], d["tierB_pred"], GREY,
           "28 captures with NO unloaded frame",
           "min |Fz| in these is 4.9–6.0 N")
    ax[0].set_ylabel("predicted [N]", fontsize=9)
    fig.suptitle("FeelAnyForce as a dataset: a valid reference frame decides "
                 "everything", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = SITE / "results_faf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    sparsh_three_way()
    faf_dataset()
