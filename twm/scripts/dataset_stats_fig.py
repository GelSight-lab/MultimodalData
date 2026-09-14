"""Draw the dataset-card figures from `dataset_stats.gather()`'s output.

Four panels, each answering a different question rather than drawing the same
numbers four times:

  A  Scale       how much trainable data per task, and from how many recordings
  B  Force       the unsaturated contact-force distribution, with the pipeline
                 clip drawn explicitly
  C  Occupancy   one-handed vs two-handed contact -- the point of a bimanual set
  D  Validity    GelSight's 17.8 Hz against the 29.8 Hz write tick, beside the
                 saturated fraction
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ORDER = ["motherboard", "pushT", "rope", "validation"]
_PAD = 0.35          # category-axis padding; see `panel`
COL = {"motherboard": "#2F6FB5", "pushT": "#C8622B", "rope": "#3F8F5B",
       "validation": "#7A5AA8"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "figure.dpi": 200,
})


def panel(d: dict, ceiling: float, path: Path, title: str) -> None:
    ts = [t for t in ORDER if t in d]
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))

    mins = [d[t]["minutes"] for t in ts]
    bars = ax[0].barh(ts, mins, color=[COL[t] for t in ts], height=.6)
    for t, r in zip(ts, bars):
        ax[0].text(r.get_width() + max(mins) * .02,
                   r.get_y() + r.get_height() / 2,
                   f"{d[t]['minutes']:.1f} min\n{d[t]['segs']} seg / "
                   f"{d[t]['sources']} rec", va="center", fontsize=7)
    ax[0].set_xlim(0, max(mins) * 1.55)
    ax[0].set_xlabel("published minutes")
    ax[0].set_title("A  Scale", loc="left", fontweight="bold")
    # Pad the category axis explicitly. Left to autoscale, a single-task
    # figure (data/validation) draws its one bar as a slab filling the axes,
    # because matplotlib fits the limits to the one category.
    ax[0].set_ylim(len(ts) - 0.5 + _PAD, -0.5 - _PAD)

    q = np.linspace(0, 1, 101)
    for t in ts:
        ax[1].plot(d[t]["f_ecdf_x"], q, color=COL[t], lw=1.5, label=t)
    ax[1].axvline(ceiling, color="#B00", lw=1.1, ls="--")
    # The annotation sits above the curves and clear of the legend; set
    # vertically beside the line it overlapped the legend and could not be read.
    ax[1].annotate(f"{ceiling} N pipeline clip", xy=(ceiling, .97),
                   xytext=(ceiling - 1.55, .97), color="#B00", fontsize=6.6,
                   va="center", ha="right",
                   arrowprops=dict(arrowstyle="->", color="#B00", lw=.8))
    ax[1].set_xlim(0, ceiling * 1.04)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("normal contact force (N)")
    ax[1].set_ylabel("cumulative fraction of contact frames")
    ax[1].set_title("B  Contact force (unsaturated)", loc="left", fontweight="bold")
    ax[1].legend(frameon=False, loc="lower right")

    x = np.arange(len(ts))
    w = .26
    for i, (key, alpha) in enumerate([("contact_pct_L", .95),
                                      ("contact_pct_R", .62),
                                      ("both_pct", .32)]):
        ax[2].bar(x + (i - 1) * w, [d[t][key] for t in ts], w,
                  color=[COL[t] for t in ts], alpha=alpha)
    # Neutral grey swatches: shade means left/right/both, colour means task.
    # Drawing the legend in one task's colour reads as "all three are that task".
    ax[2].legend(handles=[Patch(facecolor="#555", alpha=a, label=l)
                          for a, l in ((.95, "left"), (.62, "right"),
                                       (.32, "both hands"))],
                 frameon=False, loc="upper right", fontsize=7)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(ts, rotation=12, ha="right")
    ax[2].set_xlim(-0.5 - _PAD, len(ts) - 0.5 + _PAD)
    ax[2].set_ylabel("% of all frames")
    ax[2].set_title("C  Contact occupancy", loc="left", fontweight="bold")

    ax[3].bar(x - .16, [d[t]["new_pct"] for t in ts], .3,
              color=[COL[t] for t in ts], label="new tactile frames")
    ax[3].bar(x + .16, [d[t]["sat_pct"] for t in ts], .3,
              color=[COL[t] for t in ts], alpha=.42, hatch="///",
              edgecolor="white", label="force at ceiling (of contact)")
    ax[3].axhline(100 * 17.8 / 29.8, color="#666", lw=.9, ls=":")
    # Axes-fraction x, so the label starts just inside the plot whatever the
    # category count; at a fixed data x it fell outside the axes and over the
    # tick labels on the single-task figure.
    ax[3].text(0.02, 100 * 17.8 / 29.8 + 1.5, "17.8/29.8 Hz sensor ceiling",
               transform=ax[3].get_yaxis_transform(),
               fontsize=6.3, ha="left", va="bottom", color="#444")
    # rope saturates on 0.6 % of contact frames, which is invisible as a bar;
    # label the value or a reader takes it for zero.
    for xi, t in zip(x, ts):
        ax[3].text(xi + .16, d[t]["sat_pct"] + 1.2, f'{d[t]["sat_pct"]:.1f}',
                   ha="center", fontsize=6.3, color="#333")
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(ts, rotation=12, ha="right")
    ax[3].set_xlim(-0.5 - _PAD, len(ts) - 0.5 + _PAD)
    ax[3].set_ylabel("percent")
    ax[3].set_ylim(0, 88)
    ax[3].set_title("D  Tactile validity / force saturation", loc="left",
                    fontweight="bold")
    ax[3].legend(frameon=False, loc="upper right", fontsize=6.8)

    fig.suptitle(title, x=.005, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def draw(stats: dict, out_dir: Path) -> list[Path]:
    ceiling = stats["ceiling"]
    made = []
    if stats.get("new_era"):
        p = out_dir / "stats_wrist_era.png"
        panel(stats["new_era"], ceiling, p,
              "REACT - wrist-camera era (from 2026-09-10): "
              "data/{motherboard, pushT, rope}")
        made.append(p)
    if stats.get("old_era"):
        p = out_dir / "stats_arducam_session.png"
        panel(stats["old_era"], ceiling, p,
              "REACT - earlier Arducam session (2026-09-09): data/validation")
        made.append(p)
    return made
