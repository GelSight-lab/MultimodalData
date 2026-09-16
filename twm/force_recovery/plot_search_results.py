"""Generate the force search's audit figure from measured report artifacts."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model_search import ROOT


def main():
    report = json.loads((ROOT / "report.json").read_text())
    mixed = json.loads((ROOT / "multishape_report.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    rows = report["test_predictions"]
    limit = max(8, max(r["prediction_n"] for r in rows)) + 0.2
    ax.scatter([r["f"] for r in rows], [r["prediction_n"] for r in rows],
               s=18, color="#147d92", alpha=0.7, edgecolors="none")
    ax.plot([0, limit], [0, limit], color="#666666", linewidth=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Measured force (N)", ylabel="Predicted force (N)",
           title=f"Round: 158 held-out frames, MAE {report['heldout']['mae_n']:.3f} N")
    ax = axes[0, 1]
    features = ("basic", "geometry", "image", "combined")
    best = [min(r["cv_mae_n"] for r in report["search"] if r["feature"] == key) for key in features]
    ax.bar(features, best, color=["#7c8587", "#147d92", "#ae557e", "#2d8a62"])
    ax.axhline(0.5, color="#b83838", linestyle="--", linewidth=1)
    ax.set(ylabel="Training grouped-CV MAE (N)", title="Feature comparison (selection scores)")
    for i, value in enumerate(best):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    ax = axes[1, 0]
    families = list(mixed["per_shape_heldout"])
    values = [mixed["per_shape_heldout"][f]["mae_n"] for f in families]
    ax.bar(families, values, color="#2d8a62")
    ax.axhline(0.5, color="#b83838", linestyle="--", linewidth=1)
    ax.set(ylabel="Held-out-position MAE (N)", title=f"Six trained shapes: 805 frames, MAE {mixed['heldout']['mae_n']:.3f} N")
    ax.tick_params(axis="x", labelrotation=20)
    ax = axes[1, 1]
    folds = mixed["unseen_shapes"]["folds"]
    ax.bar([r["heldout_shape"] for r in folds], [r["mae_n"] for r in folds], color="#ae557e")
    ax.axhline(0.5, color="#b83838", linestyle="--", linewidth=1)
    ax.set(ylabel="Unseen-shape MAE (N)", title=f"Nested leave-one-shape-out: MAE {mixed['unseen_shapes']['mae_n']:.3f} N")
    ax.tick_params(axis="x", labelrotation=20)
    for ax in axes.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Force estimation: calibration gains and transfer limits", fontsize=15)
    fig.savefig(ROOT / "search_results.png", dpi=180)
    fig.savefig(ROOT / "search_results.pdf")
    print(ROOT / "search_results.png")


if __name__ == "__main__":
    main()
