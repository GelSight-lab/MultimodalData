"""recon_compare.png — three React presses through three reconstructions.

WHY THIS FILE EXISTS AT ALL. The figure was already on the live site with no
producer anywhere in the repo: a PNG sitting in `_recover/` that
`calibfree_page` copied into the assets folder. Nothing could regenerate it, so
nothing could correct it, and it had drifted twice —

  * its difference column was captioned "(×3)" while `visualize.DIFF_GAIN` had
    been 1.0 for weeks, and
  * its title had been written with CJK text that the matplotlib font could not
    render, so the live site showed a row of tofu boxes.

An artifact with no producer is a claim nobody can re-derive. This regenerates
it from the current laws: `eval_panel` columns, `diff_caption` from the gain,
`mesh` from the one uncropped renderer, and the flat-gel leak computed here
rather than transcribed.

The comparison: React's own LUT (cnc_mini_26), the Sparsh LUT, and the
calibration-free solve. The leak is `calib_free.flat_gel_leak` — mean |depth|
off-contact over peak depth, which needs no force label and is zero for a
physically coherent reconstruction.

    xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.react_leak_figure
"""
from __future__ import annotations

import numpy as np

from .lut_calibration import CAL_OUT, MM_PER_PIXEL, crop
from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "site2" / "assets" / "recon_compare.png"

# The rows the published figure used, kept so the new figure is comparable with
# the old one rather than a different sample making a different point.
ROWS = (4822, 1122, 4496)
TASK, DATE, EPISODE, SIDE = ("motherboard", "2026-05-10",
                            "episode_000", "left")


def _luts() -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for label, fn in (("React LUT (cnc_mini_26)", "glowtact_lut.npz"),
                      ("Sparsh LUT", "sparsh_lut.npz")):
        p = CAL_OUT / fn
        if not p.exists():
            raise SystemExit(f"missing {p} — the figure compares LUTs, so a "
                             f"missing one is a missing column, not a default")
        z = np.load(p)
        out.append((label, z["lut"], z["count"]))
    return out


def build() -> "object":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import calib_free as CF
    from . import eval_panel as EP
    from .recon_study import stages_full
    from .showcase import _react_context

    luts = _luts()
    _, _, _, _, frame, ref, h5 = _react_context(TASK, DATE, EPISODE, SIDE)

    # Titles are COLUMN HEADERS, written once on row 0; per-frame numbers are
    # drawn INSIDE their own axes. Per-cell titles were tried first and every
    # one after row 0 was clipped by the image above it — a caption that
    # depends on layout luck is a caption that will go missing again.
    ncol = 2 + 3 * 2                      # raw, diff, then depth+mesh per method
    fig, ax = plt.subplots(len(ROWS), ncol,
                           figsize=(3.05 * ncol, 2.6 * len(ROWS)),
                           constrained_layout=True)
    ax = np.atleast_2d(ax)

    def note(a, text, dark_bg: bool) -> None:
        a.text(0.03, 0.97, text, transform=a.transAxes, va="top", fontsize=8.5,
               color="white" if dark_bg else "black",
               bbox=dict(facecolor="black" if dark_bg else "white",
                         alpha=0.45, edgecolor="none", pad=1.6))

    for i, row in enumerate(ROWS):
        img = crop(frame(row)).astype(np.float32)
        cells = [(np.clip(img, 0, 255).astype(np.uint8), None,
                  (f"row {row}", False)),
                 (EP.diff_rgb(img, ref), None, None)]
        for label, lut, cnt in luts:
            st = stages_full(img, ref, lut, cnt)
            d = np.clip(st["depth"], 0, None)
            leak = CF.flat_gel_leak(d, st["valid"])
            cells += [(d, "inferno",
                       (f"leak {leak:.3f}   peak {d.max():.2f} mm", True)),
                      (EP.mesh(d), None, None)]
        r = CF.reconstruct(img, ref)
        dcf = np.clip(r["depth"], 0, None)
        leak = CF.flat_gel_leak(dcf, r["valid"])
        cells += [(dcf / max(float(dcf.max()), 1e-12), "inferno",
                   (f"leak {leak:.3f}", True)),
                  (EP.mesh(dcf, relative=True), None, None)]
        for a, (data, cmap, ann) in zip(ax[i], cells):
            a.imshow(data, cmap=cmap)
            a.axis("off")
            if ann:
                note(a, ann[0], ann[1])
    heads = ["raw frame", EP.diff_caption("difference  dI = frame − ref"),
             f"{luts[0][0]}\ndepth", f"{luts[0][0]}\nmesh",
             f"{luts[1][0]}\ndepth", f"{luts[1][0]}\nmesh",
             "calibration-free\ndepth (relative)",
             "calibration-free\nmesh (relative z)"]
    for a, htxt in zip(ax[0], heads):
        a.set_title(htxt, fontsize=9)
    fig.suptitle(f"React {TASK}/{EPISODE} {SIDE} — three reconstructions of the "
                 f"same contact.  Flat-gel leak: mean |depth| off-contact / "
                 f"peak depth, 0 is coherent.", fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    h5.close()
    print(f"-> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    return OUT


if __name__ == "__main__":
    build()
