"""What the calibration-free solve actually measures on React, before integration.

Depth is an INTEGRAL of the estimated surface normals. When a depth map looks
wrong, the normal field says whether the measurement was wrong or the
integration was — and on React it was the integration: the same normals, put
through a solver that does not pin the frame border to zero, give a surface
without the cliff (see `force_recovery.poisson`).

Columns, per contact:

    raw · difference · normal map · gradient magnitude ·
    depth (Dirichlet, retired) · depth (Neumann) · mesh (Neumann)

The two depth columns are the SAME normals integrated two ways. Everything
else is held fixed, so the difference between them is the boundary condition
and nothing else.

    xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.react_normals_figure
"""
from __future__ import annotations

import numpy as np

from .lut_calibration import crop
from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "site2" / "assets" / "react_normals.png"
ROWS = (4822, 1122, 4496)
TASK, DATE, EPISODE, SIDE = ("motherboard", "2026-05-10",
                             "episode_000", "left")


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import calib_free as CF
    from . import eval_panel as EP
    from .showcase import _react_context

    _, _, _, _, frame, ref, h5 = _react_context(TASK, DATE, EPISODE, SIDE)
    # ONE display gain for the whole figure, derived from the data (the
    # smallest gain any row needs), so the three normal maps stay comparable
    # with each other and the number in the header is not a dial.
    recs = {r: CF.reconstruct(crop(frame(r)).astype(np.float32), ref,
                              normalize=True) for r in ROWS}
    GAIN = min(CF.display_gain(v["gx"], v["gy"]) for v in recs.values())
    heads = ["raw frame", EP.diff_caption("difference  dI = frame − ref"),
             "calibration-free normal map\nn = (−gx, −gy, 1)/|·|"
             f"   gradients ×{GAIN:g} for display",
             "gradient magnitude  |∇z|",
             "depth — Dirichlet solver\n(border forced to 0, RETIRED)",
             "depth — Neumann solver\n(border free)",
             "mesh — Neumann (relative z)"]
    fig, ax = plt.subplots(len(ROWS), len(heads),
                           figsize=(3.05 * len(heads), 2.6 * len(ROWS)),
                           constrained_layout=True)
    ax = np.atleast_2d(ax)

    def note(a, text, dark):
        a.text(0.03, 0.97, text, transform=a.transAxes, va="top", fontsize=8.5,
               color="white" if dark else "black",
               bbox=dict(facecolor="black" if dark else "white", alpha=0.45,
                         edgecolor="none", pad=1.6))

    for i, row in enumerate(ROWS):
        img = crop(frame(row)).astype(np.float32)
        rn = recs[row]
        rd = CF.reconstruct(img, ref, normalize=True, solver="dirichlet")
        gmag = np.hypot(rn["gx"], rn["gy"])
        v = rn["valid"]
        touch = [s for s, hit in (("top", v[0].any()), ("bottom", v[-1].any()),
                                  ("left", v[:, 0].any()),
                                  ("right", v[:, -1].any())) if hit]
        cells = [
            (np.clip(img, 0, 255).astype(np.uint8), None,
             (f"row {row}", False)),
            (EP.diff_rgb(img, ref), None,
             (f"contact reaches: {', '.join(touch) or 'no border'}", False)),
            (CF.normal_rgb(rn["gx"], rn["gy"], GAIN), None, None),
            (gmag, "magma", (f"|∇z| p99 {np.percentile(gmag, 99):.2f}", True)),
            (rd["depth"], "inferno", None),
            (rn["depth"], "inferno", None),
            (EP.mesh(rn["depth"], relative=True), None, None),
        ]
        for a, (data, cmap, ann) in zip(ax[i], cells):
            a.imshow(data, cmap=cmap)
            a.axis("off")
            if ann:
                note(a, ann[0], ann[1])
    for a, h in zip(ax[0], heads):
        a.set_title(h, fontsize=9)
    fig.suptitle("React motherboard/episode_000 left — the calibration-free "
                 "normals, and the same normals integrated two ways. "
                 "Analytic edge-contact error: Dirichlet 15.4% of peak, "
                 "Neumann 0.4%.", fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    h5.close()
    print(f"-> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    return OUT


if __name__ == "__main__":
    build()
