"""What a truncated press is, and why its depth is not recoverable.

A press counts as fully imaged only if BOTH hold:

  1. the commanded position is inside the sensor's field of view, and
  2. the imaged contact core — |dI| smoothed, thresholded well above the noise
     floor — touches none of the four borders.

Either alone lets the wrong frames through. A probe centred outside the view
can still put an interior-looking blob in frame, and a contact commanded inside
can still spill over an edge.

When a press is truncated, part of the indentation is physically outside the
image. The free-boundary integrator then has a slope running to the border and
nothing on the far side to stop it, so the height ramps. The size of that
effect is COUNTED by this module over the frames it draws from and written to
`truncation.json`; the site reads it from there. It used to be a number typed
into four separate files.

This draws examples of both, with the contact core outlined, so the criterion
is a picture rather than a sentence.

    xvfb-run -a python -m force_recovery.truncation_figure
"""
from __future__ import annotations

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "site2" / "assets" / "truncation.png"
STAT = OUT_ROOT / "feature_cache" / "truncation.json"
N = 3
SCAN = 500          # frames measured for the over-gel fraction quoted below


def build():
    """One writer at a time — see `artifact_lock` for why."""
    from .artifact_lock import one_writer
    with one_writer(STAT):
        return _build()


def _build():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import calib_free as CF
    from . import eval_panel as EP
    from .cjk_font import use_cjk
    from .debug_gallery import stages
    from .force_recon_matrix import _rows
    from .lut_calibration import GEL_THICKNESS_MM
    from .visible_eval import CORE_FACTOR, in_fov, visible

    rows, get = _rows("cnc_mini_26")
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))]

    # MEASURE the claim this figure illustrates, do not quote it. The "14.5% of
    # truncated frames reconstruct deeper than the gel" was measured once, by
    # hand, and then typed into this docstring, this figure's title, the site's
    # caption and a comment in `calib_free` — four copies, none of which would
    # notice the reconstruction changing under them. It is now counted here,
    # over the same frames the examples are drawn from, and written to
    # `truncation.json` for the site to read.
    # The gel ceiling is a MILLIMETRE claim, so it can only be asked of the
    # millimetre reconstruction. The first version of this counted
    # `CF.reconstruct(...)["depth"] > GEL_THICKNESS_MM`, but the
    # calibration-free solve carries no millimetre scale at all
    # (`calib_free.RETURNS_MILLIMETRES = False`) — its p99.8 runs 6 to 25 in
    # arbitrary units against the LUT's 1.7 to 4.25 mm — so that comparison was
    # a unit error and reported 93% of truncated frames over the gel against
    # 88% of whole ones, i.e. no contrast at all. `stages` measures the real
    # thing: `over_gel_frac` is computed BEFORE it clips to the ceiling.
    whole, trunc = [], []
    over = {True: 0, False: 0}
    seen = {True: 0, False: 0}
    for fr in rows[:SCAN]:
        img, ref = get(fr)
        ok = bool(in_fov(fr) and visible(img, ref))
        seen[ok] += 1
        over[ok] += int(stages(img, ref)["over_gel_frac"] > 0)
        if len(whole if ok else trunc) < N:
            (whole if ok else trunc).append((fr, img, ref))
    stat = {"n_scanned": SCAN,
            "n_whole": seen[True], "n_truncated": seen[False],
            "over_gel_frac_truncated": over[False] / max(seen[False], 1),
            "over_gel_frac_whole": over[True] / max(seen[True], 1),
            "gel_thickness_mm": GEL_THICKNESS_MM}
    STAT.parent.mkdir(parents=True, exist_ok=True)
    STAT.write_text(__import__("json").dumps(stat, indent=1))
    print(f"  {seen[False]} truncated / {seen[True]} whole scanned; "
          f"over the {GEL_THICKNESS_MM} mm gel: "
          f"{stat['over_gel_frac_truncated']*100:.1f}% vs "
          f"{stat['over_gel_frac_whole']*100:.1f}%", flush=True)
    sel = [(*w, True) for w in whole[:N]] + [(*t, False) for t in trunc[:N]]

    heads = ["原图 + 接触核心轮廓", "差分 dI(彩色)", "重建深度", "深度剖面"]
    use_cjk(heads + ["完整成像", "被截断(压痕有一部分在画面外)", "胶厚上限"])

    fig, ax = plt.subplots(len(sel), 4, figsize=(15, 3.2 * len(sel)),
                           constrained_layout=True)
    for i, (fr, img, ref, ok) in enumerate(sel):
        dI = img.astype(np.float32) - ref.astype(np.float32)
        mag = cv2.GaussianBlur(np.abs(dI).max(axis=2), (5, 5), 1.5)
        core = (mag > CORE_FACTOR * CF.VALID_DI).astype(np.uint8)
        d = CF.reconstruct(img, ref)["depth"]
        show = np.clip(img, 0, 255).astype(np.uint8).copy()
        cont, _ = cv2.findContours(core, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(show, cont, -1, (255, 60, 60), 2)
        row = ax[i]
        row[0].imshow(show)
        row[1].imshow(EP.diff_rgb(img, ref))
        row[2].imshow(d / max(d.max(), 1e-12), cmap="inferno")
        # a horizontal slice through the deepest row, so the ramp is visible
        r0 = int(np.argmax(d.max(axis=1)))
        row[3].plot(d[r0] / max(d.max(), 1e-12), lw=1.6,
                    color="#2a9d8f" if ok else "#e76f51")
        row[3].set_ylim(0, 1.05)
        row[3].set_xlabel("列")
        row[3].set_ylabel("相对深度")
        row[3].grid(alpha=0.25)
        for a in row[:3]:
            a.axis("off")
        row[0].text(0.03, 0.96,
                    ("完整成像" if ok else "被截断(压痕有一部分在画面外)")
                    + f"\n{fr.get('group','?')}  F={float(fr['f']):.1f} N",
                    transform=row[0].transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="#e9ffe9" if ok else "#ffe9e9",
                              alpha=0.9, edgecolor="none"))
    for a, h in zip(ax[0], heads):
        a.set_title(h, fontsize=10)
    fig.suptitle("截断帧:接触核心(红色轮廓)触到画面边缘,说明压痕有一部分在视野之外\n"
                 f"这类帧的深度不可辨识 —— 自由边界会朝画外外推,"
                 f"{stat['over_gel_frac_truncated']*100:.1f}% 的截断帧重建深度"
                 f"超过 {GEL_THICKNESS_MM} mm 胶厚"
                 f"(完整帧 {stat['over_gel_frac_whole']*100:.1f}%)",
                 fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return OUT


if __name__ == "__main__":
    build()
