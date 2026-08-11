"""Sparsh before and after the channel-order fix, at every stage.

The calibration-free solve reads each COLOUR CHANNEL as one LED direction, so
a channel permutation does not merely tint the image — it rotates the entire
recovered gradient field, and with it the surface.

Sparsh's frames reach us with R and B exchanged relative to this project's
Mini. Measured on 30 sphere presses per sensor, using the fact that a sphere's
surface gradient points radially outward so each channel's dI dipole direction
IS that channel's LED azimuth:

                     rest hue       R        G        B
      our Mini        172.1 deg   259.2     5.1     51.1
      Sparsh, as-is    42.1 deg    75.7     4.3    259.8
      Sparsh, R<->B   197.9 deg   259.8     4.3     75.7

This figure shows what that does, stage by stage: raw, signed difference,
gx, gy, and the reconstructed depth — before on top, after below, for the same
frames.

    xvfb-run -a python -m force_recovery.sparsh_channel_fix
"""
from __future__ import annotations

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "site2" / "assets" / "sparsh_channel_fix.png"
N = 3


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import calib_free as CF
    from . import eval_panel as EP
    from .cjk_font import use_cjk
    from .debug_gallery import load_sparsh

    rows, get = load_sparsh(n=300)
    sph = [r for r in rows if r["group"].startswith("sphere")]
    # The strongest presses THAT THE SENSOR IMAGES WHOLE. A random draw gave
    # 0.3 N contacts that show almost nothing; taking the strongest gave
    # presses hanging off the frame edge, where the depth panel is dominated
    # by the truncation rather than by the shape this figure is about.
    from .visible_eval import visible
    whole = [r for r in sph if visible(*get(r))]
    sel = sorted(whole, key=lambda r: -float(r["f"]))[:N]
    print(f"  {len(whole)}/{len(sph)} sphere presses imaged whole; "
          f"showing the {N} strongest", flush=True)

    heads = ["原图 raw", "差分 dI(彩色)", "gx", "gy", "重建深度"]
    use_cjk(heads + ["修复前(R/B 颠倒)", "修复后(R↔B)", "同一帧", "球压"])

    fig, ax = plt.subplots(2 * N, 5, figsize=(16.5, 6.4 * N),
                           constrained_layout=True)
    for i, fr in enumerate(sel):
        img_fixed, ref_fixed = get(fr)                       # loader now fixes
        img_raw = np.ascontiguousarray(img_fixed[..., ::-1])  # undo it
        ref_raw = np.ascontiguousarray(ref_fixed[..., ::-1])
        for j, (img, ref, tag) in enumerate(
                ((img_raw, ref_raw, "修复前(R/B 颠倒)"),
                 (img_fixed, ref_fixed, "修复后(R↔B)"))):
            r = CF.reconstruct(img, ref)
            gm = max(np.abs(r["gx"]).max(), np.abs(r["gy"]).max(), 1e-9) * 0.35
            cells = [(np.clip(img, 0, 255).astype(np.uint8), None, None),
                     (EP.diff_rgb(img, ref), None, None),
                     (r["gx"], "coolwarm", (-gm, gm)),
                     (r["gy"], "coolwarm", (-gm, gm)),
                     # a surface, not a hue ramp — see `error_analysis`
                     (EP.mesh(r["depth"], relative=True), None, None)]
            row = ax[2 * i + j]
            for a, (cell, cm, lim) in zip(row, cells):
                a.imshow(cell, cmap=cm, **({} if lim is None
                                        else {"vmin": lim[0], "vmax": lim[1]}))
                a.axis("off")
            row[0].text(0.03, 0.96, f"{tag}\n{fr['group']}  F={fr['f']:.2f} N",
                        transform=row[0].transAxes, va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none"))
    for a, h in zip(ax[0], heads):
        a.set_title(h, fontsize=11)
    fig.suptitle("Sparsh 通道序修复 · 免标定重建把每个颜色通道当作一个 LED 方向,"
                 "所以 R/B 颠倒会整体旋转梯度场\n"
                 "每帧上下两行:上=修复前,下=修复后(R↔B);gx/gy 同色标",
                 fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    plt.close(fig)
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return OUT


if __name__ == "__main__":
    build()
