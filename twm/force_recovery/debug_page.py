"""Build the bilingual step-by-step pipeline debug page for the HF Space."""
from __future__ import annotations

import shutil
from pathlib import Path

from force_recovery.results_page import CSS  # shared style
from force_recovery.run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"
GAL = OUT_ROOT / "site_assets" / "debug_gallery"
PICKS = [1, 4, 8, 11, 14, 17]          # 6 of 18, spread across force range
DSETS = [
    ("glowtact", "GlowTact — markerless, the LUT's home sensor (control)"),
    ("cnc", "FoTa cnc_Mini — markerless, foreign sensor, press grid larger "
            "than the field of view"),
    ("feats", "FEATS — dot/marker gel; these panels are the FORCE path, so the "
     "dots are still in them (marker inpainting ships for geometry only)"),
]

ABLATION = """<table><tr><th>scope (cnc, 3351 frames)</th><th>n</th>
<th>per-probe &rho; (A&ndash;F)</th><th>median &rho;</th><th>MAE</th></tr>
<tr><td>all press positions</td><td>3351</td><td>0.09&ndash;0.16</td>
<td>0.113</td><td>0.74 N</td></tr>
<tr><td>interior x&isin;[3.5,14.5]</td><td>1232</td><td>0.79&ndash;0.85</td>
<td>0.835</td><td>0.39 N</td></tr>
<tr><td><b>strictly in view x&isin;[5,13]</b></td><td>671</td>
<td><b>0.94 0.94 0.93 0.91 0.95 0.95</b></td><td><b>0.941</b></td>
<td><b>0.26 N</b></td></tr></table>"""

EN = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pipeline debug — raw image to force, step by step</title>
<style>@@CSS@@ .srow img{width:100%;max-width:1700px;background:#fff;
border-radius:6px;margin:4px 0}</style></head><body><div class="wrap">
<div class="kicker">React force recovery · pipeline debug
<span style="float:right"><a href="debug_pipeline_zh.html">中文</a></span></div>
<h1>From raw image to force, step by step</h1>
<p>Each row: raw &rarr; reference &rarr; dI=img&minus;ref &rarr; valid mask
&rarr; |LUT gradient| &rarr; Poisson depth &rarr; features and predicted vs
ground-truth force. Same pipeline, three datasets.</p>
<h2>Why cnc_Mini looked broken — and what it actually was</h2>
<p>Neither bad labels nor a broken reconstruction: the cnc press grid spans
the full 20&times;16 mm pad while the cropped camera view sees only
&asymp;13&times;10 mm, so two thirds of the presses are partially outside the
image and their contact features are systematically underestimated. Restricted
to presses fully in view, the same reconstruction reaches GlowTact-level
accuracy. Label quality is fine (force CV 4.5% at fixed probe/position/depth).
This retracts our earlier &ldquo;dataset ceiling&rdquo; argument: in-view
&rho;=0.94 exceeds the commanded-depth proxy (0.78&ndash;0.88) that argument
relied on.</p>
@@ABLATION@@
<p class="note">Per-probe half/half fit, 7 seeds, LUT pipeline with
z-supervised gain field. Dot-type control: FEATS presses are always centered,
so the raw pipeline already gives &rho;=0.72 there, vs 0.46 raw on GlowTact —
the gain field and scope, not markers, are the binding factors.</p>
<h2>The same failure mode, on four datasets</h2>
<p>The cnc story above is not a cnc story. Exact per-pixel ground truth on
Tactile MNIST shows the identical effect from the other side: 420/420 touches
have a contact that runs off the pad, and a control that moves one sphere cap
from mid-pad to the edge collapses its reconstructed peak <b>1.39 &rarr; 0.30
mm</b> against a 0.90 mm truth. Sparsh shows it a third time — its clipped
frames carry the <i>highest</i> median force and still score worse. The cause
is structural: the Poisson solve's zero boundary cannot represent a surface
that leaves the frame, so a clipped contact is integrated as if it ended at the
image edge. <b>Contact visibility, not force range or gel type, is this
pipeline's biggest external failure mode.</b></p>
<p>The same ground truth also puts a number on the mask you see in column 4:
against the true contact region it scores IoU <b>0.614</b> with recall
<b>0.917</b> and over-segmentation <b>0.531</b> — it finds nearly all of the
contact, then adds half as much again in halo. That is why a halo pedestal has
to be subtracted before any 3D view.</p>
@@GALLERIES@@
<footer>React force recovery · <a href="index.html">overview</a> ·
<a href="results.html">results</a> · <a href="method.html">method design</a>
</footer></div></body></html>"""

ZH = [
    ("<h2>The same failure mode, on four datasets</h2>",
     "<h2>同一个失效模式，出现在四个数据集上</h2>"),
    ("""<p>The cnc story above is not a cnc story. Exact per-pixel ground truth on
Tactile MNIST shows the identical effect from the other side: 420/420 touches
have a contact that runs off the pad, and a control that moves one sphere cap
from mid-pad to the edge collapses its reconstructed peak <b>1.39 &rarr; 0.30
mm</b> against a 0.90 mm truth. Sparsh shows it a third time — its clipped
frames carry the <i>highest</i> median force and still score worse. The cause
is structural: the Poisson solve's zero boundary cannot represent a surface
that leaves the frame, so a clipped contact is integrated as if it ended at the
image edge. <b>Contact visibility, not force range or gel type, is this
pipeline's biggest external failure mode.</b></p>""",
     "<p>上面这个 cnc 的故事其实不只属于 cnc。Tactile MNIST 上精确的逐像素真值"
     "从另一侧展示了完全相同的效应：420/420 次触碰的接触都跑出了 pad，而把单个球冠"
     "从 pad 中心移到边缘的对照，使重建峰值从 <b>1.39 塌到 0.30 mm</b>（真值 0.90 mm）。"
     "Sparsh 是第三次印证——它被裁切的帧力中位数<i>最高</i>，得分却更差。原因是结构性的："
     "Poisson 求解的零边界无法表示跑出画面的曲面，所以被裁切的接触会被当成在图像边缘就结束了。"
     "<b>接触可见性——而不是力程或 gel 类型——才是这条管线最大的外部失效模式。</b></p>"),
    ("""<p>The same ground truth also puts a number on the mask you see in column 4:
against the true contact region it scores IoU <b>0.614</b> with recall
<b>0.917</b> and over-segmentation <b>0.531</b> — it finds nearly all of the
contact, then adds half as much again in halo. That is why a halo pedestal has
to be subtracted before any 3D view.</p>""",
     "<p>同一份真值也给第 4 列那个掩码定了量：相对真实接触区域，它的 IoU 为 <b>0.614</b>，"
     "召回 <b>0.917</b>，过分割 <b>0.531</b>——它几乎找全了接触，然后又多加了半个接触面积的光晕。"
     "这正是任何 3D 视图之前都必须先减掉光晕基座的原因。</p>"),
    ('<html lang="en">', '<html lang="zh-CN">'),
    ("<title>Pipeline debug — raw image to force, step by step</title>",
     "<title>管线调试 — 从原始图像到力，逐步展示</title>"),
    ('pipeline debug\n<span style="float:right">'
     '<a href="debug_pipeline_zh.html">中文</a></span>',
     '管线调试\n<span style="float:right">'
     '<a href="debug_pipeline.html">English</a></span>'),
    ("<h1>From raw image to force, step by step</h1>",
     "<h1>从原始图像到力，逐步展示</h1>"),
    ("<p>Each row: raw &rarr; reference &rarr; dI=img&minus;ref &rarr; valid "
     "mask\n&rarr; |LUT gradient| &rarr; Poisson depth &rarr; features and "
     "predicted vs\nground-truth force. Same pipeline, three datasets.</p>",
     "<p>每行：原图 &rarr; 参考帧 &rarr; 差分 dI=img&minus;ref &rarr; 有效掩码 "
     "&rarr; |LUT 梯度| &rarr; Poisson 深度 &rarr; 特征与预测力 vs 真值。"
     "同一管线，三个数据集。</p>"),
    ("<h2>Why cnc_Mini looked broken — and what it actually was</h2>",
     "<h2>cnc_Mini 为何看起来失效 — 真正的原因</h2>"),
    ("Neither bad labels nor a broken reconstruction: the cnc press grid spans"
     "\nthe full 20&times;16 mm pad while the cropped camera view sees only"
     "\n&asymp;13&times;10 mm, so two thirds of the presses are partially "
     "outside the\nimage and their contact features are systematically "
     "underestimated. Restricted\nto presses fully in view, the same "
     "reconstruction reaches GlowTact-level\naccuracy. Label quality is fine "
     "(force CV 4.5% at fixed probe/position/depth).\nThis retracts our "
     "earlier &ldquo;dataset ceiling&rdquo; argument: in-view\n&rho;=0.94 "
     "exceeds the commanded-depth proxy (0.78&ndash;0.88) that argument\n"
     "relied on.",
     "既不是标签质量差，也不是重建失效：cnc 的按压网格覆盖整块 20&times;16 mm "
     "凝胶，而裁剪后的相机视野只有约 13&times;10 mm——三分之二的按压部分在画面外，"
     "接触特征被系统性低估。仅取完全在视野内的按压，同一重建管线达到 GlowTact "
     "同级精度。标签质量合格（固定探头/位置/深度下力的变异系数 4.5%）。"
     "这也撤回了我们此前的“数据集天花板”论证：视野内 &rho;=0.94 超过了该论证"
     "依赖的指令深度代理值（0.78&ndash;0.88）。"),
    ("<th>scope (cnc, 3351 frames)</th>", "<th>作用域（cnc，3351 帧）</th>"),
    ("<td>all press positions</td>", "<td>全部按压位置</td>"),
    ("<td>interior x&isin;[3.5,14.5]</td>", "<td>内部 x&isin;[3.5,14.5]</td>"),
    ("<td><b>strictly in view x&isin;[5,13]</b></td>",
     "<td><b>严格视野内 x&isin;[5,13]</b></td>"),
    ("Per-probe half/half fit, 7 seeds, LUT pipeline with\nz-supervised gain "
     "field. Dot-type control: FEATS presses are always centered,\nso the raw "
     "pipeline already gives &rho;=0.72 there, vs 0.46 raw on GlowTact —\nthe "
     "gain field and scope, not markers, are the binding factors.",
     "每探头半半分拟合，7 种子，LUT 管线 + z 监督增益场。dot 类型对照：FEATS "
     "按压恒居中，原始管线即得 &rho;=0.72，而 GlowTact 原始条件仅 0.46——"
     "起决定作用的是增益场与作用域，而非 marker。"),
    ("GlowTact — markerless, the LUT's home sensor (control)",
     "GlowTact — 无 marker，LUT 标定所用传感器（对照）"),
    ("FoTa cnc_Mini — markerless, foreign sensor, press grid larger "
     "than the field of view",
     "FoTa cnc_Mini — 无 marker，外来传感器，按压网格大于视野"),
    ("FEATS — dot/marker gel; these panels are the FORCE path, so the "
     "dots are still in them (marker inpainting ships for geometry only)",
     "FEATS — dot/marker gel；这里的面板是力的路径，所以点仍然在图里"
     "（marker 修补只用于几何）"),
]


def build():
    dst = SITE / "assets" / "debug"
    dst.mkdir(parents=True, exist_ok=True)
    gal_html = []
    for name, blurb in DSETS:
        gal_html.append(f"<h2>{blurb}</h2><div class='srow'>")
        for k in PICKS:
            src = GAL / name / f"sample_{k:02d}.png"
            tgt = dst / f"{name}_{k:02d}.png"
            shutil.copyfile(src, tgt)
            gal_html.append(f"<img src='assets/debug/{tgt.name}' "
                            f"loading='lazy'>")
        gal_html.append("</div>")
    en = (EN.replace("@@CSS@@", CSS)
            .replace("@@ABLATION@@", ABLATION)
            .replace("@@GALLERIES@@", "\n".join(gal_html)))
    (SITE / "debug_pipeline.html").write_text(en)
    zh = en
    for old, new in ZH:
        assert old in zh, f"ZH miss: {old[:60]!r}"
        zh = zh.replace(old, new, 1)
    (SITE / "debug_pipeline_zh.html").write_text(zh)
    print(SITE / "debug_pipeline.html")


if __name__ == "__main__":
    build()
