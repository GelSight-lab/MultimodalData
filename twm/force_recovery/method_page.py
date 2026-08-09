"""Build method.html / method_zh.html — a one-page method overview.

Deliberately short: the pipeline, the two validation figures, the numbers
that matter, and a related-work note answering "has anyone compared NN vs
model-based force estimation on the GelSight Mini?".
"""
from __future__ import annotations

import json
from pathlib import Path

from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"

CSS = """
:root{--paper:#0d1526;--ink:#dce7f5;--dim:#7c8db0;--grid:#1a2540;
--force:#ffb347;--target:#ff7847;--ok:#7be0a0;--card:#111b31;--line:#24345;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:'IBM Plex Sans','Noto Sans SC',sans-serif;line-height:1.65;font-size:16px;
background-image:linear-gradient(var(--grid) 1px,transparent 1px),
linear-gradient(90deg,var(--grid) 1px,transparent 1px);background-size:44px 44px}
.wrap{max-width:880px;margin:0 auto;padding:0 24px 80px}
h1,h2{font-family:'Fraunces','Noto Serif SC',serif}
h1{font-size:2.1rem;font-weight:700;line-height:1.2;margin:0 0 10px}
h2{font-size:1.25rem;font-weight:600;margin:46px 0 8px;color:var(--force)}
.kicker{font-family:'IBM Plex Mono',monospace;font-size:.76rem;letter-spacing:.16em;
text-transform:uppercase;color:var(--force);margin-bottom:14px}
header{padding:60px 0 26px;border-bottom:1px dashed var(--line)}
.sub{color:var(--dim);max-width:720px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 22px;margin:14px 0}
.step{display:grid;grid-template-columns:34px 200px 1fr;gap:12px;
padding:10px 0;border-bottom:1px solid var(--line);align-items:baseline}
.step:last-child{border-bottom:none}
.step .n{font-family:'Fraunces',serif;font-size:1.25rem;color:var(--target)}
.step .t{font-family:'IBM Plex Mono',monospace;font-size:.85rem}
.step .d{color:var(--dim);font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:10px 0;
font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}
th{color:var(--dim);font-family:'IBM Plex Mono',monospace;font-size:.72rem;
text-transform:uppercase;letter-spacing:.08em}
td:first-child,th:first-child{text-align:left}
img,video{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);
display:block;margin:12px auto}
a{color:#4fd8e0}a:hover{color:var(--force)}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;
padding:3px 12px;font-size:.75rem;font-family:'IBM Plex Mono',monospace;
color:var(--dim);margin-right:8px;margin-top:12px;text-decoration:none}
code{font-family:'IBM Plex Mono',monospace;background:#0a1120;padding:1px 5px;
border-radius:4px;font-size:.85em}
.ok{color:var(--ok)}.warn{color:#e0b34f}
footer{margin-top:60px;padding-top:18px;border-top:1px dashed var(--line);
color:var(--dim);font-size:.8rem}
@media(max-width:640px){.step{grid-template-columns:28px 1fr;grid-auto-flow:dense}
.step .d{grid-column:2}
h1{font-size:1.45rem}
table{display:block;overflow-x:auto;white-space:nowrap;-webkit-overflow-scrolling:touch}
th,td{white-space:normal;min-width:110px}}
"""

SEC_MARKER_EN = r'''<h2>Marker gels: one extra step, and what it is not for</h2>
<div class="card">
<p style="margin-top:0">FEATS is our only dotted gel. The dots occlude the gel,
so the photometric table has no valid colour underneath them and Poisson
integration bakes a dimple lattice into the depth map — visible as pockmarks
across the whole reconstructed surface. The fix is the one
<a href="https://arxiv.org/abs/2106.08851">GelSight Wedge</a>
(Wang/She/Dong/Adelson, ICRA 2021) gives in its Fig. 10 for marker holes —
detect the dots, then <i>fill</i> them rather than mask them — moved from
gradient space into image space: inpaint the dots out of <b>both</b> the
reference and the frame (cv2 Telea) and difference the inpainted pair.</p>
<table>
<tr><th>variant</th><th>dimple power at the 31.9 px pitch</th><th>force &rho;</th></tr>
<tr><td>baseline, no marker handling</td><td>1.523</td><td class="ok">0.7747</td></tr>
<tr><td><b>inpaint the image, ref + frame</b> — shipped, depth only</td>
<td class="ok"><b>0.890</b> (&times;0.65)</td><td>0.7371</td></tr>
<tr><td>inpaint the gradient field instead</td><td>0.924</td><td>0.7408</td></tr>
<tr><td>zero the gradient inside the holes</td><td class="warn">1.251</td><td>0.7620</td></tr>
</table>
<p>Two things worth reading off that table. <b>Filling beats masking</b>:
setting g&nbsp;:=&nbsp;0 on the holes barely helps (&times;0.93) because a zero
patch puts a dipole layer on every hole boundary, at exactly the lattice
frequency it is supposed to remove. And <b>the geometry column and the force
column disagree</b> — the variant that halves the dimple power is the one that
loses the most &rho;. So the step ships for <b>depth and 3D only</b>
(<code>marker_removal.stages_depth</code>, which wraps rather than edits the
force path); the force features on FEATS still come from the untouched
pipeline. Everything else on this site is markerless, where the detector finds
zero dots and the step is a bit-exact no-op.</p>
<p class="footnote">Controls, because &ldquo;inpainting helps&rdquo; is an easy
thing to fool yourself about: inpainting the same <i>area</i> of randomly
placed fake markers gives &rho; 0.7697 — no gain, so this is not smoothing.
The detector finds 63/63 dots with 0 rejects on the FEATS reference and stays
at 63 for every threshold from 3 to 16 grey levels, but exactly 0 blobs on the
cnc_mini_26 and cnc references, so it is marker-specific. It is also not a
complete fix: the dots shear with the gel (median 1.7 px, &gt;8 px on 8% of
frames) and a static reference mask leaves the displaced ones in place. Full
study: <code>python -m force_recovery.marker_study all</code>.</p>
</div>'''

SEC_MARKER_ZH = r'''<h2>有 marker 的 gel：多一步，以及它不是为了什么</h2>
<div class="card">
<p style="margin-top:0">FEATS 是我们唯一一块带点的 gel。点会遮挡 gel，
所以光度查找表在点下没有有效颜色，Poisson 积分会把一层点阵凹坑烤进深度图——
在整个重建表面上表现为麻点。修法就是
<a href="https://arxiv.org/abs/2106.08851">GelSight Wedge</a>
（Wang/She/Dong/Adelson，ICRA 2021）图 10 针对 marker 孔洞给出的做法——
检测这些点，然后<i>填充</i>而不是屏蔽——只不过从梯度域搬到了图像域：
把点从<b>参考帧和当前帧</b>里一起修补掉（cv2 Telea），再对修补后的两幅图做差分。</p>
<table>
<tr><th>方案</th><th>31.9 px 点距频率上的凹坑功率</th><th>力 &rho;</th></tr>
<tr><td>基线，不处理 marker</td><td>1.523</td><td class="ok">0.7747</td></tr>
<tr><td><b>图像域修补，参考帧 + 当前帧</b>——已上线，仅用于深度</td>
<td class="ok"><b>0.890</b>（&times;0.65）</td><td>0.7371</td></tr>
<tr><td>改为修补梯度场</td><td>0.924</td><td>0.7408</td></tr>
<tr><td>把孔洞内的梯度置零</td><td class="warn">1.251</td><td>0.7620</td></tr>
</table>
<p>这张表有两点值得读出来。<b>填充胜过屏蔽</b>：把孔洞里的 g 置零几乎没用（&times;0.93），
因为零补丁会在每个孔洞边界上放一层偶极子，频率恰好就是它本该去掉的那个点阵频率。
以及<b>几何列和力列是相反的</b>——把凹坑功率减半的那个方案，恰恰是 &rho; 掉得最多的。
所以这一步只用于<b>深度与 3D</b>（<code>marker_removal.stages_depth</code>，
它是包一层而不是改动力的路径）；FEATS 的力特征仍然来自未改动的管线。
本站其余数据都是无 marker 的，检测器在那里找不到点，这一步是逐比特的空操作。</p>
<p class="footnote">对照实验，因为&ldquo;修补有用&rdquo;是很容易自欺的结论：
把同样<i>面积</i>的随机假 marker 修补掉得到 &rho; 0.7697——没有增益，所以这不是平滑效应。
检测器在 FEATS 参考帧上得到 63/63 个点、0 个误检，阈值从 3 到 16 灰阶都保持 63；
而在 cnc_mini_26 与 cnc 参考帧上恰好是 0 个，所以它是 marker 专用的。
它也不是完全的修复：点会随 gel 剪切移动（中位 1.7 px，8% 的帧超过 8 px），
静态参考掩码会漏掉位移后的点。完整研究：
<code>python -m force_recovery.marker_study all</code>。</p>
</div>'''

SEC_GT_EN = r'''<h2>How accurate is the depth, in micrometres?</h2>
<div class="card">
<p style="margin-top:0">Every &rho; above scores <b>force</b>. The depth
underneath used to be checked only against our own analytic sphere cap. It is
now measured against exact per-pixel ground truth — ray-cast mesh depth on 420
Tactile MNIST touches, non-spherical geometry a sphere calibration cannot
self-validate — and the answer is a <b>range set by press depth</b>, not a
number:</p>
<table>
<tr><th>press depth</th><th>0.30 mm</th><th>0.60 mm</th><th>1.00 mm</th>
<th>1.50 mm</th><th>2.25 mm</th></tr>
<tr><td>MAE</td><td class="ok"><b>11 µm</b></td><td class="ok">35 µm</td>
<td>68 µm</td><td>127 µm</td><td class="warn">281 µm</td></tr>
<tr><td>peak recovered</td><td>1.00</td><td>0.97</td><td>0.77</td><td>0.68</td>
<td class="warn">0.55</td></tr>
</table>
<p>At 0.3 mm, with no per-frame alignment and no fitted indentation scale, the
Type-2 error is 96.5 µm — below all three of 3D Cal's published figures
(152.8&thinsp;/&thinsp;171.6&thinsp;/&thinsp;290.0 µm), which are reported
<i>with</i> both. By 2.25 mm we recover barely half the peak. <b>The working
range of this reconstruction is shallow contact, and no accuracy number here
should be quoted without its press depth.</b></p>
<p>The same run corrected two of our own claims and confirmed a third:
flat-top over-doming is <b>+7% to +12%</b>, not the +23&ndash;42% we implied
(compliant gel wraps a flat edge, so the true centre/rim ratio is
1.33&ndash;1.40, not 1.0); unobserved LUT bins are a <b>minor</b> factor
(correlation 0.10&ndash;0.29 with error); and the <code>|dI|&gt;8</code> valid
mask really is halo-dominated (IoU 0.614, recall 0.917, over-segmentation
0.531). The photometric table is <i>not</i> the weak link — its gradients sit
24.4&deg; from the true ones off-domain versus 26.1&deg; on its own sensor.
Full tables, controls and the Taxim caveat on the
<a href="results.html"><b>results page</b></a>.</p>
</div>'''

SEC_GT_ZH = r'''<h2>深度到底准到多少微米？</h2>
<div class="card">
<p style="margin-top:0">上面所有 &rho; 评的都是<b>力</b>。底层的深度此前只和我们自己的
解析球冠比较过。现在它已经对着精确的逐像素真值测量——420 次 Tactile MNIST 触碰的
网格光线投射深度，而且是球面标定无法自证的非球面几何——答案是一个
<b>由压入深度决定的区间</b>，而不是一个数字：</p>
<table>
<tr><th>压入深度</th><th>0.30 mm</th><th>0.60 mm</th><th>1.00 mm</th>
<th>1.50 mm</th><th>2.25 mm</th></tr>
<tr><td>MAE</td><td class="ok"><b>11 µm</b></td><td class="ok">35 µm</td>
<td>68 µm</td><td>127 µm</td><td class="warn">281 µm</td></tr>
<tr><td>峰值还原比</td><td>1.00</td><td>0.97</td><td>0.77</td><td>0.68</td>
<td class="warn">0.55</td></tr>
</table>
<p>在 0.3 mm，且没有逐帧配准、也没有拟合压入尺度的情况下，Type-2 误差为 96.5 µm——
低于 3D Cal 公布的全部三个数字（152.8&thinsp;/&thinsp;171.6&thinsp;/&thinsp;290.0 µm），
而他们的结果是<i>带</i>这两项才得到的。到 2.25 mm 我们只还原了大约一半峰值。
<b>这套重建的适用区间是浅接触，本页任何精度数字都必须连同其压入深度一起引用。</b></p>
<p>同一次实验纠正了我们自己的两个说法，并确认了第三个：平头过度穹顶化是
<b>+7% 到 +12%</b>，而不是我们暗示的 +23&ndash;42%（gel 柔顺、会包裹平边，
所以真实的中心/边缘比是 1.33&ndash;1.40，而不是 1.0）；未观测的 LUT bin 是
<b>次要</b>因素（与误差的相关性只有 0.10&ndash;0.29）；而 <code>|dI|&gt;8</code>
有效掩码确实被光晕主导（IoU 0.614、召回 0.917、过分割 0.531）。
光度查找表<i>不是</i>薄弱环节——它的梯度在域外与真值相差 24.4&deg;，
而在它自己的传感器上是 26.1&deg;。完整表格、对照与 Taxim 相关的边界说明见
<a href="results_zh.html"><b>评测结果页</b></a>。</p>
</div>'''

EN = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Method in One Page — React Force Recovery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500&display=swap" rel="stylesheet">
<style>@@CSS@@</style></head><body><div class="wrap">

<header>
<div class="kicker">React force recovery · method overview</div>
<h1>GelSight image → normal force, in one page</h1>
<p class="sub">Markerless gel, no F/T sensor, no training data from our rig.
A physics pipeline with exactly one fitted number.</p>
<a class="pill" href="results.html">↖ results matrix</a>
<a class="pill" href="index.html">overview</a>
<a class="pill" href="recon_workbench.html">3D workbench</a>
<a class="pill" href="actions.html">action transform</a>
<a class="pill" href="method_zh.html">中文</a>
</header>

<h2>The pipeline — six steps</h2>
<div class="card">
<div class="step"><span class="n">1</span><span class="t">crop 1/7 border</span>
<span class="d">the depth network was trained on the SDK's cropped view; the full frame includes LED borders it has never seen</span></div>
<div class="step"><span class="n">2</span><span class="t">RGB → surface normals</span>
<span class="d">per-pixel MLP (gsrobotics <code>nnmini</code>): three-color illumination makes color→normal invertible</span></div>
<div class="step"><span class="n">3</span><span class="t">Poisson integration</span>
<span class="d">normals → height map; subtract a per-episode zero map (median of the 15 lowest-contact frames)</span></div>
<div class="step"><span class="n">4</span><span class="t">background plane removal</span>
<span class="d">illumination drift integrates into a global tilt that can dwarf real indentation; a robust per-frame plane fit removes it</span></div>
<div class="step"><span class="n">5</span><span class="t">contact threshold</span>
<span class="d">5σ from the MAD of reference-frame residuals — per sensor, because noise varies 10–50 µm between sensors</span></div>
<div class="step"><span class="n">6</span><span class="t">volume × c → force</span>
<span class="d">Winkler foundation: F = c·Σδ·dA. The scale c is the single fitted number — from FEA ground truth, not assumed gel constants</span></div>
</div>
<p>Post-processing: a 3-tap median over <i>fresh</i> tactile frames only
(duplicated rows would let a row-wise filter count bad values three times).
Cuts single-frame spikes from 4–8% to ≈0.</p>

<h2>Photometric overhaul — classic per-sensor calibration (v2)</h2>
<div class="card">
<p style="margin-top:0">Ground-truth depth supervision (commanded press depth in
cnc_mini_26) exposed that the generic depth MLP recovers only ~25% of true
indentation and saturates (peak-depth ρ = 0.39) — and, deeper, that the
<b>gsrobotics SDK's Poisson solver returns 39% of the amplitude even on a
perfect synthetic gradient field</b> (line-integral proved the gradients were
correct at 105%). Rebuilt on the classic Dong/Yuan calibration: difference
image → per-sensor RGB lookup table, self-calibrated from cnc_mini_26's
spherical presses via the exact relation a² = d(2R−d) (R = 3.35 mm, no
external data) → exact Poisson (Dong's fast_poisson, 100.7% on the same
benchmark) → sphere-supervised spatial gain field → Drake-style stiffening
foundation p = k₁δ + k₂δ² with imprint-derived shape conditioning.</p>
<table>
<tr><th>stage (held-out, cnc_mini_26 0–20 N)</th><th>ρ</th><th>MAE</th></tr>
<tr><td>MLP + linear Winkler (v1)</td><td>0.63</td><td>4.4 N</td></tr>
<tr><td>LUT + solver fix + gain field + nonlinear foundation</td><td>0.80</td><td>2.75 N</td></tr>
<tr><td>+ imprint shape self-conditioning</td><td><b>0.82</b></td><td><b>2.46 N</b></td></tr>
<tr><td>spheres only (geometry exact — the method's ceiling)</td><td class="ok"><b>0.91–0.94</b></td><td class="ok">1.5–1.9 N</td></tr>
<tr><td><b>spheres × 0–8 N</b> (React's operating range; 7-seed median, +isotonic)</td><td class="ok"><b>0.95</b> (0.93–0.96)</td><td class="ok"><b>0.78 N</b> (0.73–0.84)</td></tr>
</table>
<p class="footnote">Ceiling context: the CNC's own commanded depth predicts force
at ρ = 0.975. The remaining pooled gap is object-dependent contact mechanics;
sub-newton MAE on 0–20 N exceeds what geometry alone carries (the 200K-frame
supervised network reaches 2.1 N on the same range). A new sensor needs one
2-minute ball-press pass — a calibration the React rig can adopt.</p>
</div>

<h2>Validation (LUT-v2 pipeline)</h2>
<div class="card">
<p style="margin-top:0">The LUT-v2 pipeline is validated on <b>four force-labeled
datasets</b> (FEATS, FoTa cnc_Mini, cnc_mini_26, Sparsh) and cross-checked against
two neural estimators on identical frames — every predicted-vs-ground-truth
scatter, per dataset, lives on the
<a href="results.html"><b>results page</b></a>. Short version: physics
0.77-0.99 everywhere (cnc_mini_26 0.99, cnc_Mini in-view 0.95, Sparsh in-view
0.97, FEATS 0.77), each reported beside a within-group label-shuffle control
that lands between &minus;0.00 and 0.26; each network 0.90+ in its own gel
domain and collapsing outside it.</p>
</div>

@@SEC_GT@@

@@SEC_MARKER@@

<h2>Per-dataset calibration</h2>
<div class="card">
<p style="margin-top:0">Worth stating plainly, because it bounds what those ρ
mean: <b>the pipeline holds no hardness or elastic-modulus constant anywhere</b>.
Nothing in the code knows a gel's Shore hardness or Young's modulus. Stiffness
enters only implicitly — absorbed into the per-group least-squares weights and
the isotonic calibration stacked on them — and those are refit <b>per dataset
and per indenter/probe group</b>: every cnc_Mini probe, every cnc_mini_26 indenter
family, every FEATS capture group gets its own fit, half the group fitting and
half held out.</p>
<p>The only physical constants shared across datasets are the sphere-calibrated
RGB lookup table and <code>MM_PER_PIXEL</code>. Everything downstream of depth
is refit. The reported ρ are therefore <b>per-group rank correlations</b> —
evidence that the geometry recovered from the image is monotone in force
<i>within</i> a group. They do <b>not</b> demonstrate a transferable
absolute-newton model across gels; the one pooled multi-object fit we tried was
much worse (ρ 0.47), which is why the sphere, with known geometry, is the
calibration object.</p>
<p>The reconstruction underneath is not the noise source it appears to be, and
that is measured, not assumed: high-frequency content in the <b>depth</b> field
is <b>0.3%</b> of peak depth (6.5 µm on a 1.9 mm press), while in the
<b>gradient</b> field it is <b>22.2%</b> — LUT bin quantisation, with 9.8% of
contact pixels landing in bins the sphere calibration never observed
(nearest-filled). Poisson integration is a low-pass, so the LUT noise is gone
by the time depth exists; the speckle visible in gradient-domain debug panels
never reaches the force features.</p>
</div>

<h2>Optimizing against ground truth</h2>
<div class="card">
<p style="margin-top:0">The cnc_Mini force labels turned the pipeline's weak spots into
measurable defects, fixed in order (each step verified on held-out data):</p>
<table>
<tr><th>step</th><th>evidence that drove it</th><th>ρ (held-out val)</th></tr>
<tr><td>baseline (volume, per-episode zeroing)</td><td>—</td><td>0.34 pooled</td></tr>
<tr><td>+ median zero map over scattered presses</td><td>only 4 of 2686 frames are truly contact-free; lowest-force references carried 1.7 N of baked-in contact</td><td>≈ same (zero map wasn't the bottleneck)</td></tr>
<tr><td>+ <b>flat-field illumination normalization</b></td><td>force-fit residuals correlated with contact position (|ρ| up to 0.6); probe×quadrant conditioning raised ρ 0.45→0.55 — the vignette modulates the depth MLP's gain</td><td>0.44 pooled</td></tr>
<tr><td>+ <b>edge filtering</b> (contact centroid &gt; 3 mm from border)</td><td>border presses sit where the vignette is steepest and imprints clip the sensor edge</td><td><b>0.65</b> (probe-median 0.64)</td></tr>
</table>
<p><b>Volume or max depth?</b> Settled empirically: the volume family wins
(vol<sup>1.5</sup>-weighted 0.65, plain volume 0.63) over max depth (0.63) and
clearly over contact area (0.35); a tiny 3-feature linear model matches ρ but
halves MAE (0.61 N). The ceiling matters too: even the CNC's own commanded
press depth only reaches ρ 0.78–0.88 <i>within</i> a probe — force at equal
depth genuinely varies with texture and position.</p>
</div>

<h2>Gallery</h2>
<div class="card">
<p style="margin-top:0">20 image samples (raw | indentation | Open3D mesh of the
reconstruction | predicted vs ground-truth force) and 10 React episode clips
(tactile | live depth | live Open3D mesh | force trace): <a href="gallery.html">browse
the full gallery</a>. The mesh panel re-renders on every fresh tactile frame.</p>
<img src="assets/gallery/cnc_08.png" alt="sample panel">
<video controls muted loop playsinline preload="metadata"
 src="assets/gallery/@@CLIP@@"></video>
</div>

<h2>NN vs model-based on the GelSight Mini — who has compared them?</h2>
<div class="card">
<p style="margin-top:0"><b>No published head-to-head that we could find.</b>
The literature runs in two camps that cite but don't benchmark each other.
Model-based: marker displacement × elasticity
(<a href="https://ieeexplore.ieee.org/document/8202149">Yuan 2017</a>),
photometric-stereo height + polynomial fit, inverse FEM
(<a href="https://arxiv.org/abs/1810.04621">GelSlim, Ma 2019</a>).
Learned, on the Mini specifically:
<a href="https://openreview.net/forum?id=dUO0QQw4FW">CANFnet</a> (F/T-labeled, normal only),
<a href="https://arxiv.org/abs/2411.03315">FEATS</a> (FEA-labeled, 3D distributions),
<a href="https://arxiv.org/abs/2410.02048">FeelAnyForce</a> (200K ATI-labeled).
Each motivates NN over physics qualitatively — FEA too slow for real time,
linear elasticity misses elastomer nonlinearity — but their reported baselines
are other <i>networks</i>, not the physics pipeline.</p>
<p>Our FEATS experiment is therefore one of the few direct data points:
the model-based pipeline reaches ρ 0.70–0.85 with <b>one</b> fitted scalar and
zero training frames, where the NNs earn sub-newton MAE in-domain but die
outside their gel (FEATS on our markerless gel: no response at all).
The trade is portability vs in-domain accuracy — and which one you need
depends on whether you can collect labels on your own sensor.</p>
</div>

<h2>Where it can still improve — and does learning help?</h2>
<div class="card">
<p>Five candidate improvements, each with a control that could have killed it.
Two survived. Reproduce with
<code>python -m force_recovery.improvement_study all</code>.</p>
<table>
<tr><th>idea</th><th>result</th><th>verdict</th></tr>
<tr><td>Extract shear-sensitive features<br>
<span class="footnote">residual correlates with shear at &rho; 0.31&ndash;0.40</span></td>
<td>feeding the <b>true</b> shear in as an oracle moves &rho; 0.9749 &rarr; 0.9757
(MAE 0.0360 &rarr; 0.0345)</td><td><b>closed</b> — the correlation was
confounded; both track force magnitude</td></tr>
<tr><td>Small neural nets on the same features</td>
<td>MLP 32&times;16 0.967 · MLP 64&times;64 0.972 · gradient boosting 0.972,
vs <b>0.974 for linear+isotonic</b></td>
<td><b>no</b> — with ~400 in-view frames per pad, flexible models overfit</td></tr>
<tr><td><b>Log features</b> (error is multiplicative)</td>
<td>&rho; 0.9735 &rarr; <b>0.9822</b>, MAE 0.0361 &rarr; <b>0.0310</b> (&minus;14%)</td>
<td><b>adopted</b></td></tr>
<tr><td>&ldquo;PINN-lite&rdquo; dimensionless contact law
<i>F/A = g(&delta;/&radic;A)</i></td>
<td>looked decisive on sphere&rarr;flat (MAE 0.426 &rarr; 0.150 N, beating an
MLP's 0.297) — then <i>F = g(volume)</i> reached <b>0.0736 N</b></td>
<td><b>rejected</b> — the gain was one monotone feature instead of five,
not the physics form</td></tr>
<tr><td><b>Match model complexity to the transfer regime</b></td>
<td>see below</td><td><b>adopted</b></td></tr>
</table>
<p>The one real structural finding: <b>a single monotone volume feature
generalises to unseen indenter shapes better than the five-feature model</b>,
while five features stay better once the shape is calibrated. On cnc_mini_26
leave-one-indenter-out — the model never sees the held-out shape — this repairs
exactly the two weak spots we reported earlier:</p>
<table>
<tr><th>held-out indenter</th><th>5-feature &rho;</th><th>volume-only &rho;</th></tr>
<tr><td>round</td><td>0.958</td><td><b>0.993</b></td></tr>
<tr><td>star</td><td>0.977</td><td><b>0.991</b></td></tr>
<tr><td>quad_small</td><td>0.989</td><td><b>0.997</b></td></tr>
<tr><td>B / quad / triangle</td><td>0.981&ndash;0.995</td><td>0.986&ndash;0.996</td></tr>
</table>
<p class="footnote">Every held-out indenter now lands at &rho; &ge; 0.986
<i>without the model ever seeing that shape</i> — a stronger claim than the
per-indenter calibrated numbers on the results page, which are fitted within
each family. The cost is absolute newtons: five features still win MAE when
the shape is known (Sparsh cross-pad 0.0372 vs 0.0443 N), so the rule is
volume-only for unseen shapes, five features once calibrated. A learned
<i>g</i> inside the physics law collapses entirely (&rho; &minus;0.249):
isotonic clips monotonically outside the training range, a network
extrapolates freely.</p>
</div>

<footer>React force recovery ·
<a href="https://huggingface.co/datasets/yxma/React">dataset</a> ·
<a href="index.html">results</a> ·
<a href="actions.html">action transform</a> ·
code: <code>twm/force_recovery/</code></footer>
</div></body></html>"""

ZH = [
    ('<h2>Where it can still improve — and does learning help?</h2>',
     '<h2>还能怎么改进——learning 有用吗？</h2>'),
    ('<p>Five candidate improvements, each with a control that could have killed it.\nTwo survived. Reproduce with\n<code>python -m force_recovery.improvement_study all</code>.</p>',
     '<p>五个候选改进，每个都配了一个足以否定它的对照。两个存活。复跑：\n<code>python -m force_recovery.improvement_study all</code>。</p>'),
    ('<tr><th>idea</th><th>result</th><th>verdict</th></tr>',
     '<tr><th>思路</th><th>结果</th><th>结论</th></tr>'),
    ('<tr><td>Extract shear-sensitive features<br>',
     '<tr><td>提取剪切敏感特征<br>'),
    ('<span class="footnote">residual correlates with shear at &rho; 0.31&ndash;0.40</span></td>',
     '<span class="footnote">残差与剪切相关 &rho; 0.31&ndash;0.40</span></td>'),
    ('<td>feeding the <b>true</b> shear in as an oracle moves &rho; 0.9749 &rarr; 0.9757\n(MAE 0.0360 &rarr; 0.0345)</td><td><b>closed</b> — the correlation was\nconfounded; both track force magnitude</td></tr>',
     '<td>把<b>真实</b>剪切作为预言机特征喂进去，&rho; 仅 0.9749 &rarr; 0.9757\n（MAE 0.0360 &rarr; 0.0345）</td><td><b>关闭</b>——相关性是混杂的，\n两者都随力的大小变化</td></tr>'),
    ('<tr><td>Small neural nets on the same features</td>',
     '<tr><td>在同样特征上用小神经网络</td>'),
    ('<td>MLP 32&times;16 0.967 · MLP 64&times;64 0.972 · gradient boosting 0.972,\nvs <b>0.974 for linear+isotonic</b></td>\n<td><b>no</b> — with ~400 in-view frames per pad, flexible models overfit</td></tr>',
     '<td>MLP 32&times;16 0.967 · MLP 64&times;64 0.972 · 梯度提升 0.972，\n而<b>线性+isotonic 为 0.974</b></td>\n<td><b>无用</b>——每块 pad 仅约 400 帧视野内数据，柔性模型过拟合</td></tr>'),
    ('<tr><td><b>Log features</b> (error is multiplicative)</td>',
     '<tr><td><b>对数特征</b>（误差是乘性的）</td>'),
    ('<td>&rho; 0.9735 &rarr; <b>0.9822</b>, MAE 0.0361 &rarr; <b>0.0310</b> (&minus;14%)</td>\n<td><b>adopted</b></td></tr>',
     '<td>&rho; 0.9735 &rarr; <b>0.9822</b>，MAE 0.0361 &rarr; <b>0.0310</b>（&minus;14%）</td>\n<td><b>采纳</b></td></tr>'),
    ('<tr><td>&ldquo;PINN-lite&rdquo; dimensionless contact law\n<i>F/A = g(&delta;/&radic;A)</i></td>',
     '<tr><td>&ldquo;PINN-lite&rdquo; 量纲无关接触律\n<i>F/A = g(&delta;/&radic;A)</i></td>'),
    ("<td>looked decisive on sphere&rarr;flat (MAE 0.426 &rarr; 0.150 N, beating an\nMLP's 0.297) — then <i>F = g(volume)</i> reached <b>0.0736 N</b></td>\n<td><b>rejected</b> — the gain was one monotone feature instead of five,\nnot the physics form</td></tr>",
     '<td>在 球&rarr;平 上看似决定性（MAE 0.426 &rarr; 0.150 N，胜过 MLP 的 0.297）——\n但 <i>F = g(体积)</i> 达到 <b>0.0736 N</b></td>\n<td><b>否决</b>——增益来自&ldquo;用一个单调特征而非五个&rdquo;，\n不是那个物理形式</td></tr>'),
    ('<tr><td><b>Match model complexity to the transfer regime</b></td>\n<td>see below</td><td><b>adopted</b></td></tr>',
     '<tr><td><b>让模型复杂度匹配迁移场景</b></td>\n<td>见下</td><td><b>采纳</b></td></tr>'),
    ('<p>The one real structural finding: <b>a single monotone volume feature\ngeneralises to unseen indenter shapes better than the five-feature model</b>,\nwhile five features stay better once the shape is calibrated. On cnc_mini_26\nleave-one-indenter-out — the model never sees the held-out shape — this repairs\nexactly the two weak spots we reported earlier:</p>',
     '<p>唯一真正的结构性发现：<b>单一单调的体积特征在未见过的压头形状上比五特征模型\n泛化得更好</b>，而一旦该形状已被标定，五特征仍然更优。在 cnc_mini_26 留一压头交叉验证中\n（模型从未见过被留出的形状），这恰好修复了我们此前报告的两个短板：</p>'),
    ('<tr><th>held-out indenter</th><th>5-feature &rho;</th><th>volume-only &rho;</th></tr>',
     '<tr><th>留出的压头</th><th>五特征 &rho;</th><th>仅体积 &rho;</th></tr>'),
    ('<p class="footnote">Every held-out indenter now lands at &rho; &ge; 0.986\n<i>without the model ever seeing that shape</i> — a stronger claim than the\nper-indenter calibrated numbers on the results page, which are fitted within\neach family. The cost is absolute newtons: five features still win MAE when\nthe shape is known (Sparsh cross-pad 0.0372 vs 0.0443 N), so the rule is\nvolume-only for unseen shapes, five features once calibrated. A learned\n<i>g</i> inside the physics law collapses entirely (&rho; &minus;0.249):\nisotonic clips monotonically outside the training range, a network\nextrapolates freely.</p>',
     '<p class="footnote">现在每个被留出的压头都达到 &rho; &ge; 0.986，<i>而模型从未见过该形状</i>——\n这比结果页上&ldquo;逐压头标定&rdquo;的数字是更强的论断（后者是在各家族内部拟合的）。\n代价在绝对牛顿：形状已知时五特征的 MAE 仍更优（Sparsh 跨 pad 0.0372 vs 0.0443 N），\n所以规则是：未见形状用仅体积，已标定则用五特征。把物理律里的 <i>g</i> 换成学习模型会完全崩溃\n（&rho; &minus;0.249）：isotonic 在训练范围外单调截断，而网络会自由外推。</p>'),
    ('<h2>Photometric overhaul — classic per-sensor calibration (v2)</h2>',
     '<h2>光度重做——经典逐传感器标定(v2)</h2>'),
    ('<h2>Validation (LUT-v2 pipeline)</h2>',
     '<h2>验证(LUT-v2 管线)</h2>'),
    ('<a class="pill" href="results.html">↖ results matrix</a>\n<a class="pill" href="index.html">overview</a>',
     '<a class="pill" href="results_zh.html">↖ 评测结果</a>\n<a class="pill" href="index.html">总览(英文)</a>'),
    ('<p style="margin-top:0">The LUT-v2 pipeline is validated on <b>four force-labeled\ndatasets</b> (FEATS, FoTa cnc_Mini, cnc_mini_26, Sparsh) and cross-checked against\ntwo neural estimators on identical frames — every predicted-vs-ground-truth\nscatter, per dataset, lives on the\n<a href="results.html"><b>results page</b></a>. Short version: physics\n0.77-0.99 everywhere (cnc_mini_26 0.99, cnc_Mini in-view 0.95, Sparsh in-view\n0.97, FEATS 0.77), each reported beside a within-group label-shuffle control\nthat lands between &minus;0.00 and 0.26; each network 0.90+ in its own gel\ndomain and collapsing outside it.</p>',
     '<p style="margin-top:0">LUT-v2 管线在<b>四个带力标注的数据集</b>(FEATS、FoTa cnc_Mini、cnc_mini_26、Sparsh)上验证,并与两个神经网络估计器同帧对比——每个数据集的预测 vs 真值散点图都在<a href="results_zh.html"><b>评测结果页</b></a>。一句话版:物理方法在所有域 0.77–0.99(cnc_mini_26 0.99、cnc_Mini 视野内 0.95、Sparsh 视野内 0.97、FEATS 0.77),每个数字旁边都给出组内标签打乱对照(落在 &minus;0.00 到 0.26 之间);每个网络在自己的 gel 域 0.90+,出域即塌。</p>'),
    ('<html lang="en">',
     '<html lang="zh-CN">'),
    ('<title>Method in One Page — React Force Recovery</title>',
     '<title>方法一页纸 — React 力恢复</title>'),
    ('<div class="kicker">React force recovery · method overview</div>',
     '<div class="kicker">React 力恢复 · 方法简介</div>'),
    ('<h1>GelSight image → normal force, in one page</h1>',
     '<h1>从 GelSight 图像到法向力,一页讲完</h1>'),
    ('<p class="sub">Markerless gel, no F/T sensor, no training data from our rig.\nA physics pipeline with exactly one fitted number.</p>',
     '<p class="sub">无 marker gel,无 F/T 传感器,没有一帧来自我们平台的训练数据。\n一条只有一个拟合参数的物理管线。</p>'),
    ('<a class="pill" href="actions.html">action transform</a>\n<a class="pill" href="method_zh.html">中文</a>',
     '<a class="pill" href="actions_zh.html">动作变换</a>\n<a class="pill" href="method.html">English</a>'),
    ('<h2>The pipeline — six steps</h2>',
     '<h2>管线——六步</h2>'),
    ('<span class="t">crop 1/7 border</span>',
     '<span class="t">裁掉 1/7 边缘</span>'),
    ('<span class="d">the depth network was trained on the SDK\'s cropped view; the full frame includes LED borders it has never seen</span>',
     '<span class="d">深度网络在 SDK 的裁剪视野上训练;整幅画面含它从未见过的 LED 边缘</span>'),
    ('<span class="t">RGB → surface normals</span>',
     '<span class="t">RGB → 表面法向</span>'),
    ('<span class="d">per-pixel MLP (gsrobotics <code>nnmini</code>): three-color illumination makes color→normal invertible</span>',
     '<span class="d">逐像素 MLP(gsrobotics <code>nnmini</code>):三色照明使 颜色→法向 可逆</span>'),
    ('<span class="t">Poisson integration</span>',
     '<span class="t">Poisson 积分</span>'),
    ('<span class="d">normals → height map; subtract a per-episode zero map (median of the 15 lowest-contact frames)</span>',
     '<span class="d">法向 → 高度图;减去逐 episode 零图(15 个最低接触帧的中位数)</span>'),
    ('<span class="t">background plane removal</span>',
     '<span class="t">背景平面移除</span>'),
    ('<span class="d">illumination drift integrates into a global tilt that can dwarf real indentation; a robust per-frame plane fit removes it</span>',
     '<span class="d">照明漂移积分后成为全局倾斜,可淹没真实压入;逐帧稳健平面拟合将其去除</span>'),
    ('<span class="t">contact threshold</span>',
     '<span class="t">接触阈值</span>'),
    ('<span class="d">5σ from the MAD of reference-frame residuals — per sensor, because noise varies 10–50 µm between sensors</span>',
     '<span class="d">参考帧残差 MAD 的 5σ——逐传感器,因为噪声在 10–50 µm 间浮动</span>'),
    ('<span class="t">volume × c → force</span>',
     '<span class="t">体积 × c → 力</span>'),
    ('<span class="d">Winkler foundation: F = c·Σδ·dA. The scale c is the single fitted number — from FEA ground truth, not assumed gel constants</span>',
     '<span class="d">Winkler 弹性地基:F = c·Σδ·dA。刻度 c 是唯一拟合量——来自 FEA 真值,而非假设的 gel 常数</span>'),
    ('<p>Post-processing: a 3-tap median over <i>fresh</i> tactile frames only\n(duplicated rows would let a row-wise filter count bad values three times).\nCuts single-frame spikes from 4–8% to ≈0.</p>',
     '<p>后处理:只在 <i>fresh</i> 触觉帧上做 3 点中值(逐行滤波会把重复行里的坏值数三次)。\n单帧尖峰从 4–8% 降到 ≈0。</p>'),
    ('<h2>Per-dataset calibration</h2>',
     '<h2>逐数据集标定</h2>'),
    ('<p style="margin-top:0">Worth stating plainly, because it bounds what those ρ\nmean: <b>the pipeline holds no hardness or elastic-modulus constant anywhere</b>.\nNothing in the code knows a gel\'s Shore hardness or Young\'s modulus. Stiffness\nenters only implicitly — absorbed into the per-group least-squares weights and\nthe isotonic calibration stacked on them — and those are refit <b>per dataset\nand per indenter/probe group</b>: every cnc_Mini probe, every cnc_mini_26 indenter\nfamily, every FEATS capture group gets its own fit, half the group fitting and\nhalf held out.</p>',
     '<p style="margin-top:0">这一点值得直说,因为它界定了上面那些 ρ 的含义:<b>管线里没有任何硬度或弹性模量常数</b>。代码不知道任何 gel 的邵氏硬度或杨氏模量。刚度只是隐式进入——被逐组最小二乘的权重、以及叠在其上的保序(isotonic)标定吸收掉了——而这些都是<b>逐数据集、逐压头/探头组</b>重新拟合的:每个 cnc_Mini 探头、每个 cnc_mini_26 压头族、每个 FEATS 采集组各自拟合,组内一半拟合、一半留出。</p>'),
    ('<p>The only physical constants shared across datasets are the sphere-calibrated\nRGB lookup table and <code>MM_PER_PIXEL</code>. Everything downstream of depth\nis refit. The reported ρ are therefore <b>per-group rank correlations</b> —\nevidence that the geometry recovered from the image is monotone in force\n<i>within</i> a group. They do <b>not</b> demonstrate a transferable\nabsolute-newton model across gels; the one pooled multi-object fit we tried was\nmuch worse (ρ 0.47), which is why the sphere, with known geometry, is the\ncalibration object.</p>',
     '<p>跨数据集共享的物理常数只有两个:球面标定出的 RGB 查找表,以及 <code>MM_PER_PIXEL</code>。深度之后的一切都会重新拟合。因此报告的 ρ 是<b>逐组的秩相关</b>——它说明从图像恢复的几何在组<i>内</i>与力单调相关,但<b>不</b>说明存在一个可跨 gel 迁移的绝对牛顿值模型;我们唯一试过的多物体混合拟合差得多(ρ 0.47),这正是选用几何已知的球体作标定物的原因。</p>'),
    ('<p>The reconstruction underneath is not the noise source it appears to be, and\nthat is measured, not assumed: high-frequency content in the <b>depth</b> field\nis <b>0.3%</b> of peak depth (6.5 µm on a 1.9 mm press), while in the\n<b>gradient</b> field it is <b>22.2%</b> — LUT bin quantisation, with 9.8% of\ncontact pixels landing in bins the sphere calibration never observed\n(nearest-filled). Poisson integration is a low-pass, so the LUT noise is gone\nby the time depth exists; the speckle visible in gradient-domain debug panels\nnever reaches the force features.</p>',
     '<p>下面的重建也不是它看上去的那个噪声源,而且这是量出来的、不是假设的:<b>深度</b>场里的高频成分只占峰值深度的 <b>0.3%</b>(1.9 mm 按压上是 6.5 µm),而<b>梯度</b>场里是 <b>22.2%</b>——来自 LUT 分箱量化,9.8% 的接触像素落在球面标定从未观测到的 bin 里(用最近邻填充)。Poisson 积分本身就是低通,所以到深度这一步 LUT 噪声已经没了;梯度域调试面板里看到的麻点根本进不到力特征里。</p>'),
    ('<h2>Optimizing against ground truth</h2>',
     '<h2>用真值驱动的优化</h2>'),
    ('<p style="margin-top:0">The cnc_Mini force labels turned the pipeline\'s weak spots into\nmeasurable defects, fixed in order (each step verified on held-out data):</p>',
     '<p style="margin-top:0">cnc_Mini 的力标签把管线的弱点变成了可测量的缺陷,依次修复\n(每一步都在留出集上验证):</p>'),
    ('<tr><th>step</th><th>evidence that drove it</th><th>ρ (held-out val)</th></tr>',
     '<tr><th>步骤</th><th>驱动它的证据</th><th>ρ(留出 val)</th></tr>'),
    ('<td>baseline (volume, per-episode zeroing)</td><td>—</td><td>0.34 pooled</td>',
     '<td>基线(体积,逐集零图)</td><td>—</td><td>0.34 混合</td>'),
    ("<td>+ median zero map over scattered presses</td><td>only 4 of 2686 frames are truly contact-free; lowest-force references carried 1.7 N of baked-in contact</td><td>≈ same (zero map wasn't the bottleneck)</td>",
     '<td>+ 散布按压的中位数零图</td><td>2686 帧里只有 4 帧真无接触;最低力参考帧带着 1.7 N 的既有接触</td><td>≈ 不变(零图不是瓶颈)</td>'),
    ("<td>+ <b>flat-field illumination normalization</b></td><td>force-fit residuals correlated with contact position (|ρ| up to 0.6); probe×quadrant conditioning raised ρ 0.45→0.55 — the vignette modulates the depth MLP's gain</td><td>0.44 pooled</td>",
     '<td>+ <b>照明平场归一</b></td><td>力拟合残差与接触位置相关(|ρ| 达 0.6);按探头×象限条件化 ρ 0.45→0.55——渐晕在调制深度 MLP 的增益</td><td>0.44 混合</td>'),
    ('<td>+ <b>edge filtering</b> (contact centroid &gt; 3 mm from border)</td><td>border presses sit where the vignette is steepest and imprints clip the sensor edge</td><td><b>0.65</b> (probe-median 0.64)</td>',
     '<td>+ <b>边缘过滤</b>(接触质心距边 &gt; 3 mm)</td><td>贴边按压处渐晕最陡,印痕还会出传感器边界</td><td><b>0.65</b>(探头中位 0.64)</td>'),
    ("<p><b>Volume or max depth?</b> Settled empirically: the volume family wins\n(vol<sup>1.5</sup>-weighted 0.65, plain volume 0.63) over max depth (0.63) and\nclearly over contact area (0.35); a tiny 3-feature linear model matches ρ but\nhalves MAE (0.61 N). The ceiling matters too: even the CNC's own commanded\npress depth only reaches ρ 0.78–0.88 <i>within</i> a probe — force at equal\ndepth genuinely varies with texture and position.</p>",
     '<p><b>体积还是最大深度?</b>实证裁决:体积家族胜出(δ<sup>1.5</sup> 加权 0.65、\n纯体积 0.63)优于最大深度(0.63),明显优于接触面积(0.35);3 特征小线性模型 ρ 持平\n但 MAE 减半(0.61 N)。天花板也要认识:连 CNC 自己的指令压深在<i>探头内</i>也只有\nρ 0.78–0.88——同样深度下力确实随纹理和位置变化。</p>'),
    ('<h2>Gallery</h2>',
     '<h2>图库</h2>'),
    ('<p style="margin-top:0">20 image samples (raw | indentation | Open3D mesh of the\nreconstruction | predicted vs ground-truth force) and 10 React episode clips\n(tactile | live depth | live Open3D mesh | force trace): <a href="gallery.html">browse\nthe full gallery</a>. The mesh panel re-renders on every fresh tactile frame.</p>',
     '<p style="margin-top:0">20 个图像样本(原图 | 压入图 | 重建的 Open3D 网格 | 预测 vs 真值力)\n与 10 个 React 片段(触觉 | 实时深度 | 实时 Open3D 网格 | 力曲线):<a href="gallery.html">浏览完整图库</a>。网格面板在每一个新触觉帧上重新渲染。</p>'),
    ('<h2>NN vs model-based on the GelSight Mini — who has compared them?</h2>',
     '<h2>GelSight Mini 上 NN 与 model-based 谁对比过?</h2>'),
    ('<p style="margin-top:0"><b>No published head-to-head that we could find.</b>\nThe literature runs in two camps that cite but don\'t benchmark each other.\nModel-based: marker displacement × elasticity\n(<a href="https://ieeexplore.ieee.org/document/8202149">Yuan 2017</a>),\nphotometric-stereo height + polynomial fit, inverse FEM\n(<a href="https://arxiv.org/abs/1810.04621">GelSlim, Ma 2019</a>).\nLearned, on the Mini specifically:\n<a href="https://openreview.net/forum?id=dUO0QQw4FW">CANFnet</a> (F/T-labeled, normal only),\n<a href="https://arxiv.org/abs/2411.03315">FEATS</a> (FEA-labeled, 3D distributions),\n<a href="https://arxiv.org/abs/2410.02048">FeelAnyForce</a> (200K ATI-labeled).\nEach motivates NN over physics qualitatively — FEA too slow for real time,\nlinear elasticity misses elastomer nonlinearity — but their reported baselines\nare other <i>networks</i>, not the physics pipeline.</p>',
     '<p style="margin-top:0"><b>没有找到公开发表的正面对比。</b>\n文献分成互相引用但不互相 benchmark 的两个阵营。\nModel-based:marker 位移 × 弹性理论\n(<a href="https://ieeexplore.ieee.org/document/8202149">Yuan 2017</a>)、\n光度立体高度图 + 多项式拟合、逆有限元\n(<a href="https://arxiv.org/abs/1810.04621">GelSlim, Ma 2019</a>)。\n学习类(专门在 Mini 上):\n<a href="https://openreview.net/forum?id=dUO0QQw4FW">CANFnet</a>(F/T 标注,仅法向)、\n<a href="https://arxiv.org/abs/2411.03315">FEATS</a>(FEA 标注,3D 分布)、\n<a href="https://arxiv.org/abs/2410.02048">FeelAnyForce</a>(20 万 ATI 标注)。\n它们都只定性地论证 NN 优于物理方法——FEA 太慢、线性弹性抓不住弹性体非线性——\n但论文里对比的 baseline 全是其他<i>网络</i>,没有物理管线。</p>'),
    ('<p>Our FEATS experiment is therefore one of the few direct data points:\nthe model-based pipeline reaches ρ 0.70–0.85 with <b>one</b> fitted scalar and\nzero training frames, where the NNs earn sub-newton MAE in-domain but die\noutside their gel (FEATS on our markerless gel: no response at all).\nThe trade is portability vs in-domain accuracy — and which one you need\ndepends on whether you can collect labels on your own sensor.</p>',
     '<p>因此我们的 FEATS 实验算是少数直接对比数据点之一:物理管线只用<b>一个</b>拟合标量、\n零训练帧,达到 ρ 0.70–0.85;NN 在域内能到亚牛顿 MAE,但出了自己的 gel 就失效\n(FEATS 在我们的无点 gel 上完全无响应)。这是可移植性与域内精度的取舍——\n选哪个取决于你能否在自己的传感器上采标注。</p>'),
    ('<footer>React force recovery ·\n<a href="https://huggingface.co/datasets/yxma/React">dataset</a> ·\n<a href="index.html">results</a> ·\n<a href="actions.html">action transform</a> ·\ncode: <code>twm/force_recovery/</code></footer>',
     '<footer>React 力恢复 ·\n<a href="https://huggingface.co/datasets/yxma/React">数据集</a> ·\n<a href="index.html">完整结果</a> ·\n<a href="actions_zh.html">动作变换</a> ·\n代码:<code>twm/force_recovery/</code></footer>'),
]



def build_gallery() -> Path:
    """gallery.html — every sample panel and clip, one scrolling page."""
    g = SITE / "assets" / "gallery"
    imgs = sorted(p.name for p in g.glob("*.png"))
    vids = sorted(p.name for p in g.glob("*.mp4"))
    items = "\n".join(
        f'<figure><img loading="lazy" src="assets/gallery/{n}">'
        f"<figcaption>{n}</figcaption></figure>" for n in imgs)
    clips = "\n".join(
        f'<figure><video controls muted loop playsinline preload="none" '
        f'src="assets/gallery/{n}"></video><figcaption>{n}</figcaption></figure>'
        for n in vids)
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gallery — React Force Recovery</title>
<style>@@CSS@@
figure{{margin:14px 0}} figcaption{{color:var(--dim);font-size:.72rem;
font-family:'IBM Plex Mono',monospace;margin-top:2px}}</style></head>
<body><div class="wrap">
<header><div class="kicker">React force recovery · gallery</div>
<h1>Samples: every stage from raw frame to force</h1>
<p>raw &rarr; reference &rarr; difference image &rarr; valid mask &rarr; LUT surface
gradient &rarr; depth &rarr; Open3D mesh &rarr; predicted vs ground truth.</p>
<a class="pill" href="method.html">↖ method</a>
<a class="pill" href="index.html">results</a></header>
<h2>{len(imgs)} image samples (cnc_Mini + FEATS, with ground truth)</h2>
{items}
<h2>{len(vids)} video samples (React episodes)</h2>
{clips}
</div></body></html>""".replace("@@CSS@@", CSS)
    out = SITE / "gallery.html"
    out.write_text(page)
    return out


def build() -> tuple[Path, Path]:
    clips = sorted(p.name for p in (SITE / "assets" / "gallery").glob("*.mp4"))
    assert clips, "no gallery clips found — run showcase.video_samples first"
    en = (EN.replace("@@CSS@@", CSS).replace("@@CLIP@@", clips[0])
            .replace("@@SEC_MARKER@@", SEC_MARKER_EN)
            .replace("@@SEC_GT@@", SEC_GT_EN))
    # new sections are single-sourced: the ZH pair matches the same constant
    # by construction, so a long block cannot drift between the languages
    pairs = ZH + [(SEC_MARKER_EN, SEC_MARKER_ZH), (SEC_GT_EN, SEC_GT_ZH),
                  ('<a class="pill" href="recon_workbench.html">3D workbench</a>',
                   '<a class="pill" href="recon_workbench.html">3D 工作台</a>')]
    zh = en
    for old, new in pairs:
        assert old in zh, f"ZH replacement missed: {old[:60]!r}"
        zh = zh.replace(old, new, 1)
    for name, page in (("method.html", en), ("method_zh.html", zh)):
        assert "@@" not in page, f"{name}: unreplaced token"
        (SITE / name).write_text(page)
    return SITE / "method.html", SITE / "method_zh.html"


if __name__ == "__main__":
    for p in build():
        print(p)
    print(build_gallery())
