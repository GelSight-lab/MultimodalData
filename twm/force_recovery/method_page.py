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
GlowTact) exposed that the generic depth MLP recovers only ~25% of true
indentation and saturates (peak-depth ρ = 0.39) — and, deeper, that the
<b>gsrobotics SDK's Poisson solver returns 39% of the amplitude even on a
perfect synthetic gradient field</b> (line-integral proved the gradients were
correct at 105%). Rebuilt on the classic Dong/Yuan calibration: difference
image → per-sensor RGB lookup table, self-calibrated from GlowTact's
spherical presses via the exact relation a² = d(2R−d) (R = 3.35 mm, no
external data) → exact Poisson (Dong's fast_poisson, 100.7% on the same
benchmark) → sphere-supervised spatial gain field → Drake-style stiffening
foundation p = k₁δ + k₂δ² with imprint-derived shape conditioning.</p>
<table>
<tr><th>stage (held-out, GlowTact 0–20 N)</th><th>ρ</th><th>MAE</th></tr>
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
<p style="margin-top:0">The LUT-v2 pipeline is validated on <b>three force-labeled
datasets</b> (FEATS, FoTa cnc_Mini, GlowTact) and cross-checked against two
neural estimators on identical frames — every predicted-vs-ground-truth
scatter, per dataset, lives on the
<a href="results.html"><b>results page</b></a>. Short version: physics
0.74-0.99 everywhere (GlowTact 0.98, cnc_Mini in-view 0.94, FEATS 0.77);
each network 0.90+ in its own gel domain and collapsing
outside it.</p>
</div>

<h2>Per-dataset calibration</h2>
<div class="card">
<p style="margin-top:0">Worth stating plainly, because it bounds what those ρ
mean: <b>the pipeline holds no hardness or elastic-modulus constant anywhere</b>.
Nothing in the code knows a gel's Shore hardness or Young's modulus. Stiffness
enters only implicitly — absorbed into the per-group least-squares weights and
the isotonic calibration stacked on them — and those are refit <b>per dataset
and per indenter/probe group</b>: every cnc_Mini probe, every GlowTact indenter
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

<footer>React force recovery ·
<a href="https://huggingface.co/datasets/yxma/React">dataset</a> ·
<a href="index.html">results</a> ·
<a href="actions.html">action transform</a> ·
code: <code>twm/force_recovery/</code></footer>
</div></body></html>"""

ZH = [
    ('<h2>Photometric overhaul — classic per-sensor calibration (v2)</h2>',
     '<h2>光度重做——经典逐传感器标定(v2)</h2>'),
    ('<h2>Validation (LUT-v2 pipeline)</h2>',
     '<h2>验证(LUT-v2 管线)</h2>'),
    ('<a class="pill" href="results.html">↖ results matrix</a>\n<a class="pill" href="index.html">overview</a>',
     '<a class="pill" href="results_zh.html">↖ 评测结果</a>\n<a class="pill" href="index.html">总览(英文)</a>'),
    ('<p style="margin-top:0">The LUT-v2 pipeline is validated on <b>three force-labeled\ndatasets</b> (FEATS, FoTa cnc_Mini, GlowTact) and cross-checked against two\nneural estimators on identical frames — every predicted-vs-ground-truth\nscatter, per dataset, lives on the\n<a href="results.html"><b>results page</b></a>. Short version: physics\n0.74-0.99 everywhere (GlowTact 0.98, cnc_Mini in-view 0.94, FEATS 0.77);\neach network 0.90+ in its own gel domain and collapsing\noutside it.</p>',
     '<p style="margin-top:0">LUT-v2 管线在<b>三个带力标注的数据集</b>(FEATS、FoTa cnc_Mini、GlowTact)上验证,并与两个神经网络估计器同帧对比——每个数据集的预测 vs 真值散点图都在<a href="results_zh.html"><b>评测结果页</b></a>。一句话版:物理方法在所有域 0.74–0.99(GlowTact 0.98、cnc_Mini 视野内 0.94、FEATS 0.77);每个网络在自己的 gel 域 0.90+,出域即塌。</p>'),
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
    ('<p style="margin-top:0">Worth stating plainly, because it bounds what those ρ\nmean: <b>the pipeline holds no hardness or elastic-modulus constant anywhere</b>.\nNothing in the code knows a gel\'s Shore hardness or Young\'s modulus. Stiffness\nenters only implicitly — absorbed into the per-group least-squares weights and\nthe isotonic calibration stacked on them — and those are refit <b>per dataset\nand per indenter/probe group</b>: every cnc_Mini probe, every GlowTact indenter\nfamily, every FEATS capture group gets its own fit, half the group fitting and\nhalf held out.</p>',
     '<p style="margin-top:0">这一点值得直说,因为它界定了上面那些 ρ 的含义:<b>管线里没有任何硬度或弹性模量常数</b>。代码不知道任何 gel 的邵氏硬度或杨氏模量。刚度只是隐式进入——被逐组最小二乘的权重、以及叠在其上的保序(isotonic)标定吸收掉了——而这些都是<b>逐数据集、逐压头/探头组</b>重新拟合的:每个 cnc_Mini 探头、每个 GlowTact 压头族、每个 FEATS 采集组各自拟合,组内一半拟合、一半留出。</p>'),
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
<h1>Samples: raw | depth | Open3D mesh | force</h1>
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
    en = EN.replace("@@CSS@@", CSS).replace("@@CLIP@@", clips[0])
    (SITE / "method.html").write_text(en)
    zh = en
    for old, new in ZH:
        assert old in zh, f"ZH replacement missed: {old[:60]!r}"
        zh = zh.replace(old, new, 1)
    (SITE / "method_zh.html").write_text(zh)
    return SITE / "method.html", SITE / "method_zh.html"


if __name__ == "__main__":
    for p in build():
        print(p)
    print(build_gallery())
