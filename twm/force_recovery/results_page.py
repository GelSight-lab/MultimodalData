"""Build results.html / results_zh.html — the method x dataset matrix.

One section per ground-truth dataset, each with a predicted-vs-GT scatter row
showing all three estimators under identical conditions — dataset quality is
controlled within a row, so differences between panels are differences
between methods, not between datasets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"
CACHE = OUT_ROOT / "feature_cache"

ORANGE, PURPLE, GREEN = "#d95f02", "#7570b3", "#1b9e77"


def _load_ours_cnc():
    va = json.loads((OUT_ROOT / "feature_cache_ff" / "cnc_mini_val.json").read_text())["rows"]
    tr = json.loads((OUT_ROOT / "feature_cache_ff" / "cnc_mini_train.json").read_text())["rows"]
    s = np.median([r["force_true"] / r["vol15"] for r in tr
                   if r["vol15"] > 0.2 and r["force_true"] > 0.5])
    t = np.array([r["force_true"] for r in va])
    return t, np.array([r["vol15"] for r in va]) * s


def _load_ours_glowtact():
    gv = json.loads((OUT_ROOT / "glowtact_validation.json").read_text())
    rows = gv["per_frame"]
    t = np.array([r["force"] for r in rows])
    return t, np.array([r["vol15"] for r in rows]) * gv["scale_n_per_mm3"]


def _load_ours_feats():
    rep = json.loads((OUT_ROOT / "feats_validation_val.json").read_text())
    rows = rep["per_frame"]
    t = np.array([r["f_true"] for r in rows])
    return t, np.array([r["volume_mm3"] for r in rows]) * rep["scale_n_per_mm3"]


def _load_json_pred(name):
    rows = json.loads((CACHE / name).read_text())
    t = np.array([r["force_true"] for r in rows])
    p = np.array([r.get("pred", r.get("fz")) for r in rows])
    if "fz" in rows[0] and "pred" not in rows[0]:
        p = -p
    return t, p


def dataset_figures() -> dict:
    """One 3-panel scatter figure per GT dataset + the React agreement plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10, "axes.grid": True,
        "grid.alpha": .18, "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "axes.titlesize": 10,
        "figure.titlesize": 12, "figure.titleweight": "bold",
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})

    out_dir = SITE / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures, metrics = {}, {}

    datasets = {
        "feats": {
            "title": "Marker-dot gel: only the in-domain network is precise — "
                     "physics transfers, the markerless network degrades",
            "panels": [
                ("Ours (physics)", _load_ours_feats, ORANGE, "transfer"),
                ("FEATS U-net", lambda: _load_json_pred("feats_on_feats_val.json"),
                 PURPLE, "IN-DOMAIN"),
                ("FeelAnyForce", lambda: _load_json_pred("anyforce_on_feats_val.json"),
                 GREEN, "transfer (marker gel)"),
            ]},
        "cnc": {
            "title": "Markerless gel (cnc_Mini): the marker-gel network goes blind — "
                     "physics and FeelAnyForce both track force",
            "panels": [
                ("Ours (physics)", _load_ours_cnc, ORANGE, "transfer"),
                ("FEATS U-net",
                 lambda: _load_json_pred_cnc_feats(), PURPLE, "transfer (markerless)"),
                ("FeelAnyForce",
                 lambda: _load_json_pred("anyforce_cnc_train.json"), GREEN,
                 "near-domain (markerless)"),
            ]},
        "glowtact": {
            "title": "Markerless gel (GlowTact, 0-20 N): supervision buys the "
                     "object-dependent gain physics cannot know",
            "panels": [
                ("Ours (physics)", _load_ours_glowtact, ORANGE, "transfer"),
                ("FEATS U-net",
                 lambda: _load_json_pred("feats_on_glowtact.json"), PURPLE,
                 "transfer (markerless)"),
                ("FeelAnyForce",
                 lambda: _load_json_pred("anyforce_glowtact.json"), GREEN,
                 "near-domain (markerless)"),
            ]},
    }

    for key, spec in datasets.items():
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
        metrics[key] = {}
        for ax, (name, loader, color, domain) in zip(axes, spec["panels"]):
            try:
                t, p = loader()
            except FileNotFoundError:
                ax.set_title(f"{name}\n(not available)"); ax.axis("off")
                continue
            rho = float(spearmanr(p, t).statistic)
            mae = float(np.abs(p - t).mean())
            metrics[key][name] = {"rho": rho, "mae": mae, "n": len(t),
                                  "domain": domain}
            ax.scatter(t, p, s=9, alpha=.55, color=color)
            lim = max(t.max(), 1) * 1.06
            ax.plot([0, lim], [0, lim], "k--", lw=.8, alpha=.5)
            ax.set_xlim(-lim * .03, lim)
            ax.set_ylim(min(-lim * .03, np.percentile(p, 0.5)),
                        max(lim, np.percentile(p, 99.5)))
            ax.set_title(f"{name}\n{domain}", fontsize=9)
            ax.text(0.04, 0.96, f"ρ = {rho:.2f}",
                    transform=ax.transAxes, fontsize=13, fontweight="bold",
                    color=color, va="top")
            ax.text(0.04, 0.84, f"MAE {mae:.2f} N",
                    transform=ax.transAxes, fontsize=8.5, color="#666",
                    va="top")
            ax.set_xlabel("ground truth [N]")
        axes[0].set_ylabel("predicted [N]")
        fig.suptitle(spec["title"], fontsize=10)
        fig.tight_layout()
        f = out_dir / f"results_{key}.png"
        fig.savefig(f); plt.close(fig)
        figures[key] = f.name

    # React agreement (no GT)
    rows = json.loads((CACHE / "anyforce_react_ep0.json").read_text())
    ours = np.array([r["ours"] for r in rows])
    af = -np.array([r["fz"] for r in rows])
    rho = float(spearmanr(ours, af).statistic)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(ours, af, s=12, alpha=.6, color=ORANGE)
    ax.set_xlabel("Ours (physics) [N]")
    ax.set_ylabel("FeelAnyForce [N]")
    ax.set_title(f"React: zero-training physics and 200K-frame network\n"
                 f"agree at ρ = {rho:.2f} (n={len(rows)}) — neither can copy the other",
                 fontsize=9)
    fig.tight_layout()
    f = out_dir / "results_react.png"
    fig.savefig(f); plt.close(fig)
    figures["react"] = f.name
    metrics["react"] = {"agreement_rho": rho, "n": len(rows)}

    (CACHE / "results_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"figures": figures, "metrics": metrics}


def _load_json_pred_cnc_feats():
    rows = json.loads((OUT_ROOT / "feature_cache_ff" / "cnc_mini_val.json"
                       ).read_text())["rows"]
    t = np.array([r["force_true"] for r in rows])
    return t, np.array([r["feats_pred_n"] for r in rows])


CSS = (Path(__file__).parent / "method_page.py").read_text().split('CSS = """')[1].split('"""')[0]


SEC_FORCE_EN = r'''<h2>Every number beside the control that could have killed it</h2>
<div class="card">
<p style="margin-top:0">Re-run end to end after marker inpainting was added to
the depth pipeline. One protocol on all four datasets: within each group
(indenter family / probe / capture group / gel pad) half the frames fit a
5-feature least-squares model calibrated by isotonic regression, the other half
are scored; 5 seeds, median reported. The control column repeats the identical
protocol with the force labels <b>permuted within each group</b> — group
structure and force distribution untouched, only the frame-to-force pairing
destroyed.</p>
@@TBL_FORCE@@
<p class="footnote">Nothing moved. That is the expected result and it is worth
being explicit about: marker inpainting was adopted for <b>geometry only</b>,
so the force path was deliberately left byte-identical, and a spot check that
recomputes cached features from the raw frames confirms it
(max |cached − fresh| = 0 over 40 cnc frames). The last row is the same FEATS
frames and the same splits with the marker-inpainted features fed to the force
model instead — it <b>loses</b> 0.037 ρ, which is why it did not ship there.</p>
<p class="footnote">A within-group shuffle is the right control here, not a
global one: on FeelAnyForce the pooled ρ survived <i>global</i> shuffling at
0.442 vs 0.455, which is how we caught that its frame join had never been
demonstrated. Reading the table: cnc and GlowTact sit ~0.9 above their
controls; FEATS sits 0.78 above a control that is flat at 0.00.</p>

<h3 style="margin-bottom:6px">Datasets we evaluated but did not include</h3>
<p class="footnote" style="margin-top:0">Four GelSight-Mini force datasets are
rows above. Two more were worked on and left out, and it is worth saying why
rather than leaving a reader to wonder.</p>
<table><tr><th>dataset</th><th>why it is not a row</th></tr>
<tr><td><b>FeelAnyForce</b> (2410.02048)<br>
<span class="footnote">appears above as a <i>method</i> — its trained network is
one of our three baselines — but not as a dataset</span></td>
<td>The 48,197 images we hold locally carry <b>no force labels at all</b>
(every force column null). The labels are downloadable separately, but joining
them to the frames requires inferring a per-capture frame index, and that join
<b>was never demonstrated</b>: pooled &rho; was 0.455 while shuffling the labels
<i>within</i> each capture still gave 0.442, i.e. the correlation was entirely
between-capture structure and would survive a completely wrong frame mapping.
Including it would have meant publishing a number we could not defend. Fixing
it needs the original timestamped images (&asymp;157 GB) or an official frame
map.</td></tr>
<tr><td><b>Tactile MNIST</b> (real split)</td>
<td>Used for reconstruction ground truth, not for force — it has no force
labels. The <i>simulated</i> split is what gives us the per-pixel depth
validation above.</td></tr></table>
<p class="footnote">Also checked and rejected for having no force ground truth
in newtons, or the wrong sensor: Touch&nbsp;and&nbsp;Go, TVL, TacQuad, GelSLAM,
3DCal, DAR_OTS (GelSight model unstated; its marker-gel force labels are also a
reordered copy of the markerless run), GenForce (marker images only, no RGB),
AllSight (round finger, not a flat Mini gel), DIGIT and Wedge datasets.
<b>TacVerse</b> is the one verified GelSight-Mini candidate still open — Mini
with and without markers, 2&nbsp;mm sphere, 0.1&nbsp;mm steps — and it is
access-gated.</p>
</div>'''

SEC_FORCE_ZH = r'''<h2>每个数字都配上足以否定它的对照</h2>
<div class="card">
<p style="margin-top:0">在深度管线加入 marker 修补之后端到端重跑。四个数据集使用同一套协议：
在每个组内（压头族 / 探头 / 采集组 / gel pad）一半帧拟合 5 特征最小二乘模型并用保序回归标定，
另一半用于评分；5 个随机种子，报告中位数。对照列用完全相同的协议，但把力标签
<b>在组内打乱</b>——组结构和力的分布都不变，只破坏帧与力的配对。</p>
@@TBL_FORCE@@
<p class="footnote">没有任何数字变动。这正是预期结果，也值得明确说出来：marker 修补只被采纳用于
<b>几何</b>，力的路径被刻意保持逐字节不变；从原始帧重算缓存特征的抽查也证实了这一点
（40 帧 cnc 上 max |缓存 − 重算| = 0）。最后一行是同样的 FEATS 帧、同样的划分，
只是把 marker 修补后的特征喂给力模型——ρ <b>下降</b> 0.037，这就是它没有进入力路径的原因。</p>
<p class="footnote">这里正确的对照是组内打乱，而不是全局打乱：在 FeelAnyForce 上，
混合后的 ρ 在<i>全局</i>打乱下仍有 0.442（对比 0.455），我们正是这样发现它的帧对齐从未被验证。
读表方式：cnc 与 GlowTact 高出各自对照约 0.9；FEATS 高出 0.78，而它的对照平在 0.00。</p>

<h3 style="margin-bottom:6px">评测过但未纳入的数据集</h3>
<p class="footnote" style="margin-top:0">上表是四个 GelSight Mini 力数据集。另有两个做过工作但未纳入，
与其让读者猜，不如说清楚为什么。</p>
<table><tr><th>数据集</th><th>为什么它不是表中一行</th></tr>
<tr><td><b>FeelAnyForce</b>（2410.02048）<br>
<span class="footnote">它在上文是作为<i>方法</i>出现的——其训练好的网络是我们三个基线之一——
而不是作为数据集</span></td>
<td>我们本地持有的 48,197 张图像<b>完全没有力标签</b>（所有力字段为空）。标签可以单独下载，
但要与帧对应就必须推断每个 capture 内的帧索引，而这个对齐<b>从未被验证</b>：
混合 &rho; 为 0.455，而在每个 capture <i>内部</i>打乱标签后仍有 0.442——
也就是说相关性完全来自 capture 之间的结构，即使帧映射完全错误也会保留。
纳入它就等于发布一个我们无法辩护的数字。要修正需要带时间戳的原始图像（约 157 GB）
或官方的帧映射表。</td></tr>
<tr><td><b>Tactile MNIST</b>（真实划分）</td>
<td>用于重建真值而非力——它没有力标签。上文的逐像素深度验证用的是它的<i>仿真</i>划分。</td></tr></table>
<p class="footnote">另外经核查因缺少牛顿单位力真值、或传感器不符而排除：Touch and Go、TVL、
TacQuad、GelSLAM、3DCal、DAR_OTS（GelSight 型号未声明，且其 marker gel 的力标签是
markerless 组的重排副本）、GenForce（仅 marker 图像，无 RGB）、AllSight（圆柱指，非平面 Mini gel）、
DIGIT 与 Wedge 系数据集。<b>TacVerse</b> 是唯一确认符合的 GelSight Mini 候选——
有/无 marker、2 mm 球、0.1 mm 步进——但需申请访问权限。</p>
</div>'''

SEC_MARKER_EN = r'''<h3>Removing the marker dots: a geometry win, not a force win</h3>
<img src="assets/feats_marker_removal.png" alt="FEATS before and after marker inpainting">
<p class="footnote">The dots occlude the gel, so the photometric table has no
valid colour under them and Poisson integrates a dimple lattice into the depth
map — visible as pockmarks all over the <b>3D mesh BEFORE</b> panels. Detecting
the dots on the reference and inpainting them out of <b>both</b> the reference
and the frame before differencing (cv2 Telea, the image-space cousin of
GelSight Wedge's Fig. 10 hole interpolation) removes it: lattice power at the
31.9 px marker pitch drops <b>1.523 → 0.890</b> (×0.65), lower on <b>91%</b> of
120 frames above 1 N, Wilcoxon p = 2.6e-19. The detector is marker-specific,
not a blob finder — 63/63 dots and 0 rejects on this reference, stable for
every threshold from 3 to 16 grey levels, and exactly <b>0</b> blobs on the
markerless GlowTact and cnc references, where the step is a bit-exact no-op.</p>
<p class="footnote">Two things it does <b>not</b> do. It does not help force:
on identical frames and splits ρ goes 0.7747 → 0.7371 and every paired median
delta is negative, so the force features still come from the untouched
pipeline. And it does not catch every dot — the dots shear with the gel
(median 1.7 px, &gt;8 px on 8% of frames), so a static reference mask leaves
the displaced ones behind, which is what the residual dots in column 3 are.
Controls: inpainting the same <i>area</i> of randomly placed fake markers gives
0.7697, i.e. no gain, so the small changes are not "inpainting = smoothing".
What actually caps FEATS is the reference, not the dots — on the 20 lightest
presses |dI| already sits at 11 grey levels off-dot and 88% of off-dot pixels
pass the |dI|&gt;8 valid test, so the mask is nearly the whole frame and the
features integrate reference mismatch; a per-indenter light-press reference
made it worse still (0.7747 → 0.7261).</p></div>'''

SEC_MARKER_ZH = r'''<h3>去掉 marker 点：几何上的胜利，不是力上的</h3>
<img src="assets/feats_marker_removal.png" alt="FEATS marker 修补前后对比">
<p class="footnote">点会遮挡 gel，因此光度查找表在点下没有有效颜色，Poisson 积分会把一层
点阵凹坑积进深度图——在 <b>3D mesh BEFORE</b> 面板上表现为满屏麻点。在参考帧上检测这些点，
并在做差分<b>之前</b>把它们从参考帧<b>和</b>当前帧里一起修补掉（cv2 Telea，相当于 GelSight
Wedge 图 10 中孔洞插值的图像域版本），凹坑就消失了：31.9 px 点距对应频率上的功率从
<b>1.523 降到 0.890</b>（×0.65），在 1 N 以上的 120 帧中有 <b>91%</b> 变低，
Wilcoxon p = 2.6e-19。检测器是 marker 专用的，不是通用斑点检测器——在该参考帧上 63/63 个点、
0 个误检，阈值从 3 到 16 灰阶都稳定；而在无 marker 的 GlowTact 与 cnc 参考帧上恰好检出
<b>0</b> 个，此时该步骤是逐比特的空操作。</p>
<p class="footnote">它<b>做不到</b>两件事。第一，它对力没有帮助：在相同帧、相同划分下
ρ 从 0.7747 降到 0.7371，且每个配对中位差都是负的，所以力特征仍然来自未改动的管线。
第二，它抓不全所有点——点会随 gel 剪切移动（中位 1.7 px，8% 的帧超过 8 px），
静态参考掩码追不上位移的点，第 3 列里残留的点就是它们。对照实验：把同样<i>面积</i>的
随机假 marker 修补掉得到 0.7697，即没有增益，说明这些微小变化并不是"修补=平滑=更好"。
真正卡住 FEATS 的是参考帧而非点——在最轻的 20 次按压上，点外区域的 |dI| 已经有 11 个灰阶，
88% 的点外像素通过 |dI|&gt;8 的有效性判据，掩码几乎覆盖整幅画面，特征积分的是参考帧失配；
换成逐压头的轻压参考帧反而更差（0.7747 → 0.7261）。</p></div>'''

SEC_SCOPE_EN = r'''<p><b>And state the scope with it.</b> Those ρ are
<i>force</i> scores, per group. The <i>depth</i> underneath is now measured
against exact per-pixel ground truth and is a strong function of press depth —
11 µm MAE at 0.3 mm, 281 µm at 2.25 mm — so the honest one-line summary is:
<b>accurate shallow-contact geometry, monotone force within a calibrated
group, and neither claim survives a contact that leaves the field of view.</b>
Marker gels get one extra step (dots inpainted before differencing) which
buys geometry and not force.</p>'''

SEC_SCOPE_ZH = r'''<p><b>并且要把适用范围一起说清楚。</b>上面那些 ρ 是逐组的<i>力</i>得分。
底层的<i>深度</i>现在已经用精确逐像素真值测过，它是压入深度的强函数——
0.3 mm 处 MAE 11 µm，2.25 mm 处 281 µm——所以诚实的一句话总结是：
<b>浅接触几何精确、标定组内力单调，而一旦接触跑出视野，两条结论都不成立。</b>
有 marker 的 gel 多一步（差分前把点修补掉），它买到的是几何，不是力。</p>'''

SEC_GT_EN = r'''<h2>First external per-pixel ground truth — and it moves the headline</h2>
<div class="card">
<p style="margin-top:0">Everything above scores <b>force</b>. Until now the
<b>depth</b> underneath was only ever checked against our own analytic sphere
cap, with the cap's amplitude anchored on the reconstruction itself. Tactile
MNIST supplies what was missing: exact per-pixel depth, ray-cast from the
3D-printed digit meshes, on <b>non-spherical</b> geometry a sphere calibration
cannot self-validate — 420 touches, 106 objects. The pose bookkeeping is
verified end to end rather than assumed (re-rendering the ground-truth height
map reproduces the shipped image to ~2/255 grey levels).</p>
<img src="assets/mnist_examples.png" alt="reconstruction vs exact mesh ground truth">
<p><b>The finding is a range, not a number: accuracy is a steep function of
press depth.</b> Same digit meshes, re-rendered at five penetrations, no
per-frame alignment and no fitted indentation scale:</p>
<table class="matrix">
<tr><th>press depth [mm]</th><th>0.30</th><th>0.60</th><th>1.00</th><th>1.50</th>
<th>2.25 — what the dataset ships</th></tr>
<tr><td>MAE [µm]</td><td class="best"><b>11.2</b></td><td class="best">35.0</td>
<td>67.8</td><td>127.4</td><td class="dead">281.1</td></tr>
<tr><td>Type-2 error [µm]</td><td class="best"><b>96.5</b></td><td>186.3</td>
<td>308.6</td><td>514.6</td><td class="dead">961.8</td></tr>
<tr><td>peak recovered (ours / GT)</td><td>1.00</td><td>0.97</td><td>0.77</td>
<td>0.68</td><td class="dead">0.55</td></tr>
</table>
<p class="footnote">At 0.3 mm the Type-2 error is <b>96.5 µm</b>, below every
one of 3D Cal's three published figures (152.8 / 171.6 / 290.0 µm) — and
theirs are reported <i>with</i> a 2D cross-correlation alignment and a fitted
indentation scale, ours with neither. At 0.6 mm (186.3 µm) we sit inside their
range. At the 2.25 mm press this dataset actually ships we are an order of
magnitude worse and recover barely half the peak. <b>So: no accuracy number on
this site should be quoted without the press depth it was measured at, and the
working range of this reconstruction is shallow contact.</b></p>
<p><b>What it corrects, what it confirms.</b></p>
<table class="matrix">
<tr><th>earlier claim</th><th>after per-pixel ground truth</th></tr>
<tr><td>Flat-topped indenters over-dome badly — centre/rim 1.23&ndash;1.42
where 1.0 would be correct</td>
<td class="dead"><b>retracted.</b> The premise was wrong: compliant gel wraps
around a flat edge, so c/r &gt; 1 is <i>expected</i>. GT says the true depth
map of a pressed digit has c/r <b>1.400</b> (the gel surface itself 1.334)
while we reconstruct 1.539&ndash;1.562 &mdash; <b>+10&ndash;12%</b>; on an
enclosed plateau control whose truth is exactly 1.000 we measure <b>1.069</b>,
<b>+7%</b>. The over-doming is real and small, not the +23&ndash;42% we
implied.</td></tr>
<tr><td>Up to 22% of contact pixels land in unobserved LUT bins &mdash; a
leading defect</td>
<td class="dead"><b>demoted.</b> 13.8% / 16.9% on the GT set, correlating with
per-touch Type-2 error at only <b>0.098 / 0.294</b>.</td></tr>
<tr><td>The <code>|dI| &gt; 8</code> valid mask is halo-dominated</td>
<td class="best"><b>confirmed and quantified.</b> Against the true contact
region: IoU <b>0.614</b>, recall <b>0.917</b>, over-segmentation <b>0.531</b>
&mdash; it finds nearly all the contact and then adds half as much again in
halo.</td></tr>
<tr><td>The photometric table might be the weak link off-domain</td>
<td class="best"><b>ruled out.</b> LUT gradient vs true gel gradient is
<b>24.4&deg;</b> on these renders against <b>26.1&deg;</b> for the same table
on its own real sensor, and refitting the table on in-domain sphere renders
does not improve the digits (316.9 vs 273.4 µm).</td></tr>
</table>
<p><b>One failure mode, now seen on four datasets.</b> 420/420 of these touches
have a contact that runs off the pad, and a control that moves a single sphere
cap from mid-pad to the edge collapses its peak <b>1.39 &rarr; 0.30 mm</b>
against a 0.90 mm truth (Type-2 292 &rarr; 450 µm). That is the same effect as
cnc_Mini's press grid being larger than the field of view (&rho; 0.11 &rarr;
0.94 in the field-of-view ablation) and Sparsh's clipped-disc frames
scoring worse despite carrying the highest median force. <b>Contact
visibility, not force range or gel type, is this pipeline's single biggest
external failure mode</b> &mdash; the Poisson solve's zero boundary cannot
represent a surface that leaves the frame.</p>
<p class="footnote"><b>Honest caveat.</b> These images are Taxim renders, not
real GelSight frames, so this validates the <i>geometry solver</i> &mdash;
table, mask, integration &mdash; rather than the sensor model, and the domain
check above is what licenses reading it that way. Taxim's gel also follows the
object geometry to within ~38 µm, so it barely models real gel
non-conformance: the true numbers on a physical sensor at the same press depth
will be worse, not better. Reproduce with
<code>python -m force_recovery.mnist_validation stage1|controls|sweep</code>.</p>
</div>'''

SEC_GT_ZH = r'''<h2>第一份外部逐像素真值——它改变了主结论</h2>
<div class="card">
<p style="margin-top:0">上面评的都是<b>力</b>。在此之前，底层的<b>深度</b>只被拿来和我们自己的
解析球冠比较，而球冠的幅值还是从重建本身锚定的。Tactile MNIST 补上了缺失的一环：
由 3D 打印数字网格光线投射得到的精确逐像素深度，且是球面标定无法自证的<b>非球面</b>几何——
420 次触碰、106 个物体。位姿对应关系是端到端验证过的而非假设的
（用真值高度图重渲染可复现出厂图像，误差约 2/255 灰阶）。</p>
<img src="assets/mnist_examples.png" alt="重建与精确网格真值对比">
<p><b>结论是一个区间，不是一个数字：精度是压入深度的陡峭函数。</b>
同样的数字网格，在五个压入深度上重渲染，没有逐帧配准，也没有拟合压入尺度：</p>
<table class="matrix">
<tr><th>压入深度 [mm]</th><th>0.30</th><th>0.60</th><th>1.00</th><th>1.50</th>
<th>2.25 — 数据集实际发布的</th></tr>
<tr><td>MAE [µm]</td><td class="best"><b>11.2</b></td><td class="best">35.0</td>
<td>67.8</td><td>127.4</td><td class="dead">281.1</td></tr>
<tr><td>Type-2 误差 [µm]</td><td class="best"><b>96.5</b></td><td>186.3</td>
<td>308.6</td><td>514.6</td><td class="dead">961.8</td></tr>
<tr><td>峰值还原比（我们 / 真值）</td><td>1.00</td><td>0.97</td><td>0.77</td>
<td>0.68</td><td class="dead">0.55</td></tr>
</table>
<p class="footnote">在 0.3 mm 处 Type-2 误差为 <b>96.5 µm</b>，低于 3D Cal 公布的全部三个数字
（152.8 / 171.6 / 290.0 µm）——而且他们的结果是<i>带</i>二维互相关配准和拟合压入尺度得到的，
我们两者都没有。在 0.6 mm（186.3 µm）我们落在他们的区间内。而在该数据集实际使用的 2.25 mm
压入下，我们差了一个数量级，峰值只还原了一半左右。<b>所以：本站任何精度数字都必须连同
它所对应的压入深度一起引用，而这套重建的适用区间是浅接触。</b></p>
<p><b>它纠正了什么，又确认了什么。</b></p>
<table class="matrix">
<tr><th>此前的说法</th><th>逐像素真值之后</th></tr>
<tr><td>平头压头过度穹顶化——中心/边缘比 1.23&ndash;1.42，而 1.0 才是正确值</td>
<td class="dead"><b>撤回。</b>前提本身就是错的：gel 是柔顺的，会包裹平边，所以
c/r &gt; 1 是<i>应当出现</i>的。真值显示被压数字的真实深度图 c/r 为 <b>1.400</b>
（gel 表面本身是 1.334），而我们重建出 1.539&ndash;1.562——<b>+10&ndash;12%</b>；
在真值恰为 1.000 的封闭平台对照上我们测得 <b>1.069</b>，即 <b>+7%</b>。
过度穹顶化真实存在但很小，不是我们暗示的 +23&ndash;42%。</td></tr>
<tr><td>多达 22% 的接触像素落在未观测过的 LUT bin 里——一个主要缺陷</td>
<td class="dead"><b>降级。</b>在真值集上为 13.8% / 16.9%，
与逐次触碰 Type-2 误差的相关性仅 <b>0.098 / 0.294</b>。</td></tr>
<tr><td><code>|dI| &gt; 8</code> 有效掩码被光晕主导</td>
<td class="best"><b>确认并量化。</b>相对真实接触区域：IoU <b>0.614</b>，
召回 <b>0.917</b>，过分割 <b>0.531</b>——它几乎找全了接触，然后又多加了半个接触面积的光晕。</td></tr>
<tr><td>出域时光度查找表可能是薄弱环节</td>
<td class="best"><b>排除。</b>LUT 梯度与真实 gel 梯度的夹角在这些渲染上是 <b>24.4&deg;</b>，
而同一张表在它自己的真实传感器上是 <b>26.1&deg;</b>；用域内球压重渲染重新拟合查找表
也没有改善数字（316.9 vs 273.4 µm）。</td></tr>
</table>
<p><b>同一个失效模式，现在已在四个数据集上看到。</b>这里 420/420 次触碰的接触都跑出了 pad，
而一个把单个球冠从 pad 中心移到边缘的对照，使峰值从 <b>1.39 塌到 0.30 mm</b>
（真值 0.90 mm，Type-2 292 &rarr; 450 µm）。这与 cnc_Mini 的按压网格大于视野
（视野消融实验里 &rho; 0.11 &rarr; 0.94）、以及 Sparsh 中被裁切的接触圆盘虽然力中位数最高
却得分更差，是同一回事。<b>接触可见性——而不是力程或 gel 类型——才是这条管线最大的外部失效模式</b>：
Poisson 求解的零边界无法表示一个跑出画面的曲面。</p>
<p class="footnote"><b>诚实的边界。</b>这些图像是 Taxim 渲染，不是真实 GelSight 帧，
所以它验证的是<i>几何求解器</i>（查找表、掩码、积分），而不是传感器模型；
上面的域检查正是允许这样解读的依据。另外 Taxim 的 gel 与物体几何的偏差只有约 38 µm，
几乎没有建模真实 gel 的非贴合性：在同样压深下，真实传感器上的数字只会更差，不会更好。
复现：<code>python -m force_recovery.mnist_validation stage1|controls|sweep</code>。</p>
</div>'''

EN = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Results — Force Estimation on GelSight Mini</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500&display=swap" rel="stylesheet">
<style>@@CSS@@
.matrix td.best{color:var(--ok);font-weight:700}
.matrix td.dead{color:#c0392b}
</style></head><body><div class="wrap">

<header>
<div class="kicker">React force recovery · results</div>
<h1>Three estimators × four ground-truth datasets</h1>
<p class="sub">Every method evaluated on every dataset with force-sensor labels,
predicted vs ground truth per dataset — so dataset quality is controlled within
each row, and differences between panels are differences between methods.</p>
<a class="pill" href="method.html">↖ how the method is designed</a>
<a class="pill" href="recon_workbench.html">3D workbench</a>
<a class="pill" href="index.html">overview</a>
<a class="pill" href="gallery.html">gallery</a>
<a class="pill" href="results_zh.html">中文</a>
</header>

<h2>The matrix (Spearman ρ, predicted vs F/T ground truth)</h2>
<div class="card"><table class="matrix">
<tr><th>dataset (gel type)</th><th>Ours — physics,<br>0 training frames</th>
<th>FEATS U-net<br>(trained: marker gel)</th>
<th>FeelAnyForce<br>(trained: markerless)</th></tr>
<tr><td>FEATS val (marker)</td><td>@@O_FEATS@@</td>
<td class="best">@@N_FEATS@@ · in-domain</td><td class="dead">@@A_FEATS@@</td></tr>
<tr><td>FoTa cnc_Mini (markerless)</td><td>@@O_CNC@@ (in view)</td>
<td class="dead">@@N_CNC@@</td><td class="best">@@A_CNC@@</td></tr>
<tr><td>GlowTact (markerless)</td><td>@@O_GLOW@@</td>
<td class="dead">@@N_GLOW@@</td><td class="best">@@A_GLOW@@</td></tr>
<tr><td>Sparsh / Meta (markerless, 10 gel pads)</td>
<td>@@O_SPARSH@@ (in view, self-calibrated table)</td>
<td colspan="2">not run &mdash; no published predictions</td></tr>
</table>
<p>The pattern is the finding: <b>each network dominates its own gel domain and
collapses outside it</b> (FEATS 0.96 → 0.04–0.07; FeelAnyForce 0.90 → 0.43),
while <b>the physics pipeline is the only estimator that works everywhere</b>
(@@RANGE@@) — it never sees training data, so it has no domain to leave.</p>
</div>

@@SEC_FORCE@@

<h2>FEATS dataset — marker-dot gel</h2>
<div class="card"><img src="assets/results_feats.png" alt="FEATS dataset panels">
<p class="footnote">In-domain, the FEATS U-net is excellent (ρ=0.96) — the
negative results elsewhere are domain effects, not a weak model. FeelAnyForce,
markerless-trained, degrades on the dotted gel (0.43): the same knife cuts both
ways.</p>
@@SEC_MARKER@@

<h2>FoTa cnc_Mini — markerless gel</h2>
<div class="card"><img src="assets/results_cnc.png" alt="cnc_Mini panels">
<p class="footnote">Hard conditions: only 4 contact-free frames, 62% of presses
near the pad border — the press grid is larger than the field of view (see
<a href="debug_pipeline.html">pipeline debug</a>). Strictly in view our ρ is
@@O_CNC@@ (MAE @@M_CNC@@ N); FeelAnyForce reaches 0.92 non-edge.</p></div>

<h2>GlowTact — markerless gel, cleaned</h2>
<div class="card"><img src="assets/results_glowtact.png" alt="GlowTact panels">
<p class="footnote">Friendliest ground truth (centred presses, 10 free frames,
0–20 N). Per-indenter, calibrated within each family under the physical scope
(contact fully in view, gel not bottomed out), ρ is 0.975–0.992 across all six
indenters with MAE ≤ 0.73 N. Caveat: that calibration is refit per family, so
it measures rank agreement within a group, not a transferable absolute-newton
scale — see <a href="method.html">per-dataset calibration</a>.</p></div>

<h2>Sparsh (Meta) — a fourth dataset, and where the method breaks</h2>
<div class="card"><img src="assets/results_sparsh.png" alt="Sparsh results">
<p class="footnote">10 gel pads (6 sphere, 2 flat, 2 sharp), force in newtons.
Our GlowTact table applied to this foreign sensor reaches &rho;=0.878 on
in-view frames. Rebuilding the table from <b>Sparsh's own sphere presses</b>
— 708 frames, radius fitted at R=2.44 mm from a&sup2;=d(2R&minus;d) — takes it
to <b>&rho;=0.968, MAE 0.042 N</b>, against a labels-shuffled-within-pad
control of 0.23. Fitting on one pad and applying it <b>unchanged</b> to another
costs nothing measurable (0.96&ndash;0.98 everywhere): one table, six gel pads.</p>
<img src="assets/sparsh_dome.png" alt="dome before and after">
<p class="footnote">Why the table matters more than the fit: with the wrong
sensor's table a sphere press integrates to a <b>bilobed shape with a central
dip</b>; with the self-calibrated table it is a single dome matching the
analytic spherical cap (residual RMS 0.179 &rarr; <b>0.0545 mm</b>). Measured
before any integration, the LUT gradient sits <b>93.3&deg;</b> from the analytic
sphere gradient — chance is 90&deg; — and self-calibration brings it to
<b>4.5&deg;</b> (within 30&deg;: 15% &rarr; 99%). Reverse control: the Sparsh
table fails on GlowTact frames too, so this is a per-sensor property, not a
bad table.</p>
<p class="footnote"><b>What this costs and where it still fails.</b> The price
is one set of sphere presses with logged depth on the target sensor: this is
<b>calibrate once per sensor</b>, not zero-shot. Rank order transfers across
indenter shapes but <i>absolute newtons do not</i> — a sphere-fitted model
applied to a flat punch degrades to MAE 0.37&ndash;0.40 N, and it got
<i>worse</i> with the correct table, because true geometry widens the real
feature-scale gap between a sphere and a punch. The sharp indenter is
unsupported (in-view &rho; 0.58). Shear stays out of reach by construction —
the top shear decile keeps 1.6&times; the residual whichever table is used.
Frames are restricted to a visible contact disc: 36% of presses show none and
11% are clipped, and the clipped subset carries the <i>highest</i> median force
yet scores worse, so this is visibility, not force-range filtering. Three
dataset defects had to be fixed first: flat/sharp trajectories ship 5 more
frame indices than forces (silently drifting the labels, &rho;&asymp;0 until
paired within each trajectory), sharp/batch_2 is stored BGR while the other
nine are RGB, and flat/batch_2 ships only 3 of 4 image files.</p></div>

@@SEC_GT@@

<h2>React — no ground truth, so: do independent methods agree?</h2>
<div class="card"><img src="assets/results_react.png" alt="React agreement">
<p class="footnote">On the dataset we actually care about, the two surviving
estimators — physics (zero training) and FeelAnyForce (200K frames) — agree at
ρ=0.91, and FeelAnyForce reads ≈0 N on every frame the physics pipeline calls
contact-free. Neither can copy the other's mistakes.</p></div>

<h2>Takeaway</h2>
<div class="card"><p style="margin-top:0"><b>For labelling React</b> (markerless
Mini): FeelAnyForce as the primary labeller, the physics pipeline as an
independent audit, disagreement rows flagged. <b>For any new gel or sensor</b>
where no trained model matches the domain: the physics pipeline is the only
option that works out of the box — and its FEATS-dataset score (@@O_FEATS@@) shows
what it does on a domain nobody tuned it for.</p>
@@SEC_SCOPE@@</div>

<p><a href="debug_pipeline.html"><b>Pipeline debug page</b></a>: raw
image &rarr; force step by step on all three datasets, and the cnc
field-of-view ablation (in-view &rho;=@@O_CNC@@).</p>

<footer>React force recovery · <a href="index.html">overview</a> ·
<a href="method.html">method design</a> · <a href="gallery.html">gallery</a> ·
data: FEATS (2411.03315) · FoTa/T3 (2406.13640) · GlowTact
(dacongming666/GlowTact_Datasets) · FeelAnyForce (2410.02048) ·
Tactile MNIST (TUDa-RL) · Sparsh (facebook/gelsight-force-estimation)</footer>
</div></body></html>"""

ZH = [
    ('<h2>Sparsh (Meta) — a fourth dataset, and where the method breaks</h2>',
     '<h2>Sparsh (Meta) — 第四个数据集，以及方法的边界</h2>'),
    ('<p class="footnote">10 gel pads (6 sphere, 2 flat, 2 sharp), force in newtons.\nOur GlowTact table applied to this foreign sensor reaches &rho;=0.878 on\nin-view frames. Rebuilding the table from <b>Sparsh\'s own sphere presses</b>\n— 708 frames, radius fitted at R=2.44 mm from a&sup2;=d(2R&minus;d) — takes it\nto <b>&rho;=0.968, MAE 0.042 N</b>, against a labels-shuffled-within-pad\ncontrol of 0.23. Fitting on one pad and applying it <b>unchanged</b> to another\ncosts nothing measurable (0.96&ndash;0.98 everywhere): one table, six gel pads.</p>\n<img src="assets/sparsh_dome.png" alt="dome before and after">\n<p class="footnote">Why the table matters more than the fit: with the wrong\nsensor\'s table a sphere press integrates to a <b>bilobed shape with a central\ndip</b>; with the self-calibrated table it is a single dome matching the\nanalytic spherical cap (residual RMS 0.179 &rarr; <b>0.0545 mm</b>). Measured\nbefore any integration, the LUT gradient sits <b>93.3&deg;</b> from the analytic\nsphere gradient — chance is 90&deg; — and self-calibration brings it to\n<b>4.5&deg;</b> (within 30&deg;: 15% &rarr; 99%). Reverse control: the Sparsh\ntable fails on GlowTact frames too, so this is a per-sensor property, not a\nbad table.</p>',
     '<p class="footnote">10 块 gel pad（6 球、2 平、2 尖），力为牛顿。我们用 GlowTact\n标定的表应用到这个外来传感器，视野内 &rho;=0.878。改用 <b>Sparsh 自己的球压帧</b>\n重建查找表（708 帧，由 a&sup2;=d(2R&minus;d) 拟合出 R=2.44 mm）后达到\n<b>&rho;=0.968，MAE 0.042 N</b>，对照组（pad 内打乱标签）为 0.23。\n在一块 pad 上拟合后<b>原样</b>应用到另一块没有可测代价（各处 0.96&ndash;0.98）：\n一张表，六块 gel pad。</p>\n<img src="assets/sparsh_dome.png" alt="重建前后对比">\n<p class="footnote">为什么"表"比"拟合"更关键：用错传感器的表，球压积分出的是\n<b>双瓣+中心凹陷</b>；用自标定的表则是与解析球冠吻合的单一圆顶\n（残差 RMS 0.179 &rarr; <b>0.0545 mm</b>）。在任何积分之前测量：LUT 梯度与解析球面\n梯度夹角为 <b>93.3&deg;</b>（随机水平是 90&deg;），自标定后降到 <b>4.5&deg;</b>\n（30&deg; 以内占比 15% &rarr; 99%）。反向对照：Sparsh 的表在 GlowTact 帧上同样失败，\n所以这是逐传感器属性，而非"我们的表不好"。</p>'),
    ('<p class="footnote"><b>What this costs and where it still fails.</b> The price\nis one set of sphere presses with logged depth on the target sensor: this is\n<b>calibrate once per sensor</b>, not zero-shot. Rank order transfers across\nindenter shapes but <i>absolute newtons do not</i> — a sphere-fitted model\napplied to a flat punch degrades to MAE 0.37&ndash;0.40 N, and it got\n<i>worse</i> with the correct table, because true geometry widens the real\nfeature-scale gap between a sphere and a punch. The sharp indenter is\nunsupported (in-view &rho; 0.58). Shear stays out of reach by construction —\nthe top shear decile keeps 1.6&times; the residual whichever table is used.\nFrames are restricted to a visible contact disc: 36% of presses show none and\n11% are clipped, and the clipped subset carries the <i>highest</i> median force\nyet scores worse, so this is visibility, not force-range filtering. Three\ndataset defects had to be fixed first: flat/sharp trajectories ship 5 more\nframe indices than forces (silently drifting the labels, &rho;&asymp;0 until\npaired within each trajectory), sharp/batch_2 is stored BGR while the other\nnine are RGB, and flat/batch_2 ships only 3 of 4 image files.</p>',
     '<p class="footnote"><b>代价与仍然失败之处。</b>代价是在目标传感器上采一组带深度记录的\n球压：这是<b>每个传感器标定一次</b>，不是零样本。秩序可以跨压头形状迁移，但\n<i>绝对牛顿不行</i>——球拟合的模型用到平头上 MAE 退化到 0.37&ndash;0.40 N，而且\n用正确的表反而<i>更差</i>，因为真实几何拉大了球与平头之间本就存在的特征尺度差异。\n尖头不受支持（视野内 &rho; 0.58）。剪切在结构上无法覆盖——无论用哪张表，\n剪切最高十分位的残差都保持 1.6 倍。帧被限制为接触圆盘可见：36% 的按压检测不到圆盘、\n11% 被裁切，而被裁切子集的力中位数<i>最高</i>却得分更差，说明这是可见性问题而非力程筛选。\n另外还先修了三个数据集缺陷：flat/sharp 的轨迹里帧索引比力多 5 个（标签会静默错位，\n逐轨迹配对前 &rho;&asymp;0）、sharp/batch_2 以 BGR 存储而其余九个是 RGB、\nflat/batch_2 只有 3 个图像文件。</p>'),
    ("Pipeline debug page</b></a>: raw\nimage &rarr; force step by step on all three datasets, and the cnc\nfield-of-view ablation (in-view &rho;=@@O_CNC@@).",
     "管线调试页</b></a>：三个数据集上从原始图像到力的逐步展示，以及 cnc 视野消融（视野内 &rho;=@@O_CNC@@）。"),
    ('<html lang="en">', '<html lang="zh-CN">'),
    ("<title>Results — Force Estimation on GelSight Mini</title>",
     "<title>评测结果 — GelSight Mini 力估计</title>"),
    ('<div class="kicker">React force recovery · results</div>',
     '<div class="kicker">React 力恢复 · 评测结果</div>'),
    ("<h1>Three estimators × four ground-truth datasets</h1>",
     "<h1>三个估计器 × 四个真值数据集</h1>"),
    ("""<p class="sub">Every method evaluated on every dataset with force-sensor labels,
predicted vs ground truth per dataset — so dataset quality is controlled within
each row, and differences between panels are differences between methods.</p>""",
     """<p class="sub">每个方法在每个有力传感器标注的数据集上评测,逐数据集画预测 vs 真值——
数据集质量在行内被控制,面板之间的差异就是方法之间的差异。</p>"""),
    ('<a class="pill" href="method.html">↖ how the method is designed</a>',
     '<a class="pill" href="method_zh.html">↖ 方法设计</a>'),
    ('<a class="pill" href="index.html">overview</a>\n<a class="pill" href="gallery.html">gallery</a>\n<a class="pill" href="results_zh.html">中文</a>',
     '<a class="pill" href="index.html">总览(英文)</a>\n<a class="pill" href="gallery.html">图库</a>\n<a class="pill" href="results.html">English</a>'),
    ("<h2>The matrix (Spearman ρ, predicted vs F/T ground truth)</h2>",
     "<h2>矩阵(Spearman ρ,预测 vs F/T 真值)</h2>"),
    ("<tr><th>dataset (gel type)</th><th>Ours — physics,<br>0 training frames</th>",
     "<tr><th>数据集(gel 类型)</th><th>我们——物理,<br>零训练帧</th>"),
    ("<th>FEATS U-net<br>(trained: marker gel)</th>",
     "<th>FEATS U-net<br>(训练域:有点 gel)</th>"),
    ("<th>FeelAnyForce<br>(trained: markerless)</th></tr>",
     "<th>FeelAnyForce<br>(训练域:无点)</th></tr>"),
    ('<td>FEATS val (marker)</td>', '<td>FEATS val(有点)</td>'),
    ('<td class="best">@@N_FEATS@@ · in-domain</td>', '<td class="best">0.96 · 本域</td>'),
    ("<td>FoTa cnc_Mini (markerless)</td>",
     "<td>FoTa cnc_Mini(无点)</td>"),
    ("(in view)</td>", "(视野内)</td>"),
    ("<td>GlowTact (markerless)</td>", "<td>GlowTact(无点)</td>"),
    ("""<p>The pattern is the finding: <b>each network dominates its own gel domain and
collapses outside it</b> (FEATS 0.96 → 0.04–0.07; FeelAnyForce 0.90 → 0.43),
while <b>the physics pipeline is the only estimator that works everywhere</b>
(@@RANGE@@) — it never sees training data, so it has no domain to leave.</p>""",
     """<p>规律本身就是发现:<b>每个网络只统治自己的 gel 域,出域即塌</b>
(FEATS 0.96 → 0.04–0.07;FeelAnyForce 0.90 → 0.43);
而<b>物理管线是唯一在所有域都工作的估计器</b>(@@RANGE@@)——
它没见过训练数据,所以也没有可以离开的域。</p>"""),
    ("<h2>FEATS dataset — marker-dot gel</h2>", "<h2>FEATS 数据集——有点 gel</h2>"),
    ("""<p class="footnote">In-domain, the FEATS U-net is excellent (ρ=0.96) — the
negative results elsewhere are domain effects, not a weak model. FeelAnyForce,
markerless-trained, degrades on the dotted gel (0.43): the same knife cuts both
ways.</p>""",
     """<p class="footnote">本域内 FEATS U-net 非常出色(ρ=0.96)——它在别处的失败是域效应,
不是模型弱。FeelAnyForce(无点训练)在有点 gel 上退化到 0.43:这把刀两边都割。</p>"""),
    ("<h2>FoTa cnc_Mini — markerless gel</h2>", "<h2>FoTa cnc_Mini——无点 gel</h2>"),
    ("""<p class="footnote">Hard conditions: only 4 contact-free frames, 62% of presses
near the pad border — the press grid is larger than the field of view (see
<a href="debug_pipeline.html">pipeline debug</a>). Strictly in view our ρ is
@@O_CNC@@ (MAE @@M_CNC@@ N); FeelAnyForce reaches 0.92 non-edge.</p>""",
     """<p class="footnote">条件苛刻:只有 4 帧无接触,62% 的按压贴边。我们严格视野内 ρ=@@O_CNC@@(MAE @@M_CNC@@ N,见管线调试页);
FeelAnyForce 非边缘达 0.92。</p>"""),
    ("<h2>GlowTact — markerless gel, cleaned</h2>", "<h2>GlowTact——无点 gel,清洗过</h2>"),
    ("""<p class="footnote">Friendliest ground truth (centred presses, 10 free frames,
0–20 N). Per-indenter, calibrated within each family under the physical scope
(contact fully in view, gel not bottomed out), ρ is 0.975–0.992 across all six
indenters with MAE ≤ 0.73 N. Caveat: that calibration is refit per family, so
it measures rank agreement within a group, not a transferable absolute-newton
scale — see <a href="method.html">per-dataset calibration</a>.</p>""",
     """<p class="footnote">最友好的真值(按压居中、10 帧自由参考、0–20 N)。逐 indenter 在
物理作用域内(接触完全在视野、凝胶未触底)家族内标定后,六种 indenter 的 ρ 为
0.975–0.992,MAE ≤ 0.73 N。注意:该标定逐家族重拟合,衡量的是组内秩一致性,
并非可迁移的绝对牛顿刻度——见<a href="method_zh.html">逐数据集标定</a>。</p>"""),
    ("<h2>React — no ground truth, so: do independent methods agree?</h2>",
     "<h2>React——没有真值,那就问:独立方法是否一致?</h2>"),
    ("""<p class="footnote">On the dataset we actually care about, the two surviving
estimators — physics (zero training) and FeelAnyForce (200K frames) — agree at
ρ=0.91, and FeelAnyForce reads ≈0 N on every frame the physics pipeline calls
contact-free. Neither can copy the other's mistakes.</p>""",
     """<p class="footnote">在我们真正关心的数据集上,两个幸存的估计器——物理(零训练)与
FeelAnyForce(20 万帧)——一致性 ρ=0.91,且物理管线判无接触的每一帧它都读 ≈0 N。
彼此无法抄袭对方的错误。</p>"""),
    ("<h2>Takeaway</h2>", "<h2>结论</h2>"),
    ("""<p style="margin-top:0"><b>For labelling React</b> (markerless
Mini): FeelAnyForce as the primary labeller, the physics pipeline as an
independent audit, disagreement rows flagged. <b>For any new gel or sensor</b>
where no trained model matches the domain: the physics pipeline is the only
option that works out of the box — and its FEATS-dataset score (@@O_FEATS@@) shows
what it does on a domain nobody tuned it for.</p>""",
     """<p style="margin-top:0"><b>给 React 打标签</b>(无点 Mini):FeelAnyForce 主标,
物理管线独立审计,分歧行打标。<b>换任何新 gel 或新传感器</b>、没有训练模型匹配该域时:
物理管线是唯一开箱即用的选项——它在 FEATS 数据集上的 @@O_FEATS@@ 就是"无人为它调过的域"上的表现。</p>"""),
    ("""<footer>React force recovery · <a href="index.html">overview</a> ·
<a href="method.html">method design</a> · <a href="gallery.html">gallery</a> ·
data: FEATS (2411.03315) · FoTa/T3 (2406.13640) · GlowTact
(dacongming666/GlowTact_Datasets) · FeelAnyForce (2410.02048) ·
Tactile MNIST (TUDa-RL) · Sparsh (facebook/gelsight-force-estimation)</footer>""",
     """<footer>React 力恢复 · <a href="index.html">总览</a> ·
<a href="method_zh.html">方法设计</a> · <a href="gallery.html">图库</a> ·
数据:FEATS (2411.03315) · FoTa/T3 (2406.13640) · GlowTact
(dacongming666/GlowTact_Datasets) · FeelAnyForce (2410.02048) ·
Tactile MNIST (TUDa-RL) · Sparsh (facebook/gelsight-force-estimation)</footer>"""),
]


def _force_row(key: str) -> dict:
    d = json.loads((CACHE / "force_matrix.json").read_text())["datasets"]
    return next(v for k, v in d.items() if k.lower().startswith(key.lower()))


def force_table() -> str:
    """The refreshed metric table, straight out of force_matrix.json."""
    d = json.loads((CACHE / "force_matrix.json").read_text())["datasets"]
    head = ("<tr><th>dataset</th><th>n (eval)</th><th>&rho;</th>"
            "<th>&rho; across seeds</th><th>MAE [N]</th>"
            "<th>within-group shuffle</th></tr>")
    body = ""
    for k, v in d.items():
        cls = "dead" if "rejected" in k else "best"
        body += (f"<tr><td>{k}</td><td>{v['n_eval']}</td>"
                 f"<td class='{cls}'>"
                 f"<b>{v['rho']:.3f}</b></td>"
                 f"<td>{v['rho_min']:.3f}&ndash;{v['rho_max']:.3f}</td>"
                 f"<td>{v['mae']:.3f}</td>"
                 f"<td>{v['shuffle_rho']:+.3f}</td></tr>")
    return f'<table class="matrix">{head}{body}</table>'


def build_pages() -> tuple[Path, Path]:
    """EN -> ZH -> numbers, in that order.

    Translating BEFORE substituting the metrics is what keeps the two pages
    honest: the ZH pairs then match on tokens, never on a rendered number, so
    a metric can move without silently un-translating a paragraph (or, worse,
    leaving the Chinese page quoting last week's rho).
    """
    m = json.loads((CACHE / "results_metrics.json").read_text())
    o = lambda d, k: m[d][k]["rho"]                             # noqa: E731
    tok = {
        "@@O_FEATS@@": f"{o('feats', 'Ours (physics)'):.2f}",
        "@@N_FEATS@@": f"{o('feats', 'FEATS U-net'):.2f}",
        "@@A_FEATS@@": f"{o('feats', 'FeelAnyForce'):.2f}",
        "@@O_CNC@@":   f"{o('cnc', 'Ours (physics)'):.2f}",
        "@@N_CNC@@":   f"{o('cnc', 'FEATS U-net'):.2f}",
        "@@A_CNC@@":   f"{o('cnc', 'FeelAnyForce'):.2f}",
        "@@O_GLOW@@":  f"{o('glowtact', 'Ours (physics)'):.2f}",
        "@@N_GLOW@@":  f"{o('glowtact', 'FEATS U-net'):.2f}",
        "@@A_GLOW@@":  f"{o('glowtact', 'FeelAnyForce'):.2f}",
        "@@M_CNC@@":   f"{m['cnc']['Ours (physics)']['mae']:.2f}",
        "@@RANGE@@":   (f"{min(o(d, 'Ours (physics)') for d in m if d != 'react'):.2f}"
                        f"&ndash;0.99"),
        "@@O_SPARSH@@": f"{_force_row('Sparsh')['rho']:.2f}",
        "@@TBL_FORCE@@": force_table(),
    }
    en = (EN.replace("@@CSS@@", CSS)
            .replace("@@SEC_FORCE@@", SEC_FORCE_EN)
            .replace("@@SEC_MARKER@@", SEC_MARKER_EN)
            .replace("@@SEC_SCOPE@@", SEC_SCOPE_EN)
            .replace("@@SEC_GT@@", SEC_GT_EN))
    # The new sections are single-sourced: EN goes in by token above and the
    # ZH pair below matches the same constant by construction, so a long
    # block cannot silently drift between the two languages.
    zh_pairs = ZH + [(SEC_FORCE_EN, SEC_FORCE_ZH),
                     (SEC_MARKER_EN, SEC_MARKER_ZH),
                     (SEC_SCOPE_EN, SEC_SCOPE_ZH),
                     (SEC_GT_EN, SEC_GT_ZH),
                     ('<a class="pill" href="recon_workbench.html">3D workbench</a>',
                      '<a class="pill" href="recon_workbench.html">3D 工作台</a>'),
                     ('<tr><td>Sparsh / Meta (markerless, 10 gel pads)</td>\n'
                      '<td>@@O_SPARSH@@ (in view, self-calibrated table)</td>\n'
                      '<td colspan="2">not run &mdash; no published predictions</td></tr>',
                      '<tr><td>Sparsh / Meta(无点,10 块 gel pad)</td>\n'
                      '<td>@@O_SPARSH@@(视野内,自标定查找表)</td>\n'
                      '<td colspan="2">未运行 &mdash; 没有公开的预测结果</td></tr>')]
    zh = en
    for old, new in zh_pairs:
        assert old in zh, f"ZH miss: {old[:60]!r}"
        zh = zh.replace(old, new, 1)
    for name, page in (("results.html", en), ("results_zh.html", zh)):
        for k, v in tok.items():
            page = page.replace(k, v)
        assert "@@" not in page, f"{name}: unreplaced token"
        (SITE / name).write_text(page)
    return SITE / "results.html", SITE / "results_zh.html"




# --- LUT-v2 pipeline loaders (override the MLP-era ones above) -----------
_DG = OUT_ROOT / "site_assets" / "debug_gallery"


def _basis(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def _fit_group_pred(X, f, groups, seed=0):
    """The MEDIAN-rho seed of the protocol in `force_eval_all`.

    The panels and the refreshed table have to be one estimator, not two, or
    a reader can catch the site quoting two different numbers for the same
    row. `force_eval_all.evaluate` reports the median over 5 seeds, so the
    scatter shows exactly the seed that produced it — real predictions, and
    the annotated rho is that seed's own.

    (Previously this used one RNG re-seeded identically for every group,
    which made a different split again.)"""
    from scipy.stats import spearmanr

    from .force_eval_all import SEEDS, _one_seed
    X, f, groups = (np.asarray(X, float), np.asarray(f, float),
                    np.asarray(groups))
    runs = [_one_seed(X, f, groups, s)[:2] for s in range(SEEDS)]
    rr = [spearmanr(p, t).statistic for t, p in runs]
    return runs[int(np.argsort(rr)[len(rr) // 2])]


def _load_ours_glowtact():
    rows = json.loads((CACHE / "lut_full.json").read_text())
    rows = [r for r in rows if r["f"] > 0.15 and np.isfinite(r.get("cx", np.nan))]
    a = lambda k: np.array([r[k] for r in rows])
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    V, V2, A, D, cx, cy = a("vol"), a("vol2"), a("area"), a("maxd"), a("cx"), a("cy")
    grp = np.array([r["fam"] for r in rows])
    m = (grp == "round") & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5)
    PHI = _basis(x[m], y[m])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[m][:, None], -PHI]), D[m], rcond=None)
    u = 1.0 / np.clip(_basis(x, y) @ w[:6], 0.15, 3.0)
    X = np.column_stack([V * u, V2 * u ** 2, D * u,
                         np.sqrt(np.clip(A, 0, None)) * D * u, A])
    r_eff = np.sqrt(np.clip(A, 0, None) / np.pi)
    sc = ((cx - r_eff > 24) & (cx + r_eff < 296) & (cy - r_eff > 20)
          & (cy + r_eff < 220) & (z <= 4.2)
          & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5))
    return _fit_group_pred(X[sc], f[sc], grp[sc])


def _load_ours_cnc():
    rows = json.loads((_DG / "features_cnc_full.json").read_text())
    a = lambda k: np.array([r[k] for r in rows])
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    grp = np.array([r["group"] for r in rows])
    inner = (x > 5) & (x < 13) & (y > 4) & (y < 12)
    PHI = _basis(x[inner], y[inner])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[inner][:, None], -PHI]),
                            a("maxd")[inner], rcond=None)
    u = 1.0 / np.clip(_basis(x, y) @ w[:6], 0.15, 3.0)
    X = np.column_stack([a("vol") * u, a("vol2") * u ** 2, a("maxd") * u,
                         a("area"),
                         np.sqrt(np.clip(a("area"), 0, None)) * a("maxd") * u])
    return _fit_group_pred(X[inner], f[inner], grp[inner])


def _load_ours_feats():
    # the frames force_eval_all actually scored (written by ds_feats), so the
    # panel and the refreshed table are the same 390 frames
    src = CACHE / "feats_rows.json"
    rows = json.loads((src if src.exists()
                       else _DG / "features_feats.json").read_text())
    a = lambda k: np.array([r[k] for r in rows])
    X = np.column_stack([a("vol"), a("vol2"), a("maxd"), a("area"), a("h1")])
    return _fit_group_pred(X, a("f"), np.array([r["group"] for r in rows]))


if __name__ == "__main__":
    dataset_figures()
    for p in build_pages():
        print(p)
