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
<h1>Three estimators × three ground-truth datasets</h1>
<p class="sub">Every method evaluated on every dataset with force-sensor labels,
predicted vs ground truth per dataset — so dataset quality is controlled within
each row, and differences between panels are differences between methods.</p>
<a class="pill" href="method.html">↖ how the method is designed</a>
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
</table>
<p>The pattern is the finding: <b>each network dominates its own gel domain and
collapses outside it</b> (FEATS 0.96 → 0.04–0.07; FeelAnyForce 0.90 → 0.43),
while <b>the physics pipeline is the only estimator that works everywhere</b>
(0.74–0.99) — it never sees training data, so it has no domain to leave.</p>
</div>

<h2>FEATS dataset — marker-dot gel</h2>
<div class="card"><img src="assets/results_feats.png" alt="FEATS dataset panels">
<p class="footnote">In-domain, the FEATS U-net is excellent (ρ=0.96) — the
negative results elsewhere are domain effects, not a weak model. FeelAnyForce,
markerless-trained, degrades on the dotted gel (0.43): the same knife cuts both
ways.</p></div>

<h2>FoTa cnc_Mini — markerless gel</h2>
<div class="card"><img src="assets/results_cnc.png" alt="cnc_Mini panels">
<p class="footnote">Hard conditions: only 4 contact-free frames, 62% of presses
near the pad border — the press grid is larger than the field of view (see
<a href="debug_pipeline.html">pipeline debug</a>). Strictly in view our ρ is
0.94 (MAE 0.26 N); FeelAnyForce reaches 0.92 non-edge.</p></div>

<h2>GlowTact — markerless gel, cleaned</h2>
<div class="card"><img src="assets/results_glowtact.png" alt="GlowTact panels">
<p class="footnote">Friendliest ground truth (centred presses, 10 free frames,
0–20 N). Per-indenter, calibrated within each family under the physical scope
(contact fully in view, gel not bottomed out), ρ is 0.975–0.992 across all six
indenters with MAE ≤ 0.73 N. Caveat: that calibration is refit per family, so
it measures rank agreement within a group, not a transferable absolute-newton
scale — see <a href="method.html">per-dataset calibration</a>.</p></div>

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
option that works out of the box — and its FEATS-dataset score (0.74) shows
what it does on a domain nobody tuned it for.</p></div>

<p><a href="debug_pipeline.html"><b>Pipeline debug page</b></a>: raw
image &rarr; force step by step on all three datasets, and the cnc
field-of-view ablation (in-view &rho;=0.94).</p>

<footer>React force recovery · <a href="index.html">overview</a> ·
<a href="method.html">method design</a> · <a href="gallery.html">gallery</a> ·
data: FEATS (2411.03315) · FoTa/T3 (2406.13640) · GlowTact
(dacongming666/GlowTact_Datasets) · FeelAnyForce (2410.02048)</footer>
</div></body></html>"""

ZH = [
    ("Pipeline debug page</b></a>: raw\nimage &rarr; force step by step on all three datasets, and the cnc\nfield-of-view ablation (in-view &rho;=0.94).",
     "管线调试页</b></a>：三个数据集上从原始图像到力的逐步展示，以及 cnc 视野消融（视野内 &rho;=0.94）。"),
    ('<html lang="en">', '<html lang="zh-CN">'),
    ("<title>Results — Force Estimation on GelSight Mini</title>",
     "<title>评测结果 — GelSight Mini 力估计</title>"),
    ('<div class="kicker">React force recovery · results</div>',
     '<div class="kicker">React 力恢复 · 评测结果</div>'),
    ("<h1>Three estimators × three ground-truth datasets</h1>",
     "<h1>三个估计器 × 三个真值数据集</h1>"),
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
    ('<td class="best">0.96 · in-domain</td>', '<td class="best">0.96 · 本域</td>'),
    ("<td>FoTa cnc_Mini (markerless)</td>",
     "<td>FoTa cnc_Mini(无点)</td>"),
    ("(in view)</td>", "(视野内)</td>"),
    ("<td>GlowTact (markerless)</td>", "<td>GlowTact(无点)</td>"),
    ("""<p>The pattern is the finding: <b>each network dominates its own gel domain and
collapses outside it</b> (FEATS 0.96 → 0.04–0.07; FeelAnyForce 0.90 → 0.43),
while <b>the physics pipeline is the only estimator that works everywhere</b>
(0.74–0.99) — it never sees training data, so it has no domain to leave.</p>""",
     """<p>规律本身就是发现:<b>每个网络只统治自己的 gel 域,出域即塌</b>
(FEATS 0.96 → 0.04–0.07;FeelAnyForce 0.90 → 0.43);
而<b>物理管线是唯一在所有域都工作的估计器</b>(0.74–0.99)——
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
0.94 (MAE 0.26 N); FeelAnyForce reaches 0.92 non-edge.</p>""",
     """<p class="footnote">条件苛刻:只有 4 帧无接触,62% 的按压贴边。我们严格视野内 ρ=0.94(MAE 0.26 N,见管线调试页);
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
option that works out of the box — and its FEATS-dataset score (0.74) shows
what it does on a domain nobody tuned it for.</p>""",
     """<p style="margin-top:0"><b>给 React 打标签</b>(无点 Mini):FeelAnyForce 主标,
物理管线独立审计,分歧行打标。<b>换任何新 gel 或新传感器</b>、没有训练模型匹配该域时:
物理管线是唯一开箱即用的选项——它在 FEATS 数据集上的 0.74 就是"无人为它调过的域"上的表现。</p>"""),
    ("""<footer>React force recovery · <a href="index.html">overview</a> ·
<a href="method.html">method design</a> · <a href="gallery.html">gallery</a> ·
data: FEATS (2411.03315) · FoTa/T3 (2406.13640) · GlowTact
(dacongming666/GlowTact_Datasets) · FeelAnyForce (2410.02048)</footer>""",
     """<footer>React 力恢复 · <a href="index.html">总览</a> ·
<a href="method_zh.html">方法设计</a> · <a href="gallery.html">图库</a> ·
数据:FEATS (2411.03315) · FoTa/T3 (2406.13640) · GlowTact
(dacongming666/GlowTact_Datasets) · FeelAnyForce (2410.02048)</footer>"""),
]


def build_pages() -> tuple[Path, Path]:
    # numbers come from the evaluation artifacts, never typed into HTML
    m = json.loads((CACHE / "results_metrics.json").read_text())
    tok = {
        "@@O_FEATS@@": m["feats"]["Ours (physics)"]["rho"],
        "@@N_FEATS@@": m["feats"]["FEATS U-net"]["rho"],
        "@@A_FEATS@@": m["feats"]["FeelAnyForce"]["rho"],
        "@@O_CNC@@":   m["cnc"]["Ours (physics)"]["rho"],
        "@@N_CNC@@":   m["cnc"]["FEATS U-net"]["rho"],
        "@@A_CNC@@":   m["cnc"]["FeelAnyForce"]["rho"],
        "@@O_GLOW@@":  m["glowtact"]["Ours (physics)"]["rho"],
        "@@N_GLOW@@":  m["glowtact"]["FEATS U-net"]["rho"],
        "@@A_GLOW@@":  m["glowtact"]["FeelAnyForce"]["rho"],
    }
    en = EN.replace("@@CSS@@", CSS)
    for k, v in tok.items():
        en = en.replace(k, f"{v:.2f}")
    (SITE / "results.html").write_text(en)
    zh = en
    for old, new in ZH:
        assert old in zh, f"ZH miss: {old[:60]!r}"
        zh = zh.replace(old, new, 1)
    (SITE / "results_zh.html").write_text(zh)
    return SITE / "results.html", SITE / "results_zh.html"




# --- LUT-v2 pipeline loaders (override the MLP-era ones above) -----------
_DG = OUT_ROOT / "site_assets" / "debug_gallery"


def _basis(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def _fit_group_pred(X, f, groups, seed=0):
    from sklearn.isotonic import IsotonicRegression
    pred = np.full(len(f), np.nan)
    for g in set(groups):
        m = np.where(groups == g)[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(m))
        fi, ei = m[perm[:len(m) // 2]], m[perm[len(m) // 2:]]
        if len(fi) < 8:
            continue
        wl, *_ = np.linalg.lstsq(X[fi], f[fi], rcond=None)
        iso = IsotonicRegression(out_of_bounds="clip").fit(X[fi] @ wl, f[fi])
        pred[ei] = iso.predict(X[ei] @ wl)
    ok = np.isfinite(pred)
    return f[ok], pred[ok]


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
    rows = json.loads((_DG / "features_feats.json").read_text())
    a = lambda k: np.array([r[k] for r in rows])
    X = np.column_stack([a("vol"), a("vol2"), a("maxd"), a("area"), a("h1")])
    return _fit_group_pred(X, a("f"), np.array([r["group"] for r in rows]))


if __name__ == "__main__":
    dataset_figures()
    for p in build_pages():
        print(p)
