"""Build actions.html — the force-informed action showcase page.

A focused companion to index.html: how the pose-only actions are processed
into force-informed actions, demonstrated on real episode data with a
scrubbable live chart (no chart library — inline SVG + vanilla JS reading
the exported action_trace.json).
"""
from __future__ import annotations

import json
from pathlib import Path

from .dexforce import STIFFNESS_N_PER_M
from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"


def build() -> Path:
    ev = json.loads((SITE / "eval.json").read_text())
    m2 = ev["method2"]
    inv = max(r["invariance_max_offset_m"] for r in m2)
    rt = max(r["roundtrip_max_err_n"] for r in m2)
    pen_max = max(r["penetration_max_mm"] for r in m2)
    pens = sorted(r["penetration_p50_mm"] for r in m2)
    pen_p50 = pens[len(pens) // 2]
    n_sides = len(m2)

    page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Force-Informed Actions — React Dataset</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#0d1526; --ink:#dce7f5; --dim:#7c8db0; --grid:#1a2540;
  --pose:#4fd8e0; --force:#ffb347; --target:#ff7847; --ok:#7be0a0;
  --card:#111b31; --line:#24345;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:'IBM Plex Sans',sans-serif;line-height:1.6;font-size:16px;
  background-image:
    linear-gradient(var(--grid) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:44px 44px;}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px 90px}
h1,h2{font-family:'Fraunces',serif}
h1{font-size:2.6rem;font-weight:700;line-height:1.15;margin:0 0 14px}
h2{font-size:1.5rem;font-weight:600;margin:64px 0 10px;color:var(--force)}
h2::before{content:"§ ";color:var(--dim)}
mono,.mono{font-family:'IBM Plex Mono',monospace}
header{padding:72px 0 30px;border-bottom:1px dashed var(--line)}
.kicker{font-family:'IBM Plex Mono',monospace;font-size:.78rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--force);margin-bottom:16px}
.sub{color:var(--dim);max-width:760px;font-size:1.04rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:22px 24px;margin:18px 0;position:relative}
.card::after{content:"";position:absolute;inset:0;border-radius:10px;pointer-events:none;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
.formula{font-family:'IBM Plex Mono',monospace;text-align:center;font-size:1.15rem;
  padding:20px;background:#0a1120;border:1px solid var(--line);border-radius:8px;
  color:var(--ink);overflow-x:auto}
.formula b{color:var(--target)}
.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;justify-content:center;margin:22px 0}
.fstep{background:#0a1120;border:1px solid var(--line);border-radius:8px;padding:12px 16px;
  font-size:.83rem;text-align:center;min-width:120px;font-family:'IBM Plex Mono',monospace}
.fstep small{display:block;color:var(--dim);margin-top:4px;font-size:.72rem}
.farr{align-self:center;color:var(--force);font-size:1.2rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:12px 0;
  font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th{color:var(--dim);font-family:'IBM Plex Mono',monospace;font-size:.74rem;
  text-transform:uppercase;letter-spacing:.08em}
td:first-child,th:first-child{text-align:left}
.big{font-family:'Fraunces',serif;font-size:2rem;color:var(--ok)}
.stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:20px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.stat .lbl{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--dim);
  text-transform:uppercase;letter-spacing:.1em}
.stat .val{font-family:'Fraunces',serif;font-size:1.7rem;margin-top:4px;color:var(--ink)}
.stat .val.ok{color:var(--ok)}
img,video{max-width:100%;border-radius:8px;border:1px solid var(--line);display:block;margin:12px auto}
a{color:var(--pose)} a:hover{color:var(--force)}
code{font-family:'IBM Plex Mono',monospace;background:#0a1120;padding:2px 6px;border-radius:4px;font-size:.85em}
pre{background:#0a1120;border:1px solid var(--line);border-radius:8px;padding:16px;
  overflow-x:auto;font-size:.82rem;line-height:1.5}
pre code{background:none;padding:0}
.legend{display:flex;gap:20px;font-family:'IBM Plex Mono',monospace;font-size:.76rem;
  color:var(--dim);flex-wrap:wrap;margin:8px 2px}
.legend span::before{content:"—— ";font-weight:700}
.legend .l-pose::before{color:var(--pose)} .legend .l-force::before{color:var(--force)}
.legend .l-tgt::before{color:var(--target)}
#readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
  margin-top:12px;font-family:'IBM Plex Mono',monospace}
#readout .r{background:#0a1120;border:1px solid var(--line);border-radius:6px;padding:10px 12px}
#readout .r .k{font-size:.68rem;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
#readout .r .v{font-size:1.25rem;margin-top:2px}
#chartbox{cursor:crosshair;touch-action:none}
.hint{font-family:'IBM Plex Mono',monospace;font-size:.74rem;color:var(--dim);text-align:center;margin-top:6px}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;
  padding:3px 12px;font-size:.75rem;font-family:'IBM Plex Mono',monospace;
  color:var(--dim);margin-right:8px;margin-top:14px}
footer{margin-top:70px;padding-top:20px;border-top:1px dashed var(--line);
  color:var(--dim);font-size:.82rem}
@media (max-width:700px){h1{font-size:1.9rem}}
</style></head><body><div class="wrap">

<header>
<div class="kicker">React dataset · action processing</div>
<h1>The missing half of the action:<br>putting force back into pose-only demonstrations</h1>
<p class="sub">React's actions are computed from OptiTrack sensor poses — like all
UMI-style data, the demonstrated pose <i>is</i> the achieved pose. A stiffness
controller exerts <span class="mono">F&nbsp;=&nbsp;k·(target&nbsp;−&nbsp;actual)</span>,
so replaying these actions reproduces the motion but presses with <b>zero
intended force</b>. This page shows how each action is transformed, on real
episode data, and what the transformation is worth.</p>
<span class="pill">36 episodes · 72 sensor-sides</span>
<span class="pill">no F/T sensor involved</span>
<span class="pill"><a href="index.html" style="color:inherit;text-decoration:none">↖ full methods &amp; validation</a></span>
<span class="pill"><a href="actions_zh.html" style="color:inherit;text-decoration:none">中文版</a></span>
</header>

<h2>How an action is processed</h2>
<div class="card">
<div class="flow">
<div class="fstep">pose p<sub>t</sub><small>OptiTrack, 30 Hz</small></div>
<div class="farr">+</div>
<div class="fstep">GelSight frame<small>640×480 RGB</small></div>
<div class="farr">→</div>
<div class="fstep">F̂<sub>n</sub><small>LUT depth → force,<br>GlowTact-calibrated</small></div>
<div class="farr">→</div>
<div class="fstep">n̂ = R<sub>t</sub>·a<sub>gel</sub><small>dual-ball calibrated axis</small></div>
<div class="farr">→</div>
<div class="fstep" style="border-color:var(--target)">action target<small>virtual, past the surface</small></div>
</div>
<div class="formula">p<sub><b>target</b></sub> = p<sub>observed</sub> + ( F̂<sub>n</sub> / k ) · n̂
&nbsp;&nbsp;&nbsp;k = @@K_NM@@ N/m</div>
<p>The action stays a pose. In free space F̂<sub>n</sub>&nbsp;=&nbsp;0 and the target
<i>is</i> the observed pose — nothing changes. In contact, the target moves past
the surface along the gel normal by exactly the displacement an impedance
controller at stiffness k needs to exert the demonstrated force. No new action
dimension, no force interface at deployment — a DexForce-style transform
(arXiv:2501.10356) driven by tactile-estimated rather than measured force.</p>
</div>

<h2>Live on real data — drag across the trace</h2>
<div class="card">
<p style="margin-top:0">90 s of <span class="mono">motherboard/2026-05-10/episode_000</span>
(left sensor), centred on the strongest press. Top: estimated normal force.
Bottom: the transform's entire effect on the action — the target's offset from
the observed pose along the gel normal. The cyan zero-line <i>is</i> the
original action; the sensor itself sweeps ±170 mm through this window, which
is why the offset is drawn on its own millimetre scale.</p>
<div id="chartbox"></div>
<div class="legend"><span class="l-force">F̂ normal [N]</span>
<span class="l-pose">observed pose = zero offset (action before)</span>
<span class="l-tgt">target offset F̂/k along n̂ (action after)</span></div>
<div id="readout">
<div class="r"><div class="k">t</div><div class="v" id="ro-t">–</div></div>
<div class="r"><div class="k">F̂ normal</div><div class="v" id="ro-f" style="color:var(--force)">–</div></div>
<div class="r"><div class="k">target offset F̂/k</div><div class="v" id="ro-p" style="color:var(--target)">–</div></div>
<div class="r"><div class="k">state</div><div class="v" id="ro-s">–</div></div>
</div>
<p class="hint">drag / hover to scrub · data is the actual per-row output, not a mock-up</p>
</div>

<h2>The effect, measured</h2>
<div class="stat-row">
<div class="stat"><div class="lbl">free-space invariance</div>
<div class="val ok">__INV__ m</div>
<div class="lbl" style="margin-top:6px">max |target − pose| when F̂=0, all __NS__ sides</div></div>
<div class="stat"><div class="lbl">round-trip error</div>
<div class="val ok">__RT__ N</div>
<div class="lbl" style="margin-top:6px">k·‖target−pose‖ vs F̂ — machine precision</div></div>
<div class="stat"><div class="lbl">penetration in contact</div>
<div class="val">__P50__ mm <span style="font-size:.9rem;color:var(--dim)">median</span></div>
<div class="lbl" style="margin-top:6px">max __PMAX__ mm, i.e. the hardest __FMAX__ N press at k=__KNMM__ N/mm</div></div>
</div>
<div class="card">
<p style="margin-top:0"><b>Why this matters for training.</b> Policies trained on
raw poses learn "touch the surface and stop": the label says the fingertip halts
at the contact plane, so at deployment the controller exerts whatever residual
force tracking error happens to produce. With force-informed targets the label
itself encodes <i>how hard</i> — DexForce measured near-zero task success without
this correction and 76% with it, on kinesthetic demonstrations with measured
forces; here the same transform runs from tactile-estimated force, with the
estimator validated against FEA ground truth (ρ=0.70 pooled, 0.85 on unseen
indenter shapes — see the <a href="index.html">main page</a>).</p>
<img src="assets/dexforce_motherboard_episode_000_left.png" alt="virtual target offsets">
<video controls muted loop playsinline preload="metadata" src="assets/clip_motherboard_episode_000_left.mp4"></video>
<p class="hint">the force signal driving the action transform, live under the tactile stream</p>
</div>

<h2>Using it</h2>
<div class="card">
<pre><code># per-episode force estimates ship as npz next to the release
import numpy as np
from force_recovery.dexforce import force_informed_targets, gel_axis
from force_recovery.evaluate import median3_fresh

z = np.load("force_recovery/motherboard/2026-05-10/episode_000_left.npz")
force = median3_fresh(z["force_normal_n"], is_new)      # de-spike on fresh frames
act = force_informed_targets(pose, force, gel_axis("motherboard", "left"))
train_targets = act.target_pos                          # (T,3) — drop-in pose labels</code></pre>
<p>Caveats, stated plainly: absolute newtons carry the GlowTact-calibration
uncertainty (cross-sensor scale drifts 2–4×; within-episode relative force is
the reliable part), shear-dominant contact is a blind spot of the normal-force
estimator, and legacy recordings update tactile at ~8.5 fps — the
<code>tactile_*_is_new</code> flags mark which rows carry fresh force evidence.</p>
</div>

<footer>React force recovery · data <a href="https://huggingface.co/datasets/yxma/React">yxma/React</a>
· methods &amp; external validation on the <a href="index.html">main page</a>
· transform: DexForce (2501.10356) · force: per-sensor RGB-LUT photometric calibration + Poisson (LUT-v2), GlowTact-calibrated</footer>
</div>

<script>
fetch('assets/action_trace.json').then(r=>r.json()).then(D=>{
const W=940,H1=170,H2=250,GAP=34,PAD_L=56,PAD_R=16,H=H1+H2+GAP+40;
const N=D.force_n.length,fps=D.meta.fps;
const box=document.getElementById('chartbox');
const svgNS='http://www.w3.org/2000/svg';
const svg=document.createElementNS(svgNS,'svg');
svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.style.width='100%';
box.appendChild(svg);
const X=i=>PAD_L+(W-PAD_L-PAD_R)*i/(N-1);
const fmax=Math.max(...D.force_n)*1.08;
const Yf=v=>H1-(H1-8)*v/fmax;
// bottom panel: the OFFSET itself. The trajectory sweeps +-170mm here while
// the offset stays <=5mm — plotting raw positions hides the transform.
const smin=0,smax=Math.max(...D.pen_mm)*1.12;
const Y2=v=>H1+GAP+H2-(H2-8)*(v-smin)/(smax-smin);
function poly(ys,color,w,dash){const p=document.createElementNS(svgNS,'path');
  let d='M';for(let i=0;i<N;i++){d+=X(i).toFixed(1)+' '+ys(i).toFixed(1)+(i<N-1?' L':'');}
  p.setAttribute('d',d);p.setAttribute('fill','none');p.setAttribute('stroke',color);
  p.setAttribute('stroke-width',w);if(dash)p.setAttribute('stroke-dasharray',dash);
  svg.appendChild(p);return p;}
// axes
[[0,'0'],[fmax/1.08,fmax.toFixed(0)]].forEach(([v,t])=>{
  const l=document.createElementNS(svgNS,'text');l.textContent=t+' N';
  l.setAttribute('x',PAD_L-8);l.setAttribute('y',Yf(v)+4);l.setAttribute('text-anchor','end');
  l.setAttribute('fill','#7c8db0');l.setAttribute('font-size','11');
  l.setAttribute('font-family','IBM Plex Mono');svg.appendChild(l);});
[[0,'0 mm'],[smax/1.12,(smax/1.12).toFixed(1)+' mm']].forEach(([v,t])=>{
  const l=document.createElementNS(svgNS,'text');l.textContent=t;
  l.setAttribute('x',PAD_L-8);l.setAttribute('y',Y2(v)+4);l.setAttribute('text-anchor','end');
  l.setAttribute('fill','#7c8db0');l.setAttribute('font-size','11');
  l.setAttribute('font-family','IBM Plex Mono');svg.appendChild(l);});
// commanded-penetration band between the zero-line (observed) and the target
const band=document.createElementNS(svgNS,'path');
let d='M';for(let i=0;i<N;i++)d+=X(i).toFixed(1)+' '+Y2(D.pen_mm[i]).toFixed(1)+(i<N-1?' L':'');
d+=' L'+X(N-1).toFixed(1)+' '+Y2(0)+' L'+X(0).toFixed(1)+' '+Y2(0);
band.setAttribute('d',d+' Z');band.setAttribute('fill','rgba(255,120,71,.28)');svg.appendChild(band);
poly(i=>Yf(D.force_n[i]),'#ffb347',1.4);
poly(i=>Y2(0),'#4fd8e0',1.3);
poly(i=>Y2(D.pen_mm[i]),'#ff7847',1.3);
// cursor
const cur=document.createElementNS(svgNS,'line');
cur.setAttribute('y1',0);cur.setAttribute('y2',H1+GAP+H2);
cur.setAttribute('stroke','#dce7f5');cur.setAttribute('stroke-width','1');
cur.setAttribute('stroke-dasharray','3 3');svg.appendChild(cur);
const dotF=document.createElementNS(svgNS,'circle');dotF.setAttribute('r',4);
dotF.setAttribute('fill','#ffb347');svg.appendChild(dotF);
const dotT=document.createElementNS(svgNS,'circle');dotT.setAttribute('r',4);
dotT.setAttribute('fill','#ff7847');svg.appendChild(dotT);
function setI(i){i=Math.max(0,Math.min(N-1,i));
  const x=X(i);cur.setAttribute('x1',x);cur.setAttribute('x2',x);
  dotF.setAttribute('cx',x);dotF.setAttribute('cy',Yf(D.force_n[i]));
  dotT.setAttribute('cx',x);dotT.setAttribute('cy',Y2(D.pen_mm[i]));
  document.getElementById('ro-t').textContent=(i/fps).toFixed(2)+' s';
  document.getElementById('ro-f').textContent=D.force_n[i].toFixed(2)+' N';
  document.getElementById('ro-p').textContent=(D.pen_mm[i]).toFixed(2)+' mm';
  const s=document.getElementById('ro-s');
  if(D.force_n[i]>0.1){s.textContent='CONTACT';s.style.color='var(--target)';}
  else{s.textContent='free — target ≡ pose';s.style.color='var(--ok)';}}
function fromEvent(e){const r=svg.getBoundingClientRect();
  const px=((e.touches?e.touches[0].clientX:e.clientX)-r.left)/r.width*W;
  setI(Math.round((px-PAD_L)/(W-PAD_L-PAD_R)*(N-1)));}
['mousemove','mousedown','touchmove','touchstart'].forEach(ev=>
  box.addEventListener(ev,e=>{fromEvent(e);if(ev!=='mousemove')e.preventDefault();},{passive:false}));
setI(Math.round(N*0.5));
});
</script>
</body></html>"""
    page = (page
            .replace("__INV__", f"{inv:.1e}")
            .replace("__RT__", f"{rt:.0e}")
            .replace("__P50__", f"{pen_p50:.1f}")
            .replace("__PMAX__", f"{pen_max:.1f}")
            .replace("__FMAX__", f"{pen_max * STIFFNESS_N_PER_M / 1000.0:.1f}")
            .replace("__KNMM__", f"{STIFFNESS_N_PER_M / 1000.0:g}")
            .replace("__NS__", str(n_sides)))
    out = SITE / "actions.html"
    page = page.replace("@@K_NM@@", f"{STIFFNESS_N_PER_M:.0f}")
    out.write_text(page)
    return out


# Ordered exact replacements EN -> ZH. Every pair is asserted to hit, so an
# edit to the English template that would silently desynchronise the Chinese
# page fails the build instead.
ZH = [
    ('<html lang="en">', '<html lang="zh-CN">'),
    ("<title>Force-Informed Actions — React Dataset</title>",
     "<title>力信息化动作 — React 数据集</title>"),
    ("family=IBM+Plex+Sans:wght@400;600&display=swap",
     "family=IBM+Plex+Sans:wght@400;600&family=Noto+Serif+SC:wght@600;700"
     "&family=Noto+Sans+SC:wght@400;500&display=swap"),
    ("font-family:'IBM Plex Sans',sans-serif;line-height:1.6",
     "font-family:'IBM Plex Sans','Noto Sans SC',sans-serif;line-height:1.75"),
    ("h1,h2{font-family:'Fraunces',serif}",
     "h1,h2{font-family:'Fraunces','Noto Serif SC',serif}"),
    ('<div class="kicker">React dataset · action processing</div>',
     '<div class="kicker">React 数据集 · 动作处理</div>'),
    ("<h1>The missing half of the action:<br>putting force back into pose-only demonstrations</h1>",
     "<h1>动作缺失的另一半:<br>把力还给只有位姿的演示数据</h1>"),
    ("""<p class="sub">React's actions are computed from OptiTrack sensor poses — like all
UMI-style data, the demonstrated pose <i>is</i> the achieved pose. A stiffness
controller exerts <span class="mono">F&nbsp;=&nbsp;k·(target&nbsp;−&nbsp;actual)</span>,
so replaying these actions reproduces the motion but presses with <b>zero
intended force</b>. This page shows how each action is transformed, on real
episode data, and what the transformation is worth.</p>""",
     """<p class="sub">React 的动作由 OptiTrack 跟踪的传感器位姿计算而来——和所有
UMI 类数据一样,演示位姿<i>就是</i>实际达到的位姿。刚度控制器输出
<span class="mono">F&nbsp;=&nbsp;k·(target&nbsp;−&nbsp;actual)</span>,
因此复现这些动作只能复现运动,按压的<b>意图力恒为零</b>。
本页在真实 episode 数据上展示每个动作如何被变换,以及这个变换值多少。</p>"""),
    ('<span class="pill">36 episodes · 72 sensor-sides</span>',
     '<span class="pill">36 个 episode · 72 个传感器侧</span>'),
    ('<span class="pill">no F/T sensor involved</span>',
     '<span class="pill">全程无 F/T 传感器</span>'),
    ('>↖ full methods &amp; validation</a>', '>↖ 完整方法与验证(英文)</a>'),
    ('<a href="actions_zh.html" style="color:inherit;text-decoration:none">中文版</a>',
     '<a href="actions.html" style="color:inherit;text-decoration:none">English</a>'),
    ("<h2>How an action is processed</h2>", "<h2>动作是怎么被处理的</h2>"),
    ("<small>OptiTrack, 30 Hz</small>", "<small>OptiTrack,30 Hz</small>"),
    ("GelSight frame<small>640×480 RGB</small>", "GelSight 图像<small>640×480 RGB</small>"),
    ("<small>LUT depth → force,<br>GlowTact-calibrated</small>",
     "<small>LUT 深度 → 力,<br>GlowTact 定标</small>"),
    ("<small>dual-ball calibrated axis</small>", "<small>双球标定的 gel 轴</small>"),
    ("action target<small>virtual, past the surface</small>",
     "动作目标<small>虚拟,穿过接触面</small>"),
    ("""<p>The action stays a pose. In free space F̂<sub>n</sub>&nbsp;=&nbsp;0 and the target
<i>is</i> the observed pose — nothing changes. In contact, the target moves past
the surface along the gel normal by exactly the displacement an impedance
controller at stiffness k needs to exert the demonstrated force. No new action
dimension, no force interface at deployment — a DexForce-style transform
(arXiv:2501.10356) driven by tactile-estimated rather than measured force.</p>""",
     """<p>动作仍然是位姿。自由空间中 F̂<sub>n</sub>&nbsp;=&nbsp;0,目标<i>就是</i>观测位姿——什么都不变。
接触时,目标沿 gel 法向越过接触面,偏移量恰好是刚度为 k 的阻抗控制器要输出演示力所需的位移。
不新增动作维度,部署端不需要力控接口——即 DexForce 式变换(arXiv:2501.10356),
只是驱动它的力来自触觉估计而非力传感器测量。</p>"""),
    ("<h2>Live on real data — drag across the trace</h2>",
     "<h2>真实数据实时演示——在曲线上拖动</h2>"),
    ("""<p style="margin-top:0">90 s of <span class="mono">motherboard/2026-05-10/episode_000</span>
(left sensor), centred on the strongest press. Top: estimated normal force.
Bottom: the transform's entire effect on the action — the target's offset from
the observed pose along the gel normal. The cyan zero-line <i>is</i> the
original action; the sensor itself sweeps ±170 mm through this window, which
is why the offset is drawn on its own millimetre scale.</p>""",
     """<p style="margin-top:0"><span class="mono">motherboard/2026-05-10/episode_000</span>
(左传感器)以最强按压为中心的 90 秒。上:估计的法向力。
下:变换对动作的全部效果——目标相对观测位姿沿 gel 法向的偏移。
青色零线<i>就是</i>原始动作;这段窗口里传感器本身扫过 ±170 mm,
所以偏移单独用毫米刻度画出。</p>"""),
    ('<span class="l-pose">observed pose = zero offset (action before)</span>',
     '<span class="l-pose">观测位姿 = 零偏移(变换前的动作)</span>'),
    ('<span class="l-tgt">target offset F̂/k along n̂ (action after)</span>',
     '<span class="l-tgt">沿 n̂ 的目标偏移 F̂/k(变换后的动作)</span>'),
    ('<span class="l-force">F̂ normal [N]</span>', '<span class="l-force">法向力 F̂ [N]</span>'),
    ('<div class="k">F̂ normal</div>', '<div class="k">法向力 F̂</div>'),
    ('<div class="k">target offset F̂/k</div>', '<div class="k">目标偏移 F̂/k</div>'),
    ('<div class="k">state</div>', '<div class="k">状态</div>'),
    ('<p class="hint">drag / hover to scrub · data is the actual per-row output, not a mock-up</p>',
     '<p class="hint">拖动 / 悬停查看 · 数据为逐行真实输出,非示意图</p>'),
    ("<h2>The effect, measured</h2>", "<h2>效果,用数字说话</h2>"),
    ('<div class="lbl">free-space invariance</div>', '<div class="lbl">自由空间不变性</div>'),
    ('max |target − pose| when F̂=0, all __NS__ sides'.replace("__NS__", "{ns}"),
     'F̂=0 时 |target − pose| 的最大值,全部 {ns} 个侧'),
    ('<div class="lbl">round-trip error</div>', '<div class="lbl">往返误差</div>'),
    ("k·‖target−pose‖ vs F̂ — machine precision", "k·‖target−pose‖ 对比 F̂ —— 机器精度"),
    ('<div class="lbl">penetration in contact</div>', '<div class="lbl">接触期穿透量</div>'),
    ('>median</span>', '>中位数</span>'),
    ("max {pmax} mm, i.e. the hardest {fmax} N press at k={knmm} N/mm",
     "最大 {pmax} mm，即 k={knmm} N/mm 下最硬的 {fmax} N 按压"),
    ("""<p style="margin-top:0"><b>Why this matters for training.</b> Policies trained on
raw poses learn "touch the surface and stop": the label says the fingertip halts
at the contact plane, so at deployment the controller exerts whatever residual
force tracking error happens to produce. With force-informed targets the label
itself encodes <i>how hard</i> — DexForce measured near-zero task success without
this correction and 76% with it, on kinesthetic demonstrations with measured
forces; here the same transform runs from tactile-estimated force, with the
estimator validated against FEA ground truth (ρ=0.70 pooled, 0.85 on unseen
indenter shapes — see the <a href="index.html">main page</a>).</p>""",
     """<p style="margin-top:0"><b>为什么这对训练重要。</b>在原始位姿上训练的策略学到的是
"碰到表面就停":标签说指尖停在接触面上,部署时控制器输出的力只是跟踪误差碰巧产生的残余。
用力信息化目标后,标签本身编码了<i>按多重</i>——DexForce 在带力测量的拖动示教上测得:
不做这个修正任务成功率接近零,做了是 76%;这里同样的变换由触觉估计的力驱动,
估计器已对 FEA 真值验证(混合 ρ=0.70,未见过的按压头形状 0.85——见<a href="index.html">主页(英文)</a>)。</p>"""),
    ('<p class="hint">the force signal driving the action transform, live under the tactile stream</p>',
     '<p class="hint">驱动动作变换的力信号,与触觉流实时对齐</p>'),
    ("<h2>Using it</h2>", "<h2>怎么用</h2>"),
    ("# per-episode force estimates ship as npz next to the release",
     "# 每个 episode 的力估计以 npz 形式随 release 一起发布"),
    ("# de-spike on fresh frames", "# 只在 fresh 帧上去尖峰"),
    ("# (T,3) — drop-in pose labels", "# (T,3) —— 直接替换位姿标签"),
    ("""<p>Caveats, stated plainly: absolute newtons carry the GlowTact-calibration
uncertainty (cross-sensor scale drifts 2–4×; within-episode relative force is
the reliable part), shear-dominant contact is a blind spot of the normal-force
estimator, and legacy recordings update tactile at ~8.5 fps — the
<code>tactile_*_is_new</code> flags mark which rows carry fresh force evidence.</p>""",
     """<p>需要直说的告诫:绝对牛顿值带有 GlowTact 定标的不确定性(跨传感器刻度漂移 2–4×;
集内相对力才是可靠的部分);剪切主导的接触是法向力估计器的盲区;
旧录制的触觉有效更新率约 8.5 fps——<code>tactile_*_is_new</code>
标记了哪些行携带新的力证据。</p>"""),
    ("""<footer>React force recovery · data <a href="https://huggingface.co/datasets/yxma/React">yxma/React</a>
· methods &amp; external validation on the <a href="index.html">main page</a>
· transform: DexForce (2501.10356) · force: per-sensor RGB-LUT photometric calibration + Poisson (LUT-v2), GlowTact-calibrated</footer>""",
     """<footer>React 力恢复 · 数据集 <a href="https://huggingface.co/datasets/yxma/React">yxma/React</a>
· 方法与外部验证见<a href="index.html">主页(英文)</a>
· 变换:DexForce (2501.10356) · 力估计:逐传感器 RGB-LUT 光度标定 + Poisson(LUT-v2),GlowTact 定标</footer>"""),
    # JS-embedded UI strings
    ("s.textContent='CONTACT';", "s.textContent='接触中';"),
    ("s.textContent='free — target ≡ pose';", "s.textContent='自由 — 目标 ≡ 位姿';"),
]


def build_zh() -> Path:
    """Chinese page, derived from the freshly built English page."""
    en = build().read_text()
    # numeric placeholders inside translated strings
    import re
    ns = re.search(r"all (\d+) sides", en).group(1)
    pmax = re.search(r"max ([\d.]+) mm, i\.e\.", en).group(1)
    fmax = re.search(r"hardest ([\d.]+) N press", en).group(1)
    knmm = re.search(r"press at k=([\d.]+) N/mm", en).group(1)
    zh = en
    for old, new in ZH:
        sub = lambda t: (t.replace("{ns}", ns).replace("{pmax}", pmax)
                          .replace("{fmax}", fmax).replace("{knmm}", knmm))
        old, new = sub(old), sub(new)
        assert old in zh, f"ZH replacement missed: {old[:60]!r}"
        zh = zh.replace(old, new, 1)
    out = SITE / "actions_zh.html"
    zh = zh.replace("@@K_NM@@", f"{STIFFNESS_N_PER_M:.0f}")
    out.write_text(zh)
    return out


if __name__ == "__main__":
    print(build())
    print(build_zh())
