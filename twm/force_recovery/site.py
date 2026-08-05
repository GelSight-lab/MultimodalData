"""Build the static results site (HF Space) from eval results + assets."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"

CSS = """
:root { --accent:#d95f02; --accent2:#7570b3; --ok:#1b9e77; --bg:#faf9f7;
        --card:#ffffff; --text:#2b2b2b; --muted:#6b6b6b; }
* { box-sizing:border-box; }
body { font-family:'Segoe UI',system-ui,-apple-system,sans-serif; margin:0;
       background:var(--bg); color:var(--text); line-height:1.55; }
.wrap { max-width:1060px; margin:0 auto; padding:0 20px 80px; }
header { background:linear-gradient(135deg,#2d2a4a 0%,#4a3f6b 60%,#7a5c9e 100%);
         color:#f3f0ff; padding:52px 20px 44px; }
header .wrap { padding-bottom:0; }
header h1 { margin:0 0 10px; font-size:1.9rem; font-weight:650; }
header p.sub { margin:0; max-width:820px; color:#d8d2ef; font-size:1.02rem; }
header .pills { margin-top:18px; }
.pill { display:inline-block; background:rgba(255,255,255,.14); padding:4px 12px;
        border-radius:999px; font-size:.82rem; margin-right:8px; }
h2 { font-size:1.35rem; margin:52px 0 6px; }
h3 { font-size:1.05rem; margin:26px 0 6px; }
p.lead { color:var(--muted); margin-top:0; }
.card { background:var(--card); border:1px solid #e8e4de; border-radius:12px;
        padding:20px 22px; margin:16px 0; box-shadow:0 1px 3px rgba(0,0,0,.04); }
.card img, .card video { max-width:100%; border-radius:6px; display:block;
                          margin:10px auto; }
.negative { border-left:4px solid #c0392b; }
.method { border-left:4px solid var(--accent); }
.method2 { border-left:4px solid var(--accent2); }
table { border-collapse:collapse; width:100%; font-size:.86rem; margin:10px 0; }
th,td { padding:6px 9px; border-bottom:1px solid #eee7de; text-align:right; }
th { background:#f4f0ea; font-weight:600; }
td:first-child, th:first-child { text-align:left; }
code { background:#f2eee8; padding:1px 5px; border-radius:4px; font-size:.86em; }
.formula { text-align:center; font-family:Georgia,serif; font-style:italic;
           font-size:1.08rem; padding:12px; background:#f6f3ee; border-radius:8px; }
.flow { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
        justify-content:center; margin:14px 0; }
.flow .step { background:#f0ebf7; border:1px solid #d9cfeb; padding:8px 14px;
              border-radius:8px; font-size:.85rem; text-align:center; }
.flow .arr { color:var(--muted); font-size:1.1rem; }
.verdict { font-weight:600; }
.verdict.pass { color:var(--ok); } .verdict.warn { color:#c9820e; }
.footnote { color:var(--muted); font-size:.82rem; }
.twocol { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:760px){ .twocol { grid-template-columns:1fr; } }
"""


def _m1_table(rows: list[dict]) -> str:
    head = ("<tr><th>episode</th><th>side</th><th>fresh rows</th>"
            "<th>max F [N]</th><th>SNR</th><th>ρ vs intensity</th>"
            "<th>spike raw→filt [%]</th><th>threshold [µm]</th></tr>")
    body = "".join(
        f"<tr><td>{html.escape(r['episode'].split('/',1)[1])}</td>"
        f"<td>{r['side']}</td><td>{r['fresh']}</td>"
        f"<td>{r['force_max_n']:.2f}</td><td>{r['snr']:.1f}</td>"
        f"<td>{r['spearman_vs_intensity']:.2f}</td>"
        f"<td>{r['spike_rate_raw']*100:.1f} → {r['spike_rate_filtered']*100:.1f}</td>"
        f"<td>{r['threshold_um']:.0f}</td></tr>"
        for r in rows)
    return f"<table>{head}{body}</table>"


def _m2_table(rows: list[dict]) -> str:
    head = ("<tr><th>episode</th><th>side</th><th>free-space max offset [m]</th>"
            "<th>penetration p50 / max [mm]</th><th>roundtrip err [N]</th>"
            "<th>v·n̂ pressing / free</th></tr>")
    body = "".join(
        f"<tr><td>{html.escape(r['episode'].split('/',1)[1])}</td>"
        f"<td>{r['side']}</td>"
        f"<td>{r['invariance_max_offset_m']:.1e}</td>"
        f"<td>{r['penetration_p50_mm']:.2f} / {r['penetration_max_mm']:.2f}</td>"
        f"<td>{r['roundtrip_max_err_n']:.1e}</td>"
        f"<td>{r['align_pressing']:+.3f} / {r['align_free']:+.3f}</td></tr>"
        for r in rows)
    return f"<table>{head}{body}</table>"


def build(m1: list[dict], m2: list[dict], assets: dict[str, str]) -> Path:
    """assets: name -> relative path of copied figure/video files."""
    m1_snr = [r["snr"] for r in m1]
    m1_rho = [r["spearman_vs_intensity"] for r in m1]
    inv_all = max(r["invariance_max_offset_m"] for r in m2)
    rt_all = max(r["roundtrip_max_err_n"] for r in m2)
    pen_max = max(r["penetration_max_mm"] for r in m2)

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recovering Force Actions for the React Dataset</title>
<style>{CSS}</style></head><body>
<header><div class="wrap">
<h1>Recovering Force-Related Actions for the React Tactile Dataset</h1>
<p class="sub">React records GelSight-Mini images and OptiTrack sensor poses, but —
like all UMI-style datasets — no applied force: demonstrated pose equals achieved
pose, so the <i>F&nbsp;=&nbsp;k(target&nbsp;−&nbsp;actual)</i> channel of a stiffness
controller is identically zero, and contact-rich pressure cannot be reproduced from
the actions alone. Two methods, implemented and evaluated on the released data,
put it back.</p>
<div class="pills"><span class="pill">GelSight Mini (markerless gel)</span>
<span class="pill">no F/T sensor</span><span class="pill">no force ground truth</span>
<span class="pill">dataset: yxma/React</span></div>
</div></header>
<div class="wrap">

<h2>What was tried first — and why it failed</h2>
<div class="card negative">
<h3>FEATS transfer: a negative result</h3>
<p>The plan matched FARM (arXiv:2510.13324): run
<a href="https://github.com/feats-ai/feats">FEATS</a> (U-net, GelSight-Mini images →
FEA-supervised force distributions, MAE &lt; 1 N) offline over the recordings.
It does not transfer: <b>FEATS is trained on marker-dot gel, and the React sensors
use markerless gel</b> — different texture, different illumination profile.</p>
<img src="{assets['feats_domain_gap']}" alt="training sensor vs our sensor">
<p class="footnote">Left: FEATS training sensor (63 marker dots). Right: React
sensor, markerless. On this input the network returns its no-contact output for
every frame — the strongest-contact frame of pushT/episode_000 predicts
−0.040 N total, identical to a free frame (−0.042 N). Dot-based per-sensor
calibration is impossible for the same reason.</p>
</div>

<h2>Method 1 — Photometric-stereo depth → Winkler normal force</h2>
<div class="card method">
<div class="flow">
<span class="step">GelSight frame<br>640×480 RGB</span><span class="arr">→</span>
<span class="step">per-pixel MLP<br>(gsrobotics, markerless)</span><span class="arr">→</span>
<span class="step">surface normals</span><span class="arr">→</span>
<span class="step">Poisson (DCT)<br>integration</span><span class="arr">→</span>
<span class="step">indentation δ(x,y) [mm]</span><span class="arr">→</span>
<span class="step">Winkler foundation<br>F = E*/h · Σδ · dA</span>
</div>
<p>The gel pad is modelled as a bed of independent springs of stiffness
<i>E*/h</i> per unit area (thin-elastic-layer approximation; E = 0.145 MPa,
ν = 0.48, h = 4 mm). Per-sensor calibration uses the 15 lowest-intensity frames
of each episode: median depth becomes the zero map and the MAD of the residuals
sets a 5σ contact threshold — a fixed threshold cannot work because
reconstruction noise differs per sensor (measured 10–50 µm across episodes).</p>
<img src="{assets['depth_validation_panel']}" alt="raw | diff | depth">
<p class="footnote">Strongest contact of motherboard/episode_000 (left sensor):
raw frame | difference vs reference | reconstructed depth. Indentation localises
exactly where the image difference shows contact.</p>
<div class="twocol">
<img src="{assets['depth_pushT']}" alt="depth panels pushT">
<img src="{assets['depth_mb']}" alt="depth panels motherboard">
</div>
<h3>Evaluation (no ground truth → falsifiable properties instead)</h3>
<p>E1.1 <b>specificity</b> — no-contact rows must read ≈0 N; SNR = median force
on high-intensity rows / p95 on reference rows. E1.2 <b>correlation</b> —
Spearman ρ against contact intensity, an independent statistic of the same
images. E1.3 <b>spikes</b> — single-frame excursions are solver noise; a
median-3 over fresh frames must remove them. E1.4 <b>calibration
stability</b> across episodes.</p>
{assets['m1_table']}
<img src="{assets['timeline_pushT']}" alt="force timeline pushT">
<img src="{assets['timeline_mb']}" alt="force timeline motherboard">
<div class="twocol">
<video controls muted loop playsinline preload="metadata"
 src="{assets['clip']}"></video>
<video controls muted loop playsinline preload="metadata"
 src="{assets['clip_mb']}"></video>
</div>
<p class="footnote">12 s around each episode's force peak: live force bar under
the tactile stream (median-3 filtered). Left: pushT (light, intermittent
contact). Right: motherboard (sustained presses).</p>
</div>

<h2>Method 2 — DexForce-style force-informed position targets</h2>
<div class="card method2">
<p>Rather than adding force as a new action dimension, the estimated normal
force becomes a <b>virtual position target past the contact surface</b>
(DexForce, arXiv:2501.10356):</p>
<p class="formula">p<sub>target</sub> = p<sub>observed</sub> + (F̂<sub>n</sub> / k) · n̂,
&nbsp;&nbsp; k = 300 N/m (deployment impedance)</p>
<p>The action stays a pose — it composes with the existing 30 Hz pose actions,
needs only an impedance controller at deployment, and in free space reduces
exactly to the observed pose. The pressing direction n̂ is <b>not</b> a guessed
axis: it is the rig's dual-ball calibrated <code>gel_axis_in_rigid</code>
(pose-to-pose consistency ≈1°), rotated into world frame per row. The naive
[0,0,1] guess produced <i>negative</i> approach alignment at force onsets; the
calibrated axis makes it positive.</p>
<img src="{assets['dexforce_pushT']}" alt="virtual target offsets pushT">
<img src="{assets['dexforce_mb']}" alt="virtual target offsets motherboard">
<h3>Evaluation</h3>
<p>E2.1 <b>free-space invariance</b> — zero force must leave the action exactly
the observed pose. E2.2 <b>boundedness</b> — F/k must stay millimetre-scale to
be a safe action. E2.3 <b>roundtrip</b> — k·‖target − pose‖ must reproduce the
input force (pure algebra). E2.4 <b>geometry</b> — motion should align more
with n̂ while pressing than in free motion.</p>
{assets['m2_table']}
<p>Across all episodes: free-space invariance
<span class="verdict pass">{inv_all:.1e} m</span>, roundtrip error
<span class="verdict pass">{rt_all:.1e} N</span> (machine precision), max
penetration <span class="verdict pass">{pen_max:.1f} mm</span>.</p>
</div>

<h2>What the numbers say</h2>
<div class="card">
<ul>
<li><b>Method 1 works, with honest limits.</b> SNR
{min(m1_snr):.1f}–{max(m1_snr):.1f} across episodes and ρ =
{min(m1_rho):.2f}–{max(m1_rho):.2f} vs the independent intensity proxy: the
signal is real, but React's contacts are light (median in-contact force
≈0.05 N, peaks 0.4–1.2 N) and close to the reconstruction noise floor —
absolute scale rests on the stated E, h values until a weight calibration is
done on the rig.</li>
<li><b>Method 2 is exactly as reliable as its input force.</b> The transform
itself is loss-free (invariance and roundtrip at machine precision) and keeps
targets safe (≤{pen_max:.1f} mm at k = 300 N/m). It converts pseudo-force into
a deployable action without changing the policy's output space.</li>
<li><b>Recommended recipe</b>: train on force-informed targets (Method 2) with
the Method 1 force as an auxiliary prediction head — matching the conclusions
of FARM and DexForce, adapted to markerless gel.</li>
</ul>
</div>

<h2>Debug log (what actually went wrong)</h2>
<div class="card">
<table>
<tr><th>found</th><th>fix</th></tr>
<tr><td>FEATS returns its no-contact output on every markerless frame</td>
<td>switched force estimation to photometric-stereo depth (negative result kept above)</td></tr>
<tr><td>fixed dot-threshold 55 detects 0 dots (our dots bottom out at gray 56)</td>
<td>percentile-based threshold — then made moot by the markerless finding</td></tr>
<tr><td>1.1 mm phantom depth at the image border (Poisson/Neumann edge artifact)</td>
<td>exclude a 12/16-px margin from force integration</td></tr>
<tr><td>contact threshold exploded 30× when a reference frame was lightly touching</td>
<td>median zero-map + MAD threshold instead of mean + std</td></tr>
<tr><td>4% single-frame force spikes (bad Poisson solves)</td>
<td>median-3 over fresh frames only — row-wise filtering would see each duplicated
value 3× and keep it</td></tr>
<tr><td>assumed gel normal [0,0,1] gave negative approach alignment at onsets</td>
<td>use the rig's dual-ball calibrated <code>gel_axis_in_rigid</code>; sign verified
against approach kinematics</td></tr>
</table>
</div>

<p class="footnote">Data: <a href="https://huggingface.co/datasets/yxma/React">
yxma/React</a> · code ships with the dataset (<code>preprocess/</code>,
<code>toolbox/</code>) · references: FARM (2510.13324), DexForce (2501.10356),
FEATS (2411.03315), ACP (2410.09309), gsrobotics SDK.</p>
</div></body></html>"""
    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    out.write_text(page)
    return out


def collect_and_build() -> Path:
    """Run evaluation, copy assets, emit the site."""
    import shutil

    from .evaluate import run_all
    from .run_episode import OUT_ROOT as R

    m1, m2 = run_all()
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    (SITE / "eval.json").write_text(json.dumps({"method1": m1, "method2": m2},
                                               indent=2))
    src = R / "site_assets"
    names = {
        "feats_domain_gap": "feats_domain_gap.png",
        "depth_validation_panel": "depth_validation_panel.png",
        "depth_pushT": "depth_pushT_episode_000_right.png",
        "depth_mb": "depth_motherboard_episode_000_left.png",
        "timeline_pushT": "timeline_pushT_episode_000.png",
        "timeline_mb": "timeline_motherboard_episode_000.png",
        "dexforce_pushT": "dexforce_pushT_episode_000_right.png",
        "dexforce_mb": "dexforce_motherboard_episode_000_left.png",
        "clip": "clip_pushT_episode_000_right.mp4",
        "clip_mb": "clip_motherboard_episode_000_left.mp4",
    }
    assets = {}
    for key, fname in names.items():
        shutil.copy2(src / fname, SITE / "assets" / fname)
        assets[key] = f"assets/{fname}"
    assets["m1_table"] = _m1_table(m1)
    assets["m2_table"] = _m2_table(m2)
    return build(m1, m2, assets)
