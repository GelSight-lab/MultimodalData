"""Build the static results site (HF Space) from eval results + assets."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .dexforce import STIFFNESS_N_PER_M
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
details { margin:10px 0; }
details summary { cursor:pointer; color:var(--accent2); font-size:.85rem;
  font-weight:600; padding:8px 12px; background:#f4f0ea; border-radius:8px; }
details[open] summary { border-radius:8px 8px 0 0; }
.twocol { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:760px){ .twocol { grid-template-columns:1fr; }
header h1{font-size:1.4rem}
table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{min-width:100px} }
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


def _force_row(key: str) -> dict:
    """One refreshed dataset row out of force_matrix.json.

    The index used to hard-code these four numbers, which is how a landing
    page ends up a release behind its own results page.
    """
    d = json.loads((OUT_ROOT / "feature_cache" / "force_matrix.json"
                    ).read_text())["datasets"]
    for k, v in d.items():
        if k.lower().startswith(key.lower()):
            return v
    raise KeyError(key)


def build(m1: list[dict], m2: list[dict], assets: dict[str, str],
          ext: dict | None = None) -> Path:
    """assets: name -> relative path of copied figure/video files."""
    fm = {k: _force_row(k) for k in ("Mini CNC", "FoTa", "FEATS", "Sparsh")}
    m1_snr = [r["snr"] for r in m1]
    m1_rho = [r["spearman_vs_intensity"] for r in m1]
    inv_all = max(r["invariance_max_offset_m"] for r in m2)
    rt_all = max(r["roundtrip_max_err_n"] for r in m2)
    pen_max = max(r["penetration_max_mm"] for r in m2)
    pens = sorted(r["penetration_p50_mm"] for r in m2)
    pen_p50 = pens[len(pens) // 2]
    ext = ext or {}

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recovering Force Actions for the React Dataset</title>
<style>{CSS}</style></head><body>
<header><div class="wrap">
<h1>Recovering Force-Related Actions for the React Tactile Dataset</h1>
<p class="sub">React records GelSight-Mini images and OptiTrack poses — but, like
all UMI-style data, no applied force: demonstrated pose equals achieved pose, so
replaying the actions presses with zero intended force. Two methods put it back:
tactile-estimated normal force, and DexForce-style force-informed action targets.</p>
<div class="pills"><span class="pill">GelSight Mini (markerless gel)</span>
<span class="pill">no F/T sensor</span><span class="pill">no force ground truth</span>
<span class="pill">dataset: yxma/React</span>
<a class="pill" href="method.html" style="text-decoration:none;color:#ffd9a0;border:1px solid rgba(255,255,255,.3)">→ method design</a>
<a class="pill" href="results.html" style="text-decoration:none;color:#ffd9a0;border:1px solid rgba(255,255,255,.3)">→ results matrix</a>
<a class="pill" href="reconstruction.html" style="text-decoration:none;color:#ffd9a0;border:1px solid rgba(255,255,255,.3)">→ how force is estimated</a>
<a class="pill" href="actions.html" style="text-decoration:none;color:#ffd9a0;border:1px solid rgba(255,255,255,.3)">→ action processing deep-dive</a>
<a class="pill" href="recon_workbench.html" style="text-decoration:none;color:#ffd9a0;border:1px solid rgba(255,255,255,.3)">→ 3D reconstruction workbench</a></div>
</div></header>
<div class="wrap">

<h2>The negative result that shaped everything</h2>
<div class="card negative">
<p style="margin-top:0"><b>FEATS (U-net, FEA-labeled) does not transfer:</b> it is
trained on marker-dot gel, React's sensors are markerless — on our frames it
returns its no-contact output regardless of contact. Verified in-domain at
ρ&nbsp;=&nbsp;0.96 on its own dataset, so this is a domain effect, not a weak model
(<a href="results.html">full matrix</a>).</p>
<img src="{assets['feats_domain_gap']}" alt="training sensor vs our sensor">
<p class="footnote">Left: FEATS training gel (63 marker dots). Right: React gel,
markerless.</p>
</div>

<h2>Method 1 — physics: depth reconstruction → normal force</h2>
<div class="card method">
<div class="flow">
<span class="step">GelSight frame</span><span class="arr">→</span>
<span class="step">dI = img − ref<br>(difference image)</span><span class="arr">→</span>
<span class="step">per-sensor RGB LUT<br>→ surface gradients</span><span class="arr">→</span>
<span class="step">Poisson<br>integration</span><span class="arr">→</span>
<span class="step">indentation δ</span><span class="arr">→</span>
<span class="step">F(vol, area, max δ)</span>
</div>
<p>Zero training frames from our rig; the lookup table is self-calibrated from
spherical presses (classic Dong/Yuan calibration, LUT-v2). Every design
decision was driven by a measured defect — the full pipeline,
the optimization journey, and the step-by-step reconstruction debug are on the
<a href="method.html"><b>method page</b></a> and the
<a href="debug_pipeline.html"><b>pipeline debug page</b></a>.</p>
<img src="{assets['depth_validation_panel']}" alt="raw | diff | depth">
<p class="footnote">Strongest motherboard presses: raw | difference |
LUT-reconstructed depth. More examples (20 panels, 10 clips) in the
<a href="gallery.html">gallery</a>; every stage of the reconstruction, with
its knobs, in the <a href="recon_workbench.html">3D workbench</a>. On marker
gels the depth and 3D products get one extra step — the dots are inpainted out
of reference and frame before differencing, which removes the dimple lattice
from the geometry but is deliberately <b>not</b> applied to the force
features, where it costs 0.04 ρ.</p>
</div>

<h2>How accurate is the geometry itself?</h2>
<div class="card method">
<p style="margin-top:0">Measured against exact per-pixel ground truth for the
first time — ray-cast mesh depth on 420 Tactile MNIST touches, non-spherical
geometry a sphere calibration cannot self-validate. The answer is a range set
by press depth, not a single number:</p>
<table>
<tr><th>press depth</th><th>0.30 mm</th><th>0.60 mm</th><th>1.00 mm</th>
<th>1.50 mm</th><th>2.25 mm</th></tr>
<tr><td>depth MAE</td><td><b>11 µm</b></td><td>35 µm</td><td>68 µm</td>
<td>127 µm</td><td>281 µm</td></tr>
<tr><td>peak recovered</td><td>1.00</td><td>0.97</td><td>0.77</td><td>0.68</td>
<td>0.55</td></tr>
</table>
<p>At 0.3 mm, with no per-frame alignment and no fitted indentation scale, we
are below every published 3D Cal figure; by 2.25 mm we recover barely half the
peak. <b>The working range is shallow contact, and no accuracy number should
be quoted without its press depth.</b> The same run cut our own over-doming
claim to +7–12%, demoted unobserved LUT bins to a minor factor, and confirmed
that the valid mask is halo-dominated (IoU 0.614, over-segmentation 0.531) —
details and the Taxim caveat on the
<a href="results.html">results page</a>.</p>
</div>

<h2>Validated against four force-labeled datasets</h2>
<div class="card method">
<table>
<tr><th>dataset (gel)</th><th>ours</th><th>shuffle control</th>
<th>FEATS U-net</th><th>FeelAnyForce</th></tr>
<tr><td>FEATS val (marker)</td><td>{fm['FEATS']['rho']:.2f}</td>
<td>{fm['FEATS']['shuffle_rho']:+.2f}</td><td><b>0.96</b> in-domain</td><td>0.43</td></tr>
<tr><td>FoTa cnc_Mini (markerless)</td><td>{fm['FoTa']['rho']:.2f} (in view)</td>
<td>{fm['FoTa']['shuffle_rho']:+.2f}</td><td>0.07</td><td><b>0.83</b></td></tr>
<tr><td>cnc_mini_26 (markerless)</td><td><b>{fm['Mini CNC']['rho']:.2f}</b></td>
<td>{fm['Mini CNC']['shuffle_rho']:+.2f}</td><td>0.04</td><td>0.90</td></tr>
<tr><td>Sparsh / Meta (markerless, 10 pads)</td>
<td><b>{fm['Sparsh']['rho']:.2f}</b> *</td>
<td>{fm['Sparsh']['shuffle_rho']:+.2f}</td>
<td colspan="2">not run — no published predictions</td></tr>
<tr><td>React (no GT — agreement)</td><td colspan="4">physics vs FeelAnyForce ρ = 0.91; both read ≈0 N off-contact</td></tr>
</table>
<p>Each network dominates its own gel domain and collapses
outside it; the physics pipeline ({min(v['rho'] for v in fm.values()):.2f}–0.99)
is the only one that works everywhere. Every row is a per-group half/half fit
with isotonic calibration, 5 seeds, reported beside the same protocol with the
force labels shuffled <i>within</i> each group. Predicted-vs-ground-truth
scatters, per dataset, on the
<a href="results.html"><b>results page</b></a>.</p>
<p class="footnote" style="margin-bottom:0">* Sparsh is a <b>foreign sensor</b>.
Our cnc_mini_26 table reaches 0.878 there; rebuilding the table from Sparsh's own
sphere presses gives <b>0.968 (MAE 0.042 N)</b> and turns a bilobed
reconstruction into a true dome. That is <b>calibrate once per sensor</b>, not
zero-shot — and absolute newtons still do not transfer across indenter shapes.
Details on the <a href="results.html">results page</a>.</p>
</div>

<h2>Method 2 — force-informed action targets</h2>
<div class="card method2">
<p class="formula">p<sub>target</sub> = p<sub>observed</sub> + (F̂<sub>n</sub> / k) · n̂,
&nbsp;&nbsp; k = @@K_NM@@ N/m</p>
<p>The estimated force becomes a virtual position target past the contact
surface (DexForce-style): the action stays a pose, free space is untouched, and
an impedance controller reproduces the demonstrated force at deployment. The
transform is loss-free — invariance {inv_all:.0e} m, roundtrip {rt_all:.0e} N,
penetration median {pen_p50:.1f} mm. Interactive walkthrough on the
<a href="actions.html"><b>actions page</b></a>.</p>
<img src="{assets['dexforce_mb']}" alt="virtual target offsets">
</div>

<h2>Takeaway</h2>
<div class="card">
<ul style="margin:0">
<li><b>Labelling React</b>: FeelAnyForce as primary labeller (strongest on
markerless Mini), the physics pipeline as an independent audit — they agree at
ρ = 0.91 and cannot share failure modes.</li>
<li><b>Training</b>: use force-informed targets (Method 2) with the estimated
force as an auxiliary head; absolute newtons are coarse (±25–40%), trends are
reliable.</li>
<li><b>New gel / new sensor</b>: the physics pipeline is the only estimator
that works out of the box.</li>
</ul>
</div>

<h2>Debug log</h2>
<div class="card">
<details><summary>10 things that actually went wrong, and their fixes (click to expand)</summary>
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
<tr><td>fed the full camera view to a depth MLP trained on the SDK's 15%-cropped view</td>
<td>same 1/7 border crop for React frames and FEATS images — geometry now identical
between inference and validation</td></tr>
<tr><td>FEATS transfer started at ρ = 0.42: illumination tilt dwarfed real indentation,
dot imprints leaked into depth</td>
<td>per-frame robust background-plane removal + marker inpainting → ρ = 0.70;
quadratic background tried and rejected (absorbed contact)</td></tr>
<tr><td>shear-loaded captures anti-correlate (ρ = −0.15)</td>
<td>scoped the estimator to normal loading; documented as a limitation (React pushing
involves shear — treat high-shear force estimates with caution)</td></tr>
<tr><td>theoretical Winkler scale off by ~40× (depth-unit + gel-constant assumptions)</td>
<td>absolute scale fitted on FEATS ground truth; React forces in FEATS-calibrated
newtons, uncertainty stated</td></tr>
</table></details>
</div>

<p class="footnote">Data: <a href="https://huggingface.co/datasets/yxma/React">
yxma/React</a> · code ships with the dataset (<code>preprocess/</code>,
<code>toolbox/</code>) · references: FARM (2510.13324), DexForce (2501.10356),
FEATS (2411.03315), ACP (2410.09309), gsrobotics SDK.</p>
</div></body></html>"""
    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    page = page.replace("@@K_NM@@", f"{STIFFNESS_N_PER_M:.0f}")
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
    ext = {}
    ext_path = R / "feats_validation_val.json"
    if ext_path.exists():
        ext = {k: v for k, v in json.loads(ext_path.read_text()).items()
               if k != "per_frame"}

    src = R / "site_assets"
    names = {
        "feats_domain_gap": "feats_domain_gap.png",
        "feats_validation": "feats_validation.png",
        "fota_validation": "fota_validation.png",
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
    assets["m1_table"] = (
        "<details><summary>per-episode table — 72 sensor-sides "
        "(click to expand)</summary>" + _m1_table(m1) + "</details>")
    assets["m2_table"] = (
        "<details><summary>per-episode table — 72 sensor-sides "
        "(click to expand)</summary>" + _m2_table(m2) + "</details>")
    return build(m1, m2, assets, ext)


def rebuild_from_cache() -> Path:
    """Re-emit index.html from the cached eval.json (no episode re-runs).

    Assets are assumed to be in place under site/assets — used when only a
    figure or the page template changed, e.g. the LUT-v2 asset refresh."""
    data = json.loads((SITE / "eval.json").read_text())
    names = {
        "feats_domain_gap": "feats_domain_gap.png",
        "depth_validation_panel": "depth_validation_panel.png",
        "dexforce_mb": "dexforce_motherboard_episode_000_left.png",
    }
    assets = {k: f"assets/{v}" for k, v in names.items()}
    return build(data["method1"], data["method2"], assets)


if __name__ == "__main__":
    print(rebuild_from_cache())
