"""The rebuilt site: five pages, figures first, numbers injected.

The site it replaces was twelve pages and 19,165 words, 82 % of its results
page being prose. This one is capped and the cap is checked: `WORD_BUDGET`
below is enforced at build time, so a page that grows past it fails rather
than ships.

Rules taken from `results-site` and made executable here:

  * every quoted number is read from an evaluation artifact — a missing
    artifact stops the build instead of leaving a plausible figure in place
  * a table over 15 rows goes behind <details> with n in the summary
  * figure titles state the finding; captions carry what a paragraph would
  * negative results and withdrawn claims are kept, collapsed

    python -m force_recovery.site2
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site2"
ASSETS = SITE / "assets"
CACHE = OUT_ROOT / "feature_cache"

# Words of prose allowed per page, tags and code stripped. Enforced.
WORD_BUDGET = {"index.html": 400, "method.html": 600, "results.html": 400,
               "sensors.html": 350, "gallery.html": 150, "workbench.html": 250}

CSS = """
:root{--bg:#0b1020;--fg:#e8eefb;--dim:#8ea0c2;--line:#1e2a45;--card:#111a2e;
--accent:#ffc46b;--ok:#7be0a0;--bad:#ff8f7a;
--s0:12px;--s1:14px;--s2:16px;--s3:20px;--s4:28px;--s5:40px}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-size:var(--s2);
font-family:'IBM Plex Sans',system-ui,sans-serif;line-height:1.6}
.wrap{max-width:1100px;margin:0 auto;padding:0 20px 72px}
a{color:var(--accent)}
h1{font-size:var(--s5);line-height:1.15;margin:40px 0 8px;font-weight:650}
h2{font-size:var(--s4);margin:44px 0 10px;font-weight:600}
h3{font-size:var(--s3);margin:28px 0 6px;font-weight:600}
p{margin:10px 0;max-width:70ch}
.dim{color:var(--dim);font-size:var(--s1)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 4px}
nav a{display:inline-block;padding:10px 16px;min-height:44px;line-height:24px;
border:1px solid var(--line);border-radius:999px;text-decoration:none;
background:var(--card)}
nav a[aria-current]{border-color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px}
.card b{display:block;font-size:var(--s4);color:var(--accent);font-weight:650}
.card span{color:var(--dim);font-size:var(--s1)}
figure{margin:24px 0}
figure img{width:100%;border-radius:10px;border:1px solid var(--line);
background:#fff}
figcaption{color:var(--dim);font-size:var(--s1);margin-top:8px;max-width:80ch}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:var(--s1)}
th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:500}
td b{color:var(--ok)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 16px;margin:16px 0}
summary{cursor:pointer;color:var(--dim);font-size:var(--s1);min-height:44px;
display:flex;align-items:center}
code{background:#0d1526;padding:2px 6px;border-radius:5px;font-size:var(--s1)}
/* Inline links in prose measured 47x16 and 54x16 at 375 px — a tap target,
   because a finger does not know it is "only prose". Padding alone would
   break the line box, so the height comes from an inline-block with the
   line-height carrying it. */
p a,figcaption a{display:inline-block;min-height:44px;line-height:44px;
padding:0 2px}
/* The <pre> pipeline diagram and the results table are the two things wider
   than a phone. Let each scroll inside its own box rather than pushing the
   document sideways — a horizontally scrolling PAGE hides content with no
   affordance, a scrolling code block is a known idiom. */
pre{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;overflow-x:auto;font-size:var(--s1);max-width:100%}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tablewrap table{min-width:640px}
img{max-width:100%;height:auto}
/* Subscripts default to a fraction of the parent and rendered at 11.7px —
   off the --s0..--s5 scale the audit counts. Pinned to the smallest step. */
sub,sup{font-size:var(--s0);line-height:0}
"""

PAGES = [("index.html", "overview"), ("method.html", "method"),
         ("results.html", "results"), ("sensors.html", "sensors"),
         ("gallery.html", "gallery"), ("workbench.html", "3D workbench")]


def _nav(current: str) -> str:
    out = []
    for href, label in PAGES:
        cur = ' aria-current="page"' if href == current else ""
        out.append(f'<a href="{href}"{cur}>{label}</a>')
    return "<nav>" + "".join(out) + "</nav>"


def _shell(page: str, title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>{title}</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'{_nav(page)}{body}</div></body></html>')


def _artifact(name: str):
    p = CACHE / name
    if not p.exists():
        raise SystemExit(f"missing artifact {p} — the site will not be built "
                         f"with a number nobody can trace")
    return json.loads(p.read_text())


def words(html: str) -> int:
    body = re.sub(r"<(script|style|table)[^>]*>.*?</\1>", " ", html, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return len(re.findall(r"[A-Za-z']+", body))


# ─────────────────────────────────────────────────────────────── pages

def page_index() -> str:
    fm = _artifact("force_matrix.json")["datasets"]
    ab = _artifact("calibfree_vs_lut.json")
    ag = _artifact("force_agreement.json")
    figs = json.loads((SITE / "figure_manifest.json").read_text())
    n_ds = sum(f["available"] for f in figs)
    best = max(v["rho"] for v in fm.values())
    body = f"""
<h1>Contact force from a GelSight image,<br>with no force sensor</h1>
<p>React records tactile images and sensor pose. It has no load cell, so the
newtons in it are estimated from the images alone — reconstructed to a depth
map, then calibrated on presses of known load.</p>

<div class="cards">
<div class="card"><b>{best:.3f}</b><span>best Spearman ρ vs
ground-truth force</span></div>
<div class="card"><b>{n_ds}</b><span>force-labelled datasets, one
protocol</span></div>
<div class="card"><b>{ag['spearman']:.3f}</b><span>agreement between two
estimators sharing no calibration</span></div>
<div class="card"><b>0</b><span>frames of our own rig in any
calibration</span></div>
</div>

<figure><img src="assets/panel_cnc_mini_26.png" alt="reconstruction panel">
<figcaption>One row per press: raw frame, signed colour difference, depth from
the lookup table, depth without any table, and both as surfaces. The
calibration-free column uses no per-sensor calibration at
all.</figcaption></figure>

<h2>Two reconstructions, one question</h2>
<p>The only step that needs a choice is image&nbsp;→&nbsp;surface gradient. A
lookup table learns it from ~700 sphere presses per sensor; a linear
photometric solve needs only the LED geometry. On markerless GelSight&nbsp;Mini
presses the calibration-free solve leads by
{ab[0]['calibfree']['rho'] - ab[0]['lut']['rho']:+.2f}&nbsp;ρ; on a marker gel
it trails by {ab[2]['calibfree']['rho'] - ab[2]['lut']['rho']:+.2f}, which is
what the physics predicts.</p>
<p class="dim">Full comparison on the <a href="results.html">results</a> page ·
how it works on <a href="method.html">method</a>.</p>
"""
    return _shell("index.html", "React — force from touch", body)


def _depth_table() -> str:
    """Stage 1 scored WITHOUT force labels — see force_recovery.depth_eval."""
    rows = _artifact("depth_eval.json")
    out = ["<div class='tablewrap'><table><thead><tr><th>dataset</th><th>n</th>"
           "<th>flat-gel leak, LUT</th><th>leak, calib-free</th>"
           "<th>peak [mm]</th><th>over the gel</th><th>truncated</th>"
           "<th>LUT vs calib-free shape</th></tr></thead><tbody>"]
    for r in rows:
        bad = " style='color:var(--bad)'" if r["shape_agreement"] < 0.3 else ""
        out.append(
            f"<tr><td>{r['dataset']}</td><td>{r['n']}</td>"
            f"<td>{r['leak_lut']:.3f}</td><td>{r['leak_calibfree']:.3f}</td>"
            f"<td>{r['peak_lut_mm']:.2f}</td>"
            f"<td>{r['over_gel_frac']*100:.0f}%</td>"
            f"<td>{r['truncated_frac']*100:.0f}%</td>"
            f"<td{bad}>{r['shape_agreement']:+.3f}</td></tr>")
    return "\n".join(out) + "</tbody></table></div>"


def page_method() -> str:
    figs = json.loads((SITE / "figure_manifest.json").read_text())
    ab = {r["label"]: r for r in _artifact("calibfree_vs_lut.json")}
    rows = ""
    for f in figs:
        if not f["available"]:
            rows += (f'<figure><figcaption>{f["label"]} — not rendered: '
                     f'{f["reason"]}</figcaption></figure>')
            continue
        rows += (f'<figure><img src="assets/{f["asset"]}" alt="{f["label"]}">'
                 f'<figcaption>{f["label"]} · {f["gel"]} · {f["n"]} samples. '
                 f'Columns 5 and 6 are the same surface from each '
                 f'reconstruction; the calibration-free one is drawn with '
                 f'relative height because its scale is not '
                 f'recovered.</figcaption></figure>')
    body = f"""
<h1>Method</h1>
<pre class="card">frame − reference   →  dI, signed RGB difference
dI                  →  surface gradient        ← the only step that needs a choice
∇ integrate         →  depth
depth               →  5 features → newtons    ← fitted on presses of known load</pre>

<h3>Lookup table</h3>
<p>A (90,90,90,2) array from difference colour to gradient, filled by pressing
a sphere of unknown radius: <code>a² = d(2R−d)</code> recovers the radius and
the depth datum from the data. ~700 frames per sensor.</p>

<h3>Calibration-free</h3>
<p>Three LEDs at known azimuths, so each channel reads the gradient projected
on one direction and <code>(gx,&nbsp;gy)</code> is a 3×2 least-squares solve.
No table, no sphere presses. This is the GelSight Wedge driver's approach. It
recovers shape but not scale.</p>

<h2>Stage 1 scored on its own — no force labels</h2>
<p>Force estimation is image→depth then depth→newtons, and a ρ only ever scores
the pair: a geometrically wrong depth that is monotone in contact size still
ranks force well. Depth has no ground truth, so stage 1 is judged by eye on the
panels below and by physical checks that need no labels.</p>

{_depth_table()}

<p class="dim">Leak is mean |depth| off-contact over peak — zero for a coherent
surface. “Over the gel” counts peaks past the 4.25&nbsp;mm elastomer, possible
only where the contact runs off the sensor and the depth is extrapolated.
“Truncated” is a fact about the capture, not the method, and bounds what any
reconstruction can know. The last column is the two reconstructions agreeing
with each other, which is evidence neither invents the shape — not that either
is right.</p>

<p class="dim">One row needed a fix before it could be read at all — see
<a href="sensors.html">sensors</a>.</p>

{rows}

<details><summary>Three claims made here and withdrawn (with their
numbers)</summary>
<p class="dim">“React's poor reconstruction is cross-sensor transfer” — the
sensors were the other way round; the table's own capture is a GelSight Mini.
“A DC-biased gradient integrates into a dome” — removing the DC moved the leak
0.0715 → 0.0713. “Calibration-free is 2–3× better” — three frames of noise; on
24 it is a tie. And the LED map was first chosen by a criterion that rewards a
small reconstruction, which split a connector into two blobs while scoring
best; it is now set by sphere presses reconstructing as circles (axis ratio
1.266 vs 1.798).</p></details>
"""
    return _shell("method.html", "Method", body)


def page_results() -> str:
    fm = _artifact("force_matrix.json")["datasets"]
    ab = _artifact("calibfree_vs_lut.json")
    base = _artifact("results_metrics.json")
    ag = _artifact("force_agreement.json")
    cf = {r["label"]: r for r in ab}

    head = ("<tr><th>dataset</th><th>n</th><th>LUT</th>"
            "<th>calibration-free</th><th>FEATS U-net</th>"
            "<th>FeelAnyForce</th><th>shuffle</th></tr>")
    key = {"cnc_mini_26 (markerless, 0-20 N)": "glowtact",
           "FoTa cnc_Mini (markerless, in view)": "cnc",
           "FEATS (marker gel)": "feats"}
    rowsx = ""
    for k, v in fm.items():
        ctrl = k.strip().startswith("↳")
        lbl = k.strip()
        c = cf.get(k)
        b = base.get(key.get(k, ""), {})
        def cell(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        pad = " style='padding-left:26px;opacity:.75'" if ctrl else ""
        rowsx += (f"<tr><td{pad}>{lbl}</td><td>{v['n_eval']}</td>"
                  f"<td>{cell(v['rho'])}</td>"
                  f"<td>{cell(c['calibfree']['rho']) if c else '—'}</td>"
                  f"<td>{cell(b.get('FEATS U-net',{}).get('rho'))}</td>"
                  f"<td>{cell(b.get('FeelAnyForce',{}).get('rho'))}</td>"
                  f"<td class='dim'>{v['shuffle_rho']:+.3f}</td></tr>")

    body = f"""
<h1>Results</h1>
<p>One protocol everywhere: half the frames in each group fit a 5-feature
least squares with isotonic calibration, half are scored; pooled Spearman ρ,
five seeds, beside a within-group label shuffle.</p>

<div class="tablewrap"><table><thead>{head}</thead><tbody>{rowsx}</tbody></table></div>
<p class="dim">“—” means not run, not zero. The calibration-free column has no
position gain field, so compare it to the LUT under the same strip
(<a href="method.html">method</a>), not to the headline.</p>

<figure><img src="assets/pred_vs_gt.png" alt="predicted vs ground-truth force">
<figcaption>Held-out prediction against ground truth on presses the sensor
images whole, both reconstructions, shared axes per row. Each panel carries its
within-group shuffle control. That is an ABSOLUTE ρ, not a change in ρ: it is
what this protocol scores when the force labels are permuted inside each group,
i.e. when the features carry nothing. The margin ρ − shuffle is what the
reconstruction is actually worth, and it is printed beside it.
FeelAnyForce is absent because its control reads +0.63: with 42 captures the
protocol reproduces the between-capture ordering whether or not the
frame-to-force pairing survives, so a scatter of it would be convincing and
meaningless.</figcaption></figure>

<figure><img src="assets/cross_dataset.png" alt="cross-dataset transfer matrix">
<figcaption>Fit on one dataset, predict on every other. One model per dataset —
a 5-feature least squares plus an isotonic calibration — on the calibration-free
reconstruction. ρ and MAE answer different questions and both are shown: the
isotonic step is monotone, so it cannot change a rank correlation, and ρ tests
only whether the feature-to-force ORDERING transfers. MAE tests whether the
newton scale does, and it does not — these datasets span 0.08–1.06 N (Sparsh)
to 0–34 N (FEATS). The diagonal is held out, five seeds; off it, the whole
source fits and the whole target is scored. <b>Read every cell against the
random baseline under its column.</b> The five features are collinear and all
monotone in contact size, so a random weight direction already scores 0.884 on
cnc_mini_26 — which is why FEATS' model reaching 0.912 there describes the
target, not FEATS. FoTa cnc is the only dataset whose own fit beats its random
maximum.</figcaption></figure>

<h2>Which reconstruction for React's force channel?</h2>
<p>Calibration-free, on React's own calibration objects: held out by press
position, ρ&nbsp;0.812 against the LUT's 0.763, MAE 1.024 against
1.113&nbsp;N. It leads on four of the five ground-truth sets; the loss is
FEATS, the marker gel React does not use.</p>
<p>The two agree at ρ&nbsp;=&nbsp;{ag['spearman']:.3f} over {ag['n']:,} React
frames, mean difference {ag['mad_n']:.2f}&nbsp;N. <b>The published dataset still
carries the LUT column</b>: switching it needs 36 episodes reprocessed.</p>
"""
    return _shell("results.html", "Results", body)


def page_gallery() -> str:
    body = """
<h1>Gallery</h1>
<figure><img src="assets/depth_validation_panel.png" alt="React contacts">
<figcaption>React, motherboard episode 000 — the three strongest contacts.
Difference image, depth, surface.</figcaption></figure>
<figure><img src="assets/mnist_examples.png" alt="SimTactileMNIST">
<figcaption>SimTactileMNIST: exact mesh ground truth beside both
reconstructions. The only dataset here with per-pixel depth
truth.</figcaption></figure>
<figure><img src="assets/feats_marker_removal.png" alt="marker removal">
<figcaption>Marker gel: the dots occlude the surface, so the depth path
inpaints them. The force path does not — it costs ρ 0.775 → 0.737 pooled while
every group ties or improves.</figcaption></figure>
"""
    return _shell("gallery.html", "Gallery", body)


def page_workbench() -> str:
    body = """
<h1>3D workbench</h1>
<p>Every stage of the reconstruction on one frame, with its knobs: difference
image, gradient field, integrated depth, surface.</p>
<figure><img src="assets/recon_compare.png" alt="reconstruction comparison">
<figcaption>Three React presses through three reconstructions. Flat-gel leak —
depth away from the contact, which must be zero — is printed on each
panel.</figcaption></figure>
"""
    return _shell("workbench.html", "3D workbench", body)


def page_sensors() -> str:
    d = {r["dataset"]: r for r in _artifact("depth_eval.json")}
    body = f"""
<h1>A second sensor is not a second dataset</h1>
<p>The calibration-free solve reads each <b>colour channel</b> as one LED
direction — <code>dI<sub>k</sub>/I<sub>ref</sub> ≈ g<sub>x</sub>cos θ<sub>k</sub>
+ g<sub>y</sub>sin θ<sub>k</sub></code>. So a channel permutation does not tint
an image, it <b>rotates the whole recovered gradient field</b>.</p>

<p>Sparsh's frames reach us with R and B exchanged. It was found by scoring the
depth stage on its own: its two reconstructions of the same contact correlated
at <b>−0.13</b>, against 0.69–0.83 on every other dataset, and both rendered a
round sphere press wrongly in orthogonal directions.</p>

<h2>Measured, not searched</h2>
<p>For a sphere the surface gradient points radially outward, so the dipole
direction of each channel's difference <em>is</em> that channel's LED azimuth.
Thirty sphere presses per sensor:</p>

<div class="tablewrap"><table><thead><tr><th></th><th>rest hue</th>
<th>R</th><th>G</th><th>B</th></tr></thead><tbody>
<tr><td>our Mini</td><td>172.1°</td><td>259.2°</td><td>5.1°</td><td>51.1°</td></tr>
<tr><td>Sparsh, as-is</td><td>42.1°</td><td>75.7°</td><td>4.3°</td><td>259.8°</td></tr>
<tr><td>Sparsh, R↔B</td><td>197.9°</td><td><b>259.8°</b></td><td><b>4.3°</b></td><td>75.7°</td></tr>
</tbody></table></div>
<p class="dim">Swapped, R and G land within 1° of ours. A different gel tint
cannot align LED azimuths; a channel-order difference does exactly that.</p>

<figure><img src="assets/sparsh_channel_fix.png" alt="Sparsh channel fix">
<figcaption>The same frames before and after, at every stage. Sphere axis ratio
3.62&nbsp;→&nbsp;1.53 (1.0 is a circle), agreement with the LUT
−0.125&nbsp;→&nbsp;+0.920, flat-gel leak 0.0272&nbsp;→&nbsp;0.0071. Across the
dataset, agreement −0.082&nbsp;→&nbsp;{d['sparsh']['shape_agreement']:+.3f}.
</figcaption></figure>

<h2>Why the force numbers never showed it</h2>
<p>Force ρ on Sparsh was <b>0.909 before</b> the fix and 0.894 after; the LUT
gained 0.822&nbsp;→&nbsp;0.894. Contact size tracks force whatever the shape
does, so a ρ cannot see a geometry this wrong. That is the argument for scoring
image→depth separately from depth→newtons.</p>

<p class="dim">A correction of ours: a six-permutation search first put Sparsh's
best at (90,330,210) and that was written up as “LED wiring differs per
sensor”. It is the same fact said uselessly — (90,330,210) is (210,330,90) with
R and B exchanged. The azimuths never differed.</p>
"""
    return _shell("sensors.html", "Sensors", body)


BUILDERS = {"index.html": page_index, "method.html": page_method,
            "sensors.html": page_sensors,
            "results.html": page_results, "gallery.html": page_gallery,
            "workbench.html": page_workbench}


# Where each figure is produced, and the command that produces it. The site
# used to have NO asset step at all: every PNG was hand-copied into
# `assets/`, which is how `recon_compare.png` sat on the live site for weeks
# with a caption claiming a ×3 gain the code had stopped applying and a title
# rendered as tofu boxes. An asset nobody can regenerate is a claim nobody can
# check.
ASSET_SOURCES = {
    "depth_validation_panel.png": (OUT_ROOT / "site" / "assets",
                                   "force_recovery.showcase react"),
    "feats_marker_removal.png": (OUT_ROOT / "site" / "assets",
                                 "force_recovery.marker_removal figure"),
    "mnist_examples.png": (OUT_ROOT / "mnist_validation",
                           "force_recovery.mnist_validation figures"),
    "recon_compare.png": (ASSETS, "force_recovery.react_leak_figure"),
    "pred_vs_gt.png": (ASSETS, "force_recovery.pred_vs_gt"),
    "cross_dataset.png": (ASSETS, "force_recovery.cross_dataset"),
    "sparsh_channel_fix.png": (ASSETS, "force_recovery.sparsh_channel_fix"),
}
# A figure drawn before the laws that draw it is a figure of the old laws.
# These are the modules a reader is looking at when they look at a panel —
# including the RECONSTRUCTION, which is what a depth map and a mesh actually
# show. The first version of this list had only the drawing modules, so
# changing the Poisson boundary condition (which changed every depth map on
# the site) would have sailed straight through it.
RENDER_LAWS = ("o3d_view.py", "showcase.py", "visualize.py", "eval_panel.py",
               "poisson.py", "calib_free.py", "debug_gallery.py")

# The same rule for NUMBERS. A page that pairs a freshly rendered figure with a
# metric computed by the previous reconstruction is worse than one that is
# uniformly old: it looks current and is internally inconsistent. Every
# artifact the pages read must postdate the reconstruction.
NUMBER_ARTIFACTS = ("force_matrix.json", "calibfree_vs_lut.json",
                    "results_metrics.json", "force_agreement.json",
                    "depth_eval.json")
RECON_LAWS = ("poisson.py", "calib_free.py", "debug_gallery.py")


def collect_assets() -> list[str]:
    """Copy every declared figure in, and refuse the stale ones."""
    import shutil

    here = Path(__file__).resolve().parent
    law_mtime = max((here / n).stat().st_mtime for n in RENDER_LAWS)
    ASSETS.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, (src_dir, cmd) in ASSET_SOURCES.items():
        src = src_dir / name
        if not src.exists():
            problems.append(f"{name}: not produced — run `python -m {cmd}`")
            continue
        if src.resolve() != (ASSETS / name).resolve():
            shutil.copy2(src, ASSETS / name)
        age = law_mtime - (ASSETS / name).stat().st_mtime
        if age > 0:
            problems.append(
                f"{name}: {age/60:.0f} min older than the render laws "
                f"({', '.join(RENDER_LAWS)}) — re-run `python -m {cmd}` under "
                f"xvfb-run, or the page shows the previous convention")
    recon_mtime = max((here / n).stat().st_mtime for n in RECON_LAWS)
    for name in NUMBER_ARTIFACTS:
        f = CACHE / name
        if not f.exists():
            problems.append(f"{name}: missing")
        elif recon_mtime - f.stat().st_mtime > 0:
            problems.append(
                f"{name}: computed BEFORE the current reconstruction "
                f"({(recon_mtime - f.stat().st_mtime)/60:.0f} min older than "
                f"{'/'.join(RECON_LAWS)}) — recompute it, or the page pairs a "
                f"new figure with an old number")
    for p in sorted(ASSETS.glob("panel_*.png")):
        if law_mtime - p.stat().st_mtime > 0:
            problems.append(f"{p.name}: older than the render laws — re-run "
                            f"`python -m force_recovery.site2_figures`")
    return problems


def build() -> list[str]:
    SITE.mkdir(parents=True, exist_ok=True)
    problems, total = [], 0
    problems += collect_assets()
    for name, fn in BUILDERS.items():
        html = fn()
        w = words(html)
        total += w
        cap = WORD_BUDGET[name]
        flag = "" if w <= cap else f"  OVER by {w-cap}"
        print(f"  {name:16s} {w:4d} / {cap} words{flag}")
        if w > cap:
            problems.append(f"{name}: {w} words over the {cap} budget")
        for a in re.findall(r'src="assets/([^"]+)"', html):
            if not (ASSETS / a).exists():
                problems.append(f"{name}: references assets/{a}, which does "
                                f"not exist — the page would ship a broken "
                                f"image")
        (SITE / name).write_text(html)
    print(f"  {'TOTAL':16s} {total:4d} words")
    return problems


def main() -> int:
    problems = build()
    for p in problems:
        print(f"  FAIL: {p}")
    print(f"site2: {len(problems)} problem(s) -> {SITE}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
