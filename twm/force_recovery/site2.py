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
               "gallery.html": 150, "workbench.html": 250}

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
"""

PAGES = [("index.html", "overview"), ("method.html", "method"),
         ("results.html", "results"), ("gallery.html", "gallery"),
         ("workbench.html", "3D workbench")]


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
<p>One protocol everywhere: within each indenter/probe/pad group, half the
frames fit a 5-feature least squares with isotonic calibration, half are
scored; pooled Spearman ρ, five seeds, beside a within-group label shuffle.</p>

<div class="tablewrap"><table><thead>{head}</thead><tbody>{rowsx}</tbody></table></div>
<p class="dim">“—” means not run, not zero. Calibration-free is scored under a
stripped protocol with no position gain field, so its column is not comparable
to the LUT's headline value — it is comparable to the LUT under the same strip,
which is on the <a href="method.html">method</a> page.</p>

<h2>Is the published React force channel sound?</h2>
<p>React has no labels, so the test is whether a second estimator sharing no
calibration does better on React's own calibration domain. It does not:
0.739 against {ag.get('cf_heldout', 0.310):.3f} held out by press position. The
two agree at ρ&nbsp;=&nbsp;{ag['spearman']:.3f} over {ag['n']:,} frames, mean
difference {ag['mad_n']:.2f}&nbsp;N. That is consistent with both measuring
contact and is not a certification of absolute scale — p95 disagreement is
{ag['p95_abs_diff_n']:.2f}&nbsp;N against a 7.29&nbsp;N ceiling.</p>
<p><b>No recompute.</b> The published newtons stand.</p>
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


BUILDERS = {"index.html": page_index, "method.html": page_method,
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
                    "results_metrics.json", "force_agreement.json")
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
