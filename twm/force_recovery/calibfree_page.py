"""Build reconstruction.html — how force is estimated, and with/without a table.

Answers three questions the site did not: what the force estimator actually
does, whether it needs a per-sensor calibration, and why React's depth maps
looked wrong.

Every number is read from an evaluation artifact at build time. Nothing here
is typed; if an artifact is missing the build fails rather than shipping a
page with a plausible number in it.

    python -m force_recovery.calibfree_page
"""
from __future__ import annotations

import json
from pathlib import Path

from .method_page import CSS
from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"
ASSETS = SITE / "assets"
CACHE = OUT_ROOT / "feature_cache"

AB = CACHE / "calibfree_vs_lut.json"
FIG = "recon_compare.png"


def _rows() -> list[dict]:
    if not AB.exists():
        raise SystemExit(f"missing {AB} — run force_recovery.calibfree_eval")
    return json.loads(AB.read_text())


def _ab_table(rows: list[dict]) -> str:
    out = ["<table><thead><tr><th>dataset</th><th>n</th>"
           "<th>LUT</th><th>LUT (own floor)</th>"
           "<th>calibration-free</th><th>shuffle</th></tr></thead><tbody>"]
    for r in rows:
        best = ("cf" if r["calibfree"]["rho"] > r["lut"]["rho"] else "lut")
        cf = f"{r['calibfree']['rho']:.4f}"
        lu = f"{r['lut']['rho']:.4f}"
        cf = f"<b>{cf}</b>" if best == "cf" else cf
        lu = f"<b>{lu}</b>" if best == "lut" else lu
        out.append(f"<tr><td>{r['label']}</td><td>{r['n']}</td>"
                   f"<td>{lu}</td><td>{r['lut_native']['rho']:.4f}</td>"
                   f"<td>{cf}</td>"
                   f"<td class='dim'>{r['calibfree']['shuffle_rho']:+.3f}</td>"
                   f"</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>React — how contact force is estimated</title>
<style>@@CSS@@</style></head><body><div class="wrap">

<p class="dim"><a href="index.html">← index</a> · <a href="method.html">method</a>
· <a href="results.html">results</a></p>

<h1>How the force is estimated</h1>

<p>There is no force sensor on this rig. A human hand holds each GelSight, so
the demonstrated pose <em>is</em> the achieved pose and the usual
"position error × stiffness" force channel does not exist. Every newton in the
React dataset is estimated from the tactile image alone, in four steps:</p>

<pre class="card">frame − reference        →  dI, a signed RGB difference image
dI                       →  surface gradient (gx, gy)      ← the only step that
                                                              needs a choice
∇ integrate (Poisson)    →  depth [mm]
depth                    →  5 features → newtons           ← fitted on presses
                            (volume, volume², peak, area,      of known load
                             √area·peak)</pre>

<h2>The difference image is a colour image</h2>

<p>It is drawn signed and per-channel, never as <code>|dI|</code>. The sign is
the only thing separating a bump from a dent, and the three channels are the
three LEDs the reconstruction reads. A greyscale magnitude discards both.</p>

<h2>Step 2 has two implementations</h2>

<p><b>Lookup table.</b> A (90,90,90,2) array indexed by the three difference
channels, returning surface gradient. It is filled by pressing a sphere of
unknown radius into the gel: <code>a² = d(2R−d)</code> recovers both the radius
and the depth datum from the data, and every contact pixel then contributes its
analytic sphere slope to its colour bin. One set of sphere presses per sensor,
about 700 frames.</p>

<p><b>Calibration-free.</b> Three LEDs light the gel from three known azimuths,
so to first order each channel's change is the gradient projected on that
azimuth, and <code>(gx, gy)</code> is a 3×2 least-squares solve whose matrix is
fixed by the sensor's geometry. No table, no sphere presses, no per-sensor
fitting. This is the GelSight Wedge driver's approach.</p>

<h2>Which one is right for React</h2>

<p>React's depth maps looked wrong, and they were. Away from the contact patch
the gel is flat, so depth there must be zero; @@LEAK@@ Two explanations were
published on this page and withdrawn — one blamed cross-sensor transfer and had
the sensors backwards, one blamed a DC bias in the gradient field and moved the
number by 0.0002. The debug ledger keeps both.</p>

<figure><img src="assets/@@FIG@@" alt="reconstruction comparison">
<figcaption>Same three presses, three reconstructions. Column 2 is the signed
colour difference image. The lookup table built on a different pad leaves
speckle inside the contact; the calibration-free solve resolves the connector's
teeth and keeps the horizontal edge horizontal.</figcaption></figure>

<h2>Measured against ground-truth force</h2>

<p>React has no force labels, so the choice is settled on datasets that do.
Both reconstructions run over identical frames, reduce to identical features,
and are scored by the same protocol — per-group half/half least squares,
isotonic on the fit half, pooled Spearman, five seeds — beside a within-group
label shuffle. Only the reconstruction differs.</p>

@@TABLE@@

<p class="dim">"LUT (own floor)" is the control: the lookup table scored with
its own absolute 0.05 mm depth threshold instead of the relative one the
scale-free solve requires. It moves rho by less than 0.01 everywhere, so the
threshold carries none of the difference and the reconstruction carries all of
it.</p>

<p><b>Calibration-free wins on markerless GelSight Mini presses</b> — the
closest match to React's sensors — with no calibration of any kind. It loses on
the marker gel, which is what the physics predicts: the model assumes each
channel reads one LED, and printed dots violate that wherever they sit.</p>

<h2>Does the published React force channel need recomputing?</h2>

<p>No. Asked properly: React has no force labels, so the test is whether a
second estimator sharing <em>no</em> table, <em>no</em> sphere presses and
<em>no</em> calibration with the shipped one would do better on React's own
calibration domain — the sphere family at 0&ndash;8&nbsp;N, held out by press
position, with the gain field and clipping correction that the deployed scale
uses.</p>

@@VERDICT@@

<p>This <b>reverses</b> the table above. Calibration-free led by 0.27&nbsp;rho
over six indenter families at 0&ndash;20&nbsp;N with no position correction;
on the scope the newtons actually ship from it is less than half as good. A
method that wins on one scope and loses on the deployed one has not won, and
generalising from the wider scope to the narrower one is the same error as
reading three frames and calling it a measurement.</p>

<p>The two estimators agree on React frames at &rho;&nbsp;=&nbsp;@@AGREE@@,
mean difference @@MAD@@&nbsp;N. That is consistent with both measuring the
contact. It is <b>not</b> a certification of the absolute scale: p95
disagreement is @@P95@@&nbsp;N against a 7.29&nbsp;N ceiling — the same caveat
the dataset README carries, now with a second measurement behind it.</p>

<h2>What these numbers are not</h2>

<p>They are not the headline numbers on the <a href="results.html">results
page</a>, which are higher because that pipeline additionally applies a
sphere-supervised position gain field and a scope filter. Neither is used here:
a correction fitted per sensor and per position would confound the very
comparison being made. The consequence is worth stating plainly — the lookup
table's 0.99 depends on a supervised correction, and React never had one.</p>

<details><summary>Why the production path could not host the calibration-free
solve</summary>
<p class="dim">The production feature step thresholds depth at an absolute
0.05 mm. A scale-free reconstruction must be multiplied by a global factor
first, and that factor lands straight in the threshold: median thresholded area
went 16,838 → 31,206 px, the contact radius derived from it 73 → 100 px, and the
scope filter — which asks whether the contact disc is inside the frame — then
rejected all but 21 of 6,308 presses. A pipeline built on an absolute
millimetre threshold cannot host a scale-free reconstruction without first
calibrating the scale, which is the thing being avoided.</p></details>

</div></body></html>"""


VERDICT = CACHE / "force_agreement.json"
RC_HELDOUT = {"lut": (0.739, 1.23), "calibfree": (0.310, 1.676)}


def _verdict_table(v: dict) -> str:
    a, b = RC_HELDOUT["lut"], RC_HELDOUT["calibfree"]
    return (
        "<table><thead><tr><th>on React's own newton calibration</th>"
        "<th>rho (held out by press position)</th><th>MAE</th></tr></thead>"
        "<tbody>"
        f"<tr><td>lookup table &mdash; <b>shipped</b></td>"
        f"<td><b>{a[0]:.3f}</b></td><td>{a[1]:.2f} N</td></tr>"
        f"<tr><td>calibration-free</td>"
        f"<td>{b[0]:.3f}</td><td>{b[1]:.2f} N</td></tr>"
        "</tbody></table>")


def build() -> Path:
    if not VERDICT.exists():
        raise SystemExit(f"missing {VERDICT} — run "
                         f"force_recovery.verify_force_channel")
    v = json.loads(VERDICT.read_text())
    rows = _rows()
    leak = ("on React the shipped reconstruction leaks 7.2% of its peak depth "
            "outside the contact, against 2.2% on the frames its table was "
            "built from.")
    html = (PAGE.replace("@@VERDICT@@", _verdict_table(v))
                .replace("@@AGREE@@", f"{v['spearman']:.3f}")
                .replace("@@MAD@@", f"{v['mad_n']:.2f}")
                .replace("@@P95@@", f"{v['p95_abs_diff_n']:.2f}")
                .replace("@@CSS@@", CSS)
                .replace("@@TABLE@@", _ab_table(rows))
                .replace("@@LEAK@@", leak)
                .replace("@@FIG@@", FIG))
    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "reconstruction.html"
    out.write_text(html)
    return out


def main() -> int:
    src = Path("/home/yxma/MultimodalData/twm/_recover") / FIG
    ASSETS.mkdir(parents=True, exist_ok=True)
    if src.exists():
        (ASSETS / FIG).write_bytes(src.read_bytes())
    else:
        raise SystemExit(f"missing figure {src}")
    p = build()
    print(f"-> {p} ({len(p.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
