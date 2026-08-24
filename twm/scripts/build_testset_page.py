"""Example page for the probe test set: the README's claims, shown.

A README can assert that the overlay lands on the sensor. A page with 72 of
them lets a reader check. Static stills, not clips: the question here is where
ground truth IS, not how it moves — the clips already exist under /probes/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path("/media/yxma/Disk1/twm/probe_testset")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/yxma/Disk1/twm/testset_page")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(SRC / "overlays", out / "overlays")
    shutil.copy(SRC / "overlay_example.jpg", out / "overlay_example.jpg")

    man = json.loads((SRC / "manifest.json").read_text())
    runs = [json.loads((SRC / p["meta"]).read_text()) for p in man["probes"]]
    allp = [q for r in runs for q in r["probes"]]
    tr = [q["amplitude"] for q in allp if q["amplitude_unit"] == "m"]
    ro = [q["amplitude"] for q in allp if q["amplitude_unit"] == "deg"]
    pc = [q["speed_percentile"] for q in allp]
    sep = min(q["min_separation_m"] for q in allp)
    px = man["overlay_error_budget_px"]["camera_reprojection"]

    cards = [
        (f"{len(allp)}", f"probes &middot; {len(runs)} start frames"),
        (f"{min(tr):.2f}&ndash;{max(tr):.2f} m",
         f"translation &middot; {min(ro):.0f}&ndash;{max(ro):.0f}&deg; rotation"),
        (f"p{min(pc):.0f}&ndash;p{max(pc):.0f}", "speed vs the dataset"),
        (f"~{max(px.values()):.0f} px", "worst reprojection noise floor"),
        (f"{sep:.3f} m", "closest the hands come (rule 0.12)"),
    ]
    sec = []
    for r in runs:
        tiles = "".join(
            f"<figure><img loading='lazy' src='overlays/run{r['run']}_{q['name']}.jpg'>"
            f"<figcaption>{q['name']} &middot; {q['amplitude']:g}"
            f"{'m' if q['amplitude_unit'] == 'm' else '&deg;'} &middot; "
            f"{q['horizon_s']:.2f}s &middot; p{q['speed_percentile']:.0f}</figcaption>"
            f"</figure>" for q in r["probes"])
        sec.append(
            f"<h2>run{r['run']} &mdash; {r['episode']}, rows "
            f"{r['context_rows'][0]}&ndash;{r['context_rows'][-1]} &nbsp;"
            f"<small>moving {r['moving_side']}, holding {r['held_side']}</small></h2>"
            f"<div class='grid'>{tiles}</div>")

    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>React probe test set</title><style>
:root{--bg:#0b1020;--fg:#e8eefb;--dim:#8ea0c2;--line:#1e2a45;--card:#111a2e;--accent:#ffc46b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 'IBM Plex Sans',system-ui,sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:28px 20px 70px}
h1{font-size:28px;margin:0 0 8px}h2{font-size:16px;color:var(--accent);margin:30px 0 8px}
h2 small{color:var(--dim);font-weight:400}
p{color:var(--dim);max-width:78ch}code{background:#0d1526;padding:1px 5px;border-radius:4px}
a{color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:22px;color:var(--accent);font-weight:650}
.card span{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:7px}
img{width:100%;border-radius:5px;display:block}
figcaption{color:var(--dim);font-size:12px;margin-top:5px}
.hero{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px;margin:18px 0}
</style></head><body><div class="wrap">
<h1>React probe test set</h1>
<p>__N__ commanded action sequences over __R__ start frames, for scoring a tactile
world model's rollouts against ground truth that is <b>geometric, not photometric</b>.
Nobody performed these motions, so there is no ground-truth future image &mdash; what
is ground truth is where the sensor <i>would be</i>, and its projection into each
camera. Full method and usage in the
<a href="README.md">README</a>; per-probe clips are under
<a href="../probes/index.html">/probes/</a>.</p>
<div class="hero"><img src="overlay_example.jpg">
<figcaption style="color:var(--dim);font-size:13px;margin-top:8px">Yellow: commanded
ground truth. Red: a deliberately wrong rollout, offset 25&nbsp;mm in world x &mdash;
it reads 18&ndash;19&nbsp;px against a ~6&nbsp;px noise floor. Dimmed: the hand that
must stay still.</figcaption></div>
<div class="cards">__CARDS__</div>
<p><b>Start frames come only from 2026-05-10 and 2026-05-11.</b> 2026-05-19 is
excluded: its OptiTrack world was redefined and the yaw about the table normal
remains unmeasured to &plusmn;2.3&deg;, which is 16&nbsp;px at the workspace &mdash;
three times the noise floor. Projected ground truth is the point of this set, so a
session whose projection carries an unstated bias is left out.</p>
__SECS__
</div></body></html>"""
    html = (html.replace("__N__", str(len(allp)))
                .replace("__R__", str(len(runs)))
                .replace("__CARDS__", "".join(
                    f"<div class='card'><b>{a}</b><span>{b}</span></div>" for a, b in cards))
                .replace("__SECS__", "".join(sec)))
    (out / "index.html").write_text(html)
    shutil.copy(SRC / "README.md", out / "README.md")
    print(f"{len(allp)} overlays -> {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
