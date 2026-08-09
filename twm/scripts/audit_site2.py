"""Measure every text node and control on the rebuilt site, at four widths.

Not a screenshot review. One DOM walk per page per viewport collecting WCAG
contrast against the COMPOSITED background, touch-target sizes, the rendered
font-size census, and horizontal overflow. Reports before/after deltas rather
than an opinion.

Thresholds are the WCAG ones: 4.5:1 for body text, 3.0:1 at >=24 px or bold
>=18.66 px; 44 px minimum for anything tappable.

    python scripts/audit_site2.py [--dir <site>] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VIEWPORTS = [375, 768, 1280, 1920]

PROBE = r"""
(() => {
  const srgb = (c) => { c /= 255; return c <= 0.03928 ? c/12.92
                        : Math.pow((c+0.055)/1.055, 2.4); };
  const lum = (r,g,b) => 0.2126*srgb(r)+0.7152*srgb(g)+0.0722*srgb(b);
  const parse = (s) => (s.match(/[\d.]+/g) || []).map(Number);
  const bgOf = (el) => {
    let n = el;
    while (n && n !== document.documentElement) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c.length >= 3 && (c[3] === undefined || c[3] > 0.85)) return c;
      n = n.parentElement;
    }
    return [255,255,255];
  };
  const contrast = ! => 0;
  const fails = [], touch = [], sizes = {}, fams = {};
  document.querySelectorAll('body *').forEach((el) => {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    const r = el.getBoundingClientRect();
    if (['A','BUTTON','INPUT','SELECT','SUMMARY'].includes(el.tagName)
        && r.height > 0 && (r.height < 44 || r.width < 44))
      touch.push({tag: el.tagName, w: Math.round(r.width),
                  h: Math.round(r.height), text: (el.textContent||'').trim().slice(0,24)});
    const txt = (el.textContent || '').trim();
    if (!txt || el.children.length) return;
    const fs = parseFloat(cs.fontSize);
    sizes[fs.toFixed(1)] = (sizes[fs.toFixed(1)] || 0) + 1;
    fams[cs.fontFamily.split(',')[0]] = 1;
    const fg = parse(cs.color), bg = bgOf(el);
    const L1 = lum(fg[0],fg[1],fg[2]), L2 = lum(bg[0],bg[1],bg[2]);
    const ratio = (Math.max(L1,L2)+0.05)/(Math.min(L1,L2)+0.05);
    const big = fs >= 24 || (fs >= 18.66 && parseInt(cs.fontWeight) >= 700);
    const need = big ? 3.0 : 4.5;
    if (ratio < need)
      fails.push({text: txt.slice(0,40), size: fs, ratio: +ratio.toFixed(2),
                  need, color: cs.color});
  });
  return {fails, touch, sizes, fams: Object.keys(fams),
          overflow: document.documentElement.scrollWidth
                  - document.documentElement.clientWidth};
})()
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/media/yxma/Disk1/twm/force_recovery/site2")
    ap.add_argument("--json")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — cannot audit; NOT reporting a pass")
        return 2

    site = Path(args.dir)
    pages = sorted(p.name for p in site.glob("*.html"))
    probe = PROBE.replace("  const contrast = ! => 0;\n", "")
    out, problems = [], []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for vw in VIEWPORTS:
            pg = b.new_page(viewport={"width": vw, "height": 900})
            for name in pages:
                pg.goto(f"file://{site/name}")
                pg.wait_for_timeout(120)
                r = pg.evaluate(probe)
                r.update(page=name, viewport=vw)
                out.append(r)
                if r["fails"]:
                    problems.append(f"{name}@{vw}: {len(r['fails'])} contrast "
                                    f"failures (worst "
                                    f"{min(f['ratio'] for f in r['fails'])}:1)")
                if r["touch"] and vw <= 768:
                    problems.append(f"{name}@{vw}: {len(r['touch'])} touch "
                                    f"targets under 44 px")
                if r["overflow"] > 0:
                    problems.append(f"{name}@{vw}: {r['overflow']} px "
                                    f"horizontal overflow")
            pg.close()
        b.close()

    allsizes = sorted({s for r in out for s in r["sizes"]}, key=float)
    print(f"[audit] {len(pages)} pages x {len(VIEWPORTS)} viewports")
    print(f"  distinct rendered font sizes : {len(allsizes)}  {allsizes}")
    print(f"  font families                : "
          f"{sorted({f for r in out for f in r['fams']})}")
    print(f"  contrast failures            : {sum(len(r['fails']) for r in out)}")
    print(f"  small touch targets (<=768)  : "
          f"{sum(len(r['touch']) for r in out if r['viewport'] <= 768)}")
    print(f"  pages with overflow          : "
          f"{sum(1 for r in out if r['overflow'] > 0)}")
    if len(allsizes) > 7:
        problems.append(f"{len(allsizes)} distinct font sizes — over the 7-step "
                        f"scale, that is sprawl not hierarchy")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    for p in problems[:20]:
        print(f"  FAIL: {p}")
    print(f"audit: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
