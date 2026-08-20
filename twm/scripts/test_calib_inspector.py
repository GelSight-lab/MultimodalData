"""The inspector's in-browser projection agrees with the library's.

The page recomputes the projection in JavaScript so the sliders can move the
overlay live. That makes it a SECOND implementation of a calculation this
project has already got wrong four separate ways (wrong calibration epoch,
gel centre defaulted to the origin, world offset missing, world offset applied
backwards). A second implementation that silently disagrees is worse than no
interactivity at all, because it would be a control panel for the wrong
geometry — and it would look completely plausible while you tuned an offset
on it.

So the test drives the real page in a browser, calls the projection it uses to
draw, and compares against `react_toolbox.calibration.project_gel_frame`.

It also checks the sliders MEAN what they say: a world offset of (d,0,0) must
move the projected point by the same amount the library reports when the same
offset is added to the pose. A slider whose label does not match its effect is
how you talk yourself into a wrong recalibration.

    python scripts/test_calib_inspector.py [--batch a]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
VIEWS = ("left", "middle", "right")
TOL_PX = 0.5


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


async def probe(url: str, calls, world=(0, 0, 0)):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page()
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        await pg.goto(url, wait_until="load")
        await pg.wait_for_function("() => typeof window.__probe === 'function'")
        for axis, v in zip("xyz", world):
            if v:
                await pg.eval_on_selector(f"#w{axis}",
                                          f"e => {{ e.value = {v}; e.dispatchEvent(new Event('input')); }}")
        out = []
        for i, view, side in calls:
            out.append(await pg.evaluate(f"window.__probe({i}, '{view}', '{side}')"))
        await b.close()
        return out, errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="a")
    ap.add_argument("--root", default="/media/yxma/Disk1/twm/calib_inspector")
    args = ap.parse_args()

    from react_toolbox.calibration import load_calibration, project_gel_frame
    from twm.calib_epoch import calib_dir

    root = Path(args.root) / args.batch
    d = json.loads((root / "data.json").read_text())
    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = load_calibration(stage)

    calls = [(i, v, s) for i in range(len(d["frames"])) for v in VIEWS
             for s in ("left", "right")]
    url = (root / "index.html").resolve().as_uri()

    got, errs = asyncio.run(probe(url, calls))
    bad, n_ok = [], 0
    for (i, v, s), g in zip(calls, got):
        want = project_gel_frame(d["frames"][i]["pose"][s], cal[f"gel_{s}"],
                                 cal["cams"][v])
        if want is None or g is None:
            if (want is None) != (g is None):
                bad.append(f"f{i}/{v}/{s}: one side returned nothing")
            continue
        e = float(np.hypot(g[0] - want["centre"][0], g[1] - want["centre"][1]))
        n_ok += 1
        if e > TOL_PX:
            bad.append(f"f{i}/{v}/{s}: {e:.2f} px")
    check(not bad and n_ok == len(calls),
          "the page projects where the library projects",
          f"{n_ok}/{len(calls)} tile-sensors within {TOL_PX} px"
          + (f"; worst {bad[:3]}" if bad else ""))
    check(not errs, "no console errors", f"{len(errs)} errors"
          + (f": {errs[:2]}" if errs else ""))

    # THE SLIDER MEANS WHAT IT SAYS. +25 mm on world x must reproduce the
    # library's answer for a pose whose x is 25 mm larger — not merely "move
    # the dot to the right", which any sign error also does.
    D = 25.0
    got2, _ = asyncio.run(probe(url, calls, world=(D, 0, 0)))
    bad2, n2 = [], 0
    for (i, v, s), g in zip(calls, got2):
        p = np.asarray(d["frames"][i]["pose"][s], float).copy()
        p[0] += D / 1000.0
        want = project_gel_frame(p, cal[f"gel_{s}"], cal["cams"][v])
        if want is None or g is None:
            continue
        n2 += 1
        e = float(np.hypot(g[0] - want["centre"][0], g[1] - want["centre"][1]))
        if e > TOL_PX:
            bad2.append(f"f{i}/{v}/{s}: {e:.2f} px")
    check(not bad2 and n2 == len(calls),
          f"the world-offset slider adds exactly what it claims (+{D:g} mm x)",
          f"{n2}/{len(calls)} match the library's answer for a shifted pose"
          + (f"; worst {bad2[:3]}" if bad2 else ""))

    # ...and the shift is big enough to see, or the control is decorative.
    moved = [float(np.hypot(a[0] - b[0], a[1] - b[1]))
             for a, b in zip(got, got2) if a and b]
    check(min(moved) > 3.0,
          "and that offset is visible on every tile",
          f"{D:g} mm moves the marker {min(moved):.1f}-{max(moved):.1f} px "
          f"across the {len(moved)} tile-sensors")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    nf = sum(not ok for ok, _, _ in RESULTS)
    print(f"\ncalib inspector: {len(RESULTS)} checks, {nf} failing")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
