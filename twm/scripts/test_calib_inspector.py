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

    # THE CONTROL MUST REACH THE HYPOTHESIS IT EXISTS TO TEST. The first
    # version capped the world slider at +/-60 mm while this date's own offset
    # is (230, 0, 175) mm, so the single most relevant setting was off the end
    # of the scale. A range that cannot express the question is indistinguishable
    # from a range that answers it "no".
    woff = np.asarray(d["world_offset_mm"], float)
    reach = float(np.max(np.abs(woff)))
    check(d["world_range_mm"] >= reach,
          "the world control reaches this date's own offset",
          f"slider spans +/-{d['world_range_mm']} mm; the offset baked into "
          f"{d['date']} is {list(woff)} mm (needs {reach:g})")

    # ...and the preset button applies exactly that offset, not something near it.
    async def preset(url):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch(); pg = await b.new_page()
            await pg.goto(url, wait_until="load")
            await pg.wait_for_function("() => typeof window.__probe === 'function'")
            await pg.click("#pAdd")
            got = [await pg.evaluate(f"window.__probe({i}, '{v}', '{s_}')")
                   for i, v, s_ in calls]
            await b.close()
            return got
    got3 = asyncio.run(preset(url))
    bad3, n3 = [], 0
    for (i, v, s_), g in zip(calls, got3):
        p_ = np.asarray(d["frames"][i]["pose"][s_], float).copy()
        p_[:3] += woff / 1000.0
        want = project_gel_frame(p_, cal[f"gel_{s_}"], cal["cams"][v])
        if want is None or g is None:
            continue
        n3 += 1
        e = float(np.hypot(g[0] - want["centre"][0], g[1] - want["centre"][1]))
        if e > TOL_PX:
            bad3.append(f"f{i}/{v}/{s_}: {e:.2f} px")
    moved3 = [float(np.hypot(a[0]-b[0], a[1]-b[1])) for a, b in zip(got, got3) if a and b]
    check(not bad3 and n3 == len(calls),
          "the preset applies exactly that offset",
          f"{n3}/{len(calls)} match the library for a pose shifted by the "
          f"published offset; it moves the marker "
          f"{min(moved3):.0f}-{max(moved3):.0f} px"
          + (f"; worst {bad3[:3]}" if bad3 else ""))

    # a typed value beyond the slider's max must still take effect
    async def typed(url, v):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch(); pg = await b.new_page()
            await pg.goto(url, wait_until="load")
            await pg.wait_for_function("() => typeof window.__probe === 'function'")
            await pg.eval_on_selector("#wxn",
                f"e => {{ e.value = {v}; e.dispatchEvent(new Event('input')); }}")
            out = await pg.evaluate("window.__probe(0, 'middle', 'left')")
            await b.close()
            return out
    big = d["world_range_mm"] + 250.0
    g4 = asyncio.run(typed(url, big))
    p4 = np.asarray(d["frames"][0]["pose"]["left"], float).copy()
    p4[0] += big / 1000.0
    w4 = project_gel_frame(p4, cal["gel_left"], cal["cams"]["middle"])
    e4 = float(np.hypot(g4[0]-w4["centre"][0], g4[1]-w4["centre"][1])) if g4 else float("inf")
    check(e4 < TOL_PX, "a typed value past the slider's end still applies",
          f"{big:g} mm typed into the box lands {e4:.2f} px from the library's "
          f"answer (the slider itself stops at {d['world_range_mm']})")

    # THE ROTATION CONTROL IS A RIGID TRANSFORM, not a nudge of the position.
    # A world-frame correction turns the ORIENTATION too; rotating only the
    # position would move the dot plausibly while leaving the triad pointing
    # the old way, and would look fine on a still frame.
    from scipy.spatial.transform import Rotation

    async def rotated(url, rv):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch(); pg = await b.new_page()
            await pg.goto(url, wait_until="load")
            await pg.wait_for_function("() => typeof window.__probe === 'function'")
            for ax, v in zip("xyz", rv):
                await pg.eval_on_selector(
                    f"#r{ax}n", f"e => {{ e.value = {v}; e.dispatchEvent(new Event('input')); }}")
            out = [await pg.evaluate(f"window.__probe({i}, '{v}', '{s_}')")
                   for i, v, s_ in calls]
            await b.close()
            return out
    RV = np.asarray(d["tilt_fix_deg"], float)
    Rd = Rotation.from_rotvec(np.radians(RV))
    got5 = asyncio.run(rotated(url, RV))
    bad5, n5 = [], 0
    for (i, v, s_), g in zip(calls, got5):
        p_ = np.asarray(d["frames"][i]["pose"][s_], float).copy()
        q = np.concatenate([Rd.apply(p_[:3]),
                            (Rd * Rotation.from_quat(p_[3:7])).as_quat()])
        want = project_gel_frame(q, cal[f"gel_{s_}"], cal["cams"][v])
        if want is None or g is None:
            continue
        n5 += 1
        e = float(np.hypot(g[0]-want["centre"][0], g[1]-want["centre"][1]))
        if e > TOL_PX:
            bad5.append(f"f{i}/{v}/{s_}: {e:.2f} px")
    moved5 = [float(np.hypot(a[0]-b[0], a[1]-b[1])) for a, b in zip(got, got5) if a and b]
    check(not bad5 and n5 == len(calls),
          "the rotation control applies a full rigid transform",
          f"{n5}/{len(calls)} match scipy rotating BOTH position and "
          f"orientation; the measured tilt moves the marker "
          f"{min(moved5):.0f}-{max(moved5):.0f} px"
          + (f"; worst {bad5[:3]}" if bad5 else ""))

    # THE OVERLAY MUST SIT ON THE PHOTO. Every check above calls
    # window.__probe, which works in CANVAS coordinates — so they all passed
    # while the canvas was a different size from the image underneath it and
    # the drawn marker sat somewhere else on screen. Measured: tile <img> was
    # 464x480 while its <canvas> was 464x348, because the width/height HTML
    # attributes act as CSS presentational hints and the stylesheet overrode
    # only `width`. Every overlay y was scaled by 0.725 against the photo, and
    # the photo itself was stretched off 4:3.
    #
    # This is layout, not arithmetic, so it is measured from the rendered
    # boxes rather than from the numbers.
    async def boxes(url):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            pg = await b.new_page(viewport={"width": 1500, "height": 1000})
            await pg.goto(url, wait_until="load")
            await pg.wait_for_function("() => typeof window.__probe === 'function'")
            await pg.wait_for_timeout(400)
            tiles = await pg.evaluate("""(()=>[...document.querySelectorAll('.tile')].map(t=>{
                const i=t.querySelector('img').getBoundingClientRect();
                const c=t.querySelector('canvas').getBoundingClientRect();
                return [i.x,i.y,i.width,i.height,c.x,c.y,c.width,c.height];}))()""")
            await pg.click(".tile"); await pg.wait_for_timeout(600)
            z = await pg.evaluate("""(()=>{const i=document.getElementById('zimg')
                .getBoundingClientRect(), c=document.getElementById('zcv')
                .getBoundingClientRect();
                return [i.x,i.y,i.width,i.height,c.x,c.y,c.width,c.height];})()""")
            await b.close()
            return tiles, z

    tiles, z = asyncio.run(boxes(url))
    def gap(r):
        return max(abs(r[0]-r[4]), abs(r[1]-r[5]), abs(r[2]-r[6]), abs(r[3]-r[7]))
    worst_t = max(gap(r) for r in tiles)
    check(worst_t <= 1.0, "the overlay canvas covers its photo exactly (tiles)",
          f"{len(tiles)} tiles, worst image/canvas box mismatch {worst_t:.1f} px "
          f"(img {tiles[0][2]:.0f}x{tiles[0][3]:.0f}, canvas "
          f"{tiles[0][6]:.0f}x{tiles[0][7]:.0f})")
    check(gap(z) <= 1.0, "...and in the enlarged view",
          f"mismatch {gap(z):.1f} px (img {z[2]:.0f}x{z[3]:.0f}, "
          f"canvas {z[6]:.0f}x{z[7]:.0f})")

    # and the photo is not stretched off its own aspect ratio
    ar = [r[2]/r[3] for r in tiles] + [z[2]/z[3]]
    want = d["frames"][0]["w"] / d["frames"][0]["h"]
    check(max(abs(a-want) for a in ar) < 0.01,
          "the photo keeps its 4:3 aspect ratio",
          f"displayed {min(ar):.3f}-{max(ar):.3f} vs {want:.3f}")

    # READ THE MARKER BACK OUT OF THE RENDERED PIXELS, in both the tile and
    # the enlarged view. The box-geometry check above would have caught the
    # layout defect, but only this one proves the thing a reader actually
    # looks at is in the right place — and the defect it is guarding against
    # (image and canvas scaled differently) left every arithmetic check green
    # while putting the marker ~93 image px from the sensor.
    async def shots(url):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            pg = await b.new_page(viewport={"width": 1500, "height": 1000})
            await pg.goto(url, wait_until="load")
            await pg.wait_for_function("() => typeof window.__probe === 'function'")
            for cid in ("sR", "ax", "st", "gh"):
                await pg.eval_on_selector(
                    f"#{cid}", "e=>{e.checked=false;e.dispatchEvent(new Event('change'))}")
            await pg.wait_for_timeout(400)
            tiles = await pg.query_selector_all(".tile")
            out = [await tiles[k].screenshot() for k in range(3)]
            await tiles[1].click(); await pg.wait_for_timeout(700)
            out.append(await (await pg.query_selector("#zcv")).screenshot())
            # A SECOND shot with the axes back ON. The four above deliberately
            # hide them so the marker dot is the only coloured thing; sampling
            # axis colours from those hit the dark board and reported every
            # axis as blue — a defect in the test, not the page.
            await pg.eval_on_selector(
                "#ax", "e=>{e.checked=true;e.dispatchEvent(new Event('change'))}")
            await pg.wait_for_timeout(400)
            out.append(await (await pg.query_selector("#zcv")).screenshot())
            await b.close()
            return out

    import cv2
    def centroid(png, rgb=(255, 210, 63), tol=40):
        a = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].astype(int)
        m = np.abs(a - np.array(rgb)).max(2) < tol
        if m.sum() < 4:
            return None, a.shape
        ys, xs = np.nonzero(m)
        return np.array([xs.mean(), ys.mean()]), a.shape

    pngs = asyncio.run(shots(url))
    errs, miss = [], []
    for k, v in enumerate(VIEWS + ("middle",)):
        got, shape = centroid(pngs[k])
        if got is None:
            miss.append(v); continue
        want = project_gel_frame(d["frames"][0]["pose"]["left"],
                                 cal["gel_left"], cal["cams"][v])["centre"]
        errs.append(float(np.linalg.norm(got / (shape[1]/640.0) - np.asarray(want))))
    check(not miss and errs and max(errs) < 2.0,
          "the DRAWN marker is on the sensor, in tiles and enlarged",
          f"read back from pixels: {min(errs):.2f}-{max(errs):.2f} px from the "
          f"library's answer (3 tiles + the enlarged view)"
          + (f"; not found in {miss}" if miss else ""))

    # PARITY WITH THE PREVIEW. The page must not merely be self-consistent; it
    # must agree with `react_toolbox.viz.draw_sensor_frame`, which is what the
    # published clips use. Geometry is shared via project_gel_frame, so what is
    # left to diverge is which colour goes on which axis.
    from react_toolbox.viz import AXIS_BGR_RGB
    r0 = project_gel_frame(d["frames"][0]["pose"]["left"], cal["gel_left"],
                           cal["cams"]["middle"])
    c0 = np.asarray(r0["centre"])
    a = cv2.imdecode(np.frombuffer(pngs[4], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    sc = a.shape[1] / 640.0
    order, bad6 = [], []
    for ti, tip in enumerate(r0["tips"]):
        p_ = (c0 + 0.72 * (np.asarray(tip) - c0)) * sc
        x_, y_ = int(round(p_[0])), int(round(p_[1]))
        patch = a[max(0, y_-3):y_+4, max(0, x_-3):x_+4].reshape(-1, 3).astype(int)
        px_ = patch[int(np.argmax(patch.max(1) - patch.min(1)))]
        dom = int(np.argmax(px_))
        want_dom = int(np.argmax(AXIS_BGR_RGB[ti]))
        order.append("xyz"[ti] + "=" + "RGB"[dom])
        if dom != want_dom:
            bad6.append(f"axis {'xyz'[ti]}: page {'RGB'[dom]}, preview {'RGB'[want_dom]}")
    check(not bad6, "axis colours match the published preview overlay",
          f"{' '.join(order)} — same mapping as viz.AXIS_BGR_RGB"
          + (f"; {bad6}" if bad6 else ""))

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    nf = sum(not ok for ok, _, _ in RESULTS)
    print(f"\ncalib inspector: {len(RESULTS)} checks, {nf} failing")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
