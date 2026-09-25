"""The interactive sensor simulator: does the overlay obey the action?

The page moves a sensor by keyboard and redraws its projected frame over a
frozen photo. Nothing about that is self-evidently right, and three of the
ways it can be wrong were live bugs in this project within the last day:

  - a rotation that pivots on the marker cluster instead of the gel swings
    the gel 30-50 mm across the table while looking like a clean spin;
  - a canvas whose CSS size does not match its photo puts every marker tens
    of pixels off, while every number inside the page stays correct;
  - poses in one up-axis convention with extrinsics in the other project
    165 px away and raise nothing.

So the checks below compare the PAGE against the Python library that built
it, and against physics -- not against the page's own arithmetic.

    python scripts/test_sim.py [--root <dir>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402

from react_paths import out_root                               # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
TOL_PX = 0.5


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


async def drive(url: str, script: str):
    """Open the page, run `script` in it, return (result, console errors)."""
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page(viewport={"width": 1500, "height": 1000})
        errs = []
        pg.on("console",
              lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(url, wait_until="load")
        await pg.wait_for_function("() => typeof window.__sim === 'function'")
        out = await pg.evaluate(script)
        await b.close()
        return out, errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(out_root("sim")))
    a = ap.parse_args()
    root = Path(a.root)
    if not (root / "index.html").exists():
        check(False, "the simulator page exists",
              f"{root}/index.html is missing — run scripts/build_sim.py")
        return _report()

    from react_toolbox.calibration import load_calibration, project_gel_frame
    from react_toolbox.frames import require_up_axis
    from scipy.spatial.transform import Rotation

    cal = load_calibration(root)
    require_up_axis(cal, where=f"{root}/calibration")
    D = json.loads((root / "sim.json").read_text())
    url = (root / "index.html").resolve().as_uri()

    # ---- 1. it loads clean, and shows what it claims to ------------------
    got, errs = asyncio.run(drive(url, """(() => {
        const t = [...document.querySelectorAll('[data-stream]')]
                    .map(e => e.dataset.stream);
        return {streams: t,
                tactileNote: !!document.querySelector('[data-tactile-note]'),
                selected: window.__sim().side};
    })()"""))
    check(not errs and len(got["streams"]) == 5,
          "the page shows three views and both tactile streams",
          f"{got['streams']}; {len(errs)} console errors")

    # ---- 2. at rest, the page projects where the library projects --------
    calls = [(v, s) for v in ("left", "middle", "right")
             for s in ("left", "right")]
    got, _ = asyncio.run(drive(url, "(() => window.__sim().points)()"))
    worst, n = 0.0, 0
    for v, s in calls:
        r = project_gel_frame(np.asarray(D["pose"][s], float),
                              cal[f"gel_{s}"], cal["cams"][v])
        p = got[v][s]["centre"]
        if r is None or p is None:
            continue
        n += 1
        worst = max(worst, float(np.hypot(p[0] - r["centre"][0],
                                          p[1] - r["centre"][1])))
    check(n == 6 and worst < TOL_PX,
          "at rest the page projects where the library projects",
          f"{n}/6 sensor-view pairs, worst {worst:.3f} px")

    # ---- 3. a translation key moves the gel by what it says --------------
    step = D["step_mm"]
    got, _ = asyncio.run(drive(url, f"""(() => {{
        const a = window.__sim(); window.__key('ArrowRight');
        const b = window.__sim();
        return {{a, b}};
    }})()"""))
    side = got["a"]["side"]
    d = np.asarray(got["b"]["gel_mm"], float) - np.asarray(got["a"]["gel_mm"], float)
    check(abs(np.linalg.norm(d) - step) < 1e-6
          and abs(d[0] - step) < 1e-6,
          "one translation key steps the gel by exactly one step, on one axis",
          f"+x key moved the gel {np.round(d, 4).tolist()} mm "
          f"(step {step} mm); off-axis {max(abs(d[1]), abs(d[2])):.2e}")

    # ---- 4. THE ONE THAT MATTERS: rotation pivots on the GEL -------------
    got, _ = asyncio.run(drive(url, """(() => {
        const a = window.__sim();
        for (let i = 0; i < 6; i++) window.__key('KeyE');   // yaw +
        const b = window.__sim();
        return {a, b};
    })()"""))
    gel_move = float(np.linalg.norm(
        np.asarray(got["b"]["gel_mm"], float) - np.asarray(got["a"]["gel_mm"], float)))
    org_move = float(np.linalg.norm(
        np.asarray(got["b"]["origin_mm"], float) - np.asarray(got["a"]["origin_mm"], float)))
    ang = float(np.degrees((Rotation.from_quat(got["b"]["quat"]).inv()
                            * Rotation.from_quat(got["a"]["quat"])).magnitude()))
    # ANCHOR THE PREDICTION IN THE LIBRARY, not in the page's own report.
    # The first version of this check took v = origin - gel from the page's
    # OWN output, so when the pivot was moved to the marker cluster the page
    # relabelled both points, v kept its magnitude, and the check passed on
    # the exact bug it was written for. Self-referential checks are how the
    # 153 px overlay error stayed green all day.
    p0 = np.asarray(D["pose"][side], float)
    gel_lib = p0[:3] * 1000.0 + Rotation.from_quat(p0[3:7]).as_matrix() @ \
        np.asarray(cal[f"gel_{side}"], float)
    at_rest = float(np.linalg.norm(np.asarray(got["a"]["gel_mm"], float) - gel_lib))
    after = float(np.linalg.norm(np.asarray(got["b"]["gel_mm"], float) - gel_lib))
    v = p0[:3] * 1000.0 - gel_lib               # gel -> marker cluster, world
    Rz = Rotation.from_rotvec(np.deg2rad(6 * D["step_deg"]) * np.array([0, 0, 1.]))
    want = float(np.linalg.norm(Rz.apply(v) - v))
    check(at_rest < 1e-6 and after < 1e-6 and want > 1.0
          and abs(org_move - want) < 0.05
          and abs(ang - 6 * D["step_deg"]) < 1e-6,
          "rotation pivots on the gel, not on the marker cluster",
          f"{ang:.1f} deg about world z. Distance from the library's gel "
          f"centre: {at_rest:.2e} mm before, {after:.2e} mm after — near zero "
          f"means it is THE GEL that held still; a large value means the page "
          f"pivoted somewhere else and relabelled the points. The cluster "
          f"moved {org_move:.2f} mm "
          f"against {want:.2f} mm predicted from the published pose "
          f"(perpendicular radius "
          f"{want / (2 * np.sin(np.deg2rad(3 * D['step_deg']))):.1f} mm)")

    # ---- 5. only the selected sensor moves --------------------------------
    got, _ = asyncio.run(drive(url, """(() => {
        const a = window.__sim();
        for (let i = 0; i < 4; i++) window.__key('ArrowUp');
        window.__key('KeyE');
        const b = window.__sim();
        return {side: a.side, a: a.other_gel_mm, b: b.other_gel_mm};
    })()"""))
    om = float(np.linalg.norm(np.asarray(got["b"], float)
                              - np.asarray(got["a"], float)))
    check(om < 1e-9, "the sensor that is not selected does not move",
          f"held sensor moved {om:.2e} mm while the selected one was driven")

    # ---- 6. switching sensors drives the other one -----------------------
    got, _ = asyncio.run(drive(url, """(() => {
        const a = window.__sim(); window.__key('Tab');
        const b = window.__sim(); window.__key('ArrowRight');
        const c = window.__sim();
        return {a: a.side, b: b.side,
                moved: [c.gel_mm[0] - b.gel_mm[0], c.gel_mm[1] - b.gel_mm[1],
                        c.gel_mm[2] - b.gel_mm[2]]};
    })()"""))
    check(got["a"] != got["b"] and abs(got["moved"][0] - step) < 1e-6,
          "switching the selection drives the other sensor",
          f"{got['a']} -> {got['b']}, then +x moved it "
          f"{np.round(got['moved'], 3).tolist()} mm")

    # ---- 7. reset restores the start pose exactly ------------------------
    got, _ = asyncio.run(drive(url, """(() => {
        const a = window.__sim();
        for (let i = 0; i < 5; i++) { window.__key('ArrowRight'); window.__key('KeyE'); }
        window.__key('KeyR');
        return {a, b: window.__sim()};
    })()"""))
    dp = float(np.abs(np.asarray(got["b"]["gel_mm"], float)
                      - np.asarray(got["a"]["gel_mm"], float)).max())
    dq = float(np.abs(np.asarray(got["b"]["quat"], float)
                      - np.asarray(got["a"]["quat"], float)).max())
    check(dp < 1e-9 and dq < 1e-12, "reset returns to the start pose exactly",
          f"worst position {dp:.2e} mm, worst quaternion {dq:.2e}")

    # ---- 8. the canvas covers its photo exactly --------------------------
    got, errs = asyncio.run(drive(url, """(() => {
        return [...document.querySelectorAll('[data-stream]')].map(t => {
            const i = t.querySelector('img'), c = t.querySelector('canvas');
            if (!i || !c) return null;
            const a = i.getBoundingClientRect(), b = c.getBoundingClientRect();
            return {s: t.dataset.stream,
                    d: Math.max(Math.abs(a.x-b.x), Math.abs(a.y-b.y),
                                Math.abs(a.width-b.width), Math.abs(a.height-b.height))};
        }).filter(Boolean);
    })()"""))
    worst = max((t["d"] for t in got), default=999)
    check(got and worst < 0.5,
          "every overlay canvas covers its photo exactly",
          f"{len(got)} tiles, worst box mismatch {worst:.2f} px — a canvas "
          f"that does not match its photo moves every marker while the "
          f"numbers inside the page stay right")

    # ---- 9. the tactile tiles say they are NOT simulated ------------------
    got, _ = asyncio.run(drive(url, """(() => {
        const n = document.querySelector('[data-tactile-note]');
        const t = [...document.querySelectorAll('[data-stream]')]
                    .filter(e => e.dataset.stream.startsWith('tactile'));
        return {note: n ? n.textContent.trim() : "",
                tiles: t.length,
                marked: t.filter(e => e.querySelector('[data-frozen]')).length};
    })()"""))
    check(got["tiles"] == 2 and got["marked"] == 2 and len(got["note"]) > 20,
          "the tactile tiles are marked as frozen, not predicted",
          f"{got['marked']}/{got['tiles']} tiles marked; note: "
          f"{got['note'][:70]!r}")

    # ---- 10. the start frame is a legitimate evaluation start -------------
    from react_paths import release_root
    sp = json.loads((release_root("motherboard") / "splits.json").read_text())
    key, row = D["key"], D["row"]
    iv = (sp["episodes"].get(key) or {}).get("test") or []
    check(bool(iv) and any(lo <= row <= hi for lo, hi in iv),
          "the start frame lies in a held-out interval",
          f"{key} row {row}; {len(iv)} test intervals on that episode "
          f"-- a simulator seeded from training frames would be showing the "
          f"model its own homework")

    # ---- 11. a REAL keypress, not the test hook --------------------------
    # Every check above drives through window.__key, which calls apply()
    # directly. If the keydown listener were never attached -- or attached to
    # the wrong target, or swallowed by a focus problem -- all of them would
    # still pass and the page would be dead to the keyboard.
    async def real_keys():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            pg = await b.new_page(viewport={"width": 1500, "height": 1000})
            await pg.goto(url, wait_until="load")
            await pg.wait_for_function("() => typeof window.__sim === 'function'")
            before = await pg.evaluate("window.__sim()")
            await pg.keyboard.press("ArrowRight")
            await pg.keyboard.press("ArrowRight")
            mid = await pg.evaluate("window.__sim()")
            await pg.keyboard.press("Tab")
            after = await pg.evaluate("window.__sim()")
            await b.close()
            return before, mid, after
    b0, m0, a0 = asyncio.run(real_keys())
    moved = float(np.asarray(m0["gel_mm"], float)[0]
                  - np.asarray(b0["gel_mm"], float)[0])
    check(abs(moved - 2 * step) < 1e-6 and a0["side"] != m0["side"],
          "the page responds to a real keypress, not just the test hook",
          f"two -> keys moved the gel {moved:.1f} mm (expected {2 * step:.1f}), "
          f"Tab switched {m0['side']} -> {a0['side']}")

    # ---- 12. it works opened straight off disk ---------------------------
    # The data is inlined precisely so this works; a page that fetch()es its
    # own JSON is blank under file:// and only ever runs on the Space.
    got, errs = asyncio.run(drive(url, "(() => window.__sim().side)()"))
    check(bool(got) and not errs,
          "the page runs from file:// with no server",
          f"opened {url[:44]}... and __sim() answered {got!r}; "
          f"{len(errs)} console errors")

    # ---- 13. the document has a real skeleton ----------------------------
    # Hugging Face injects a <script> into every static Space. With no
    # <html>/<head>/<body>, the parser put that injection in the body and the
    # deployed page showed `window.huggingface={variables:...}` as a line of
    # text above the title. Local rendering is perfect; only the live URL
    # shows it. So assert the structure that makes the injection land in the
    # head, and assert no page text looks like injected script.
    got, _ = asyncio.run(drive(url, """(() => ({
        head: !!document.head && document.head.children.length > 2,
        body: !!document.body,
        stray: (document.body.innerText || "").slice(0, 400)
    }))()"""))
    leaks = [t for t in ("window.huggingface", "SPACE_CREATOR", "={variables")
             if t in got["stray"]]
    html = (root / "index.html").read_text()
    skeleton = "<html" in html[:120] and "<head>" in html[:300] and "<body" in html
    check(got["head"] and got["body"] and not leaks and skeleton,
          "the document has html/head/body, so injected script cannot render",
          f"head/body present: {got['head']}/{got['body']}; explicit skeleton "
          f"in the file: {skeleton}; script-looking text in the body: "
          f"{leaks or 'none'}")

    # ---- 14. the inlined copy of the data is the file beside it ----------
    html = (root / "index.html").read_text()
    blob = (root / "sim.json").read_text()
    check(blob.strip() in html,
          "the page's inlined data is byte-identical to sim.json",
          f"sim.json is {len(blob)} bytes and appears verbatim in the page — "
          f"two copies that can drift is how a page starts describing a frame "
          f"it is no longer showing"
          if blob.strip() in html else "the two copies DIFFER")

    return _report()


def _report() -> int:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\nsim: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
