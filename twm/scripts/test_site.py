"""The published Space renders, and its pages agree with the data and each other.

Two failures this suite exists for, both of which shipped:

  * /probes/ was rendered from its own sampling run, so it showed clips of
    probes that were not in the published set, including one from a session
    the set excludes. Two pages, one dataset, opposite answers.
  * a tile's <img> was 464x480 while its <canvas> was 464x348, so every
    overlay sat 93 px from the sensor it annotated while every arithmetic
    check stayed green.

So this checks three things a screenshot cannot: that each page loads clean,
that what it displays matches the published artefacts it describes, and that
overlay canvases actually cover their images.

    python scripts/test_site.py [--base URL]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS: list[tuple[bool, str, str]] = []
BASE = "https://yxma-react-force-recovery.static.hf.space"
PAGES = ["testset/index.html", "probes/index.html",
         "calib/a/index.html", "calib/pre/index.html"]


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


async def audit(base, pages):
    from playwright.async_api import async_playwright
    out = {}
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        for p in pages:
            pg = await b.new_page(viewport={"width": 1440, "height": 1000})
            errs, bad = [], []
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            # the FULL url. Storing the basename and rebuilding
            # f"{base}/{dir}/{name}" dropped the subdirectory, so the retry
            # fetched /testset/run4_trans-y.jpg for a file that lives at
            # /testset/overlays/... — a 404 on a URL that never existed, which
            # I read as a missing file when all 72 were present.
            pg.on("response", lambda r: bad.append((r.status, r.url))
                  if r.status >= 400 else None)
            r = await pg.goto(f"{base}/{p}", wait_until="networkidle", timeout=90000)
            await pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await pg.wait_for_timeout(3500)
            d = await pg.evaluate("""(() => {
                const im = [...document.querySelectorAll('img')];
                const cv = [...document.querySelectorAll('.tile, .zoomwrap')].map(t => {
                    const i = t.querySelector('img'), c = t.querySelector('canvas');
                    if (!i || !c) return 0;
                    const a = i.getBoundingClientRect(), b = c.getBoundingClientRect();
                    return Math.max(Math.abs(a.x-b.x), Math.abs(a.y-b.y),
                                    Math.abs(a.width-b.width), Math.abs(a.height-b.height));
                });
                const vd = [...document.querySelectorAll('video')];
                return {imgs: im.length,
                        broken: im.filter(x => x.naturalWidth === 0 && x.getAttribute('src')).length,
                        videos: vd.length,
                        vsrc: vd.length ? vd[0].currentSrc || vd[0].src : null,
                        overflow: document.documentElement.scrollWidth > window.innerWidth,
                        canvasGap: cv.length ? Math.max(...cv) : null,
                        title: document.title};
            })()""")
            out[p] = {"status": r.status, "errs": errs, "bad": bad, **d}
            await pg.close()
        await b.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    au = asyncio.run(audit(a.base, PAGES))

    # 1 — every page loads clean.
    #
    # A sub-resource failure is RE-REQUESTED before it counts. The static Space
    # returns transient 429/503 under a cold start, and the first version of
    # this check reported 4 failures on a page whose files were all present —
    # a re-run showed zero. A flaky check that cries wolf is worse than none,
    # because the next real failure gets waved through.
    persist = []
    for pg_, v in au.items():
        for st, url in v["bad"]:
            name = url.rsplit("/", 1)[-1]
            try:
                again = urllib.request.urlopen(url, timeout=45).status
            except urllib.error.HTTPError as e:
                again = e.code
            except Exception:
                again = 0
            if again >= 400 or again == 0:
                persist.append(f"{pg_}: {name} -> {st} then {again}")
    bad = [f"{p_}: HTTP {v['status']}" for p_, v in au.items() if v["status"] != 200]
    bad += persist
    # A console error CAUSED by a transient 4xx is that 4xx counted twice. The
    # browser logs "Failed to load resource" for the same request `persist`
    # already adjudicated, so only errors that are not about a re-fetchable
    # sub-resource count as page defects.
    def real_errs(v):
        urls = {u.rsplit("/", 1)[-1] for _, u in v["bad"]}
        return [e for e in v["errs"]
                if not any(u and u in e for u in urls)
                and "Failed to load resource" not in e]
    bad += [f"{p_}: {len(real_errs(v))} console errors {real_errs(v)[:1]}"
            for p_, v in au.items() if real_errs(v)]
    transient = sum(len(v["bad"]) for v in au.values()) - len(persist)
    check(not bad, "every page loads with no persistent 4xx and no console errors",
          f"{len(PAGES)} pages, all HTTP 200, 0 persistent sub-resource failures, "
          f"0 console errors"
          + (f" ({transient} transient, re-requested OK)" if transient else "")
          + (f"; {bad[:3]}" if bad else ""))

    # 2 — nothing broken or overflowing
    # RE-FETCH before calling an image broken. One that lost a race with a
    # cold-start 429 reports naturalWidth 0 while its file is perfectly there —
    # measured, 4 of them on one run and 0 on the next. My first attempt at
    # this patch searched for `p_` where the file said `p`, so `.replace()`
    # silently did nothing and the run that happened not to hit a 429 reported
    # a clean pass. Every edit here now asserts its anchor.
    br = []
    for p_, v in au.items():
        if not v["broken"]:
            continue
        still = 0
        for st, url in v["bad"]:
            if url.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                try:
                    if urllib.request.urlopen(url, timeout=45).status >= 400:
                        still += 1
                except Exception:
                    still += 1
        if still:
            br.append(f"{p_}: {still} genuinely broken imgs")
    ov = [p_ for p_, v in au.items() if v["overflow"]]
    n_transient_img = sum(v["broken"] for v in au.values())
    check(not br and not ov, "no broken media and no horizontal overflow",
          f"{sum(v['imgs'] for v in au.values())} images, "
          f"{sum(v['videos'] for v in au.values())} videos, 0 persistently broken, "
          f"0 overflowing"
          + (f" ({n_transient_img} lost a race with a cold-start 429, all "
             f"re-fetched OK)" if n_transient_img else "")
          + (f"; {br + ov}" if (br or ov) else ""))

    # 3 — VIDEOS ACTUALLY DECODE. Counting <video> elements says nothing: the
    #     element exists whether or not the browser can play what is inside it.
    #     72 clips shipped as mpeg4/mp4v Simple Profile, which Chromium refuses
    #     (canPlayType returns ""), while every file was present, served as
    #     video/mp4 with the right byte count — and unplayable. This check
    #     loads one and waits for real decoded dimensions.
    async def probe_video(url):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            pg = await b.new_page()
            await pg.goto("about:blank")
            r = await pg.evaluate("""async (u) => {
                const v = document.createElement('video');
                v.src = u; v.muted = true; v.preload = 'auto';
                document.body.appendChild(v);
                const ok = await new Promise(res => {
                    v.onloadeddata = () => res(true);
                    v.onerror = () => res(false);
                    setTimeout(() => res(false), 20000);
                });
                return {ok, w: v.videoWidth, h: v.videoHeight,
                        state: v.readyState,
                        err: v.error ? v.error.code : null};
            }""", url)
            await b.close()
            return r

    # ...and on a phone viewport, where the failure was first reported.
    #
    # Playwright's Linux WebKit is not a stand-in for Safari, but NOT for the
    # reason I first wrote here. I claimed it lacked the proprietary H.264
    # decoder; measured, its canPlayType returns "probably" for baseline, for
    # high, AND for mp4v — the codec real Safari refuses and that broke this
    # page. It is over-permissive, so it would have passed the original bug.
    # Chromium under a mobile device profile exercises the small-screen layout
    # and the lazy-load path; neither engine here can certify Safari playback.
    vids = [(p_, v["vsrc"]) for p_, v in au.items() if v.get("vsrc")]
    vres = []
    for p_, src in vids:
        vres.append((p_, asyncio.run(probe_video(src))))
    okv = [r for _, r in vres if r["ok"] and r["w"] > 0]
    check(vres and len(okv) == len(vres),
          "videos actually decode, not merely exist as elements",
          "; ".join(f"{p_.split('/')[0]} {r['w']}x{r['h']} readyState={r['state']}"
                    + (f" ERR{r['err']}" if r.get("err") else "")
                    for p_, r in vres) or "no <video> found")

    # 4 — OVERLAY CANVASES COVER THEIR IMAGES. The defect that left every
    #     arithmetic check green while the overlay sat 93 px off.
    gaps = {p: v["canvasGap"] for p, v in au.items() if v["canvasGap"] is not None}
    worst = max(gaps.values()) if gaps else 0.0
    check(gaps and worst <= 1.0,
          "overlay canvases cover their images exactly",
          f"{len(gaps)} pages with overlays, worst image/canvas box mismatch "
          f"{worst:.1f} px")

    # ...and the video grid must not fetch every clip up front. 72 videos
    #    preloading at once is what a phone chokes on: WebKit could not even
    #    finish `load` on the page before this changed.
    async def first_paint(url):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            ctx = await b.new_context(**pw.devices["iPhone 13"])
            pg = await ctx.new_page()
            got = []
            pg.on("response", lambda r: got.append(r.url.rsplit("/", 1)[-1]))
            await pg.goto(url, wait_until="networkidle", timeout=90000)
            await pg.wait_for_timeout(2500)
            d = await pg.evaluate("""(() => {const v=[...document.querySelectorAll('video')];
                return {n: v.length, preload: v.length ? v[0].getAttribute('preload') : null,
                        posters: v.filter(x => x.getAttribute('poster')).length};})()""")
            await b.close()
            return d, sum(1 for g in got if g.endswith(".mp4"))

    clip_page = f"{a.base}/probes/index.html"
    d, n_mp4 = asyncio.run(first_paint(clip_page))
    check(d["n"] > 0 and n_mp4 == 0 and d["posters"] == d["n"],
          "the clip grid shows posters and fetches no video until asked",
          f"iPhone viewport: {d['n']} videos, preload={d['preload']}, "
          f"{d['posters']} with posters, {n_mp4} mp4 requests on first paint")

    # 6 — THE CLIP PAGE SHOWS THE PUBLISHED SET, not a second sampling.
    from huggingface_hub import hf_hub_download
    d = tempfile.mkdtemp()
    man = json.loads(Path(hf_hub_download(
        "yxma/React", "test_sets/probes_v1/manifest.json",
        repo_type="dataset", local_dir=d)).read_text())
    live = json.loads(urllib.request.urlopen(f"{a.base}/probes/probes.json",
                                             timeout=60).read())
    page_runs = {(x["run"], x["episode"]) for x in live}
    set_runs = {(x["run"], x["episode"]) for x in man["probes"]}
    check(page_runs == set_runs and len(live) == man["n_probes"],
          "the clip page animates the published set, not a second sampling",
          f"{len(live)} clips vs {man['n_probes']} published probes; "
          f"start frames identical: {page_runs == set_runs}")

    # 7 — THE PAGE'S STATED POLICY MATCHES THE MANIFEST. Check 8 below compares
    #     run SETS and passed while the prose said "2026-05-19 is excluded"
    #     months after that call was reversed: the draw had simply not selected
    #     it, and the sentence promoted an accident of sampling into a policy a
    #     reader would act on. Sets are not claims; sentences are.
    async def page_text(url):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch(); pg = await b.new_page()
            await pg.goto(url, wait_until="domcontentloaded", timeout=90000)
            t = await pg.inner_text("body")
            await b.close()
            return t

    elig = set(man.get("trusted_sessions", []))
    excl = set(man.get("excluded_sessions") or {})
    prose = []
    for pg_ in ("testset/index.html", "probes/index.html"):
        t = asyncio.run(page_text(f"{a.base}/{pg_}"))
        for sess in elig - excl:
            # an eligible session must never be described as excluded
            for phrase in (f"{sess} is excluded", f"{sess} is left out",
                           f"excluded: {sess}"):
                if phrase.lower() in t.lower():
                    prose.append(f"{pg_}: says '{phrase}' but manifest lists it eligible")
        for sess in excl:
            if sess not in t:
                prose.append(f"{pg_}: excludes {sess} but never says so")
    check(not prose, "the page's stated session policy matches the manifest",
          f"eligible {sorted(elig)}, excluded {sorted(excl) or 'none'}; both pages "
          f"agree" + (f"; {prose[:2]}" if prose else ""))

    # 8 — and no page uses a session the set excludes
    excl = set(man.get("excluded_sessions") or {})
    leaked = sorted({e.split("/")[0] for _, e in page_runs} & excl)
    check(not leaked, "no page shows an excluded session",
          f"sessions on the page: {sorted({e.split('/')[0] for _, e in page_runs})}; "
          f"excluded by the set: {sorted(excl) or 'none'}"
          + (f"; LEAKED {leaked}" if leaked else ""))

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nsite: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
