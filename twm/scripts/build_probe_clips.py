"""Render the PUBLISHED test set as clips. One source, so they cannot disagree.

The old /probes/ page was rendered from its own sampling run. It therefore
showed 60 clips from five start frames that are not in the test set at all,
and one of them came from 2026-05-19 — the session the test set excludes
because its world frame is unpinned. Two published pages, the same dataset,
opposite answers about the same session.

This reads the exported test set: its poses, its held hand, its context
frames, its calibration. A clip is an animation OF the data you download, not
a second sampling that happens to look similar.

    python scripts/build_probe_clips.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_paths import out_root, testset_root   # noqa: E402

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402

from react_toolbox.calibration import load_calibration          # noqa: E402
from react_toolbox.frames import require_up_axis           # noqa: E402

SRC = testset_root()
FPS = 30.0


class _H264Writer:
    """Frames -> H.264, by pipe, because cv2 here cannot encode it.

    The clips shipped as `mpeg4 / mp4v / Simple Profile`, which browsers refuse:
    Chromium's canPlayType for `mp4v.20.8` returns the empty string. The files
    were present, served with the right Content-Type and the right byte count —
    and unplayable. The dataset's own 248 videos are avc1; only these were not.

    OpenCV in this environment has no H.264 encoder ("Could not find encoder for
    codec_id=27"), so frames go straight to ffmpeg. Piping rather than
    transcoding an mp4v intermediate avoids a second lossy pass.

    `yuv420p` and `+faststart` are not decoration: some players reject 4:4:4,
    and without faststart the moov atom lands at the end so playback cannot
    begin until the whole file is fetched.
    """

    def __init__(self, path, w, h, fps):
        import subprocess
        self.p = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-",
             "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             # BASELINE, level 3.0, no B-frames. libx264's default is High with
             # B-frames, which modern desktops decode fine and older phones and
             # Android WebViews do not. Desktop Chromium reported 640x480 and
             # readyState 4 on the High-profile files, which is one device's
             # evidence, not "it plays".
             "-profile:v", "baseline", "-level", "3.0", "-bf", "0",
             "-preset", "veryfast", "-crf", "23",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)

    def write(self, bgr):
        self.p.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def release(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode()[-400:]
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err}")


def render(frame_rgb, poses, cal, cam, out_mp4, hud, held_pose, held_gel,
           gel_m, collision_m, color):
    """The moving sensor's frame walking its trajectory over a frozen scene.

    Frozen on purpose: the camera image at step k is unknown — that is the
    whole point of a probe — so animating anything but the sensors would be
    inventing pixels.
    """
    from react_toolbox.calibration import project_gel_frame
    from react_toolbox.viz import (draw_collision_circle, draw_sensor_frame,
                                   draw_world_gizmo)

    h, w = frame_rgb.shape[:2]
    base = draw_collision_circle(frame_rgb, held_pose, held_gel, cam,
                                 collision_m, (120, 120, 120))
    base = draw_sensor_frame(base, held_pose, held_gel, cam, stem=True,
                             dim=True, label="held")
    # on the frozen background, so it is present in every frame for free
    base = draw_world_gizmo(base, cam, corner="tl", margin=26, title="world (z-up)")
    vw = _H264Writer(out_mp4, w, h, FPS)
    poster = Path(str(out_mp4).replace(".mp4", ".jpg"))
    trail, n = [], 0
    for i, q in enumerate(np.asarray(poses, float)):
        img = base.copy()
        pr = project_gel_frame(q, gel_m, cam)
        if pr is not None:
            u, v = pr["centre"]
            if 0 <= u < w and 0 <= v < h:
                trail.append((int(round(u)), int(round(v))))
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
        if trail:
            cv2.circle(img, trail[0], 3, (200, 200, 200), 1, cv2.LINE_AA)
        img = draw_collision_circle(img, q, gel_m, cam, collision_m, color)
        img = draw_sensor_frame(img, q, gel_m, cam, stem=True)
        cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(img, f"{hud}  t={i/FPS:4.2f}s", (5, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        if n == 0:
            # A POSTER, so the page can show 72 stills without fetching 72
            # videos. `preload="metadata"` on a grid this size means the phone
            # opens 72 connections before anything is visible; WebKit here
            # could not even finish `load` on the page for that reason.
            # Half size: the tile is ~290 px wide in the grid, so a 640x480
            # poster ships four times the pixels it can show. 72 of them is the
            # page's whole first-paint cost on a phone.
            small = cv2.resize(img[:, :, ::-1], (w // 2, h // 2),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(poster), small, [cv2.IMWRITE_JPEG_QUALITY, 78])
        vw.write(img[:, :, ::-1]); n += 1
    vw.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(out_root("probe_clips")))
    args = ap.parse_args()
    out = Path(args.out)
    # real/ is built by a different script and must survive this rebuild.
    # Without this the page would show the real clips exactly once, until the
    # next time anyone regenerated the synthetic ones -- and it would look
    # like they had simply never been built.
    import tempfile
    stash = None
    if (out / "real").is_dir():
        stash = Path(tempfile.mkdtemp()) / "real"
        shutil.move(str(out / "real"), str(stash))
    if out.exists():
        shutil.rmtree(out)
    (out / "clips").mkdir(parents=True)
    if stash is not None:
        shutil.move(str(stash), str(out / "real"))

    cal = load_calibration(SRC)
    require_up_axis(cal, where=f"{SRC}/calibration")
    man = json.loads((SRC / "manifest.json").read_text())
    runs = [json.loads((SRC / p["meta"]).read_text()) for p in man["probes"]]
    cam = cal["cams"]["middle"]
    recs = []
    for m in runs:
        frame = cv2.imread(str(SRC / f"probes/run{m['run']}/context/ctx3_view_middle.jpg"))[:, :, ::-1]
        gel_m = cal[f"gel_{m['moving_side']}"]
        gel_o = cal[f"gel_{m['held_side']}"]
        col = (255, 210, 63) if m["moving_side"] == "left" else (79, 195, 247)
        for q in m["probes"]:
            d = np.load(SRC / q["file"])
            name = f"run{m['run']}_{q['name']}"
            amp = (f"{q['amplitude']*100:.1f}cm" if q["amplitude_unit"] == "m"
                   else f"{q['amplitude']:.1f}deg")
            hud = (f"{q['name']}  {amp}  {q['horizon_s']:.2f}s  "
                   f"p{q['speed_percentile']:.0f}")
            nf = render(frame, d["poses"], cal, cam, out / "clips" / f"{name}.mp4",
                        hud, d["held_pose"], gel_o, gel_m,
                        m["collision_diameter_m"], col)
            recs.append({**q, "run": m["run"], "episode": m["episode"],
                         "moving_side": m["moving_side"],
                         "clip": f"clips/{name}.mp4", "frames": nf})
            print(f"  {name:22s} {nf:4d}f  {hud}", flush=True)
    (out / "probes.json").write_text(json.dumps(recs, indent=1))
    _page(out, recs, man)
    print(f"\n{len(recs)} clips -> {out}/index.html")
    return 0


def _sessions_note(man, recs):
    """Read the policy from the manifest, not from prose written months ago.

    This page once said "2026-05-19 is excluded" after that call was reversed:
    the draw simply had not selected it, and the sentence promoted an accident
    of sampling to a stated policy.
    """
    used = sorted({r["episode"].split("/")[0] for r in recs})
    elig = sorted(man.get("trusted_sessions", []))
    excl = man.get("excluded_sessions") or {}
    out = [f"Start frames come from <b>held-out intervals</b> of "
           f"<code>splits.json</code>, never training frames. "
           f"Eligible: {', '.join(elig)}; this draw selected {', '.join(used)}"]
    miss = sorted(set(elig) - set(used))
    if miss:
        out.append(f"({', '.join(miss)} was not sampled this time &mdash; chance, "
                   f"not a judgement).")
    else:
        out.append(".")
    if excl:
        out.append("Excluded by policy: " + "; ".join(
            f"<b>{k}</b> &mdash; {v}" for k, v in excl.items()) + ".")
    return " ".join(out)


def _page(out: Path, recs, man) -> None:
    import statistics as st
    order = [f"{k}{s}{a}" for k in ("trans", "rot") for a in "xyz" for s in "+-"]
    order = sorted({r["name"] for r in recs},
                   key=lambda n: (n.startswith("rot"), n[-2:], n))
    amps_t = [r["amplitude"] for r in recs if r["amplitude_unit"] == "m"]
    amps_r = [r["amplitude"] for r in recs if r["amplitude_unit"] == "deg"]
    pct = [r["speed_percentile"] for r in recs]
    seps = [r["min_separation_m"] for r in recs]
    hands = sum(1 for r in recs if r["moving_side"] == "left")
    cards = [
        (f"{len(recs)}", f"clips &middot; {man['n_runs']} start frames"),
        (f"{hands}L / {len(recs)-hands}R", "moving hand"),
        (f"{min(amps_t):.2f}&ndash;{max(amps_t):.2f} m",
         f"translation &middot; {min(amps_r):.0f}&ndash;{max(amps_r):.0f}&deg; rotation"),
        (f"p{min(pct):.0f}&ndash;p{max(pct):.0f}",
         f"speed vs the dataset (median p{st.median(pct):.0f})"),
        (f"{min(seps):.3f} m", "closest the hands come (rule 0.12)"),
    ]
    # REAL counterparts, if they have been built. The synthetic probes show a
    # commanded motion over a frozen frame; nobody performed them. Beside each
    # one, the same signed axis as it actually occurs in the recordings, found
    # by the criteria the probes are BUILT with (translation holds
    # orientation, rotation holds the gel centre) rather than a looser
    # "mostly along x".
    real = {}
    rj = out / "real" / "real_motion.json"
    if rj.exists():
        for r in json.loads(rj.read_text()):
            real[r["name"]] = r

    def _real_fig(name):
        r = real.get(name)
        if not r:
            return ""
        return (f"<figure class='real'><video src='{r['clip']}' controls loop "
                f"muted playsinline preload='none' "
                f"poster='{r['clip'].replace('.mp4', '.jpg')}'></video>"
                f"<figcaption><b>real</b> &middot; {r['date']}/{r['episode']}"
                f"<br>{r['amount']:.0f}"
                f"{'mm' if r['unit'] == 'mm' else '&deg;'} &middot; "
                f"{r['purity_kind']} {r['purity']:.2f} &middot; "
                f"per-step {r['step_dominance']:.2f}<br>"
                f"holds {r['counter']:.1f}"
                f"{'&deg;' if r['counter_unit'] == 'deg' else 'mm'} &middot; "
                f"{r['window']}f &middot; {r['side']}</figcaption></figure>")

    secs = "".join(
        f"<h2>{name}</h2><div class='grid'>" + _real_fig(name) + "".join(
            f"<figure><video src='{r['clip']}' controls loop muted playsinline "
            f"poster='{r['clip'].replace('.mp4', '.jpg')}' preload='none'>"
            f"</video><figcaption>run{r['run']} &middot; "
            f"{r['episode']}<br>{r['amplitude']:g}"
            f"{'m' if r['amplitude_unit']=='m' else '&deg;'} &middot; "
            f"{r['horizon_s']:.2f}s &middot; p{r['speed_percentile']:.0f} &middot; "
            f"{r['moving_side']}</figcaption></figure>"
            for r in recs if r["name"] == name) + "</div>"
        for name in order)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>React probe clips</title><style>
:root{--bg:#0b1020;--fg:#e8eefb;--dim:#8ea0c2;--line:#1e2a45;--card:#111a2e;--accent:#ffc46b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 'IBM Plex Sans',system-ui,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:28px 20px 70px}
h1{font-size:28px;margin:0 0 8px}h2{font-size:16px;color:var(--accent);margin:28px 0 8px}
p{color:var(--dim);max-width:78ch}a{color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:24px;color:var(--accent);font-weight:650}
.card span{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
video{width:100%;border-radius:6px;background:#000}
figcaption{color:var(--dim);font-size:12px;margin-top:6px}
figure.real{border-color:var(--accent)}
figure.real figcaption b{color:var(--accent)}
</style></head><body><div class="wrap">
<h1>React probe clips</h1>
<p>Every clip here is an animation <b>of the published test set</b> &mdash; the same
poses, held hand, context frame and calibration you download from
<a href="https://huggingface.co/datasets/yxma/React/tree/main/test_sets/probes_v1">test_sets/probes_v1</a>.
Rendered from that package, not from a second sampling run, so a clip cannot
drift from the data it illustrates. Static overlays and the method are on the
<a href="../testset/index.html">test set page</a>.</p>
<p><small style="color:var(--dim)">Each tile shows the first frame; tap or click
to load and play. 72 videos preloading at once is what a phone chokes on.</small></p>
<p>The background is <b>frozen on purpose</b>. The camera image at step k is
unknown &mdash; that is the whole point of a probe &mdash; so animating anything
but the sensors would be inventing pixels. What moves is the only thing known:
where the sensor would be. The dimmed hand is the one that must stay still; the
circles are the 0.12&nbsp;m exclusion zones.</p>
<p>__SESSIONS__</p>
<div class="cards">__CARDS__</div>
__SECS__
</div></body></html>"""
    (out / "index.html").write_text(
        html.replace("__SESSIONS__", _sessions_note(man, recs))
            .replace("__CARDS__", "".join(
            f"<div class='card'><b>{a}</b><span>{b}</span></div>" for a, b in cards))
            .replace("__SECS__", secs))


if __name__ == "__main__":
    raise SystemExit(main())
