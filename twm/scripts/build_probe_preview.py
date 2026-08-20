"""Render the synthetic probe sets as videos, and a page to review them.

There is no ground-truth future image for a synthetic probe, so the thing to
review is the GT sensor-pose MOTION: the marker sweeping along the projected
trajectory over the frozen start frame, with the path it has covered trailing
behind it. That is exactly what a model rollout must be compared against, so
it is what gets rendered.

Five independent start frames, twelve directions each. The start frames are
sampled without reference to the action sets (`sample_probe` accepts or
rejects a frame against the probes, never adjusts a probe to fit a frame), and
every probe is required to keep its marker inside the view — the default —
because a marker that leaves the image asks the model about a configuration
the training data never contains.

    python scripts/build_probe_preview.py [--runs 5] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2                                                      # noqa: E402
import h5py                                                     # noqa: E402
import hdf5plugin                                               # noqa: E402,F401
import numpy as np                                              # noqa: E402
import pyarrow.parquet as pq                                    # noqa: E402

import react_toolbox as T                                       # noqa: E402
from twm.calib_epoch import calib_dir                           # noqa: E402

RELEASE = Path("/media/yxma/Disk1/twm/release_force/motherboard/meta")
H5ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")
CAM_H5 = {"left": 1, "middle": 2, "right": 0}
FPS = 30.0


def _episodes():
    out = []
    for d in sorted(RELEASE.iterdir()):
        for p in sorted(d.glob("*.parquet")):
            if (H5ROOT / d.name / f"{p.stem}.h5").exists():
                out.append((d.name, p.stem))
    return out


def _render_probe(frame_rgb, probe, cal, cam, out_mp4, hud, held_pose,
                  collision_m):
    """One mp4: the moving sensor's FRAME walking its trajectory.

    Frozen background on purpose. The camera image at step k is unknown —
    that is the whole point of the probe — so animating anything but the
    sensors would be inventing pixels. What moves is the only thing known.

    Both hands are drawn: the moving one bright, the held one dimmed, each
    with its exclusion circle, so the clearance the sampler enforced
    numerically is also visible. A viewer can see that the rule held rather
    than take the metadata's word for it.
    """
    from react_toolbox.viz import draw_collision_circle, draw_sensor_frame

    side, other = probe["moving_side"], probe["held_side"]
    gel_m, gel_o = cal[f"gel_{side}"], cal[f"gel_{other}"]
    h, w = frame_rgb.shape[:2]
    tmp = Path(tempfile.mkdtemp())
    col = (0, 220, 255) if probe["kind"] == "translation" else (255, 100, 220)

    # the held hand and its circle never change: draw once, reuse
    base = draw_collision_circle(frame_rgb, held_pose, gel_o, cam,
                                 collision_m, (120, 120, 120))
    base = draw_sensor_frame(base, held_pose, gel_o, cam,
                             label=other[0].upper(), dim=True)

    from react_toolbox.calibration import project_gel_frame
    trail, n = [], 0
    for i, q in enumerate(np.asarray(probe["poses"], float)):
        img = base.copy()
        pr = project_gel_frame(q, gel_m, cam)
        if pr is not None:
            u, v = pr["centre"]
            if 0 <= u < w and 0 <= v < h:
                trail.append((int(round(u)), int(round(v))))
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, a, b, col, 1, cv2.LINE_AA)
        if trail:
            cv2.circle(img, trail[0], 3, (200, 200, 200), 1, cv2.LINE_AA)
        img = draw_collision_circle(img, q, gel_m, cam, collision_m, col)
        img = draw_sensor_frame(img, q, gel_m, cam, label=side[0].upper())
        cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(img, f"{hud}  t={i/FPS:4.2f}s  sep={probe['min_separation_m']:.2f}m",
                    (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        cv2.imwrite(str(tmp / f"{i:05d}.png"), img[..., ::-1])
        n = i
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4)],
                   check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return n + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--view", default="middle")
    ap.add_argument("--out", default="/media/yxma/Disk1/twm/probe_preview")
    ap.add_argument("--allow-leaving-view", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = T.load_calibration(stage)
    cam = cal["cams"][args.view]

    eps = _episodes()
    rng = np.random.default_rng(0)
    records, run = [], 0
    tried = 0
    while run < args.runs and tried < args.runs * 12:
        tried += 1
        date, ep = eps[int(rng.integers(len(eps)))]
        t = pq.read_table(RELEASE / date / f"{ep}.parquet",
                          columns=["sensor_left_pose", "sensor_right_pose",
                                   "source_h5_frame"]).to_pydict()
        poses = {sd: np.asarray([x for x in t[f"sensor_{sd}_pose"]], float)
                 for sd in ("left", "right")}
        trim = int(np.asarray(t["source_h5_frame"])[0])
        try:
            r = T.sample_probe(poses, cal, seed=int(rng.integers(1 << 30)),
                               view=args.view,
                               allow_leaving_view=args.allow_leaving_view)
        except ValueError:
            continue                     # this frame cannot hold all 12; try another
        row = int(r["context_rows"][-1])
        with h5py.File(str(H5ROOT / date / f"{ep}.h5"), "r") as f5:
            frame = f5[f"realsense/cam{CAM_H5[args.view]}/color"][trim + row][..., ::-1]
        for p in r["probes"]:
            amp = (p["amplitude_m"] if p["kind"] == "translation"
                   else np.radians(p["amplitude_deg"]) * 1.0)
            speed = (p["per_step_mm"] * FPS / 1000.0 if p["kind"] == "translation"
                     else p["per_step_deg"] * FPS)
            name = f"run{run}_{p['name']}"
            hud = (f"{p['name']}  "
                   + (f"{p['amplitude_m']*100:.1f}cm" if p["kind"] == "translation"
                      else f"{p['amplitude_deg']:.1f}deg")
                   + f"  {p['horizon_s']:.2f}s  p{p['speed_percentile']:.0f}")
            nf = _render_probe(frame, p, cal, cam,
                               out / "clips" / f"{name}.mp4", hud,
                               r["held_pose"], r["collision_m"])
            records.append({
                "run": run, "name": p["name"], "kind": p["kind"],
                "moving_side": p["moving_side"],
                "min_separation_m": round(p["min_separation_m"], 4),
                "axis": p["axis"], "clip": f"clips/{name}.mp4",
                "episode": f"{date}/{ep}", "context_rows": [int(x) for x in r["context_rows"]],
                "steps": p["n_steps"], "horizon_s": round(p["horizon_s"], 3),
                "amplitude": round(p["amplitude_m"], 4) if p["kind"] == "translation"
                             else round(p["amplitude_deg"], 2),
                "amplitude_unit": "m" if p["kind"] == "translation" else "deg",
                "speed": round(speed, 4),
                "speed_unit": "m/s" if p["kind"] == "translation" else "deg/s",
                "per_step": round(p["per_step_mm"], 4) if p["kind"] == "translation"
                            else round(p["per_step_deg"], 4),
                "per_step_unit": "mm/step" if p["kind"] == "translation" else "deg/step",
                "speed_percentile": round(p["speed_percentile"], 1),
                "in_view": bool(p.get("in_view", True)),
                "frames": nf,
            })
            print(f"  run{run} [{p['moving_side'][0].upper()}] {p['name']:8s} "
                  f"{nf:4d}f  {hud}  sep={p['min_separation_m']:.3f}m", flush=True)
        run += 1

    (out / "probes.json").write_text(json.dumps(records, indent=1))
    _write_page(out, records, args)
    print(f"\n{len(records)} clips -> {out}/index.html")
    return 0


def _write_page(out: Path, recs, args) -> None:
    from collections import defaultdict
    by = defaultdict(list)
    for r in recs:
        by[r["name"]].append(r)
    order = [f"trans{a}" for a in ("+x", "-x", "+y", "-y", "+z", "-z")] + \
            [f"rot{a}" for a in ("+x", "-x", "+y", "-y", "+z", "-z")]
    dp = "p25 0.971 / p50 2.813 / p90 10.158 mm per step"
    da = "p25 0.320 / p50 0.699 / p90 2.296 deg per step"
    rows = []
    for name in order:
        for r in by.get(name, []):
            rows.append(
                f"<tr><td>{r['name']}</td><td>{r['run']}</td>"
                f"<td>{r['moving_side'][0].upper()}</td>"
                f"<td>{r['amplitude']} {r['amplitude_unit']}</td>"
                f"<td>{r['speed']} {r['speed_unit']}</td>"
                f"<td>{r['per_step']} {r['per_step_unit']}</td>"
                f"<td>p{r['speed_percentile']:.0f}</td>"
                f"<td>{r['horizon_s']} s</td><td>{r['steps']}</td>"
                f"<td>{r['min_separation_m']} m</td>"
                f"<td class='{'ok' if r['in_view'] else 'bad'}'>"
                f"{'in view' if r['in_view'] else 'LEAVES VIEW'}</td>"
                f"<td class='dim'>{r['episode']}</td></tr>")
    # THE COLLAPSED STATE MUST STILL SAY SOMETHING. Sixty rows pushed the
    # clips off the first two screens, and the clips are the point of the
    # page; but a <details> that hides everything trades one problem for
    # another. These five numbers are what a reader checks before watching
    # anything, so they stay visible and the per-row detail folds away.
    import statistics as _st
    amps_t = [r["amplitude"] for r in recs if r["kind"] == "translation"]
    amps_r = [r["amplitude"] for r in recs if r["kind"] == "rotation"]
    pcts = [r["speed_percentile"] for r in recs]
    seps = [r["min_separation_m"] for r in recs]
    hands = sum(1 for r in recs if r["moving_side"] == "left")
    cards_summary = [
        f"<div class='card'><b>{len(recs)}</b><span>probes &middot; "
        f"{len(set(r['run'] for r in recs))} start frames</span></div>",
        f"<div class='card'><b>{hands}L / {len(recs)-hands}R</b>"
        f"<span>moving hand, drawn at random</span></div>",
        f"<div class='card'><b>{min(amps_t):.2f}&ndash;{max(amps_t):.2f} m</b>"
        f"<span>translation &middot; {min(amps_r):.0f}&ndash;{max(amps_r):.0f}&deg; rotation</span></div>",
        f"<div class='card'><b>p{min(pcts):.0f}&ndash;p{max(pcts):.0f}</b>"
        f"<span>speed vs the dataset (median p{_st.median(pcts):.0f})</span></div>",
        f"<div class='card'><b>{min(seps):.3f} m</b>"
        f"<span>closest the hands ever come (rule: 0.12 m)</span></div>",
    ]
    cards = []
    for name in order:
        clips = "".join(
            f"<figure><video src='{r['clip']}' controls loop muted "
            f"preload='metadata'></video><figcaption>run {r['run']} &middot; "
            f"{r['amplitude']} {r['amplitude_unit']} &middot; {r['speed']} "
            f"{r['speed_unit']} &middot; {r['horizon_s']} s &middot; "
            f"p{r['speed_percentile']:.0f}</figcaption></figure>"
            for r in by.get(name, []))
        cards.append(f"<h3>{name}</h3><div class='grid'>{clips}</div>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic probe review</title><style>
:root{{--bg:#0b1020;--fg:#e8eefb;--dim:#8ea0c2;--line:#1e2a45;--card:#111a2e;
--ok:#7be0a0;--bad:#ff8f7a;--accent:#ffc46b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 'IBM Plex Sans',system-ui,sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;padding:0 20px 72px}}
h1{{font-size:32px;margin:32px 0 8px}}h3{{margin:32px 0 8px;color:var(--accent)}}
p{{max-width:76ch;color:var(--dim)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:8px}}
video{{width:100%;border-radius:6px;background:#000}}
figcaption{{color:var(--dim);font-size:13px;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--dim);font-weight:500}}
.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}.dim{{color:var(--dim)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:10px;margin:20px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 14px}}
.card b{{display:block;font-size:24px;color:var(--accent);font-weight:650}}
.card span{{color:var(--dim);font-size:12px}}
details{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;margin:16px 0}}
summary{{cursor:pointer;color:var(--dim);font-size:14px;min-height:32px;
display:flex;align-items:center}}
</style></head><body><div class="wrap">
<h1>Synthetic probe review</h1>
<p>Twelve axis-aligned probes &times; {args.runs} independent start frames.
Each clip freezes the start camera frame and moves only the GT sensor-pose
marker along its projected trajectory &mdash; the future image is unknown, so
animating anything else would be inventing pixels. This is the reference a
model rollout is compared against.</p>
<p>Speeds are quoted against the measured dataset distribution over 480,008
published rows at 30&nbsp;Hz: translation {dp}; rotation {da}. A 1.5&nbsp;s
horizon floor and dataset-median speed are incompatible at the small end of
the requested amplitudes, so short probes run slower than the median &mdash;
the percentile column says by how much.</p>
<p>Start frames were sampled independently of the action sets and accepted or
rejected against them; probes are never adjusted to fit a frame. Every probe
here keeps its marker inside the view{'' if not args.allow_leaving_view else
' &mdash; EXCEPT this run was built with --allow-leaving-view'}.</p>
<div class="cards">{''.join(cards_summary)}</div>
<details><summary>Per-probe numbers &mdash; {len(recs)} rows</summary>
<div style="overflow-x:auto"><table><thead><tr><th>probe</th><th>run</th>
<th>hand</th><th>amplitude</th><th>speed</th><th>per step</th>
<th>vs dataset</th><th>horizon</th><th>steps</th><th>sep</th><th>view</th>
<th>source</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></details>
{''.join(cards)}
</div></body></html>"""
    (out / "index.html").write_text(html)


if __name__ == "__main__":
    raise SystemExit(main())
