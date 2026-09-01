"""Find REAL clips that move the way each synthetic probe commands, and render them.

The probe page shows twelve commanded motions — one signed axis each — drawn
over a frozen frame. They are synthetic by construction: nobody performed
them. A reader has no way to see what a "+x translation" or a "−y rotation"
actually looks like on this rig.

So: search the published poses for windows that really do move that way, and
render them from the real video.

The criteria mirror how the probes are BUILT, not some looser notion of
"mostly x":

  translation  the gel centre moves along one world axis (dominance > 0.85)
               by more than 8 mm, while the orientation turns less than 6° —
               `make_translation_set` holds orientation fixed.
  rotation     the orientation turns about one world axis by more than 6°
               while the gel centre moves less than 12 mm —
               `make_rotation_set` pivots ON the gel, so the gel stays put.

Both thresholds sit near the middle of the measured distribution (single-axis
dominance has a median of 0.894 for translation and 0.856 for rotation over
77,418 windows), so this selects the clean end of ordinary motion rather than
a handful of freak windows.

    python scripts/build_real_motion_clips.py
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402
from scipy.spatial.transform import Rotation                   # noqa: E402

from react_paths import force_meta, out_root, release_root     # noqa: E402
from react_toolbox.calibration import load_calibration, project_gel_frame  # noqa: E402
from react_toolbox.frames import require_up_axis               # noqa: E402

TASK = "motherboard"
W = 30                       # window length, frames
STRIDE = 5
TRANS_DOM, TRANS_MIN_MM, TRANS_MAX_DEG = 0.85, 8.0, 6.0
ROT_DOM, ROT_MIN_DEG, ROT_MAX_MM = 0.85, 6.0, 12.0
COL = {"left": (63, 210, 255), "right": (247, 195, 79)}        # BGR
AXES = [(a, s) for a in range(3) for s in (+1, -1)]


def _candidates(cal):
    """Every window, with the two quantities that decide what it is."""
    out = []
    for f in sorted(glob.glob(str(force_meta(TASK) / "*" / "*.parquet"))):
        date, ep = Path(f).parts[-2], Path(f).stem
        for side in ("left", "right"):
            t = pq.read_table(f, columns=[f"sensor_{side}_pose"]).to_pydict()
            P = np.asarray([x for x in t[f"sensor_{side}_pose"]], float)
            ok = np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)
            idx = np.where(ok)[0]
            if len(idx) < W + 1:
                continue
            R = Rotation.from_quat(P[idx, 3:7])
            g = np.tile(cal[f"gel_{side}"], (len(idx), 1))
            gel = P[idx, :3] * 1000.0 + R.apply(g)
            for s in range(0, len(idx) - W, STRIDE):
                e = s + W
                if idx[e] - idx[s] != W:          # frames must be consecutive
                    continue
                dp = gel[e] - gel[s]
                rv = (R[e] * R[s].inv()).as_rotvec(degrees=True)
                nd, na = float(np.linalg.norm(dp)), float(np.linalg.norm(rv))
                out.append({"date": date, "episode": ep, "side": side,
                            "row0": int(idx[s]), "row1": int(idx[e]),
                            "dp_mm": dp.tolist(), "rv_deg": rv.tolist(),
                            "trans_mm": nd, "rot_deg": na})
    return out


def _pick(cands):
    """The best real window per signed axis, for each kind."""
    best = {}
    for c in cands:
        dp, rv = np.asarray(c["dp_mm"]), np.asarray(c["rv_deg"])
        for ax, sg in AXES:
            # EXACTLY the name synth_actions.AXES produces: sign first. Mine
            # put the sign last, so every lookup on the page missed and the
            # real clips were simply absent -- no error, no empty box.
            nm = ("+" if sg > 0 else "-") + "xyz"[ax]
            # translation: one axis dominates the MOVE, orientation holds
            if c["trans_mm"] > TRANS_MIN_MM and c["rot_deg"] < TRANS_MAX_DEG:
                dom = abs(dp[ax]) / c["trans_mm"]
                if dom > TRANS_DOM and np.sign(dp[ax]) == sg:
                    k = f"trans{nm}"
                    score = dom * min(c["trans_mm"] / 60.0, 1.0)
                    if score > best.get(k, {}).get("_score", 0):
                        best[k] = {**c, "_score": score, "kind": "translation",
                                   "axis": nm, "dominance": dom,
                                   "amount": c["trans_mm"], "unit": "mm",
                                   "counter": c["rot_deg"], "counter_unit": "deg"}
            # rotation: one axis dominates the TURN, the gel stays put
            if c["rot_deg"] > ROT_MIN_DEG and c["trans_mm"] < ROT_MAX_MM:
                dom = abs(rv[ax]) / c["rot_deg"]
                if dom > ROT_DOM and np.sign(rv[ax]) == sg:
                    k = f"rot{nm}"
                    score = dom * min(c["rot_deg"] / 20.0, 1.0)
                    if score > best.get(k, {}).get("_score", 0):
                        best[k] = {**c, "_score": score, "kind": "rotation",
                                   "axis": nm, "dominance": dom,
                                   "amount": c["rot_deg"], "unit": "deg",
                                   "counter": c["trans_mm"], "counter_unit": "mm"}
    return best


def _render(rec, cal, out_mp4: Path, fps: float = 10.0) -> int:
    """The real video for that window, with the sensor frame drawn per frame."""
    src = (release_root(TASK) / "videos" / rec["date"] / rec["episode"]
           / "view_middle.mp4")
    t = pq.read_table(str(force_meta(TASK) / rec["date"]
                          / f"{rec['episode']}.parquet"),
                      columns=[f"sensor_{rec['side']}_pose"]).to_pydict()
    P = np.asarray([x for x in t[f"sensor_{rec['side']}_pose"]], float)
    cam, gel = cal["cams"]["middle"], cal[f"gel_{rec['side']}"]
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec["row0"])
    frames, trail = [], []
    for r in range(rec["row0"], rec["row1"] + 1):
        ok, fr = cap.read()
        if not ok:
            break
        pr = project_gel_frame(P[r], gel, cam)
        if pr is not None:
            c = pr["centre"]
            trail.append((int(c[0]), int(c[1])))
            for k, tip in enumerate(pr["tips"]):
                if tip is None:
                    continue
                cv2.line(fr, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])),
                         [(0, 0, 255), (0, 255, 0), (255, 128, 0)][k], 2,
                         cv2.LINE_AA)
            for i in range(1, len(trail)):
                cv2.line(fr, trail[i - 1], trail[i], COL[rec["side"]], 2,
                         cv2.LINE_AA)
            cv2.circle(fr, trail[0], 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(fr, (int(c[0]), int(c[1])), 5, COL[rec["side"]], 2,
                       cv2.LINE_AA)
        hud = (f"real {rec['kind'][:5]} {rec['axis']}  "
               f"{rec['amount']:.0f}{rec['unit']}  "
               f"dom {rec['dominance']:.2f}  {rec['side']}")
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(fr, hud, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (235, 235, 235), 1, cv2.LINE_AA)
        frames.append(fr)
    cap.release()
    if not frames:
        return 0
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out_mp4)], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    # A poster, or the browser shows a black box until someone presses play —
    # and this clip sits FIRST in its section, so the black box is the first
    # thing on the row.
    cv2.imwrite(str(out_mp4.with_suffix(".jpg")), frames[0],
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(out_root("probe_clips") / "real"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cal = load_calibration(release_root(TASK))
    require_up_axis(cal, where=f"{release_root(TASK)}/calibration")
    cands = _candidates(cal)
    best = _pick(cands)
    print(f"{len(cands):,} windows scanned, {len(best)}/12 axes matched")

    recs = []
    for k in sorted(best, key=lambda n: (n.startswith("rot"), n[-1], n[-2])):
        r = {kk: vv for kk, vv in best[k].items() if not kk.startswith("_")}
        n = _render(r, cal, out / f"{k}.mp4")
        r.update({"name": k, "clip": f"real/{k}.mp4", "frames": n})
        recs.append(r)
        print(f"  {k:9s} {r['date']}/{r['episode']} {r['side']:5s} "
              f"rows {r['row0']}-{r['row1']}  {r['amount']:6.1f}{r['unit']}  "
              f"dom {r['dominance']:.3f}  counter {r['counter']:.1f}"
              f"{r['counter_unit']}  {n}f", flush=True)
    (out / "real_motion.json").write_text(json.dumps(recs, indent=1))
    print(f"\n{len(recs)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
