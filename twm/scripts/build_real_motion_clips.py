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
WINDOWS = (12, 18, 24, 30, 45)   # frames; short windows are likelier to be pure
STRIDE = 4

# PURITY IS A PROPERTY OF THE PATH, NOT THE ENDPOINTS. The first version of
# this search compared the first and last pose only. A window can wander
# 110 mm off-axis and reverse eight times and still finish where a clean
# motion would have -- measured on the clip it selected for trans+z:
# straightness 0.227, per-step dominance 0.109. Every one of the six rotation
# clips it picked swung the gel 37-50 mm out and back, while its endpoint
# test read "the gel held to within 9.9 mm".
STRAIGHTNESS = 0.90      # |net| / path length          (translation)
MONOTONICITY = 0.80      # |net angle| / summed |steps|  (rotation)
STEP_DOM_T = 0.90        # median per-step axis share    (translation)
STEP_DOM_R = 0.85        # median per-step axis share    (rotation)
TRANS_MIN_MM, TRANS_MAX_DEG = 8.0, 6.0
ROT_MIN_DEG, ROT_MAX_GEL_MM = 6.0, 15.0   # MAX gel excursion over the path
COL = {"left": (63, 210, 255), "right": (247, 195, 79)}        # BGR
AXES = [(a, s) for a in range(3) for s in (+1, -1)]


def _purity(G, Q):
    """Path statistics for one window: (N,3) gel mm and N rotations."""
    d = np.diff(G, axis=0)
    L = np.linalg.norm(d, axis=1)
    net = G[-1] - G[0]
    nl = float(np.linalg.norm(net))
    rs = np.array([(Q[i + 1] * Q[i].inv()).as_rotvec(degrees=True)
                   for i in range(len(Q) - 1)])
    rl = np.linalg.norm(rs, axis=1)
    netr = (Q[-1] * Q[0].inv()).as_rotvec(degrees=True)
    nr = float(np.linalg.norm(netr))
    return {
        "d": d, "L": L, "net": net, "trans_mm": nl,
        "rs": rs, "rl": rl, "netr": netr, "rot_deg": nr,
        "straightness": nl / max(L.sum(), 1e-9),
        "monotonicity": nr / max(rl.sum(), 1e-9),
        # the gel's WORST excursion, not where it ended up
        "gel_excursion_mm": float(np.linalg.norm(G - G[0], axis=1).max()),
    }


def _scan(cal):
    """Every window at every length, already scored for purity."""
    out = []
    for W in WINDOWS:
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
                    if idx[e] - idx[s] != W:
                        continue
                    st = _purity(gel[s:e + 1], R[s:e + 1])
                    st.update({"date": date, "episode": ep, "side": side,
                               "row0": int(idx[s]), "row1": int(idx[e]),
                               "window": W})
                    out.append(st)
    return out


def _pick(cands):
    best = {}
    for c in cands:
        L, d, rl, rs = c["L"], c["d"], c["rl"], c["rs"]
        mt, mr = L > 0.2, rl > 0.15
        for ax, sg in AXES:
            nm = ("+" if sg > 0 else "-") + "xyz"[ax]
            # TRANSLATION: straight path, each step along the axis, no turn
            if (c["trans_mm"] > TRANS_MIN_MM and c["rot_deg"] < TRANS_MAX_DEG
                    and mt.sum() > 6 and c["straightness"] > STRAIGHTNESS
                    and np.sign(c["net"][ax]) == sg):
                dom = float(np.median(np.abs(d[mt][:, ax]) / L[mt]))
                if dom > STEP_DOM_T:
                    k = f"trans{nm}"
                    score = c["straightness"] * dom * min(c["trans_mm"] / 60, 1)
                    if score > best.get(k, {}).get("_score", 0):
                        best[k] = {**c, "_score": score, "kind": "translation",
                                   "axis": nm, "step_dominance": dom,
                                   "amount": c["trans_mm"], "unit": "mm",
                                   "purity": c["straightness"],
                                   "purity_kind": "straightness",
                                   "counter": c["rot_deg"], "counter_unit": "deg"}
            # ROTATION: monotonic turn, each step about the axis, gel held
            if (c["rot_deg"] > ROT_MIN_DEG and mr.sum() > 6
                    and c["monotonicity"] > MONOTONICITY
                    and c["gel_excursion_mm"] < ROT_MAX_GEL_MM
                    and np.sign(c["netr"][ax]) == sg):
                dom = float(np.median(np.abs(rs[mr][:, ax]) / rl[mr]))
                if dom > STEP_DOM_R:
                    k = f"rot{nm}"
                    score = c["monotonicity"] * dom * min(c["rot_deg"] / 20, 1)
                    if score > best.get(k, {}).get("_score", 0):
                        best[k] = {**c, "_score": score, "kind": "rotation",
                                   "axis": nm, "step_dominance": dom,
                                   "amount": c["rot_deg"], "unit": "deg",
                                   "purity": c["monotonicity"],
                                   "purity_kind": "monotonicity",
                                   "counter": c["gel_excursion_mm"],
                                   "counter_unit": "mm"}
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
               f"pure {rec['purity']:.2f}  {rec['side']}")
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
    cands = _scan(cal)
    best = _pick(cands)
    print(f"{len(cands):,} windows scanned over {len(WINDOWS)} lengths, "
          f"{len(best)}/12 axes matched")
    missing = sorted({f"{k}{s2}{a}" for k in ("trans", "rot")
                      for s2 in "+-" for a in "xyz"} - set(best))
    if missing:
        print(f"  NO CLEAN WINDOW EXISTS for: {missing} — these are reported "
              f"as absent rather than filled with a clip that is not what it "
              f"claims")

    recs = []
    for k in sorted(best, key=lambda n: (n.startswith("rot"), n[-1], n[-2])):
        r = {kk: vv for kk, vv in best[k].items()
             if not kk.startswith("_") and not isinstance(vv, np.ndarray)}
        n = _render(r, cal, out / f"{k}.mp4")
        r.update({"name": k, "clip": f"real/{k}.mp4", "frames": n})
        recs.append(r)
        print(f"  {k:9s} W={r['window']:2d} {r['date']}/{r['episode']} "
              f"{r['side']:5s} {r['amount']:6.1f}{r['unit']}  "
              f"{r['purity_kind'][:5]} {r['purity']:.3f}  "
              f"step {r['step_dominance']:.3f}  "
              f"holds {r['counter']:.1f}{r['counter_unit']}  {n}f", flush=True)
    (out / "real_motion.json").write_text(json.dumps(recs, indent=1))
    print(f"\n{len(recs)} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
