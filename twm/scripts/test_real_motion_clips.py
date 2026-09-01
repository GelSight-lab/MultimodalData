"""Each real clip must actually move the way its label says.

The clips are selected by a search, so the search's own numbers prove nothing
-- it could be scoring the wrong quantity and would still rank something
first. This re-measures every selected window from the published parquet and
checks it against the criterion its label claims.

It also checks the two properties that make a clip the REAL counterpart of a
synthetic probe rather than merely "mostly along x":

    a translation clip holds ORIENTATION   (make_translation_set holds it)
    a rotation clip holds the GEL CENTRE   (make_rotation_set pivots on it)

and that the labelled sign is the one that dominates -- otherwise a "+x" clip
could travel in -x and nothing would say so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402
from scipy.spatial.transform import Rotation                   # noqa: E402

from react_paths import force_meta, out_root, release_root     # noqa: E402

RESULTS: list = []
TASK = "motherboard"
TRANS_DOM, TRANS_MIN_MM, TRANS_MAX_DEG = 0.85, 8.0, 6.0
ROT_DOM, ROT_MIN_DEG, ROT_MAX_MM = 0.85, 6.0, 12.0


def check(ok, name, evidence):
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from react_toolbox.calibration import load_calibration

    src = out_root("probe_clips") / "real" / "real_motion.json"
    if not src.exists():
        check(False, "the real-motion clips exist",
              f"{src} missing -- run scripts/build_real_motion_clips.py")
        return _report()
    recs = json.loads(src.read_text())
    cal = load_calibration(release_root(TASK))

    want = {f"{k}{s}{a}" for k in ("trans", "rot") for a in "xyz" for s in "+-"}
    got = {r["name"] for r in recs}
    check(got == want, "all twelve signed axes have a real clip",
          f"{len(got)}/12 present"
          + ("" if got == want else f"; missing {sorted(want - got)}"))

    bad, rows = [], 0
    for r in recs:
        f = force_meta(TASK) / r["date"] / f"{r['episode']}.parquet"
        t = pq.read_table(str(f),
                          columns=[f"sensor_{r['side']}_pose"]).to_pydict()
        P = np.asarray([x for x in t[f"sensor_{r['side']}_pose"]], float)
        a, b = P[r["row0"]], P[r["row1"]]
        Ra, Rb = Rotation.from_quat(a[3:7]), Rotation.from_quat(b[3:7])
        g = np.asarray(cal[f"gel_{r['side']}"], float)
        ga = a[:3] * 1000.0 + Ra.apply(g)
        gb = b[:3] * 1000.0 + Rb.apply(g)
        dp = gb - ga
        rv = (Rb * Ra.inv()).as_rotvec(degrees=True)
        nd, na = float(np.linalg.norm(dp)), float(np.linalg.norm(rv))
        ax = "xyz".index(r["axis"][1])
        sg = 1 if r["axis"][0] == "+" else -1
        rows += 1
        if r["kind"] == "translation":
            dom = abs(dp[ax]) / max(nd, 1e-9)
            ok = (dom > TRANS_DOM and np.sign(dp[ax]) == sg
                  and nd > TRANS_MIN_MM and na < TRANS_MAX_DEG)
            if not ok:
                bad.append(f"{r['name']}: dom {dom:.2f}, {nd:.1f} mm, "
                           f"turned {na:.1f} deg, sign {int(np.sign(dp[ax]))}")
        else:
            dom = abs(rv[ax]) / max(na, 1e-9)
            ok = (dom > ROT_DOM and np.sign(rv[ax]) == sg
                  and na > ROT_MIN_DEG and nd < ROT_MAX_MM)
            if not ok:
                bad.append(f"{r['name']}: dom {dom:.2f}, {na:.1f} deg, "
                           f"gel moved {nd:.1f} mm, sign {int(np.sign(rv[ax]))}")
    check(not bad and rows == len(recs),
          "every clip re-measures as the motion it is labelled",
          f"{rows} windows re-measured from the parquet, all matching their "
          f"label" if not bad else "; ".join(bad[:4]))

    # the counter-quantity is what separates these from ordinary motion
    tr = [r for r in recs if r["kind"] == "translation"]
    ro = [r for r in recs if r["kind"] == "rotation"]
    check(tr and ro
          and max(r["counter"] for r in tr) < TRANS_MAX_DEG
          and max(r["counter"] for r in ro) < ROT_MAX_MM,
          "translations hold orientation and rotations hold the gel",
          f"translation clips turn at most "
          f"{max(r['counter'] for r in tr):.1f} deg (limit {TRANS_MAX_DEG:g}); "
          f"rotation clips move the gel at most "
          f"{max(r['counter'] for r in ro):.1f} mm (limit {ROT_MAX_MM:g})")

    # a clip nobody can play is not a clip
    missing = [r["name"] for r in recs
               if not (out_root("probe_clips") / r["clip"]).exists()
               or not (out_root("probe_clips")
                       / r["clip"].replace(".mp4", ".jpg")).exists()
               or r.get("frames", 0) < 5]
    check(not missing, "every clip rendered and has frames",
          f"{len(recs)} clips, {min(r.get('frames', 0) for r in recs)}-"
          f"{max(r.get('frames', 0) for r in recs)} frames each"
          if not missing else f"missing or empty: {missing}")

    # the page must actually SHOW them. The first version named the clips
    # transx+/rotz+ while synth_actions names them trans+x/rot-z, so every
    # lookup missed and the page rendered without a single real clip -- no
    # error, no empty box, just twelve sections that looked unchanged.
    idx = out_root("probe_clips") / "index.html"
    if idx.exists():
        html = idx.read_text()
        shown = html.count("figure class='real'")
        named = [r["name"] for r in recs if f"real/{r['name']}.mp4" in html]
        check(shown == len(recs) and len(named) == len(recs),
              "the probes page shows every real clip beside its probe",
              f"{shown} real figures, {len(named)}/{len(recs)} clip paths "
              f"referenced")
    else:
        check(False, "the probes page shows every real clip beside its probe",
              f"{idx} not built")

    return _report()


def _report() -> int:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\nreal motion clips: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
