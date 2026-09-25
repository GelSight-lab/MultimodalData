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
# PURITY IS A PROPERTY OF THE PATH. The endpoint version of this check passed
# a trans+z clip that wandered 110 mm off-axis with a per-step dominance of
# 0.109, and passed six rotation clips whose gel swung 37-50 mm out and back
# while their endpoint test read "held to within 9.9 mm".
STRAIGHTNESS, MONOTONICITY = 0.90, 0.80
STEP_DOM_T, STEP_DOM_R = 0.90, 0.85
TRANS_MIN_MM, TRANS_MAX_DEG = 8.0, 6.0
ROT_MIN_DEG, ROT_MAX_GEL_MM = 6.0, 15.0


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
        P = np.asarray([x for x in t[f"sensor_{r['side']}_pose"]],
                       float)[r["row0"]:r["row1"] + 1]
        R = Rotation.from_quat(P[:, 3:7])
        g = np.tile(np.asarray(cal[f"gel_{r['side']}"], float), (len(P), 1))
        G = P[:, :3] * 1000.0 + R.apply(g)
        d = np.diff(G, axis=0); L = np.linalg.norm(d, axis=1)
        net = G[-1] - G[0]; nl = float(np.linalg.norm(net))
        rs = np.array([(R[i + 1] * R[i].inv()).as_rotvec(degrees=True)
                       for i in range(len(R) - 1)])
        rl = np.linalg.norm(rs, axis=1)
        netr = (R[-1] * R[0].inv()).as_rotvec(degrees=True)
        nr = float(np.linalg.norm(netr))
        ax = "xyz".index(r["axis"][1])
        sg = 1 if r["axis"][0] == "+" else -1
        rows += 1
        if r["kind"] == "translation":
            st = nl / max(L.sum(), 1e-9)
            m = L > 0.2
            dom = float(np.median(np.abs(d[m][:, ax]) / L[m])) if m.any() else 0
            ok = (st > STRAIGHTNESS and dom > STEP_DOM_T and nl > TRANS_MIN_MM
                  and nr < TRANS_MAX_DEG and np.sign(net[ax]) == sg)
            if not ok:
                bad.append(f"{r['name']}: straightness {st:.2f}, step "
                           f"dominance {dom:.2f}, {nl:.0f} mm, turned "
                           f"{nr:.1f} deg")
        else:
            mo = nr / max(rl.sum(), 1e-9)
            m = rl > 0.15
            dom = float(np.median(np.abs(rs[m][:, ax]) / rl[m])) if m.any() else 0
            exc = float(np.linalg.norm(G - G[0], axis=1).max())
            ok = (mo > MONOTONICITY and dom > STEP_DOM_R and nr > ROT_MIN_DEG
                  and exc < ROT_MAX_GEL_MM and np.sign(netr[ax]) == sg)
            if not ok:
                bad.append(f"{r['name']}: monotonicity {mo:.2f}, step "
                           f"dominance {dom:.2f}, {nr:.1f} deg, gel excursion "
                           f"{exc:.1f} mm")
    check(not bad and rows == len(recs),
          "every clip's PATH is the motion it is labelled, not just its ends",
          f"{rows} windows re-measured from the parquet: straightness / "
          f"monotonicity, per-step axis share, and the gel's WORST excursion"
          if not bad else "; ".join(bad[:4]))

    # the numbers the page prints must be the ones the data has
    drift = []
    for r in recs:
        if not (0 <= r["purity"] <= 1) or not (0 <= r["step_dominance"] <= 1):
            drift.append(r["name"])
    tr = [r for r in recs if r["kind"] == "translation"]
    ro = [r for r in recs if r["kind"] == "rotation"]
    check(not drift and tr and ro
          and min(r["purity"] for r in tr) > STRAIGHTNESS
          and min(r["purity"] for r in ro) > MONOTONICITY
          and max(r["counter"] for r in ro) < ROT_MAX_GEL_MM,
          "purity is reported, and every clip clears its floor",
          f"translation straightness {min(r['purity'] for r in tr):.3f}-"
          f"{max(r['purity'] for r in tr):.3f}; rotation monotonicity "
          f"{min(r['purity'] for r in ro):.3f}-"
          f"{max(r['purity'] for r in ro):.3f}; worst gel excursion "
          f"{max(r['counter'] for r in ro):.1f} mm")

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
