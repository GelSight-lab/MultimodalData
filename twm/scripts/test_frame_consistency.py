"""One world frame across all episodes — checked against the depth camera.

WHY THE OTHER CHECKS CANNOT SEE THIS

`test_world_frame_declared` verifies each episode reproduces its OWN
fingerprint, and `test_world_transform` compares table normals. Both are
invariant to a shared world transform, which is exactly the error a
recalibration produces. Nothing compared episodes against something OUTSIDE
the mocap chain.

THE INVARIANT USED HERE

At contact the gel is on the board, the board is on the table, and the table
is fixed relative to the cameras. So the gel's depth in camera coordinates —
which comes from the MOCAP chain — must agree with the depth the camera
MEASURES nearby, which comes from the camera alone. That couples the two
chains and is independent of where the operator put their hands, which is why
a projected-pixel scatter (23 px here) cannot do the job.

Depth is sampled in an ANNULUS: the tool occludes its own contact point, so
the centre pixel sees the tool while the ring sees the board it presses.

WHAT IS AND IS NOT CLAIMED

Absolute agreement is not claimed — depth may not be registered to colour, and
the board has relief. That bias is constant per camera (measured +27 / +7 /
+42 mm for left / middle / right) and CANCELS when dates are compared, which
is the only comparison made.

Measured: 2026-05-10 and 2026-05-11 agree to about 1 mm per camera.
2026-05-19 sits 8-20 mm away depending on the camera. Solving the three
camera axes for one translation puts it 20-26 mm from the reference — against
a reference-to-reference floor of 8.7 mm, so the effect is roughly twice the
method's own noise, not cleanly separated. The test therefore asserts the
FLOOR (the trusted dates must agree) and REPORTS 05-19 rather than failing on
it: a threshold tighter than the instrument would just be a coin flip.

    python scripts/test_frame_consistency.py
"""
from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import force_meta, raw_root   # noqa: E402

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
REL = force_meta("motherboard")
H5R = raw_root("motherboard")
CAM_H5 = {"left": 1, "middle": 2, "right": 0}
VIEWS = ("left", "middle", "right")
TRUSTED = ("2026-05-10", "2026-05-11")
R_IN, R_OUT = 26, 46
FORCE_MIN = 3.0


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def residuals(date, ep, view, cal, n=14):
    import h5py, hdf5plugin                                    # noqa: F401
    from scipy.spatial.transform import Rotation
    t = pq.read_table(REL / date / f"{ep}.parquet").to_pydict()
    trim = int(np.asarray(t["source_h5_frame"])[0])
    cam = cal["cams"][view]
    T = np.asarray(cam["T_mocap_to_cam"], float); K = cam["intrinsics"]
    out = []
    with h5py.File(str(H5R / date / f"{ep}.h5"), "r") as f:
        ds = f[f"realsense/cam{CAM_H5[view]}/depth"]
        for side in ("left", "right"):
            F = np.asarray(t[f"force_{side}_normal_n"], float)
            S = np.asarray([x for x in t[f"sensor_{side}_pose"]], float)
            m = (F > FORCE_MIN) & np.isfinite(S).all(1) \
                & (np.linalg.norm(S[:, 3:], axis=1) > .5)
            rows = np.flatnonzero(m)
            if len(rows) < 4:
                continue
            for r in rows[np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)]:
                if not (0 <= trim + r < len(ds)):
                    continue
                q = S[r]
                g = q[:3]*1000.0 + Rotation.from_quat(q[3:7]).as_matrix() @ cal[f"gel_{side}"]
                X = T[:3, :3] @ g + T[:3, 3]
                u = K["fx"]*X[0]/X[2] + K["ppx"]; v = K["fy"]*X[1]/X[2] + K["ppy"]
                if not (R_OUT < u < 640 - R_OUT and R_OUT < v < 480 - R_OUT):
                    continue
                patch = ds[trim + int(r)][int(v)-R_OUT:int(v)+R_OUT+1,
                                          int(u)-R_OUT:int(u)+R_OUT+1].astype(np.float32)
                yy, xx = np.mgrid[int(v)-R_OUT:int(v)+R_OUT+1, int(u)-R_OUT:int(u)+R_OUT+1]
                rr = np.hypot(xx - u, yy - v)
                sel = (rr >= R_IN) & (rr <= R_OUT) & (patch > 200)
                if sel.sum() < 60:
                    continue
                out.append(float(X[2]) - float(np.median(patch[sel])))
    return np.asarray(out)


def main() -> int:
    from react_toolbox.calibration import load_calibration
    from twm.calib_epoch import calib_dir

    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = load_calibration(stage)

    per = {v: {} for v in VIEWS}
    for p in sorted(REL.glob("*/*.parquet")):
        key = f"{p.parent.name}/{p.stem}"
        for v in VIEWS:
            d = residuals(p.parent.name, p.stem, v, cal)
            if len(d) >= 6:
                per[v][key] = float(np.median(d))

    def by_date(v):
        g = {}
        for k, x in per[v].items():
            g.setdefault(k.split("/")[0], []).append(x)
        return {d: (float(np.median(a)), float(np.std(a)), len(a)) for d, a in g.items()}

    D = {v: by_date(v) for v in VIEWS}

    # 1 — the trusted sessions must agree, per camera. This is the floor.
    gaps = []
    for v in VIEWS:
        a, b = D[v].get(TRUSTED[0]), D[v].get(TRUSTED[1])
        if a and b:
            gaps.append((v, abs(a[0] - b[0])))
    worst = max(gaps, key=lambda x: x[1])
    check(worst[1] < 8.0,
          "the two untouched sessions place the gel at the same depth",
          "  ".join(f"{v} {g:.1f}mm" for v, g in gaps) +
          f"  (worst {worst[0]} {worst[1]:.1f} mm; this is the method's floor)")

    # 2 — every episode is close to its own date's median. A single episode in
    #     a different frame would show here even if the date as a whole is fine.
    far = []
    for v in VIEWS:
        for k, x in per[v].items():
            med = D[v][k.split("/")[0]][0]
            if abs(x - med) > 25.0:
                far.append(f"{v} {k} {x-med:+.0f}mm")
    check(not far, "no single episode sits in a different frame from its session",
          f"{sum(len(per[v]) for v in VIEWS)} episode-camera pairs, all within "
          f"25 mm of their session median" + (f"; {far[:3]}" if far else ""))

    # 3 — REPORT the recalibrated session rather than failing on it. Three
    #     camera axes solve for one translation; the reference-to-reference
    #     control says the instrument's own floor is comparable to the effect.
    if "2026-05-19" in D["middle"]:
        A = np.array([np.asarray(cal["cams"][v]["T_mocap_to_cam"], float)[2, :3]
                      for v in VIEWS])
        def solve(tgt, ref):
            b = np.array([D[v][tgt][0] - D[v][ref][0] for v in VIEWS])
            return np.linalg.solve(A, b)
        ctl = np.linalg.norm(solve(TRUSTED[1], TRUSTED[0]))
        got = np.linalg.norm(solve("2026-05-19", TRUSTED[0]))
        check(True, "2026-05-19's residual, reported not asserted",
              f"{got:.1f} mm from {TRUSTED[0]} against a "
              f"{TRUSTED[1]}-vs-{TRUSTED[0]} floor of {ctl:.1f} mm — about "
              f"2x the instrument's own noise, so it is recorded, not corrected")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nframe consistency: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
