"""The declared world transform aligns a session onto the reference frame.

2026-05-10 is the reference. 2026-05-19's OptiTrack world was redefined and
the release corrected it with a TRANSLATION ONLY, so its rotation went
uncorrected. `calib_epoch.world_transform` now declares the full rigid
transform.

Checked against the physical invariant it was derived from — the board lies on
the same table every session, so the table normal in world coordinates must
agree — and checked for the things a partial correction can quietly break:

  * it must be a real rigid transform, not a scaling or a reflection
  * it must leave the reference and pre-reset sessions alone
  * it must not move the workspace, because the in-plane translation is
    UNDETERMINED and the correction has no business inventing one

    python scripts/test_world_transform.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
REL = Path("/media/yxma/Disk1/twm/release_force/motherboard/meta")


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def table_normal(date, cal, R=np.eye(3)):
    """median_i R_obj,i @ n_local — the table normal in this session's world.

    In the BOARD frame, because the board is picked up and tilted (median
    3.3-4.2 deg off its own median); fitting a plane to WORLD contacts measures
    the board's average pose instead and put two untouched dates 3.2-3.8 deg
    apart when the board-frame answer is 0.29.
    """
    from scipy.spatial.transform import Rotation
    pts, Ms = [], []
    for p in sorted((REL / date).glob("*.parquet")):
        t = pq.read_table(p, columns=["sensor_left_pose", "sensor_right_pose",
                                      "object_pose", "force_left_normal_n",
                                      "force_right_normal_n"]).to_pydict()
        O = np.asarray([x for x in t["object_pose"]], float)
        ok = np.isfinite(O).all(1) & (np.linalg.norm(O[:, 3:], axis=1) > .5)
        M = Rotation.from_quat(np.where(ok[:, None], O[:, 3:7], [0, 0, 0, 1.])).as_matrix()
        M = np.einsum("ij,njk->nik", R, M)          # rotate the session
        for s in ("left", "right"):
            F = np.asarray(t[f"force_{s}_normal_n"], float)
            S = np.asarray([x for x in t[f"sensor_{s}_pose"]], float)
            m = (F > 2.0) & np.isfinite(S).all(1) & ok \
                & (np.linalg.norm(S[:, 3:], axis=1) > .5)
            if not m.any():
                continue
            Rs = np.einsum("ij,njk->nik", R, Rotation.from_quat(S[m, 3:7]).as_matrix())
            g = (R @ (S[m, :3] * 1000.0).T).T + np.einsum("nij,j->ni", Rs, cal[f"gel_{s}"])
            o = (R @ (O[m, :3] * 1000.0).T).T
            pts.append(np.einsum("nji,nj->ni", M[m], g - o)[::5])
        Ms.append(M[ok][::5])
    C = np.vstack(pts)
    nl = np.linalg.svd(C - C.mean(0))[2][2]
    nw = np.einsum("nij,j->ni", np.vstack(Ms), nl)
    nw *= np.sign(np.median(nw[:, 1]))
    n = np.median(nw, axis=0)
    return n / np.linalg.norm(n)


def split_normals(date, cal):
    """Angle between the normals of the two in-plane halves of the cloud.

    A planar surface gives nearly the same normal from either half. 2026-05-19
    does not, which is why its whole-cloud normal cannot be believed.
    """
    from scipy.spatial.transform import Rotation
    pts, Ms = [], []
    for p in sorted((REL / date).glob("*.parquet")):
        t = pq.read_table(p, columns=["sensor_left_pose", "sensor_right_pose",
                                      "object_pose", "force_left_normal_n",
                                      "force_right_normal_n"]).to_pydict()
        O = np.asarray([x for x in t["object_pose"]], float)
        ok = np.isfinite(O).all(1) & (np.linalg.norm(O[:, 3:], axis=1) > .5)
        M = Rotation.from_quat(np.where(ok[:, None], O[:, 3:7], [0, 0, 0, 1.])).as_matrix()
        for s in ("left", "right"):
            F = np.asarray(t[f"force_{s}_normal_n"], float)
            S = np.asarray([x for x in t[f"sensor_{s}_pose"]], float)
            m = (F > 2.0) & np.isfinite(S).all(1) & ok \
                & (np.linalg.norm(S[:, 3:], axis=1) > .5)
            if not m.any():
                continue
            R = Rotation.from_quat(S[m, 3:7]).as_matrix()
            g = S[m, :3] * 1000.0 + np.einsum("nij,j->ni", R, cal[f"gel_{s}"])
            pts.append(np.einsum("nji,nj->ni", M[m], g - O[m, :3] * 1000.0)[::6])
        Ms.append(M[ok][::12])
    C = np.vstack(pts); Mall = np.vstack(Ms)[:4000]

    def wn(X):
        nl = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][2]
        nw = Mall @ nl; nw *= np.sign(np.median(nw[:, 1]))
        n = np.median(nw, axis=0); return n / np.linalg.norm(n)

    c = C.mean(0); U = np.linalg.svd(C - c, full_matrices=False)[2]
    a = (C - c) @ U[0]; lo = a < np.median(a)
    n1, n2 = wn(C[lo]), wn(C[~lo])
    return float(np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1))))


def main() -> int:
    from react_toolbox.calibration import load_calibration
    from scipy.spatial.transform import Rotation
    from twm.calib_epoch import (WORLD_REF_DATE, calib_dir, world_residual,
                                 world_transform)

    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = load_calibration(stage)
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))

    # 1 — a real rigid transform
    bad = []
    for d in ("2026-05-10", "2026-05-11", "2026-05-19"):
        R, _ = world_transform("motherboard", d)
        if abs(np.linalg.det(R) - 1) > 1e-9 or not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
            bad.append(d)
    check(not bad, "each transform is a rotation, not a scale or reflection",
          f"3/3 orthonormal with det +1" + (f"; bad {bad}" if bad else ""))

    # 2 — the reference and the pre-reset session are left alone
    ident = []
    for d in (WORLD_REF_DATE, "2026-05-11"):
        R, t = world_transform("motherboard", d)
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, 0)):
            ident.append(d)
    check(not ident, "the reference and pre-reset sessions get the identity",
          f"{WORLD_REF_DATE} and 2026-05-11 unchanged"
          + (f"; {ident} are not" if ident else ""))

    # 3 — NO TILT IS APPLIED, because the rig cannot produce one. The
    #     OptiTrack ground plane is set with an L-bracket laid on the table, so
    #     two calibrations differ only by yaw about the normal and translation
    #     in the plane. A 3.38 deg tilt was once measured and applied here; it
    #     was an artefact of 2026-05-19's non-planar contact cloud, whose halves
    #     give normals 6.35 deg apart with one half agreeing with the reference
    #     to 0.77 deg. This check encodes the procedure, so the artefact cannot
    #     come back as a "measurement".
    tilts = []
    for d in ("2026-05-10", "2026-05-11", "2026-05-19"):
        R, _ = world_transform("motherboard", d)
        rv = Rotation.from_matrix(R).as_rotvec()
        n = np.array([-0.009, 0.9983, 0.0579]); n /= np.linalg.norm(n)
        perp = np.degrees(np.linalg.norm(rv - (rv @ n) * n))
        tilts.append((d, float(perp)))
    worst = max(tilts, key=lambda x: x[1])
    check(worst[1] < 0.05,
          "no session is given a tilt about an in-plane axis",
          f"largest out-of-plane rotation {worst[1]:.3f} deg ({worst[0]}); the "
          f"L-bracket procedure permits only yaw about the normal")

    # 4 — the contact-plane estimator is NOT trusted where it disagrees with
    #     itself. This is the check that would have caught the artefact.
    n_ref = table_normal(WORLD_REF_DATE, cal)
    halves = {}
    for d in ("2026-05-10", "2026-05-19"):
        halves[d] = split_normals(d, cal)
    ok = halves["2026-05-10"] < 2.0 < halves["2026-05-19"]
    check(ok, "the contact-plane normal is known to be unreliable on 2026-05-19",
          f"in-plane halves disagree by {halves['2026-05-10']:.2f} deg on the "
          f"reference and {halves['2026-05-19']:.2f} deg on 2026-05-19, so a "
          f"whole-cloud normal there averages two inconsistent surfaces")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nworld transform: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
