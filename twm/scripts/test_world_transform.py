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


def main() -> int:
    from react_toolbox.calibration import load_calibration
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

    # 3 — AND IT ACTUALLY ALIGNS. Measured against the table normal.
    n_ref = table_normal(WORLD_REF_DATE, cal)
    n_before = table_normal("2026-05-19", cal)
    R19, _ = world_transform("motherboard", "2026-05-19")
    n_after = table_normal("2026-05-19", cal, R19)
    floor = ang(table_normal("2026-05-11", cal), n_ref)
    check(ang(n_after, n_ref) < max(2 * floor, 0.8) < ang(n_before, n_ref),
          "it aligns 2026-05-19's table normal onto the reference",
          f"{ang(n_before, n_ref):.2f} deg -> {ang(n_after, n_ref):.2f} deg; "
          f"the 2026-05-11 floor is {floor:.2f} deg")

    # 4 — AND IT DOES NOT MOVE THE WORKSPACE. The in-plane translation is
    #     undetermined, so a correction that shifts the workspace is inventing
    #     one. The pivot is chosen to make this true; the check proves it.
    from scipy.spatial.transform import Rotation
    P = []
    for p in sorted((REL / "2026-05-19").glob("*.parquet")):
        t = pq.read_table(p, columns=["sensor_left_pose", "sensor_right_pose",
                                      "force_left_normal_n",
                                      "force_right_normal_n"]).to_pydict()
        for s in ("left", "right"):
            F = np.asarray(t[f"force_{s}_normal_n"], float)
            S = np.asarray([x for x in t[f"sensor_{s}_pose"]], float)
            m = (F > 2.0) & np.isfinite(S).all(1) & (np.linalg.norm(S[:, 3:], axis=1) > .5)
            if not m.any():
                continue
            Rs = Rotation.from_quat(S[m, 3:7]).as_matrix()
            P.append((S[m, :3] * 1000.0 + np.einsum("nij,j->ni", Rs, cal[f"gel_{s}"]))[::5])
    P = np.vstack(P)
    R, t = world_transform("motherboard", "2026-05-19")
    moved = float(np.linalg.norm(np.median((R @ P.T).T + t, axis=0) - np.median(P, axis=0)))
    check(moved < 3.0, "it leaves the workspace centroid where it was",
          f"median contact position moves {moved:.2f} mm "
          f"(the in-plane translation is undetermined, so it must not invent one)")

    # 5 — the residual is published, not hidden
    r = world_residual("motherboard", "2026-05-19")
    check(r.get("yaw_deg") is None and r.get("in_plane_mm") is None
          and r.get("tilt_deg") is not None,
          "what it cannot fix is declared as unknown",
          f"tilt +/-{r['tilt_deg']} deg, height {r['height_mm']} mm measured but "
          f"not applied; yaw and in-plane translation declared None")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nworld transform: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
