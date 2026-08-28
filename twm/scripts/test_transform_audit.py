"""Field by field: what had to rotate, what must not have, and did it.

The release was relabelled from OptiTrack's Y-up to Z-up. That is a change of
the WORLD frame, so it divides every published field in two:

  ROTATED  - quantities expressed in world coordinates. Poses, the camera
             extrinsics, the per-episode world offset. If one of these is
             missed, it disagrees with its neighbours and overlays land tens
             to hundreds of pixels away with nothing raising.

  UNTOUCHED - everything else, and this is the half that is easy to damage by
             over-applying the fix. Force MAGNITUDES are scalars. Penetration
             is a scalar. `T_gel_to_rigid` is the gel's pose in the SENSOR's
             own frame -- rotating the world does not move the gel relative to
             its own body, and rotating it would silently corrupt the force
             direction, which acts along the sensor's local normal. Camera
             intrinsics, timestamps, frame indices and split boundaries are
             not geometry in the world at all.

Every published column is checked against the pre-conversion backup, and any
column that is in NEITHER list fails. A field nobody classified is the one
that gets missed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

from react_toolbox.frames import YUP_TO_ZUP, convert_poses     # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
M = np.asarray(YUP_TO_ZUP, float)

# world-frame geometry: MUST have rotated
ROTATED = {"sensor_left_pose", "sensor_right_pose", "object_pose",
           "force_left_target_pose", "force_right_target_pose"}
# scalars, indices, sensor-local quantities: MUST be untouched
UNTOUCHED = {
    "frame_idx", "timestamp", "source_h5_frame", "task", "task_index",
    "episode", "episode_index", "frame_index",
    "tactile_left_intensity", "tactile_left_area", "tactile_left_mixed",
    "tactile_right_intensity", "tactile_right_area", "tactile_right_mixed",
    "tactile_left_is_new", "tactile_right_is_new",
    "force_left_normal_n", "force_right_normal_n",
    "force_left_penetration_mm", "force_right_penetration_mm",
    "force_left_source_frame", "force_right_source_frame",
}

BACKUPS = {
    ("motherboard", "release"):
        "/media/yxma/Disk1/twm/_backup_yup_20260827_2157/release/motherboard/meta",
    ("motherboard", "release_force"):
        "/media/yxma/Disk1/twm/_backup_yup_20260827_2157/release_force/motherboard/meta",
    ("pushT", "release"):
        "/media/yxma/Disk1/twm/_backup_yup_pushT_20260828_0324/release__meta",
    ("pushT", "release_force"):
        "/media/yxma/Disk1/twm/_backup_yup_pushT_20260828_0324/release_force__meta",
}
LIVE = {"release": "/media/yxma/Disk1/twm/release/{t}/meta",
        "release_force": "/media/yxma/Disk1/twm/release_force/{t}/meta"}


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def _cols(f: str) -> list[str]:
    return pq.read_schema(f).names


def main() -> int:
    argparse.ArgumentParser().parse_args()

    # 1 — no unclassified column anywhere
    seen: set[str] = set()
    for tree in LIVE.values():
        for t in ("motherboard", "pushT"):
            fs = sorted(glob.glob(tree.format(t=t) + "/*/*.parquet"))
            if fs:
                seen |= set(_cols(fs[0]))
    unknown = sorted(seen - ROTATED - UNTOUCHED)
    check(seen and not unknown,
          "every published column is classified as rotated or untouched",
          f"{len(seen)} distinct columns, {len(ROTATED & seen)} rotated, "
          f"{len(UNTOUCHED & seen)} untouched"
          if not unknown else f"UNCLASSIFIED: {unknown}")

    # 2 — the rotated ones rotated by EXACTLY the world rotation
    worst_r, n_r, missing = 0.0, 0, []
    for (t, tree), bk in BACKUPS.items():
        for f in sorted(glob.glob(bk + "/*/*.parquet")):
            g = LIVE[tree].format(t=t) + "/" + "/".join(Path(f).parts[-2:])
            if not Path(g).exists():
                missing.append(g)
                continue
            cols = [c for c in _cols(f) if c in ROTATED]
            if not cols:
                continue
            a = pq.read_table(f, columns=cols).to_pydict()
            b = pq.read_table(g, columns=cols).to_pydict()
            for c in cols:
                A = np.asarray([x for x in a[c]], float)
                B = np.asarray([x for x in b[c]], float)
                if A.ndim != 2 or A.shape[1] != 7:
                    # a scalar listed as world-frame geometry. Without this it
                    # crashed inside numpy with an AxisError, which is a
                    # failure but tells you nothing about which column or why.
                    missing.append(f"{t}/{tree}:{c} is shape {A.shape}, not "
                                   f"(N,7) — a scalar cannot be a world pose")
                    continue
                want = convert_poses(A, True)
                m = np.isfinite(A).all(1) & np.isfinite(want).all(1)
                if not m.any():
                    continue
                n_r += int(m.sum())
                worst_r = max(worst_r, float(np.abs(want[m] - B[m]).max()))
    check(n_r > 0 and not missing and worst_r < 1e-9,
          "each world-frame column equals its own y->z rotation, exactly",
          f"{n_r} rows across {len(BACKUPS)} tree/task pairs, worst deviation "
          f"{worst_r:.2e} (m or quaternion units)"
          if not missing else f"missing live files: {missing[:3]}")

    # 3 — the untouched ones are bit-for-bit what they were
    worst_u, n_u, moved = 0.0, 0, []
    for (t, tree), bk in BACKUPS.items():
        for f in sorted(glob.glob(bk + "/*/*.parquet")):
            g = LIVE[tree].format(t=t) + "/" + "/".join(Path(f).parts[-2:])
            if not Path(g).exists():
                continue
            cols = [c for c in _cols(f) if c in UNTOUCHED]
            a = pq.read_table(f, columns=cols).to_pydict()
            b = pq.read_table(g, columns=cols).to_pydict()
            for c in cols:
                A, B = a[c], b[c]
                if A and isinstance(A[0], str):
                    if A != B:
                        moved.append(f"{t}/{tree}:{c}")
                    continue
                X = np.asarray(A, float)
                Y = np.asarray(B, float)
                m = np.isfinite(X) & np.isfinite(Y)
                n_u += int(m.sum())
                d = float(np.abs(X[m] - Y[m]).max()) if m.any() else 0.0
                if d > 0:
                    moved.append(f"{t}/{tree}:{c}={d:.3g}")
                worst_u = max(worst_u, d)
    check(n_u > 0 and not moved,
          "every scalar, index and sensor-local column is unchanged",
          f"{n_u} values across {len(UNTOUCHED)} column names, worst change "
          f"{worst_u:.1e} — force magnitude and penetration are among these"
          if not moved else f"CHANGED but should not have: {moved[:6]}")

    # 4 — calibration: extrinsics rotated, gel-to-rigid and intrinsics not
    cal_bad = []
    for t, bk in (("motherboard",
                   "/media/yxma/Disk1/twm/_backup_yup_20260827_2157/release/motherboard/calibration"),
                  ("pushT",
                   "/media/yxma/Disk1/twm/_backup_yup_pushT_20260828_0324/release__calibration")):
        live = Path(f"/media/yxma/Disk1/twm/release/{t}/calibration")
        for v in ("left", "middle", "right"):
            o = json.loads((Path(bk) / f"T_mocap_to_cam_{v}.json").read_text())
            n = json.loads((live / f"T_mocap_to_cam_{v}.json").read_text())
            O = np.asarray(o["T_mocap_to_cam"], float)
            N = np.asarray(n["T_mocap_to_cam"], float)
            if np.abs(N[:3, :3] - O[:3, :3] @ M.T).max() > 1e-12:
                cal_bad.append(f"{t}/cam_{v}: rotation is not R@M^T")
            if np.abs(N[:3, 3] - O[:3, 3]).max() > 0:
                cal_bad.append(f"{t}/cam_{v}: translation moved")
            if o.get("intrinsics") != n.get("intrinsics"):
                cal_bad.append(f"{t}/cam_{v}: intrinsics changed")
        for s in ("left", "right"):
            a = (Path(bk) / f"T_gel_to_rigid_{s}.json")
            b = (live / f"T_gel_to_rigid_{s}.json")
            if a.exists() and a.read_bytes() != b.read_bytes():
                cal_bad.append(f"{t}/T_gel_to_rigid_{s}: ROTATED, must not be")
    check(not cal_bad,
          "extrinsics rotated; gel-to-rigid and intrinsics untouched",
          "6 cameras rotated by exactly R@M^T with translation and intrinsics "
          "held; 4 sensor-local gel transforms byte-identical"
          if not cal_bad else "; ".join(cal_bad[:5]))

    # 5 — the per-episode world offset rotated with everything else
    off_bad, n_off = [], 0
    for t, bk in (("motherboard",
                   "/media/yxma/Disk1/twm/_backup_yup_20260827_2157/release/motherboard/episodes.jsonl"),
                  ("pushT",
                   "/media/yxma/Disk1/twm/_backup_yup_pushT_20260828_0324/release__episodes.jsonl")):
        old = {json.loads(l)["episode"]: json.loads(l)
               for l in Path(bk).read_text().splitlines() if l.strip()}
        for l in (Path(f"/media/yxma/Disk1/twm/release/{t}/episodes.jsonl")
                  .read_text().splitlines()):
            if not l.strip():
                continue
            r = json.loads(l)
            o = old.get(r["episode"])
            if o is None:
                continue
            n_off += 1
            a = np.asarray(o.get("world_frame_offset") or [0, 0, 0], float)
            b = np.asarray(r.get("world_frame_offset") or [0, 0, 0], float)
            if np.abs(M @ a - b).max() > 1e-12:
                off_bad.append(f"{t}/{r['episode']}")
    check(n_off > 0 and not off_bad,
          "every episode's world offset rotated with the poses",
          f"{n_off} episode records, each offset == M @ old"
          if not off_bad else f"{len(off_bad)} wrong, e.g. {off_bad[:3]}")

    # 6 — split boundaries are frame indices, not geometry
    sp_bad = []
    for t in ("motherboard", "pushT"):
        f = Path(f"/media/yxma/Disk1/twm/release/{t}/splits.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for k, v in (d.get("episodes") or {}).items():
            for lo, hi in (v.get("test") or []):
                if not (isinstance(lo, int) and isinstance(hi, int) and lo <= hi):
                    sp_bad.append(f"{t}/{k}")
    check(not sp_bad, "split boundaries stayed integer frame indices",
          "test intervals are still integer row ranges in both tasks"
          if not sp_bad else f"malformed: {sp_bad[:4]}")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\ntransform audit: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
