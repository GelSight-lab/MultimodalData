"""Measure everything the compression-force page claims, into one artifact.

Nothing on that page is typed by hand. Every number here is produced by
re-running the measurement, so a claim that goes stale becomes a build
difference rather than a sentence nobody rechecks.

    python -m force_recovery.press_axis_eval

Writes CACHE/press_axis.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402
from scipy.spatial.transform import Rotation                   # noqa: E402

from react_paths import force_meta, release_root               # noqa: E402
from force_recovery.dexforce import (                          # noqa: E402
    STIFFNESS_N_PER_M, gel_axis)
from force_recovery.site2 import CACHE                         # noqa: E402

TASK = "motherboard"
SIDES = ("left", "right")
SRCS = ("dual_ball", "body_y")
UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP
N_IMG_EPISODES = 5


def _eps(task):
    return [str(Path(p).relative_to(force_meta(task))).replace(".parquet", "")
            for p in sorted(glob.glob(str(force_meta(task) / "*" / "*.parquet")))]


def _load(task, ep, side):
    t = pq.read_table(str(force_meta(task) / f"{ep}.parquet"),
                      columns=[f"sensor_{side}_pose",
                               f"force_{side}_normal_n"]).to_pydict()
    P = np.asarray([x for x in t[f"sensor_{side}_pose"]], float)
    F = np.asarray(t[f"force_{side}_normal_n"], float)
    return P, F


def board_normal_test(out):
    """Angle between R(q)@axis and world down, pressing hard on a level board."""
    res = {}
    for side in SIDES:
        Rs = []
        for ep in _eps(TASK):
            P, F = _load(TASK, ep, side)
            m = (np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)
                 & (F > 6))
            if m.sum() >= 20:
                Rs.append(Rotation.from_quat(P[m, 3:7]).as_matrix())
        R = np.vstack(Rs)
        res[side] = {"n_frames": int(len(R))}
        for src in SRCS:
            a = gel_axis(TASK, side, source=src)
            v = np.einsum("nij,j->ni", R, a / np.linalg.norm(a))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            ang = np.degrees(np.arccos(np.clip(v @ DOWN, -1, 1)))
            res[side][src] = {"median_deg": float(np.median(ang)),
                              "p25_deg": float(np.percentile(ang, 25)),
                              "p75_deg": float(np.percentile(ang, 75))}
    out["board_normal"] = res


def motion_force_test(out):
    """corr(dF, v.n) per episode. No world frame, no table assumption."""
    res = {}
    for side in SIDES:
        per = {s: [] for s in SRCS}
        for ep in _eps(TASK):
            P, F = _load(TASK, ep, side)
            ok = (np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)
                  & np.isfinite(F))
            if ok.sum() < 200:
                continue
            Pk, Fk = P[ok], F[ok]
            R = Rotation.from_quat(Pk[:, 3:7]).as_matrix()
            v = np.gradient(Pk[:, :3], axis=0) * 1000.0
            dF = np.gradient(Fk)
            m = Fk > 0.5
            if m.sum() < 100:
                continue
            for src in SRCS:
                n = np.einsum("nij,j->ni", R, gel_axis(TASK, side, source=src))
                c = np.corrcoef(dF[m], np.sum(v * n, axis=1)[m])[0, 1]
                if np.isfinite(c):
                    per[src].append(float(c))
        a = np.array(per["dual_ball"]); b = np.array(per["body_y"])
        k = min(len(a), len(b))
        res[side] = {
            "n_episodes": int(k),
            "dual_ball": {"median": float(np.median(a))},
            "body_y": {"median": float(np.median(b))},
            "body_y_better_frac": float((b[:k] > a[:k]).mean()),
        }
    out["motion_force"] = res


def conditioning(out):
    """Can the axis be recovered from motion at all? sv ratio of mean(R)."""
    res = {}
    for side in SIDES:
        Rs = []
        for ep in _eps(TASK):
            P, F = _load(TASK, ep, side)
            m = (np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)
                 & (F > 4))
            if m.sum() >= 20:
                Rs.append(Rotation.from_quat(P[m, 3:7]).as_matrix())
        R = np.vstack(Rs)
        sv = np.linalg.svd(R.mean(0), compute_uv=False)
        res[side] = {"sv": [float(x) for x in sv],
                     "sv1_over_sv2": float(sv[0] / sv[1]),
                     "n_frames": int(len(R))}
    out["conditioning"] = res


def image_tilt_test(out):
    """The gel's own deformation: real signal, but does the pose predict it?"""
    res = {}
    eps = _eps(TASK)[:N_IMG_EPISODES]
    ys, xs = np.mgrid[0:480, 0:640]

    def _basis(a):
        a = a / np.linalg.norm(a)
        t = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1., 0])
        e1 = np.cross(a, t); e1 /= np.linalg.norm(e1)
        return e1, np.cross(a, e1)

    for side in SIDES:
        OBS, POSE, coh = [], [], []
        for ep in eps:
            vf = release_root(TASK) / "videos" / ep / f"tactile_{side}.mp4"
            if not vf.exists():
                continue
            P, F = _load(TASK, ep, side)
            cap = cv2.VideoCapture(str(vf))
            buf = []
            for i in np.where(F < 0.2)[0][:400][::20]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if ok:
                    buf.append(fr.astype(np.float32))
            if len(buf) < 5:
                cap.release(); continue
            ref = np.median(np.stack(buf), 0)
            rows = np.where(F > 4)[0]
            prev = None
            for i in rows[::7]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if not ok:
                    continue
                d = np.abs(fr.astype(np.float32) - ref).mean(2)
                m = d > max(8.0, np.percentile(d, 96))
                if m.sum() < 800:
                    continue
                X = np.c_[np.ones(m.sum()), xs[m] - 320, ys[m] - 240]
                g = np.linalg.lstsq(X, d[m], rcond=None)[0][1:]
                nrm = np.linalg.norm(g)
                if nrm < 1e-9:
                    continue
                g = g / nrm
                OBS.append(g); POSE.append(P[i])
                if prev is not None and prev[0] >= i - 21:   # <= 7 rows apart
                    coh.append(float(np.degrees(np.arccos(
                        np.clip(prev[1] @ g, -1, 1)))))
                prev = (i, g)
            cap.release()
        if len(OBS) < 80:
            continue
        OBS = np.asarray(OBS); Pm = np.asarray(POSE)
        R = Rotation.from_quat(Pm[:, 3:7]).as_matrix()
        entry = {"n_frames": int(len(OBS)),
                 # sampled every 7th contact row (~0.23 s apart), NOT adjacent
                 # frames -- calling it "adjacent" overstated the coherence.
                 "sample_gap_rows": 7,
                 "gap_angle_median_deg": float(np.median(coh)) if coh else None}
        for src in SRCS:
            a = gel_axis(TASK, side, source=src); a = a / np.linalg.norm(a)
            e1, e2 = _basis(a)
            gb = np.einsum("nji,j->ni", R, UP)
            tilt = gb - (gb @ (-a))[:, None] * (-a)
            W = np.c_[tilt @ e1, tilt @ e2]
            k = np.linalg.norm(W, axis=1) > 1e-6
            M = np.linalg.lstsq(W[k], OBS[k], rcond=None)[0]
            pred = W[k] @ M
            entry[src] = {"r2": float(
                1 - ((OBS[k] - pred) ** 2).sum()
                / ((OBS[k] - OBS[k].mean(0)) ** 2).sum())}
        res[side] = entry
    out["image_tilt"] = res


def vertical_gap(out):
    """How wrong 'just push along world z' is, per task."""
    res = {}
    for task in sorted(p.name for p in release_root().iterdir()
                       if (p / "episodes.jsonl").exists()):
        # BOTH conventions. Quoting only the active one would argue that
        # world vertical is wrong using the very axis that is in dispute.
        per = {src: [] for src in SRCS}
        for ep in _eps(task):
            P, F = _load(task, ep, "left")
            m = (np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)
                 & (F > 1))
            if m.sum() < 10:
                continue
            R = Rotation.from_quat(P[m, 3:7]).as_matrix()
            for src in SRCS:
                n = np.einsum("nij,j->ni", R,
                              gel_axis(task, "left", source=src))
                n /= np.linalg.norm(n, axis=1, keepdims=True)
                per[src].append(np.degrees(np.arccos(np.clip(n @ DOWN, -1, 1))))
        entry = {}
        for src in SRCS:
            a = np.concatenate(per[src])
            entry[src] = {"median_deg": float(np.median(a)),
                          "p90_deg": float(np.percentile(a, 90)),
                          "frac_over_15deg": float((a > 15).mean())}
            entry["n_frames"] = int(len(a))
        res[task] = entry
    out["vertical_gap"] = res


def axis_values(out):
    out["stiffness_n_per_m"] = float(STIFFNESS_N_PER_M)
    out["axes"] = {
        side: {src: [float(x) for x in gel_axis(TASK, side, source=src)]
               for src in SRCS} for side in SIDES}
    out["axis_separation_deg"] = {
        side: float(np.degrees(np.arccos(abs(np.clip(
            np.dot(gel_axis(TASK, side, source="dual_ball"),
                   gel_axis(TASK, side, source="body_y")), -1, 1)))))
        for side in SIDES}
    # what the calibration file itself reports
    cal = {}
    for side in SIDES:
        j = json.loads((release_root(TASK) / "calibration"
                        / f"T_gel_to_rigid_{side}.json").read_text())
        cal[side] = {"depth_offset_mm": j.get("depth_offset_mm"),
                     "n_poses": len(j.get("raw_data", [])),
                     "axis_max_angle_deg": (j.get("consistency") or {})
                     .get("axis_max_angle_deg")}
    out["calibration_file"] = cal


def target_shift(out):
    """How far switching the axis moves the published target poses."""
    per = {}
    for task in ("motherboard", "pushT"):
        tot = []
        for ep in _eps(task):
            for side in SIDES:
                P, F = _load(task, ep, side)
                m = (np.isfinite(P).all(1)
                     & (np.linalg.norm(P[:, 3:], axis=1) > .5) & (F > 0))
                if m.sum() < 5:
                    continue
                R = Rotation.from_quat(P[m, 3:7]).as_matrix()
                d = (F[m][:, None] / STIFFNESS_N_PER_M) * (
                    np.einsum("nij,j->ni", R, gel_axis(task, side, "body_y"))
                    - np.einsum("nij,j->ni", R,
                                gel_axis(task, side, "dual_ball")))
                tot.append(np.linalg.norm(d, axis=1) * 1000)
        a = np.concatenate(tot)
        per[task] = {"n_rows": int(len(a)),
                     "median_mm": float(np.median(a)),
                     "max_mm": float(a.max())}
    out["target_shift"] = per


def main() -> int:
    out = {}
    axis_values(out)
    conditioning(out)
    board_normal_test(out)
    motion_force_test(out)
    vertical_gap(out)
    target_shift(out)
    image_tilt_test(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "press_axis.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {CACHE / 'press_axis.json'}")
    for k in out:
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
