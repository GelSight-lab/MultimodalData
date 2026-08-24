"""The exported probe test set is self-contained, self-consistent, and scorable.

`probes.json` used to be the "test set": amplitudes, speeds and an episode
name. Reproducing a probe from it required the raw HDF5, which is not
published, so nothing could actually be evaluated. These checks are the ones
that would have caught that.

  SELF-CONTAINED  everything needed to score a rollout ships in the package —
                  context images, the action, the held hand, the calibration.
  SELF-CONSISTENT the ground-truth pixels recompute from the calibration IN
                  the package. A stored projection that only agrees with the
                  calibration on my disk is a trap.
  SCORABLE        the scorer returns zero on the ground truth and recovers a
                  known injected error. A metric that cannot be shown to move
                  cannot be shown to mean anything.

    python scripts/test_probe_testset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
ROOT = Path("/media/yxma/Disk1/twm/probe_testset")


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import cv2

    from react_toolbox.calibration import load_calibration
    from react_toolbox.probe_eval import project_gt, rollout_error
    from scipy.spatial.transform import Rotation

    man = json.loads((ROOT / "manifest.json").read_text())
    # loaded from the PACKAGE, not from the repo — that is the point
    cal = load_calibration(ROOT)

    runs = [json.loads((ROOT / p["meta"]).read_text()) for p in man["probes"]]
    files = [(r, q) for r in runs for q in r["probes"]]

    # 1 — everything a scorer needs is present
    missing = []
    for r, q in files:
        f = ROOT / q["file"]
        if not f.exists():
            missing.append(q["file"]); continue
        d = np.load(f)
        need = {"poses", "held_pose", "gel_pos_m", "delta_gel_pos_m",
                "delta_gel_rotvec_rad", "delta_rigid_pos_m",
                "delta_rigid_rotvec_rad", "action_scalar", "action_axis",
                "action_sign", "context_poses_moving", "context_poses_held"} | \
               {f"gt_px_{v}" for v in man["views"]}
        if not need <= set(d.files):
            missing.append(f"{q['file']}: {sorted(need - set(d.files))}")
    n_ctx = sum(len(list((ROOT / f"probes/run{r['run']}/context").glob("*.jpg")))
                for r in runs)
    check(not missing and n_ctx == len(runs) * man["context_frames"] * len(man["views"]),
          "every probe ships its action, ground truth and context",
          f"{len(files)} probes, {n_ctx} context images "
          f"({len(runs)} runs x {man['context_frames']} frames x {len(man['views'])} views)"
          + (f"; missing {missing[:2]}" if missing else ""))

    # 2 — the stored ground-truth pixels recompute from the PACKAGED calibration
    worst, n = 0.0, 0
    for r, q in files[:24]:
        d = np.load(ROOT / q["file"])
        gel = cal[f"gel_{r['moving_side']}"]
        for v in man["views"]:
            got = project_gt(d["poses"], gel, cal["cams"][v])
            a, b = got, d[f"gt_px_{v}"]
            m = np.isfinite(a).all(1) & np.isfinite(b).all(1)
            if m.any():
                worst = max(worst, float(np.max(np.linalg.norm(a[m] - b[m], axis=1))))
                n += int(m.sum())
    check(worst < 1e-6, "stored ground-truth pixels recompute from the package",
          f"worst disagreement {worst:.2e} px over {n} projected points")

    # 3 — the deltas reconstruct the absolute poses
    bad = []
    for r, q in files:
        d = np.load(ROOT / q["file"])
        P = d["poses"]
        # BOTH deltas must integrate: the rigid one back to `poses`, the gel
        # one back to `gel_pos_m`. They are different trajectories — that is
        # the point — so checking only one would let the other rot.
        pos = P[0, :3] + np.cumsum(d["delta_rigid_pos_m"], axis=0)
        e = float(np.max(np.linalg.norm(pos - P[1:, :3], axis=1)))
        g = d["gel_pos_m"][0] + np.cumsum(d["delta_gel_pos_m"], axis=0)
        eg = float(np.max(np.linalg.norm(g - d["gel_pos_m"][1:], axis=1)))
        qq = Rotation.from_quat(P[0, 3:7])
        for rv in d["delta_gel_rotvec_rad"]:
            qq = Rotation.from_rotvec(rv) * qq        # world-frame: pre-multiply
        ang = float(np.degrees((qq.inv() * Rotation.from_quat(P[-1, 3:7])).magnitude()))
        if e > 1e-9 or eg > 1e-9 or ang > 1e-6:
            bad.append(f"{q['file']}: rigid {e:.2e} m, gel {eg:.2e} m, {ang:.2e} deg")
    check(not bad, "the published deltas integrate back to the poses",
          f"{len(files)}/{len(files)} exact to 1e-9 m and 1e-6 deg"
          + (f"; {bad[:2]}" if bad else ""))

    # 4 — THE SCORER IS ZERO ON TRUTH AND MOVES BY A KNOWN AMOUNT.
    r, q = files[0]
    d = np.load(ROOT / q["file"])
    gel = cal[f"gel_{r['moving_side']}"]
    z = rollout_error(d["poses"], d["poses"], gel, cal["cams"]["middle"])
    inj = d["poses"].copy(); inj[:, 0] += 0.010            # 10 mm along world x
    e = rollout_error(inj, d["poses"], gel, cal["cams"]["middle"])
    check(z["pos_mm_final"] < 1e-9 and abs(e["pos_mm_final"] - 10.0) < 1e-6,
          "the scorer is zero on truth and recovers an injected 10 mm",
          f"truth {z['pos_mm_final']:.2e} mm; injected 10 mm reads "
          f"{e['pos_mm_final']:.4f} mm and {e['px_final']:.1f} px")

    # 5 — no probe comes from a session whose world frame is unpinned
    bad5 = [p["episode"] for p in man["probes"]
            if p["episode"].split("/")[0] not in man["trusted_sessions"]]
    check(not bad5 and man["excluded_sessions"],
          "start frames come only from sessions with a pinned world frame",
          f"{len(man['probes'])} runs from {sorted({p['episode'].split('/')[0] for p in man['probes']})}; "
          f"excluded {sorted(man['excluded_sessions'])}"
          + (f"; leaked {bad5}" if bad5 else ""))

    # 6 — the overlay runs and puts the marker where the stored truth says
    from react_toolbox.probe_eval import overlay_gt
    r, q = files[0]
    d = np.load(ROOT / q["file"])
    img = cv2.imread(str(ROOT / f"probes/run{r['run']}/context/ctx3_middle.jpg"))[:, :, ::-1]
    vis = overlay_gt(img, d["poses"], cal[f"gel_{r['moving_side']}"],
                     cal["cams"]["middle"], held_pose7=d["held_pose"],
                     held_gel_mm=cal[f"gel_{r['held_side']}"])
    diff = int((np.abs(vis.astype(int) - img.astype(int)).sum(2) > 25).sum())
    start = d["gt_px_middle"][0]
    near = vis[max(0, int(start[1])-4):int(start[1])+5,
               max(0, int(start[0])-4):int(start[0])+5]
    check(vis.shape == img.shape and diff > 200 and near.max() > 240,
          "overlay_gt draws the commanded path on a context frame",
          f"{diff} pixels changed; the start marker is bright at the stored "
          f"ground-truth pixel {np.round(start, 1).tolist()}")

    # 7 — GROUND TRUTH STAYS CLEAR OF THE EDGE. In frame is not enough: a path
    #     ending 15 px from the border cannot be scored, because a rollout that
    #     overshoots even slightly leaves the image entirely. The preview used
    #     an 8 px margin, which is right for looking and wrong for measuring.
    close = []
    for r, q in files:
        d = np.load(ROOT / q["file"])
        p_ = d["gt_px_middle"]
        m = np.isfinite(p_).all(1)
        if not m.any():
            close.append(f"{q['file']}: nothing in view"); continue
        e = float(min(p_[m][:, 0].min(), p_[m][:, 1].min(),
                      (640 - p_[m][:, 0]).min(), (480 - p_[m][:, 1]).min()))
        if e < man["view_margin_px"] - 1:
            close.append(f"{q['file']}: {e:.0f} px")
    check(not close, "ground truth keeps a scoring margin from the edge",
          f"{len(files)}/{len(files)} stay >= {man['view_margin_px']:.0f} px "
          f"inside the middle view"
          + (f"; {close[:2]}" if close else ""))

    # 8 — EACH ACTION MOVES ALONG EXACTLY ONE AXIS. This is the defining
    #     property of the set and nothing checked it. Measured at the GEL:
    #     the pose 7-vec is the marker cluster's and rotations pivot on the
    #     gel 65.7 mm away, so in RIGID-BODY coordinates a "pure rotation"
    #     carries up to 75.7 mm of translation and a model fed that action
    #     reads "translate 76 mm AND rotate 79 deg".
    off = []
    for r, q in files:
        d = np.load(ROOT / q["file"])
        ax = int(d["action_axis"])
        dp, dr = d["delta_gel_pos_m"], d["delta_gel_rotvec_rad"]
        if q["kind"] == "translation":
            cross = float(np.abs(np.delete(dp, ax, axis=1)).max())
            other = float(np.abs(dr).max())
            unit = "m"
        else:
            cross = float(np.abs(np.delete(dr, ax, axis=1)).max())
            other = float(np.abs(dp).max())
            unit = "rad"
        if cross > 1e-12 or other > 1e-9:
            off.append(f"{q['file']}: off-axis {cross:.1e} {unit}, "
                       f"other-kind {other:.1e}")
        # and the 1-D form must reconstruct the full delta
        recon = np.zeros_like(dp)
        recon[:, ax] = d["action_scalar"]
        tgt = dp if q["kind"] == "translation" else dr
        if float(np.abs(recon - tgt).max()) > 1e-15:
            off.append(f"{q['file']}: action_scalar does not reconstruct")
    check(not off, "every action moves along exactly one axis, at the gel",
          f"{len(files)}/{len(files)} have zero off-axis and zero other-kind "
          f"motion, and action_scalar reconstructs the delta exactly"
          + (f"; {off[:2]}" if off else ""))

    # ...and the rigid-body delta is NOT zero for rotations, which is the whole
    # reason the gel frame is the primary one. Asserted so the distinction
    # cannot quietly collapse back.
    rots = [(r, q) for r, q in files if q["kind"] == "rotation"]
    mx = max(float(np.abs(np.load(ROOT / q["file"])["delta_rigid_pos_m"]).sum(0).max())
             for _, q in rots)
    check(mx > 0.005, "the rigid-body action is documented as different",
          f"rotation probes carry up to {mx*1000:.0f} mm of marker-cluster "
          f"translation, which is why delta_gel_* is primary")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    nf = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nprobe test set: {len(RESULTS)} checks, {nf} failing")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
