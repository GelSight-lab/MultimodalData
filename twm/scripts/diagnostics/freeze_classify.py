"""Lower the freeze threshold to 0.25 s and auto-classify each event using
cross-modal motion as ground truth.

Idea: if the sensor pose is bitwise-frozen but the cam2 image and / or the
GelSight tactile stream are actively changing, the sensor must have been
moving — OptiTrack lost track. If both video sources are also static, the
operator really did pause. The latter doesn't need to be cut.

Output:
- /tmp/freeze_events_0.25s.csv          one row per event, sortable
- .../freeze_classify/summary.json      same data + classification verdict
- .../freeze_classify/held_still/*.gif  15 randomly-sampled "real still" clips
- .../freeze_classify/ot_lost/*.gif     15 randomly-sampled "real OT-loss" clips
- .../freeze_classify/ambiguous/*.gif   up to 15 sampled borderline events
"""
import csv
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")
# Make sibling scripts importable (e.g. `from inspect_freezes_v3 import ...`)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image

# Reuse the viewer-style renderer
from inspect_freezes_v3 import (
    render_clip, load_cam_calibs,
    freeze_intervals, merge_adjacent,
    CALIB_DIR, FPS, EPS,
)
from twm.viz import cam_aligned_pose, CAM_CALIB_NAME

H5_ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")
TASKS_JSON = Path("/tmp/tasks_local.json")
OUT_ROOT = Path("/media/yxma/Disk1/twm/figures/dataset_figures/freeze_classify")
CSV_OUT = Path("/tmp/freeze_events_0.25s.csv")

THRESHOLD_S = 0.25
N_MOTION_SAMPLES = 16
SEED = 42

# Classification thresholds, mean abs uint8 diff (per channel, per pixel).
# These are conservative: a real still scene typically sits at ~1–3 (camera
# noise floor), so anything >5 already represents visible motion.
TAU_VIEW_STILL = 4.0       # cam2_motion < this → still
TAU_VIEW_MOVE  = 6.0       # cam2_motion > this → moving
TAU_TAC_STILL  = 3.0
TAU_TAC_MOVE   = 5.0

N_SAMPLES_PER_CLASS = 15


def active_for(tasks, task, date):
    n = tasks.get("tasks", {}).get(task, {}).get("per_date_notes", {}).get(date, {})
    return tuple(n.get("active_sensors") or ("left", "right"))


def motion_in_interval(h5, dataset_key, a, b, n_samples):
    """Mean inter-frame absolute diff across n_samples pairs in [a, b-1]."""
    ds = h5[dataset_key]
    if b - a < 1:
        return 0.0
    if b - a <= n_samples:
        pairs = list(range(a, b))
    else:
        pairs = np.linspace(a, b - 1, n_samples).astype(int).tolist()
    diffs = []
    for t in pairs:
        x0 = ds[t].astype(np.int16)
        x1 = ds[t + 1].astype(np.int16)
        diffs.append(float(np.abs(x1 - x0).mean()))
    return float(np.mean(diffs))


def classify(view_mot, tac_mot):
    if view_mot < TAU_VIEW_STILL and tac_mot < TAU_TAC_STILL:
        return "held_still"
    if view_mot > TAU_VIEW_MOVE or tac_mot > TAU_TAC_MOVE:
        return "ot_lost"
    return "ambiguous"


def main():
    tasks = json.loads(TASKS_JSON.read_text())
    cam_calibs = load_cam_calibs()
    gel_L = np.array(
        json.loads((CALIB_DIR / "T_gel_to_rigid_left.json").read_text())["gel_center_in_rigid_mm"],
        np.float64,
    )
    gel_R = np.array(
        json.loads((CALIB_DIR / "T_gel_to_rigid_right.json").read_text())["gel_center_in_rigid_mm"],
        np.float64,
    )
    min_frames = int(round(THRESHOLD_S * FPS))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("held_still", "ot_lost", "ambiguous"):
        d = OUT_ROOT / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.gif"):
            old.unlink()

    # ── Phase 1: enumerate + classify every event ───────────────────────────
    events = []
    for date_dir in sorted(H5_ROOT.iterdir()):
        date = date_dir.name
        if date == "2026-03-23" or not date_dir.is_dir():
            continue
        for h5_path in sorted(date_dir.glob("episode_*.h5")):
            ep_stem = h5_path.stem
            with h5py.File(h5_path, "r") as f:
                cam_ts = f["timestamps"][:]
                sl_ts = f["optitrack/sensor_left/timestamps"][:]
                sl_poses = f["optitrack/sensor_left/pose"][:]
                sr_ts = f["optitrack/sensor_right/timestamps"][:]
                sr_poses = f["optitrack/sensor_right/pose"][:]
                pose_L = cam_aligned_pose(cam_ts, sl_ts, sl_poses)
                pose_R = cam_aligned_pose(cam_ts, sr_ts, sr_poses)
                active = active_for(tasks, "motherboard", date)
                for side in active:
                    pose = pose_L if side == "left" else pose_R
                    ivs = merge_adjacent(freeze_intervals(pose, min_frames))
                    for a, b in ivs:
                        view_mot = motion_in_interval(
                            f, "realsense/cam2/color", a, b, N_MOTION_SAMPLES
                        )
                        tac_mot = motion_in_interval(
                            f, f"gelsight/{side}/frames", a, b, N_MOTION_SAMPLES
                        )
                        verdict = classify(view_mot, tac_mot)
                        events.append({
                            "episode": f"{date}/{ep_stem}",
                            "side": side,
                            "a": a, "b": b,
                            "length_s": round((b - a + 1) / FPS, 3),
                            "view_motion": round(view_mot, 3),
                            "tac_motion": round(tac_mot, 3),
                            "verdict": verdict,
                        })
            del pose_L, pose_R
            gc.collect()

    # ── Phase 2: dump CSV + summary ─────────────────────────────────────────
    by_verdict = {"held_still": [], "ot_lost": [], "ambiguous": []}
    for e in events:
        by_verdict[e["verdict"]].append(e)

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else
                                    ["episode", "side", "a", "b", "length_s",
                                     "view_motion", "tac_motion", "verdict"])
        w.writeheader()
        w.writerows(events)
    print(f"Wrote {len(events)} events to {CSV_OUT}")
    print(f"  held_still : {len(by_verdict['held_still'])}")
    print(f"  ot_lost    : {len(by_verdict['ot_lost'])}")
    print(f"  ambiguous  : {len(by_verdict['ambiguous'])}")
    print(f"\nThresholds: view_still<{TAU_VIEW_STILL}, view_move>{TAU_VIEW_MOVE}, "
          f"tac_still<{TAU_TAC_STILL}, tac_move>{TAU_TAC_MOVE}")
    print(f"\nMotion-stat distribution (view, tactile):")
    for verdict in ("held_still", "ot_lost", "ambiguous"):
        if by_verdict[verdict]:
            vs = [e["view_motion"] for e in by_verdict[verdict]]
            ts = [e["tac_motion"] for e in by_verdict[verdict]]
            print(f"  {verdict:11s}  view={min(vs):.2f}-{max(vs):.2f} (med {sorted(vs)[len(vs)//2]:.2f})  "
                  f"tac={min(ts):.2f}-{max(ts):.2f} (med {sorted(ts)[len(ts)//2]:.2f})")

    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "threshold_s": THRESHOLD_S,
        "tau_view_still": TAU_VIEW_STILL, "tau_view_move": TAU_VIEW_MOVE,
        "tau_tac_still":  TAU_TAC_STILL,  "tau_tac_move":  TAU_TAC_MOVE,
        "n_motion_samples_per_event": N_MOTION_SAMPLES,
        "counts": {k: len(v) for k, v in by_verdict.items()},
        "events": events,
    }, indent=2))

    # ── Phase 3: sample 15 from each class, render clips ────────────────────
    rng = random.Random(SEED)
    for verdict in ("held_still", "ot_lost", "ambiguous"):
        pool = by_verdict[verdict]
        if not pool:
            print(f"\n[{verdict}] no events — skipping render")
            continue
        n = min(N_SAMPLES_PER_CLASS, len(pool))
        picks = rng.sample(pool, n)
        # Sort picks by motion stat for easier inspection (lowest first within each)
        picks.sort(key=lambda e: (e["view_motion"] + e["tac_motion"]))
        print(f"\n[{verdict}] rendering {n}/{len(pool)} sampled clips:")
        for ci, e in enumerate(picks):
            date, ep_stem = e["episode"].split("/")
            h5_path = H5_ROOT / date / f"{ep_stem}.h5"
            # Reload poses for render_clip
            with h5py.File(h5_path, "r") as f:
                cam_ts = f["timestamps"][:]
                sl_ts = f["optitrack/sensor_left/timestamps"][:]
                sl_poses = f["optitrack/sensor_left/pose"][:]
                sr_ts = f["optitrack/sensor_right/timestamps"][:]
                sr_poses = f["optitrack/sensor_right/pose"][:]
            pose_L = cam_aligned_pose(cam_ts, sl_ts, sl_poses)
            pose_R = cam_aligned_pose(cam_ts, sr_ts, sr_poses)
            T_total = pose_L.shape[0]
            # Cap each verification clip at 5 s (150 frames @ 30 fps). For
            # short freezes that means [freeze − 1 s, freeze + 1 s]; for long
            # freezes (> 3 s of freeze) we just show the onset region so the
            # 1 s pre-freeze tracking is visible and the file stays small.
            pad = int(round(FPS))
            MAX_CLIP_FR = int(round(5 * FPS))
            length = e["b"] - e["a"] + 1
            if length + 2 * pad <= MAX_CLIP_FR:
                ca = max(0, e["a"] - pad)
                cb = min(T_total - 1, e["b"] + pad)
            else:
                ca = max(0, e["a"] - pad)
                cb = min(T_total - 1, ca + MAX_CLIP_FR - 1)
            # Temporarily override the OUT dir of render_clip via monkey-patching the module global
            import inspect_freezes_v3 as _v3
            saved_out = _v3.OUT
            _v3.OUT = OUT_ROOT / verdict
            try:
                p, sz = render_clip(
                    date, ep_stem, e["side"], e["a"], e["b"], ca, cb,
                    f"{verdict}_v{e['view_motion']:.1f}_t{e['tac_motion']:.1f}",
                    ci, n,
                    pose_L, pose_R, h5_path, cam_calibs, gel_L, gel_R,
                )
                print(f"  {p.name}  ({sz / 1024:.0f} KB)")
            finally:
                _v3.OUT = saved_out
            del pose_L, pose_R
            gc.collect()

    print(f"\nDone. Clips in {OUT_ROOT}/{{held_still,ot_lost,ambiguous}}/")


if __name__ == "__main__":
    main()
