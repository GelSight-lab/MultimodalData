"""Diagnose what each freeze event actually is — pipeline stall, OT solver
glitch (translation OR rotation teleport), OT dropout, or real held-still.

For each bitwise-frozen pose interval (≥ 0.25 s on an active sensor):

  trans_vel_near : max translation velocity in [a − 1 s, b + 1 s]  (m/s)
  ang_vel_near   : max angular velocity in same window             (rad/s)
  opt_gap_s      : max gap in OT timestamp stream during [a, b]    (s)
  cam_dup_max    : fraction of bitwise-identical consecutive cam frames
                   in the window (max over cam0/1/2)
  tac_dup        : same on the affected GelSight stream
  view_motion    : mean inter-frame abs diff on cam2                (uint8)
  tac_motion     : mean inter-frame abs diff on affected GelSight  (uint8)

Verdict priority (first match wins). All freeze events in this dataset
correspond to OptiTrack track loss (the recorder holds the last pose and
on re-acquire the solver may flip). `tac_dup` is *not* a reliable freeze
indicator — GelSight only changes under gel deformation, so a sensor
moved in free air looks "duplicated" while being perfectly healthy.

  ot_loss        : any pipeline-level signal of mocap failure —
                   opt_gap_s ≥ 0.1, or trans_vel ≥ 5 m/s, or ang_vel ≥ 15
                   rad/s near the event, or the pose is bitwise-held for
                   the full window (= solver lost track, kept emitting).
  pipeline_stall : ALL three camera streams duplicating ≥ 30 % (recorder
                   itself stuck — not observed in current dataset).
  real_still     : no OT-loss signals and cross-modal motion at noise.
  ambiguous      : none of the above.
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

import cv2  # noqa: F401 (used transitively in render_clip)
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from inspect_freezes_v3 import (
    render_clip, load_cam_calibs,
    freeze_intervals, merge_adjacent,
    CALIB_DIR, FPS, EPS,
)
from twm.viz import cam_aligned_pose

H5_ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")
TASKS_JSON = Path("/tmp/tasks_local.json")
OUT_ROOT = Path("/media/yxma/Disk1/twm/figures/dataset_figures/freeze_diagnose")
CSV_OUT = Path("/tmp/freeze_diagnose.csv")

THRESHOLD_S = 0.25
N_MOTION_SAMPLES = 16
N_DUP_SAMPLES = 32
NEAR_WINDOW_S = 1.0
SEED = 42
N_PER_CLASS = 15

# Classification thresholds
TAU_TRANS_MPS = 5.0
TAU_ANG_RAD_S = 15.0          # ~860 °/s — well above fast wrist motion
TAU_OPT_GAP_S = 0.10
TAU_DUP_FRAC = 0.30
TAU_VIEW_STILL = 2.0   # mean abs diff per pixel (uint8). Anything above ≈2 is visible motion;
                       # below 2 is camera noise. Was 4.0 (too lax — let the ep_005/017
                       # OT-offline prefixes slip into real_still even though hands were
                       # visibly moving at ~3.7 mean diff).
TAU_TAC_STILL = 3.0


def active_for(tasks, task, date):
    n = tasks.get("tasks", {}).get(task, {}).get("per_date_notes", {}).get(date, {})
    return tuple(n.get("active_sensors") or ("left", "right"))


def angular_velocity(poses, ts):
    """Angular velocity (rad/s) at each consecutive pair. Returns (T-1,)."""
    q = np.asarray(poses[:, 3:], np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dots = np.abs(np.einsum("ti,ti->t", q[:-1], q[1:]))
    dots = np.clip(dots, 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    dt = np.maximum(np.diff(ts), 1e-9)
    return angles / dt


def translation_velocity(poses, ts):
    """Translation speed (m/s)."""
    d = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    dt = np.maximum(np.diff(ts), 1e-9)
    return d / dt


def max_in_window(values, ts_values, t_lo, t_hi):
    """Max of `values` (defined at midpoints of OT samples) within wall-clock
    window [t_lo, t_hi]. `ts_values` is the per-pair midpoint time array."""
    mask = (ts_values >= t_lo) & (ts_values <= t_hi)
    return float(values[mask].max()) if mask.any() else 0.0


def max_opt_gap(ts, t_lo, t_hi):
    """Largest inter-sample gap (s) in `ts` intersecting [t_lo, t_hi].

    Includes both internal gaps (between samples in the window) and edge gaps
    (no sample at the start or end of the window). When the OT stream is
    silent for the whole window, returns (t_hi − t_lo) — earlier versions
    returned 0 here, which silently misclassified the OT-uninit prefixes as
    `real_still`.
    """
    if len(ts) == 0:
        return float(t_hi - t_lo)
    in_win = (ts >= t_lo) & (ts <= t_hi)
    if not in_win.any():
        return float(t_hi - t_lo)
    win_ts = ts[in_win]
    candidates = []
    if len(win_ts) >= 2:
        candidates.append(float(np.diff(win_ts).max()))
    # Edge gaps — silence at start or end of window
    candidates.append(float(win_ts.min() - t_lo))
    candidates.append(float(t_hi - win_ts.max()))
    return max(candidates)


def motion_stat(h5, key, a, b, n_samples):
    ds = h5[key]
    if b - a < 1:
        return 0.0, 0.0
    if b - a <= n_samples:
        pairs = list(range(a, b))
    else:
        pairs = np.linspace(a, b - 1, n_samples).astype(int).tolist()
    diffs = []
    dup_count = 0
    for t in pairs:
        x0 = ds[t]
        x1 = ds[t + 1]
        if np.array_equal(x0, x1):
            dup_count += 1
            diffs.append(0.0)
        else:
            diffs.append(float(np.abs(x1.astype(np.int16) - x0.astype(np.int16)).mean()))
    return float(np.mean(diffs)), dup_count / len(pairs)


def diagnose_event(f, a, b, side, cam_ts, sl_ts, sl_poses, sr_ts, sr_poses):
    """Compute every diagnostic stat for one freeze event."""
    # OT-stream stats: window is the wall-clock cam time of [a-30, b+30]
    t_lo = float(cam_ts[max(0, a - int(round(NEAR_WINDOW_S * FPS)))])
    t_hi = float(cam_ts[min(len(cam_ts) - 1, b + int(round(NEAR_WINDOW_S * FPS)))])

    opt_ts = sl_ts if side == "left" else sr_ts
    opt_poses = sl_poses if side == "left" else sr_poses
    trans_vel = translation_velocity(opt_poses, opt_ts)
    ang_vel = angular_velocity(opt_poses, opt_ts)
    pair_ts = 0.5 * (opt_ts[:-1] + opt_ts[1:])
    trans_max = max_in_window(trans_vel, pair_ts, t_lo, t_hi)
    ang_max = max_in_window(ang_vel, pair_ts, t_lo, t_hi)
    opt_gap = max_opt_gap(opt_ts, t_lo, t_hi)

    # Frame stats: cam0/1/2 + affected GelSight
    cam_dups = []
    cam_motions = []
    for c in (0, 1, 2):
        m, d = motion_stat(f, f"realsense/cam{c}/color", a, b, N_DUP_SAMPLES)
        cam_motions.append(m)
        cam_dups.append(d)
    tac_motion, tac_dup = motion_stat(f, f"gelsight/{side}/frames", a, b, N_DUP_SAMPLES)

    return {
        "view_motion": round(cam_motions[2], 3),
        "cam0_motion": round(cam_motions[0], 3),
        "cam1_motion": round(cam_motions[1], 3),
        "tac_motion":  round(tac_motion, 3),
        "cam0_dup":    round(cam_dups[0], 3),
        "cam1_dup":    round(cam_dups[1], 3),
        "cam2_dup":    round(cam_dups[2], 3),
        "tac_dup":     round(tac_dup, 3),
        "trans_vel_near_mps":   round(trans_max, 3),
        "ang_vel_near_rad_s":   round(ang_max, 3),
        "opt_gap_s":            round(opt_gap, 4),
    }


def verdict_for(e):
    # A genuine recorder stall would show duplication on ALL three cam
    # streams. tac_dup alone is uninformative (gel only changes under
    # contact deformation), so it's excluded from the stall test.
    if min(e["cam0_dup"], e["cam1_dup"], e["cam2_dup"]) >= TAU_DUP_FRAC:
        return "pipeline_stall"
    # Anything else with a bitwise-frozen pose is an OT track loss —
    # confirmed by any of: OT timestamp gap, large translation velocity
    # near the event, or large angular velocity near the event.
    if (e["opt_gap_s"] >= TAU_OPT_GAP_S
        or e["trans_vel_near_mps"] >= TAU_TRANS_MPS
        or e["ang_vel_near_rad_s"] >= TAU_ANG_RAD_S):
        return "ot_loss"
    # No OT-loss signal and nothing is moving — operator really paused.
    if (e["view_motion"] < TAU_VIEW_STILL and e["tac_motion"] < TAU_TAC_STILL and
        e["cam0_motion"] < TAU_VIEW_STILL and e["cam1_motion"] < TAU_VIEW_STILL):
        return "real_still"
    # Pose is bitwise-frozen with no other signal and not obviously
    # static either — treat as OT-loss-without-extra-evidence by default.
    return "ot_loss"


def main():
    tasks = json.loads(TASKS_JSON.read_text())
    cam_calibs = load_cam_calibs()
    gel_L = np.array(json.loads((CALIB_DIR / "T_gel_to_rigid_left.json").read_text())["gel_center_in_rigid_mm"], np.float64)
    gel_R = np.array(json.loads((CALIB_DIR / "T_gel_to_rigid_right.json").read_text())["gel_center_in_rigid_mm"], np.float64)
    min_frames = int(round(THRESHOLD_S * FPS))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    classes = ["ot_loss", "pipeline_stall", "real_still", "ambiguous"]
    for sub in classes:
        d = OUT_ROOT / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.gif"):
            old.unlink()

    # ── Phase 1: enumerate every event with full diagnostics ────────────────
    # Trim offsets per episode — freezes that fall entirely within the trimmed
    # prefix are dropped (they're not in the published .pt anymore).
    TRIM_OFFSETS = {
        "2026-05-11/episode_005": 2429,
        "2026-05-11/episode_012": 9719,
        "2026-05-11/episode_017": 19228,
    }

    events = []
    n_dropped_prefix = 0
    for date_dir in sorted(H5_ROOT.iterdir()):
        date = date_dir.name
        if date == "2026-03-23" or not date_dir.is_dir():
            continue
        for h5_path in sorted(date_dir.glob("episode_*.h5")):
            ep_stem = h5_path.stem
            ep_key = f"{date}/{ep_stem}"
            trim_off = TRIM_OFFSETS.get(ep_key, 0)
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
                    for a, b in merge_adjacent(freeze_intervals(pose, min_frames)):
                        # Skip events that fall entirely in the trimmed prefix:
                        # they're not part of the published dataset anymore.
                        if b < trim_off:
                            n_dropped_prefix += 1
                            continue
                        diag = diagnose_event(
                            f, a, b, side, cam_ts, sl_ts, sl_poses, sr_ts, sr_poses
                        )
                        # Report in TRIMMED-pt coordinates so they match bad_frames.json
                        a_pt = max(0, a - trim_off)
                        b_pt = b - trim_off
                        ev = {
                            "episode": ep_key,
                            "side": side, "a": a_pt, "b": b_pt,
                            "_h5_a": a, "_h5_b": b,         # for render_clip (private)
                            "length_s": round((b_pt - a_pt + 1) / FPS, 3),
                            **diag,
                        }
                        ev["verdict"] = verdict_for(ev)
                        events.append(ev)
            del pose_L, pose_R
            gc.collect()
    print(f"  dropped {n_dropped_prefix} events entirely inside the trimmed prefix")

    # ── Phase 2: dump CSV / summary, print distribution ─────────────────────
    if not events:
        print("No events found.")
        return
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)
    print(f"Wrote {len(events)} events to {CSV_OUT}\n")

    by = {v: [e for e in events if e["verdict"] == v] for v in classes}
    print("Verdict distribution:")
    for v in classes:
        print(f"  {v:18s}: {len(by[v])}")
    print(f"\nThresholds: trans={TAU_TRANS_MPS} m/s  ang={TAU_ANG_RAD_S} rad/s  "
          f"opt_gap={TAU_OPT_GAP_S}s  dup={TAU_DUP_FRAC}  "
          f"view_still<{TAU_VIEW_STILL}  tac_still<{TAU_TAC_STILL}")
    print("\nPer-verdict signal summary:")
    for v in classes:
        if not by[v]:
            continue
        print(f"  [{v}]  n={len(by[v])}")
        for stat in ("trans_vel_near_mps", "ang_vel_near_rad_s", "opt_gap_s",
                     "cam0_dup", "cam2_dup", "tac_dup", "view_motion", "tac_motion"):
            vals = [e[stat] for e in by[v]]
            print(f"     {stat:22s}  min={min(vals):.3f}  med={sorted(vals)[len(vals)//2]:.3f}  max={max(vals):.3f}")

    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "threshold_s": THRESHOLD_S,
        "thresholds": {
            "trans_mps": TAU_TRANS_MPS, "ang_rad_s": TAU_ANG_RAD_S,
            "opt_gap_s": TAU_OPT_GAP_S, "dup_frac": TAU_DUP_FRAC,
            "view_still": TAU_VIEW_STILL, "tac_still": TAU_TAC_STILL,
        },
        "counts": {v: len(by[v]) for v in classes},
        "events": events,
    }, indent=2))

    # ── Phase 3: sample up to N_PER_CLASS, render 5s viewer-layout clips ───
    rng = random.Random(SEED)
    import inspect_freezes_v3 as _v3
    saved_out = _v3.OUT
    try:
        for v in classes:
            pool = by[v]
            if not pool:
                print(f"\n[{v}] no events")
                continue
            n = min(N_PER_CLASS, len(pool))
            picks = rng.sample(pool, n)
            picks.sort(key=lambda e: (e["view_motion"] + e["tac_motion"]))
            print(f"\n[{v}] rendering {n}/{len(pool)} clips:")
            for ci, e in enumerate(picks):
                date, ep_stem = e["episode"].split("/")
                h5_path = H5_ROOT / date / f"{ep_stem}.h5"
                with h5py.File(h5_path, "r") as f:
                    cam_ts = f["timestamps"][:]
                    sl_ts = f["optitrack/sensor_left/timestamps"][:]
                    sl_poses = f["optitrack/sensor_left/pose"][:]
                    sr_ts = f["optitrack/sensor_right/timestamps"][:]
                    sr_poses = f["optitrack/sensor_right/pose"][:]
                pose_L = cam_aligned_pose(cam_ts, sl_ts, sl_poses)
                pose_R = cam_aligned_pose(cam_ts, sr_ts, sr_poses)
                T_total = pose_L.shape[0]
                pad = int(round(FPS))
                MAX_CLIP_FR = int(round(5 * FPS))
                # render_clip indexes the H5 directly — use H5 coords here.
                h5_a, h5_b = e.get("_h5_a", e["a"]), e.get("_h5_b", e["b"])
                length = h5_b - h5_a + 1
                if length + 2 * pad <= MAX_CLIP_FR:
                    ca = max(0, h5_a - pad)
                    cb = min(T_total - 1, h5_b + pad)
                else:
                    ca = max(0, h5_a - pad)
                    cb = min(T_total - 1, ca + MAX_CLIP_FR - 1)
                _v3.OUT = OUT_ROOT / v
                label = (f"{v}_v{e['view_motion']:.1f}_t{e['tac_motion']:.1f}"
                         f"_av{e['ang_vel_near_rad_s']:.1f}"
                         f"_d{max(e['cam0_dup'], e['cam2_dup'], e['tac_dup']):.2f}")
                p, sz = render_clip(
                    date, ep_stem, e["side"], h5_a, h5_b, ca, cb, label,
                    ci, n, pose_L, pose_R, h5_path, cam_calibs, gel_L, gel_R,
                )
                print(f"  {p.name}  ({sz/1024:.0f} KB)")
                del pose_L, pose_R
                gc.collect()
    finally:
        _v3.OUT = saved_out

    print(f"\nDone. Clips in {OUT_ROOT}/{{{','.join(classes)}}}/")


if __name__ == "__main__":
    main()
