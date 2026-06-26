"""Real-time freeze inspection clips, full recording-viewer layout.

All visualization primitives come from `twm.viz` (single source). This script
just sits on top, computing freeze schedules and feeding each clip range
through the panel builder + projection overlay (with optional FROZEN ring on
the affected sensor for in-freeze frames).

Output: native 1280×480 GIFs at 30 fps (real time), ≤4 MB after PIL palette
quantization.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from twm.data_collection import make_preview   # = twm.viz.build_preview_panel
from twm.viz import (
    CAM_CALIB_NAME, DISPLAY_ORDER,
    cam_aligned_pose,
    draw_projection_overlay,
    load_optitrack, optitrack_at,
    save_gif,
)

PT_ROOT = Path("/media/yxma/Disk1/twm/processed/mode1_v1/motherboard")
H5_ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")
TASKS_JSON = Path("/tmp/tasks_local.json")
CALIB_DIR = Path("/home/yxma/MultimodalData/twm/calibration/result")
OUT = Path("/media/yxma/Disk1/twm/figures/dataset_figures/freeze_check")

FPS = 30.0
EPS = 1e-7
MIN_S = 1.0
PAD_S = 1.0
CLIP_S = 5.0
LONG_THRESHOLD_S = 15.0
MAX_MID_CLIPS = 6
MAX_KB = 8192   # native 1280x480 with 30 fps; ~6-8 MB at 256-color palette

PAD_FR = int(round(PAD_S * FPS))
CLIP_FR = int(round(CLIP_S * FPS))


# ──────────────────────────────────────────────────────────────────────────────
# Freeze interval detection (kept here — analysis logic, not visualization)
# ──────────────────────────────────────────────────────────────────────────────

def active_for(tasks, task, date):
    n = tasks.get("tasks", {}).get(task, {}).get("per_date_notes", {}).get(date, {})
    return tuple(n.get("active_sensors") or ("left", "right"))


def freeze_intervals(pose, min_frames):
    T = pose.shape[0]
    same = np.zeros(T, dtype=bool)
    same[1:] = np.all(np.abs(np.diff(pose, axis=0)) < EPS, axis=1)
    ivs = []
    i = 1
    while i < T:
        if not same[i]:
            i += 1
            continue
        j = i
        while j < T and same[j]:
            j += 1
        a, b = i - 1, j - 1
        if (b - a + 1) >= min_frames:
            ivs.append((a, b))
        i = j
    return ivs


def merge_adjacent(intervals, max_gap=1):
    if not intervals:
        return []
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a - out[-1][1] <= max_gap + 1:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def schedule_clips(a, b, T_total):
    """One or more (clip_a, clip_b, label) windows per freeze event.

    Short freezes (≤ 5 s) → whole event + 1 s pad each side.
    Long  freezes (>  5 s) → onset + offset + up to 6 mid samples (each 5 s).
    """
    L_s = (b - a + 1) / FPS
    if L_s <= 5.0:
        return [(max(0, a - PAD_FR), min(T_total - 1, b + PAD_FR), "full")]
    clips = []
    onset_end = min(b, a + CLIP_FR - PAD_FR - 1)
    clips.append((max(0, a - PAD_FR), onset_end, "onset"))
    offset_start = max(a, b - CLIP_FR + PAD_FR + 1)
    clips.append((offset_start, min(T_total - 1, b + PAD_FR), "offset"))
    if L_s > LONG_THRESHOLD_S:
        n_mid = min(MAX_MID_CLIPS, max(2, int(round(L_s / 60))))
        mid_lo = a + CLIP_FR
        mid_hi = b - CLIP_FR - CLIP_FR + 1
        if mid_hi > mid_lo:
            for s in np.linspace(mid_lo, mid_hi, n_mid).astype(int):
                clips.append((int(s), int(s + CLIP_FR - 1), "mid"))
    return clips


def load_cam_calibs():
    """Return {cam_idx: {T_mocap_to_cam, intrinsics}} keyed by H5 cam idx."""
    out = {}
    for cam_idx, name in CAM_CALIB_NAME.items():
        d = json.loads((CALIB_DIR / name).read_text())
        out[cam_idx] = {
            "T_mocap_to_cam": np.array(d["T_mocap_to_cam"], np.float64),
            "intrinsics": d["intrinsics"],
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-clip renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_clip(date, ep_stem, side, freeze_a, freeze_b,
                clip_a, clip_b, label, clip_idx, total_in_event,
                pose_L, pose_R, h5_path, cam_calibs, gel_L, gel_R):
    """Render a [clip_a, clip_b] window as a 1280×480 30 fps GIF and return
    (out_path, size_bytes). The freeze span [freeze_a, freeze_b] gets a red
    FROZEN ring on the affected sensor's projected dot in every cam thumb.
    """
    project_cams = [
        {"index": c,
         "T_mocap_to_cam": cam_calibs[c]["T_mocap_to_cam"],
         "intrinsics": cam_calibs[c]["intrinsics"]}
        for c in (0, 1, 2)
    ]

    # Prefer the per-episode p01 reference (least-contacted moment of the
    # episode) over the clip's first frame. The p01 PNGs are shipped with
    # the dataset; if they aren't present we fall back to clip[0].
    PT_ROOT_LOCAL = Path("/media/yxma/Disk1/twm/processed/mode1_v1/motherboard")
    ref_L_path = PT_ROOT_LOCAL / date / f"{ep_stem}.gs_ref_left.png"
    ref_R_path = PT_ROOT_LOCAL / date / f"{ep_stem}.gs_ref_right.png"
    ep_ref = None
    if ref_L_path.is_file() and ref_R_path.is_file():
        from PIL import Image as _Img
        ep_ref = [np.array(_Img.open(ref_L_path).convert("RGB")),
                  np.array(_Img.open(ref_R_path).convert("RGB"))]

    with h5py.File(h5_path, "r") as h5:
        timestamps = h5["timestamps"][clip_a:clip_b + 1]
        cam_slices = {c: h5[f"realsense/cam{c}/color"][clip_a:clip_b + 1] for c in (0, 1, 2)}
        gs_left = h5["gelsight/left/frames"][clip_a:clip_b + 1]
        gs_right = h5["gelsight/right/frames"][clip_a:clip_b + 1]
        gs_ref = ep_ref if ep_ref is not None else [gs_left[0].copy(), gs_right[0].copy()]
        optitrack_lookup = load_optitrack(h5)
        try:
            task_name = str(h5["metadata"].attrs.get("task", ""))
        except Exception:
            task_name = ""

    L_s = (freeze_b - freeze_a + 1) / FPS
    frames_rgb = []
    for t_rel in range(clip_b - clip_a + 1):
        t = clip_a + t_rel
        color_frames = {c: cam_slices[c][t_rel] for c in (0, 1, 2)}
        gs_frames = [gs_left[t_rel], gs_right[t_rel]]
        cam_t = float(timestamps[t_rel])
        optitrack_poses = optitrack_at(optitrack_lookup, cam_t)
        elapsed = float(timestamps[t_rel] - timestamps[0])

        # Compose custom status line — replaces the live UI's REC/IDLE bar
        in_freeze = (freeze_a <= t <= freeze_b)
        flag = "** FROZEN **" if in_freeze else "tracking"
        status = (f"[{task_name}]  freeze inspect  ep {date}/{ep_stem}  "
                  f"side={side}  freeze=[{freeze_a},{freeze_b}] ({L_s:.1f}s)  "
                  f"t={t}  clip={label} {clip_idx + 1}/{total_in_event}  {flag}")

        preview = make_preview(
            color_frames, gs_frames, gs_ref, optitrack_poses,
            False, t, elapsed, 0, FPS, task_name, status,
        )
        draw_projection_overlay(
            preview, optitrack_poses, project_cams, gel_L, gel_R,
            frozen_side=side if in_freeze else None,
        )

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        frames_rgb.append(preview_rgb)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{date}_{ep_stem}_{side}_{freeze_a}-{freeze_b}_clip{clip_idx:02d}_{label}.gif"
    size = save_gif(frames_rgb, out_path, fps=FPS, max_kb=MAX_KB,
                    palette_colors=256, min_palette=128)
    return out_path, size


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

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

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.gif"):
        old.unlink()
    print(f"Cleared old clips in {OUT}")

    min_frames = int(round(MIN_S * FPS))
    total = 0
    rows = []
    for date_dir in sorted(H5_ROOT.iterdir()):
        date = date_dir.name
        if date == "2026-03-23" or not date_dir.is_dir():
            continue
        for h5 in sorted(date_dir.glob("episode_*.h5")):
            ep_stem = h5.stem
            with h5py.File(h5, "r") as f:
                cam_ts = f["timestamps"][:]
                sl_ts = f["optitrack/sensor_left/timestamps"][:]
                sl_poses = f["optitrack/sensor_left/pose"][:]
                sr_ts = f["optitrack/sensor_right/timestamps"][:]
                sr_poses = f["optitrack/sensor_right/pose"][:]
            pose_L = cam_aligned_pose(cam_ts, sl_ts, sl_poses)
            pose_R = cam_aligned_pose(cam_ts, sr_ts, sr_poses)
            T_total = pose_L.shape[0]
            active = active_for(tasks, "motherboard", date)
            events = []
            for side in active:
                pose = pose_L if side == "left" else pose_R
                for a, b in merge_adjacent(freeze_intervals(pose, min_frames)):
                    events.append((side, a, b))
            if not events:
                continue
            print(f"\n{date}/{ep_stem}: {len(events)} freeze event(s)")
            for side, a, b in events:
                L_s = (b - a + 1) / FPS
                schedule = schedule_clips(a, b, T_total)
                print(f"  side={side} [{a},{b}] len={L_s:.1f}s → {len(schedule)} clip(s)")
                for ci, (ca, cb, label) in enumerate(schedule):
                    try:
                        p, sz = render_clip(
                            date, ep_stem, side, a, b, ca, cb, label,
                            ci, len(schedule),
                            pose_L, pose_R, h5, cam_calibs, gel_L, gel_R,
                        )
                        print(f"    -> {p.name}  ({sz / 1024:.0f} KB)")
                        rows.append({
                            "episode": f"{date}/{ep_stem}", "side": side,
                            "freeze_a": a, "freeze_b": b,
                            "freeze_len_s": round(L_s, 3),
                            "clip_a": ca, "clip_b": cb, "label": label,
                            "gif": str(p),
                        })
                        total += 1
                    except Exception:
                        import traceback
                        traceback.print_exc()
            del pose_L, pose_R
            gc.collect()

    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nTotal: {total} clips in {OUT}")


if __name__ == "__main__":
    main()
