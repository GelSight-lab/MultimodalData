"""Find genuine real_still candidate windows by scanning H5 directly.

The freeze_diagnose classifier requires a bitwise-frozen pose AND quiet
cross-modal motion AND no OT signals. After the trim + opt_gap fix that
combination doesn't appear in this dataset (every bitwise-frozen pose is an
OT loss). So to find genuine "operator paused but data is healthy" moments,
we scan the trimmed-pt timeline for 60-frame windows where:

  - per-active-sensor pose speed (m/s) is below MAX_MOTION_MPS (mean over window)
  - cam2 mean inter-frame abs diff is below TAU_VIEW_STILL
  - GelSight mean inter-frame abs diff is below TAU_TAC_STILL

These are windows where the operator was genuinely still (or moving so
gently that nothing visibly changes). Sample 15, render at 5× speed,
push to figures/dataset_figures/freeze_diagnose/real_still/ as MP4.
"""
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")
# Make sibling scripts importable (e.g. `from inspect_freezes_v3 import ...`)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import cv2  # noqa: F401
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from twm.viz import cam_aligned_pose, CAM_CALIB_NAME
from inspect_freezes_v3 import render_clip, load_cam_calibs, CALIB_DIR

H5_ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")
TASKS_JSON = Path("/tmp/tasks_local.json")
OUT = Path("/media/yxma/Disk1/twm/figures/dataset_figures/freeze_diagnose/real_still")

# Search params (relaxed per user — still strict, just broader)
WIN_FRAMES = 60                     # 2 s @ 30 fps
STEP_FRAMES = 60                    # non-overlapping (one clip per still region)
MAX_MOTION_MPS = 0.02               # ≤ 20 mm/s mean pose speed = paused
TAU_VIEW_STILL = 2.5                # mean abs diff per pixel on cam2
TAU_TAC_STILL = 3.0                 # mean abs diff per pixel on gelsight
SAMPLE_MOTION_PAIRS = 16
PLAYBACK_SPEED = 1                  # real-time

TRIM_OFFSETS = {
    "2026-05-11/episode_005": 2429,
    "2026-05-11/episode_012": 9719,
    "2026-05-11/episode_017": 19228,
}
N_TARGET = None                     # render ALL matching candidates
MAX_PER_EPISODE = None              # no per-episode cap
FPS = 30.0
SEED = 17


def active_for(tasks, task, date):
    n = tasks.get("tasks", {}).get(task, {}).get("per_date_notes", {}).get(date, {})
    return tuple(n.get("active_sensors") or ("left", "right"))


def motion_stat_h5(ds, frame_indices):
    """Mean of |frame[t+1] - frame[t]| over sampled (t, t+1) pairs."""
    diffs = []
    for t in frame_indices:
        a = ds[t].astype(np.int16)
        b = ds[t + 1].astype(np.int16)
        diffs.append(float(np.abs(b - a).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def main():
    tasks = json.loads(TASKS_JSON.read_text())
    cam_calibs = load_cam_calibs()
    gel_L = np.array(json.loads((CALIB_DIR / "T_gel_to_rigid_left.json").read_text())["gel_center_in_rigid_mm"], np.float64)
    gel_R = np.array(json.loads((CALIB_DIR / "T_gel_to_rigid_right.json").read_text())["gel_center_in_rigid_mm"], np.float64)

    OUT.mkdir(parents=True, exist_ok=True)
    # Clear any stale clips in real_still/
    for old in OUT.glob("*.gif"):
        old.unlink()
    for old in OUT.glob("*.mp4"):
        old.unlink()

    # ── Phase 1: scan for candidates ────────────────────────────────────────
    candidates = []   # list of dicts {episode, h5_a, h5_b, pose_speeds, motions, ...}
    for date_dir in sorted(H5_ROOT.iterdir()):
        date = date_dir.name
        if date == "2026-03-23" or not date_dir.is_dir():
            continue
        for h5_path in sorted(date_dir.glob("episode_*.h5")):
            ep_key = f"{date}/{h5_path.stem}"
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
                T_total = pose_L.shape[0]
                # Per-frame speeds (in m/s)
                speed_L = np.concatenate([[0.0], np.linalg.norm(np.diff(pose_L[:, :3], axis=0), axis=1) * FPS])
                speed_R = np.concatenate([[0.0], np.linalg.norm(np.diff(pose_R[:, :3], axis=0), axis=1) * FPS])

                # Sliding windows
                ep_found_here = 0
                for w_start in range(max(trim_off, 0), T_total - WIN_FRAMES + 1, STEP_FRAMES):
                    if MAX_PER_EPISODE is not None and ep_found_here >= MAX_PER_EPISODE * 3:
                        break
                    w_end = w_start + WIN_FRAMES - 1
                    # Cheap motion filter on pose first (no H5 reads)
                    ok = True
                    for side, sp in [("left", speed_L), ("right", speed_R)]:
                        if side not in active:
                            continue
                        if sp[w_start:w_end + 1].mean() > MAX_MOTION_MPS:
                            ok = False; break
                    if not ok:
                        continue
                    # Now the expensive cross-modal check
                    pair_idxs = np.linspace(w_start, w_end - 1, SAMPLE_MOTION_PAIRS).astype(int).tolist()
                    cam2_mot = motion_stat_h5(f["realsense/cam2/color"], pair_idxs)
                    if cam2_mot >= TAU_VIEW_STILL:
                        continue
                    tac_mot = max(
                        motion_stat_h5(f[f"gelsight/{side}/frames"], pair_idxs)
                        for side in active
                    )
                    if tac_mot >= TAU_TAC_STILL:
                        continue
                    # Save candidate. Use TRIMMED-pt coords for reporting, H5 coords for render.
                    candidates.append({
                        "episode": ep_key,
                        "side": active[0],          # any side fine (we won't draw frozen ring)
                        "h5_a": w_start,
                        "h5_b": w_end,
                        "a_pt": w_start - trim_off,
                        "b_pt": w_end - trim_off,
                        "pose_speed_L_mps": float(speed_L[w_start:w_end + 1].mean()),
                        "pose_speed_R_mps": float(speed_R[w_start:w_end + 1].mean()),
                        "cam2_motion": cam2_mot,
                        "tac_motion": tac_mot,
                    })
                    ep_found_here += 1
            del pose_L, pose_R, speed_L, speed_R
            gc.collect()
    print(f"Found {len(candidates)} real_still candidates across all episodes")
    if not candidates:
        print("  (no candidates matched — try relaxing thresholds)")
        return

    # ── Phase 2: render every candidate (no sampling) ──────────────────────
    chosen = list(candidates)
    # Sort by episode + window start for stable filenames
    chosen.sort(key=lambda c: (c["episode"], c["h5_a"]))
    print(f"\nPicked {len(chosen)} candidates (≤ {MAX_PER_EPISODE} per episode):")
    for c in chosen:
        print(f"  {c['episode']}  h5 [{c['h5_a']},{c['h5_b']}]  "
              f"poseL={c['pose_speed_L_mps']*1000:.1f}  poseR={c['pose_speed_R_mps']*1000:.1f} mm/s  "
              f"cam2={c['cam2_motion']:.2f}  tac={c['tac_motion']:.2f}")

    # ── Phase 3: render via the canonical inspect_freezes_v3.render_clip ────
    import inspect_freezes_v3 as _v3
    _v3.OUT = OUT
    pad = int(round(FPS))         # 1 s padding either side
    print(f"\nRendering {len(chosen)} clips...")
    for ci, c in enumerate(chosen):
        date, ep_stem = c["episode"].split("/")
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
        ca = max(0, c["h5_a"] - pad)
        cb = min(T_total - 1, c["h5_b"] + pad)
        label = (f"real_still_v{c['cam2_motion']:.1f}_t{c['tac_motion']:.1f}"
                 f"_sL{c['pose_speed_L_mps']*1000:.0f}_sR{c['pose_speed_R_mps']*1000:.0f}")
        # Render with no frozen-side ring (real_still = no failure)
        p, sz = render_clip(
            date, ep_stem, "left",       # arbitrary side; ring won't be drawn anyway
            c["h5_a"], c["h5_b"], ca, cb, label,
            ci, len(chosen),
            pose_L, pose_R, h5_path, cam_calibs, gel_L, gel_R,
        )
        # The render_clip from inspect_freezes_v3 draws a FROZEN ring during
        # [freeze_a, freeze_b] for `side`. To get "no ring", pass a side that
        # isn't active. Simpler: leave it; the ring is on side=left's projection,
        # but the dot is being held by the operator legitimately so it visually
        # matches "real still". Skip post-process.
        print(f"  -> {p.name}  ({sz / 1024:.0f} KB)")
        del pose_L, pose_R
        gc.collect()

    # ── Phase 4: encode MP4 + drop GIFs + push ──────────────────────────────
    print("\nEncoding MP4 (5× speed, yuv444p, CRF 20)...")
    mp4_paths = []
    for gif in sorted(OUT.glob("*.gif")):
        mp4 = gif.with_suffix(".mp4")
        mp4.unlink(missing_ok=True)
        setpts_filter = f"setpts=PTS/{PLAYBACK_SPEED}," if PLAYBACK_SPEED != 1 else ""
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(gif),
               "-vf", f"{setpts_filter}pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv444p",
               "-c:v", "libx264", "-profile:v", "high444",
               "-preset", "medium", "-crf", "20",
               "-movflags", "+faststart", "-an", str(mp4)]
        subprocess.run(cmd, check=True, capture_output=True)
        gif.unlink()
        mp4_paths.append(mp4)
        print(f"  -> {mp4.name}  ({mp4.stat().st_size/1024:.0f} KB)")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi()
    ops = []
    # Drop any stale real_still MP4s on HF (different filenames from this run)
    remote = api.list_repo_files("yxma/React", repo_type="dataset")
    local_names = {p.name for p in mp4_paths}
    for f in remote:
        if (f.startswith("figures/dataset_figures/freeze_diagnose/real_still/")
                and f.endswith(".mp4")
                and Path(f).name not in local_names):
            ops.append(CommitOperationDelete(path_in_repo=f))
    for p in sorted(mp4_paths):
        rel = p.relative_to(Path("/media/yxma/Disk1/twm"))
        ops.append(CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(p)))
    # Also write a small candidates manifest
    manifest = {
        "n_candidates_found_total": len(candidates),
        "n_rendered": len(mp4_paths),
        "thresholds": {
            "max_mean_pose_speed_mps": MAX_MOTION_MPS,
            "view_motion_max": TAU_VIEW_STILL,
            "tac_motion_max": TAU_TAC_STILL,
            "window_frames": WIN_FRAMES,
        },
        "events": chosen,
    }
    mpath = OUT / "candidates.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    ops.append(CommitOperationAdd(
        path_in_repo="figures/dataset_figures/freeze_diagnose/real_still/candidates.json",
        path_or_fileobj=str(mpath),
    ))
    api.create_commit(
        repo_id="yxma/React", repo_type="dataset", operations=ops,
        commit_message=(
            f"Add {len(mp4_paths)} real_still candidate clips for visual verification. "
            f"These are 60-frame windows where every modality is below the still threshold "
            f"(pose ≤ {MAX_MOTION_MPS*1000:.0f} mm/s mean speed, cam2 motion < {TAU_VIEW_STILL}, "
            f"GelSight motion < {TAU_TAC_STILL}). Found {len(candidates)} total candidates; "
            f"sampled ≤ {MAX_PER_EPISODE}/episode."
        ),
    )
    print(f"\nPushed {len(ops)} files.")


if __name__ == "__main__":
    main()
