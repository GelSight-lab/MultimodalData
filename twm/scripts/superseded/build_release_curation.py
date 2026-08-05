"""Task-first curation: from the per-episode `_detect.pt` sidecars produced by
build_video_release.py, build per-task:
  data/<task>/bad_frames.json   (intensity_spikes, pose_teleports_*, ot_loss_*)
  data/<task>/segments.json     (clean-segment index -> episode + frame_range)
  data/<task>/episodes.jsonl    (one row per episode)

Reuses the validated detector (detect_bad_intervals.py, 25/27 bit-identical
vs the published motherboard bad_frames.json) and the clean-segment
complement logic (build_segments.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_bad_intervals as D
from build_segments import find_clean_segments

STAGE = Path("/media/yxma/Disk1/twm/release")
MIN_SEGMENT_FRAMES = 16
TACTILE_THRESHOLD = 0.4
FPS = 30.0


def detect_from_sidecar(det_path: Path):
    ep = torch.load(str(det_path), weights_only=False, map_location="cpu")
    T = int(ep["timestamps"].shape[0])
    cm = ep["_contact_meta"]
    active = cm.get("active_sensors", ["left", "right"])
    iL = ep["tactile_left_intensity"].numpy()
    iR = ep["tactile_right_intensity"].numpy()
    pL = ep["sensor_left_pose"].numpy()
    pR = ep["sensor_right_pose"].numpy()
    entry = {
        "n_frames": T,
        "duration_s": round(T / FPS, 3),
        "intensity_spikes": D.detect_intensity_spikes(iL, iR, T),
        "pose_teleports_L": D.detect_pose_teleports(pL, T) if "left" in active else [],
        "pose_teleports_R": D.detect_pose_teleports(pR, T) if "right" in active else [],
        "ot_loss_L": D.detect_pose_freezes(pL, T) if "left" in active else [],
        "ot_loss_R": D.detect_pose_freezes(pR, T) if "right" in active else [],
    }
    mask = np.zeros(T, bool)
    for iv in (entry["intensity_spikes"], entry["pose_teleports_L"], entry["pose_teleports_R"],
               entry["ot_loss_L"], entry["ot_loss_R"]):
        for a, b in iv:
            mask[max(0, a):min(T, b + 1)] = True
    entry["total_bad_frames"] = int(mask.sum())
    entry["bad_fraction"] = round(entry["total_bad_frames"] / T, 4) if T else 0.0
    return entry, cm, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    task = args.task
    meta_root = STAGE / task / "meta"

    sidecars = sorted(meta_root.rglob("*._detect.pt"))
    if not sidecars:
        print(f"no _detect.pt under {meta_root}", file=sys.stderr); sys.exit(1)

    episodes_bf = {}
    segments = []
    episodes_jsonl = []
    for det in sidecars:
        date = det.parent.name
        ep_stem = det.name.replace("._detect.pt", "")
        ep_key = f"{date}/{ep_stem}"
        entry, cm, ep = detect_from_sidecar(det)
        episodes_bf[ep_key] = entry
        T = entry["n_frames"]

        # clean segments = complement of union(bad intervals)
        bad = (entry["intensity_spikes"] + entry["pose_teleports_L"] + entry["pose_teleports_R"]
               + entry["ot_loss_L"] + entry["ot_loss_R"])
        bad = [(int(a), int(b)) for a, b in bad]
        clean = find_clean_segments(T, bad)
        seg_idx = 0
        mL = ep["tactile_left_intensity"].numpy() if "tactile_left_intensity" in ep else None
        for a, b in clean:
            L = b - a + 1
            if L < MIN_SEGMENT_FRAMES:
                continue
            segments.append({
                "task": task,
                "source_episode": ep_key,
                "segment_idx": seg_idx,
                "frame_range": [a, b],          # inclusive, in episode-video frame coords
                "n_frames": L,
                "duration_s": round(L / FPS, 3),
            })
            seg_idx += 1

        episodes_jsonl.append({
            "episode": ep_key,
            "date": date,
            "n_frames": T,
            "duration_s": entry["duration_s"],
            "active_sensors": cm.get("active_sensors", ["left", "right"]),
            "trim_offset": int(cm.get("trim_offset", 0)),
            "world_frame_offset": cm.get("world_frame_offset_applied", [0.0, 0.0, 0.0]),
            "n_segments": seg_idx,
            "total_bad_frames": entry["total_bad_frames"],
        })

    out_dir = STAGE / task
    # bad_frames.json
    total_frames = sum(e["n_frames"] for e in episodes_bf.values())
    total_bad = sum(e["total_bad_frames"] for e in episodes_bf.values())
    bf = {
        "task": task,
        "tau_intensity": D.TAU_INTENSITY, "tau_velocity_mps": D.TAU_VELOCITY_MPS,
        "tau_angular_rad_per_s": D.TAU_ANGULAR_RAD_PS, "freeze_threshold_s": D.FREEZE_THRESHOLD_S,
        "buffer_frames": D.BUFFER_FRAMES,
        "summary": {"n_episodes": len(episodes_bf), "total_frames": total_frames,
                    "total_bad_frames": total_bad,
                    "bad_fraction_overall": round(total_bad / total_frames, 4) if total_frames else 0.0},
        "episodes": episodes_bf,
    }
    (out_dir / "bad_frames.json").write_text(json.dumps(bf, indent=2))

    # segments.json
    seg_total = sum(s["n_frames"] for s in segments)
    sj = {
        "task": task, "schema": "segments_v2_video",
        "description": ("Each entry indexes a contiguous clean span within an episode's "
                        "videos (data/<task>/videos/<date>/episode_NNN/*.mp4) and parquet. "
                        "frame_range is [a,b] inclusive in episode-video frame coords."),
        "n_segments": len(segments), "total_frames": seg_total,
        "total_duration_min": round(seg_total / FPS / 60, 2),
        "min_segment_frames_kept": MIN_SEGMENT_FRAMES,
        "segments": sorted(segments, key=lambda s: (s["source_episode"], s["segment_idx"])),
    }
    (out_dir / "segments.json").write_text(json.dumps(sj, indent=2))

    # episodes.jsonl
    with open(out_dir / "episodes.jsonl", "w") as f:
        for row in sorted(episodes_jsonl, key=lambda r: r["episode"]):
            f.write(json.dumps(row) + "\n")

    print(f"[curation] {task}: {len(episodes_bf)} episodes, {len(segments)} segments, "
          f"{total_frames:,} frames, {total_bad} bad ({100*total_bad/total_frames:.2f}%), "
          f"clean {seg_total:,} ({seg_total/FPS/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
