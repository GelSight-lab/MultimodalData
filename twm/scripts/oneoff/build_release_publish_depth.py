"""Publish the depth + object_pose addition (incremental, on top of the
existing video release).

Uploads:
  data/<task>/depth/...                 (FFV1 16-bit MKV)
  data/<task>/meta/*.parquet            (now with object_pose column)
  data/<task>/episodes.jsonl            (now with object_tracked)
  tasks.json (refreshed), README.md (depth+object_pose), loader
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationAdd

REPO = "yxma/React"
STAGE = Path("/media/yxma/Disk1/twm/release")
REPO_ROOT = Path("/home/yxma/MultimodalData/twm")
TASKS = ("motherboard", "pushT")


def refresh_tasks_json():
    R = STAGE
    def summary(task, calib_id, created, unit, gel_left, note):
        bf = json.loads((R/task/"bad_frames.json").read_text())
        sj = json.loads((R/task/"segments.json").read_text())
        eps = [json.loads(l) for l in (R/task/"episodes.jsonl").read_text().splitlines()]
        n_obj = sum(1 for e in eps if e.get("object_tracked"))
        n_depth = len(list((R/task/"depth").rglob("depth_middle.mkv"))) if (R/task/"depth").is_dir() else 0
        return {
            "n_episodes": len(eps), "dates": sorted({e["date"] for e in eps}),
            "n_frames": bf["summary"]["total_frames"],
            "duration_min": round(bf["summary"]["total_frames"]/1800, 1),
            "n_segments": sj["n_segments"], "clean_min": sj["total_duration_min"],
            "bad_fraction": bf["summary"]["bad_fraction_overall"],
            "active_sensors": ["left", "right"],
            "calibration_id": calib_id, "calibration_created": created,
            "calibration_rmse_unit": unit,
            "object_tracked_episodes": n_obj,
            "depth_available_episodes": n_depth,
            "depth_units": "mm", "depth_invalid_value": 0,
            "gelsight_left_serial": gel_left, "note": note,
        }
    tasks = {
        "motherboard": summary("motherboard", "may-12", "2026-05-12", "mm", "2BGLKZNT/2DUPB53G",
            "Bimanual handheld tactile-visual interaction. Object pose (the board) tracked. 05-19 has a redefined OptiTrack world origin; offset (0.23,0,0.175)m baked into poses."),
        "pushT": summary("pushT", "june-26", "2026-06-26", "px", "2DUPB53G",
            "Push-T manipulation. Recalibrated cameras (June-26). Object rigid body was not tracked (object_pose = NaN). episode_004 source H5 corrupt, excluded."),
    }
    out = {"dataset": "React",
           "format": "video (LeRobot-style: per-camera MP4 + per-episode parquet; depth as FFV1 16-bit MKV)",
           "resolution": "640x480", "fps": 30,
           "video_streams": ["view_left", "view_middle", "view_right", "tactile_left", "tactile_right"],
           "depth_streams": ["depth_left", "depth_middle", "depth_right"],
           "parquet_columns": ["frame_idx", "timestamp", "sensor_left_pose", "sensor_right_pose",
                               "object_pose", "tactile_{L,R}_{intensity,area,mixed}", "source_h5_frame"],
           "decoded_color": "RGB", "tasks": tasks}
    p = "/tmp/tasks_v3.json"
    Path(p).write_text(json.dumps(out, indent=2))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    api = HfApi()

    for task in TASKS:
        depth_dir = STAGE / task / "depth"
        if depth_dir.is_dir():
            print(f"[publish-depth] uploading data/{task}/depth/ ...", flush=True)
            if not args.dry_run:
                api.upload_folder(repo_id=REPO, repo_type="dataset",
                                  folder_path=str(depth_dir), path_in_repo=f"data/{task}/depth",
                                  commit_message=f"Add {task} depth (FFV1 16-bit, lossless mm)")
        # re-upload parquet (now with object_pose) + episodes.jsonl
        print(f"[publish-depth] uploading data/{task}/meta + episodes.jsonl ...", flush=True)
        if not args.dry_run:
            api.upload_folder(repo_id=REPO, repo_type="dataset",
                              folder_path=str(STAGE / task / "meta"), path_in_repo=f"data/{task}/meta",
                              ignore_patterns=["*._detect.pt"],
                              commit_message=f"Add object_pose column to {task} parquet")
            api.upload_file(path_or_fileobj=str(STAGE / task / "episodes.jsonl"),
                            path_in_repo=f"data/{task}/episodes.jsonl", repo_id=REPO, repo_type="dataset",
                            commit_message=f"Update {task} episodes.jsonl (object_tracked)")

    print("[publish-depth] refreshing tasks.json + README + loader ...", flush=True)
    tasks_p = refresh_tasks_json()
    if not args.dry_run:
        ops = [
            CommitOperationAdd("tasks.json", tasks_p),
            CommitOperationAdd("README.md", str(REPO_ROOT / "docs/superpowers/specs/README_v2_release.md")),
            CommitOperationAdd("examples/react_video_dataset.py", str(REPO_ROOT / "examples/react_video_dataset.py")),
        ]
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message="Depth + object_pose: refresh tasks.json, README, loader")
    print("[publish-depth] done", flush=True)


if __name__ == "__main__":
    main()
