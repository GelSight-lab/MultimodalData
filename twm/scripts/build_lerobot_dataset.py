"""Build a LeRobot v2.1 dataset (yxma/React-lerobot) from the React video
release, reusing the existing MP4 via hardlinks (no re-encode, no copy).

Layout produced under STAGE_LR:
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  data/chunk-000/episode_{idx:06d}.parquet
  videos/chunk-000/observation.images.<key>/episode_{idx:06d}.mp4   (hardlinks)

Feature design (valid for both tasks, no NaN):
  observation.state            float32[14]  sensor_left_pose(7)+sensor_right_pose(7)
  observation.object_pose      float32[7]   object body pose (0-filled where untracked)
  observation.object_pose_valid float32[1]  1.0 if object tracked else 0.0
  observation.tactile          float32[6]   L/R intensity,area,mixed
  action                       float32[14]  next-frame observation.state (last repeats)
  observation.images.{view_left,view_middle,view_right,tactile_left,tactile_right} : video
  + timestamp, frame_index, episode_index, index, task_index
Depth is intentionally excluded (LeRobot video is lossy h264).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REL = Path("/media/yxma/Disk1/twm/release")
STAGE_LR = Path("/media/yxma/Disk1/twm/lerobot")
FPS = 30
CHUNK = 0
VIDEO_KEYS = ["observation.images.view_left", "observation.images.view_middle",
              "observation.images.view_right", "observation.images.tactile_left",
              "observation.images.tactile_right"]
SRC_STREAM = {  # lerobot key -> release mp4 filename
    "observation.images.view_left": "view_left", "observation.images.view_middle": "view_middle",
    "observation.images.view_right": "view_right", "observation.images.tactile_left": "tactile_left",
    "observation.images.tactile_right": "tactile_right"}
TASK_STRINGS = {"motherboard": "bimanual handheld tactile interaction with a motherboard",
                "pushT": "push a T-shaped object"}
TASK_ORDER = ["motherboard", "pushT"]


def list_episodes():
    """Return ordered list of (task, date, ep_stem, parquet_path, video_dir)."""
    out = []
    for task in TASK_ORDER:
        for pqf in sorted((REL / task / "meta").rglob("episode_*.parquet")):
            date = pqf.parent.name
            stem = pqf.stem
            out.append((task, date, stem, pqf, REL / task / "videos" / date / stem))
    return out


def img_stats_sample(video_path, n=20):
    """Per-channel mean/std/min/max in [0,1] from a strided sample of frames."""
    c = av.open(str(video_path))
    frames = []
    total = c.streams.video[0].frames or 0
    step = max(1, total // n) if total else 1
    for i, fr in enumerate(c.decode(c.streams.video[0])):
        if i % step == 0:
            frames.append(fr.to_ndarray(format="rgb24"))
        if len(frames) >= n:
            break
    c.close()
    arr = np.stack(frames).astype(np.float32) / 255.0     # (n,H,W,3)
    # lerobot image stats shape: (3,1,1)
    mean = arr.mean(axis=(0, 1, 2)).reshape(3, 1, 1)
    std = arr.std(axis=(0, 1, 2)).reshape(3, 1, 1)
    mn = arr.min(axis=(0, 1, 2)).reshape(3, 1, 1)
    mx = arr.max(axis=(0, 1, 2)).reshape(3, 1, 1)
    return {"mean": mean.tolist(), "std": std.tolist(),
            "min": mn.tolist(), "max": mx.tolist(), "count": [arr.shape[0]]}


def vec_stats(a):
    a = np.asarray(a, np.float32)
    return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist(), "count": [a.shape[0]]}


def main():
    STAGE_LR.mkdir(parents=True, exist_ok=True)
    (STAGE_LR / "meta").mkdir(exist_ok=True)
    (STAGE_LR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for k in VIDEO_KEYS:
        (STAGE_LR / "videos" / "chunk-000" / k).mkdir(parents=True, exist_ok=True)

    eps = list_episodes()
    episodes_jsonl = []
    episodes_stats = []
    global_index = 0
    total_frames = 0

    for gidx, (task, date, stem, pqf, vdir) in enumerate(eps):
        tbl = pq.read_table(pqf)
        T = tbl.num_rows
        slp = np.array(tbl.column("sensor_left_pose").to_pylist(), np.float32)
        srp = np.array(tbl.column("sensor_right_pose").to_pylist(), np.float32)
        op = np.array(tbl.column("object_pose").to_pylist(), np.float32)
        obj_valid = np.isfinite(op).all(axis=1).astype(np.float32)
        op = np.nan_to_num(op, nan=0.0)
        tac = np.stack([
            tbl.column("tactile_left_intensity").to_numpy(),
            tbl.column("tactile_left_area").to_numpy(),
            tbl.column("tactile_left_mixed").to_numpy(),
            tbl.column("tactile_right_intensity").to_numpy(),
            tbl.column("tactile_right_area").to_numpy(),
            tbl.column("tactile_right_mixed").to_numpy()], axis=1).astype(np.float32)
        state = np.concatenate([slp, srp], axis=1)          # (T,14)
        action = np.concatenate([state[1:], state[-1:]], axis=0)  # next-frame, last repeats
        ts = (np.arange(T) / FPS).astype(np.float32)
        task_index = TASK_ORDER.index(task)

        out = pa.table({
            "observation.state": list(state),
            "observation.object_pose": list(op),
            "observation.object_pose_valid": obj_valid,
            "observation.tactile": list(tac),
            "action": list(action),
            "timestamp": ts,
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, gidx, np.int64),
            "index": np.arange(global_index, global_index + T, dtype=np.int64),
            "task_index": np.full(T, task_index, np.int64),
        })
        pq.write_table(out, str(STAGE_LR / "data" / "chunk-000" / f"episode_{gidx:06d}.parquet"))

        # hardlink videos
        for k in VIDEO_KEYS:
            src = vdir / f"{SRC_STREAM[k]}.mp4"
            dst = STAGE_LR / "videos" / "chunk-000" / k / f"episode_{gidx:06d}.mp4"
            if dst.exists():
                dst.unlink()
            os.link(src, dst)

        # stats
        st = {"observation.state": vec_stats(state), "action": vec_stats(action),
              "observation.object_pose": vec_stats(op),
              "observation.object_pose_valid": vec_stats(obj_valid.reshape(-1, 1)),
              "observation.tactile": vec_stats(tac)}
        for k in VIDEO_KEYS:
            st[k] = img_stats_sample(vdir / f"{SRC_STREAM[k]}.mp4")
        episodes_stats.append({"episode_index": gidx, "stats": st})
        episodes_jsonl.append({"episode_index": gidx, "tasks": [TASK_STRINGS[task]], "length": T})

        global_index += T
        total_frames += T
        print(f"  ep{gidx:03d} {task}/{date}/{stem}  T={T}", flush=True)

    # meta/tasks.jsonl
    with open(STAGE_LR / "meta" / "tasks.jsonl", "w") as f:
        for t in TASK_ORDER:
            f.write(json.dumps({"task_index": TASK_ORDER.index(t), "task": TASK_STRINGS[t]}) + "\n")
    with open(STAGE_LR / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes_jsonl:
            f.write(json.dumps(e) + "\n")
    with open(STAGE_LR / "meta" / "episodes_stats.jsonl", "w") as f:
        for e in episodes_stats:
            f.write(json.dumps(e) + "\n")

    # video info from a real file
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt", "-of", "json",
        str(STAGE_LR / "videos" / "chunk-000" / VIDEO_KEYS[0] / "episode_000000.mp4")],
        capture_output=True, text=True)
    vs = json.loads(probe.stdout)["streams"][0]

    def vfeat():
        return {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"],
                "info": {"video.fps": float(FPS), "video.height": 480, "video.width": 640,
                         "video.channels": 3, "video.codec": vs["codec_name"],
                         "video.pix_fmt": vs["pix_fmt"], "video.is_depth_map": False,
                         "has_audio": False}}
    features = {k: vfeat() for k in VIDEO_KEYS}
    features.update({
        "observation.state": {"dtype": "float32", "shape": [14],
            "names": [f"{s}_{c}" for s in ("left", "right") for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]},
        "observation.object_pose": {"dtype": "float32", "shape": [7],
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
        "observation.object_pose_valid": {"dtype": "float32", "shape": [1], "names": None},
        "observation.tactile": {"dtype": "float32", "shape": [6],
            "names": ["L_intensity", "L_area", "L_mixed", "R_intensity", "R_area", "R_mixed"]},
        "action": {"dtype": "float32", "shape": [14],
            "names": [f"{s}_{c}" for s in ("left", "right") for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    info = {
        "codebase_version": "v2.1", "robot_type": "handheld_gelsight",
        "total_episodes": len(eps), "total_frames": total_frames,
        "total_tasks": len(TASK_ORDER), "total_videos": len(eps) * len(VIDEO_KEYS),
        "total_chunks": 1, "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{len(eps)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (STAGE_LR / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[lerobot] built {len(eps)} episodes, {total_frames:,} frames -> {STAGE_LR}", flush=True)


if __name__ == "__main__":
    main()
