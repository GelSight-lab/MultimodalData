"""Episode build: source H5 -> published videos + per-frame parquet.

    from react_preprocess import pipeline
    pipeline.build_episode(Path(".../episode_000.h5"), task="pushT")

Output layout (mirrors the HF dataset):

    <stage>/<task>/videos/<date>/<episode>/{view_*,tactile_*}.mp4
    <stage>/<task>/depth/<date>/<episode>/depth_*.mkv        (--with-depth)
    <stage>/<task>/meta/<date>/<episode>.parquet
    <stage>/<task>/meta/<date>/<episode>._detect.pt          (quality sidecar)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers BLOSC for the recorded files)
import numpy as np

from . import meta as meta_mod
from .config import CAM_STREAM, CHUNK, GEL_STREAM, SIDES, stage_dirs
from .encode import depth_writer, rgb_writer
from .h5io import open_episode
from .tactile import process_side


@dataclass
class BuildReport:
    episode: str
    status: str
    timestamped: bool = False
    duration_s: float = 0.0
    detail: str = ""

    def __str__(self):
        return f"{self.episode}: {self.status}" + (f" — {self.detail}" if self.detail else "")


def _encode_cameras(f, source, video_dir: Path) -> None:
    for cam_idx, name in CAM_STREAM.items():
        key = f"realsense/cam{cam_idx}/color"
        if key not in f:
            continue
        ds = f[key]                                   # (N, H, W, 3) BGR
        with rgb_writer(video_dir / f"{name}.mp4") as w:
            for s in range(0, source.T, CHUNK):
                e = min(s + CHUNK, source.T)
                w.write(ds[source.trim + s:source.trim + e])


def _encode_depth(f, source, depth_dir: Path) -> int:
    written = 0
    for cam_idx, name in CAM_STREAM.items():
        key = f"realsense/cam{cam_idx}/depth"
        if key not in f:
            continue
        ds = f[key]                                   # (N, H, W) uint16 mm
        out = depth_dir / f"{name.replace('view_', 'depth_')}.mkv"
        with depth_writer(out) as w:
            for s in range(0, source.T, CHUNK):
                e = min(s + CHUNK, source.T)
                w.write(np.asarray(ds[source.trim + s:source.trim + e], np.uint16))
        written += 1
    return written


def _object_pose(f, source) -> np.ndarray | None:
    """Nearest-timestamp pose of the manipulated object, if it was tracked."""
    from .h5io import cam_align_poses

    for body in (source.task, "object", "motherboard"):
        grp = f"optitrack/{body}"
        if grp in f and len(f[f"{grp}/timestamps"]) > 0:
            pose = cam_align_poses(source.trimmed_cam_ts,
                                   f[f"{grp}/timestamps"][:], f[f"{grp}/pose"][:]).copy()
            off = source.world_offset
            pose[:, 0] += off[0]; pose[:, 1] += off[1]; pose[:, 2] += off[2]
            return pose
    return np.full((source.T, 7), np.nan, np.float32)


def _write_detect_sidecar(path: Path, source, tactile) -> None:
    """Small torch sidecar consumed by the quality detector."""
    import torch

    torch.save({
        "timestamps": torch.from_numpy(source.trimmed_cam_ts.astype(np.float64)),
        "sensor_left_pose": torch.from_numpy(source.pose_left),
        "sensor_right_pose": torch.from_numpy(source.pose_right),
        "tactile_left_intensity": torch.from_numpy(tactile["left"].intensity),
        "tactile_right_intensity": torch.from_numpy(tactile["right"].intensity),
        "_contact_meta": {
            "trim_offset": int(source.trim),
            "active_sensors": source.active,
            "ref_p01_idx_left": int(tactile["left"].ref_index),
            "ref_p01_idx_right": int(tactile["right"].ref_index),
            "world_frame_offset_applied": list(source.world_offset),
            "tactile_timestamped": bool(source.timestamped),
            "tactile_stats": {s: tactile[s].stats for s in SIDES},
        },
    }, str(path))


def build_episode(h5_path: Path, task: str, force: bool = False,
                  with_depth: bool = False, encode_video: bool = True) -> BuildReport:
    """Build every published artefact for one source recording."""
    h5_path = Path(h5_path)
    t0 = time.time()
    try:
        source = open_episode(h5_path, task)
    except Exception as exc:                                    # noqa: BLE001
        return BuildReport(h5_path.stem, "FAIL", detail=f"unreadable ({exc})")

    video_dir, meta_dir = stage_dirs(task, source.date, source.episode)
    pq_path = meta_dir / f"{source.episode}.parquet"
    if pq_path.exists() and not force:
        return BuildReport(source.episode, "skipped", detail="already built")

    with h5py.File(str(h5_path), "r") as f:
        if encode_video:
            _encode_cameras(f, source, video_dir)
        tactile = {
            side: process_side(f, side, source.align[side],
                               video_dir / f"{GEL_STREAM[side]}.mp4",
                               encode=encode_video)
            for side in SIDES
        }
        obj_pose = _object_pose(f, source)
        if with_depth:
            depth_dir = video_dir.parent.parent.parent / "depth" / source.date / source.episode
            _encode_depth(f, source, depth_dir)

    table = meta_mod.build_table(source, tactile, obj_pose)
    meta_mod.write_table(table, pq_path)
    _write_detect_sidecar(meta_dir / f"{source.episode}._detect.pt", source, tactile)

    lstat = tactile["left"].stats
    detail = (f"T={source.T} "
              f"{'timestamped' if source.timestamped else 'legacy'} "
              f"tactile {lstat['effective_fps']:.1f}fps "
              f"({lstat['duplicate_ratio']*100:.0f}% dup)")
    return BuildReport(source.episode, "OK", source.timestamped,
                       time.time() - t0, detail)
