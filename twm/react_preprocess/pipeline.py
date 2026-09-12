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

import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers BLOSC for the recorded files)
import numpy as np

from . import meta as meta_mod
from . import repair
from .config import CAM_STREAM, CHUNK, GEL_STREAM, SIDES, WRIST_STREAM, stage_dirs
from twm.recorder.frames import decode_arducam

from .encode import depth_writer, rgb_writer
from .h5io import open_episode
from twm.wrist_tone import apply_tone_curve, camera_kind_from_serials, gamma_for_camera
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


def _rgb_plan(f, source) -> tuple[list, dict]:
    """The colour streams to encode: [(dataset, name, gamma_or_None)].

    ``gamma`` is None for a RealSense stream (published as recorded) and the
    wrist camera's tone exponent otherwise; see ``_encode_wrist``.
    """
    plan = [(f[k], name, None)
            for cam_idx, name in CAM_STREAM.items()
            if (k := f"realsense/cam{cam_idx}/color") in f]
    gammas = {}
    if "arducam" in f:
        gamma = gamma_for_camera(_wrist_camera_kind(f), source.task)
        for slot, name in WRIST_STREAM.items():
            if (k := f"arducam/{slot}/frames") in f:
                plan.append((f[k], name, gamma))
                gammas[slot] = round(float(gamma), 4)
    return plan, gammas


def _encode_rgb_single_pass(f, source, video_dir: Path) -> dict:
    """Every colour stream, from ONE traversal of the recording.

    Identical output to ``_encode_cameras`` + ``_encode_wrist``; the only
    difference is the order the file is read in, and that order is what the
    recording's layout makes expensive.

    The recorder writes one chunk per frame per stream, tick by tick, so on
    disk the file reads
    ``cam0₀ cam1₀ cam2₀ gelL₀ gelR₀ wristL₀ wristR₀ | cam0₁ cam1₁ …``.
    Measured on rope/2026-09-11/episode_011: each chunk is 0.62 MB and the
    next chunk of the SAME stream sits 3.87 MB further on. Encoding one stream
    at a time therefore seeks across that stride for every chunk, and does it
    once per stream.

    What that costs is TIME, not bytes. Measured on episode_008, reading 600
    frames of all five colour streams two ways (different frame ranges, so
    neither ran off the page cache):

        one pass, streams interleaved   1.81 GB read   190 s
        per stream, as built today      1.90 GB read   351 s

    The same data comes off the platter either way — 1.09x vs 1.14x of the
    bytes actually used, which is the same number. The per-stream order is
    simply seek-bound: 1.8x the wall time to move identical bytes. An earlier
    version of this comment claimed a ~6x read amplification; the measurement
    above says that was wrong.

    This is also why raising ``read_ahead_kb`` did nothing (measured
    128 KB -> 2 MB: 32.0 -> 30.3 MB/s aggregate, i.e. nothing). No bytes were
    being wasted for a bigger readahead to recover.

    Not yet the default. It changes the core build path and the difference is
    invisible in the output, so it wants a full session's worth of evidence
    before it becomes what every build does.
    """
    from contextlib import ExitStack

    plan, gammas = _rgb_plan(f, source)
    if not plan:
        return gammas
    with ExitStack() as stack:
        writers = [stack.enter_context(rgb_writer(video_dir / f"{name}.mp4"))
                   for _, name, _ in plan]
        for s in range(0, source.T, CHUNK):
            e = min(s + CHUNK, source.T)
            for (ds, _, gamma), w in zip(plan, writers):
                block = ds[source.trim + s:source.trim + e]
                if gamma is not None:
                    block = apply_tone_curve(
                        np.stack([decode_arducam(fr) for fr in block]), gamma)
                w.write(block)
    return gammas


def _encode_wrist(f, source, video_dir: Path) -> dict:
    """The two wrist streams, decoded here rather than at record time.

    Cut at `source.trim` like every other stream: a wrist video that starts
    at frame 0 while the tactile beside it starts at the trim plays ahead of
    it, and nothing in the file says so.

    Each stream is published through the power curve its camera generation
    calls for — the wrist cameras record much darker than the rest of the rig
    and cannot be fixed at the camera (no gain control, exposure already at the
    30 fps ceiling). See `tone` for where the exponents come from and why they
    are per camera rather than per episode. Returns the exponent used per slot
    so it reaches the episode metadata; 1.0 means the stream was published
    exactly as recorded.
    """
    if "arducam" not in f:
        return {}
    gamma = gamma_for_camera(_wrist_camera_kind(f), source.task)
    gammas = {}
    for slot, name in WRIST_STREAM.items():
        key = f"arducam/{slot}/frames"
        if key not in f:
            continue
        ds = f[key]
        with rgb_writer(video_dir / f"{name}.mp4") as w:
            for s in range(0, source.T, CHUNK):
                e = min(s + CHUNK, source.T)
                block = ds[source.trim + s:source.trim + e]
                # decode_arducam is a no-op on the raw-BGR episodes recorded
                # before 2026-09, so both layouts take this one path.
                w.write(apply_tone_curve(
                    np.stack([decode_arducam(fr) for fr in block]), gamma))
        gammas[slot] = round(float(gamma), 4)
    return gammas


def _wrist_camera_kind(f) -> str | None:
    """Which wrist camera this recording used, or None when it had none.

    The Arducams (serial-keyed, through 2026-09-09) and the generic USB pair
    (port-keyed, 2026-09-10 on) differ in field of view, exposure and
    distortion. A `has_wrist` boolean would merge them into one modality.
    """
    raw = f["metadata"].attrs.get("arducam_config", None)
    if not raw:
        return None
    try:
        cams = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return camera_kind_from_serials([c.get("serial") for c in cams])


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
    """Nearest-timestamp pose of the manipulated object, if this task has one.

    Which task tracks an object is DECLARED (`config.OBJECT_TRACKED_TASKS`),
    not searched for. This used to try the bodies (task, "object",
    "motherboard") in turn and take whichever the recording happened to
    contain -- so a pushT session recorded while Motive still had a
    `motherboard` body defined got that body's output as its object pose. On
    2026-09-09 that was 3 stray samples over 8.9 minutes, 157 mm apart, and
    the nearest-neighbour fill spread them across all 13484 rows. The column
    then reads 100% valid while every other pushT episode is all NaN, and
    nothing in the data says which one to believe.
    """
    from .h5io import cam_align_poses
    from .config import OBJECT_BODY, OBJECT_TRACKED_TASKS

    none = np.full((source.T, 7), np.nan, np.float32)
    if source.task not in OBJECT_TRACKED_TASKS:
        return none
    grp = f"optitrack/{OBJECT_BODY[source.task]}"
    if grp not in f or len(f[f"{grp}/timestamps"]) == 0:
        return none
    pose = cam_align_poses(source.trimmed_cam_ts,
                           f[f"{grp}/timestamps"][:], f[f"{grp}/pose"][:]).copy()
    off = source.world_offset
    pose[:, 0] += off[0]; pose[:, 1] += off[1]; pose[:, 2] += off[2]
    return pose


def _write_detect_sidecar(path: Path, source, tactile, extra_meta=None) -> None:
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
            **(extra_meta or {}),
        },
    }, str(path))


def build_episode(h5_path: Path, task: str, force: bool = False,
                  with_depth: bool = False, encode_video: bool = True,
                  auto_repair: bool = True, single_pass: bool = False) -> BuildReport:
    """Build every published artefact for one source recording.

    A recording that will not open is diagnosed and, for the one signature
    that is safely recoverable, repaired — see ``repair``. The recovered file
    then has to pass ``release_eligibility`` like anything else. It usually
    will not: recovery gets back what HDF5 had evicted from its metadata
    cache, which is the pixels and rarely the timestamps, and an episode whose
    timestamps were invented is worse than an episode that is missing.
    """
    h5_path = Path(h5_path)
    t0 = time.time()

    try:
        source = open_episode(h5_path, task)
    except Exception as exc:                                    # noqa: BLE001
        opened, note = repair.ensure_readable(h5_path, auto=auto_repair)
        if opened is None:
            return BuildReport(h5_path.stem, "FAIL",
                               detail=f"unreadable ({exc}) — {note}")
        ok, why = repair.release_eligibility(opened)
        if not ok:
            return BuildReport(
                h5_path.stem, "RECOVERED-NOT-PUBLISHABLE",
                detail=f"{note}, but it cannot become an episode: {why}")
        h5_path = opened
        source = open_episode(h5_path, task)

    video_dir, meta_dir = stage_dirs(task, source.date, source.episode)
    pq_path = meta_dir / f"{source.episode}.parquet"
    if pq_path.exists() and not force:
        return BuildReport(source.episode, "skipped", detail="already built")

    with h5py.File(str(h5_path), "r") as f:
        # The tone exponent is a property of (camera, task), not of this run,
        # so it is recorded even when the video is not re-encoded. Leaving it
        # empty under --meta-only made the metadata claim no curve was applied
        # to mp4s that were in fact published through one.
        kind = _wrist_camera_kind(f)
        wrist = {"wrist_camera": kind,
                 "wrist_tone_gamma": {slot: round(gamma_for_camera(kind, source.task), 4)
                                      for slot in WRIST_STREAM
                                      if f"arducam/{slot}/frames" in f}}
        if encode_video:
            if single_pass:
                wrist["wrist_tone_gamma"] = _encode_rgb_single_pass(
                    f, source, video_dir)
            else:
                _encode_cameras(f, source, video_dir)
                wrist["wrist_tone_gamma"] = _encode_wrist(f, source, video_dir)
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
    _write_detect_sidecar(meta_dir / f"{source.episode}._detect.pt", source, tactile,
                          extra_meta=wrist)

    lstat = tactile["left"].stats
    detail = (f"T={source.T} "
              f"{'timestamped' if source.timestamped else 'legacy'} "
              f"tactile {lstat['effective_fps']:.1f}fps "
              f"({lstat['duplicate_ratio']*100:.0f}% dup)")
    return BuildReport(source.episode, "OK", source.timestamped,
                       time.time() - t0, detail)
