"""Per-task curation indices built from the per-episode detect sidecars.

Produces three files next to the data:

``bad_frames.json``   detector thresholds plus every flagged interval
``segments.json``     the clean spans, indexed into episode video/parquet coords
``episodes.jsonl``    one row per episode

Frame ranges are inclusive ``[a, b]`` in episode-video coordinates, so
``frame_range`` indexes the MP4s and the parquet directly — no offset applies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import detect as D
from .config import FPS, SIDES, STAGE_ROOT

MIN_SEGMENT_FRAMES = 16

# react_preprocess copies poses straight from the HDF5, which is Y-up.
UP_AXIS_BUILT = "y"



def force_flag(parquet) -> bool:
    """Does this published parquet carry the force channel?

    Recorded in `episodes.jsonl` so a session published WITHOUT force says so.
    146 segments on the Hub carry it; a reader joining a force-free one against
    them gets NaN or a KeyError depending on the loader, and nothing in the
    data would explain why. An absence that is written down is a fact; one that
    is not is a hole.
    """
    import pyarrow.parquet as _pq
    from twm.dataset_layout import FORCE_COLUMNS
    try:
        names = set(_pq.read_schema(str(parquet)).names)
    except Exception:                                    # noqa: BLE001
        return False
    return bool(set(FORCE_COLUMNS) & names)

def _sidecar_arrays(path: Path) -> tuple[dict, dict]:
    import torch

    ep = torch.load(str(path), weights_only=False, map_location="cpu")
    cm = dict(ep["_contact_meta"])
    cm.setdefault("source", "sidecar")
    return ep, cm


def _parquet_arrays(path: Path) -> tuple[dict, dict]:
    """The same arrays, derived from the published parquet.

    32 of the 43 published motherboard episodes predate 2026-07 and their
    source H5 is deleted, so their `_detect.pt` can never be rebuilt. Every
    array the sidecar carries is also a column here, so the release can be
    curated from itself — and `source` records that it was, because a
    derivation must not be mistaken for a build artefact.
    """
    import pyarrow.parquet as pq
    import torch

    from .config import WORLD_OFFSET

    table = pq.read_table(str(path))
    need = ("timestamp", "sensor_left_pose", "sensor_right_pose",
            "tactile_left_intensity", "tactile_right_intensity")
    missing = [c for c in need if c not in table.column_names]
    if missing:
        raise KeyError(f"{path.parent.name}/{path.stem}: no sidecar, and the "
                       f"parquet cannot stand in for one — missing {missing}")

    def col(name):
        return np.asarray(table[name].to_pylist(), dtype=np.float64)

    ep = {name: torch.from_numpy(col(src)) for name, src in (
        ("timestamps", "timestamp"),
        ("sensor_left_pose", "sensor_left_pose"),
        ("sensor_right_pose", "sensor_right_pose"),
        ("tactile_left_intensity", "tactile_left_intensity"),
        ("tactile_right_intensity", "tactile_right_intensity"))}
    trim = (int(np.asarray(table["source_h5_frame"].to_pylist())[0])
            if "source_h5_frame" in table.column_names else 0)
    # A side counts as tracked when its pose is not all-NaN, the same thing
    # `_object_pose` writes for a body that was never broadcast.
    active = [s for s in SIDES
              if np.isfinite(col(f"sensor_{s}_pose")).any()]
    task, date = path.parents[2].name, path.parent.name
    return ep, {"source": "parquet", "trim_offset": trim,
                "active_sensors": active,
                "world_frame_offset_applied": list(
                    WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))),
                "tactile_timestamped": date > "2026-06-18"}


BAD_KEYS = ("intensity_spikes", "pose_teleports_L", "pose_teleports_R",
            "ot_loss_L", "ot_loss_R", "cam_corruption", "tactile_corruption",
            "tactile_freeze_L", "tactile_freeze_R")


def episode_report(path: Path, video_dir: Path | None = None) -> tuple[dict, dict]:
    """Run every detector on one sidecar; returns (report, contact_meta).

    `video_dir` points at the episode's published videos; without it the two
    video-corruption detectors are skipped — which is the pre-2026-08 state,
    where nothing in curation could see a torn camera frame.
    """
    ep, cm = (_sidecar_arrays(path) if str(path).endswith("._detect.pt")
              else _parquet_arrays(path))
    T = int(ep["timestamps"].shape[0])
    active = cm.get("active_sensors", ["left", "right"])
    pose_l = ep["sensor_left_pose"].numpy()
    pose_r = ep["sensor_right_pose"].numpy()

    report = {
        "n_frames": T,
        "duration_s": round(T / FPS, 3),
        "intensity_spikes": D.detect_intensity_spikes(
            ep["tactile_left_intensity"].numpy(),
            ep["tactile_right_intensity"].numpy(), T),
        "pose_teleports_L": D.detect_pose_teleports(pose_l, T) if "left" in active else [],
        "pose_teleports_R": D.detect_pose_teleports(pose_r, T) if "right" in active else [],
        "ot_loss_L": D.detect_pose_freezes(pose_l, T) if "left" in active else [],
        "ot_loss_R": D.detect_pose_freezes(pose_r, T) if "right" in active else [],
        # A held GelSight frame is not missing data the reader can see: it is
        # the previous frame again, with the previous frame's metrics.
        "tactile_freeze_L": D.detect_tactile_freezes(
            ep["tactile_left_intensity"].numpy(), T),
        "tactile_freeze_R": D.detect_tactile_freezes(
            ep["tactile_right_intensity"].numpy(), T),
        "cam_corruption": [],
        "tactile_corruption": [],
    }
    if video_dir is not None and Path(video_dir).is_dir():
        # From the EPISODE name, not by string surgery on the input path:
        # `path` is the parquet when there is no sidecar, and a replace that
        # does not match left `cache` pointing at the parquet itself, which
        # then went to json.loads.
        stem = path.name.replace("._detect.pt", "").replace(".parquet", "")
        cache = path.with_name(f"{stem}._camscan.json")
        report.update(D.detect_video_corruption(video_dir, T, cache=cache))

    mask = np.zeros(T, bool)
    for key in BAD_KEYS:
        for a, b in report[key]:
            mask[max(0, a):min(T, b + 1)] = True
    report["total_bad_frames"] = int(mask.sum())
    report["bad_fraction"] = round(report["total_bad_frames"] / T, 4) if T else 0.0
    return report, cm


def _bad_intervals(report: dict) -> list[tuple[int, int]]:
    return [(int(a), int(b))
            for key in BAD_KEYS
            for a, b in report[key]]


def build_task(task: str, stage_root: Path = STAGE_ROOT,
               write: bool = True) -> dict:
    """Build the three curation files for one task."""
    out_dir = Path(stage_root) / task
    # Discovery is by PARQUET, not by sidecar. Sidecars only exist for
    # episodes built by the current pipeline, so discovering by sidecar
    # quietly rebuilds the indices from whatever subset happens to have one:
    # on the real motherboard tree that was 3 of 35, and the 32 dropped rows
    # were not noticed until the force export refused an episode two steps
    # later. Refuse instead, naming what is missing.
    parquets = sorted((out_dir / "meta").rglob("episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no episode parquet under {out_dir/'meta'}")
    # A sidecar when there is one, the parquet itself when there is not: the
    # pre-2026-07 episodes' source H5 is deleted, so their sidecar can never be
    # rebuilt, and refusing on that account made the whole task uncurateable.
    # Nothing is dropped either way — that protection is what the refusal was
    # for, and `episode_report` now derives the same arrays from the release.
    sidecars = []
    for pq in parquets:
        det = pq.with_suffix("")
        det = det.with_name(det.name + "._detect.pt")
        sidecars.append(det if det.is_file() else pq)

    # An existing row's up_axis is preserved: react_preprocess itself writes
    # no axis convention, the published rows carry "z" from the Z-up staging
    # step, and dropping the field makes calib_epoch read every Z-up world
    # offset as Y-up.
    prior = {}
    jsonl = out_dir / "episodes.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                prior[r.get("episode")] = r

    episodes, segments, rows = {}, [], []
    for det in sidecars:
        date = det.parent.name
        stem = det.name.replace("._detect.pt", "").replace(".parquet", "")
        key = f"{date}/{stem}"
        report, cm = episode_report(
            det, video_dir=out_dir / "videos" / date / stem)
        episodes[key] = report
        T = report["n_frames"]

        n_seg = 0
        for a, b in D.find_clean_segments(T, _bad_intervals(report)):
            length = b - a + 1
            if length < MIN_SEGMENT_FRAMES:
                continue
            segments.append({
                "task": task, "source_episode": key, "segment_idx": n_seg,
                "frame_range": [a, b], "n_frames": length,
                "duration_s": round(length / FPS, 3),
            })
            n_seg += 1

        rows.append({
            "episode": key, "date": date, "n_frames": T,
            "duration_s": report["duration_s"],
            "active_sensors": cm.get("active_sensors", ["left", "right"]),
            "trim_offset": int(cm.get("trim_offset", 0)),
            "world_frame_offset": cm.get("world_frame_offset_applied", [0.0, 0.0, 0.0]),
            "n_segments": n_seg,
            "total_bad_frames": report["total_bad_frames"],
            # Whether this unit carries the force channel. Recorded because the
            # absence has to be a FACT a reader can act on: 146 published
            # segments have it, and one that does not gives NaN or a KeyError
            # depending on the loader, with nothing in the data saying why.
            "force": force_flag(out_dir / "meta" / date / f"{stem}.parquet"),
            # Which wrist camera, not whether: the Arducam and USB pairs are
            # different optics. None means the session predates them.
            "wrist_camera": cm.get("wrist_camera"),
            # The power curve each wrist stream was published through (1.0 =
            # as recorded). A declared photometric change; see `tone`.
            "wrist_tone_gamma": cm.get("wrist_tone_gamma", {}),
        })
        # Declared, always. react_preprocess copies poses straight out of the
        # HDF5, which is Y-up as recorded; an existing row's value wins because
        # a later stage may have rotated that episode. A row with no up_axis at
        # all makes every consumer guess, and the guess put the DexForce target
        # hundreds of mm off for 2026-09-09.
        rows[-1]["up_axis"] = prior.get(key, {}).get("up_axis", UP_AXIS_BUILT)

    total = sum(e["n_frames"] for e in episodes.values())
    bad = sum(e["total_bad_frames"] for e in episodes.values())
    seg_frames = sum(s["n_frames"] for s in segments)

    bad_frames = {
        "task": task, **D.thresholds(),
        "summary": {
            "n_episodes": len(episodes), "total_frames": total,
            "total_bad_frames": bad,
            "bad_fraction_overall": round(bad / total, 4) if total else 0.0,
        },
        "episodes": episodes,
    }
    segments_doc = {
        "task": task, "schema": "segments_v2_video",
        "description": ("Each entry indexes a contiguous clean span within an "
                        "episode's videos (data/<task>/videos/<date>/episode_NNN/*.mp4) "
                        "and parquet. frame_range is [a,b] inclusive in "
                        "episode-video frame coords."),
        "n_segments": len(segments), "total_frames": seg_frames,
        "total_duration_min": round(seg_frames / FPS / 60, 2),
        "min_segment_frames_kept": MIN_SEGMENT_FRAMES,
        "segments": sorted(segments, key=lambda s: (s["source_episode"], s["segment_idx"])),
    }

    if write:
        (out_dir / "bad_frames.json").write_text(json.dumps(bad_frames, indent=2))
        (out_dir / "segments.json").write_text(json.dumps(segments_doc, indent=2))
        with open(out_dir / "episodes.jsonl", "w") as fh:
            for row in sorted(rows, key=lambda r: r["episode"]):
                fh.write(json.dumps(row) + "\n")

    return {
        "task": task, "episodes": len(episodes), "segments": len(segments),
        "total_frames": total, "bad_frames": bad,
        "bad_fraction": bad / total if total else 0.0,
        "clean_frames": seg_frames, "clean_minutes": seg_frames / FPS / 60,
    }
