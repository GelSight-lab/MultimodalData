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
from .config import FPS, STAGE_ROOT

MIN_SEGMENT_FRAMES = 16


def _sidecar_arrays(path: Path) -> tuple[dict, dict]:
    import torch

    ep = torch.load(str(path), weights_only=False, map_location="cpu")
    return ep, ep["_contact_meta"]


def episode_report(path: Path) -> tuple[dict, dict]:
    """Run every detector on one sidecar; returns (report, contact_meta)."""
    ep, cm = _sidecar_arrays(path)
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
    }

    mask = np.zeros(T, bool)
    for key in ("intensity_spikes", "pose_teleports_L", "pose_teleports_R",
                "ot_loss_L", "ot_loss_R"):
        for a, b in report[key]:
            mask[max(0, a):min(T, b + 1)] = True
    report["total_bad_frames"] = int(mask.sum())
    report["bad_fraction"] = round(report["total_bad_frames"] / T, 4) if T else 0.0
    return report, cm


def _bad_intervals(report: dict) -> list[tuple[int, int]]:
    return [(int(a), int(b))
            for key in ("intensity_spikes", "pose_teleports_L", "pose_teleports_R",
                        "ot_loss_L", "ot_loss_R")
            for a, b in report[key]]


def build_task(task: str, stage_root: Path = STAGE_ROOT,
               write: bool = True) -> dict:
    """Build the three curation files for one task."""
    out_dir = Path(stage_root) / task
    sidecars = sorted((out_dir / "meta").rglob("*._detect.pt"))
    if not sidecars:
        raise FileNotFoundError(f"no _detect.pt sidecars under {out_dir/'meta'}")

    episodes, segments, rows = {}, [], []
    for det in sidecars:
        date, stem = det.parent.name, det.name.replace("._detect.pt", "")
        key = f"{date}/{stem}"
        report, cm = episode_report(det)
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
        })

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
