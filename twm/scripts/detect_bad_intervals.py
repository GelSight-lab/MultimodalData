"""Run failure-mode detection on per-episode .pt files (under
`processed/episodes/`) and emit `bad_frames.json` entries compatible with
the existing schema.

Detection rules (reverse-engineered from existing bad_frames.json on the
27 published episodes, using the thresholds stored in that file's header):

  tau_intensity     = 30.0   (mean L2 contact intensity)
  tau_velocity_mps  = 5.0    (per-frame translational velocity)
  tau_angular_rad_per_s = 15.0   (per-frame angular velocity)
  freeze_threshold_s = 0.25  (minimum freeze duration to flag as ot_loss)
  buffer_frames     = 3      (symmetric padding around each detected event)

Logic
-----
intensity_spikes:
    union over both sides of frames where tactile_<side>_intensity > tau_intensity.
    Each event's bracketing endpoints are flagged, then merged + padded.

pose_teleports_{L,R}:
    For each side, compute per-frame translational and angular velocity from
    sensor_<side>_pose (xyz + unit quaternion). Flag BOTH endpoint frames of
    any velocity sample that exceeds either threshold. Then merge + pad.

ot_loss_{L,R}:
    For each side, find contiguous runs of frames where sensor_<side>_pose
    is bit-identical (per-element |diff| < 1e-7) for at least
    freeze_threshold_s * 30 fps frames. These are the held-pose intervals
    produced when OptiTrack lost the rigid body and the recorder kept
    emitting the last sample. Then pad.

    NOTE: for the existing 27 episodes, these intervals were filtered by a
    cross-modal classifier (freeze_classify.py) that distinguished true OT
    loss from "real still" (deliberate hold). On NEW data we apply the
    conservative version — all bit-identical freezes are flagged as
    ot_loss. This may over-cut (false positives) but won't under-cut.

Outputs
-------
Per-episode dict of:
    {
        "n_frames":       T,
        "duration_s":     T / 30.0,
        "intensity_spikes": [[a, b], ...],
        "pose_teleports_L": [[a, b], ...],
        "pose_teleports_R": [[a, b], ...],
        "ot_loss_L":      [[a, b], ...],
        "ot_loss_R":      [[a, b], ...],
        "total_bad_frames": int,
        "bad_fraction":     float,
    }
and a trim_offsets dict {ep_key: trim_offset} pulled from _contact_meta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


TAU_INTENSITY        = 30.0
TAU_VELOCITY_MPS     = 5.0
TAU_ANGULAR_RAD_PS   = 15.0
FREEZE_THRESHOLD_S   = 0.25
FPS                  = 30.0
BUFFER_FRAMES        = 3
EPS_POSE_BIT         = 1e-7


def pad_and_merge(events: list[tuple[int, int]], T: int, buffer: int) -> list[list[int]]:
    """Each event is (a, b) inclusive. Pad ±buffer, clip to [0, T-1], merge."""
    if not events:
        return []
    padded = [(max(0, a - buffer), min(T - 1, b + buffer)) for a, b in events]
    padded.sort()
    merged: list[list[int]] = [list(padded[0])]
    for a, b in padded[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[a, b] for a, b in merged]


def detect_intensity_spikes(intens_L: np.ndarray, intens_R: np.ndarray, T: int) -> list[list[int]]:
    """Frames where either side's intensity > TAU_INTENSITY."""
    above = (intens_L > TAU_INTENSITY) | (intens_R > TAU_INTENSITY)
    idx = np.where(above)[0]
    events = [(int(i), int(i)) for i in idx]
    return pad_and_merge(events, T, BUFFER_FRAMES)


def detect_pose_teleports(pose: np.ndarray, T: int) -> list[list[int]]:
    """Per-frame translational + angular velocity exceedence; flag both endpoints."""
    if T < 2:
        return []
    xyz = pose[:, :3]
    quat = pose[:, 3:]
    qn = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    trans_vel = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FPS               # (T-1,)
    dot = np.abs((qn[:-1] * qn[1:]).sum(axis=1)).clip(-1.0, 1.0)
    ang_vel = 2.0 * np.arccos(dot) * FPS                                          # (T-1,)
    # Use AND (both translational AND angular), to match the published
    # rotation-aware detector. OR over-flags ordinary fast motion.
    flag = (trans_vel > TAU_VELOCITY_MPS) & (ang_vel > TAU_ANGULAR_RAD_PS)
    idx = np.where(flag)[0]
    events = [(int(i), int(i + 1)) for i in idx]
    return pad_and_merge(events, T, BUFFER_FRAMES)


def detect_pose_freezes(pose: np.ndarray, T: int) -> list[list[int]]:
    """Contiguous bit-identical pose runs >= FREEZE_THRESHOLD_S in duration."""
    if T < 2:
        return []
    same = np.zeros(T, dtype=bool)
    same[1:] = np.all(np.abs(np.diff(pose, axis=0)) < EPS_POSE_BIT, axis=1)
    min_frames = int(round(FREEZE_THRESHOLD_S * FPS))
    events = []
    i = 1
    while i < T:
        if not same[i]:
            i += 1
            continue
        j = i
        while j < T and same[j]:
            j += 1
        # Run includes the "anchor" at i-1 plus same[i..j-1]; len = j - (i - 1)
        run_a, run_b = i - 1, j - 1
        if (run_b - run_a + 1) >= min_frames:
            events.append((run_a, run_b))
        i = j
    # ot_loss intervals are stored raw (no padding) in the published bad_frames.json.
    return pad_and_merge(events, T, 0)


def detect_episode(pt_path: Path, ep_key: str) -> tuple[dict, int]:
    ep = torch.load(str(pt_path), weights_only=False, map_location="cpu")
    T = int(ep["timestamps"].shape[0])
    trim_offset = int(ep["_contact_meta"].get("trim_offset", 0))
    active = ep["_contact_meta"].get("active_sensors", ["left", "right"])

    intens_L = ep["tactile_left_intensity"].numpy()
    intens_R = ep["tactile_right_intensity"].numpy()
    pose_L   = ep["sensor_left_pose"].numpy()
    pose_R   = ep["sensor_right_pose"].numpy()

    intensity_spikes  = detect_intensity_spikes(intens_L, intens_R, T)
    pose_teleports_L  = detect_pose_teleports(pose_L, T) if "left"  in active else []
    pose_teleports_R  = detect_pose_teleports(pose_R, T) if "right" in active else []
    ot_loss_L         = detect_pose_freezes(pose_L, T)   if "left"  in active else []
    ot_loss_R         = detect_pose_freezes(pose_R, T)   if "right" in active else []

    # Total bad frames = |union of all intervals|
    mask = np.zeros(T, dtype=bool)
    for intervals in (intensity_spikes, pose_teleports_L, pose_teleports_R,
                      ot_loss_L, ot_loss_R):
        for a, b in intervals:
            mask[max(0, a):min(T, b + 1)] = True
    total_bad = int(mask.sum())

    entry = {
        "n_frames":         T,
        "duration_s":       round(T / FPS, 3),
        "intensity_spikes": intensity_spikes,
        "pose_teleports_L": pose_teleports_L,
        "pose_teleports_R": pose_teleports_R,
        "ot_loss_L":        ot_loss_L,
        "ot_loss_R":        ot_loss_R,
        "total_bad_frames": total_bad,
        "bad_fraction":     round(total_bad / T, 4) if T else 0.0,
    }
    return entry, trim_offset


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True,
                    help="Date subfolder under processed/episodes/<task>/, e.g. 2026-05-19")
    ap.add_argument("--task", default="motherboard")
    ap.add_argument("--episodes_root",
                    default="/media/yxma/Disk1/twm/processed/episodes")
    ap.add_argument("--out", default="/tmp/bad_frames_new.json",
                    help="Write the new entries dict here.")
    args = ap.parse_args()

    ep_dir = Path(args.episodes_root) / args.task / args.date
    pt_files = sorted(ep_dir.glob("episode_*.pt"))
    if not pt_files:
        print(f"No episode_*.pt under {ep_dir}", file=sys.stderr); sys.exit(1)

    print(f"Detecting failures on {len(pt_files)} episode(s) from {args.date}...", flush=True)
    episodes_entries: dict[str, dict] = {}
    trim_offsets: dict[str, int] = {}
    for pt in pt_files:
        ep_key = f"{args.date}/{pt.stem}"
        entry, trim = detect_episode(pt, ep_key)
        episodes_entries[ep_key] = entry
        if trim > 0:
            trim_offsets[ep_key] = trim
        nb = entry["total_bad_frames"]; T = entry["n_frames"]
        print(f"  {ep_key}: T={T:>5d}  spikes={len(entry['intensity_spikes']):>2d}  "
              f"telL={len(entry['pose_teleports_L']):>2d}  telR={len(entry['pose_teleports_R']):>2d}  "
              f"otL={len(entry['ot_loss_L']):>2d}  otR={len(entry['ot_loss_R']):>2d}  "
              f"bad={nb}/{T} ({100*nb/T:.1f}%)", flush=True)

    out = {
        "tau_intensity":           TAU_INTENSITY,
        "tau_velocity_mps":        TAU_VELOCITY_MPS,
        "tau_angular_rad_per_s":   TAU_ANGULAR_RAD_PS,
        "tau_opt_gap_s":           0.1,
        "freeze_threshold_s":      FREEZE_THRESHOLD_S,
        "buffer_frames":           BUFFER_FRAMES,
        "trim_offsets":            trim_offsets,
        "episodes":                episodes_entries,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
