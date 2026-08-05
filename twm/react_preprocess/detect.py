"""Detectors for unusable frames, and the clean-span complement.

Three failure modes are flagged per episode:

``intensity_spikes``   a GelSight reading far above anything contact produces —
                       usually the sensor being knocked or re-seated
``pose_teleports_*``   OptiTrack solving to the wrong marker set, which moves
                       the sensor implausibly far *and* rotates it implausibly
                       fast in a single frame
``ot_loss_*``          the tracker dropping out, which shows up as a run of
                       bit-identical poses rather than as missing samples

Thresholds are the ones validated against the published motherboard
``bad_frames.json`` (25/27 episodes bit-identical).

Previously split across ``detect_bad_intervals.py`` and ``build_segments.py``
in ``twm/scripts/``; the latter has since been archived, so this module is now
the only live copy of ``find_clean_segments``.
"""
from __future__ import annotations

import numpy as np

from .config import FPS

TAU_INTENSITY = 30.0
TAU_VELOCITY_MPS = 5.0
TAU_ANGULAR_RAD_PS = 15.0
FREEZE_THRESHOLD_S = 0.25
BUFFER_FRAMES = 3
EPS_POSE_BIT = 1e-7


def merge_intervals(events, gap: int = 1) -> list[list[int]]:
    """Merge inclusive ``(a, b)`` intervals that touch or overlap."""
    if not events:
        return []
    ordered = sorted((int(a), int(b)) for a, b in events)
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def pad_and_merge(events, T: int, buffer: int) -> list[list[int]]:
    """Pad each interval by ``±buffer``, clip to ``[0, T-1]``, then merge."""
    if not events:
        return []
    return merge_intervals([(max(0, a - buffer), min(T - 1, b + buffer))
                            for a, b in events])


def detect_intensity_spikes(intens_l: np.ndarray, intens_r: np.ndarray,
                            T: int) -> list[list[int]]:
    """Frames where either sensor reads above ``TAU_INTENSITY``."""
    above = (intens_l > TAU_INTENSITY) | (intens_r > TAU_INTENSITY)
    return pad_and_merge([(int(i), int(i)) for i in np.where(above)[0]],
                         T, BUFFER_FRAMES)


def detect_pose_teleports(pose: np.ndarray, T: int) -> list[list[int]]:
    """Frames whose pose jump is implausible in translation *and* rotation.

    The conjunction matters: ordinary fast motion trips the translational
    threshold on its own, so requiring both is what separates a tracking error
    from a quick reach.
    """
    if T < 2:
        return []
    xyz, quat = pose[:, :3], pose[:, 3:]
    qn = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    trans_vel = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FPS
    dot = np.abs((qn[:-1] * qn[1:]).sum(axis=1)).clip(-1.0, 1.0)
    ang_vel = 2.0 * np.arccos(dot) * FPS
    flag = (trans_vel > TAU_VELOCITY_MPS) & (ang_vel > TAU_ANGULAR_RAD_PS)
    return pad_and_merge([(int(i), int(i + 1)) for i in np.where(flag)[0]],
                         T, BUFFER_FRAMES)


def detect_pose_freezes(pose: np.ndarray, T: int) -> list[list[int]]:
    """Runs of bit-identical pose lasting at least ``FREEZE_THRESHOLD_S``.

    OptiTrack repeats its last solution when it loses the marker set, so a
    frozen pose is track loss rather than genuine stillness — a real hold still
    jitters in the last decimal places.

    Reported unpadded, matching the published ``bad_frames.json``.
    """
    if T < 2:
        return []
    same = np.zeros(T, dtype=bool)
    same[1:] = np.all(np.abs(np.diff(pose, axis=0)) < EPS_POSE_BIT, axis=1)
    min_frames = int(round(FREEZE_THRESHOLD_S * FPS))

    events, i = [], 1
    while i < T:
        if not same[i]:
            i += 1
            continue
        j = i
        while j < T and same[j]:
            j += 1
        # the run includes the anchor frame at i-1 that the copies match
        run_a, run_b = i - 1, j - 1
        if (run_b - run_a + 1) >= min_frames:
            events.append((run_a, run_b))
        i = j
    return pad_and_merge(events, T, 0)


def find_clean_segments(T: int, bad_intervals) -> list[tuple[int, int]]:
    """Complement of the bad intervals: inclusive ``[a, b]`` clean spans."""
    segments, prev_end = [], -1
    for a, b in merge_intervals(bad_intervals):
        if a > prev_end + 1:
            segments.append((prev_end + 1, a - 1))
        prev_end = max(prev_end, b)
    if prev_end < T - 1:
        segments.append((prev_end + 1, T - 1))
    return segments


def thresholds() -> dict:
    """The detector settings, for recording alongside the results."""
    return {
        "tau_intensity": TAU_INTENSITY,
        "tau_velocity_mps": TAU_VELOCITY_MPS,
        "tau_angular_rad_per_s": TAU_ANGULAR_RAD_PS,
        "freeze_threshold_s": FREEZE_THRESHOLD_S,
        "buffer_frames": BUFFER_FRAMES,
    }
