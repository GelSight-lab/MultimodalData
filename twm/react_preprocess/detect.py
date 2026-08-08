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

# Camera/tactile VIDEO corruption. Everything above reads the sidecar scalars;
# none of it can see a torn or garbage camera frame, which is why "partially
# corrupted images" survived curation unflagged. Two signatures on a 160x120
# grayscale decode:
#   spike — a frame that differs sharply from BOTH neighbours (full-frame
#           garbage; real motion differs from one side only)
#   tear  — the per-row diff profile is a step: a contiguous band of rows
#           changed hard while the rest of the frame stayed put (partial
#           corruption from a torn capture)
CAM_SCAN_W, CAM_SCAN_H = 160, 120
CAM_SPIKE_ABS = 25.0          # a frame-to-frame diff this hot is a boundary
CAM_SPIKE_REL = 4.0           # ... and > REL * median + 5 (motion-adaptive)
CAM_BURST_MAX = 15            # two boundaries <= this far apart bracket a burst
CAM_TEAR_ROW_HOT = 30.0       # a row counts as "changed hard"
CAM_TEAR_HOT_FRAC = 0.15      # >=15% of rows hot ...
CAM_TEAR_COLD_FRAC = 0.40     # ... while >=40% of rows essentially static (<5)


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


def _video_diff_stats(mp4) -> tuple[np.ndarray, np.ndarray]:
    """(frame mean diff, per-row mean diff) for one video, low-res RGB.

    Per pixel the diff is the MAX over channels: a GelSight flicker swaps the
    gel's colour balance (magenta <-> green) while barely moving the greyscale
    mean, so a grey decode measured the 2026-05-11/episode_003 flicker at
    under half its chromatic amplitude and missed it entirely.
    """
    import subprocess
    W, H = CAM_SCAN_W, CAM_SCAN_H
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    n = len(raw) // (W * H * 3)
    a = np.frombuffer(raw[:n * W * H * 3], np.uint8) \
          .reshape(n, H, W, 3).astype(np.int16)
    d = (np.abs(np.diff(a, axis=0)).max(axis=3) if n > 1
         else np.zeros((0, H, W), np.int16))
    return d.mean(axis=(1, 2)), d.mean(axis=2)


def detect_video_corruption(video_dir, T: int, cache=None) -> dict[str, list[list[int]]]:
    """Corrupted frames in the episode's published videos, per stream family.

    Returns ``{"cam_corruption": [...], "tactile_corruption": [...]}`` in
    episode-frame coords. Decoding every stream costs ~10 s per episode, so
    the result is cached beside the sidecar keyed on the videos' mtimes.
    """
    import json
    from pathlib import Path

    video_dir = Path(video_dir)
    mp4s = sorted(video_dir.glob("*.mp4"))
    # The stamp includes the detector settings: a cache keyed on video mtimes
    # alone would keep serving results from a superseded algorithm — the same
    # silent-staleness failure this repo keeps re-earning gates against.
    stamp = {p.name: p.stat().st_mtime for p in mp4s}
    stamp["_thresholds"] = thresholds()["video_corruption"]
    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            d = json.loads(cache.read_text())
            if d.get("stamp") == stamp:
                return {k: d[k] for k in ("cam_corruption", "tactile_corruption")}

    out = {"cam_corruption": [], "tactile_corruption": []}
    for mp4 in mp4s:
        fam = "tactile_corruption" if "tactile" in mp4.name else "cam_corruption"
        fmean, rows = _video_diff_stats(mp4)
        n = len(fmean) + 1
        med = float(np.median(fmean)) if len(fmean) else 0.0
        thr = max(CAM_SPIKE_ABS, CAM_SPIKE_REL * med + 5)
        ev = []
        # Boundary-pair ("burst") logic instead of the single-frame spike
        # test: `fmean[i-1] hot AND fmean[i] hot` only fires when the anomaly
        # is exactly one frame long. A sustained flicker — 9 frames of
        # magenta at 2026-05-11/episode_003 — has calm interior diffs and one
        # hot boundary at each end, so the single-frame test scored it 0.
        # Two hot boundaries at most CAM_BURST_MAX frames apart bracket the
        # anomalous run; an unpaired boundary is scene motion and is ignored.
        hot = [int(i) for i in np.where(fmean > thr)[0]]   # diff k -> k+1
        for b1, b2 in zip(hot, hot[1:]):
            if b2 - b1 <= CAM_BURST_MAX:
                ev.append((b1 + 1, b2))
        for i in range(1, n - 1):                          # torn single frames
            r = rows[i - 1]
            if ((r > CAM_TEAR_ROW_HOT).mean() >= CAM_TEAR_HOT_FRAC
                    and (r < 5).mean() >= CAM_TEAR_COLD_FRAC):
                ev.append((i, i))
        out[fam] += ev
    out = {k: pad_and_merge(v, T, BUFFER_FRAMES) for k, v in out.items()}
    if cache is not None:
        cache.write_text(json.dumps({"stamp": stamp, **out}))
    return out


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
        "video_corruption": {
            "scan_res": [CAM_SCAN_W, CAM_SCAN_H], "metric": "rgb_max_channel",
            "spike_abs": CAM_SPIKE_ABS, "spike_rel": CAM_SPIKE_REL,
            "burst_max_frames": CAM_BURST_MAX,
            "tear_row_hot": CAM_TEAR_ROW_HOT,
            "tear_hot_frac": CAM_TEAR_HOT_FRAC,
            "tear_cold_frac": CAM_TEAR_COLD_FRAC,
        },
    }
