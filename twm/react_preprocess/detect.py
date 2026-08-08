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

# VIDEO corruption. Everything above reads the sidecar scalars; none of it can
# see a torn or garbage frame, which is why "partially corrupted images"
# survived curation unflagged.
#
# ONE signature, on a 160x120 RGB decode with the per-pixel diff taken as the
# MAX over channels (a GelSight flicker swaps the gel's colour balance while
# barely moving the greyscale mean — a grey decode measured the known
# 2026-05-11/ep003 flicker at under half its amplitude):
#
#   burst — two hot frame-to-frame boundaries close together bracket an
#           anomalous run. The obvious "frame differs from BOTH neighbours"
#           test scores a 9-frame flicker exactly 0: its interior diffs are
#           calm and each boundary is hot one way only.
#
#   magenta — pixels where R and B both sit far above G. A GelSight is lit by
#           three coloured LEDs from three sides; the colours it can produce
#           are bounded by that illumination, and MAGENTA is outside it. A
#           torn frame injects saturated magenta bands the sensor physically
#           cannot generate. Burst alone misses these when the band is a
#           minority of the frame: the 2026-05-11/ep003 right-sensor tear
#           moved the whole-frame mean only 21.6 against a floor of 25.
#           This is a physical constraint, not a tuned threshold, and it
#           separates absolutely: measured over the labelled set, real tears
#           read 0.133-0.748 of the frame and real contacts read exactly
#           0.0000 — including sharp edges and tips, whose strong orange/blue
#           photometric gradients defeated every amplitude-based test.
#
# Five discriminators were tried and killed by measurement, not opinion. Each
# is recorded because "we tried X" is the cheapest thing to lose and the most
# expensive to rediscover:
#   * row-band tear ("a band changed hard while the rest stayed put"): 8
#     intervals over the whole release, every one a fast arm sweep on a colour
#     view — 56 frames of good data lost, zero corruption caught.
#   * reversion ("same rows hot going in and coming out"): 0.000 on true AND
#     false positives alike; a corrupt band persists several frames, so it
#     never reverts within one frame-diff.
#   * dropping burst's absolute floor so it adapts to a static gel: catches
#     the tear, and also every real CONTACT — the intensity step is identical
#     in the whole-frame mean.
#   * closure ("the scene returns after the burst"): 3.07 for a real tear vs
#     5.60 and 8.04 for two real contacts. Backwards, and overlapping.
#   * off-gamut channel spread, by level (92 intervals in one episode) and by
#     step (7 intervals, 4 of them sharp-edge contacts — a knife edge in the
#     gel steps exactly like a tear).
# Magenta needs neither a step nor a pairing rule, because contact scores
# exactly zero: it is a statement about the sensor's physics, not a threshold
# fitted to this release.
CAM_SCAN_W, CAM_SCAN_H = 160, 120
CAM_SPIKE_ABS = 25.0          # a frame-to-frame diff this hot is a boundary
CAM_SPIKE_REL = 4.0           # ... and > REL * median + 5 (motion-adaptive)
CAM_BURST_MAX = 15            # two boundaries <= this far apart bracket a burst
GEL_MAGENTA_MARGIN = 50       # R and B this far above G = off-illumination
GEL_MAGENTA_FRAC = 0.01       # 13x below the weakest real tear (0.133)


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


def _video_stats(mp4) -> tuple[np.ndarray, np.ndarray]:
    """(frame-to-frame mean diff, magenta pixel fraction) for one video.

    Decoded at CAM_SCAN_W x CAM_SCAN_H in RGB. The diff is the MAX over
    channels per pixel: a GelSight tear swaps the gel's colour balance while
    barely moving the greyscale mean, so a grey decode measured the known
    2026-05-11/ep003 flicker at under half its amplitude.
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
    if n < 2:
        return np.zeros(0), np.zeros(max(n, 0))
    fmean = np.abs(np.diff(a, axis=0)).max(axis=3).mean(axis=(1, 2))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    magenta = (r > g + GEL_MAGENTA_MARGIN) & (b > g + GEL_MAGENTA_MARGIN)
    return fmean, magenta.mean(axis=(1, 2))


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
        is_gel = "tactile" in mp4.name
        fam = "tactile_corruption" if is_gel else "cam_corruption"
        fmean, chroma = _video_stats(mp4)
        med = float(np.median(fmean)) if len(fmean) else 0.0
        ev = []
        # (1) BURST. Boundary pairs, not a single-frame spike test: the
        # obvious "differs from BOTH neighbours" scores a 9-frame flicker 0,
        # because its interior diffs are calm and each boundary is hot one
        # way only. Two hot boundaries close together bracket the anomalous
        # run; an unpaired boundary is scene motion and is ignored.
        thr = max(CAM_SPIKE_ABS, CAM_SPIKE_REL * med + 5)
        hot = [int(i) for i in np.where(fmean > thr)[0]]   # diff k -> k+1
        ev += [(b1 + 1, b2) for b1, b2 in zip(hot, hot[1:])
               if b2 - b1 <= CAM_BURST_MAX]
        # (2) MAGENTA, gel only. The absolute floor above is right for a
        # moving colour view and far too high for a static gel (median diff
        # 0.17), so a banded tear that moved the whole-frame mean by 21.6
        # slipped through. Dropping the floor catches it — and also flags
        # real CONTACT, whose intensity step looks the same in the mean. What
        # contact cannot do is produce colours outside the gel's own
        # illumination gamut, so the discriminator is off-gamut area, not
        # amplitude.
        if is_gel and len(chroma):
            ev += [(int(i), int(i))
                   for i in np.where(chroma > GEL_MAGENTA_FRAC)[0]]
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
            "gel_magenta_margin": GEL_MAGENTA_MARGIN,
            "gel_magenta_frac": GEL_MAGENTA_FRAC,
        },
    }
