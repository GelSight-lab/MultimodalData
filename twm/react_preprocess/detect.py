"""Detectors for unusable frames, and the clean-span complement.

Three failure modes are flagged per episode:

``intensity_spikes``   a GelSight reading far above anything contact produces —
                       usually the sensor being knocked or re-seated
``pose_teleports_*``   OptiTrack solving to the wrong marker set, which moves
                       the sensor implausibly far *and* rotates it implausibly
                       fast in a single frame
``ot_loss_*``          the tracker dropping out, which shows up as a run of
                       bit-identical poses rather than as missing samples
``tactile_freeze_*``   a GelSight dropping out, which shows up the same way in
                       its intensity trace: the recorder holds the last frame

Thresholds are the ones validated against the published motherboard
``bad_frames.json`` (25/27 episodes bit-identical).

Previously split across ``detect_bad_intervals.py`` and ``build_segments.py``
in ``twm/scripts/``; the latter has since been archived, so this module is now
the only live copy of ``find_clean_segments``.
"""
from __future__ import annotations

import numpy as np

from .config import FPS

class UnreadableVideo(RuntimeError):
    """A published stream that will not decode at all.

    Distinct from "this stream has corrupt frames": those are intervals to cut
    around, this is a file that cannot be read and must be rebuilt.
    """


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
#   magenta ROW FILL — pixels where R and B both sit far above G. A GelSight is
#           lit by three coloured LEDs from three sides; the colours it can
#           produce are bounded by that illumination, and MAGENTA is outside
#           it. A torn frame injects saturated magenta bands the sensor
#           physically cannot generate. Burst alone misses these when the band
#           is a minority of the frame: the 2026-05-11/ep003 right-sensor tear
#           moved the whole-frame mean only 21.6 against a floor of 25.
#
#           What is thresholded is the SHAPE of the magenta, not how much of it
#           there is and not how fast it arrived. Per frame, take the widest
#           single image row that is magenta, as a fraction of image width. A
#           lost or duplicated scanline is written edge to edge, so it fills
#           its rows; an object pressed into the gel produces its off-gamut rim
#           around the contact patch, which stays compact however deep it goes.
#           Measured: 0.725-1.000 for the nine labelled tears, 0.069-0.075 for
#           a probe tip. Corruption is row-structured because video is stored
#           in rows; contact is not, because gel is not.
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
#   * magenta by AREA: separated on the 5-episode labelled set, and an earlier
#     header here claimed "real contacts read exactly 0.0000". That claim was
#     FALSE and is retracted. The first full-release run found a probe tip at
#     2026-05-19/ep000 f=1383-1392 reading 0.0110 against a 0.0100 threshold,
#     while the weakest real tear reads only 0.0130 — 18% apart, no threshold
#     between them. Five episodes is not a validation set.
#   * magenta by ONSET ("appeared faster than gel can deform"): killed within
#     the hour by the same episode. The tip's area ramps 0.0069 -> 0.0110 at
#     0.0014/frame once pressed, which looked like clean separation from the
#     0.0130/frame of the weakest tear — but TOUCHDOWN at f=1358 goes 0.0000
#     -> 0.0064 in one frame. Contact is continuous only after it starts.
#     Note the shape of that mistake: the margin was measured in a window
#     centred on the deep press, which excluded the moment of first contact.
#     A window chosen around the phenomenon you already believe in will
#     confirm it.
# The two surviving rules each say something physical: burst, that a run of
# frames is bracketed by two hot boundaries; magenta row fill, that off-gamut
# colour is laid out in scanlines rather than around a contact patch.
CAM_SCAN_W, CAM_SCAN_H = 160, 120
CAM_SPIKE_ABS = 25.0          # floor, for a camera so still its median is ~0
CAM_SPIKE_REL = 18.0          # ... else REL * median + 5 (motion-adaptive)
CAM_BURST_MAX = 15            # two boundaries <= this far apart bracket a burst


def cam_burst_threshold(median: float) -> float:
    """How hot a frame-to-frame difference has to be to bound a burst.

    `CAM_SPIKE_REL` was 4.0, calibrated on the three STATIC scene cameras. For
    them the floor decides: their median is about 1.1, so 4*1.1+5 = 9.4 loses
    to 25.0, which sits 23x above their median and 2.6x above the worst frame
    of a whole toy episode. Scene motion cannot reach it.

    The wrist cameras ride the hand, and the same numbers put the threshold ON
    the 99th percentile of their own motion. Measured on
    toy/2026-09-17/episode_006:

        stream        median   p99    p99.9   max    old threshold
        view_middle     1.08    4.7     6.0    9.6       25.0
        wrist_left      3.28   25.3    33.3   40.0       25.0
        wrist_right     6.81   34.3    51.8   64.8       32.2

    A real tear is an outlier far from the bulk. These maxima are 1.6-2.0x the
    threshold with the percentiles rising smoothly through it, which is the
    tail of one distribution rather than a second one. The tail being cut is
    ordinary fast hand motion, and cutting it cost toy 9,923 of its 15,608 bad
    frames -- publishable yield 15.8% against 92.7%, 13.0 minutes against
    76.4, with three episodes reduced to no segments at all. One of them was
    reviewed by eye and had nothing wrong with it.

    The multiplier is set from the ratio each stream's WORST frame bears to
    its own median, measured across four episodes and two tasks rather than
    fitted to the one that prompted this:

        scene cameras   max/median  3.2 - 9.9   (12 streams)
        wrist cameras   max/median  9.0 - 17.3  (8 streams)

    A camera on the hand has a motion tail about twice as long as a fixed one,
    which is the whole asymmetry. 15.4 would put every observed wrist maximum
    just under the line; 18.0 leaves margin for a faster episode than any of
    these. The static case is untouched -- for a median of 1.1 the 25.0 floor
    still wins, and the scene cameras' worst frame in these episodes is 11.0.

    pushT has the same shape (wrist max/median 16.2 and 17.3) and was saved
    only by its lower median, where the floor decided. Its published yield was
    never affected, so this widens a threshold it never reached.

    A tear is still caught: it moves the frame mean by tens of units in one
    frame, against neighbours that differ by a few, so it stands far outside
    the motion tail on any camera.
    """
    return max(CAM_SPIKE_ABS, CAM_SPIKE_REL * float(median) + 5.0)
GEL_MAGENTA_MARGIN = 50       # R and B this far above G = off-illumination
GEL_MAGENTA_ROWFILL = 0.25    # widest magenta row, as a fraction of image width.
                              # Nine labelled tears: 0.725-1.000. A probe tip
                              # pressed into the gel: 0.069 at touchdown, 0.075
                              # at full depth. ~3x of margin on both sides.


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
    return pad_and_merge(_frozen_runs(same, T, min_frames), T, 0)


def _frozen_runs(same: np.ndarray, T: int, min_frames: int) -> list[tuple[int, int]]:
    """Inclusive spans of ``min_frames`` or more frames that repeat their
    predecessor. ``same[i]`` means frame ``i`` equals frame ``i-1``."""
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
    return events


def detect_tactile_freezes(intensity: np.ndarray, T: int) -> list[list[int]]:
    """Runs of bit-identical tactile intensity lasting at least
    ``FREEZE_THRESHOLD_S`` — a GelSight that stopped delivering frames.

    The recorder HOLDS a sensor's last frame at every tick until a new one
    arrives, so an outage is never missing data: it is the same frame written
    over and over, and the metrics computed from it repeat to the bit. That is
    the same signature ``detect_pose_freezes`` reads for OptiTrack.

    A GelSight runs at 15-18 Hz against a 30 Hz tick, so SHORT repeat runs are
    the normal state of every episode — 1 to 3 ticks, 4 at the slowest. Only
    the threshold separates that from an outage, and on the pushT 2026-09-10
    session the two populations are 4 and >= 144 ticks apart.

    Padded by ``BUFFER_FRAMES``, unlike the pose version: the driver's first
    frames after a reopen are underexposed (measured mean 55, then 73, against
    a settled 75), and those land just past the end of the freeze.
    """
    if T < 2:
        return []
    same = np.zeros(T, dtype=bool)
    same[1:] = np.diff(np.asarray(intensity, np.float64)[:T]) == 0.0
    return pad_and_merge(_frozen_runs(same, T, int(round(FREEZE_THRESHOLD_S * FPS))),
                         T, BUFFER_FRAMES)


def _video_stats(mp4) -> tuple[np.ndarray, np.ndarray]:
    """(frame-to-frame mean diff, widest magenta ROW FILL) for one video.

    Decoded at CAM_SCAN_W x CAM_SCAN_H in RGB. The diff is the MAX over
    channels per pixel: a GelSight tear swaps the gel's colour balance while
    barely moving the greyscale mean, so a grey decode measured the known
    2026-05-11/ep003 flicker at under half its amplitude.

    The second return is per frame: of all image rows, the largest fraction of
    that row which is off-gamut magenta. NOT the magenta area — see the module
    header. Area cannot tell a corrupt scanline from a probe tip pressed into
    the gel (0.0130 vs 0.0110); row fill can (0.725 vs 0.075), because the two
    differ in shape, not amount.
    """
    import subprocess
    W, H = CAM_SCAN_W, CAM_SCAN_H
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    raw = proc.stdout
    n = len(raw) // (W * H * 3)
    # A stream that will not decode is not a clean stream. This used to take
    # ffmpeg's stdout unconditionally, so a truncated file -- one ffmpeg was
    # killed before it could write the trailer, leaving "moov atom not found"
    # -- came back as zero bytes, scored as a 0-frame video, and reported NO
    # corruption. Two of the eight 2026-09-11 motherboard episodes were in
    # exactly that state (episode_004/wrist_left, episode_006/view_right) and
    # both passed curation silently; the cut caught them only because ffprobe
    # happens to be run before encoding. Counting files, or trusting a
    # detector that cannot fail, is not the same as checking the data.
    if proc.returncode != 0 or n == 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise UnreadableVideo(
            f"{mp4} did not decode ({'exit ' + str(proc.returncode)}): "
            f"{err[0] if err else 'no output'}")
    a = np.frombuffer(raw[:n * W * H * 3], np.uint8) \
          .reshape(n, H, W, 3).astype(np.int16)
    if n < 2:
        return np.zeros(0), np.zeros(max(n, 0))
    fmean = np.abs(np.diff(a, axis=0)).max(axis=3).mean(axis=(1, 2))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    magenta = (r > g + GEL_MAGENTA_MARGIN) & (b > g + GEL_MAGENTA_MARGIN)
    return fmean, magenta.mean(axis=2).max(axis=1)


def _magenta_bands(rowfill: np.ndarray) -> list[tuple[int, int]]:
    """Frames holding an off-gamut magenta band that spans its row.

    ``rowfill[k]`` is the widest single row of frame k that is magenta. A lost
    or duplicated scanline is written edge to edge, so it fills its rows; an
    object pressed into the gel produces its off-gamut rim around the contact
    patch, which is compact whatever its depth. Measured: 0.725-1.000 for the
    nine labelled tears, 0.069-0.075 for a probe tip at touchdown and at full
    depth. The threshold sits ~3x from each.
    """
    return [(int(i), int(i))
            for i in np.where(rowfill > GEL_MAGENTA_ROWFILL)[0]]


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
        fmean, rowfill = _video_stats(mp4)
        med = float(np.median(fmean)) if len(fmean) else 0.0
        ev = []
        # (1) BURST. Boundary pairs, not a single-frame spike test: the
        # obvious "differs from BOTH neighbours" scores a 9-frame flicker 0,
        # because its interior diffs are calm and each boundary is hot one
        # way only. Two hot boundaries close together bracket the anomalous
        # run; an unpaired boundary is scene motion and is ignored.
        thr = cam_burst_threshold(med)
        hot = [int(i) for i in np.where(fmean > thr)[0]]   # diff k -> k+1
        ev += [(b1 + 1, b2) for b1, b2 in zip(hot, hot[1:])
               if b2 - b1 <= CAM_BURST_MAX]
        # (2) MAGENTA ONSET, gel only. The absolute floor above is right for a
        # moving colour view and far too high for a static gel (median diff
        # 0.17), so a banded tear that moved the whole-frame mean by 21.6
        # slipped through. Dropping the floor catches it — and also flags
        # real CONTACT, whose intensity step looks the same in the mean. What
        # contact cannot do is produce colour outside the gel's own
        # illumination gamut FASTER THAN GEL DEFORMS. Thresholding the level
        # instead of the rise flagged a real probe tip at 0.0110 against a
        # weakest-tear level of 0.0130; see the module header.
        if is_gel and len(rowfill):
            ev += _magenta_bands(rowfill)
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
            "gel_magenta_rowfill": GEL_MAGENTA_ROWFILL,
        },
    }
