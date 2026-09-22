"""Repair the short tracking glitches OptiTrack leaves in the pose stream.

OptiTrack occasionally solves the wrong marker correspondence for a frame or
three, swinging the rigid body's orientation far away and snapping it back.
Measured on the published motherboard/2026-09-11/episode_006_seg00, right
sensor: 24 such excursions, the worst rotating 168.7 degrees out and back while
its two neighbours sit 1.0 degrees apart. The left sensor of the same recording
is clean -- a per-marker-set solve failure, not a rig-wide one.

`detect.detect_pose_teleports` catches none of these. It requires translation
AND rotation to exceed threshold, for a stated reason: "ordinary fast motion
trips the translational threshold on its own, so requiring both is what
separates a tracking error from a quick reach". That is true of a reach, and it
leaves the detector blind to the rotation-only case, which is the common one.

THE SIGNATURE USED HERE IS THE RETURN, not the magnitude. Frame i is far from
i-1 and far from i+k, while i-1 and i+k are close to each other. Real motion
goes somewhere and stays; a solve failure goes and comes back. That test needs
no velocity threshold, so it does not have to be retuned per task or per speed.
"""
from __future__ import annotations

import numpy as np

# How far ahead the detector looks for the RETURN. It must exceed the longest
# loss to be found: a window shorter than the loss never has clean frames on
# both sides, so a 9-frame drift was invisible at 5 while the wandering ones
# showed up only as spurious fragments. The longest measured across motherboard
# including the archives is 25 frames (0.83 s), so 40 leaves headroom.
#
# This is NOT a cap on what may be filled. It used to be both, and the double
# duty is what hid the long losses.
MAX_EXCURSION_FRAMES = 40

# Only OVERLAPPING or touching detections fuse. A wider gap chains independent
# glitches across good tracking: on 2026-05-11/episode_016_seg01 a gap of 5
# merged ~22 separate jumps spread over 5 s into one 150-frame span whose
# median per-frame change was 1.2 degrees -- 128 good frames would have been
# interpolated away to fix 22 bad ones. With the detection window wide enough
# to span a real loss outright, fusing is only needed to union overlapping
# candidates, which touching covers.
MERGE_GAP_FRAMES = 1

# How far the pose must swing to count as an excursion at all. Well above the
# measured p99 of 1.8 rad/s (~3.4 degrees per frame) on a clean stream.
MIN_EXCURSION_RAD = np.radians(12.0)

# The endpoints must agree this much better than the excursion is large. At
# 0.35 an excursion of 90 degrees needs its neighbours within 31 degrees --
# the measured glitches return to within 1 degree, and real motion does not
# return at all.
RETURN_RATIO = 0.35

# A single-frame orientation change this large is not motion. Measured on a
# clean stream the per-frame change sits at p99 ~ 3.4 degrees; a solve failure
# jumps tens of degrees between two consecutive frames.
MIN_JUMP_RAD = np.radians(12.0)

# Longest run that gets FILLED. Beyond it the span is reported and the poses
# are left exactly as measured: half a second is already a lot of invented
# motion, and the longest runs the detector finds are stretches where tracking
# was lost repeatedly rather than a single recoverable glitch.
REPAIR_LIMIT_FRAMES = 15


def _unit(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _angle(a, b):
    """Rotation angle between two quaternions, sign-insensitive."""
    d = np.abs(np.sum(a * b, axis=-1)).clip(-1.0, 1.0)
    return 2.0 * np.arccos(d)


def find_excursions(pose: np.ndarray) -> list[tuple[int, int]]:
    """`(start, n_frames)` for every short swing-out-and-back.

    Scanned shortest-first so a 1-frame glitch is reported as 1 frame rather
    than as the leading edge of a longer span.
    """
    if len(pose) < 3:
        return []
    q = _unit(np.asarray(pose, float)[:, 3:])
    T = len(q)
    out: list[tuple[int, int]] = []
    # Vectorised over i for each span length. The scalar version walked every
    # frame in Python -- a 41k-row episode times five span lengths times two
    # sides turned a whole-library pass into tens of minutes.
    for n in range(1, MAX_EXCURSION_FRAMES + 1):
        if T < n + 2:
            break
        i = np.arange(1, T - n)
        before, after = q[i - 1], q[i + n]
        swing = np.max(np.stack([_angle(q[i + k], before) for k in range(n)]), 0)
        back = np.max(np.stack([_angle(q[i + k], after) for k in range(n)]), 0)
        span = _angle(before, after)
        # The boundaries must be DISCONTINUOUS. A tracking loss jumps to the
        # wrong solution in one frame and jumps back in one frame; a real
        # reach-and-return travels there continuously. Without this the return
        # test alone flags a 2.6 s reach as an excursion -- which is exactly
        # the false positive `detect_pose_teleports` used its translation
        # conjunction to avoid, and why widening the window needed a second
        # condition rather than a looser first one.
        jump_in = _angle(q[i], q[i - 1])
        jump_out = _angle(q[i + n], q[i + n - 1])
        # EVERY interior frame must leave the path between the two anchors.
        # Without this a window spanning two separate glitches with good
        # tracking between them still satisfies everything above -- that is how
        # 22 scattered jumps over 5 s came out as one 150-frame span whose
        # median per-frame change was 1.2 degrees. A good frame hugs the
        # interpolated path; a lost one does not.
        off = np.full(len(i), np.inf)
        for k in range(n):
            t_ = (k + 1) / (n + 1)
            off = np.minimum(off, _angle(q[i + k], _slerp_many(before, after, t_)))
        hit = ((swing > MIN_EXCURSION_RAD) & (back > MIN_EXCURSION_RAD)
               & (span < swing * RETURN_RATIO)
               & (jump_in > MIN_JUMP_RAD) & (jump_out > MIN_JUMP_RAD)
               & (off > MIN_EXCURSION_RAD))
        # Every candidate is kept. Claiming frames shortest-first discarded the
        # span that covered a whole loss whenever a shorter one inside it also
        # qualified -- a 15-frame loss came out as (33, 9). Overlap is what
        # `_merge` is for.
        out.extend((int(i[j]), n) for j in np.where(hit)[0])
    return _merge(sorted(out))


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fuse touching or near-touching spans, then drop what is too long.

    A long track loss does not hold one wrong orientation -- it WANDERS, so
    sub-windows inside it satisfy the return test on their own. Measured on
    motherboard/2026-09-11/episode_007_seg02, rows 5401-5415 came out as eight
    separate spans, and every anchor but the outermost two was another bad
    frame: `repair_poses` then interpolated between two garbage poses, which
    is smooth, wrong, and survives a second pass unchanged.

    Merged runs are returned WHATEVER their length. A long one is half a
    second of invented motion, which is why `repair_poses(mark=True)` hands
    back a per-frame `pose_interpolated` mask and the release publishes it: a
    fabricated pose that looks measured is the thing to avoid, a labelled one
    is a choice the user gets to make. MAX_EXCURSION_FRAMES stays the
    DETECTION window -- how far ahead to look for the return -- not a cap on
    what may be filled.
    """
    if not spans:
        return []
    fused: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for i, n in spans[1:]:
        s, k = fused[-1]
        # Detections inside one long loss come out separated: the frames
        # between them have bad neighbours on BOTH sides, so the return test
        # never fires there. Anything within MAX_EXCURSION_FRAMES is treated as
        # the same event. The bias is deliberate -- two genuinely separate
        # glitches that close together merge into one span too long to repair,
        # and are left alone rather than interpolated on doubtful anchors.
        if i <= s + k + MERGE_GAP_FRAMES:
            fused[-1][1] = max(k, i + n - s)
        else:
            fused.append([i, n])
    return [(s, n) for s, n in fused]


def _slerp_many(a, b, t):
    """SLERP row-wise between two (N,4) arrays at scalar `t`."""
    a, b = _unit(a), _unit(b)
    b = np.where((a * b).sum(1, keepdims=True) < 0, -b, b)
    d = np.clip((a * b).sum(1), -1.0, 1.0)
    th = np.arccos(d)[:, None]
    s = np.sin(th)
    near = (s < 1e-8).ravel()
    out = np.where(np.repeat(near[:, None], 4, 1),
                   _unit(a * (1 - t) + b * t),
                   (np.sin((1 - t) * th) * a + np.sin(t * th) * b)
                   / np.where(s < 1e-8, 1.0, s))
    return _unit(out)


def _slerp(a, b, t):
    a, b = _unit(a), _unit(b)
    if np.dot(a, b) < 0:          # shortest arc
        b = -b
    d = np.clip(np.dot(a, b), -1.0, 1.0)
    th = np.arccos(d)
    if th < 1e-8:
        return _unit(a * (1 - t) + b * t)
    s = np.sin(th)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / s


# A flicker frame must sit at least this far from the local settled
# orientation. Same scale as MIN_JUMP_RAD: below it, it is jitter.
FLICKER_MIN_RAD = np.radians(25.0)

# How far to look either side for the settled orientation. Wide enough to reach
# past a burst of flicker, short enough to track real motion.
FLICKER_CONTEXT = 15

# A frame agreeing with fewer than this share of its neighbourhood is the
# minority branch. Real motion carries its neighbours along, so a frame in a
# genuine turn still agrees with the frames beside it.
#
# THE LIMIT OF THE IDEA, stated rather than hidden: "minority" stops meaning
# anything as the wrong branch approaches half the window. At 39% bad frames a
# synthetic burst already loses one frame at 0.35, which is why this sits at
# 0.45 -- and a burst worse than ~45% would need something other than support
# to separate the branches.
FLICKER_SUPPORT = 0.45

# Longest contiguous run still treated as flicker. Beyond it the frames are a
# dropout, not an alternation, and go to the excursion path so the
# fill-or-mark decision stays in one place.
FLICKER_MAX_RUN = 3


def find_flicker(pose: np.ndarray) -> list[int]:
    """Rows where the tracker resolved to a SECOND, wrong orientation.

    Different in kind from an excursion. A dropout is a contiguous run to
    interpolate through; a flicker ALTERNATES -- measured on
    motherboard/2026-09-11/episode_004_seg00, right sensor, 40 frames toggling
    between two orientations ~100 degrees apart, most of them good. Filling the
    run would destroy the good frames; skipping it leaves ~100 degree steps in
    the action stream, which is what made downstream drop whole training
    windows.

    A frame is flicker when it is far from the orientation its neighbourhood
    settles on, where "settles on" is the per-component median over a window
    that excludes the frame itself. The median is what makes this work on an
    alternating signal: the wrong branch never holds the majority for long, so
    the median tracks the right one.
    """
    q = _unit(np.asarray(pose, float)[:, 3:])
    T = len(q)
    if T < 5:
        return []
    # SUPPORT, not a median over components. A quaternion and its negation are
    # the same rotation, so aligning signs to a reference is ambiguous when the
    # two flicker states sit ~180 degrees apart -- measured on
    # motherboard/2026-05-11/episode_016 rows 7950-7953, 175 degrees apart,
    # where a per-component median came out meaningless and the burst was
    # missed. Counting how many neighbours AGREE with a frame uses only
    # `_angle`, which is sign-insensitive by construction.
    agree = np.zeros((T, 2 * FLICKER_CONTEXT + 1), bool)
    valid = np.zeros_like(agree)
    for j, o in enumerate(range(-FLICKER_CONTEXT, FLICKER_CONTEXT + 1)):
        if o == 0:
            continue
        lo_i = max(0, -o)
        hi_i = min(T, T - o)
        if hi_i <= lo_i:
            continue
        a = _angle(q[lo_i:hi_i], q[lo_i + o:hi_i + o])
        agree[lo_i:hi_i, j] = a <= FLICKER_MIN_RAD
        valid[lo_i:hi_i, j] = True
    n = valid.sum(1)
    support = np.where(n > 0, agree.sum(1) / np.maximum(n, 1), 1.0)
    # A settled frame agrees with most of its neighbourhood. A frame the
    # tracker mis-solved agrees with only the other mis-solved ones, which are
    # the minority -- that is what makes the alternation legible without ever
    # naming which orientation is "right".
    flagged = np.where((n >= 4) & (support < FLICKER_SUPPORT))[0]
    if not len(flagged):
        return []
    # FLICKER ALTERNATES; a dropout does not. Flagged frames that run
    # contiguously for more than a few frames are a loss, and a loss is the
    # excursion path's business, where REPAIR_LIMIT_FRAMES decides whether to
    # fill or to mark. Without this split the flicker path would fill a
    # 40-frame dropout one frame at a time and invent 1.3 s of motion behind
    # the policy's back.
    out, run = [], [flagged[0]]
    for i in flagged[1:]:
        if i == run[-1] + 1:
            run.append(int(i))
        else:
            if len(run) <= FLICKER_MAX_RUN:
                out += run
            run = [int(i)]
    if len(run) <= FLICKER_MAX_RUN:
        out += run
    return out


def repair_poses(pose: np.ndarray, mark: bool = False):
    """`(repaired, spans)`, or `(repaired, spans, interpolated)` with `mark`.

    Position is linear, orientation is SLERP, both between the good frames on
    either side of the WHOLE span -- never between two frames inside the same
    loss, which is what produced a smooth path through garbage on the first
    attempt.

    With `mark`: `(repaired, filled, interpolated, skipped)`.

    `interpolated` is a per-frame bool, True where the pose was invented. It
    exists because filling a gap is defensible only if the filling is visible.
    Ship it beside the poses; never let a fabricated pose look measured.

    `skipped` is the spans deliberately NOT filled -- longer than
    REPAIR_LIMIT_FRAMES, or with no clean anchor on one side. Those poses stay
    exactly as measured and are reported so they can be marked instead.
    """
    pose = np.asarray(pose, float)
    interp = np.zeros(len(pose), bool)

    # FLICKER FIRST, and separately. It alternates rather than holding, so the
    # excursion path would fuse it into one long span and refuse the lot --
    # which is what left 40 frames of ~100 degree steps in episode_004_seg00.
    # Here only the wrong-branch frames move; every good frame inside the run
    # is kept.
    flick = find_flicker(pose)
    if flick:
        pose = pose.copy()
        bad = np.zeros(len(pose), bool); bad[flick] = True
        good = np.where(~bad)[0]
        for i in flick:
            lo = good[good < i]
            hi = good[good > i]
            if not len(lo) or not len(hi):
                continue          # no clean frame on one side
            a, b = pose[lo[-1]], pose[hi[0]]
            t = (i - lo[-1]) / (hi[0] - lo[-1])
            pose[i, :3] = a[:3] * (1 - t) + b[:3] * t
            pose[i, 3:] = _slerp(a[3:], b[3:], t)
            interp[i] = True

    spans = find_excursions(pose)
    if not spans:
        return (pose, [], interp, []) if mark else (pose, [])

    out = pose.copy()
    filled, skipped = [], []
    for i, n in spans:
        lo, hi = i - 1, i + n
        if lo < 0 or hi >= len(pose):
            skipped.append((i, n))    # no clean anchor on one side
            continue
        if n > REPAIR_LIMIT_FRAMES:
            skipped.append((i, n))    # too long to invent; reported, not filled
            continue
        a, b = pose[lo], pose[hi]
        for k in range(n):
            t = (k + 1) / (n + 1)
            out[i + k, :3] = a[:3] * (1 - t) + b[:3] * t
            out[i + k, 3:] = _slerp(a[3:], b[3:], t)
        interp[i:i + n] = True
        filled.append((i, n))
    if mark:
        return out, filled, interp, skipped
    return out, filled
