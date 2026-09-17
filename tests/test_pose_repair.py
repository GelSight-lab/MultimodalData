"""Repair the short tracking glitches OptiTrack leaves in the pose stream.

Measured on the published motherboard/2026-09-11/episode_006_seg00, right
sensor: 24 excursions where the quaternion swings away and returns within a
few frames. The worst rotates 168.7 degrees out and back while its two
neighbours are 1.0 degrees apart -- no physical object does that in 1/15 s.
The left sensor of the same episode is clean, which is what a per-marker-set
solve failure looks like.

`detect_pose_teleports` catches NONE of them. It requires translation AND
rotation to exceed threshold, and the comment says why: "ordinary fast motion
trips the translational threshold on its own, so requiring both is what
separates a tracking error from a quick reach". True for a reach, and it makes
the detector blind to a rotation-only glitch, which is the common one.

The signature used here is the one that identifies the failure rather than its
magnitude: an excursion that RETURNS. Frame i is far from i-1 and far from
i+k, while i-1 and i+k are close to each other. Real motion goes somewhere and
stays; a solve failure snaps out and back.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.pose_repair import (find_excursions, repair_poses,
                                              REPAIR_LIMIT_FRAMES,
                                              MAX_EXCURSION_FRAMES)


def _stream(n=60):
    """A smooth rotation about z, plus a steady translation."""
    ang = np.linspace(0, 0.6, n)
    q = R.from_euler("z", ang).as_quat()
    xyz = np.stack([np.linspace(0, 0.3, n), np.zeros(n), np.zeros(n)], 1)
    return np.concatenate([xyz, q], 1)


def _glitch(pose, at, frames=1, degrees=90.0):
    p = pose.copy()
    bad = (R.from_euler("x", np.radians(degrees))
           * R.from_quat(p[at, 3:])).as_quat()
    for k in range(frames):
        p[at + k, 3:] = bad
    return p


def test_a_rotation_only_excursion_is_found():
    p = _glitch(_stream(), 30, frames=1, degrees=90)
    got = find_excursions(p)
    assert [s for s, _ in got] == [30]


def test_a_multi_frame_excursion_is_found_whole():
    p = _glitch(_stream(), 30, frames=3, degrees=120)
    got = find_excursions(p)
    assert got and got[0][0] == 30 and got[0][1] == 3


def test_real_motion_is_left_alone():
    """A turn that goes somewhere and STAYS is not an excursion."""
    p = _stream()
    p[30:, 3:] = (R.from_euler("x", np.radians(90))
                  * R.from_quat(p[30:, 3:])).as_quat()
    assert find_excursions(p) == []


def test_a_long_deviation_is_not_repaired():
    """Beyond a few frames it is either real or a loss too long to invent
    through. Reported, never silently interpolated."""
    p = _glitch(_stream(80), 30, frames=8, degrees=90)
    # detected as ONE span covering the whole run, not eight pieces
    assert [n for _, n in find_excursions(p)] == [8]


def test_repair_puts_the_quaternion_back_on_the_smooth_path():
    """Asserts the OUTCOME, not which path produced it.

    A single-frame glitch is now caught by the flicker pass before the
    excursion pass sees it, so `spans` is empty and the pose is still fixed.
    Binding the test to `spans` bound it to an implementation detail.
    """
    clean = _stream()
    p = _glitch(clean, 30, frames=1, degrees=90)
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert interp[30], "frame 30 was neither repaired nor flagged"
    a = R.from_quat(fixed[30, 3:]); b = R.from_quat(clean[30, 3:])
    assert np.degrees((a.inv() * b).magnitude()) < 1.0


def test_repair_leaves_every_other_frame_bit_identical():
    clean = _stream()
    p = _glitch(clean, 30, frames=1, degrees=90)
    fixed, _ = repair_poses(p)
    mask = np.ones(len(p), bool); mask[30] = False
    assert np.array_equal(fixed[mask], p[mask])


def test_repair_interpolates_position_too():
    clean = _stream()
    p = clean.copy()
    p[30, :3] += [0.5, 0, 0]                       # a position spike as well
    p = _glitch(p, 30, frames=1, degrees=90)
    fixed, _ = repair_poses(p)
    assert abs(fixed[30, 0] - clean[30, 0]) < 1e-3


def test_a_clean_stream_is_returned_untouched():
    p = _stream()
    fixed, spans = repair_poses(p)
    assert spans == [] and np.array_equal(fixed, p)


def test_the_repaired_quaternions_stay_unit_norm():
    p = _glitch(_stream(), 30, frames=2, degrees=150)
    fixed, _ = repair_poses(p)
    assert np.allclose(np.linalg.norm(fixed[:, 3:], axis=1), 1.0, atol=1e-9)


def test_an_excursion_at_the_very_start_is_not_repaired():
    """There is no earlier good pose to interpolate from; inventing one is
    worse than leaving it flagged."""
    p = _glitch(_stream(), 0, frames=1, degrees=90)
    fixed, spans = repair_poses(p)
    assert spans == []


def _cluster(pose, at, n, degrees=100.0):
    """A RUN of bad frames, which is what real track loss looks like."""
    p = pose.copy()
    for k in range(n):
        p[at + k, 3:] = (R.from_euler("x", np.radians(degrees + 7 * k))
                         * R.from_quat(p[at + k, 3:])).as_quat()
    return p


def test_a_cluster_is_anchored_outside_itself(tmp_path=None):
    """Repairing frame-by-frame anchored `pose[i-1]` on another BAD frame.

    Measured on motherboard/2026-09-11/episode_007_seg02: rows 5401-5415 are
    one 15-frame loss, and the first pass split it into eight spans whose
    anchors were, except at the two ends, inside the cluster. Interpolating
    between two bad poses produces a smooth path through garbage, so the run
    survived the repair and the whole thing was not idempotent.
    """
    clean = _stream(80)
    p = _cluster(clean, 30, 4)
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert interp[30:34].all(), f"cluster not repaired: {spans}"
    for k in range(4):
        a = R.from_quat(fixed[30 + k, 3:]); b = R.from_quat(clean[30 + k, 3:])
        assert np.degrees((a.inv() * b).magnitude()) < 2.0


def test_a_cluster_longer_than_the_limit_is_filled_and_flagged():
    """Policy changed 2026-09-17: fill it, and SAY it was filled.

    This used to assert the long cluster was left alone. Leaving a 0.5 s hole
    keeps the data honest but breaks continuity, and the operator chose
    continuity plus a label over a gap. What must never happen is filling it
    silently, which is why the mask exists.
    """
    clean = _stream(80)
    p = _cluster(clean, 30, 9)          # 0.3 s, longer than a glitch
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert spans and interp[30:39].all() and skipped == []


def test_repair_is_idempotent_on_a_cluster():
    clean = _stream(80)
    p = _cluster(clean, 30, 4)
    once, _ = repair_poses(p)
    twice, spans2 = repair_poses(once)
    assert spans2 == [] and np.array_equal(once, twice)


def test_a_wandering_cluster_is_not_split_into_anchored_pieces():
    """The real failure. A long loss does not hold one wrong orientation --
    it wanders, so sub-windows inside it satisfy the return test on their own.
    motherboard/2026-09-11/episode_007_seg02 rows 5401-5415 came out as eight
    spans, and every anchor except the outermost two was another bad frame.

    Adjacent spans must merge, and a merged run longer than the limit must be
    reported rather than interpolated.
    """
    rng = np.random.default_rng(3)
    clean = _stream(90)
    p = clean.copy()
    for k in range(15):                       # 15 frames, wandering
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        p[30 + k, 3:] = (R.from_rotvec(ax * np.radians(70 + 30 * rng.random()))
                         * R.from_quat(clean[30 + k, 3:])).as_quat()
    fixed, spans = repair_poses(p)
    assert len(spans) == 1, f"split into anchored pieces: {spans}"
    s, n = spans[0]
    assert s <= 30 and s + n >= 45, f"span {spans[0]} does not cover the loss"


# ── pose_interpolated: repair everything, but say which frames were invented ──

def test_repair_reports_an_interpolated_mask():
    """The operator's call on 2026-09-17: fill the gap, but ship a flag so a
    user decides whether to trust it. A fabricated pose that looks like a
    measured one is the thing to avoid; a labelled one is a usable choice."""
    p = _glitch(_stream(), 30, frames=1, degrees=90)
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert interp.dtype == bool and len(interp) == len(p)
    assert interp[30] and interp.sum() == 1


def test_a_long_loss_is_now_filled_and_flagged():
    """Previously dropped. 25 frames is 0.83 s of invented motion -- allowed,
    because it is labelled, and refused silence is worse than labelled guess."""
    clean = _stream(90)
    p = _cluster(clean, 30, 15)
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert interp[30:45].all(), "the long loss was not filled"
    assert not interp[:30].any() and not interp[45:].any()


def test_a_wandering_long_loss_is_filled_as_one_span():
    """The cluster that split into eight anchored pieces. It must come out as
    ONE span anchored outside the whole run, not eight interpolations whose
    anchors are each other."""
    rng = np.random.default_rng(3)
    clean = _stream(90)
    p = clean.copy()
    for k in range(15):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        p[30 + k, 3:] = (R.from_rotvec(ax * np.radians(70 + 30 * rng.random()))
                         * R.from_quat(clean[30 + k, 3:])).as_quat()
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert len(spans) == 1, f"not merged into one: {spans}"
    s, n = spans[0]
    assert s <= 30 and s + n >= 45, f"span {spans[0]} does not cover 30..44"
    # anchored OUTSIDE the loss, so the filled path runs between clean poses
    a = R.from_quat(fixed[s, 3:]); b = R.from_quat(clean[s, 3:])
    assert np.degrees((a.inv() * b).magnitude()) < 25.0


def test_untouched_frames_are_not_flagged():
    p = _stream()
    fixed, spans, interp, skipped = repair_poses(p, mark=True)
    assert not interp.any() and spans == []


def test_mark_false_keeps_the_two_value_return():
    """Existing callers keep working."""
    p = _glitch(_stream(), 30, frames=1, degrees=90)
    got = repair_poses(p)
    assert len(got) == 2


def test_a_run_past_the_repair_limit_is_reported_not_filled():
    """The operator's call: long dropouts get marked, not invented. A 5 s run
    of repeated loss is not one recoverable glitch."""
    clean = _stream(120)
    p = _cluster(clean, 30, REPAIR_LIMIT_FRAMES + 10)
    fixed, filled, interp, skipped = repair_poses(p, mark=True)
    assert filled == [] and skipped
    assert not interp.any()
    assert np.array_equal(fixed, p), "a skipped span was modified"
