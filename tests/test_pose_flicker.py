"""Two-state flicker is not a dropout, and must not be repaired like one.

Measured on motherboard/2026-09-11/episode_004_seg00, right sensor, native
frames 8549-8589: the quaternion ALTERNATES between two orientations ~100
degrees apart, over and over, for 40 frames. One of them matches the settled
pose on both sides of the run to within a degree; the other is OptiTrack
resolving the marker set to a second, wrong solution.

`find_excursions` sees it -- each B frame swings out and returns -- and the
detections fuse into one 40-frame span. That span is past REPAIR_LIMIT_FRAMES,
so it was reported in pose_gaps.json and left alone, which is right for a
dropout and wrong here: most of those 40 frames are GOOD. Interpolating
through them would destroy real motion; skipping them leaves ~100 degree steps
in the action stream, which is what the downstream `valid` flag then had to
drop whole training windows for.

The repair a flicker needs is different in kind: identify the wrong branch and
replace only its frames, keeping every good one.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.pose_repair import find_flicker, repair_poses


def _stream(n=80):
    ang = np.linspace(0, 0.4, n)
    q = R.from_euler("z", ang).as_quat()
    xyz = np.stack([np.linspace(0, 0.2, n), np.zeros(n), np.zeros(n)], 1)
    return np.concatenate([xyz, q], 1)


def _flicker(pose, start, stop, bad_rows, degrees=100.0, shift_mm=29.0):
    """Put `bad_rows` (absolute indices) into a second, wrong orientation."""
    p = pose.copy()
    for i in bad_rows:
        p[i, 3:] = (R.from_euler("y", np.radians(degrees))
                    * R.from_quat(pose[i, 3:])).as_quat()
        p[i, 0] += shift_mm / 1000.0
    return p


BAD = [30, 31, 35, 42, 43, 45, 48, 49, 52, 55, 56, 60, 62, 65]


def test_the_wrong_branch_is_found():
    p = _flicker(_stream(), 28, 68, BAD)
    rows = find_flicker(p)
    assert sorted(rows) == sorted(BAD), f"got {sorted(rows)}"


def test_the_good_frames_inside_the_run_are_not_touched():
    """The whole point. 40 frames of flicker are mostly GOOD frames."""
    clean = _stream()
    p = _flicker(clean, 28, 68, BAD)
    fixed, filled, interp, skipped = repair_poses(p, mark=True)
    good = [i for i in range(28, 68) if i not in BAD]
    for i in good:
        assert np.array_equal(fixed[i], p[i]), f"good frame {i} was rewritten"


def test_the_bad_frames_come_back_to_the_settled_orientation():
    clean = _stream()
    p = _flicker(clean, 28, 68, BAD)
    fixed, *_ = repair_poses(p, mark=True)
    for i in BAD:
        a = R.from_quat(fixed[i, 3:]); b = R.from_quat(clean[i, 3:])
        assert np.degrees((a.inv() * b).magnitude()) < 3.0, f"frame {i} still off"


def test_every_repaired_flicker_frame_is_flagged():
    p = _flicker(_stream(), 28, 68, BAD)
    fixed, filled, interp, skipped = repair_poses(p, mark=True)
    assert set(np.where(interp)[0]) >= set(BAD)


def test_no_transition_above_thirty_degrees_survives():
    """The downstream guard drops a window if any step rotates > 30 deg. That
    is what this repair exists to remove."""
    p = _flicker(_stream(), 28, 68, BAD)
    fixed, *_ = repair_poses(p, mark=True)
    q = fixed[:, 3:] / np.linalg.norm(fixed[:, 3:], axis=1, keepdims=True)
    d = np.degrees(2 * np.arccos(np.abs((q[:-1] * q[1:]).sum(1)).clip(-1, 1)))
    assert d.max() < 30.0, f"largest step still {d.max():.1f} deg"


def test_a_clean_stream_has_no_flicker():
    assert find_flicker(_stream()) == []


def test_real_motion_is_not_read_as_flicker():
    """A genuine turn visits new orientations and stays; it never returns to
    one repeated value."""
    p = _stream()
    p[40:, 3:] = (R.from_euler("y", np.radians(90))
                  * R.from_quat(p[40:, 3:])).as_quat()
    assert find_flicker(p) == []


def test_flicker_is_found_when_the_two_states_are_near_180_degrees():
    """The component-median version missed exactly this.

    Measured on motherboard/2026-05-11/episode_016, rows 7950-7953 alternate
    between two orientations 175 degrees apart. A quaternion and its negation
    are the same rotation, so aligning signs to a global reference is ambiguous
    at ~180 deg and the per-component median comes out meaningless. Counting
    how many neighbours AGREE with a frame is immune to the sign.
    """
    clean = _stream(90)
    p = clean.copy()
    for i in (40, 42, 44, 46):
        p[i, 3:] = (R.from_euler("y", np.radians(175))
                    * R.from_quat(clean[i, 3:])).as_quat()
    rows = find_flicker(p)
    assert sorted(rows) == [40, 42, 44, 46], f"got {sorted(rows)}"


def test_flicker_at_the_very_start_is_still_found():
    """`episode_016_seg04` begins three frames into a flicker burst, so the
    segment's own row 1-3 carried 175 deg steps. Context is one-sided there."""
    clean = _stream(60)
    p = clean.copy()
    for i in (2, 4):
        p[i, 3:] = (R.from_euler("y", np.radians(120))
                    * R.from_quat(clean[i, 3:])).as_quat()
    assert set(find_flicker(p)) >= {2, 4}


def test_a_contiguous_loss_is_left_to_the_excursion_path():
    """Flicker alternates. A run of consecutive bad frames is a dropout, and
    filling it frame-by-frame here would invent motion behind the back of
    REPAIR_LIMIT_FRAMES."""
    clean = _stream(90)
    p = clean.copy()
    rng = np.random.default_rng(1)
    for k in range(20):                      # one contiguous 20-frame loss
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        p[35 + k, 3:] = (R.from_rotvec(ax * np.radians(80 + 40 * rng.random()))
                         * R.from_quat(clean[35 + k, 3:])).as_quat()
    assert not (set(find_flicker(p)) & set(range(35, 55))), \
        "a contiguous dropout was taken as flicker"
    fixed, filled, interp, skipped = repair_poses(p, mark=True)
    assert not interp[35:55].any(), "a 20-frame loss was invented through"
