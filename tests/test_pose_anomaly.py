from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.pose_anomaly import detect_pose_events


def poses(yaw_deg, x_mm=None):
    yaw_deg = np.asarray(yaw_deg, float)
    if x_mm is None:
        x_mm = np.arange(len(yaw_deg), dtype=float)
    out = np.zeros((len(yaw_deg), 7), float)
    out[:, 0] = np.asarray(x_mm, float) / 1000.0
    out[:, 3:] = R.from_euler("z", yaw_deg, degrees=True).as_quat()
    return out


def test_one_frame_jump_out_and_back_is_repairable():
    events = detect_pose_events(
        poses([0, 1, 101, 2, 3], [0, 1, 31, 2, 3]), "right")
    assert len(events) == 1
    event = events[0]
    assert (event.start, event.end, event.kind) == (2, 2, "returning_excursion")
    assert event.repairable
    assert event.bad_frames == (2,)
    assert event.evidence["return_rotation_deg"] < 12
    assert event.evidence["return_translation_mm"] < 20


def test_short_contiguous_wrong_branch_is_repairable_with_two_anchors():
    events = detect_pose_events(
        poses([0, 1, 91, 92, 3, 4], [0, 1, 31, 32, 3, 4]), "left")
    assert len(events) == 1
    assert (events[0].start, events[0].end) == (2, 3)
    assert events[0].kind == "returning_excursion"
    assert events[0].repairable


def test_alternating_rotation_and_translation_branch_is_flicker():
    events = detect_pose_events(
        poses([0, 1, 101, 2, 102, 3, 4], [0, 1, 31, 2, 32, 3, 4]), "right")
    assert len(events) == 1
    assert events[0].kind == "branch_flicker"
    assert events[0].bad_frames == (2, 4)
    assert events[0].repairable
    assert events[0].evidence["branch_translation_mm"] > 20


def test_continuing_large_turn_is_reviewed_but_not_called_tracking_loss():
    events = detect_pose_events(
        poses([0, 8, 24, 60, 86, 101], [0, 1, 2, 3, 4, 5]), "right")
    assert len(events) == 1
    assert events[0].kind == "plausible_motion"
    assert not events[0].repairable


def test_persistent_pose_branch_change_is_not_interpolated():
    events = detect_pose_events(
        poses([0, 1, 111, 112, 113, 114], [0, 1, 41, 42, 43, 44]), "left")
    assert len(events) == 1
    assert events[0].kind == "persistent_branch"
    assert not events[0].repairable


def test_episode_edge_has_no_two_sided_anchor():
    events = detect_pose_events(
        poses([120, 0, 1, 2], [40, 0, 1, 2]), "right")
    assert len(events) == 1
    assert events[0].kind == "edge_discontinuity"
    assert not events[0].repairable


def test_known_long_gap_is_emitted_once_and_never_repaired():
    p = poses([0] * 30)
    events = detect_pose_events(p, "left", known_gaps=[(5, 20)])
    assert len(events) == 1
    assert (events[0].start, events[0].end) == (5, 24)
    assert events[0].kind == "long_gap"
    assert not events[0].repairable


def test_fast_translation_without_rotation_is_not_a_tracking_error():
    events = detect_pose_events(
        poses([0, 1, 2, 3, 4], [0, 1, 81, 82, 83]), "left")
    assert events == []
