from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.mocap_repair import (
    Confidence,
    TaskGate,
    detect_bouts,
    repair_pose_stream,
)


def smooth_pose(n: int = 100) -> np.ndarray:
    """Slow, slightly curved motion that is clean at 30 fps."""
    t = np.linspace(0.0, 1.0, n)
    xyz = np.stack([0.20 * t, 0.015 * t**2, 0.005 * np.sin(t)], axis=1)
    quat = R.from_euler("zy", np.stack([20.0 * t, 3.0 * t**2], axis=1),
                        degrees=True).as_quat()
    return np.concatenate([xyz, quat], axis=1)


def alternate_branch(pose: np.ndarray, rows: list[int]) -> np.ndarray:
    out = pose.copy()
    wrong = R.from_euler("yx", [100.0, 8.0], degrees=True)
    for row in rows:
        out[row, :3] += [0.029, -0.003, 0.0]
        out[row, 3:] = (wrong * R.from_quat(pose[row, 3:])).as_quat()
    return out


def continuous_turn(n: int = 90, degrees: float = 120.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    xyz = np.stack([0.10 * t, np.zeros(n), np.zeros(n)], axis=1)
    quat = R.from_euler("z", degrees * t, degrees=True).as_quat()
    return np.concatenate([xyz, quat], axis=1)


def curved_pose(n: int = 120) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n)
    xyz = np.stack([0.12 * t, 0.008 * t**2, 0.004 * t**3], axis=1)
    quat = R.from_euler(
        "zyx", np.stack([14.0 * t, 2.0 * t**2, t], axis=1),
        degrees=True).as_quat()
    return np.concatenate([xyz, quat], axis=1)


def wrong_branch(pose: np.ndarray, rows: range) -> np.ndarray:
    return alternate_branch(pose, list(rows))


def passing_gate() -> TaskGate:
    return TaskGate(high_confidence_enabled=True, validated_max_frames=60)


def rotation_errors_deg(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    delta = R.from_quat(actual[:, 3:]).inv() * R.from_quat(expected[:, 3:])
    return np.degrees(delta.magnitude())


def test_repeated_ab_toggles_are_one_bout():
    bad_rows = [40, 42, 44, 47, 49]
    bouts = detect_bouts(alternate_branch(smooth_pose(), bad_rows), "right")

    assert len(bouts) == 1
    assert bouts[0].start <= 40 and bouts[0].end >= 49
    assert set(bouts[0].seed_frames) >= set(bad_rows)


def test_bout_closes_only_after_ten_stable_frames():
    bad_rows = [40, 42, 44, 51]
    bouts = detect_bouts(alternate_branch(smooth_pose(120), bad_rows), "left")

    assert len(bouts) == 1
    assert bouts[0].end >= 51
    assert bouts[0].right_context_start >= 52
    assert bouts[0].right_context_start + 9 < 120


def test_genuine_continuous_large_turn_is_not_a_bout():
    assert detect_bouts(continuous_turn(), "left") == []


def test_nonfinite_and_known_gap_rows_seed_one_bout():
    pose = smooth_pose()
    pose[40:43] = np.nan

    bouts = detect_bouts(pose, "right", known_gaps=[(40, 3)])

    assert len(bouts) == 1
    assert (bouts[0].start, bouts[0].end, bouts[0].kind) == (
        40, 42, "known_gap")
    assert bouts[0].seed_frames == (40, 41, 42)


def test_majority_wrong_branch_keeps_correct_observations():
    truth = curved_pose()
    observed = wrong_branch(truth, range(42, 58))
    observed[47] = truth[47]

    result = repair_pose_stream(observed, "right", task_gate=passing_gate())

    assert not result.repaired[47]
    expected_replaced = set(range(42, 58)) - {47}
    assert expected_replaced <= set(np.flatnonzero(result.repaired))
    assert np.percentile(rotation_errors_deg(result.pose[42:58], truth[42:58]), 95) < 5


def test_two_sided_linear_motion_can_repair_sixty_frames():
    truth = smooth_pose(140)
    observed = wrong_branch(truth, range(40, 100))

    result = repair_pose_stream(
        observed, "left", known_gaps=[(40, 60)], task_gate=passing_gate())

    assert result.repaired[40:100].all()
    assert (result.confidence[40:100] == Confidence.HIGH).all()
    assert result.valid[40:100].all()


def test_medium_candidate_is_reconstructed_but_invalid():
    truth = smooth_pose(150)
    observed = wrong_branch(truth, range(40, 101))

    result = repair_pose_stream(
        observed, "left", known_gaps=[(40, 61)], task_gate=passing_gate())

    mask = result.confidence == Confidence.MEDIUM
    assert mask[40:101].all()
    assert result.repaired[40:101].all()
    assert not result.valid[40:101].any()


def test_missing_anchor_is_low_and_original_is_preserved():
    observed = wrong_branch(smooth_pose(80), range(0, 20))

    result = repair_pose_stream(
        observed, "right", known_gaps=[(0, 20)], task_gate=passing_gate())

    assert np.array_equal(result.pose[:20], observed[:20])
    assert (result.confidence[:20] == Confidence.LOW).all()
    assert not result.valid[:20].any()


def test_robust_fit_records_bounded_iteration_and_convergence():
    truth = curved_pose()
    observed = wrong_branch(truth, range(42, 58))
    observed[45] = truth[45]
    observed[51] = truth[51]

    event = repair_pose_stream(
        observed, "right", task_gate=passing_gate()).events[0]

    assert 1 <= event.evidence["fit_iterations"] <= 5
    assert event.evidence["fit_converged"] is True
    assert 0 < event.evidence["translation_inlier_threshold_mm"] <= 20
    assert 0 < event.evidence["rotation_inlier_threshold_deg"] <= 12
    assert event.evidence["branch_unambiguous"] is True


def test_frames_outside_repair_mask_are_bit_identical():
    truth = curved_pose()
    observed = wrong_branch(truth, range(42, 58))
    result = repair_pose_stream(observed, "right", task_gate=passing_gate())

    assert np.array_equal(result.pose[~result.repaired],
                          observed[~result.repaired])


def test_repair_is_idempotent():
    truth = curved_pose()
    observed = wrong_branch(truth, range(42, 58))
    once = repair_pose_stream(observed, "right", task_gate=passing_gate())
    twice = repair_pose_stream(once.pose, "right", task_gate=passing_gate())

    assert np.array_equal(twice.pose, once.pose)
    assert not twice.repaired.any()


def test_quaternion_sign_changes_do_not_seed_a_bout():
    pose = smooth_pose()
    pose[30:60, 3:] *= -1

    assert detect_bouts(pose, "left") == []


def test_persistent_branch_without_return_is_low_and_not_replaced():
    truth = smooth_pose()
    observed = truth.copy()
    observed[40:] = wrong_branch(truth, range(40, len(truth)))[40:]

    result = repair_pose_stream(observed, "left", task_gate=passing_gate())

    assert np.array_equal(result.pose, observed)
    assert not result.repaired.any()
    assert result.events
    assert all(event.confidence == Confidence.LOW for event in result.events)
