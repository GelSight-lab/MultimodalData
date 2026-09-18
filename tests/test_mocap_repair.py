from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.mocap_repair import detect_bouts


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
