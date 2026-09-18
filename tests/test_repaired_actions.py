from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.repaired_actions import (
    ActionSeries,
    actions_fps15,
    native_actions,
)


def poses(n: int = 5) -> np.ndarray:
    out = np.zeros((n, 7), dtype=float)
    out[:, 0] = np.arange(n) * 0.01
    out[:, 3:] = R.from_euler("z", np.arange(n) * 5.0,
                              degrees=True).as_quat()
    return out


def test_native_action_is_translation_plus_relative_rot6d():
    p = poses()
    actions = native_actions(
        p, np.ones(len(p), bool), np.zeros(len(p), bool),
        np.full(len(p), "", dtype=object))

    assert actions.values.shape == (len(p) - 1, 9)
    assert_allclose(actions.values[:, :3], np.diff(p[:, :3], axis=0), atol=1e-7)
    expected = (R.from_quat(p[1:, 3:]) * R.from_quat(p[:-1, 3:]).inv()).as_matrix()
    expected_rot6d = np.concatenate([expected[:, :, 0], expected[:, :, 1]], axis=1)
    assert_allclose(actions.values[:, 3:], expected_rot6d, atol=1e-7)


def test_native_valid_requires_both_endpoints_and_physical_step():
    p = poses(6)
    pose_valid = np.ones(len(p), bool)
    pose_valid[2] = False
    p[5, 0] += 0.06

    actions = native_actions(
        p, pose_valid, np.zeros(len(p), bool),
        np.full(len(p), "", dtype=object))

    assert actions.valid.tolist() == [True, False, False, True, False]


def test_native_repaired_and_event_provenance_use_both_endpoints():
    p = poses(4)
    repaired = np.array([False, True, False, False])
    event_id = np.array(["", "right:1-1:flicker", "", ""], dtype=object)

    actions = native_actions(p, np.ones(4, bool), repaired, event_id)

    assert actions.repaired.tolist() == [True, True, False]
    assert actions.event_ids == (
        ("right:1-1:flicker",), ("right:1-1:flicker",), ())


def test_fps15_valid_is_and_and_repaired_is_or():
    p = poses(5)
    native = ActionSeries(
        values=np.zeros((4, 9), np.float32),
        valid=np.array([True, False, True, True]),
        repaired=np.array([False, True, False, True]),
        event_ids=((), ("a",), (), ("b",)),
        start_rows=np.arange(4, dtype=np.int32),
        end_rows=np.arange(1, 5, dtype=np.int32),
        rotation_deg=np.zeros(4),
        translation_mm=np.zeros(4),
    )

    half = actions_fps15(p, native)

    assert half.values.shape == (2, 9)
    assert half.valid.tolist() == [False, True]
    assert half.repaired.tolist() == [True, True]
    assert half.event_ids == (("a",), ("b",))
    assert half.start_rows.tolist() == [0, 2]
    assert half.end_rows.tolist() == [2, 4]


def test_invalid_quaternion_never_reaches_rotation_constructor():
    p = poses(4)
    p[2, 3:] = np.nan

    actions = native_actions(
        p, np.ones(4, bool), np.zeros(4, bool),
        np.full(4, "", dtype=object))

    assert actions.valid.tolist() == [True, False, False]
    assert np.isfinite(actions.values).all()
