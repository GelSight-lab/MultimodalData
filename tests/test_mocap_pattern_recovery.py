import numpy as np
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.mocap_branch import find_branch_candidates


def motion(n=300):
    t = np.arange(n)
    return np.c_[.001*t, .01*np.sin(t/13), .01*np.cos(t/17),
                 R.from_euler('z', .15*t, degrees=True).as_quat()]


def switch(raw, rows):
    out = raw.copy()
    r = R.from_quat(raw[rows, 3:])
    out[rows, :3] += r.apply([.02, -.01, .004])
    out[rows, 3:] = (r*R.from_euler('x', 70, degrees=True)).as_quat()
    return out


def recover(*args):
    from twm.react_preprocess import mocap_pattern_recovery as m
    return m.recover_pattern(*args)


def test_flicker_uses_repeated_same_recording_reference():
    truth = motion()
    rows = np.r_[30:40, 80:90, 151:156, 157:160, 163:164, 168:170]
    raw = switch(truth, rows)
    refs = find_branch_candidates(raw)
    result = recover(raw, 151, 169, refs)
    assert result is not None
    fixed, evidence = result
    assert evidence['method'] == 'repeated_body_branch_flicker'
    assert evidence['corrected_frames'] == 11
    np.testing.assert_allclose(fixed[:, :3], truth[151:170, :3], atol=.0003)
    np.testing.assert_array_equal(fixed[5], raw[156])


def test_mixed_event_preserves_clean_interior_and_branch_delta():
    truth = motion()
    raw = switch(truth, np.r_[50:55, 82:107])
    fixed, evidence = recover(raw, 50, 106, [])
    assert evidence['method'] == 'split_return_branches'
    assert evidence['corrected_frames'] == 30
    np.testing.assert_array_equal(fixed[5:32], raw[55:82])
    np.testing.assert_allclose(fixed[32:, :3], truth[82:107, :3], atol=.0003)


def test_missing_return_is_not_repaired():
    assert recover(switch(motion(), np.arange(50, 300)), 50, 100, []) is None


def test_nonfinite_rows_are_not_branch_motion():
    raw = switch(motion(), np.r_[50:55, 82:107]); raw[90] = np.nan
    assert recover(raw, 50, 106, []) is None


def test_overlength_branch_is_not_admitted():
    assert recover(switch(motion(), np.arange(50, 90)), 50, 89, []) is None


def test_unflagged_smooth_motion_has_no_recovery():
    assert recover(motion(), 50, 80, []) is None


def test_segment_edge_has_no_invented_anchor():
    assert recover(switch(motion(), np.arange(0, 15)), 0, 14, []) is None


def test_static_held_wrong_branch_is_not_retained_as_observed_motion():
    raw = switch(motion(), np.arange(50, 75)); raw[51:74] = raw[50]
    assert recover(raw, 50, 74, []) is None
