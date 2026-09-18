import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation as R


def table(n=80):
    pose = np.c_[np.arange(n)*.001, np.zeros((n, 2)), np.tile([0, 0, 0, 1], (n, 1))]
    cols = {'timestamp': np.arange(n)/30, 'source_h5_frame': np.arange(n)}
    for side in ('left', 'right'):
        cols.update({f'sensor_{side}_pose': pose.tolist(),
                     f'sensor_{side}_pose_repaired': pose.tolist(),
                     f'pose_{side}_valid': np.ones(n, bool),
                     f'pose_{side}_repaired': np.zeros(n, bool),
                     f'pose_{side}_repair_confidence': np.zeros(n, np.uint8),
                     f'pose_{side}_repair_event_id': ['']*n,
                     f'pose_{side}_repair_method': ['']*n})
    return pa.table(cols)


def event(start, end, confidence='MEDIUM'):
    return dict(start=start, end=end, local_start_row=start, local_end_row=end,
                confidence=confidence, side='left', event_id='test', method='endpoint_se3', evidence={})


def accept(t, es):
    from twm.react_preprocess.accepted_repairs import accept_table
    return accept_table(t, es, recover=False)


def test_33_frames_accepted_34_refused():
    out, sides, events = accept(table(), [event(10, 42)])
    assert events[0]['policy_accepted']
    assert np.asarray(out['action_valid_left'])[9:43].all()
    out, _, events = accept(table(), [event(10, 43)])
    assert not events[0]['policy_accepted']
    assert not np.asarray(out['action_valid_left'])[9:44].any()


def test_low_precedence_and_fps15_does_not_bridge_skipped_rows():
    out, sides, _ = accept(table(), [event(10, 25), event(16, 17, 'LOW')])
    assert not np.asarray(out['pose_left_valid'])[16:18].any()
    assert not np.asarray(out['action_valid_left'])[15:18].any()
    assert not sides['left']['fps15'].valid[7:9].any()
    assert not np.asarray(out['action_valid'])[-1]


def test_raw_preserved_and_missing_force_not_zero():
    t = table();out, _, _ = accept(t, [event(10, 12)])
    for name in ('timestamp', 'source_h5_frame', 'sensor_left_pose'):
        assert out[name].equals(t[name])
    assert not np.asarray(out['action_force_valid']).any()


def test_force_target_transport_and_zero_force_identity():
    from twm.react_preprocess.accepted_repairs import transport_force_target
    raw = np.array([[1, 2, 3, 0, 0, 0, 1.], [2, 3, 4, 0, 0, 0, 1.]])
    fixed = raw.copy(); fixed[:, :3] += .1
    fixed[:, 3:] = R.from_euler('z', [90, 90], degrees=True).as_quat()
    target = raw.copy(); target[0, 0] += .002
    out, valid = transport_force_target(raw, fixed, target, np.array([2., 0.]))
    np.testing.assert_allclose(out[0, :3], fixed[0, :3]+[0, .002, 0])
    np.testing.assert_array_equal(out[1], fixed[1])
    assert valid.all()


def test_contradictory_force_target_is_invalid():
    from twm.react_preprocess.accepted_repairs import transport_force_target
    p = np.array([[0, 0, 0, 0, 0, 0, 1.]])
    target = p.copy(); target[0, 0] = .1
    _, valid = transport_force_target(p, p, target, np.array([2.]))
    assert not valid[0]


def test_joint_force_action_mask_uses_both_current_and_next_rows():
    t = table(8)
    for side in ('left', 'right'):
        t = t.append_column(f'force_{side}_target_pose', t[f'sensor_{side}_pose'])
        t = t.append_column(f'force_{side}_penetration_mm', pa.array(np.zeros(8)))
        t = t.append_column(f'force_{side}_normal_n', pa.array(np.zeros(8)))
    out, _, _ = accept(t, [event(3, 3, 'LOW')])
    mask = np.asarray(out['action_force_valid'])
    assert mask[0] and mask[1] and mask[4]
    assert not mask[2] and not mask[3] and not mask[-1]


def test_publication_validator_detects_raw_force_overwrite():
    import pytest
    from twm.scripts.accept_mocap_repairs import validate_table
    t = table().append_column('force_left_normal_n', pa.array(np.ones(80)))
    out, sides, es = accept(t, [event(10, 12)])
    validate_table(t, out, sides, es)
    broken = out.set_column(out.schema.get_field_index('force_left_normal_n'),
                            'force_left_normal_n', pa.array(np.zeros(80)))
    with pytest.raises(ValueError, match='preserved'):
        validate_table(t, broken, sides, es)


def test_publication_validator_detects_next_frame_action_mismatch():
    import pytest
    from twm.scripts.accept_mocap_repairs import validate_table
    t = table(); out, sides, es = accept(t, [])
    out = out.set_column(out.schema.get_field_index('action'), 'action', pa.array(np.zeros((80, 14)).tolist()))
    with pytest.raises(ValueError, match='action'):
        validate_table(t, out, sides, es)
