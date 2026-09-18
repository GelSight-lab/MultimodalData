import numpy as np
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.mocap_branch import find_branch_candidates


def motion(n=240):
    t = np.arange(n, dtype=float)
    xyz = np.column_stack((0.001*t, 0.015*np.sin(t/30), 0.002*np.cos(t/17)))
    quat = R.from_euler('xyz', np.column_stack((0.15*t, 0.25*t, 0.4*t)), degrees=True).as_quat()
    return np.column_stack((xyz, quat))


def branch(pose, start, stop):
    out = pose.copy()
    r = R.from_quat(pose[start:stop, 3:])
    out[start:stop, :3] += r.apply([0.025, -0.009, 0.004])
    out[start:stop, 3:] = (r * R.from_euler('yx', [110, 12], degrees=True)).as_quat()
    return out


def test_persistent_body_branch_preserves_varying_observed_motion():
    truth = motion()
    observed = branch(truth, 50, 150)
    original = observed.copy()
    candidates = find_branch_candidates(observed)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (candidate.start, candidate.end) == (50, 149)
    assert candidate.confidence == 'MEDIUM'
    assert candidate.evidence['bidirectional_transform_agrees']
    assert np.max(np.linalg.norm(candidate.pose[:, :3]-truth[50:150, :3], axis=1)) < .001
    error = (R.from_quat(candidate.pose[:, 3:]).inv()*R.from_quat(truth[50:150, 3:])).magnitude()
    assert np.max(np.degrees(error)) < .2
    np.testing.assert_array_equal(original, observed)


def test_missing_return_has_no_candidate():
    assert find_branch_candidates(branch(motion(), 50, 240)) == []


def test_true_fast_continuous_motion_is_preserved():
    truth = motion()
    truth[:, 3:] = R.from_euler('z', 35*np.arange(len(truth)), degrees=True).as_quat()
    assert find_branch_candidates(truth) == []


def test_long_gap_is_not_branch_evidence():
    pose = branch(motion(), 50, 150)
    assert find_branch_candidates(pose, known_gaps=[(80, 10)]) == []
    pose[90] = np.nan
    assert find_branch_candidates(pose) == []


def test_untrusted_return_anchor_disallows_candidate():
    pose = branch(motion(), 50, 150)
    trusted = np.ones(len(pose), dtype=bool)
    trusted[150] = False
    assert find_branch_candidates(pose, trusted=trusted) == []


def test_repeated_branch_support_and_quaternion_sign_invariance():
    truth = motion(320)
    pose = branch(branch(truth, 40, 70), 180, 220)
    pose[::2, 3:] *= -1
    candidates = find_branch_candidates(pose)
    assert [(c.start, c.end) for c in candidates] == [(40, 69), (180, 219)]
    assert all(c.evidence['matching_return_pairs'] == 2 for c in candidates)
    assert all(c.confidence == 'MEDIUM' for c in candidates)


def test_a_single_frame_flicker_has_both_return_boundaries():
    candidates = find_branch_candidates(branch(motion(), 50, 51))
    assert len(candidates) == 1
    assert (candidates[0].start, candidates[0].end) == (50, 50)


def test_different_exit_transform_is_not_a_return():
    pose = branch(motion(), 50, 150)
    pose[120:150, 3:] = (R.from_quat(pose[120:150, 3:])*R.from_euler('z', 20, degrees=True)).as_quat()
    assert find_branch_candidates(pose) == []


def test_abrupt_out_and_back_cannot_be_automatically_certified_as_tracking_error():
    # The same poses could be a true extremely abrupt movement. There is no
    # sensor-independent way to tell from these poses alone.
    pose = branch(motion(), 50, 150)
    before = pose.copy()
    candidates = find_branch_candidates(pose)
    assert all(c.confidence == 'MEDIUM' for c in candidates)
    assert all(not c.evidence['benchmark_validated'] for c in candidates)
    assert all(not c.evidence['independent_anchor_mask_supplied'] for c in candidates)
    np.testing.assert_array_equal(pose, before)


def test_gap_wholly_before_stream_does_not_mask_observations():
    candidates = find_branch_candidates(branch(motion(), 50, 150), known_gaps=[(-20, 5)])
    assert len(candidates) == 1


def test_branch_benchmark_scores_exact_interval_and_keeps_small_sample_gate_closed():
    from twm.scripts.benchmark_mocap_branch import benchmark_branch_task
    report = benchmark_branch_task([motion(1200)], task='synthetic', max_intervals=8)
    assert report['qualifying_intervals'] > 0
    assert report['metrics']['orientation_deg']['p95'] < 1
    assert not report['gate']['high_confidence_enabled']
    assert report['negative_controls']['continuous_fast_turn_candidates'] == 0
    assert report['negative_controls']['abrupt_out_back_automatic_admissions'] == 0


def test_review_proposal_is_medium_and_does_not_modify_primary_candidates(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from twm.scripts.review_mocap_branches import write_branch_review
    source = tmp_path/'source'
    source_file = source/'toy/meta/day/episode.parquet'
    source_file.parent.mkdir(parents=True)
    pose = branch(motion(), 50, 150)
    pq.write_table(pa.table({'sensor_left_pose': pose.tolist()}), source_file)
    candidate = tmp_path/'candidate'
    event_file = candidate/'toy/repair_events/day/episode.json'
    event_file.parent.mkdir(parents=True)
    events = {'task':'toy', 'date':'day', 'episode':'episode',
              'source_path':'toy/meta/day/episode.parquet',
              'events':[{'event_id':'test', 'side':'left', 'start':50, 'end':50, 'confidence':'LOW'}]}
    event_file.write_text(json.dumps(events))
    before = event_file.read_bytes(), source_file.read_bytes()
    report = write_branch_review(candidate, source, tmp_path/'review', manifest_digest='test')
    assert report['proposal_count'] == 1
    artifact = json.loads((tmp_path/'review'/report['proposals'][0]['path']).read_text())
    assert artifact['confidence'] == 'MEDIUM'
    assert not artifact['training_valid']
    assert artifact['requires_independent_branch_identity_evidence']
    assert np.asarray(artifact['proposed_pose']).shape == (100, 7)
    assert before == (event_file.read_bytes(), source_file.read_bytes())


def test_review_branch_invalidates_interior_and_merges_boundary_events():
    from twm.react_preprocess.mocap_repair import repair_pose_stream, Confidence
    from twm.react_preprocess.mocap_branch_review import apply_review_branches
    raw = branch(motion(), 50, 150)
    before = repair_pose_stream(raw, 'left')
    assert before.valid[80]
    result = apply_review_branches(before, raw, 'left')
    assert not result.valid[50:150].any()
    assert result.repaired[50:150].all()
    assert len(result.events) == 1
    assert result.events[0].confidence == Confidence.MEDIUM
    assert result.events[0].bout.start <= 50 and result.events[0].bout.end >= 149
    assert result.events[0].evidence['requires_independent_branch_identity_evidence']
    assert before.valid[80]  # Input result is immutable to this operation.


def test_review_branch_never_overrides_high_confidence_event():
    from dataclasses import replace
    from twm.react_preprocess.mocap_repair import repair_pose_stream, Confidence
    from twm.react_preprocess.mocap_branch_review import apply_review_branches
    raw = branch(motion(), 50, 150)
    result = repair_pose_stream(raw, 'left')
    result = replace(result, events=(replace(result.events[0], confidence=Confidence.HIGH),)+result.events[1:])
    updated = apply_review_branches(result, raw, 'left')
    np.testing.assert_array_equal(updated.valid, result.valid)
    np.testing.assert_array_equal(updated.pose, result.pose)
    assert updated.events == result.events


def test_review_branch_merges_two_flickers_without_duplicate_event_owners():
    from twm.react_preprocess.mocap_repair import repair_pose_stream
    from twm.react_preprocess.mocap_branch_review import apply_review_branches
    raw = branch(branch(motion(), 50, 51), 58, 59)
    before = repair_pose_stream(raw, 'left')
    result = apply_review_branches(before, raw, 'left')
    assert len(result.events) == 1
    assert set(result.events[0].replaced_frames) >= {50, 58}
    assert len(set(result.event_id[50:59])) == 1


def test_loss_evidence_gate_downgrades_unknown_fast_outback_but_allows_declared_gap():
    from twm.react_preprocess.mocap_repair import repair_pose_stream, TaskGate, Confidence
    from twm.react_preprocess.mocap_branch_review import apply_loss_evidence_gate
    raw = motion()
    raw[60, 3:] = (R.from_quat(raw[60, 3:])*R.from_euler('z', 40, degrees=True)).as_quat()
    gate = TaskGate(True, 60, endpoint_max_frames=5)
    before = repair_pose_stream(raw, 'left', task_gate=gate)
    assert before.events[0].confidence == Confidence.HIGH
    reviewed = apply_loss_evidence_gate(before, raw)
    assert reviewed.events[0].confidence == Confidence.MEDIUM
    assert not reviewed.valid[60]
    np.testing.assert_array_equal(reviewed.pose, before.pose)
    declared = repair_pose_stream(raw, 'left', [(60, 1)], task_gate=gate)
    accepted = apply_loss_evidence_gate(declared, raw, [(60, 1)])
    assert accepted.valid[60]
    assert accepted.events[0].confidence == Confidence.HIGH
    unchanged = apply_loss_evidence_gate(reviewed, raw)
    assert unchanged.events == reviewed.events


def test_writer_persists_review_policy_and_replay_keeps_full_span_invalid(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from twm.react_preprocess.mocap_candidate import CandidateWriter, snapshot_inputs, apply_decisions, verify_candidate
    from twm.react_preprocess.mocap_repair import TaskGate
    source = tmp_path/'source'
    path = source/'toy/meta/day/episode.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'sensor_left_pose': branch(motion(), 50, 150).tolist()}), path)
    manifest = snapshot_inputs(source, ('toy',))
    output = tmp_path/'candidate'
    CandidateWriter(source, output, manifest,
                    task_gates={'toy': TaskGate(True, 60, endpoint_max_frames=5)},
                    branch_review=True, require_loss_evidence=True).write_all()
    config = json.loads((output/'build_config.json').read_text())
    assert config['branch_review'] and config['require_loss_evidence']
    apply_decisions(output, {'schema_version':1, 'manifest_digest':manifest.digest, 'decisions':[]})
    table = pq.read_table(output/'toy/meta/day/episode.parquet')
    assert not np.asarray(table['pose_left_valid'].to_pylist())[50:150].any()
    assert verify_candidate(output).errors == ()


def test_scoped_review_ids_select_only_one_of_two_identical_episode_events(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from twm.react_preprocess.mocap_candidate import CandidateWriter, snapshot_inputs, apply_decisions, _event_index
    source = tmp_path/'source'
    for episode in ('episode_000', 'episode_001'):
        path = source/f'toy/meta/day/{episode}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({'sensor_left_pose': branch(motion(), 50, 51).tolist()}), path)
    manifest = snapshot_inputs(source, ('toy',))
    output = tmp_path/'candidate'
    CandidateWriter(source, output, manifest).write_all()
    index = _event_index(output)
    assert len(index) == 2
    selected = next(key for key in index if key.startswith('toy/day/episode_000::'))
    apply_decisions(output, {'schema_version':1, 'manifest_digest':manifest.digest,
                             'decisions':[{'event_id':selected, 'decision':'accept_repair'}]})
    assert pq.read_table(output/'toy/meta/day/episode_000.parquet')['pose_left_valid'][50].as_py()
    assert not pq.read_table(output/'toy/meta/day/episode_001.parquet')['pose_left_valid'][50].as_py()


def test_review_alias_and_scoped_name_cannot_submit_duplicate_decisions(tmp_path):
    from types import SimpleNamespace
    import pytest
    from twm.react_preprocess.mocap_candidate import _canonical_decisions
    path = tmp_path/'toy/repair_events/day/episode.json'
    event = {'event_id':'left:50-50:branch_discontinuity'}
    scoped = 'toy/day/episode::'+event['event_id']
    entries = {scoped:(path,event), event['event_id']:(path,event)}
    payload = {'schema_version':1, 'manifest_digest':'test', 'decisions':[
        {'event_id':event['event_id'], 'decision':'accept_repair'},
        {'event_id':scoped, 'decision':'invalidate'}]}
    with pytest.raises(ValueError, match='duplicate decision'):
        _canonical_decisions(payload, SimpleNamespace(digest='test'), entries)
