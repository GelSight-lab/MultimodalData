import numpy as np
import pyarrow as pa
import pytest
from twm.react_toolbox.frames import convert_poses

from twm.scripts.propagate_mocap_repairs import align_rows, build_segment, source_relative_path


def tables(n=8):
    poses = [[i * .001, 0., 0., 0., 0., 0., 1.] for i in range(n)]
    raw = pa.table({'source_h5_frame': range(100, 100 + n),
                    'timestamp': np.arange(n) / 30,
                    'sensor_left_pose': poses, 'sensor_right_pose': poses,
                    'force_left_normal_n': np.arange(n, dtype=float)})
    candidate = raw
    for side in ('left', 'right'):
        for key, values in [('valid', [True] * n), ('repaired', [False] * n),
                            ('repair_confidence', [0] * n), ('repair_event_id', [''] * n)]:
            candidate = candidate.append_column(f'pose_{side}_{key}', pa.array(values))
    return raw, candidate


def test_segment_alignment_restarts_half_rate_and_preserves_force():
    source, candidate = tables()
    raw = source.slice(1, 6).append_column('action', pa.array([[999.] * 14] * 6))
    rows = align_rows(raw, source)
    out, sidecars = build_segment(raw, source, candidate, rows, [])
    assert out['force_left_normal_n'].equals(raw['force_left_normal_n'])
    assert out['sensor_left_pose'].equals(raw['sensor_left_pose'])
    assert out['action'][0].as_py()[0] == pytest.approx(.002)
    assert out['action'][-1].as_py()[0] == pytest.approx(.006)
    assert out['action_valid_left'].to_pylist() == [True] * 5 + [False]
    assert sidecars['left']['fps15'].start_rows.tolist() == [0, 2]
    assert sidecars['left']['fps15'].end_rows.tolist() == [2, 4]
    assert sidecars['left']['native'].values.shape == (5, 9)


def test_missing_or_noncontiguous_rows_fail_closed():
    source, candidate = tables()
    with pytest.raises(ValueError, match='missing'):
        align_rows(source, source.slice(1))
    raw = source.take(pa.array([1, 2, 4]))
    with pytest.raises(ValueError, match='contiguous'):
        align_rows(raw, source)


def test_pose_mismatch_and_duplicate_frame_ids_are_rejected():
    source, _ = tables()
    different = source.set_column(2, 'sensor_left_pose', source['sensor_right_pose'].slice(0))
    different = different.set_column(1, 'timestamp', pa.array(np.arange(8) + 10.))
    with pytest.raises(ValueError, match='timestamp'):
        align_rows(different, source)
    wrong_pose = np.asarray(source['sensor_left_pose'].to_pylist())
    wrong_pose[3, 0] += .001
    different = source.set_column(2, 'sensor_left_pose', pa.array(wrong_pose.tolist()))
    with pytest.raises(ValueError, match='sensor pose mismatch'):
        align_rows(different, source)
    duplicate = source.set_column(0, 'source_h5_frame', pa.array([100] * 8))
    with pytest.raises(ValueError, match='duplicate'):
        align_rows(source, duplicate)


def test_duplicate_episode_names_are_scoped_by_task_and_date():
    assert str(source_relative_path('data/rope/meta/2026-09-11/episode_000_seg02.parquet')) == 'rope/meta/2026-09-11/episode_000.parquet'
    assert str(source_relative_path('old_data/pushT/meta/2026-06-18/episode_000.parquet')) == 'pushT/meta/2026-06-18/episode_000.parquet'


def test_missing_candidate_validity_is_not_inferred():
    source, _ = tables()
    with pytest.raises(ValueError, match='pose_left_valid'):
        build_segment(source, source, source, np.arange(8), [])


def test_long_event_retains_full_duration_at_segment_boundary():
    source, candidate = tables(100)
    candidate = candidate.set_column(candidate.schema.get_field_index('pose_left_valid'),
                                     'pose_left_valid', pa.array([False] * 100))
    raw = source.slice(50, 4)
    out, _ = build_segment(raw, source, candidate, align_rows(raw, source),
                           [{'side': 'left', 'start': 0, 'end': 90}])
    assert out['action_long_lost_track_left'].to_pylist() == [True, True, True, False]


def test_known_coordinate_conversion_applies_to_repaired_pose_and_actions():
    source, candidate = tables()
    raw = source
    for side in ('left', 'right'):
        name = f'sensor_{side}_pose'
        raw = raw.set_column(raw.schema.get_field_index(name), name,
                             pa.array(convert_poses(source[name].to_pylist()).tolist()))
    rows = align_rows(raw, source)
    out, _ = build_segment(raw, source, candidate, rows, [])
    np.testing.assert_allclose(out['sensor_left_pose_repaired'].to_pylist(),
                               raw['sensor_left_pose'].to_pylist(), atol=1e-7)
    assert out['sensor_left_pose'].equals(raw['sensor_left_pose'])


def test_inventory_and_export_preserve_all_scopes_and_reject_stale_base(tmp_path):
    import json
    import pyarrow.parquet as pq
    from twm.scripts.propagate_mocap_repairs import inventory, export_inventory
    source, candidate = tables()
    base, src, cand, output = [tmp_path / name for name in ('base', 'src', 'cand', 'out')]
    paths = ['data/rope/meta/2026-09-11/episode_000_seg00.parquet',
             'old_data/pushT/meta/2026-06-18/episode_000.parquet',
             'data/validation/meta/2026-09-09/episode_000.parquet']
    for rel in paths:
        source_rel = source_relative_path(rel)
        for path, table in [(base / rel, source.slice(1, 6)),
                            (src / source_rel, source), (cand / source_rel, candidate)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
        task, _, date, name = source_rel.parts
        events = cand / task / 'repair_events' / date / (source_rel.stem + '.json')
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_text(json.dumps({'events': [
            dict(event_id='left:2-3:jump', side='left', start=2, end=3,
                 confidence='HIGH', method='interpolation'),
            dict(event_id='right:4-4:jump', side='right', start=4, end=4,
                 confidence='LOW', method='unrepaired_jump', evidence={'residual': float('inf')})]}))
    manifest = inventory(base, src, src, 'frozen-revision')
    assert manifest['errors'] == []
    assert len(manifest['episodes']) == 3
    result = export_inventory(manifest, cand, cand, output)
    assert len(result['episodes']) == 3
    assert len(list(output.rglob('*.npz'))) == 12
    assert len(list(output.glob('*/*/action_repair_events/*/*.json'))) == 3
    sample_manifest = json.loads((output / 'action_sample25.json').read_text())
    assert sample_manifest['actual_count'] == 6
    assert all('chart' in r and r['human_verification_status'] == 'UNVERIFIED' for r in sample_manifest['events'])
    for rel in paths:
        assert pq.read_table(output / rel)['force_left_normal_n'].equals(source.slice(1, 6)['force_left_normal_n'])
    pq.write_table(source, base / paths[0])
    with pytest.raises(ValueError, match='base digest changed'):
        export_inventory(manifest, cand, cand, output)


def test_segment_review_clips_times_qualifies_ids_and_only_links_known_videos():
    from twm.scripts.propagate_mocap_repairs import segment_reviews, review_html
    source, candidate = tables(100)
    raw = source.slice(50, 10)
    out, _ = build_segment(raw, source, candidate, np.arange(50, 60), [])
    remote = 'data/rope/meta/2026-09-11/episode_000_seg02.parquet'
    middle = 'data/rope/videos/2026-09-11/episode_000_seg02/view_middle.mp4'
    events = [dict(event_id='left:0-90:gap', side='left', start=0, end=90,
                   confidence='LOW', method='unrepaired_gap'),
              dict(event_id='right:52-53:spike', side='right', start=52, end=53,
                   confidence='HIGH', method='interpolated'),
              dict(event_id='right:80-90:gap', side='right', start=80, end=90,
                   confidence='LOW', method='unrepaired_gap')]
    records = segment_reviews(remote, raw, out, np.arange(50, 60), events,
                              {middle}, 'frozen-sha')
    assert len(records) == 2
    assert records[0]['local_start_row'] == 0
    assert records[0]['local_end_row'] == 9
    assert records[0]['local_start_seconds'] == 0
    assert records[0]['local_end_seconds'] == pytest.approx(.3)
    assert records[0]['source_start_row'] == 0
    assert records[0]['qualified_event_id'].startswith(remote + '::')
    assert set(records[0]['video_links']) == {'view_middle'}
    assert '/resolve/frozen-sha/' in records[0]['video_links']['view_middle']
    html = review_html(records)
    assert records[0]['qualified_event_id'] in html
    assert records[1]['qualified_event_id'] not in html
    assert '<svg' in html
    assert 'unrepaired_gap' in html
    sample_html = review_html(records, include_high=True)
    assert 'Play synced' in sample_html
    assert 'Pause' in sample_html
    assert 'loadedmetadata' in sample_html
    assert 'data-clip-start="0.000"' in sample_html
    assert 'measured camera footage is unchanged' in sample_html
    drifted = raw.set_column(raw.schema.get_field_index('timestamp'), 'timestamp',
                             pa.array(np.arange(raw.num_rows) / 25.))
    drift_records = segment_reviews(remote, drifted, out, np.arange(50, 60), events,
                                    {middle}, 'frozen-sha')
    assert drift_records[0]['local_end_seconds'] == pytest.approx(9 / 30)
    assert drift_records[0]['capture_end_seconds'] == pytest.approx(9 / 25)
    assert drift_records[0]['chart']['seconds'][-1] == pytest.approx(9 / 30)


def test_pose_method_uses_event_provenance_and_preserves_explicit_column():
    source, candidate = tables()
    ids = ['', 'left:1-2:spike', 'left:1-2:spike'] + [''] * 5
    candidate = candidate.set_column(candidate.schema.get_field_index('pose_left_repair_event_id'),
                                     'pose_left_repair_event_id', pa.array(ids))
    events = [dict(event_id='left:1-2:spike', side='left', start=1, end=2,
                   method='measured_method', confidence='LOW')]
    out, _ = build_segment(source, source, candidate, np.arange(8), events)
    assert out['pose_left_repair_method'].to_pylist() == [None, 'measured_method', 'measured_method'] + [None] * 5
    candidate = candidate.append_column('pose_left_repair_method', pa.array(['explicit'] * 8))
    out, _ = build_segment(source, source, candidate, np.arange(8), events)
    assert out['pose_left_repair_method'].to_pylist() == ['explicit'] * 8


def test_sample25_stratifies_final_events_and_includes_uncertainty():
    from twm.scripts.propagate_mocap_repairs import choose_review_samples, review_html
    scopes = ['data/motherboard', 'data/pushT', 'data/rope', 'data/toy',
              'old_data/motherboard', 'old_data/pushT', 'data/validation']
    records = []
    for i in range(35):
        path = f'{scopes[i % 7]}/meta/2026-09-09/episode_{i:03d}.parquet'
        duration = [2, 10, 80][i % 3]
        records.append(dict(remote_path=path, event_id=f'left:{i}:event',
                            qualified_event_id=path + f'::left:{i}:event',
                            source_start_row=0, source_end_row=duration - 1,
                            local_start_row=0, local_end_row=duration - 1,
                            local_start_seconds=0., local_end_seconds=duration / 30,
                            confidence='HIGH' if i < 25 else 'LOW',
                            kind='short_flicker' if duration == 10 else 'jump',
                            method='interpolation' if i < 25 else 'unrepaired_ambiguous_branch',
                            video_links={}))
    samples = choose_review_samples(records)
    assert len(samples) == 25
    assert sum(r['confidence'] == 'HIGH' for r in samples) == 20
    assert sum(r['confidence'] != 'HIGH' for r in samples) == 5
    assert {'/'.join(r['remote_path'].split('/')[:2]) for r in samples} == set(scopes)
    assert {r['source_end_row'] + 1 for r in samples} == {2, 10, 80}
    assert len({r['qualified_event_id'] for r in samples}) == 25
    html = review_html(samples, include_high=True, title='25 examples for human verification')
    assert 'not proof' in html
    assert all(r['qualified_event_id'] in html for r in samples)
    for i, record in enumerate(records):
        record['confidence'] = 'MEDIUM' if i < 25 else 'LOW'
        record['evidence'] = {'prior_confidence': 'HIGH', 'independent_loss_evidence': False} if i < 25 else {}
    samples = choose_review_samples(records)
    assert len(samples) == 25
    assert sum(r.get('evidence', {}).get('prior_confidence') == 'HIGH' for r in samples) == 20
    assert all(r['human_verification_status'] == 'UNVERIFIED' for r in samples)
