import pyarrow as pa
import pytest

from twm.scripts.publish_action_repairs import merge_force_table, validate_publication_manifest


def test_merge_preserves_action_and_new_force_metadata():
    base = pa.table({'source_h5_frame': [10, 11], 'timestamp': [0., 1.],
                     'sensor_left_pose': [[0.] * 7] * 2})
    action = base.append_column('action', pa.array([[1.] * 14] * 2))
    force = base.append_column(pa.field('force_left_normal_n', pa.float32(),
                                       metadata={'source': 'mp4'}),
                               pa.array([2., 3.], type=pa.float32()))
    force = force.replace_schema_metadata({b'twm.force_export': b'v8'})
    result = merge_force_table(action, force)
    assert result['action'].equals(action['action'])
    assert result['force_left_normal_n'].equals(force['force_left_normal_n'])
    assert result.schema.field('force_left_normal_n').metadata == {b'source': b'mp4'}
    assert result.schema.metadata[b'twm.force_export'] == b'v8'


def test_merge_rejects_different_source():
    action = pa.table({'timestamp': [0., 1.], 'sensor_left_pose': [[0.] * 7] * 2})
    force = pa.table({'timestamp': [0., 2.], 'sensor_left_pose': [[0.] * 7] * 2,
                      'force_left_normal_n': [1., 2.]})
    with pytest.raises(ValueError, match='timestamp'):
        merge_force_table(action, force)


def test_publication_requires_force_commit_binding():
    with pytest.raises(ValueError, match='force revision'):
        validate_publication_manifest({'force_revision': 'old', 'force_merged_episodes': 36}, 'new')
    with pytest.raises(ValueError, match='36'):
        validate_publication_manifest({'force_revision': 'new', 'force_merged_episodes': 35}, 'new')
    validate_publication_manifest({'force_revision': 'new', 'force_merged_episodes': 36}, 'new')
