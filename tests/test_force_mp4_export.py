"""MP4 recovery must preserve action columns and name actual source frames."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.force_recovery import finalize_mp4_recovery as export


@pytest.fixture
def case(tmp_path):
    table = pa.table({"frame_idx": [0, 1, 2, 3],
        "tactile_left_is_new": [True, False, True, False],
        "tactile_right_is_new": [True, False, True, False],
        "action": [[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
        "object_pose": [[float("nan"), 0.]] * 4,
        "sensor_left_pose": [[0., 0., 0., 0., 0., 0., 1.]] * 4,
        "sensor_right_pose": [[0., 0., 0., 0., 0., 0., 1.]] * 4,
    }).replace_schema_metadata({b"action_provenance": b"preserve this"})
    directory = tmp_path / "pushT" / "2026-06-18"
    directory.mkdir(parents=True)
    data = dict(force_normal_n=np.array([0, 0, 6, 6], np.float32),
        volume_mm3=np.array([0, 0, 1, 1], np.float32),
        contact_area_mm2=np.array([0, 0, 2, 2], np.float32),
        max_depth_mm=np.array([0, 0, .1, .1], np.float32),
        source_frame=np.array([0, 0, 2, 2], np.int32), reference_rows=np.array([0]),
        source_format=export.SOURCE_FORMAT, pipeline_version=export.PIPELINE_VERSION,
        force_calibration=export.CALIBRATION_NAME, force_calibration_ceiling_n=15.,
        alignment="video_frame_i_equals_parquet_row_i", lossy_input=True,
        absolute_force_validated_on_react=False)
    for side in export.SIDES:
        np.savez(directory / f"episode_000_{side}.npz", **data)
    return tmp_path, table, data


def test_force_only_preserves_actions_and_correctly_labels_mp4(case):
    root, table, _ = case
    output, sidecar = export.update_table(table, root, "pushT", "2026-06-18", "episode_000")
    assert output["action"].equals(table["action"])
    assert output.schema.metadata[b"action_provenance"] == b"preserve this"
    assert len(output.column_names) == len(table.column_names) + 4
    desc = output.schema.field("force_left_source_frame").metadata[b"twm.desc"]
    assert b"MP4 frame index" in desc and b"SOURCE H5" not in desc
    assert sidecar["lossy_input"] is True and sidecar["raw_h5_equivalent"] is False
    rerun, _ = export.update_table(output, root, "pushT", "2026-06-18", "episode_000")
    assert export.tables_equal(rerun, output)


def test_existing_derived_force_is_replaced_without_duplicate_names(case):
    root, table, _ = case
    for side in export.SIDES:
        table = table.append_column(f"force_{side}_normal_n", pa.array([99.] * 4))
        table = table.append_column(f"force_{side}_penetration_mm", pa.array([99.] * 4))
        table = table.append_column(f"force_{side}_target_pose", table[f"sensor_{side}_pose"])
    table = table.replace_schema_metadata({b"twm.force_export": json.dumps({"stiffness_n_per_mm": 2.}).encode()})
    pq.write_table(table, root / "base.parquet")
    table = pq.read_table(root / "base.parquet")
    output, _ = export.update_table(table, root, "pushT", "2026-06-18", "episode_000")
    assert len(output.column_names) == len(set(output.column_names))
    assert output["force_left_penetration_mm"].to_pylist() == [0., 0., 3., 3.]
    target = np.asarray(output["force_left_target_pose"].to_pylist())
    assert np.allclose(np.linalg.norm(target[:, :3], axis=1), [0, 0, .003, .003])
    assert output["action"].equals(table["action"])
    pq.write_table(output, root / "updated.parquet")
    assert export.tables_equal(pq.read_table(root / "updated.parquet"), output)


def test_rejects_mislabeled_alignment_even_with_correct_length(case):
    root, table, data = case
    data["source_frame"] = np.arange(4)
    path = root / "pushT/2026-06-18/episode_000_left.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="source-frame alignment"):
        export.validate_side(path, table, "left")


def test_rejects_nonfinite_geometry(case):
    root, table, data = case
    data["max_depth_mm"][2] = np.nan
    path = root / "pushT/2026-06-18/episode_000_left.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="invalid max_depth_mm"):
        export.validate_side(path, table, "left")
