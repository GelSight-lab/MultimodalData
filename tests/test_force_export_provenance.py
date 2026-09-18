"""Exported provenance must describe the axis and controller actually used."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.force_recovery import dexforce
from twm.force_recovery import export_force_columns as EX
from twm import world_frame


@pytest.mark.parametrize("source", ["body_y", "dual_ball"])
@pytest.mark.parametrize("stiffness", [EX.STIFFNESS_N_PER_MM, None])
def test_export_records_selected_axis_in_schema_and_sidecar(
        tmp_path, monkeypatch, source, stiffness):
    # World-frame fingerprinting is a separate contract and reads the host's
    # release index/calibrations. Keep this force-provenance test self-contained.
    monkeypatch.setattr(world_frame, "build_declaration", lambda *args: {})
    # Leave the production default untouched in the body_y regression.
    if source == "body_y":
        assert dexforce.GEL_AXIS_SOURCE_DEFAULT == "body_y"
        local_axis = np.array([0.0, -1.0, 0.0])
    else:
        monkeypatch.setattr(dexforce, "GEL_AXIS_SOURCE_DEFAULT", source)
        calibration = tmp_path / "calibration"
        calibration.mkdir()
        local_axis = np.array([0.0, 0.0, 1.0])
        for side in EX.SIDES:
            (calibration / f"T_gel_to_rigid_{side}.json").write_text(
                json.dumps({"gel_axis_in_rigid": local_axis.tolist()}))
        monkeypatch.setattr(dexforce, "_calib_dir", lambda task: calibration)

    n = 20
    pose = np.tile([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], (n, 1))
    pose[1::2, 5:] = np.sqrt(0.5)  # Include a rotated sensor, not just identity.
    table = pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        **{f"sensor_{side}_pose": pose.tolist() for side in EX.SIDES},
        **{f"tactile_{side}_is_new": np.ones(n, bool) for side in EX.SIDES},
    })
    relative = "rope/meta/2026-09-16/episode_000.parquet"
    stage = tmp_path / "stage"
    (stage / relative).parent.mkdir(parents=True)
    pq.write_table(table, stage / relative)
    monkeypatch.setattr(EX, "STAGE_ROOT", stage)
    force_root = tmp_path / "force"
    force_dir = force_root / "rope/2026-09-16"
    force_dir.mkdir(parents=True)
    force = np.linspace(0.0, 14.0, n)
    for side in EX.SIDES:
        np.savez(force_dir / f"episode_000_{side}.npz",
                 force_normal_n=force, max_depth_mm=force / 10,
                 source_frame=np.arange(n, dtype=np.int32),
                 pipeline_version=EX.MIN_PIPELINE_VERSION,
                 force_calibration="synthetic-provenance-test")
    monkeypatch.setattr(EX, "FORCE_ROOT", force_root)

    output = tmp_path / "export"
    EX.export_episode("rope", "2026-09-16", "episode_000", stiffness, output)
    exported = pq.read_table(output / relative)
    header = json.loads(exported.schema.metadata[b"twm.force_export"])
    sidecar = json.loads((output / relative).with_suffix(".force.json").read_text())
    assert all(sidecar[key] == value for key, value in header.items())
    assert f"source={source}" in header["press_direction"]
    if source == "body_y":
        assert "-Y" in header["press_direction"]
        assert "T_gel_to_rigid" not in header["press_direction"]
    else:
        assert "T_gel_to_rigid" in header["press_direction"]
    assert header["stiffness_n_per_mm"] == stiffness
    assert "controller" in header["penetration"]

    suffixes = ["normal_n", "source_frame"]
    if stiffness is not None:
        suffixes += ["penetration_mm", "target_pose"]
    assert set(exported.column_names) == set(table.column_names) | {
        f"force_{side}_{suffix}" for side in EX.SIDES for suffix in suffixes}
    normal = dexforce.quat_to_matrix(pose[:, 3:]) @ local_axis
    for side in EX.SIDES:
        np.testing.assert_array_equal(exported[f"force_{side}_normal_n"],
                                      force.astype(np.float32))
        np.testing.assert_array_equal(exported[f"force_{side}_source_frame"],
                                      np.arange(n))
        if stiffness is None:
            continue
        assert stiffness == dexforce.STIFFNESS_N_PER_M / 1000.0 == 2.0
        target = np.asarray(exported[f"force_{side}_target_pose"].to_pylist())
        expected = pose.copy()
        expected[:, :3] += (force / stiffness / 1000.0)[:, None] * normal
        np.testing.assert_allclose(target, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(target[0], pose[0])
        np.testing.assert_array_equal(target[:, 3:], pose[:, 3:])
        np.testing.assert_array_equal(exported[f"force_{side}_penetration_mm"],
                                      (force / stiffness).astype(np.float32))
