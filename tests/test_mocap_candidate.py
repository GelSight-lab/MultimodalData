from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.mocap_candidate import (
    CandidateVerificationError,
    CandidateWriter,
    SourceChangedError,
    apply_decisions,
    snapshot_inputs,
    verify_candidate,
)
from twm.react_preprocess.mocap_repair import Confidence, TaskGate


TASKS = ("motherboard", "pushT", "rope", "toy")


def _pose(n: int = 100, glitch: bool = False) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    xyz = np.stack([0.15 * t, 0.005 * t**2, np.zeros(n)], axis=1)
    quat = R.from_euler("z", 15.0 * t, degrees=True).as_quat()
    pose = np.concatenate([xyz, quat], axis=1)
    if glitch:
        pose[50, :3] += [0.029, 0.0, 0.0]
        pose[50, 3:] = (R.from_euler("y", 100, degrees=True)
                        * R.from_quat(pose[50, 3:])).as_quat()
    return pose


def _write_episode(path: Path, glitch: bool = False) -> Path:
    n = 100
    left, right = _pose(n, glitch), _pose(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "source_h5_frame": np.arange(1000, 1000 + n, dtype=np.int32),
        "sensor_left_pose": pa.array(left.tolist()),
        "sensor_right_pose": pa.array(right.tolist()),
        "object_pose": pa.array(np.full((n, 7), np.nan).tolist()),
        "force_left_normal_n": np.linspace(0, 5, n, dtype=np.float32),
        "force_right_normal_n": np.linspace(1, 6, n, dtype=np.float32),
    }), path, compression="zstd")
    return path


def sample_release(root: Path, *, glitch_task: str | None = None) -> Path:
    for task in TASKS:
        _write_episode(
            root / task / "meta" / "2026-09-17" / "episode_000.parquet",
            glitch=task == glitch_task)
    return root


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_manifest_is_sorted_and_contains_sha256(tmp_path):
    source = sample_release(tmp_path / "release")

    manifest = snapshot_inputs(source, TASKS)

    assert [item.path for item in manifest.files] == sorted(
        item.path for item in manifest.files)
    assert len(manifest.files) == 4
    assert all(len(item.sha256) == 64 and item.rows == 100
               and item.bytes > 0 for item in manifest.files)
    assert len(manifest.digest) == 64


def test_writer_refuses_production_or_nested_destination(tmp_path):
    source = sample_release(tmp_path / "release")
    manifest = snapshot_inputs(source, TASKS)

    with pytest.raises(ValueError, match="production"):
        CandidateWriter(source, source, manifest)
    with pytest.raises(ValueError, match="production"):
        CandidateWriter(source, source / "candidate", manifest)
    with pytest.raises(ValueError, match="production"):
        CandidateWriter(source, source.parent, manifest)


def test_writer_rejects_a_forged_manifest_digest(tmp_path):
    source = sample_release(tmp_path / "release")
    manifest = snapshot_inputs(source, TASKS)

    with pytest.raises(ValueError, match="manifest digest"):
        CandidateWriter(source, tmp_path / "review",
                        replace(manifest, digest="0" * 64))


def test_writer_refuses_source_changed_after_snapshot(tmp_path):
    source = sample_release(tmp_path / "release")
    manifest = snapshot_inputs(source, TASKS)
    changed = source / manifest.files[0].path
    table = pq.read_table(changed)
    pq.write_table(table.set_column(
        table.schema.get_field_index("force_left_normal_n"),
        "force_left_normal_n", pa.array(np.full(table.num_rows, 9, np.float32))),
        changed)
    writer = CandidateWriter(source, tmp_path / "review", manifest)

    with pytest.raises(SourceChangedError, match="digest"):
        writer.write_all()


def test_candidate_keeps_unselected_columns_and_good_poses_exact(tmp_path):
    source = sample_release(tmp_path / "release", glitch_task="motherboard")
    manifest = snapshot_inputs(source, TASKS)
    before_digest = {item.path: _sha256(source / item.path)
                     for item in manifest.files}
    output = tmp_path / "review" / "candidate_release"
    writer = CandidateWriter(
        source, output, manifest,
        task_gates={task: TaskGate(True, 60, endpoint_max_frames=5)
                    for task in TASKS})

    build = writer.write_all()

    assert len(build.episodes) == 4
    candidate = output / "motherboard/meta/2026-09-17/episode_000.parquet"
    before, after = pq.read_table(source / "motherboard/meta/2026-09-17/episode_000.parquet"), pq.read_table(candidate)
    assert before["force_left_normal_n"].equals(after["force_left_normal_n"])
    assert before["force_right_normal_n"].equals(after["force_right_normal_n"])
    raw = np.asarray(before["sensor_left_pose"].to_pylist())
    repaired_pose = np.asarray(after["sensor_left_pose"].to_pylist())
    repaired = np.asarray(after["pose_left_repaired"])
    confidence = np.asarray(after["pose_left_repair_confidence"])
    valid = np.asarray(after["pose_left_valid"])
    assert repaired[50]
    assert confidence[50] == Confidence.HIGH
    assert valid[50]
    assert np.array_equal(raw[~repaired], repaired_pose[~repaired])
    assert before["sensor_right_pose"].equals(after["sensor_right_pose"])

    native = output / "motherboard/actions_native/2026-09-17/episode_000_left.npz"
    fps15 = output / "motherboard/actions_fps15/2026-09-17/episode_000_left.npz"
    events = output / "motherboard/repair_events/2026-09-17/episode_000.json"
    with np.load(native, allow_pickle=False) as data:
        assert data["values"].shape == (99, 9)
        assert data["valid"].shape == (99,)
        assert data["repaired"].sum() == 2
    with np.load(fps15, allow_pickle=False) as data:
        assert data["values"].shape == (49, 9)
    event_payload = json.loads(events.read_text())
    assert event_payload["events"][0]["confidence"] == "HIGH"
    assert (output / "input_manifest.json").is_file()

    assert before_digest == {item.path: _sha256(source / item.path)
                             for item in manifest.files}


def _medium_candidate(tmp_path: Path):
    source = sample_release(tmp_path / "release", glitch_task="motherboard")
    manifest = snapshot_inputs(source, TASKS)
    output = tmp_path / "review" / "candidate_release"
    CandidateWriter(source, output, manifest).write_all()
    event_path = output / "motherboard/repair_events/2026-09-17/episode_000.json"
    event = json.loads(event_path.read_text())["events"][0]
    assert event["confidence"] == "MEDIUM"
    return source, manifest, output, event


def _logical_candidate_state(output: Path) -> tuple:
    parquet = output / "motherboard/meta/2026-09-17/episode_000.parquet"
    table = pq.read_table(parquet)
    native = output / "motherboard/actions_native/2026-09-17/episode_000_left.npz"
    with np.load(native, allow_pickle=False) as data:
        action_state = tuple((name, data[name].tobytes()) for name in sorted(data.files))
    def stable(value):
        if isinstance(value, float) and np.isnan(value):
            return "NaN"
        if isinstance(value, list):
            return tuple(stable(item) for item in value)
        return value

    return tuple((name, stable(table[name].to_pylist()))
                 for name in table.column_names), action_state


def test_accepting_medium_event_enables_pose_and_recomputed_actions(tmp_path):
    _, manifest, output, event = _medium_candidate(tmp_path)
    payload = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "decisions": [{"event_id": event["event_id"],
                       "decision": "accept_repair"}],
    }

    apply_decisions(output, payload)

    table = pq.read_table(
        output / "motherboard/meta/2026-09-17/episode_000.parquet")
    assert np.asarray(table["pose_left_valid"])[50]
    with np.load(
            output / "motherboard/actions_native/2026-09-17/episode_000_left.npz",
            allow_pickle=False) as data:
        assert data["valid"][49] and data["valid"][50]


def test_applying_same_decisions_twice_is_logically_idempotent(tmp_path):
    _, manifest, output, event = _medium_candidate(tmp_path)
    payload = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "decisions": [{"event_id": event["event_id"],
                       "decision": "accept_repair"}],
    }
    apply_decisions(output, payload)
    once = _logical_candidate_state(output)

    apply_decisions(output, payload)

    assert _logical_candidate_state(output) == once


def test_invalidate_restores_raw_pose_and_keeps_event_invalid(tmp_path):
    source, manifest, output, event = _medium_candidate(tmp_path)
    payload = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "decisions": [{"event_id": event["event_id"],
                       "decision": "invalidate"}],
    }

    apply_decisions(output, payload)

    raw = pq.read_table(
        source / "motherboard/meta/2026-09-17/episode_000.parquet")
    candidate = pq.read_table(
        output / "motherboard/meta/2026-09-17/episode_000.parquet")
    assert candidate["sensor_left_pose"][50].as_py() == raw["sensor_left_pose"][50].as_py()
    assert not candidate["pose_left_valid"][50].as_py()
    assert not candidate["pose_left_repaired"][50].as_py()


def test_decisions_reject_unknown_duplicate_and_wrong_manifest(tmp_path):
    _, manifest, output, event = _medium_candidate(tmp_path)
    base = {"schema_version": 1, "manifest_digest": manifest.digest}
    with pytest.raises(ValueError, match="unknown event"):
        apply_decisions(output, {**base, "decisions": [
            {"event_id": "missing", "decision": "invalidate"}]})
    with pytest.raises(ValueError, match="duplicate"):
        apply_decisions(output, {**base, "decisions": [
            {"event_id": event["event_id"], "decision": "invalidate"},
            {"event_id": event["event_id"], "decision": "unsure"}]})
    with pytest.raises(ValueError, match="manifest"):
        apply_decisions(output, {**base, "manifest_digest": "0" * 64,
                                 "decisions": []})


def test_verify_detects_undeclared_pose_change(tmp_path):
    _, _, output, _ = _medium_candidate(tmp_path)
    report = verify_candidate(output)
    assert report.episodes == 4 and report.errors == ()
    path = output / "motherboard/meta/2026-09-17/episode_000.parquet"
    table = pq.read_table(path)
    pose = np.asarray(table["sensor_left_pose"].to_pylist())
    pose[10, 0] += 0.1
    table = table.set_column(table.schema.get_field_index("sensor_left_pose"),
                             "sensor_left_pose", pa.array(pose.tolist()))
    pq.write_table(table, path)

    with pytest.raises(CandidateVerificationError, match="undeclared"):
        verify_candidate(output)
