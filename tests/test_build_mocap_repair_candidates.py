from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.mocap_candidate import SourceChangedError
from twm.react_preprocess.mocap_repair import TaskGate
from twm.scripts.build_mocap_repair_candidates import (
    TASKS,
    run_audit,
    run_benchmarks,
    run_build,
    run_verify,
    summarize_candidate,
)


def _write_episode(path: Path, n: int = 180, glitch: bool = False) -> None:
    t = np.linspace(0.0, 1.0, n)
    xyz = np.stack([0.18 * t, 0.004 * t**2, np.zeros(n)], axis=1)
    quat = R.from_euler("z", 18.0 * t, degrees=True).as_quat()
    left = np.concatenate([xyz, quat], axis=1)
    right = left.copy()
    if glitch:
        left[90, 0] += 0.029
        left[90, 3:] = (R.from_euler("y", 100, degrees=True)
                        * R.from_quat(left[90, 3:])).as_quat()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "source_h5_frame": np.arange(n, dtype=np.int32),
        "sensor_left_pose": pa.array(left.tolist()),
        "sensor_right_pose": pa.array(right.tolist()),
        "force_left_normal_n": np.zeros(n, np.float32),
    }), path)


def _release(root: Path) -> Path:
    for task in TASKS:
        _write_episode(
            root / task / "meta/2026-09-17/episode_000.parquet",
            glitch=task == "motherboard")
    return root


def disabled_gates() -> dict[str, TaskGate]:
    return {task: TaskGate(False, 0) for task in TASKS}


def test_audit_then_build_uses_exact_manifest_snapshot(tmp_path):
    source = _release(tmp_path / "release")
    output = tmp_path / "review" / "candidate_release"
    manifest = run_audit(source, output, TASKS)
    _write_episode(source / "toy/meta/2026-09-18/episode_001.parquet")

    build = run_build(output, task_gates=disabled_gates())

    assert build.manifest_digest == manifest.digest
    assert len(build.episodes) == 4
    assert not (output / "toy/meta/2026-09-18/episode_001.parquet").exists()
    assert run_verify(output).episodes == 4


def test_build_rechecks_source_digest_before_any_output(tmp_path):
    source = _release(tmp_path / "release")
    output = tmp_path / "review" / "candidate_release"
    manifest = run_audit(source, output, TASKS)
    changed = source / manifest.files[0].path
    table = pq.read_table(changed)
    pq.write_table(table.set_column(
        table.schema.get_field_index("force_left_normal_n"),
        "force_left_normal_n", pa.array(np.ones(table.num_rows, np.float32))),
        changed)

    with pytest.raises(SourceChangedError, match="digest"):
        run_build(output, task_gates=disabled_gates())


def test_benchmark_writes_one_explicit_gate_per_task(tmp_path):
    source = _release(tmp_path / "release")
    output = tmp_path / "review" / "candidate_release"
    run_audit(source, output, TASKS)

    reports = run_benchmarks(output, max_intervals=5, seed=3)

    assert set(reports) == set(TASKS)
    assert all(report.intervals <= 5 for report in reports.values())
    assert all(not report.gate.high_confidence_enabled
               for report in reports.values())
    assert (output / "task_gates.json").is_file()


def test_summary_counts_tiers_and_recovered_actions(tmp_path):
    source = _release(tmp_path / "release")
    output = tmp_path / "review" / "candidate_release"
    run_audit(source, output, TASKS)
    run_build(output, task_gates=disabled_gates())

    summary = summarize_candidate(output)

    motherboard = summary["tasks"]["motherboard"]
    assert motherboard["events"]["MEDIUM"] == 1
    assert motherboard["repaired_frames"] == 1
    assert motherboard["native_repaired_actions"] == 2
    assert motherboard["native_recovered_valid_actions"] == 0
