"""The exporter must be able to run one task at a time.

`run_export` enumerated every parquet under STAGE_ROOT and refused if ANY
sensor-side lacked an npz. So a run where three tasks are finished and the
fourth is still computing could export nothing at all, and three idle cores
waited on one. The v8 runbook names this directly: the force/export commands
"enumerate whole trees rather than respecting a per-task subset".

Scoping is per TASK and not per episode on purpose. The missing-npz refusal is
the check that catches a half-finished force run, and narrowing it to
individual episodes would let a partially computed task through.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.force_recovery import export_force_columns as EX


@pytest.fixture
def tree(tmp_path, monkeypatch):
    stage = tmp_path / "input"
    force = tmp_path / "force"
    n = 12   # >= 10: world_frame refuses to fingerprint fewer
    pose = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]] * n
    for task, ready in (("pushT", True), ("rope", False)):
        d = stage / task / "meta" / "2026-09-16"
        d.mkdir(parents=True)
        pq.write_table(pa.table({
            "frame_idx": np.arange(n, dtype=np.int32),
            "sensor_left_pose": pose, "sensor_right_pose": pose,
            "tactile_left_is_new": np.ones(n, bool),
            "tactile_right_is_new": np.ones(n, bool),
        }), str(d / "episode_000.parquet"))
        fd = force / task / "2026-09-16"
        fd.mkdir(parents=True)
        sides = ("left", "right") if ready else ("left",)   # rope half done
        for side in sides:
            np.savez(fd / f"episode_000_{side}.npz",
                     force_normal_n=np.linspace(0, 3, n),
                     max_depth_mm=np.linspace(0, 1, n),
                     source_frame=np.arange(n, dtype=np.int32),
                     pipeline_version=EX.MIN_PIPELINE_VERSION,
                     force_calibration="test")
        # The world-frame offset is READ from episodes.jsonl and an unlisted
        # episode RAISES rather than assuming zero. Without this the fixture
        # falls through to the production index and the guard fires -- which
        # is the guard doing its job, not a defect in it.
        (stage / task / "episodes.jsonl").write_text(json.dumps({
            # the `episode` field carries the DATE PREFIX; world_offset_m
            # keys on "<date>/<episode>" and a bare stem misses
            "task": task, "date": "2026-09-16",
            "episode": "2026-09-16/episode_000",
            "world_frame_offset": [0.0, 0.0, 0.0], "up_axis": "z"}) + "\n")
    monkeypatch.setenv("REACT_RELEASE", str(stage))
    monkeypatch.setattr(EX, "STAGE_ROOT", stage)
    monkeypatch.setattr(EX, "FORCE_ROOT", force)
    return tmp_path


def test_a_finished_task_exports_while_another_is_still_computing(tree):
    m = EX.run_export(2.0, tree / "out", task="pushT")
    assert m["n_episodes"] == 1
    assert {e["task"] for e in m["episodes"]} == {"pushT"}


def test_the_unfinished_task_is_still_refused_when_asked_for(tree):
    with pytest.raises(EX.MissingForceFile):
        EX.run_export(2.0, tree / "out", task="rope")


def test_without_a_task_the_half_finished_run_is_still_refused(tree):
    """The whole-tree refusal is the safety net; scoping must not remove it."""
    with pytest.raises(EX.MissingForceFile):
        EX.run_export(2.0, tree / "out")


def test_the_cli_accepts_a_task(tree, monkeypatch):
    import sys
    seen = {}
    def _fake(k, root, task=None):
        seen["task"] = task
        return {"n_episodes": 0, "n_sensor_sides": 0, "total_rows": 0}
    monkeypatch.setattr(EX, "run_export", _fake)
    monkeypatch.setattr(EX, "verify", lambda root, task=None: {
        "identity_pass": True, "alignment_pass": True, "alignment_rate": 1.0,
        "roundtrip_max_abs_err_n": 0.0, "force_only": False,
        "penetration_over_gel_thickness_frac": 0.0,
        "penetration_p50_p95_p99_max_mm": (0.1, 1.0, 2.0, 3.0)})
    monkeypatch.setattr(EX, "_print", lambda r: None)
    monkeypatch.setattr(sys, "argv", ["x", "export", "--task", "pushT"])
    assert EX.main() == 0
    assert seen["task"] == "pushT"
