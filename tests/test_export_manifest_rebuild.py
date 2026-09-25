"""The manifest can be rebuilt from the sidecars when it has been truncated.

Merging fixes the manifest going forward, but a manifest already overwritten
by a scoped run has lost the entries it dropped. They are not gone: every
exported episode writes `<episode>.force.json` beside its parquet carrying the
same task/date/episode/rows/sides/stiffness. Rebuilding from those sidecars is
a reproducible repair rather than a hand-edit, which matters because
`upload_force_columns` publishes this file and cites its sha256 as a receipt.
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
    n = 12
    pose = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]] * n
    for task in ("pushT", "rope"):
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
        for side in ("left", "right"):
            np.savez(fd / f"episode_000_{side}.npz",
                     force_normal_n=np.linspace(0, 3, n),
                     max_depth_mm=np.linspace(0, 1, n),
                     source_frame=np.arange(n, dtype=np.int32),
                     pipeline_version=EX.MIN_PIPELINE_VERSION,
                     force_calibration="test")
        (stage / task / "episodes.jsonl").write_text(json.dumps({
            "task": task, "date": "2026-09-16",
            "episode": "2026-09-16/episode_000",
            "world_frame_offset": [0.0, 0.0, 0.0], "up_axis": "z"}) + "\n")
    monkeypatch.setenv("REACT_RELEASE", str(stage))
    monkeypatch.setattr(EX, "STAGE_ROOT", stage)
    monkeypatch.setattr(EX, "FORCE_ROOT", force)
    return tmp_path


def test_rebuild_recovers_entries_a_truncated_manifest_dropped(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(2.0, out, task="rope")
    full = json.loads((out / "force_export_manifest.json").read_text())

    # simulate the pre-fix truncation
    (out / "force_export_manifest.json").write_text(json.dumps(
        {**full, "n_episodes": 1, "episodes": full["episodes"][-1:]}))

    m = EX.rebuild_manifest(out)
    assert m["n_episodes"] == 2
    assert {(e["task"], e["date"], e["episode"]) for e in m["episodes"]} == \
           {("pushT", "2026-09-16", "episode_000"),
            ("rope", "2026-09-16", "episode_000")}
    assert m["total_rows"] == full["total_rows"]
    assert json.loads((out / "force_export_manifest.json").read_text()) == m


def test_rebuild_carries_the_stiffness_each_sidecar_records(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(None, out, task="rope")          # --force-only
    m = EX.rebuild_manifest(out)
    assert {e["task"]: e["stiffness_n_per_mm"] for e in m["episodes"]} == \
           {"pushT": 2.0, "rope": None}


def test_rebuild_ignores_a_sidecar_whose_parquet_is_gone(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(2.0, out, task="rope")
    (out / "rope" / "meta" / "2026-09-16" / "episode_000.parquet").unlink()
    m = EX.rebuild_manifest(out)
    assert {e["task"] for e in m["episodes"]} == {"pushT"}
