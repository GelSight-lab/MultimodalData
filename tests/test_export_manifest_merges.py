"""A scoped export must not erase the manifest's record of the other tasks.

`run_export` rebuilt `force_export_manifest.json` from the episodes IT just
wrote. With `--task` that is a subset, so exporting one task replaced the
run-level description of the whole tree with a description of that task. On
2026-09-22 the file on disk claimed the entire force export was ONE episode
(rope/2026-09-20) while seven pushT dates sat exported beside it.

This is not cosmetic. `upload_force_columns` ships that file to the Hub as
`data/force_export_manifest.json` and records its sha256 as the receipt for
what was pushed, so the truncation is published and then cited as evidence.

The manifest must therefore describe the TREE, not the run: entries for
episodes this run wrote are replaced, entries for episodes still present under
the export root are kept, and entries whose parquet has left the tree are
dropped.
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


def _manifest(root):
    return json.loads((root / "force_export_manifest.json").read_text())


def _keys(m):
    return {(e["task"], e["date"], e["episode"]) for e in m["episodes"]}


def test_a_second_scoped_export_keeps_the_first(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(2.0, out, task="rope")
    m = _manifest(out)
    assert _keys(m) == {("pushT", "2026-09-16", "episode_000"),
                        ("rope", "2026-09-16", "episode_000")}
    assert m["n_episodes"] == 2
    assert m["n_sensor_sides"] == 4


def test_the_totals_describe_the_merged_tree(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    one = _manifest(out)["total_rows"]
    EX.run_export(2.0, out, task="rope")
    assert _manifest(out)["total_rows"] == 2 * one


def test_re_exporting_a_task_replaces_rather_than_duplicates(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(2.0, out, task="pushT")
    m = _manifest(out)
    assert m["n_episodes"] == 1
    assert len(m["episodes"]) == 1


def test_an_episode_whose_parquet_left_the_tree_is_dropped(tree):
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(2.0, out, task="rope")
    (out / "pushT" / "meta" / "2026-09-16" / "episode_000.parquet").unlink()
    EX.run_export(2.0, out, task="rope")
    m = _manifest(out)
    assert _keys(m) == {("rope", "2026-09-16", "episode_000")}


def test_each_entry_carries_the_stiffness_it_was_exported_with(tree):
    """A merged manifest can span runs; one top-level k would misdescribe them."""
    out = tree / "out"
    EX.run_export(2.0, out, task="pushT")
    EX.run_export(None, out, task="rope")      # --force-only
    by = {e["task"]: e.get("stiffness_n_per_mm") for e in _manifest(out)["episodes"]}
    assert by == {"pushT": 2.0, "rope": None}
