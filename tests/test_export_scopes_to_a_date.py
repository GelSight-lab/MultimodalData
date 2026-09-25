"""The exporter must be able to skip archives it can never re-export.

Some raw recordings no longer exist on disk. Their force npz is frozen at
`pipeline_version` 5 -- the calibration that scored rho 0.143 end to end --
and `build_side` rightly refuses to ship it. But `--task` is the coarsest
scope there is, so one frozen archive date made the whole task unexportable:

    ValueError: force_recovery/pushT/2026-06-18/episode_000_left.npz:
                pipeline_version 5 < 8 ... re-run run_episode

`run_episode` cannot be re-run for those episodes; the H5 is gone. On
2026-09-22 this stopped the pushT publish chain at its export step with four
archive dates blocking seven current ones.

`--since` narrows by date. The missing-npz refusal that `_episodes` documents
-- the check that catches a HALF-COMPUTED force run -- still applies in full
within the window asked for, which is what keeps a partial run from shipping.
A closed archive is not a half-computed run.
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
    # 2026-06-18 is a closed archive: force frozen at the void calibration.
    # 2026-09-16 is current. 2026-09-18 is current but half computed.
    plan = {"2026-06-18": (5, ("left", "right")),
            "2026-09-16": (EX.MIN_PIPELINE_VERSION, ("left", "right")),
            "2026-09-18": (EX.MIN_PIPELINE_VERSION, ("left",))}
    lines = []
    for date, (version, sides) in plan.items():
        d = stage / "pushT" / "meta" / date
        d.mkdir(parents=True)
        pq.write_table(pa.table({
            "frame_idx": np.arange(n, dtype=np.int32),
            "sensor_left_pose": pose, "sensor_right_pose": pose,
            "tactile_left_is_new": np.ones(n, bool),
            "tactile_right_is_new": np.ones(n, bool),
        }), str(d / "episode_000.parquet"))
        fd = force / "pushT" / date
        fd.mkdir(parents=True)
        for side in sides:
            np.savez(fd / f"episode_000_{side}.npz",
                     force_normal_n=np.linspace(0, 3, n),
                     max_depth_mm=np.linspace(0, 1, n),
                     source_frame=np.arange(n, dtype=np.int32),
                     pipeline_version=version, force_calibration="test")
        lines.append(json.dumps({
            "task": "pushT", "date": date,
            "episode": f"{date}/episode_000",
            "world_frame_offset": [0.0, 0.0, 0.0], "up_axis": "z"}))
    (stage / "pushT" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("REACT_RELEASE", str(stage))
    monkeypatch.setattr(EX, "STAGE_ROOT", stage)
    monkeypatch.setattr(EX, "FORCE_ROOT", force)
    return tmp_path


def test_a_frozen_archive_blocks_the_unscoped_task(tree):
    """The behaviour that stopped the chain on 2026-09-22, pinned."""
    with pytest.raises((ValueError, EX.MissingForceFile)):
        EX.run_export(2.0, tree / "out", task="pushT")


def test_since_skips_the_frozen_archive(tree):
    m = EX.run_export(2.0, tree / "out", task="pushT",
                      since="2026-09-16", until="2026-09-16")
    assert {e["date"] for e in m["episodes"]} == {"2026-09-16"}


def test_a_half_computed_date_inside_the_window_is_still_refused(tree):
    """The safety net `_episodes` documents must survive the narrower scope."""
    with pytest.raises(EX.MissingForceFile):
        EX.run_export(2.0, tree / "out", task="pushT", since="2026-09-16")


def test_the_cli_accepts_since(tree, monkeypatch):
    import sys
    seen = {}
    def _fake(k, root, task=None, since=None, until=None):
        seen.update(task=task, since=since)
        return {"n_episodes": 0, "n_sensor_sides": 0, "total_rows": 0}
    monkeypatch.setattr(EX, "run_export", _fake)
    monkeypatch.setattr(EX, "verify", lambda root, task=None, since=None, until=None: {
        "identity_pass": True, "alignment_pass": True, "alignment_rate": 1.0,
        "roundtrip_max_abs_err_n": 0.0, "force_only": False,
        "penetration_over_gel_thickness_frac": 0.0,
        "penetration_p50_p95_p99_max_mm": (0.1, 1.0, 2.0, 3.0)})
    monkeypatch.setattr(EX, "_print", lambda r: None)
    monkeypatch.setattr(sys, "argv",
                        ["x", "export", "--task", "pushT", "--since", "2026-09-10"])
    assert EX.main() == 0
    assert seen == {"task": "pushT", "since": "2026-09-10"}
