"""Renumbering must rewrite the parquet, not just the filename.

`episode` and `episode_index` are stored as DATA in every published parquet. A
folder whose files disagree with their own contents is worse than one that
never carried the columns: a loader keyed on the column and a loader keyed on
the path would disagree about which episode a row belongs to.

The cut segments arrived with none of the LeRobot index columns at all, so
they are added here rather than rewritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
from stage_validation_merge import _reindex  # noqa: E402

INDEX_COLUMNS = ("task", "task_index", "episode", "episode_index", "frame_index")


def _segment(n=40):
    """A cut segment: no index columns, and source_h5_frame left unrebased."""
    return pa.table({"frame_idx": np.arange(n, dtype=np.int32),
                     "source_h5_frame": (np.arange(n) + 9000).astype(np.int32)})


def _published(n=40, ep="2026-09-09/episode_000", idx=0):
    return pa.table({"frame_idx": np.arange(n, dtype=np.int32),
                     "task": pa.array(["pushT"] * n),
                     "task_index": pa.array([1] * n, pa.int64()),
                     "episode": pa.array([ep] * n),
                     "episode_index": pa.array([idx] * n, pa.int64()),
                     "frame_index": pa.array(np.arange(n), pa.int64())})


def test_a_segment_gains_every_index_column():
    t = _reindex(_segment(), "motherboard", "episode_007", 7)
    for c in INDEX_COLUMNS:
        assert c in t.column_names, c
    assert t.column("episode")[0].as_py() == "2026-09-09/episode_007"
    assert t.column("episode_index")[0].as_py() == 7
    assert t.column("task")[0].as_py() == "motherboard"
    assert t.column("task_index")[0].as_py() == 0


def test_frame_index_counts_rows_from_zero():
    t = _reindex(_segment(40), "motherboard", "episode_007", 7)
    fi = t.column("frame_index").to_pylist()
    assert fi == list(range(40))


def test_an_already_indexed_episode_is_renumbered_not_duplicated():
    t = _reindex(_published(), "pushT", "episode_003", 3)
    assert t.column("episode")[0].as_py() == "2026-09-09/episode_003"
    assert t.column("episode_index")[0].as_py() == 3
    for c in INDEX_COLUMNS:
        assert t.column_names.count(c) == 1, f"{c} appended instead of replaced"


def test_the_source_frame_is_left_alone():
    """It names the frame of the RECORDING, which renumbering does not move."""
    t = _reindex(_segment(), "motherboard", "episode_007", 7)
    assert t.column("source_h5_frame")[0].as_py() == 9000
