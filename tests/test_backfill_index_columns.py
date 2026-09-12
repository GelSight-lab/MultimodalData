"""The 79 published segments shipped without the LeRobot index columns.

Every folder published before the cut release carries `task`, `task_index`,
`episode`, `episode_index` and `frame_index` as DATA. The cut segments carry
none of them, so a loader keyed on the column works on the old folders and
returns nothing on the new ones.

`task_index` is append-only: the ints are published inside every parquet
already downloaded, so renumbering a task would silently relabel someone's
local copy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
from backfill_index_columns import TASK_INDEX, episode_indices  # noqa: E402

from twm.react_preprocess.meta import add_index_columns  # noqa: E402


def test_episode_index_is_zero_based_over_the_whole_task():
    keys = ["2026-09-11/episode_001_seg00", "2026-09-10/episode_000_seg00",
            "2026-09-10/episode_000_seg01"]
    assert episode_indices(keys) == {"2026-09-10/episode_000_seg00": 0,
                                     "2026-09-10/episode_000_seg01": 1,
                                     "2026-09-11/episode_001_seg00": 2}


def test_segments_sort_after_their_ninth_sibling():
    """seg00..seg10 must not order as seg00, seg01, seg10, seg02."""
    keys = [f"2026-09-11/episode_000_seg{i:02d}" for i in range(12)]
    idx = episode_indices(keys)
    assert [idx[k] for k in keys] == list(range(12))


def test_task_index_is_append_only():
    assert TASK_INDEX == {"motherboard": 0, "pushT": 1, "rope": 2}


def test_columns_match_the_published_dtypes():
    """int64, like every folder published before the cut release."""
    t = add_index_columns(pa.table({"frame_idx": np.arange(5, dtype=np.int32)}),
                          "rope", 2, "2026-09-11/episode_000_seg00", 7)
    assert t.schema.field("task_index").type == pa.int64()
    assert t.schema.field("episode_index").type == pa.int64()
    assert t.schema.field("frame_index").type == pa.int64()
    assert t.column("frame_index").to_pylist() == [0, 1, 2, 3, 4]
    assert t.column("episode_index")[0].as_py() == 7


def test_rewriting_twice_does_not_duplicate_columns():
    t = pa.table({"frame_idx": np.arange(5, dtype=np.int32)})
    once = add_index_columns(t, "rope", 2, "d/e", 1)
    twice = add_index_columns(once, "rope", 2, "d/e", 1)
    for c in ("task", "task_index", "episode", "episode_index", "frame_index"):
        assert twice.column_names.count(c) == 1


def test_upload_patterns_name_only_the_published_files():
    """The local tree holds parquets this folder does not publish.

    `meta/**/*.parquet` swept up motherboard's 2026-09-09 segments -- which
    live in `data/validation` now -- and re-created them under
    `data/motherboard` as orphans with no videos beside them.
    """
    from backfill_index_columns import upload_patterns
    keys = ["2026-09-11/episode_000_seg00", "2026-09-11/episode_003_seg01"]
    assert upload_patterns(keys) == ["meta/2026-09-11/episode_000_seg00.parquet",
                                     "meta/2026-09-11/episode_003_seg01.parquet"]
    assert not any("*" in p for p in upload_patterns(keys))
