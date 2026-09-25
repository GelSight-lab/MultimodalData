"""Updating the force channel must not touch the video.

Re-cutting was the wrong tool for this. Only the force columns changed, and a
re-cut re-encodes all seven streams -- `segment` says so itself: "This
re-encodes, so a cut stream is a second H.264 generation", which makes a
re-cut segment a THIRD. It also cost hours and, since `--force` takes no date
range, it started re-encoding 2026-05 data that is out of scope entirely.

The right operation slices the new force columns out of the uncut Z-up episode
and writes them into the segment parquet that already exists. Videos are never
opened.

Alignment is by `source_h5_frame`, never by position. The segment is a
contiguous run of the episode, so a positional guess would usually be right --
and silently wrong for exactly the episodes whose trim moved, which are the
ones worth worrying about.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.react_preprocess.refresh_force import refresh_segment, FORCE_PREFIX


def _episode(path, n=40, first=100, force_scale=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * n
    pq.write_table(pa.table({
        "source_h5_frame": np.arange(first, first + n, dtype=np.int32),
        "frame_idx": np.arange(n, dtype=np.int32),
        "sensor_left_pose": pose,
        "force_left_normal_n": (np.arange(n) * force_scale).astype(np.float32),
        "force_left_penetration_mm": (np.arange(n) * force_scale / 2).astype(np.float32),
        "force_left_target_pose": pose,
        "force_left_source_frame": np.arange(first, first + n, dtype=np.int32),
    }), str(path))
    return path


def _segment(path, lo, hi, first=100):
    """Rows [lo, hi) of the episode, carrying STALE force values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = hi - lo
    pose = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * n
    pq.write_table(pa.table({
        "source_h5_frame": np.arange(first + lo, first + hi, dtype=np.int32),
        "frame_idx": np.arange(n, dtype=np.int32),
        "sensor_left_pose": pose,
        "force_left_normal_n": np.full(n, -1.0, np.float32),      # stale
        "force_left_penetration_mm": np.full(n, -1.0, np.float32),
        "force_left_target_pose": pose,
        "force_left_source_frame": np.full(n, -1, np.int32),
        "task_index": np.zeros(n, np.int64),                      # not force
    }), str(path))
    return path


def test_the_new_force_values_land_on_the_right_rows(tmp_path):
    ep = _episode(tmp_path / "ep.parquet", force_scale=2.0)
    seg = _segment(tmp_path / "seg.parquet", 10, 25)
    refresh_segment(seg, ep)
    t = pq.read_table(str(seg))
    assert np.allclose(t["force_left_normal_n"].to_numpy(),
                       np.arange(10, 25) * 2.0)


def test_every_non_force_column_is_untouched(tmp_path):
    ep = _episode(tmp_path / "ep.parquet", force_scale=2.0)
    seg = _segment(tmp_path / "seg.parquet", 10, 25)
    before = pq.read_table(str(seg))
    refresh_segment(seg, ep)
    after = pq.read_table(str(seg))
    for name in before.column_names:
        if name.startswith(FORCE_PREFIX):
            continue
        assert before[name].equals(after[name]), f"{name} changed"


def test_the_row_count_never_changes(tmp_path):
    ep = _episode(tmp_path / "ep.parquet")
    seg = _segment(tmp_path / "seg.parquet", 10, 25)
    refresh_segment(seg, ep)
    assert pq.read_metadata(str(seg)).num_rows == 15


def test_alignment_is_by_source_frame_not_position(tmp_path):
    """The episode's window moved; a positional slice would be off by 5."""
    ep = _episode(tmp_path / "ep.parquet", first=95, force_scale=2.0)
    seg = _segment(tmp_path / "seg.parquet", 10, 25, first=100)
    refresh_segment(seg, ep)
    t = pq.read_table(str(seg))
    # segment rows are source frames 110..124; in an episode starting at 95
    # those are episode rows 15..29, so the values are 30, 32, ...
    assert np.allclose(t["force_left_normal_n"].to_numpy(),
                       np.arange(15, 30) * 2.0)


def test_a_segment_frame_absent_from_the_episode_refuses(tmp_path):
    """Refusing beats writing a wrong or zero-filled row."""
    ep = _episode(tmp_path / "ep.parquet", n=20)          # frames 100..119
    seg = _segment(tmp_path / "seg.parquet", 10, 25)      # asks up to 124
    with pytest.raises(ValueError, match="source frame"):
        refresh_segment(seg, ep)


def test_an_episode_without_force_columns_refuses(tmp_path):
    ep = _episode(tmp_path / "ep.parquet")
    t = pq.read_table(str(ep))
    t = t.drop([c for c in t.column_names if c.startswith(FORCE_PREFIX)])
    pq.write_table(t, str(ep))
    seg = _segment(tmp_path / "seg.parquet", 10, 25)
    with pytest.raises(ValueError, match="no force columns"):
        refresh_segment(seg, ep)


def test_a_column_the_segment_lacks_is_added(tmp_path):
    """Filling the 27 force-free segments is the same operation."""
    ep = _episode(tmp_path / "ep.parquet", force_scale=2.0)
    seg = _segment(tmp_path / "seg.parquet", 10, 25)
    t = pq.read_table(str(seg))
    t = t.drop(["force_left_target_pose"])
    pq.write_table(t, str(seg))
    refresh_segment(seg, ep)
    assert "force_left_target_pose" in pq.read_schema(str(seg)).names


def test_a_whole_episode_unit_is_not_missed(tmp_path):
    """Not every published unit is named `_segNN`.

    `cut_episode`: "An episode whose only span covers the whole recording is
    copied rather than re-encoded: it has nothing to cut." So the cut tree
    holds `episode_000.parquet` beside `episode_001_seg00.parquet`, and a plan
    that globs only `*_seg*` silently skips the copied ones -- 3 of
    motherboard's 25 published units on 2026-09-17, which would have shipped
    with stale force values while everything around them was refreshed.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "twm" / "scripts"))
    import refresh_release_force as RRF

    cut, zup = tmp_path / "cut", tmp_path / "zup"
    for name in ("episode_000", "episode_001_seg00", "episode_001_seg01"):
        (cut / "rope" / "meta" / "2026-09-16").mkdir(parents=True, exist_ok=True)
        (cut / "rope" / "meta" / "2026-09-16" / f"{name}.parquet").write_bytes(b"x")
    jobs = RRF.plan("rope", "2026-09-10", cut, zup)
    got = {s.stem: src.stem for s, src in jobs}
    assert got == {"episode_000": "episode_000",
                   "episode_001_seg00": "episode_001",
                   "episode_001_seg01": "episode_001"}, got


def test_out_of_scope_dates_are_left_alone(tmp_path):
    """2026-05 is not this run's business, and a --force re-cut ignoring that
    is what started re-encoding it."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "twm" / "scripts"))
    import refresh_release_force as RRF

    cut, zup = tmp_path / "cut", tmp_path / "zup"
    for date in ("2026-05-10", "2026-09-16"):
        d = cut / "rope" / "meta" / date
        d.mkdir(parents=True)
        (d / "episode_000_seg00.parquet").write_bytes(b"x")
    jobs = RRF.plan("rope", "2026-09-10", cut, zup)
    assert {s.parent.name for s, _ in jobs} == {"2026-09-16"}
