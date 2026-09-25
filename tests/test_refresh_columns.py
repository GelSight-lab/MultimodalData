"""Widening a cut segment must align by source frame and replace, not layer.

`refresh_force` established the shape: a stage that adds columns to the uncut
episode after the cut was made should not re-encode seven video streams to put
them in the segment. This generalises it to any added column, for the case it
was written for -- pushT/2026-09-17 published with 25 columns where every
other date carries 60, the difference being pose-repair provenance and the
action channel, all metadata.

Two properties are the whole point:

* Alignment is by `source_h5_frame`. A positional slice is right for an
  untrimmed episode and silently wrong for a trimmed one, which is exactly the
  episode whose rows a reader would most want to trust.
* `sensor_*_pose` is REPLACED. That segment's poses were repaired in place
  before the convention settled; leaving them while adding
  `sensor_*_pose_repaired` beside them would publish two different answers for
  one frame.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.react_preprocess.refresh_columns import refresh_segment


def _ep(path, frames, pose_scale=1.0, extra=True):
    n = len(frames)
    cols = {
        "source_h5_frame": np.asarray(frames, np.int64),
        "sensor_left_pose": [[pose_scale * i, 0, 0, 0, 0, 0, 1.0]
                             for i in range(n)],
        "timestamp": np.arange(n, dtype=np.float64),
    }
    if extra:
        cols["sensor_left_pose_repaired"] = [[pose_scale * i + 0.5, 0, 0,
                                              0, 0, 0, 1.0] for i in range(n)]
        cols["action"] = [[float(i)] * 14 for i in range(n)]
    pq.write_table(pa.table(cols), str(path))
    return path


def _seg(path, frames, pose_scale=1.0, flag=True):
    n = len(frames)
    cols = {
        "source_h5_frame": np.asarray(frames, np.int64),
        "sensor_left_pose": [[pose_scale * i, 0, 0, 0, 0, 0, 1.0]
                             for i in range(n)],
        "timestamp": np.arange(n, dtype=np.float64),
        "source_episode": ["2026-09-17/episode_000"] * n,
        "source_frame_idx": np.arange(n, dtype=np.int64),
    }
    if flag:
        cols["pose_interpolated"] = np.zeros(n, bool)
    pq.write_table(pa.table(cols), str(path))
    return path


def test_the_new_columns_arrive(tmp_path):
    ep = _ep(tmp_path / "ep.parquet", range(100))
    seg = _seg(tmp_path / "seg.parquet", range(10, 20))
    r = refresh_segment(seg, ep)
    out = pq.read_table(str(seg))
    assert {"sensor_left_pose_repaired", "action"} <= set(out.column_names)
    assert r["rows"] == 10


def test_rows_come_from_the_matching_source_frames(tmp_path):
    """Not from the first ten rows of the episode."""
    ep = _ep(tmp_path / "ep.parquet", range(100))
    seg = _seg(tmp_path / "seg.parquet", range(40, 50))
    refresh_segment(seg, ep)
    out = pq.read_table(str(seg))
    got = np.array(out["action"].to_pylist(), float)[:, 0]
    assert got.tolist() == list(range(40, 50))


def test_a_trimmed_episode_still_aligns(tmp_path):
    """The episode's row 0 is source frame 7; a positional slice would skew."""
    ep = _ep(tmp_path / "ep.parquet", range(7, 107))
    seg = _seg(tmp_path / "seg.parquet", range(40, 50))
    refresh_segment(seg, ep)
    out = pq.read_table(str(seg))
    got = np.array(out["action"].to_pylist(), float)[:, 0]
    assert got.tolist() == [f - 7 for f in range(40, 50)]


def test_the_segment_pose_is_replaced_not_kept(tmp_path):
    ep = _ep(tmp_path / "ep.parquet", range(100), pose_scale=1.0)
    seg = _seg(tmp_path / "seg.parquet", range(10, 20), pose_scale=99.0)
    refresh_segment(seg, ep)
    out = pq.read_table(str(seg))
    pose = np.array(out["sensor_left_pose"].to_pylist(), float)[:, 0]
    assert pose.tolist() == [float(f) for f in range(10, 20)]


def test_the_segment_keeps_its_own_columns(tmp_path):
    ep = _ep(tmp_path / "ep.parquet", range(100))
    seg = _seg(tmp_path / "seg.parquet", range(10, 20))
    refresh_segment(seg, ep)
    out = pq.read_table(str(seg))
    assert set(out["source_episode"].to_pylist()) == {"2026-09-17/episode_000"}
    assert out["source_frame_idx"].to_pylist() == list(range(10))


def test_a_named_column_is_dropped(tmp_path):
    """`pose_interpolated` said what `pose_*_repaired` now says."""
    ep = _ep(tmp_path / "ep.parquet", range(100))
    seg = _seg(tmp_path / "seg.parquet", range(10, 20))
    r = refresh_segment(seg, ep, drop=("pose_interpolated",))
    out = pq.read_table(str(seg))
    assert "pose_interpolated" not in out.column_names
    assert r["dropped"] == ["pose_interpolated"]


def test_a_source_frame_the_episode_lacks_is_refused(tmp_path):
    """Zero-filling or nearest-matching here would be an invented row."""
    ep = _ep(tmp_path / "ep.parquet", range(0, 30))
    seg = _seg(tmp_path / "seg.parquet", range(40, 50))
    with pytest.raises(ValueError, match="refusing to guess"):
        refresh_segment(seg, ep)


def test_it_refuses_without_the_alignment_key(tmp_path):
    ep = _ep(tmp_path / "ep.parquet", range(100))
    n = 10
    bare = tmp_path / "bare.parquet"
    pq.write_table(pa.table({"timestamp": np.arange(n, dtype=np.float64)}),
                   str(bare))
    with pytest.raises(ValueError, match="cannot align without guessing"):
        refresh_segment(bare, ep)
