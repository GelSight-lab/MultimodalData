"""A published episode's force must come from the file users get.

`ReleaseEpisode.forces` looked for `<episode>_<side>.npz` under the force
recovery tree. Published units are SEGMENTS -- `episode_006_seg00` -- while the
npz are named for the SOURCE recording, `episode_006_left.npz`. The lookup
therefore missed on every segment and returned {}, so `twm.visualize` played a
published segment with no force overlay at all, silently: an empty dict is
indistinguishable from "this episode has no force channel".

Since 2026-09-17 the segment parquet carries `force_<side>_normal_n` itself,
and that is the authority -- it is what a dataset user reads. The npz stays as
a fallback for an uncut episode, where it is still the only source.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.release_episode import ReleaseEpisode


def _episode(tmp_path, *, force_cols=True, n=12):
    md = tmp_path / "meta" / "2026-09-11"
    md.mkdir(parents=True)
    cols = {
        "frame_idx": np.arange(n, dtype=np.int32),
        "timestamp": 100.0 + np.arange(n) / 30.0,
    }
    if force_cols:
        cols["force_left_normal_n"] = np.linspace(0, 11, n, dtype=np.float32)
        cols["force_right_normal_n"] = np.linspace(11, 0, n, dtype=np.float32)
    p = md / "episode_006_seg00.parquet"
    pq.write_table(pa.table(cols), str(p))
    vd = tmp_path / "videos" / "2026-09-11" / "episode_006_seg00"
    vd.mkdir(parents=True)
    return ReleaseEpisode("motherboard", "2026-09-11", "episode_006_seg00",
                          vd, p, force_root=tmp_path / "no_npz")


def test_a_segment_reads_its_force_from_its_own_parquet(tmp_path):
    e = _episode(tmp_path)
    assert set(e.forces) == {"left", "right"}
    assert e.forces_at(0)["left"] == pytest.approx(0.0)
    assert e.forces_at(11)["left"] == pytest.approx(11.0)


def test_the_npz_is_not_consulted_when_the_parquet_has_the_columns(tmp_path):
    """The parquet is what ships; a stale npz beside it must not win."""
    e = _episode(tmp_path)
    fr = tmp_path / "npz" / "motherboard" / "2026-09-11"
    fr.mkdir(parents=True)
    np.savez(fr / "episode_006_seg00_left.npz",
             force_normal_n=np.full(12, 99.0))
    e.force_root = tmp_path / "npz"
    e._forces = None
    assert e.forces_at(5)["left"] != pytest.approx(99.0)


def test_an_episode_without_force_columns_falls_back_to_the_npz(tmp_path):
    """Uncut episodes predate the columns and still only have the npz."""
    e = _episode(tmp_path, force_cols=False)
    fr = tmp_path / "npz" / "motherboard" / "2026-09-11"
    fr.mkdir(parents=True)
    np.savez(fr / "episode_006_seg00_left.npz",
             force_normal_n=np.arange(12, dtype=float))
    e.force_root = tmp_path / "npz"
    e._forces = None
    assert e.forces_at(4)["left"] == pytest.approx(4.0)


def test_no_force_anywhere_is_still_empty(tmp_path):
    e = _episode(tmp_path, force_cols=False)
    assert e.forces_at(3) == {}
