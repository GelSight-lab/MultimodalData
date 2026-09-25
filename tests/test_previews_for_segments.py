"""A preview must exist for what is PUBLISHED, and start where it starts.

Published episodes are now cut segments (`episode_003_seg00`), which have no
H5 of their own — their frames come from the source recording, beginning part
way in. Planning previews from the source tree would name them after episodes
nobody can download; taking the source episode's trim would start every
segment's preview at the same place, showing the first segment's content for
all of them.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.react_preprocess import previews


def _published(root, task, date, episode, first_source_frame, n=50):
    vd = root / task / "videos" / date / episode
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "view_left.mp4").write_bytes(b"x")
    md = root / task / "meta" / date
    md.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "source_h5_frame": (np.arange(n) + first_source_frame).astype(np.int32),
    }), str(md / f"{episode}.parquet"))


def _h5(root, task, date, episode):
    d = root / task / date
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{episode}.h5").write_bytes(b"x")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    stage, h5root = tmp_path / "release", tmp_path / "data"
    monkeypatch.setattr(previews, "H5_ROOTS", {"pushT": h5root / "pushT"})
    monkeypatch.setattr(previews, "calib_dir", lambda task, date=None: tmp_path / "calib")
    return stage, h5root


def test_a_segment_is_planned_against_its_source_recording(rig):
    stage, h5root = rig
    _h5(h5root, "pushT", "2026-09-10", "episode_003")
    _published(stage, "pushT", "2026-09-10", "episode_003_seg00", 120)

    jobs = list(previews.plan("pushT", stage))
    assert len(jobs) == 1
    assert jobs[0]["h5"].name == "episode_003.h5"
    assert jobs[0]["source_episode"] == "episode_003"
    assert jobs[0]["out"].name == "episode_003_seg00.mp4"


def test_each_segment_starts_where_it_starts(rig):
    """Not at the source episode's trim — otherwise every segment's preview
    shows the same opening seconds."""
    stage, h5root = rig
    _h5(h5root, "pushT", "2026-09-10", "episode_003")
    _published(stage, "pushT", "2026-09-10", "episode_003_seg00", 120)
    _published(stage, "pushT", "2026-09-10", "episode_003_seg01", 9000)

    starts = {j["episode"]: j["trim_offset"] for j in previews.plan("pushT", stage)}
    assert starts == {"episode_003_seg00": 120, "episode_003_seg01": 9000}


def test_an_uncut_episode_still_works(rig):
    stage, h5root = rig
    _h5(h5root, "pushT", "2026-09-10", "episode_000")
    _published(stage, "pushT", "2026-09-10", "episode_000", 31)

    j = list(previews.plan("pushT", stage))[0]
    assert j["source_episode"] == "episode_000" and j["trim_offset"] == 31


def test_a_segment_whose_source_recording_is_gone_is_skipped(rig):
    """The deleted sessions must not reappear as preview jobs."""
    stage, _ = rig
    _published(stage, "pushT", "2026-06-18", "episode_000_seg00", 0)
    assert list(previews.plan("pushT", stage)) == []
