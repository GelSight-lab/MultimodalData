"""A build in progress must not read as finished.

`--force` leaves the previous run's parquet in place while the new videos are
written, and ffmpeg creates each file as it opens it, so a file-count test
passes throughout the rebuild. The cut consumed a half-written tactile stream
that way and reported the episode as corrupt.
"""
from __future__ import annotations

import os

from twm.react_preprocess.complete import STREAMS, is_complete


def _episode(root, task="motherboard", date="2026-09-11", ep="episode_006",
             n_streams=7):
    vd = root / task / "videos" / date / ep
    vd.mkdir(parents=True, exist_ok=True)
    (root / task / "meta" / date).mkdir(parents=True, exist_ok=True)
    for s in STREAMS[:n_streams]:
        (vd / f"{s}.mp4").write_bytes(b"x")
    (root / task / "meta" / date / f"{ep}._detect.pt").write_bytes(b"sidecar")
    return root / task / "meta" / date / f"{ep}.parquet"


def test_a_finished_build_is_complete(tmp_path):
    pq = _episode(tmp_path)
    pq.write_bytes(b"p")                       # written last, so newest
    assert is_complete(tmp_path, "motherboard", "2026-09-11", "episode_006")


def test_a_rebuild_in_progress_is_not_complete(tmp_path):
    """The exact 05:21 situation: a stale parquet and seven files still
    growing."""
    pq = _episode(tmp_path)
    pq.write_bytes(b"stale")
    old = pq.stat().st_mtime - 3600
    os.utime(pq, (old, old))                   # parquet predates the videos
    assert not is_complete(tmp_path, "motherboard", "2026-09-11", "episode_006")


def test_a_missing_stream_is_not_complete(tmp_path):
    pq = _episode(tmp_path, n_streams=6)
    pq.write_bytes(b"p")
    assert not is_complete(tmp_path, "motherboard", "2026-09-11", "episode_006")


def test_no_parquet_is_not_complete(tmp_path):
    _episode(tmp_path)
    assert not is_complete(tmp_path, "motherboard", "2026-09-11", "episode_006")
