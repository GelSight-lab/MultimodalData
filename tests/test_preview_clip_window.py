"""The renderer must play the window the PLAN chose, not the source episode's.

`build_one_preview` derived everything from `h5_path`, whose stem is the source
recording (`episode_000`) for every segment cut out of it. So all four segments
of one episode rendered the identical clip -- byte-for-byte identical mp4s under
four different names, each claiming to be a different published episode.

The row-indexed channels (force npz, release poses) stay keyed to the SOURCE
episode: they are stored per recording, and `row_for_h5_frame` maps an H5 frame
through the source trim. Only the sampling window moves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
import build_episode_previews as BEP  # noqa: E402
import build_release_previews as BRP  # noqa: E402


def test_window_starts_at_the_segment_not_the_trim():
    # source episode trimmed at 100, segment begins at 9000, 900-frame clip
    assert BEP.clip_window(100, 9000, 20000, 900) == (9000, 9900)


def test_window_falls_back_to_the_trim_for_an_uncut_episode():
    assert BEP.clip_window(100, None, 20000, 900) == (100, 1000)


def test_window_is_clamped_to_the_recording():
    assert BEP.clip_window(100, 19700, 20000, 900) == (19700, 20000)


def test_renderer_is_told_where_the_segment_starts(monkeypatch):
    seen = {}
    monkeypatch.setattr(BEP, "_parquet_trim_and_rows", lambda *a, **k: (9000, 50))
    monkeypatch.setattr(BEP, "_load_proj_calibs", lambda *a, **k: (["c"], 1, 2, 3))
    monkeypatch.setattr(BEP, "build_one_preview",
                        lambda *a, **k: seen.update(k))
    render, _ = BRP.make_renderer("pushT", 30.0, 2.0)
    render({"task": "pushT", "date": "2026-09-10", "episode": "episode_003_seg01",
            "h5": Path("/x/pushT/2026-09-10/episode_003.h5"),
            "out": Path("/x/out.mp4"), "world_offset": (0.0, 0.0, 0.0),
            "parquet": Path("/x/seg.parquet"), "trim_offset": 9000})
    assert seen["window_start"] == 9000
