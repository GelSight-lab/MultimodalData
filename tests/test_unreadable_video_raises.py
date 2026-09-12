"""A stream that will not decode must raise, not score as clean.

`_video_stats` took ffmpeg's stdout unconditionally. A truncated file — one
ffmpeg was killed before writing its trailer, so the container has no moov
atom — produced zero bytes, which became a 0-frame video with no frame
differences and therefore NO corruption reported. Two of the eight 2026-09-11
motherboard episodes were in that state and passed curation silently.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from twm.react_preprocess import detect as D


def _truncated_mp4(path):
    """A real H.264 file with its trailer missing — the exact failure seen."""
    frames = np.random.default_rng(0).integers(0, 255, (60, 32, 32, 3), dtype=np.uint8)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])        # cut the trailer off
    return path


def test_a_truncated_stream_raises(tmp_path):
    mp4 = _truncated_mp4(tmp_path / "view_right.mp4")
    with pytest.raises(D.UnreadableVideo, match="did not decode"):
        D._video_stats(mp4)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(D.UnreadableVideo):
        D._video_stats(tmp_path / "nope.mp4")


def test_a_readable_stream_still_returns_stats(tmp_path):
    """The guard must not fire on good video."""
    mp4 = tmp_path / "ok.mp4"
    frames = np.random.default_rng(1).integers(0, 255, (40, 32, 32, 3), dtype=np.uint8)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(mp4)],
        stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())

    fmean, rowfill = D._video_stats(mp4)
    assert len(fmean) == 39 and len(rowfill) == 40
