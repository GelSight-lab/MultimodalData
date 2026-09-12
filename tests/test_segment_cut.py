"""The segment stage must cut exactly where curation said, or not at all.

The stage's whole claim is that every published frame passed every detector.
That claim rests on two things being true: the span filter drops what it says
it drops, and the video cut lands on the frame `select` was given rather than
near it. Both are tested against real ffmpeg output, because an off-by-one in
the filter expression is invisible in any test that mocks the encoder.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.react_preprocess import segment as S

FPS = 30.0


def _mark(i: int) -> np.ndarray:
    """A frame stamped with its own index, in saturated black and white.

    The index is written as 16 binary pixels rather than as a grey level. A
    grey ramp does not survive the round trip: BGR -> YUV444 -> BGR rounds, so
    a 20-frame lossless probe read back as 0,1,2,3,3,5,... and made the cut
    look like it was duplicating frames when it was not. 0 and 255 convert
    exactly, so a bit pattern reads back bit-for-bit.
    """
    f = np.zeros((32, 32, 3), np.uint8)
    for bit in range(16):
        if i >> bit & 1:
            f[0, bit] = 255
    return f


def _write_marked_video(path: Path, n: int) -> None:
    """A video whose frame i carries its own index in its pixels, so a cut can
    be checked for landing on the right frames and not merely the right number
    of them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([_mark(i) for i in range(n)])
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "32x32", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-profile:v", "high444", "-preset", "ultrafast",
         "-crf", "0", "-pix_fmt", "yuv444p", "-an", str(path)],
        stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    assert p.returncode == 0


def _read_marks(path: Path) -> list[int]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True).stdout
    a = np.frombuffer(out, np.uint8).reshape(-1, 32, 32, 3)
    bits = a[:, 0, :16, 0] > 127
    return [int(sum(1 << b for b in range(16) if row[b])) for row in bits]


def _table(n: int, trim: int = 7) -> pa.Table:
    return pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "timestamp": np.arange(n, dtype=np.float64) / FPS + 1000.0,
        "tactile_left_intensity": np.arange(n, dtype=np.float32),
        "source_h5_frame": (np.arange(n) + trim).astype(np.int32),
    })


# ── span selection ───────────────────────────────────────────────────────────

def test_spans_shorter_than_the_floor_are_not_published():
    report = {"n_frames": 3000, "intensity_spikes": [[1000, 1010]],
              "tactile_freeze_L": [[1100, 1110]]}
    report = {**{k: [] for k in S.curation.BAD_KEYS}, **report}

    # Spans are [0,999], [1011,1099], [1111,2999]; the middle one is 89 frames.
    assert S.publishable_spans(report, min_frames=900) == [(0, 999), (1111, 2999)]
    assert (1011, 1099) in S.publishable_spans(report, min_frames=16)


def test_the_default_floor_is_thirty_seconds():
    """The operator chose 30s over 10s on 2026-09-11; 10s kept 3.4 more minutes
    but shipped ten episodes too short to hold a complete attempt."""
    assert S.MIN_PUBLISH_SECONDS == 30.0
    assert S.MIN_PUBLISH_FRAMES == 900


def test_an_episode_with_no_defects_yields_one_whole_span():
    report = {**{k: [] for k in S.curation.BAD_KEYS}, "n_frames": 5000}
    assert S.publishable_spans(report) == [(0, 4999)]


# ── the video cut ────────────────────────────────────────────────────────────

def test_the_cut_lands_on_exactly_the_frames_curation_flagged(tmp_path):
    """Frame-exactness, checked by content rather than by count.

    A `-ss` seek would pass a count check while starting a frame or two early.
    """
    src = tmp_path / "src.mp4"
    _write_marked_video(src, 120)
    dsts = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    S.cut_video(src, [(10, 29), (60, 89)], dsts)

    assert _read_marks(dsts[0]) == list(range(10, 30))
    assert _read_marks(dsts[1]) == list(range(60, 90))


def test_every_span_comes_out_of_a_single_decode_pass(tmp_path, monkeypatch):
    """The source is decoded once per stream, not once per span: the 12-span
    pushT episode would otherwise re-decode each of its 7 streams twelve
    times."""
    src = tmp_path / "src.mp4"
    _write_marked_video(src, 60)
    spawned = []
    real = S.subprocess.Popen
    monkeypatch.setattr(S.subprocess, "Popen",
                        lambda cmd, **kw: spawned.append(cmd) or real(cmd, **kw))
    S.cut_video(src, [(0, 9), (20, 29), (40, 49)],
                [tmp_path / f"{i}.mp4" for i in range(3)])

    ffmpeg = [c for c in spawned if c[0] == "ffmpeg"]
    reads = [c for c in ffmpeg if str(src) in c]
    assert len(reads) == 1            # the source is opened once, not per span
    assert len(ffmpeg) == 4           # that decode, plus one encoder per span


def test_frame_count_is_measured_by_decoding(tmp_path):
    src = tmp_path / "src.mp4"
    _write_marked_video(src, 47)
    assert S.frame_count(src) == 47


# ── the parquet cut ──────────────────────────────────────────────────────────

def test_rows_are_renumbered_and_provenance_is_added():
    sub = S.cut_table(_table(100), 30, 59, "2026-09-10/episode_003",
                      "episode_003_seg01")

    assert sub.num_rows == 30
    assert sub.column("frame_idx").to_pylist() == list(range(30))
    assert sub.column("source_frame_idx").to_pylist() == list(range(30, 60))
    assert set(sub.column("source_episode").to_pylist()) == {"2026-09-10/episode_003"}


def test_timestamp_and_h5_frame_are_not_rebased():
    """Both are statements about the recording, and stay true of a slice.
    Rebasing `timestamp` would make every segment claim to start at t=0 and
    destroy the only link between a segment and the OptiTrack log."""
    full = _table(100, trim=7)
    sub = S.cut_table(full, 30, 59, "d/episode_000", "episode_000_seg00")

    assert sub.column("timestamp").to_pylist() == full.column("timestamp").to_pylist()[30:60]
    assert sub.column("source_h5_frame").to_pylist() == list(range(37, 67))


def test_index_columns_are_rewritten_only_when_they_already_exist():
    """`episode_index` is assigned by the enrichment step over the whole
    release; inventing one here would collide with it."""
    plain = S.cut_table(_table(50), 0, 9, "d/episode_000", "episode_000_seg00")
    assert "frame_index" not in plain.column_names
    assert "episode" not in plain.column_names

    enriched = _table(50).append_column(
        "frame_index", pa.array(np.arange(50, dtype=np.int64))).append_column(
        "episode", pa.array(["episode_000"] * 50, pa.string()))
    cut = S.cut_table(enriched, 20, 29, "d/episode_000", "episode_000_seg01")
    assert cut.column("frame_index").to_pylist() == list(range(10))
    assert set(cut.column("episode").to_pylist()) == {"episode_000_seg01"}


def test_the_table_keeps_its_payload_columns():
    sub = S.cut_table(_table(100), 30, 59, "d/episode_000", "episode_000_seg00")
    assert sub.column("tactile_left_intensity").to_pylist() == [
        float(i) for i in range(30, 60)]


# ── one episode, end to end ──────────────────────────────────────────────────

def _build_episode(root: Path, date: str, ep: str, n: int,
                   streams=S.EXPECTED_STREAMS) -> None:
    for s in streams:
        _write_marked_video(root / "videos" / date / ep / f"{s}.mp4", n)
    out = root / "meta" / date / f"{ep}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_table(n), str(out))


def test_an_incomplete_build_is_refused_rather_than_published(tmp_path):
    src, dst = tmp_path / "rel", tmp_path / "cut"
    _build_episode(src, "2026-09-10", "episode_000", 40,
                   streams=[s for s in S.EXPECTED_STREAMS if s != "wrist_right"])

    with pytest.raises(FileNotFoundError, match="wrist_right"):
        S.cut_episode("pushT", "2026-09-10", "episode_000", [(0, 39)], src, dst)


def test_a_defect_free_episode_is_copied_not_re_encoded(tmp_path):
    """Re-encoding it would spend a generation of H.264 quality to produce the
    same frames, and would rename it for no reason."""
    src, dst = tmp_path / "rel", tmp_path / "cut"
    _build_episode(src, "2026-09-10", "episode_000", 40)
    rows = S.cut_episode("pushT", "2026-09-10", "episode_000", [(0, 39)],
                         src, dst, verify=False)

    assert [r["episode"] for r in rows] == ["2026-09-10/episode_000"]
    assert rows[0]["recut"] is False
    a = (src / "videos/2026-09-10/episode_000/view_left.mp4").read_bytes()
    b = (dst / "videos/2026-09-10/episode_000/view_left.mp4").read_bytes()
    assert a == b


def test_a_cut_episode_yields_one_episode_per_span(tmp_path):
    src, dst = tmp_path / "rel", tmp_path / "cut"
    _build_episode(src, "2026-09-10", "episode_003", 100)
    rows = S.cut_episode("pushT", "2026-09-10", "episode_003",
                         [(0, 39), (60, 99)], src, dst)

    assert [r["episode"] for r in rows] == [
        "2026-09-10/episode_003_seg00", "2026-09-10/episode_003_seg01"]
    assert [r["source_frame_range"] for r in rows] == [[0, 39], [60, 99]]
    for r, first in zip(rows, (0, 60)):
        name = r["episode"].split("/")[1]
        vid = dst / "videos/2026-09-10" / name
        assert sorted(p.stem for p in vid.glob("*.mp4")) == sorted(S.EXPECTED_STREAMS)
        assert _read_marks(vid / "tactile_left.mp4")[0] == first
        assert pq.read_table(
            str(dst / "meta/2026-09-10" / f"{name}.parquet")).num_rows == 40


def test_a_stream_that_came_out_the_wrong_length_stops_the_cut(tmp_path):
    """The stage promises row i of the parquet is frame i of every MP4. If a
    written stream disagrees, the segment must not be reported as written."""
    src, dst = tmp_path / "rel", tmp_path / "cut"
    _build_episode(src, "2026-09-10", "episode_003", 100)
    real = S.frame_count
    monkey = lambda p: real(p) - 1 if p.stem == "wrist_right" else real(p)  # noqa: E731
    S.frame_count, saved = monkey, S.frame_count
    try:
        with pytest.raises(RuntimeError, match="wrist_right"):
            S.cut_episode("pushT", "2026-09-10", "episode_003",
                          [(0, 39)], src, dst)
    finally:
        S.frame_count = saved


def test_a_tree_whose_frame_numbering_differs_is_refused(tmp_path):
    """The spans are measured on the master and applied to the published tree.
    Every stage between them preserves row count and row order; if one ever
    stops doing so, the cut lands on the wrong frames and nothing else notices.
    """
    src, dst = tmp_path / "rel", tmp_path / "cut"
    _build_episode(src, "2026-09-10", "episode_000", 40)

    with pytest.raises(RuntimeError, match="frame numbering differs"):
        S.cut_episode("pushT", "2026-09-10", "episode_000", [(0, 39)],
                      src, dst, verify=False, expected_T=41)


def test_it_cuts_what_the_published_tree_holds_not_what_was_detected(tmp_path):
    """A partial wave publishes fewer episodes than have been detected.

    Discovering by sidecar would try to cut episodes the tree being published
    does not contain — on the 2026-09 backlog, 33 sidecars against the 20
    episodes built so far.
    """
    src, det, dst = tmp_path / "wave", tmp_path / "rel", tmp_path / "cut"
    _build_episode(src / "pushT", "2026-09-10", "episode_000", 40)   # in the wave
    _build_episode(det / "pushT", "2026-09-10", "episode_000", 40)
    _build_episode(det / "pushT", "2026-09-10", "episode_999", 40)   # detected only
    for d in (det / "pushT/meta/2026-09-10").glob("*.parquet"):
        d.with_name(d.stem + "._detect.pt").write_text("x")

    seen = []
    real = S.curation.episode_report
    S.curation.episode_report = lambda p, video_dir=None: (
        seen.append(p.name) or ({**{k: [] for k in S.curation.BAD_KEYS},
                                 "n_frames": 40}, {}))
    try:
        S.build_task("pushT", src, dst, min_frames=16,
                     verify=False, detect_root=det)
    finally:
        S.curation.episode_report = real

    assert seen == ["episode_000._detect.pt"]


def test_an_episode_in_the_wave_with_no_sidecar_is_refused(tmp_path):
    """Publishing it uncut would ship exactly the defects this stage removes."""
    src, det, dst = tmp_path / "wave", tmp_path / "rel", tmp_path / "cut"
    _build_episode(src / "pushT", "2026-09-10", "episode_000", 40)
    (det / "pushT/meta/2026-09-10").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no _detect.pt"):
        S.build_task("pushT", src, dst, min_frames=16,
                     verify=False, detect_root=det)
