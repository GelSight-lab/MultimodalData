"""A session recorded before the wrist cameras existed is not a broken build.

`cut_episode` refuses an episode missing any of EXPECTED_STREAMS, so that a
half-built episode cannot be published with a stream silently absent. Right
intent, wrong test: the 2026-05 sessions predate the wrist cameras entirely,
so all 32 motherboard episodes from them are refused and the task cuts to
nothing (`kept_fraction: 0.0`).

`dataset_layout` already draws the line correctly — no wrist videos at all is a
warning ("recorded before the Arducams were wired in"), exactly ONE of the two
is a failure — and this has to agree with it, or the two gates disagree about
what a complete episode is.
"""
import pytest


def _streams(tmp_path, names):
    d = tmp_path / "videos" / "2026-05-10" / "episode_000"
    d.mkdir(parents=True)
    for n in names:
        (d / f"{n}.mp4").write_bytes(b"x")
    return d


CORE = ("view_left", "view_middle", "view_right", "tactile_left", "tactile_right")
WRIST = ("wrist_left", "wrist_right")


def test_missing_both_wrist_streams_is_accepted(tmp_path):
    from twm.react_preprocess.segment import missing_streams
    assert missing_streams(_streams(tmp_path, CORE)) == []


def test_all_seven_streams_is_accepted(tmp_path):
    from twm.react_preprocess.segment import missing_streams
    assert missing_streams(_streams(tmp_path, CORE + WRIST)) == []


def test_exactly_one_wrist_stream_is_still_refused(tmp_path):
    """Half a wrist pair means the build broke, not that the rig lacked one."""
    from twm.react_preprocess.segment import missing_streams
    assert missing_streams(_streams(tmp_path, CORE + ("wrist_left",))) == ["wrist_right"]


def test_a_missing_core_stream_is_still_refused(tmp_path):
    from twm.react_preprocess.segment import missing_streams
    got = missing_streams(_streams(tmp_path, [c for c in CORE if c != "view_middle"]))
    assert got == ["view_middle"]


def test_the_cut_loop_only_touches_streams_that_exist(tmp_path):
    """Accepting an episode without wrist videos is not enough: the cut then
    iterated EXPECTED_STREAMS anyway and died opening the wrist file it had
    just agreed was absent. All 30 pre-wrist motherboard episodes were refused
    a second time, at a different line, with `0 片段` as the result.
    """
    from twm.react_preprocess.segment import present_streams
    assert present_streams(_streams(tmp_path, CORE)) == list(CORE)
    d = _streams(tmp_path / "b", CORE + WRIST)
    assert present_streams(d) == list(CORE + WRIST)
