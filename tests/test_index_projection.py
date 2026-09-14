"""The published index must describe the published episodes -- nothing else.

Each test names a defect that reached the Hub on 2026-09-12/13.
"""
from __future__ import annotations

import json

import pytest

from twm.react_preprocess import index_projection as P


@pytest.fixture
def provenance():
    """Shaped like the real file: a body listing segments, plus a summary that
    describes the last `segment` invocation rather than the body."""
    return {
        "summary": {"task": "motherboard", "episodes": 0, "raw_frames": 0,
                    "kept_frames": 0, "kept_minutes": 0.0,
                    "discarded_frames": 0, "kept_fraction": 0.0,
                    "min_publish_seconds": 30.0, "skipped_already_cut": 11,
                    "unreadable": []},
        "thresholds": {"tau_intensity": 30.0},
        "min_publish_seconds": 30.0,
        "dropped_spans": [
            {"episode": "2026-09-11/episode_000", "frame_range": [0, 99], "n_frames": 100},
            {"episode": "2026-06-18/episode_000", "frame_range": [0, 9], "n_frames": 10},
        ],
        "segments": [
            {"episode": "2026-09-11/episode_000_seg00", "n_frames": 900, "duration_s": 30.0},
            {"episode": "2026-06-18/episode_000_seg00", "n_frames": 1800, "duration_s": 60.0},
        ],
    }


def test_a_fully_cut_task_does_not_report_zero_episodes(provenance):
    """The published motherboard file said `episodes: 0` beside 14 segments,
    because the cutter had nothing left to do on its last run."""
    out = P.project_provenance(provenance, ["2026-09-11/episode_000_seg00"])
    assert out["summary"]["published_segments"] == 1
    assert out["summary"]["kept_frames"] == 900
    assert out["summary"]["kept_minutes"] == 0.5


def test_the_invocation_counter_is_gone(provenance):
    out = P.project_provenance(provenance, ["2026-09-11/episode_000_seg00"])
    assert "skipped_already_cut" not in out["summary"]
    assert "episodes" not in out["summary"]


def test_locally_cut_but_unpublished_episodes_drop_out(provenance):
    """pushT has 11 segments cut from 2026-06-18 that the release does not
    publish. An index that lists them points downloaders at missing files."""
    out = P.project_provenance(provenance, ["2026-09-11/episode_000_seg00"])
    assert [s["episode"] for s in out["segments"]] == ["2026-09-11/episode_000_seg00"]
    assert [d["episode"] for d in out["dropped_spans"]] == ["2026-09-11/episode_000"]


def test_discarded_frames_come_from_the_kept_source_recordings(provenance):
    out = P.project_provenance(provenance, ["2026-09-11/episode_000_seg00"])
    s = out["summary"]
    assert s["discarded_frames"] == 100
    assert s["raw_frames"] == 1000
    assert s["kept_fraction"] == 0.9


def test_a_dropped_span_matches_its_segment_across_the_seg_suffix():
    """`dropped_spans` is keyed by the source recording, `segments` by the
    published segment. Matching them literally drops every span."""
    assert P._source_forms(["2026-09-11/episode_004_seg02"]) == {
        "2026-09-11/episode_004_seg02", "2026-09-11/episode_004"}


def test_an_uncut_episode_key_is_its_own_source():
    assert P._source_forms(["2026-09-09/episode_000"]) == {"2026-09-09/episode_000"}


def test_seg_in_a_name_that_is_not_a_suffix_is_not_stripped():
    assert P._source_forms(["d/episode_segment"]) == {"d/episode_segment"}


def test_segments_index_totals_follow_the_projection():
    doc = {"task": "pushT", "n_segments": 69, "total_frames": 150964,
           "total_duration_min": 83.87,
           "segments": [
               {"source_episode": "2026-09-12/e_seg00", "segment_idx": 0,
                "n_frames": 900, "duration_s": 30.0},
               {"source_episode": "2026-06-18/e_seg00", "segment_idx": 0,
                "n_frames": 1800, "duration_s": 60.0}]}
    out = P.project_segments(doc, ["2026-09-12/e_seg00"])
    assert out["n_segments"] == 1
    assert out["total_frames"] == 900
    assert out["total_duration_min"] == 0.5


def test_bad_frames_summary_follows_the_projection():
    doc = {"task": "rope", "summary": {"n_episodes": 66, "total_frames": 151003,
                                       "total_bad_frames": 36,
                                       "bad_fraction_overall": 0.0002},
           "episodes": {"a/e": {"n_frames": 100, "total_bad_frames": 5},
                        "b/e": {"n_frames": 900, "total_bad_frames": 0}}}
    out = P.project_bad_frames(doc, ["a/e"])
    assert out["summary"] == {"n_episodes": 1, "total_frames": 100,
                              "total_bad_frames": 5, "bad_fraction_overall": 0.05}


def test_episodes_jsonl_is_filtered_and_sorted():
    rows = [{"episode": "b/e"}, {"episode": "a/e"}, {"episode": "z/e"}]
    assert P.project_episodes(rows, ["z/e", "a/e"]) == [{"episode": "a/e"},
                                                        {"episode": "z/e"}]


def test_uncovered_names_episodes_a_split_file_forgets():
    """An unlisted episode does not raise -- `_split_filter` returns
    `self.split == "train"`, so it leaks into training in silence."""
    doc = {"episodes": {"a/e": {}}}
    assert P.uncovered(["a/e", "b/e"], doc) == ["b/e"]


def test_write_json_is_atomic(tmp_path):
    p = tmp_path / "x.json"
    P.write_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))
