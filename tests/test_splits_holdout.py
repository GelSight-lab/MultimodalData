"""An episode absent from splits.json silently becomes training data.

`ReactVideoDataset._split_filter` returns `self.split == "train"` for a key
it cannot find, with no warning. So a release whose `episodes.jsonl` lists an
episode that `splits.json` does not is a training-set leak that leaves no
trace in any metric — which is exactly the state the motherboard tree was in
after the 2026-09-09 session was built into it and published elsewhere as a
validation set.
"""
import numpy as np
import pytest

from twm.splits import build_splits

LONG = [{"episode": f"2026-05-10/episode_{i:03d}", "n_frames": 6000} for i in range(3)]
HELD = {"episode": "2026-09-09/episode_000", "n_frames": 6661}


def test_a_long_episode_is_normally_split_into_intervals():
    d = build_splits(LONG + [HELD])
    e = d["episodes"][HELD["episode"]]
    assert e["whole"] is None
    assert e["test"], "a long episode gets held-out intervals, most frames train"


def test_a_held_out_episode_is_whole_test_however_long_it_is():
    d = build_splits(LONG + [HELD], hold_out=[HELD["episode"]])
    e = d["episodes"][HELD["episode"]]
    assert e["whole"] == "test"
    assert e["test"] == [[0, HELD["n_frames"] - 1]]
    assert e["guard"] == []


def test_holding_one_out_does_not_move_the_others():
    a = build_splits(LONG + [HELD])["episodes"]
    b = build_splits(LONG + [HELD], hold_out=[HELD["episode"]])["episodes"]
    for k in (e["episode"] for e in LONG):
        assert a[k] == b[k]


def test_the_held_out_frames_count_as_test_in_the_stats():
    d = build_splits(LONG + [HELD], hold_out=[HELD["episode"]])
    assert d["stats"]["n_whole_test_episodes"] == 1
    assert d["stats"]["n_test_frames"] >= HELD["n_frames"]


def test_an_unknown_hold_out_key_is_refused():
    with pytest.raises(KeyError, match="2027-01-01/episode_000"):
        build_splits(LONG, hold_out=["2027-01-01/episode_000"])
