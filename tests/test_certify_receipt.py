"""Certify what changed, not what was certified last time.

The alignment half reads ~2000 rows per episode per side out of the SOURCE H5
and compares them pixel by pixel. Over the in-scope window that is 2356 GB
across 52 recordings at about 4.8 GB/min -- two to four hours, every run,
including every episode that passed last time, is already on the Hub, and
whose bytes have not moved since.

Re-reading those re-derives the same answer. What it cannot skip is an episode
whose parquet CHANGED, because a re-cut under the same name is exactly the case
a receipt keyed on names would wave through.

So the receipt records the parquet's content hash at the moment the episode
passed. An episode is re-certified when its hash is absent or different, and
nothing else is read at all.
"""
import hashlib
import json

import pytest

import twm.scripts.certify_release as C


def _parquet(p, payload=b"abc"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return p


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RELEASE", tmp_path)
    for name in ("episode_000_seg00", "episode_001_seg00"):
        _parquet(tmp_path / "rope" / "meta" / "2026-09-11" / f"{name}.parquet")
    return tmp_path


def test_with_no_receipt_everything_is_certified(tree):
    todo = C.needs_certifying("rope")
    assert len(todo) == 2


def test_an_unchanged_episode_is_skipped_next_time(tree):
    C.write_receipt("rope", C.needs_certifying("rope"))
    assert C.needs_certifying("rope") == []


def test_a_recut_episode_under_the_same_name_is_certified_again(tree):
    """The case a name-keyed receipt waves through: same key, new bytes."""
    C.write_receipt("rope", C.needs_certifying("rope"))
    _parquet(tree / "rope/meta/2026-09-11/episode_000_seg00.parquet", b"different")
    todo = C.needs_certifying("rope")
    assert todo == ["2026-09-11/episode_000_seg00"]


def test_a_new_episode_is_certified_and_the_old_ones_are_not(tree):
    C.write_receipt("rope", C.needs_certifying("rope"))
    _parquet(tree / "rope/meta/2026-09-14/episode_000_seg00.parquet", b"new")
    assert C.needs_certifying("rope") == ["2026-09-14/episode_000_seg00"]


def test_the_receipt_only_records_what_actually_passed(tree):
    """A failing episode must not be stamped, or the next run skips the very
    thing that was wrong."""
    C.write_receipt("rope", ["2026-09-11/episode_000_seg00"])
    todo = C.needs_certifying("rope")
    assert todo == ["2026-09-11/episode_001_seg00"]


def test_a_forced_run_ignores_the_receipt(tree):
    C.write_receipt("rope", C.needs_certifying("rope"))
    assert len(C.needs_certifying("rope", force=True)) == 2


def test_the_scope_filter_still_applies(tree, monkeypatch):
    monkeypatch.setattr(C, "SINCE", "2026-09-12")
    assert C.needs_certifying("rope") == []
