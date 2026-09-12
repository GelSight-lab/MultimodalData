"""Ship previews as they finish, alongside the renders that produce them.

Two ways to get this wrong, both of which publish something broken:

* uploading a file ffmpeg is still writing -- a preview with no moov atom,
  which no player opens;
* uploading a preview for an episode this folder does not publish. The local
  tree holds motherboard's 2026-09-09 previews, whose episodes live in
  `data/validation` under different numbers; sending them to
  `data/motherboard` would name previews for episodes that are not there.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
from upload_previews_incremental import pending  # noqa: E402


def test_only_published_episodes_are_sent():
    got = pending(local={"2026-09-09/episode_000_seg00", "2026-09-11/episode_000"},
                  remote=set(),
                  publishes={"2026-09-11/episode_000"})
    assert got == ["2026-09-11/episode_000"]


def test_what_is_already_up_is_not_resent():
    got = pending(local={"2026-09-11/a", "2026-09-11/b"},
                  remote={"2026-09-11/a"},
                  publishes={"2026-09-11/a", "2026-09-11/b"})
    assert got == ["2026-09-11/b"]


def test_nothing_to_do_is_empty_not_an_error():
    assert pending(local=set(), remote=set(), publishes={"x/y"}) == []


def test_a_published_episode_whose_preview_has_not_rendered_yet_is_skipped():
    """It is not an error -- the renderer has simply not reached it."""
    got = pending(local={"2026-09-11/a"}, remote=set(),
                  publishes={"2026-09-11/a", "2026-09-11/b"})
    assert got == ["2026-09-11/a"]


def test_order_is_stable():
    got = pending(local={f"2026-09-11/e{i}" for i in range(5)}, remote=set(),
                  publishes={f"2026-09-11/e{i}" for i in range(5)})
    assert got == sorted(got)
