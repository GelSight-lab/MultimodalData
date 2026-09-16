"""`calib_epoch` reads episodes.jsonl from a root nothing can redirect.

`calib_dir` already honours `REACT_CALIB` and `REACT_RELEASE`. `_episodes`,
which reads the world-frame offset out of `episodes.jsonl`, does not: it uses
a module-level `RELEASE` fixed at import.

Found by running the chain in a sandbox. Every root was redirected, every
stage wrote where it was told — and `export` reached across into the real
`/media/yxma/Disk1/twm/release` and refused, correctly, because the sandbox's
episode is not in the production index.

Two mechanisms for one fact, and the one that matters here is the hard-coded
half. A pipeline that cannot be pointed somewhere else cannot be tested
end-to-end, which is how twenty defects came to be found one at a time on real
data.
"""
import json

import pytest

import twm.calib_epoch as CE


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    r = tmp_path / "release" / "rope"
    r.mkdir(parents=True)
    (r / "episodes.jsonl").write_text(json.dumps(
        {"episode": "2026-09-20/episode_000", "date": "2026-09-20",
         "world_frame_offset": [0.0, 0.0, 0.0], "up_axis": "z"}) + "\n")
    CE._episodes.cache_clear()
    monkeypatch.setenv("REACT_RELEASE", str(tmp_path / "release"))
    yield tmp_path
    CE._episodes.cache_clear()


def test_the_release_root_follows_the_environment(sandbox):
    assert "2026-09-20/episode_000" in CE.release_episodes("rope")


def test_the_world_offset_is_read_from_that_root(sandbox):
    off = CE.world_offset_m("rope", "2026-09-20", "episode_000", up_axis="y")
    assert tuple(off) == (0.0, 0.0, 0.0)


def test_an_episode_absent_from_that_root_still_refuses(sandbox):
    """The refusal is right and must survive the redirection — assuming a zero
    offset is how the 2026-05-19 session would render 175 mm out."""
    with pytest.raises(KeyError, match="episode_099"):
        CE.world_offset_m("rope", "2026-09-20", "episode_099", up_axis="y")
