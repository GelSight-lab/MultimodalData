"""A stage must run after everything it reads.

`export` writes the force columns into the published parquet, and to do that it
stamps a world-frame declaration -- which needs the episode's world offset,
which `calib_epoch.world_offset_m` reads out of `release/<task>/episodes.jsonl`.
That file is written by `curate`, and `curate` ran AFTER `export`.

It never showed, because every episode the chain had seen before was already in
the file from an earlier run. The first genuinely new recording broke it:

    KeyError: pushT: '2026-09-15/episode_000' is not in
    release/pushT/episodes.jsonl, so its world-frame offset is unknown.
    Refusing to assume zero -- 2026-05-19 is offset (0.23, -0.175, 0) m Z-up
    and would render wrong.

The refusal is right. The order was wrong.
"""
import pytest

import twm.pipeline_stages as PS


def _position(name):
    return [s.name for s in PS.STAGES].index(name)


def test_the_indices_exist_before_anything_stamps_a_world_frame():
    """export reads episodes.jsonl; curate writes it."""
    assert _position("curate") < _position("export"), (
        "export stamps a world-frame declaration read from episodes.jsonl, "
        "which curate writes — running it first makes every NEW episode fail "
        "with an unknown offset, while every old one passes on a stale file")


def _needs(stage):
    """`needs` is a tuple when a stage genuinely has more than one
    prerequisite — export writes force columns AND stamps a world-frame
    declaration read from episodes.jsonl."""
    n = stage.needs
    return () if n is None else ((n,) if isinstance(n, str) else tuple(n))


def test_every_stage_comes_after_everything_it_declares():
    order = {s.name: i for i, s in enumerate(PS.STAGES)}
    for s in PS.STAGES:
        for n in _needs(s):
            assert order[n] < order[s.name], \
                f"{s.name} declares needs={s.needs!r} but runs before {n}"


def test_the_chain_is_still_connected_end_to_end():
    """Reordering must not leave a stage with no declared predecessor."""
    names = [s.name for s in PS.STAGES]
    assert names[0] == "build" and names[-1] == "publish"
    for s in PS.STAGES[1:]:
        assert _needs(s), f"{s.name} declares no predecessor at all"
        for n in _needs(s):
            assert n in names, f"{s.name} declares unknown predecessor {n!r}"
