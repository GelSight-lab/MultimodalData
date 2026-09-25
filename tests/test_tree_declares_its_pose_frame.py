"""Each release tree declares which frame its POSES are in.

`curation` stamped `up_axis` from `UP_AXIS_BUILT = "y"` -- true of the tree
react_preprocess BUILDS, and false of every tree derived from it. `curate` also
runs on the CUT tree, which is sliced out of `release_zup` and is therefore
Z-up, so `release_cut/*/episodes.jsonl` published `up_axis: y` over poses that
are demonstrably Z-up (bit-identical to release_zup, and satisfying
(x,y,z)->(x,-z,y) against release).

Two things went wrong at once. Published metadata told downstream users the
wrong frame -- and that field is how they decide what to do with the poses.
And `_already_zup` reads the same field to decide whether a tree still needs
rotating, so a re-run over a tree that claims "y" rotates Z-up poses a second
time: a net 180 degrees that every projection is blind to, because the
conversion rotates the calibration too.

The tree's calibration cannot answer this: `release` ships Z-up calibration
beside Y-up poses, so `up_axis` on a calibration file says nothing about the
poses. The frame is a property of WHICH TREE, so it is declared beside the
tree roots and the declaration wins over a stale stored value.
"""
from __future__ import annotations

import pytest

import twm.pipeline_stages as PS
from twm.react_preprocess.curation import tree_up_axis


def test_every_declared_tree_has_a_pose_frame():
    assert PS.TREE_UP_AXIS[PS.RELEASE] == "y"       # built straight from H5
    assert PS.TREE_UP_AXIS[PS.RELEASE_ZUP] == "z"   # convert_release_zup ran
    assert PS.TREE_UP_AXIS[PS.RELEASE_CUT] == "z"   # sliced out of the Z-up tree


def test_the_cut_tree_is_z_up():
    """The one that was published wrong."""
    assert tree_up_axis(PS.RELEASE_CUT) == "z"


def test_the_built_tree_is_still_y_up():
    assert tree_up_axis(PS.RELEASE) == "y"


def test_an_unknown_tree_refuses_rather_than_guessing():
    """A guess here is what put a DexForce target hundreds of mm off."""
    with pytest.raises(KeyError, match="up_axis"):
        tree_up_axis("/tmp/some/tree/nobody/declared")


def test_a_path_string_resolves_the_same_as_a_path_object():
    from pathlib import Path
    assert tree_up_axis(str(PS.RELEASE_CUT)) == tree_up_axis(Path(PS.RELEASE_CUT))


def test_curate_stamps_the_tree_frame_over_a_stale_stored_one(tmp_path,
                                                              monkeypatch):
    """The stored value is exactly what was wrong, so it must not win.

    The old rule was "an existing row's up_axis wins, because a later stage may
    have rotated that episode". Within ONE tree every episode shares a frame,
    so the tree's declaration is the better authority -- and preserving the
    stored value is what kept `y` on the cut tree across every re-curate.
    """
    import twm.pipeline_stages as PS
    from twm.react_preprocess import curation

    fake = tmp_path / "release_cut"
    monkeypatch.setitem(PS.TREE_UP_AXIS, fake, "z")
    prior = {"2026-09-11/episode_006_seg00": {"up_axis": "y"}}   # stale
    assert curation.row_up_axis(fake, "2026-09-11/episode_006_seg00",
                                prior) == "z"


def test_a_row_with_no_prior_also_gets_the_tree_frame(tmp_path, monkeypatch):
    import twm.pipeline_stages as PS
    from twm.react_preprocess import curation
    fake = tmp_path / "release_cut"
    monkeypatch.setitem(PS.TREE_UP_AXIS, fake, "z")
    assert curation.row_up_axis(fake, "new/episode_000", {}) == "z"


def test_an_undeclared_tree_keeps_the_old_behaviour(tmp_path):
    """Someone curating a scratch tree should not be stopped; they get the
    built-tree default and the stored value still wins, as before."""
    from twm.react_preprocess import curation
    scratch = tmp_path / "scratch"
    assert curation.row_up_axis(scratch, "k", {"k": {"up_axis": "z"}}) == "z"
    assert curation.row_up_axis(scratch, "k", {}) == curation.UP_AXIS_BUILT
