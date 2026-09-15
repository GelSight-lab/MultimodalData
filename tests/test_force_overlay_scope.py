"""The force overlay must not put an uncut episode beside the segments cut from it.

Step 4 of the publish uploads `release_force/` over the same repo paths,
because the uncut `release/` tree it was built from carries no force columns.
That works while the thing being published IS the uncut tree.

The chain now publishes `release_cut`. Its files are `episode_003_seg00`;
the force tree's are `episode_003`. Uploading the force tree wholesale does
not overlay anything — it ADDS the uncut episode next to the two segments cut
from it, and a reader summing the folder counts those frames twice.

Since the Z-up conversion folds the force columns in (tests/test_zup_force_
channel.py), a cut parquet already carries them, so for a cut publish the
overlay has nothing to do. That has to be CHECKED rather than assumed: a
segment missing the columns must be named, because no overlay can reach it.
"""
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import twm.scripts.build_release_publish as P

FORCE = ("force_left_normal_n", "force_left_penetration_mm",
         "force_left_target_pose", "force_left_source_frame",
         "force_right_normal_n", "force_right_penetration_mm",
         "force_right_target_pose", "force_right_source_frame")


def _parquet(path, *, with_force):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {"frame_idx": np.arange(4, dtype=np.int32)}
    if with_force:
        for c in FORCE:
            cols[c] = np.zeros(4)
    pq.write_table(pa.table(cols), str(path))


@pytest.fixture
def trees(tmp_path):
    stage = tmp_path / "release_cut"
    force = tmp_path / "release_force"
    # published: two segments of one recording, force already inline
    for seg in ("episode_003_seg00", "episode_003_seg01"):
        _parquet(stage / "pushT/meta/2026-09-12" / f"{seg}.parquet",
                 with_force=True)
    # the force tree holds the UNCUT recording those segments came from
    _parquet(force / "pushT/meta/2026-09-12/episode_003.parquet",
             with_force=True)
    return stage, force


def test_the_uncut_episode_is_never_uploaded_beside_its_segments(trees):
    stage, force = trees
    files, missing = P.force_overlay_plan(stage, force, ("pushT",),
                                          since="2026-09-08")
    assert files == [], \
        f"would have added an uncut episode to the segment folder: {files}"


def test_a_tree_that_already_carries_the_columns_needs_no_overlay(trees):
    stage, force = trees
    files, missing = P.force_overlay_plan(stage, force, ("pushT",),
                                          since="2026-09-08")
    assert not missing and not files


def test_a_segment_without_the_columns_is_named_because_nothing_can_reach_it(
        tmp_path):
    """No overlay can address a segment from an uncut name, so this is the
    only place it can be caught."""
    stage, force = tmp_path / "release_cut", tmp_path / "release_force"
    _parquet(stage / "pushT/meta/2026-09-12/episode_004_seg00.parquet",
             with_force=False)
    _parquet(force / "pushT/meta/2026-09-12/episode_004.parquet",
             with_force=True)
    files, missing = P.force_overlay_plan(stage, force, ("pushT",),
                                          since="2026-09-08")
    assert any("episode_004_seg00" in m for m in missing), missing
    assert files == []


def test_dates_outside_the_scope_are_left_alone(tmp_path):
    """The run publishes one week. A May segment without force columns is not
    this run's business — reporting it would make the gate un-passable for
    data nobody asked to reprocess."""
    stage, force = tmp_path / "release_cut", tmp_path / "release_force"
    _parquet(stage / "pushT/meta/2026-05-11/episode_000_seg00.parquet",
             with_force=False)
    _parquet(force / "pushT/meta/2026-05-11/episode_000.parquet",
             with_force=True)
    files, missing = P.force_overlay_plan(stage, force, ("pushT",),
                                          since="2026-09-08")
    assert files == [] and missing == []


def test_an_uncut_publish_still_gets_its_overlay(tmp_path):
    """The uncut tree is the case the overlay was written for: same names, so
    the force file DOES replace the published one. That must keep working."""
    stage, force = tmp_path / "release", tmp_path / "release_force"
    _parquet(stage / "pushT/meta/2026-09-12/episode_003.parquet",
             with_force=False)
    _parquet(force / "pushT/meta/2026-09-12/episode_003.parquet",
             with_force=True)
    files, missing = P.force_overlay_plan(stage, force, ("pushT",),
                                          since="2026-09-08")
    assert [pathlib.Path(f).name for f in files] == ["episode_003.parquet"]
    assert missing == []
