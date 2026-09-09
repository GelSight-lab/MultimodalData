"""A deleted source recording says so, instead of returning nothing.

1.19 TB of raw HDF5 was deleted on 2026-09-09 to make room for re-collection.
Every consumer finds those sessions through a glob, and a glob over a deleted
directory returns exactly what a glob over a typo'd path returns: []. The
callers then "succeed" by doing no work, which is the wrong answer -- the work
is impossible, not finished.

So the deletion is DECLARED (config.RAW_DELETED) and the resolvers raise.
"""
import pytest

from twm.calib_epoch import CALIB_SESSIONS
from twm.react_preprocess.config import (H5_ROOTS, RAW_DELETED, RAW_DELETED_ON,
                                         deleted_note)
from twm.react_preprocess.h5io import discover


def test_a_deleted_session_is_named_with_the_reason():
    note = deleted_note("pushT", "2026-06-18")
    assert note and RAW_DELETED_ON in note
    assert "re-collection" in note
    # It must say what survives, or the reader assumes the data is simply gone.
    assert "release" in note and "yxma/React" in note


def test_a_live_session_gets_no_note():
    assert deleted_note("motherboard", "2026-09-09") is None
    assert deleted_note("some_new_task") is None


def test_discover_raises_for_a_deleted_session_instead_of_returning_empty():
    with pytest.raises(FileNotFoundError, match="deleted"):
        discover("pushT", H5_ROOTS["pushT"], date="2026-06-18")


def test_discover_still_returns_empty_for_a_merely_absent_path(tmp_path):
    """Only a DECLARED deletion raises; an unknown task stays a quiet []."""
    assert discover("some_new_task", tmp_path) == []


def test_a_task_that_lost_a_session_and_gained_another_resolves_the_new_one():
    """pushT lost 2026-06-18 and was re-collected on 2026-09-09. Asking for
    the task must return the new session, not raise about the old one."""
    found = discover("pushT", H5_ROOTS["pushT"])
    if not found:
        pytest.skip("pushT has no live session on this machine")
    assert all(p.parent.name != "2026-06-18" for p in found)


def test_every_live_session_declares_a_calibration_epoch():
    """A session with no entry makes the viewer refuse it -- by design, so the
    entry has to be added when the session is recorded, not when it is read."""
    undeclared = []
    for task, root in H5_ROOTS.items():
        for d in sorted(p for p in root.glob("*") if p.is_dir()):
            if not list(d.glob("episode_*.h5")):
                continue
            if (task, d.name) not in CALIB_SESSIONS:
                undeclared.append(f"{task}/{d.name}")
    assert not undeclared, (
        f"recorded but undeclared: {undeclared} — add them to "
        f"calib_epoch.CALIB_SESSIONS or the viewer refuses these episodes")


def test_every_deleted_session_still_declares_its_calibration_epoch():
    """The derived release ships for these sessions and has to name the
    extrinsics it was rendered through, so the entries outlive the raw data."""
    for key in RAW_DELETED:
        task, date = key
        if not (H5_ROOTS[task] / date).exists():
            assert key in CALIB_SESSIONS, f"{key} lost its calibration epoch"
