"""rope has to reach the hub the same way the other tasks do.

rope was published by hand, before the scheduler existed, and it shows: no
`bad_frames.json`, no `segments.json`, no `splits.json`, and no `calibration/`
in its cut tree. Every one of those gaps had to be closed by a separate manual
command on 2026-09-15 -- and the missing splits.json meant every rope segment
read as TRAIN, because `ReactVideoDataset._split_filter` treats an unknown key
that way.

A task that is not in the list is a task somebody runs by hand, and running it
by hand is what produced the gaps. So the list is the thing under test.
"""
import pytest

import twm.pipeline_stages as PS
import twm.scripts.build_release_publish as P
from twm.react_preprocess.meta import TASK_INDEX


def test_every_task_with_a_published_index_is_in_the_scheduler():
    assert set(PS.TASKS) == set(TASK_INDEX), (
        "a task the scheduler does not know is one somebody runs by hand")


def test_the_publisher_covers_the_same_tasks_as_the_scheduler():
    """Two lists that can disagree WILL: the scheduler would build and cut a
    task the publisher then declines to upload, with nothing saying so."""
    assert set(P.TASKS) == set(PS.TASKS)


def test_every_scheduled_task_can_resolve_its_calibration_epoch():
    """`session_epoch` raises for an undeclared session, so a task in the list
    whose sessions are not declared fails at the publish gate after everything
    upstream has already run."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm"))
    from react_toolbox.calib_epoch import CALIB_SESSIONS
    from react_toolbox.calib_epoch import session_epoch
    from pathlib import Path
    for task in PS.TASKS:
        d = Path(PS.DATA_ROOT) / task
        dates = sorted({p.name for p in d.iterdir()
                        if p.is_dir() and p.name >= PS.SCOPE_SINCE}) \
            if d.is_dir() else []
        # Since 2026-09-15 a session dated on or after CURRENT_EPOCH resolves
        # without a table entry — the recorder ran on that solve. What must
        # hold is that every in-scope session RESOLVES, declared or not.
        for date in dates:
            assert session_epoch(task, date), f"{task}/{date} resolves to nothing"
