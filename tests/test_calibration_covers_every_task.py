"""Every task must DECLARE its calibration epoch.

`CALIB_DIRS` and `EXPECTED_EPOCH` are exempt from the one-list invariant in
`test_one_task_list`, and rightly so: each task's epoch is a fact about that
task, not a copy of the task list. But the exemption left nothing checking the
other direction -- that the map COVERS every task -- and `toy` was added to
`pipeline_stages.TASKS` on 2026-09-16 without ever being added here.

Measured consequence: `build_release_previews.py` takes its `--task` choices
from `CALIB_DIRS`, so the release preview builder rejected `toy` outright.
The force RUNBOOK records this as a downstream integration blocker and is
explicit that it is "not a reason to omit toy". Nothing else surfaced it,
because `session_epoch` defaults a session dated on or after CURRENT_EPOCH to
that epoch -- so every OTHER reader resolved toy correctly and only the
task-level default map was blind.

This is the coverage half of the invariant: one task list, and every task on
it declaring the epoch it was recorded through.
"""
import pytest

from twm.calib_epoch import CALIB_DIRS, EXPECTED_EPOCH, EPOCH_DIRS
from twm.pipeline_stages import TASKS


@pytest.mark.parametrize("task", TASKS)
def test_every_task_declares_a_calibration_directory(task):
    assert task in CALIB_DIRS, (
        f"{task} is in pipeline_stages.TASKS but declares no calibration "
        f"directory; readers that resolve a task-level default (the release "
        f"preview CLI among them) cannot see it")
    assert CALIB_DIRS[task].is_dir(), f"{task} names a missing epoch directory"


@pytest.mark.parametrize("task", TASKS)
def test_every_task_declares_an_expected_epoch(task):
    assert task in EXPECTED_EPOCH, f"{task} declares no expected epoch"
    assert EXPECTED_EPOCH[task] in EPOCH_DIRS, (
        f"{task} expects epoch {EXPECTED_EPOCH[task]}, which is not a known "
        f"epoch: {sorted(EPOCH_DIRS)}")
    assert CALIB_DIRS[task] == EPOCH_DIRS[EXPECTED_EPOCH[task]], (
        f"{task}'s directory and its expected epoch disagree")


def test_the_release_preview_builder_accepts_every_task():
    """The CLI that the force runbook calls in Step 7. Its `--task` choices
    are derived from CALIB_DIRS, so a task missing above is rejected here."""
    from twm.react_preprocess import previews
    for task in TASKS:
        assert task in previews.CALIB_DIRS, (
            f"build_release_previews.py --task {task} would be refused by "
            f"argparse: choices are {sorted(previews.CALIB_DIRS)}")
