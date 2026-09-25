"""Every task the scheduler runs must have a data root.

`H5_ROOTS` is a fourth place the task list lives, and the invariant that
forbids restating it only recognised literal tuples — a dict's keys walked
straight past it.

Measured consequence, 2026-09-16: `toy` was added to TASK_INDEX and TASKS, the
build stage produced its argv, and the CLI rejected it:

    react_preprocess build: error: argument --task: invalid choice: 'toy'
    (choose from 'motherboard', 'pushT', 'rope')

rope's two recordings had just built successfully, so the run reached `toy`
last and died there — after the work in front of it was already done.
"""
import pytest

import twm.pipeline_stages as PS
from twm.react_preprocess.config import H5_ROOTS


def test_every_scheduled_task_has_a_data_root():
    missing = [t for t in PS.TASKS if t not in H5_ROOTS]
    assert not missing, (
        f"{missing} are scheduled but have no H5_ROOTS entry, so the build "
        f"CLI rejects them by name after everything ahead has already run")


def test_the_roots_do_not_invent_tasks_the_scheduler_does_not_know():
    extra = [t for t in H5_ROOTS if t not in PS.TASKS]
    assert not extra, (
        f"{extra} have a data root but are not scheduled — either add them to "
        f"TASKS or drop the root; a half-registered task is how one gets "
        f"processed by hand")
