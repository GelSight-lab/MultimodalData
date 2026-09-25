"""A task with no force channel is drawn as such, not as zero contact.

`collect` read `force_left_normal_n` unconditionally and died on `toy`, which
is published without force while a new estimator is written:

    KeyError: Field "force_left_normal_n" does not exist in schema

Filling with zeros would be worse than crashing: panels B (force distribution)
and C (contact occupancy) are derived from force alone, and a zero-filled task
appears on the figure as one that never makes contact. That is a claim about
the data, and it would be false.

So the absence propagates: the task is counted in scale and tactile validity,
which do not need force, and is marked `no_force` so the force-derived panels
can leave it out and say why.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _table(tmp_path, with_force):
    n = 4
    cols = {"tactile_left_is_new": np.ones(n, bool)}
    if with_force:
        cols["force_left_normal_n"] = np.linspace(0, 3, n)
        cols["force_right_normal_n"] = np.linspace(3, 0, n)
    p = tmp_path / "e.parquet"
    pq.write_table(pa.table(cols), str(p))
    return p


def test_a_force_free_task_does_not_crash(tmp_path):
    from twm.scripts.dataset_stats import collect
    p = _table(tmp_path, with_force=False)
    o = collect("toy", ["2026-09-16/episode_000"], lambda _: str(p))
    assert o["segs"] == 1 and o["frames"] == 4


def test_it_is_marked_rather_than_zero_filled(tmp_path):
    """Zero force would draw `toy` as a task that never touches anything."""
    from twm.scripts.dataset_stats import collect
    p = _table(tmp_path, with_force=False)
    o = collect("toy", ["2026-09-16/episode_000"], lambda _: str(p))
    assert o.get("no_force") is True
    assert o["cL"] == 0 and not o["f"], (
        "a force-free task contributed contact counts it cannot support")


def test_a_task_with_force_is_unchanged(tmp_path):
    from twm.scripts.dataset_stats import collect
    p = _table(tmp_path, with_force=True)
    o = collect("rope", ["2026-09-16/episode_000"], lambda _: str(p))
    assert o.get("no_force") is False
    assert o["cL"] > 0 and len(o["f"]) == 1
