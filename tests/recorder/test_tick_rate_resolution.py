"""A 1% threshold cannot be expressed on a recording of 31 ticks.

`check_tick_rate` fails when more than 1% of ticks arrive late. On a real
session -- tens of thousands of ticks -- that is what it says. On a short one
it is not: 1% of 31 ticks is 0.31, so a single late tick scores 3.23% and the
check demands ZERO late ticks while its message still reads "1%".

Measured: the headless recorder test records 0.6 s at 60 fps and failed on
exactly this, one late tick out of 31, on a machine doing nothing unusual.

The resolution of a fraction is 1/T. Below that the threshold is not a
tolerance, it is a rounding artefact, so one late tick is tolerated whatever T
is. On any recording long enough for 1% to mean anything -- 100 ticks or more
-- this changes nothing, because 0.01 * T is already above 1.
"""
import numpy as np
import pytest

from twm.recorder.validate import check_tick_rate


class _F:
    filename = "<test>"

    def __init__(self, ts):
        self._ts = np.asarray(ts, float)

    def __getitem__(self, k):
        assert k == "timestamps"
        return self._ts


def _ticks(n, fps=60, late_at=()):
    dt = np.full(n - 1, 1.0 / fps)
    for i in late_at:
        dt[i] = 2.0 / fps            # > 1.5x expected: a late tick
    return np.concatenate([[100.0], 100.0 + np.cumsum(dt)])


def _run(ts, fps=60):
    return check_tick_rate(_F(ts), {}, fps, max_tick_gap_s=1.0)


def test_one_late_tick_in_a_short_recording_is_tolerated():
    c = _run(_ticks(31, late_at=(10,)))
    assert c.ok, c.detail


def test_two_late_ticks_in_a_short_recording_still_fail():
    """The point is the resolution of the threshold, not letting it through."""
    c = _run(_ticks(31, late_at=(10, 20)))
    assert not c.ok and "late" in c.detail


def test_a_long_recording_is_unchanged():
    """1% of 10000 is 100, so tolerating one changes nothing that matters."""
    c = _run(_ticks(10000, late_at=tuple(range(200))))
    assert not c.ok and "late" in c.detail


def test_a_long_recording_within_budget_passes():
    c = _run(_ticks(10000, late_at=tuple(range(50))))
    assert c.ok, c.detail
