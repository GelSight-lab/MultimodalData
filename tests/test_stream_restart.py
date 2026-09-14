"""Restarting a stalled camera took 5 s, of which 3 s was an unmeasured sleep.

Measured on the left GelSight (`2DUPB53G`) on 2026-09-11, 21 reopens with the
settle sleep swept from 0.0 s to 3.0 s: every reopen succeeded, and the time to
the first frame was flat at 620-720 ms regardless of the settle. The sleep was
buying nothing. (The first frame's brightness varies — 56 or 75 — with no
relation to the settle either, so that is the camera's own exposure behaviour,
not a settling effect.)

What the sleep WAS covering is the case the sweep could not produce: a device
that has just thrown EPROTO and is not ready. So the fixed wait becomes a
two-step backoff that VERIFIES a frame arrived, which the old code never did —
it slept, reopened, and returned as if it had worked even when the device was
dead. That is how the pushT 2026-09-10 left sensor "recovered" nine times and
delivered nothing after the ninth.

Worst case is now 2.9 s, no slower than the old fixed 3 s; the common case is
~0.7 s.
"""
import time

import numpy as np
import pytest

pytest.importorskip(
    "autolab_core",
    reason="camera_stream -> misc.utils -> autolab_core, the frankapy "
           "hardware stack. Absent here, this file raised at COLLECTION, "
           "and one collection error aborts the whole run -- bare pytest "
           "then executed nothing at all.")

from camera_stream.base_video_stream import BaseVideoStream


class FakeStream(BaseVideoStream):
    """A camera whose reopen behaviour the test dictates.

    `frames_after` is the number of opens that must happen before the device
    starts delivering; `fail_opens` is how many opens raise instead.
    """

    def __init__(self, frames_after=1, fail_opens=0, **kw):
        super().__init__(verbose=False, name="fake", **kw)
        self.frames_after, self.fail_opens = frames_after, fail_opens
        self.opens = self.stops = 0
        self.sleeps = []

    def start(self, create_thread=True):
        self.opens += 1
        if self.opens <= self.fail_opens:
            raise RuntimeError("cannot open camera stream")
        self.streaming = True
        if self.opens >= self.frames_after:
            self.frame = np.zeros((2, 2, 3), np.uint8)
            self.frame_ts = time.time()

    def stop(self):
        self.stops += 1
        self.streaming = False


@pytest.fixture(autouse=True)
def record_sleeps(monkeypatch):
    """Record the settle sleeps so the tests measure the schedule, not wall time.

    Only the settles are of interest; the 2 ms first-frame poll is filtered out
    by `settles()` rather than left to pollute every assertion.
    """
    slept = []
    monkeypatch.setattr("camera_stream.base_video_stream.time.sleep", slept.append)
    return slept


def settles(slept):
    return [x for x in slept if x >= 0.1]


def test_a_camera_that_delivers_on_the_first_open_never_settles(record_sleeps):
    s = FakeStream(frames_after=1)
    s.restart()
    assert s.opens == 1
    assert settles(record_sleeps) == [], "slept when nothing needed settling"


def test_a_camera_that_delivers_nothing_at_first_is_retried_with_a_settle(record_sleeps):
    s = FakeStream(frames_after=2)
    s.restart()
    assert s.opens == 2
    assert settles(record_sleeps), "retried without giving the device time"
    assert s.frame is not None


def test_a_dead_camera_raises_instead_of_pretending_to_have_recovered():
    """The supervisor counts on this: a silent return is indistinguishable from
    a real recovery, and it is what let 122 s of frozen frames be recorded."""
    s = FakeStream(frames_after=99)
    s.FIRST_FRAME_TIMEOUT_S = 0.05
    with pytest.raises(RuntimeError, match="no frame"):
        s.restart()
    assert s.opens == len(s.RESTART_SETTLE_S), "did not exhaust the backoff"


def test_an_open_that_raises_is_retried_then_reported():
    s = FakeStream(frames_after=1, fail_opens=99)
    with pytest.raises(RuntimeError):
        s.restart()
    assert s.opens == len(s.RESTART_SETTLE_S)


def test_every_retry_releases_the_previous_capture():
    """Reopening on top of an unreleased handle is how a camera ends up with
    two capture objects and a thread reading the dead one."""
    s = FakeStream(frames_after=2)
    s.FIRST_FRAME_TIMEOUT_S = 0.05
    s.restart()
    assert s.stops >= s.opens, f"{s.opens} opens but only {s.stops} releases"


def test_the_worst_case_is_no_slower_than_the_three_second_sleep_it_replaces():
    total = sum(FakeStream.RESTART_SETTLE_S) + (
        len(FakeStream.RESTART_SETTLE_S) * FakeStream.FIRST_FRAME_TIMEOUT_S)
    assert total <= 3.0, f"worst case {total:.1f}s exceeds the old fixed 3.0s"


def test_the_first_frame_timeout_clears_the_measured_reopen_time():
    """620-720 ms measured over 21 reopens; the timeout needs real margin over
    that or a healthy camera gets declared dead."""
    assert FakeStream.FIRST_FRAME_TIMEOUT_S >= 0.72 * 1.5


# ── the watchdog threshold ──────────────────────────────────────────────────

def test_the_stall_threshold_clears_the_worst_normal_gelsight_gap():
    """85,031 inter-frame gaps across the 2026-09-10/11 pushT episodes: median
    53 ms, p99.9 122 ms, worst 216 ms. The threshold has to sit above 216 ms
    with margin (or normal jitter triggers restarts) and well below the 1 s it
    replaces (every 100 ms of it is 3 frozen frames per outage)."""
    from twm.recorder.rig import STALL_AFTER_S
    worst_normal_gap_s = 0.216
    assert STALL_AFTER_S > worst_normal_gap_s * 1.5
    assert STALL_AFTER_S <= 0.6
