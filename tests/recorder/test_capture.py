import threading
import time

import numpy as np
import pytest

from twm.recorder.capture import CaptureLoop, StopRequest
from twm.recorder.frames import Tick
from twm.recorder.writer import EpisodeWriter


class FakeRig:
    """Ticks at wall-clock time; `jump` adds an artificial gap once."""
    def __init__(self):
        self.n = 0
        self.jump = 0.0
        self.fail = None
        self.frame = np.zeros((4, 4, 3), np.uint8)

    def grab(self):
        if self.fail:
            raise self.fail
        self.n += 1
        t = time.time() + self.jump
        self.jump = 0.0
        return Tick(t, gelsight=(self.frame + self.n, self.frame),
                    gelsight_ts=(t, t))

    def latest_poses(self):
        return {"motherboard": (time.time(), [0] * 7)}


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.002)
    raise AssertionError("timeout")


TICK = Tick(0.0, gelsight=(np.zeros((4, 4, 3), np.uint8),) * 2, gelsight_ts=(0.0, 0.0)).nbytes()


def make(rig=None, sink=None, capacity_ticks=100, warmup=0, max_gap=0.5):
    rig = rig or FakeRig()
    writer = EpisodeWriter(capacity_bytes=TICK * capacity_ticks, batch_size=1,
                           sink=sink or (lambda f, t: None))
    loop = CaptureLoop(rig, writer, fps=200, warmup_drop_frames=warmup,
                       max_tick_gap_s=max_gap)
    return rig, writer, loop


def test_snapshot_is_published_and_reference_can_be_reset():
    rig, writer, loop = make()
    loop.start()
    snap = _wait(loop.latest)
    first_ref = snap.gs_ref[0][0, 0, 0]
    loop.request_reset_ref()
    _wait(lambda: loop.latest().gs_ref[0][0, 0, 0] != first_ref)
    loop.stop()
    writer.stop()
    assert snap.recording is False and snap.writer.queue_items == 0


def test_warmup_frames_are_not_recorded():
    written = []
    rig, writer, loop = make(sink=lambda f, t: written.extend(t), warmup=5)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 3)
    result = loop.stop_recording()
    loop.stop()
    writer.drain()
    writer.stop()
    assert result.frame_count == len(written)
    assert result.stop_request is None
    assert rig.n >= result.frame_count + 5


def test_overload_ends_recording_without_dropping():
    gate = threading.Event()
    written = []

    def slow_sink(f, ticks):
        gate.wait(5)
        written.extend(ticks)

    rig, writer, loop = make(sink=slow_sink, capacity_ticks=3)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    snap = _wait(lambda: loop.latest() if loop.latest().stop_request else None)
    assert snap.stop_request.kind == "overload"
    assert snap.recording is False
    gate.set()
    result = loop.stop_recording()
    loop.stop()
    writer.drain()
    writer.stop()
    assert result.stop_request.kind == "overload"
    assert result.frame_count == len(written) == 3     # every accepted tick reached the sink


def test_tick_gap_ends_recording_as_capture_stall():
    rig, writer, loop = make(max_gap=0.2)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 2)
    rig.jump = 1.0
    snap = _wait(lambda: loop.latest() if loop.latest().stop_request else None)
    result = loop.stop_recording()
    loop.stop()
    writer.stop()
    assert snap.stop_request.kind == "capture_stall"
    assert result.max_gap_s >= 1.0 and result.gap_count == 1


def test_sensor_error_is_fatal_and_recording_can_still_finalize():
    rig, writer, loop = make()
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 1)
    rig.fail = TimeoutError("Arducam cam0 frame is stale")
    snap = _wait(lambda: loop.latest() if loop.latest().fatal_error else None)
    assert "cam0" in snap.fatal_error and snap.recording is True
    result = loop.stop_recording()
    loop.stop()
    writer.stop()
    assert result is not None and result.frame_count >= 1
    assert loop.stop_recording() is None


def test_stop_recording_race_does_not_leak_stop_request_into_next_episode():
    """Regression for a lost/misattributed StopRequest.

    `_record()` used to compute a StopRequest and only apply it to shared
    state (`_recording`/`_stop_request`) in a *separate*, later lock
    acquisition. A `stop_recording()` + `start_recording()` landing in that
    window could return episode 1 with `stop_request=None` (finalized as
    valid despite the overload) and then kill episode 2 at frame 0 carrying
    episode 1's stop request.

    This is made deterministic (no sleep-based timing race) by driving the
    sequence explicitly through the gate: force the overload for episode a,
    observe it, stop episode a, release the gate and drain so the writer has
    capacity again, then start episode b and confirm it runs cleanly with no
    inherited stop request.

    capacity_ticks is generous (50, not the 3 used to force episode a's
    overload) on purpose: once the gate is released the writer thread has to
    wake from an idle condition-variable wait and get scheduled again before
    it drains anything, and at fps=200 (5ms/tick) a tight capacity turns that
    ordinary wake-up latency into a *real* overload for episode b — which
    would make this test fail for a reason that has nothing to do with the
    bug it targets. The capacity only needs to be small enough to still
    trigger episode a's overload quickly, which 50 ticks (250ms of ticks)
    comfortably is.
    """
    gate = threading.Event()

    def slow_sink(f, ticks):
        gate.wait(5)

    rig, writer, loop = make(sink=slow_sink, capacity_ticks=50)
    loop.start()
    _wait(loop.latest)

    loop.start_recording("h5a")
    snap = _wait(lambda: loop.latest() if loop.latest().stop_request else None)
    assert snap.stop_request.kind == "overload"
    result_a = loop.stop_recording()

    gate.set()               # let episode a's queued/in-flight ticks flush
    writer.drain()            # ... so episode b starts with full capacity

    # `latest()` keeps returning episode a's last published CaptureSnapshot
    # (frame_count=50, stop_request=overload) until the capture thread
    # publishes a fresh one for episode b — a plain `frame_count >= 2` wait
    # would happily match that stale snapshot (50 >= 2) before a single
    # episode-b tick has run. Anchor on the tick's own timestamp, which is
    # only ever produced after `start_recording("h5b")` is called, so a
    # matching snapshot is guaranteed to belong to episode b.
    t_start_b = time.time()
    loop.start_recording("h5b")

    def fresh_episode_b_snapshot():
        snap = loop.latest()
        if snap is not None and snap.tick.timestamp > t_start_b and snap.frame_count >= 2:
            return snap
        return None

    snap_b = _wait(fresh_episode_b_snapshot)
    result_b = loop.stop_recording()

    loop.stop()
    writer.drain()
    writer.stop()

    assert result_a.stop_request.kind == "overload"
    assert snap_b.stop_request is None
    assert result_b.stop_request is None
    assert result_b.frame_count >= 2


def test_latest_poses_failure_is_fatal_and_recording_can_still_finalize():
    """Only rig.grab() used to be guarded; an exception from
    rig.latest_poses() (or writer.stats()) killed the thread silently with
    `recording` left True forever and no `fatal_error` ever published."""

    class PosesFailAfterN:
        def __init__(self, inner, n):
            self._inner = inner
            self.n = n
            self.calls = 0

        def grab(self):
            return self._inner.grab()

        def latest_poses(self):
            self.calls += 1
            if self.calls > self.n:
                raise RuntimeError("optitrack link down")
            return self._inner.latest_poses()

    rig = PosesFailAfterN(FakeRig(), n=3)
    _, writer, loop = make(rig=rig)
    loop.start()
    _wait(loop.latest)
    loop.start_recording("h5")
    _wait(lambda: loop.latest().frame_count >= 1)
    snap = _wait(lambda: loop.latest() if loop.latest().fatal_error else None)
    assert "optitrack link down" in snap.fatal_error and snap.recording is True
    result = loop.stop_recording()
    loop.stop()
    writer.stop()
    assert result is not None and result.h5_file == "h5" and result.frame_count >= 1


def test_stop_recording_frame_count_matches_writer_with_no_gating():
    """stop_recording() is called from the GUI thread on an operator
    keypress at any moment, not only after seeing a published stop request.
    Repeat many times, ungated, to catch any race between the frame_count
    returned to the caller and what actually reached the sink for that same
    open file."""
    for _ in range(20):
        written = []
        rig, writer, loop = make(sink=lambda f, t: written.extend(t))
        loop.start()
        _wait(loop.latest)
        loop.start_recording("h5")
        _wait(lambda: loop.latest().frame_count >= 1)
        result = loop.stop_recording()
        loop.stop()
        writer.drain()
        writer.stop()
        assert result.stop_request is None
        assert result.frame_count == len(written)
