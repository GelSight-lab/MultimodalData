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
