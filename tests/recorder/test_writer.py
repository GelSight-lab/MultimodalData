import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from twm.recorder.frames import Tick
from twm.recorder.writer import (EpisodeWriter, WriterFault, WriterOverloaded,
                                 queue_capacity_bytes)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class BlockingSink:
    """Records batches; blocks every write until `release` is set."""
    def __init__(self):
        self.batches = []
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()

    def __call__(self, f, ticks):
        self.entered.set()
        self.release.wait(timeout=5)
        self.batches.append((f, list(ticks)))


class FakeFile:
    def __init__(self, path):
        self.filename = path
        self.flushes = 0
        open(path, "wb").close()

    def flush(self):
        self.flushes += 1


def small_tick(ts):
    return Tick(ts, color=(np.zeros((4, 4, 3), np.uint8),))  # 48 bytes


TICK_BYTES = small_tick(0).nbytes()


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.002)
    return False


def test_queue_capacity_is_seconds_times_fps_times_tick():
    assert queue_capacity_bytes(3.0, 30, 8_294_400) == 90 * 8_294_400


def test_submit_writes_in_batches_grouped_by_file():
    sink = BlockingSink()
    sink.release.clear()
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 100, batch_size=3, sink=sink)
    f0, f1, f2 = object(), object(), object()
    w.submit(f0, small_tick(-1))          # primer: the thread blocks inside the sink
    assert sink.entered.wait(2)
    for i in range(4):
        w.submit(f1, small_tick(i))
    for i in range(2):
        w.submit(f2, small_tick(10 + i))
    sink.release.set()
    w.drain()
    w.stop()
    assert [(f is f1, len(t)) for f, t in sink.batches] == [
        (False, 1), (True, 3), (True, 1), (False, 2)]
    assert [t.timestamp for _, ts in sink.batches for t in ts] == [-1, 0, 1, 2, 3, 10, 11]


def test_submit_raises_instead_of_dropping_when_full():
    sink = BlockingSink()
    sink.release.clear()
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 2, batch_size=1, sink=sink)
    f = object()
    w.submit(f, small_tick(0))
    w.submit(f, small_tick(1))
    with pytest.raises(WriterOverloaded) as info:
        w.submit(f, small_tick(2))
    assert "queue full" in str(info.value)
    sink.release.set()
    w.drain()
    w.submit(f, small_tick(3))          # accepts again once drained
    w.drain()
    w.stop()
    assert len([t for _, ts in sink.batches for t in ts]) == 3
    assert w.stats().peak_fraction == 1.0


def test_check_reports_sustained_overload_only_after_threshold():
    clock = FakeClock()
    sink = BlockingSink()
    sink.release.clear()
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 4, batch_size=1, sink=sink,
                      overload_fraction=0.5, overload_sustained_s=3.0, clock=clock)
    f = object()
    for i in range(3):                  # 75 % occupancy
        w.submit(f, small_tick(i))
    assert w.check() is None            # first sighting starts the timer
    clock.t += 2.0
    assert w.check() is None
    clock.t += 1.1
    kind, detail = w.check()
    assert kind == "overload" and "3.1s" in detail
    sink.release.set()
    w.drain()
    assert w.check() is None            # cleared once the queue is below threshold
    w.stop()


def test_sink_error_becomes_fault_and_never_deadlocks():
    def bad_sink(f, ticks):
        raise OSError(28, "No space left on device")
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, batch_size=2, sink=bad_sink)
    f = object()
    w.submit(f, small_tick(0))
    assert _wait(lambda: w.stats().fault is not None)
    with pytest.raises(WriterFault):
        w.submit(f, small_tick(1))
    with pytest.raises(WriterFault):
        w.drain()
    assert w.check()[0] == "writer_fault"
    assert "No space left" in w.check()[1]
    w.stop()
    assert w.stats().queue_items == 0


def test_periodic_flush_and_disk_sampling(tmp_path):
    clock = FakeClock()
    f = FakeFile(str(tmp_path / "ep.h5"))
    usage = lambda path: SimpleNamespace(free=123e9)
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, batch_size=1,
                      flush_interval_s=10.0, sink=lambda f, t: None,
                      clock=clock, disk_usage=usage, min_free_gb=50.0)
    w.submit(f, small_tick(0))
    w.drain()
    assert f.flushes == 0
    clock.t += 10.5
    w.submit(f, small_tick(1))
    w.drain()
    # Poll the internal counter, not the external side effect: flush() now
    # runs outside the writer lock, so f.flushes can flip a hair before
    # stats().flushes does. Once the internal counter is visible, the flush
    # it counts has already happened (program order on the writer thread).
    assert _wait(lambda: w.stats().flushes == 1)
    assert f.flushes == 1
    s = w.stats()
    assert s.flushes == 1
    assert s.disk_free_gb == pytest.approx(123.0)
    assert w.check() is None
    w.stop()


@pytest.mark.parametrize("blocked_access", ["flush", "filename"])
def test_drain_waits_until_batch_file_access_finishes(tmp_path, blocked_access):
    entered = threading.Event()
    release = threading.Event()
    path = tmp_path / "ep.h5"
    path.touch()

    class GatedFile:
        def flush(self):
            if blocked_access == "flush":
                entered.set()
                assert release.wait(5)

        @property
        def filename(self):
            if blocked_access == "filename":
                entered.set()
                assert release.wait(5)
            return str(path)

    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 2, batch_size=1,
                      flush_interval_s=0.0, sink=lambda f, t: None)
    try:
        w.submit(GatedFile(), small_tick(0))
        assert entered.wait(2)
        with pytest.raises(WriterFault, match="drain timed out"):
            w.drain(timeout=0)
        stats = w.stats()
        assert stats.queue_items == 1
        assert stats.queue_bytes == TICK_BYTES
        assert stats.bytes_written == TICK_BYTES
        # Capture must still be able to enqueue while file I/O is blocked.
        w.submit(object(), small_tick(1))
        assert w.stats().queue_items == 2
        release.set()
        w.drain(timeout=2)
        assert w.stats().queue_items == 0
        assert w.stats().queue_bytes == 0
        assert w.stats().bytes_written == TICK_BYTES * 2
        assert w.stats().flushes == 2
    finally:
        release.set()
        w.stop()


def test_writer_thread_never_dies_silently_on_internal_error():
    """A raise anywhere in the loop -- not just from the sink -- must fault
    out, not kill the thread silently and leave drain()/check() hanging or
    reporting healthy forever."""
    calls = {"n": 0}

    def flaky_clock():
        calls["n"] += 1
        if calls["n"] > 1:          # call 1 is the constructor's _last_flush_t
            raise RuntimeError("clock exploded")
        return 1000.0

    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, sink=lambda f, t: None,
                      clock=flaky_clock)
    f = object()
    w.submit(f, small_tick(0))         # writer thread's t0 = self._clock() explodes
    assert _wait(lambda: w.stats().fault is not None)
    with pytest.raises(WriterFault):
        w.drain(timeout=2)
    assert w.check()[0] == "writer_fault"
    assert "clock exploded" in w.check()[1]
    w.stop()
    assert w.stats().queue_items == 0


def test_low_disk_reported_by_check(tmp_path):
    clock = FakeClock()
    f = FakeFile(str(tmp_path / "ep.h5"))
    usage = lambda path: SimpleNamespace(free=12e9)
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, batch_size=1,
                      flush_interval_s=0.0, sink=lambda f, t: None,
                      clock=clock, disk_usage=usage, min_free_gb=50.0)
    w.submit(f, small_tick(0))
    w.drain()
    assert _wait(lambda: w.stats().disk_free_gb is not None)
    kind, detail = w.check()
    assert kind == "disk_low" and "12.0 GB" in detail
    w.stop()


def test_stop_is_idempotent_and_stats_reset_per_episode():
    w = EpisodeWriter(capacity_bytes=TICK_BYTES * 10, sink=lambda f, t: None)
    w.submit(object(), small_tick(0))
    w.drain()
    assert w.stats().bytes_written == TICK_BYTES
    w.reset_episode_stats()
    assert w.stats().bytes_written == 0 and w.stats().peak_fraction == 0.0
    w.stop()
    w.stop()
    with pytest.raises(WriterFault):
        w.submit(object(), small_tick(1))
