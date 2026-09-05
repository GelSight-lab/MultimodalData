import time

import numpy as np

from twm.data_collection import CaptureLoop, _startup_call


class FakeRealSense:
    def __init__(self, value):
        self.color = np.full((480, 640, 3), value, np.uint8)
        self.depth = np.full((480, 640), value, np.uint16)

    def get_color_frame(self):
        return self.color.copy()

    def get_depth_frame(self):
        return self.depth.copy()


class FakeGelSight:
    def __init__(self, value):
        self.frame = np.full((480, 640, 3), value, np.uint8)

    def get_frame(self):
        return self.frame.copy()

    def get_frame_with_timestamp(self):
        return self.frame.copy(), 100.0 + self.frame[0, 0, 0]


class FakeArducam:
    def __init__(self, value, timestamp):
        self.frame = np.full((480, 640, 3), value, np.uint8)
        self.timestamp = timestamp

    def get_frame_with_timestamp(self):
        return self.frame.copy(), self.timestamp


class FailingArducam(FakeArducam):
    def __init__(self, value, timestamp):
        super().__init__(value, timestamp)
        self.fail = False

    def get_frame_with_timestamp(self):
        if self.fail:
            raise TimeoutError("Arducam cam0 frame is stale")
        return super().get_frame_with_timestamp()


class FakeOptitrack:
    def get_latest_pose(self, name):
        return (time.time(), np.array([0, 0, 0, 0, 0, 0, 1], float))


class FakeWriter:
    def __init__(self):
        self.calls = []
        self.dropped_frames = 0
        self.flushes = 0

    @property
    def queue_size(self):
        return 0

    def enqueue(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def flush(self):
        self.flushes += 1


def _legacy_inputs():
    return (
        [FakeRealSense(1), FakeRealSense(2), FakeRealSense(3)],
        FakeGelSight(4),
        FakeGelSight(5),
    )


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_latest_snapshot_contains_both_arducam_frames_and_timestamps():
    rs, left, right = _legacy_inputs()
    writer = FakeWriter()
    capture = CaptureLoop(
        rs, left, right, FakeOptitrack(), writer, fps=100,
        arducam_streams=[FakeArducam(21, 123.25), FakeArducam(37, 123.50)],
    )

    capture.start()
    latest = _wait_for(capture.latest)
    capture.stop()

    assert [int(frame[0, 0, 0]) for frame in latest["arducam_frames"]] == [21, 37]
    assert latest["arducam_timestamps"] == [123.25, 123.50]


def test_recording_enqueues_arducam_payload_with_legacy_tick():
    rs, left, right = _legacy_inputs()
    writer = FakeWriter()
    capture = CaptureLoop(
        rs, left, right, FakeOptitrack(), writer, fps=100,
        arducam_streams=[FakeArducam(21, 123.25), FakeArducam(37, 123.50)],
    )
    capture.WARMUP_DROP_FRAMES = 0

    capture.start()
    _wait_for(capture.latest)
    capture.start_recording(object())
    _wait_for(lambda: writer.calls)
    result = capture.stop_recording()
    capture.stop()

    args, kwargs = writer.calls[0]
    assert [int(frame[0, 0, 0]) for frame in kwargs["arducam_frames"]] == [21, 37]
    assert kwargs["arducam_timestamps"] == [123.25, 123.50]
    assert result[1] >= 1
    assert writer.flushes == 1


def test_capture_loop_without_arducams_preserves_legacy_snapshot_and_enqueue():
    rs, left, right = _legacy_inputs()
    writer = FakeWriter()
    capture = CaptureLoop(rs, left, right, FakeOptitrack(), writer, fps=100)
    capture.WARMUP_DROP_FRAMES = 0

    capture.start()
    latest = _wait_for(capture.latest)
    capture.start_recording(object())
    _wait_for(lambda: writer.calls)
    capture.stop_recording()
    capture.stop()

    assert latest["arducam_frames"] is None
    assert latest["arducam_timestamps"] is None
    _, kwargs = writer.calls[0]
    assert kwargs["arducam_frames"] is None
    assert kwargs["arducam_timestamps"] is None


def test_runtime_camera_failure_is_published_and_recording_can_finalize():
    rs, left, right = _legacy_inputs()
    writer = FakeWriter()
    failing = FailingArducam(21, 123.25)
    capture = CaptureLoop(
        rs, left, right, FakeOptitrack(), writer, fps=100,
        arducam_streams=[failing, FakeArducam(37, 123.50)],
    )
    capture.WARMUP_DROP_FRAMES = 0

    capture.start()
    _wait_for(capture.latest)
    capture.start_recording(object())
    _wait_for(lambda: writer.calls)
    failing.fail = True
    fatal = _wait_for(
        lambda: capture.latest() if capture.latest().get("fatal_error") else None
    )
    finalized = capture.stop_recording()
    capture.stop()

    assert "cam0" in fatal["fatal_error"]
    assert "stale" in fatal["fatal_error"]
    assert fatal["recording"] is True
    assert finalized is not None
    assert writer.flushes == 1


def test_startup_failure_stops_every_registered_resource_in_reverse_order():
    events = []

    class Resource:
        def __init__(self, name, fails=False):
            self.name = name
            self.fails = fails

        def start(self):
            events.append(f"start {self.name}")
            if self.fails:
                raise RuntimeError(f"{self.name} failed")

        def stop(self):
            events.append(f"stop {self.name}")

    first = Resource("first")
    second = Resource("second", fails=True)
    registered = [first]
    _startup_call(first.start, registered)
    registered.append(second)

    try:
        _startup_call(second.start, registered)
    except RuntimeError:
        pass
    else:
        raise AssertionError("startup failure did not propagate")

    assert events == [
        "start first", "start second", "stop second", "stop first"
    ]
