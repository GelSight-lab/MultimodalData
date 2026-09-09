import time

import cv2
import numpy as np
import pytest

from camera_stream.arducam_video_stream import ArducamVideoStream
from twm.sensor_camera import CameraSlot


class FakeCapture:
    def __init__(self, frames=(), opened=True, properties=None):
        self.frames = list(frames)
        self.opened = opened
        self.settings = []
        self.released = False
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        }
        self.properties.update(properties or {})

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def get(self, key):
        return self.properties.get(key, 0)

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        time.sleep(0.002)
        return False, None

    def release(self):
        self.released = True


def _slot():
    return CameraSlot("cam0", "usb-A", "unknown", 640, 480, 30, "MJPG")


def test_stream_configures_v4l2_and_publishes_copied_timestamped_frame():
    image = np.full((480, 640, 3), 17, np.uint8)
    capture = FakeCapture([image])
    calls = []

    def factory(device, backend):
        calls.append((device, backend))
        return capture

    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=factory)
    stream.start(timeout=0.2)
    frame, timestamp = stream.get_frame_with_timestamp(timeout=0.2)
    frame[0, 0] = 0
    second, second_timestamp = stream.get_frame_with_timestamp(timeout=0.2)
    stream.stop()

    assert calls == [("/dev/video6", cv2.CAP_V4L2)]
    assert (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) in capture.settings
    assert (cv2.CAP_PROP_FRAME_WIDTH, 640) in capture.settings
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 480) in capture.settings
    assert (cv2.CAP_PROP_FPS, 30) in capture.settings
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.settings
    assert np.all(second == 17)
    assert np.isfinite(timestamp)
    assert second_timestamp == timestamp
    assert capture.released


def test_stream_rejects_device_that_cannot_open():
    capture = FakeCapture(opened=False)
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *_: capture)

    with pytest.raises(RuntimeError, match=r"cam0.*video6.*open"):
        stream.start(timeout=0.05)

    assert capture.released


def test_stream_times_out_when_no_first_frame_arrives():
    capture = FakeCapture()
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *_: capture)

    with pytest.raises(TimeoutError, match=r"cam0.*first frame"):
        stream.start(timeout=0.03)

    assert capture.released


def test_stream_rejects_negotiated_fps_fallback():
    capture = FakeCapture(
        [np.zeros((480, 640, 3), np.uint8)],
        properties={cv2.CAP_PROP_FPS: 15},
    )
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *_: capture)

    with pytest.raises(RuntimeError, match=r"cam0.*negotiated.*fps.*15"):
        stream.start(timeout=0.2)

    assert capture.released


def test_stream_rejects_wrong_frame_shape():
    capture = FakeCapture([np.zeros((240, 320, 3), np.uint8)])
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *_: capture)

    with pytest.raises(RuntimeError, match=r"cam0.*shape.*480.*640"):
        stream.start(timeout=0.2)

    assert capture.released


def test_stop_is_idempotent():
    capture = FakeCapture([np.zeros((480, 640, 3), np.uint8)])
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *_: capture)
    stream.start(timeout=0.2)

    stream.stop()
    stream.stop()

    assert capture.released


def test_stream_reports_a_stale_frame_after_runtime_capture_stops():
    capture = FakeCapture([np.zeros((480, 640, 3), np.uint8)])
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *_: capture)
    stream.start(timeout=0.2)
    time.sleep(0.03)

    with pytest.raises(TimeoutError, match=r"cam0.*stale"):
        stream.get_frame_with_timestamp(timeout=0.1, max_age=0.01)

    stream.stop()


class BlockingReadCapture(FakeCapture):
    """read() blocks like OpenCV's select() until `unblock` is set; records
    whether release() was called while a read was still in progress."""

    def __init__(self, frame):
        super().__init__(frames=[frame])
        self.unblock = __import__("threading").Event()
        self.in_read = __import__("threading").Event()
        self.released_during_read = False

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        self.in_read.set()
        self.unblock.wait(timeout=5.0)
        self.in_read.clear()
        return False, None

    def release(self):
        if self.in_read.is_set():
            self.released_during_read = True
        self.released = True


def test_stop_never_releases_the_capture_while_a_read_is_in_progress():
    """Releasing a V4L2 capture from another thread while read() is blocked
    leaves the device streaming with pending URBs; the next open then fails
    with EPROTO (seen on the rig: works once, fails on the second open)."""
    frame = np.zeros((480, 640, 3), np.uint8)
    capture = BlockingReadCapture(frame)
    stream = ArducamVideoStream(CameraSlot("cam1", "usb-B", serial="TWMR0001"),
                                "/dev/video6", capture_factory=lambda *a: capture)
    stream.start(timeout=1.0)
    assert capture.in_read.wait(1.0), "reader never entered the blocking read"
    t0 = time.monotonic()
    stopper = __import__("threading").Thread(target=stream.stop)
    stopper.start()
    time.sleep(1.5)                       # longer than the old 1 s join timeout
    assert capture.released is False, "stop() released the capture while read() was blocked"
    capture.unblock.set()
    stopper.join(5.0)
    assert capture.released is True
    assert capture.released_during_read is False
    assert time.monotonic() - t0 < 6.0


def test_peek_frame_with_timestamp_never_waits():
    capture = FakeCapture(frames=[])
    stream = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *a: capture)
    assert stream.peek_frame_with_timestamp() == (None, None)    # before start
    stream._running.set()                                        # emulate a started stream with no frame yet
    t0 = time.monotonic()
    assert stream.peek_frame_with_timestamp() == (None, None)
    assert time.monotonic() - t0 < 0.05
    stream._running.clear()
    image = np.full((480, 640, 3), 9, np.uint8)
    capture2 = FakeCapture([image])
    stream2 = ArducamVideoStream(_slot(), "/dev/video6", capture_factory=lambda *a: capture2)
    stream2.start(timeout=0.5)
    frame, ts = stream2.peek_frame_with_timestamp()
    assert frame is not None and frame[0, 0, 0] == 9 and ts is not None
    stream2.stop()


class RawCapture(FakeCapture):
    """A capture in CONVERT_RGB=0 mode: retrieve() hands back the MJPEG
    buffer the camera sent, shaped (1, N) as OpenCV returns it."""

    def grab(self):
        return bool(self.frames)

    def retrieve(self):
        return (True, self.frames.pop(0).reshape(1, -1)) if self.frames else (False, None)


def _jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), value, np.uint8))
    assert ok
    return buf.reshape(-1)


def test_mjpeg_mode_hands_back_the_bytes_without_decoding_them():
    """The whole point: no imdecode on the capture thread."""
    payload = _jpeg(90)
    cap = RawCapture([payload.copy() for _ in range(3)])
    s = ArducamVideoStream(_slot(), "/dev/video0", capture_factory=lambda *a: cap,
                           encoding="mjpeg")
    s.start()
    try:
        frame, ts = s.get_frame_with_timestamp(timeout=2.0)
    finally:
        s.stop()
    assert frame.ndim == 1 and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, payload)
    assert (cv2.CAP_PROP_CONVERT_RGB, 0) in cap.settings


def test_mjpeg_mode_refuses_a_buffer_that_is_not_a_jpeg():
    """A driver that hands on rubbish makes the writer store rubbish. The
    refusal lands on start(), which waits for the first frame — a camera that
    is not in MJPEG must not get as far as a recording."""
    cap = RawCapture([np.frombuffer(b"nope-not-jpeg", np.uint8)])
    s = ArducamVideoStream(_slot(), "/dev/video0", capture_factory=lambda *a: cap,
                           encoding="mjpeg")
    with pytest.raises(RuntimeError, match="JPEG"):
        s.start()
    s.stop()


def test_the_default_still_decodes_to_bgr():
    frame = np.full((480, 640, 3), 5, np.uint8)
    cap = FakeCapture([frame.copy()])
    s = ArducamVideoStream(_slot(), "/dev/video0", capture_factory=lambda *a: cap)
    s.start()
    try:
        got, _ = s.get_frame_with_timestamp(timeout=2.0)
    finally:
        s.stop()
    assert got.shape == (480, 640, 3)
    assert (cv2.CAP_PROP_CONVERT_RGB, 0) not in cap.settings
