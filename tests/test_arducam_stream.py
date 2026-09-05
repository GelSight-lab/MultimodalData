import time

import cv2
import numpy as np
import pytest

from camera_stream.arducam_video_stream import ArducamVideoStream
from twm.sensor_camera import CameraSlot


class FakeCapture:
    def __init__(self, frames=(), opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.settings = []
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def get(self, key):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }
        return values.get(key, 0)

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
