"""The GelSight stream (USBVideoStream) must recover from a stalled camera
without freezing its caller, and must offer a non-blocking read."""
import threading
import time

import cv2
import numpy as np
import pytest

from camera_stream import base_video_stream, usb_video_stream
from camera_stream.usb_video_stream import USBVideoStream


def _jpeg_bytes(value):
    img = np.full((48, 64, 3), value, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.reshape(-1)


class FakeCapture:
    instances = []

    def __init__(self, *args):
        self.healthy = True
        self.released = False
        self.value = 100 + len(FakeCapture.instances)
        FakeCapture.instances.append(self)

    def set(self, *a):
        return True

    def isOpened(self):
        return True

    def grab(self):
        time.sleep(0.005)
        return self.healthy and not self.released

    def retrieve(self):
        return True, _jpeg_bytes(self.value)

    def release(self):
        self.released = True


@pytest.fixture
def stream(monkeypatch):
    FakeCapture.instances = []
    monkeypatch.setattr(usb_video_stream.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(USBVideoStream, "parse_serial", lambda self, serial: 0)
    real_sleep = time.sleep
    monkeypatch.setattr(base_video_stream.time, "sleep",
                        lambda s: real_sleep(min(s, 0.05)))   # restart's 3 s nap
    s = USBVideoStream(serial="FAKE", resolution=(64, 48), verbose=False, name="left")
    s.start()
    yield s
    s.stop()


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_peek_never_blocks_and_reports_capture_time(stream):
    assert _wait(lambda: stream.peek_frame_with_timestamp()[0] is not None)
    FakeCapture.instances[-1].healthy = False        # camera stalls
    # An in-flight frame may still land after the stall; take the reference
    # only once the stream's own clock has been quiet for a while.
    assert _wait(lambda: time.time() - stream.last_updated > 0.2)
    frame, ts = stream.peek_frame_with_timestamp()
    assert frame.shape == (48, 64, 3) and abs(ts - time.time()) < 1.0
    t0 = time.monotonic()
    stale_frame, stale_ts = stream.peek_frame_with_timestamp()
    assert time.monotonic() - t0 < 0.05                # no waiting, no restart
    assert stale_frame is not None and stale_ts == ts  # last frame, old time


def test_restart_recreates_the_update_thread_so_get_frame_returns(stream):
    assert _wait(lambda: stream.frame is not None)
    FakeCapture.instances[-1].healthy = False        # camera stalls
    # the fixture shortens time.sleep globally, so wait on the stream's own clock
    assert _wait(lambda: time.time() - stream.last_updated > 0.5)   # past 4 x max_no_update_time
    result = {}

    def reader():
        result["frame"] = stream.get_frame(max_no_update_time=0.1)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(6.0)
    assert not t.is_alive(), "get_frame() hung after the driver restarted the camera"
    assert result["frame"].shape == (48, 64, 3)
    assert len(FakeCapture.instances) == 2 and FakeCapture.instances[0].released
    assert _wait(lambda: stream.peek_frame_with_timestamp()[1] is not None
                 and time.time() - stream.peek_frame_with_timestamp()[1] < 0.5)
