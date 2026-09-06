"""Timestamped V4L2 stream for the TWM sensor-mounted Arducams."""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2

from twm.sensor_camera import CameraSlot


class ArducamVideoStream:
    """Continuously publish the freshest frame from one resolved Arducam."""

    def __init__(
        self,
        config: CameraSlot,
        device: str,
        *,
        capture_factory: Callable = cv2.VideoCapture,
    ):
        self.config = config
        self.device = device
        self._capture_factory = capture_factory
        self._capture = None
        self._frame = None
        self._frame_ts = None
        self._error = None
        self._condition = threading.Condition()
        self._running = threading.Event()
        self._thread = None

    def _tag(self) -> str:
        return f"Arducam {self.config.slot} ({self.device}, {self.config.id_path})"

    def start(self, timeout: float = 5.0):
        if self._running.is_set():
            return self
        capture = self._capture_factory(self.device, cv2.CAP_V4L2)
        self._capture = capture
        if not capture.isOpened():
            capture.release()
            self._capture = None
            raise RuntimeError(f"{self._tag()} could not open capture device")

        requested_fourcc = cv2.VideoWriter_fourcc(*self.config.pixel_format)
        requested = (
            ("pixel_format", cv2.CAP_PROP_FOURCC, requested_fourcc),
            ("width", cv2.CAP_PROP_FRAME_WIDTH, self.config.width),
            ("height", cv2.CAP_PROP_FRAME_HEIGHT, self.config.height),
            ("fps", cv2.CAP_PROP_FPS, self.config.fps),
            ("buffer_size", cv2.CAP_PROP_BUFFERSIZE, 1),
        )
        rejected = [name for name, prop, value in requested
                    if not capture.set(prop, value)]
        if rejected:
            capture.release()
            self._capture = None
            raise RuntimeError(
                f"{self._tag()} rejected capture settings: {', '.join(rejected)}"
            )
        observed = {name: float(capture.get(prop))
                    for name, prop, _ in requested}
        mismatches = []
        for name, _, expected in requested:
            actual = observed[name]
            if actual > 0 and abs(actual - expected) > (0.5 if name == "fps" else 0.1):
                shown = int(actual) if actual.is_integer() else actual
                mismatches.append(f"{name}={shown} (wanted {expected})")
        if mismatches:
            capture.release()
            self._capture = None
            raise RuntimeError(
                f"{self._tag()} negotiated unexpected mode: {', '.join(mismatches)}"
            )

        self._error = None
        self._running.set()
        self._thread = threading.Thread(
            target=self._update,
            name=f"arducam-{self.config.slot}",
            daemon=True,
        )
        self._thread.start()
        try:
            self.get_frame_with_timestamp(timeout=timeout)
        except Exception:
            self.stop()
            raise
        return self

    # A V4L2 read() can block for OpenCV's select() timeout (~10 s) when the
    # camera stalls. Releasing the capture from another thread while that read
    # is in flight leaves the UVC interface streaming with pending URBs, and
    # the NEXT open of the device fails with VIDIOC_STREAMON EPROTO (seen on
    # the rig: the camera worked once, then failed on every second open). So
    # the reader thread owns the handle and releases it only after its last
    # read has returned; stop() waits for that instead of racing it.
    STOP_JOIN_TIMEOUT_S = 15.0

    def _release_capture(self):
        with self._condition:
            capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def _update(self):
        expected = (self.config.height, self.config.width, 3)
        try:
            while self._running.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue
                timestamp = time.time()
                if frame.shape != expected:
                    with self._condition:
                        self._error = RuntimeError(
                            f"{self._tag()} returned shape {frame.shape}; expected {expected}"
                        )
                        self._running.clear()
                        self._condition.notify_all()
                    return
                with self._condition:
                    self._frame = frame.copy()
                    self._frame_ts = timestamp
                    self._condition.notify_all()
        finally:
            self._release_capture()

    def get_frame_with_timestamp(self, timeout: float = 0.5,
                                 max_age: float = 0.5):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._error is None:
                fresh = (
                    self._frame is not None
                    and self._frame_ts is not None
                    and time.time() - self._frame_ts <= max_age
                )
                if fresh:
                    return self._frame.copy(), float(self._frame_ts)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._frame is None:
                        raise TimeoutError(
                            f"{self._tag()} timed out waiting for first frame"
                        )
                    age = time.time() - self._frame_ts
                    raise TimeoutError(
                        f"{self._tag()} frame is stale ({age:.3f}s old)"
                    )
                self._condition.wait(remaining)
            if self._error is not None:
                raise self._error

    def get_frame(self, timeout: float = 0.5, max_age: float = 0.5):
        return self.get_frame_with_timestamp(timeout=timeout, max_age=max_age)[0]

    def stop(self):
        self._running.clear()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.STOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                # Still inside read(); the thread releases the handle itself
                # when that read returns. Releasing here would corrupt the
                # device state for the next open.
                print(f"[{self._tag()}] WARNING: reader still blocked in read() "
                      f"after {self.STOP_JOIN_TIMEOUT_S:.0f}s; deferring release")
                self._thread = None
                return
        self._thread = None
        self._release_capture()
