import threading
import time
import numpy as np

import pyrealsense2 as rs


class RealsenseStream:
    """
    Threaded RealSense D415 stream providing color (BGR uint8) and depth
    (uint16, millimetres) frames at a fixed fps. Depth is reprojected onto
    the color grid unless `align=False`.

    Usage:
        stream = RealsenseStream(serial="123456789012", fps=30)
        stream.start()
        color = stream.get_color_frame()   # (480, 640, 3) uint8
        depth = stream.get_depth_frame()   # (480, 640)    uint16
        stream.stop()
    """

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30,
                 align: bool = True):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        # `align=False` stores depth in the DEPTH camera's frame: the same
        # pixels rs.align would consume, without paying for the reprojection
        # on the recording machine (~0.25 of a core per camera). Align it
        # afterwards with twm.realsense_align, which reproduces rs.align.
        self.align = align

        self._color_frame = None
        self._depth_frame = None
        self._lock = threading.Lock()
        self._streaming = False
        self._last_updated = 0.0

    def start(self):
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._align = rs.align(rs.stream.color) if self.align else None
        self._pipeline.start(config)
        self._streaming = True
        threading.Thread(target=self._update, daemon=True).start()

    def stop(self):
        self._streaming = False
        if hasattr(self, '_pipeline'):
            self._pipeline.stop()

    def _update(self):
        while self._streaming:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if self._align is not None:
                    frames = self._align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if color and depth:
                    with self._lock:
                        self._color_frame = np.asanyarray(color.get_data()).copy()
                        self._depth_frame = np.asanyarray(depth.get_data()).copy()
                        self._last_updated = time.time()
            except Exception as e:
                print(f"[RealsenseStream {self.serial}] error: {e}")
                time.sleep(0.01)

    def get_color_frame(self, timeout: float = 2.0, max_age: float = 0.5) -> np.ndarray:
        """Block until a fresh color frame is available, then return a copy.

        Raises TimeoutError if no fresh frame arrives within `timeout` seconds.
        A frame is considered stale if it was captured more than `max_age` seconds ago.
        """
        t0 = time.time()
        while True:
            with self._lock:
                if self._color_frame is not None and time.time() - self._last_updated <= max_age:
                    return self._color_frame.copy()
            if time.time() - t0 > timeout:
                age = time.time() - self._last_updated
                raise TimeoutError(
                    f"No fresh color frame from RealSense serial={self.serial} "
                    f"(last updated {age:.2f}s ago)"
                )
            time.sleep(0.01)

    def get_depth_frame(self, timeout: float = 2.0, max_age: float = 0.5) -> np.ndarray:
        """Block until a fresh depth frame is available, then return a copy."""
        t0 = time.time()
        while True:
            with self._lock:
                if self._depth_frame is not None and time.time() - self._last_updated <= max_age:
                    return self._depth_frame.copy()
            if time.time() - t0 > timeout:
                age = time.time() - self._last_updated
                raise TimeoutError(
                    f"No fresh depth frame from RealSense serial={self.serial} "
                    f"(last updated {age:.2f}s ago)"
                )
            time.sleep(0.01)
