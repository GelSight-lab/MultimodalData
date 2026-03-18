import threading
import time
import numpy as np

import pyrealsense2 as rs


class RealsenseStream:
    """
    Threaded RealSense D415 stream providing aligned color (BGR uint8) and
    depth (uint16, millimetres) frames at a fixed fps.

    Usage:
        stream = RealsenseStream(serial="123456789012", fps=30)
        stream.start()
        color = stream.get_color_frame()   # (480, 640, 3) uint8
        depth = stream.get_depth_frame()   # (480, 640)    uint16
        stream.stop()
    """

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps

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
        self._align = rs.align(rs.stream.color)
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
                aligned = self._align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if color and depth:
                    color_arr = np.asanyarray(color.get_data())
                    depth_arr = np.asanyarray(depth.get_data())
                    with self._lock:
                        self._color_frame = color_arr
                        self._depth_frame = depth_arr
                        self._last_updated = time.time()
            except Exception as e:
                print(f"[RealsenseStream {self.serial}] error: {e}")
                time.sleep(0.01)

    def get_color_frame(self, timeout: float = 2.0) -> np.ndarray:
        """Block until a color frame is available, then return a copy."""
        t0 = time.time()
        while self._color_frame is None:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"No color frame from RealSense serial={self.serial}")
            time.sleep(0.01)
        with self._lock:
            return self._color_frame.copy()

    def get_depth_frame(self, timeout: float = 2.0) -> np.ndarray:
        """Block until a depth frame is available, then return a copy."""
        t0 = time.time()
        while self._depth_frame is None:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"No depth frame from RealSense serial={self.serial}")
            time.sleep(0.01)
        with self._lock:
            return self._depth_frame.copy()
