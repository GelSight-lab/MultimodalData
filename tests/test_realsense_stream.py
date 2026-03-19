import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import sys

# Mock pyrealsense2 before importing
mock_rs = MagicMock()
sys.modules['pyrealsense2'] = mock_rs

from camera_stream.realsense_stream import RealsenseStream


class TestRealsenseStream(unittest.TestCase):

    def test_get_color_frame_returns_copy(self):
        """get_color_frame() returns a numpy array copy when a frame is available."""
        stream = RealsenseStream(serial="123456")
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._color_frame = fake_frame
        stream._last_updated = __import__('time').time()

        result = stream.get_color_frame()
        self.assertEqual(result.shape, (480, 640, 3))
        # Ensure it's a copy, not the same object
        self.assertIsNot(result, fake_frame)

    def test_get_depth_frame_returns_copy(self):
        """get_depth_frame() returns a numpy array copy when a frame is available."""
        stream = RealsenseStream(serial="123456")
        fake_depth = np.zeros((480, 640), dtype=np.uint16)
        stream._depth_frame = fake_depth
        stream._last_updated = __import__('time').time()

        result = stream.get_depth_frame()
        self.assertEqual(result.shape, (480, 640))
        self.assertIsNot(result, fake_depth)

    def test_get_color_frame_times_out(self):
        """get_color_frame() raises TimeoutError if no frame arrives within timeout."""
        stream = RealsenseStream(serial="123456")
        stream._color_frame = None
        with self.assertRaises(TimeoutError):
            stream.get_color_frame(timeout=0.05)

    def test_get_depth_frame_times_out(self):
        """get_depth_frame() raises TimeoutError if no frame arrives within timeout."""
        stream = RealsenseStream(serial="123456")
        stream._depth_frame = None
        with self.assertRaises(TimeoutError):
            stream.get_depth_frame(timeout=0.05)


    def test_get_color_frame_raises_on_stale_frame(self):
        """get_color_frame() raises TimeoutError if the frame is too old."""
        stream = RealsenseStream(serial="123456")
        stream._color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._last_updated = __import__('time').time() - 10.0  # 10 seconds stale
        with self.assertRaises(TimeoutError):
            stream.get_color_frame(timeout=0.05)

    def test_get_depth_frame_raises_on_stale_frame(self):
        """get_depth_frame() raises TimeoutError if the frame is too old."""
        stream = RealsenseStream(serial="123456")
        stream._depth_frame = np.zeros((480, 640), dtype=np.uint16)
        stream._last_updated = __import__('time').time() - 10.0
        with self.assertRaises(TimeoutError):
            stream.get_depth_frame(timeout=0.05)

    def test_get_color_frame_fresh_succeeds(self):
        """get_color_frame() succeeds when _last_updated is recent."""
        stream = RealsenseStream(serial="123456")
        stream._color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stream._last_updated = __import__('time').time()  # just now
        result = stream.get_color_frame()
        self.assertEqual(result.shape, (480, 640, 3))


if __name__ == '__main__':
    unittest.main()
