import unittest
from unittest.mock import MagicMock, patch
import sys
import time

# Mock rospy and geometry_msgs before importing
sys.modules['rospy'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()

from optitrack.optitrack_stream import OptitrackStream


class TestOptitrackStream(unittest.TestCase):

    def _make_pose_msg(self, x, y, z, qx, qy, qz, qw, stamp=0.0):
        msg = MagicMock()
        msg.header.stamp.to_sec.return_value = stamp
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def test_get_latest_pose_none_before_any_message(self):
        stream = OptitrackStream()
        self.assertIsNone(stream.get_latest_pose("motherboard"))

    def test_get_latest_pose_after_callback(self):
        stream = OptitrackStream()
        msg = self._make_pose_msg(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, stamp=100.0)
        cb = stream._make_callback("motherboard")
        cb(msg)

        result = stream.get_latest_pose("motherboard")
        self.assertIsNotNone(result)
        t, pose = result
        self.assertAlmostEqual(t, 100.0)
        self.assertEqual(pose, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    def test_flush_buffer_returns_all_and_clears(self):
        stream = OptitrackStream()
        cb = stream._make_callback("sensor_left")
        for i in range(5):
            msg = self._make_pose_msg(float(i), 0, 0, 0, 0, 0, 1.0, stamp=float(i))
            cb(msg)

        data = stream.flush_buffer("sensor_left")
        self.assertEqual(len(data), 5)
        # Buffer should be cleared
        self.assertEqual(len(stream.flush_buffer("sensor_left")), 0)

    def test_flush_buffer_returns_timestamps_and_poses(self):
        stream = OptitrackStream()
        cb = stream._make_callback("sensor_right")
        msg = self._make_pose_msg(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, stamp=42.0)
        cb(msg)

        data = stream.flush_buffer("sensor_right")
        t, pose = data[0]
        self.assertAlmostEqual(t, 42.0)
        self.assertEqual(len(pose), 7)


if __name__ == '__main__':
    unittest.main()
