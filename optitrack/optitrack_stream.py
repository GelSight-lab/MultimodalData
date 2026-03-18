import threading
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped

TOPICS = {
    "motherboard": "/vrpn_client_node/motherboard/pose",
    "sensor_left":  "/vrpn_client_node/sensor_left/pose",
    "sensor_right": "/vrpn_client_node/sensor_right/pose",
}


class OptitrackStream:
    """
    Subscribes to three OptiTrack VRPN pose topics and buffers incoming poses.

    Usage:
        stream = OptitrackStream()
        stream.start()
        t, pose = stream.get_latest_pose("sensor_left")   # (float, [x,y,z,qx,qy,qz,qw])
        data = stream.flush_buffer("sensor_left")          # [(t, pose), ...]
        stream.stop()
    """

    def __init__(self, buffer_size: int = 50000):
        self._buffers = {name: deque(maxlen=buffer_size) for name in TOPICS}
        self._latest  = {name: None for name in TOPICS}
        self._lock = threading.Lock()
        self._running = False

    def _make_callback(self, name: str):
        def callback(msg: PoseStamped):
            t = msg.header.stamp.to_sec()
            pose = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
            with self._lock:
                self._buffers[name].append((t, pose))
                self._latest[name] = (t, pose)
        return callback

    def start(self):
        try:
            rospy.init_node("twm_data_collection", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass  # node already initialized
        for name, topic in TOPICS.items():
            rospy.Subscriber(topic, PoseStamped, self._make_callback(name))
        self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        self._spin_thread.start()
        self._running = True

    def stop(self):
        self._running = False

    def get_latest_pose(self, name: str):
        """Return (timestamp, [x,y,z,qx,qy,qz,qw]) or None if no data yet."""
        with self._lock:
            return self._latest[name]

    def flush_buffer(self, name: str):
        """Return all buffered (timestamp, pose) pairs and clear the buffer."""
        with self._lock:
            data = list(self._buffers[name])
            self._buffers[name].clear()
            return data
