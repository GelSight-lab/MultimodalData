"""The wrist cameras belong in the release, in step with everything else.

They reached the raw recordings in 2026-09 and stopped there: the release
video schema had three RealSense views and two GelSights and no slot for
them, so every published episode since silently dropped two streams.
"""
import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

from twm.react_preprocess.config import WRIST_STREAM
from twm.react_preprocess.pipeline import _encode_wrist
from twm.recorder.schema import append_ticks, create_episode_file
from twm.recorder.frames import Tick
from twm.sensor_camera import CameraSlot, ResolvedCamera


def _arducams():
    return (ResolvedCamera(CameraSlot("cam0", "usb-A", "left", 640, 480, 30, "MJPG"),
                           "/dev/video8", "TWML0001"),
            ResolvedCamera(CameraSlot("cam1", "usb-B", "right", 640, 480, 30, "MJPG"),
                           "/dev/video6", "TWMR0001"))


class _Source:
    def __init__(self, T, trim=0):
        self.T, self.trim = T, trim


def _episode(tmp_path, encoding, n=6):
    f, path = create_episode_file(str(tmp_path), 0, [], [], 30, arducam_config=_arducams(),
                                  include_legacy=False, arducam_encoding=encoding)
    ticks = []
    for k in range(n):
        img = np.full((480, 640, 3), 20 + 30 * k, np.uint8)
        img[:, :50] = 200                                  # something to see
        if encoding == "mjpeg":
            ok, buf = cv2.imencode(".jpg", img)
            frame = buf.reshape(-1)
        else:
            frame = img
        ticks.append(Tick(100.0 + k / 30, arducam=(frame, frame),
                          arducam_ts=(100.0 + k / 30,) * 2))
    append_ticks(f, ticks)
    f.close()
    return path


def _frame_count(mp4):
    c = cv2.VideoCapture(str(mp4))
    try:
        return int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        c.release()


def test_the_stream_names_say_which_wrist():
    assert WRIST_STREAM == {"cam0": "wrist_left", "cam1": "wrist_right"}


@pytest.mark.parametrize("encoding", ["bgr8", "mjpeg"])
def test_both_encodings_produce_one_video_per_wrist(tmp_path, encoding):
    path = _episode(tmp_path, encoding)
    out = tmp_path / "videos"
    out.mkdir()
    with h5py.File(path, "r") as f:
        written = _encode_wrist(f, _Source(6), out)
    assert written == 2
    for name in WRIST_STREAM.values():
        assert _frame_count(out / f"{name}.mp4") == 6


def test_the_wrist_video_starts_at_the_release_trim(tmp_path):
    """Every other stream is cut at the trim; a wrist video that is not
    plays two frames ahead of the tactile beside it."""
    path = _episode(tmp_path, "mjpeg", n=10)
    out = tmp_path / "videos"
    out.mkdir()
    with h5py.File(path, "r") as f:
        _encode_wrist(f, _Source(8, trim=2), out)
    assert _frame_count(out / "wrist_left.mp4") == 8


def test_an_episode_without_wrist_cameras_writes_nothing(tmp_path):
    f, path = create_episode_file(str(tmp_path), 1, ["A"], ["L", "R"], 30, n_realsense=1)
    f.close()
    out = tmp_path / "videos"
    out.mkdir()
    with h5py.File(path, "r") as g:
        assert _encode_wrist(g, _Source(0), out) == 0
    assert not list(out.iterdir())
