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
    def __init__(self, T, trim=0, task="pushT"):
        self.T, self.trim, self.task = T, trim, task


def _episode(tmp_path, encoding, n=6):
    f, path = create_episode_file(str(tmp_path), 0, [], [], 30, arducam_config=_arducams(),
                                  include_legacy=False, arducam_encoding=encoding)
    ticks = []
    for k in range(n):
        # 20..180, distinct per frame and never the 200 of the marker band.
        # `20 + 30 * k` left uint8 at k >= 8: NumPy 2 raises OverflowError,
        # and NumPy 1 wrapped silently, so frame 8 (260 -> 4) sat next to
        # frame 0 (20) and "each frame is distinguishable" was quietly false.
        fill = 20 + round(k * 160 / max(n - 1, 1))
        assert fill < 200, "fills must stay clear of the marker band"
        img = np.full((480, 640, 3), fill, np.uint8)
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
    # One entry per wrist video, carrying the tone exponent that stream was
    # published through, so the episode metadata can declare it.
    assert set(written) == set(WRIST_STREAM)
    assert all(g >= 1.0 for g in written.values())
    for name in WRIST_STREAM.values():
        assert _frame_count(out / f"{name}.mp4") == 6


def _first_frame_mean(mp4) -> float:
    c = cv2.VideoCapture(str(mp4))
    try:
        ok, frame = c.read()
        assert ok, f"{mp4.name}: could not decode frame 0"
        return float(frame[:, 100:].mean())      # past the marker band
    finally:
        c.release()


def test_the_wrist_video_starts_at_the_release_trim(tmp_path):
    """Every other stream is cut at the trim; a wrist video that is not
    plays two frames ahead of the tactile beside it.

    Checking only the FRAME COUNT cannot see that: taking source frames 0-7
    gives 8 frames just as taking 2-9 does, so the very defect this test is
    named for passed it. The fixture paints each source frame a different
    shade precisely so the content can be checked, and nothing checked it.

    Compared against a trim=0 encode rather than against a raw fill value,
    because `_encode_wrist` puts every frame through a tone curve — the
    comparison has to survive that, and a monotone curve preserves which
    source frame a published frame came from.
    """
    path = _episode(tmp_path, "mjpeg", n=10)
    cut, uncut = tmp_path / "cut", tmp_path / "uncut"
    cut.mkdir(), uncut.mkdir()
    with h5py.File(path, "r") as f:
        _encode_wrist(f, _Source(8, trim=2), cut)
        _encode_wrist(f, _Source(10, trim=0), uncut)

    assert _frame_count(cut / "wrist_left.mp4") == 8

    at_trim = _first_frame_mean(cut / "wrist_left.mp4")
    source = [_nth_frame_mean(uncut / "wrist_left.mp4", k) for k in range(10)]
    nearest = min(range(10), key=lambda k: abs(source[k] - at_trim))
    assert nearest == 2, (
        f"the cut video opens on source frame {nearest}, not the trim (2). "
        f"frame 0 mean {at_trim:.1f}; source means {[round(m, 1) for m in source]}")


def _nth_frame_mean(mp4, n: int) -> float:
    c = cv2.VideoCapture(str(mp4))
    try:
        for _ in range(n):
            assert c.read()[0]
        ok, frame = c.read()
        assert ok, f"{mp4.name}: no frame {n}"
        return float(frame[:, 100:].mean())
    finally:
        c.release()


def test_an_episode_without_wrist_cameras_writes_nothing(tmp_path):
    f, path = create_episode_file(str(tmp_path), 1, ["A"], ["L", "R"], 30, n_realsense=1)
    f.close()
    out = tmp_path / "videos"
    out.mkdir()
    with h5py.File(path, "r") as g:
        assert _encode_wrist(g, _Source(0), out) == {}
    assert not list(out.iterdir())
