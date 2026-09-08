"""The viewer shows the Arducam wrist cameras when an episode has them."""
import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (BLOSC filter)

from twm.recorder.frames import Tick, synthetic_tick
from twm.recorder.schema import append_ticks, create_episode_file, write_episode_attrs
from twm.sensor_camera import CameraSlot, ResolvedCamera
from twm.visualize import FramePrefetcher, arducam_labels


def _arducams():
    return (ResolvedCamera(CameraSlot("cam0", "usb-A", "left", 640, 480, 30, "MJPG"),
                           "/dev/video8", "TWML0001"),
            ResolvedCamera(CameraSlot("cam1", "usb-B", "right", 640, 480, 30, "MJPG"),
                           "/dev/video6", "TWMR0001"))


def _episode(tmp_path, with_arducam, n=5):
    f, path = create_episode_file(str(tmp_path), 0, ["A"], ["L", "R"], 30, n_realsense=3,
                                  arducam_config=_arducams() if with_arducam else None)
    ticks = []
    for k in range(n):
        tk = synthetic_tick(100.0 + k / 30, seed=k, n_realsense=3, n_arducam=2 if with_arducam else 0)
        ticks.append(tk)
    append_ticks(f, ticks)
    write_episode_attrs(f, {"valid": True, "invalid_reason": "", "ended_by": "operator",
                            "frame_count": n, "gap_count": 0})
    f.close()
    return path


def test_prefetcher_returns_arducam_frames_when_present(tmp_path):
    path = _episode(tmp_path, with_arducam=True)
    with h5py.File(path, "r") as f:
        pf = FramePrefetcher(path, 5, 5, 5)
        try:
            color, gs, ard = pf._read_frame(f, 3)
        finally:
            pf.stop()
        assert len(color) == 3 and len(gs) == 2
        assert len(ard) == 2
        np.testing.assert_array_equal(ard[0], f["arducam/cam0/frames"][3])
        np.testing.assert_array_equal(ard[1], f["arducam/cam1/frames"][3])
        assert arducam_labels(f) == ["cam0 left", "cam1 right"]


def test_prefetcher_returns_no_arducam_row_for_older_episodes(tmp_path):
    path = _episode(tmp_path, with_arducam=False)
    with h5py.File(path, "r") as f:
        pf = FramePrefetcher(path, 5, 5, 5)
        try:
            color, gs, ard = pf._read_frame(f, 0)
        finally:
            pf.stop()
        assert ard is None
        assert arducam_labels(f) is None
