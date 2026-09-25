"""Wrist cameras stored as the MJPEG the camera already produces.

Decoding them at record time cost 0.3 of a core and turned 1.1 MB of JPEG
into 1.1 MB/tick of raw pixels that BLOSC could not compress — 28 MB/s of
the 33 MB/s by which the rig outruns the disk. The bytes are stored as the
camera sends them and decoded later, where there is time.
"""
import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

from twm.recorder.frames import Tick, decode_arducam
from twm.recorder.schema import append_ticks, create_episode_file
from twm.sensor_camera import CameraSlot, ResolvedCamera


def _arducams():
    return (ResolvedCamera(CameraSlot("cam0", "usb-A", "left", 640, 480, 30, "MJPG"),
                           "/dev/video8", "TWML0001"),
            ResolvedCamera(CameraSlot("cam1", "usb-B", "right", 640, 480, 30, "MJPG"),
                           "/dev/video6", "TWMR0001"))


def _jpeg(value: int, size=(480, 640)):
    img = np.full((*size, 3), value, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.reshape(-1)


def test_an_mjpeg_episode_stores_the_bytes_the_camera_sent(tmp_path):
    f, path = create_episode_file(str(tmp_path), 0, [], [], 30,
                                  arducam_config=_arducams(), include_legacy=False,
                                  arducam_encoding="mjpeg")
    a, b = _jpeg(40), _jpeg(200)
    append_ticks(f, [Tick(100.0, arducam=(a, b), arducam_ts=(99.9, 99.95))])
    f.close()

    with h5py.File(path, "r") as g:
        assert g["arducam/cam0"].attrs["encoding"] == "mjpeg"
        stored = g["arducam/cam0/frames"][0]
        np.testing.assert_array_equal(np.asarray(stored, np.uint8), a)
        img = decode_arducam(stored)
        assert img.shape == (480, 640, 3)
        assert abs(int(img.mean()) - 40) <= 2        # JPEG is lossy, not unrecognisable


def test_the_default_stays_raw_bgr_so_old_readers_keep_working(tmp_path):
    f, path = create_episode_file(str(tmp_path), 1, [], [], 30,
                                  arducam_config=_arducams(), include_legacy=False)
    f.close()
    with h5py.File(path, "r") as g:
        assert g["arducam/cam0"].attrs["encoding"] == "bgr8"
        assert g["arducam/cam0/frames"].shape == (0, 480, 640, 3)


def test_decode_leaves_an_already_decoded_frame_alone():
    """Readers call it on both encodings without asking which they have."""
    img = np.full((480, 640, 3), 7, np.uint8)
    out = decode_arducam(img)
    assert out is img


def test_a_truncated_buffer_raises_instead_of_returning_none(tmp_path):
    """cv2.imdecode returns None on garbage; a reader that passes that on
    turns a corrupt frame into a crash three call sites away."""
    with pytest.raises(ValueError, match="decode"):
        decode_arducam(np.frombuffer(b"\xff\xd8not-a-jpeg", np.uint8))


def test_mjpeg_costs_a_fraction_of_the_raw_bytes(tmp_path):
    """The point of the exercise, asserted rather than assumed."""
    f, path = create_episode_file(str(tmp_path), 2, [], [], 30,
                                  arducam_config=_arducams(), include_legacy=False,
                                  arducam_encoding="mjpeg")
    rng = np.random.default_rng(0)
    ticks = []
    for k in range(20):
        img = (rng.integers(0, 40, (480, 640, 3), dtype=np.uint8)
               + np.linspace(0, 200, 640, dtype=np.uint8)[None, :, None])
        ok, buf = cv2.imencode(".jpg", img.astype(np.uint8))
        b = buf.reshape(-1)
        ticks.append(Tick(100.0 + k / 30, arducam=(b, b), arducam_ts=(0.0, 0.0)))
    append_ticks(f, ticks)
    f.close()
    with h5py.File(path, "r") as g:
        stored = g["arducam/cam0/frames"].id.get_storage_size()
    raw_per_tick = 480 * 640 * 3
    assert stored / 20 < raw_per_tick * 0.35, "JPEG should be well under a third of raw"


def test_the_recorder_records_mjpeg_by_default_and_can_be_told_not_to():
    from twm.recorder.config import parse_args
    assert parse_args(["--task", "t"]).arducam_encoding == "mjpeg"
    assert parse_args(["--task", "t", "--arducam_raw"]).arducam_encoding == "bgr8"


def test_the_encoding_reaches_the_arducam_driver_and_the_file():
    """A flag that stops at the config saves nothing."""
    from twm.recorder.config import RecorderConfig
    from twm.recorder.rig import Drivers, SensorRig

    class _S:
        def __init__(self, *a, **k):
            self._f = np.zeros(4, np.uint8)

        def start(self, timeout=None):
            return None

        def stop(self):
            return None

        def get_frame_with_timestamp(self, **kw):
            return self._f, 1.0

        def get_color_frame(self, **kw):
            return np.zeros((4, 4, 3), np.uint8)

        def get_depth_frame(self, **kw):
            return np.zeros((4, 4), np.uint16)

        def get_frame(self, **kw):
            return np.zeros((4, 4, 3), np.uint8)

    class _Cam:
        def __init__(self, slot):
            self.slot, self.id_path, self.position = slot, f"usb-{slot}", "unknown"
            self.config, self.device = {}, f"/dev/{slot}"

    seen = []
    d = Drivers(
        realsense=lambda serial, fps, align=True: _S(),
        gelsight=lambda serial, resolution, name: _S(),
        optitrack=lambda: None,
        arducam=lambda config, device, encoding="bgr8": seen.append(encoding) or _S(),
        resolve_arducams=lambda path: [_Cam("cam0"), _Cam("cam1")],
        sleep=lambda s: None,
    )
    cfg = RecorderConfig(task="t", realsense_serials=("A",), use_optitrack=False,
                         gelsight_serials={"left": "L", "right": "R"})
    rig = SensorRig.open(cfg, d)
    try:
        assert seen == ["mjpeg", "mjpeg"]          # the default
        assert rig.arducam_encoding == "mjpeg"
    finally:
        rig.close()
