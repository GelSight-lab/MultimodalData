"""One wrist camera is a valid rig, not a broken one.

The pair was two Arducams and everything from the config schema to the
preview panel hard-coded "exactly two". Testing a single camera before the
mount is built — or losing one mid-session — should not require editing the
stack.
"""
import json

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

from twm.recorder.frames import Tick
from twm.recorder.schema import append_ticks, create_episode_file
from twm.sensor_camera import (ArducamConfigError, CameraSlot, ResolvedCamera,
                               VideoDevice, register_wrist_cameras,
                               resolve_slots, validate_config)
from twm.viz import build_preview_panel

ONE = {"cameras": [{"slot": "cam0", "id_path": "pci-0:1:1.0", "position": "left",
                    "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"}]}


def test_a_one_camera_config_validates():
    slots = validate_config(ONE)
    assert len(slots) == 1 and slots[0].slot == "cam0"


def test_a_config_with_no_cameras_is_still_refused():
    with pytest.raises(ArducamConfigError):
        validate_config({"cameras": []})


def test_a_one_camera_config_resolves_to_one_device():
    devs = [VideoDevice("/dev/video12", "pci-0:1:1.0", "S", True, "1e45")]
    got = resolve_slots(validate_config(ONE), devs)
    assert len(got) == 1 and got[0].device == "/dev/video12"


def test_an_episode_with_one_wrist_camera_records_it(tmp_path):
    cam = ResolvedCamera(CameraSlot("cam0", "pci-0:1:1.0", "left", 640, 480, 30, "MJPG"),
                         "/dev/video12", "S")
    f, path = create_episode_file(str(tmp_path), 0, [], [], 30, arducam_config=(cam,),
                                  include_legacy=False, arducam_encoding="mjpeg")
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8))
    append_ticks(f, [Tick(100.0, arducam=(buf.reshape(-1),), arducam_ts=(99.9,))])
    f.close()
    with h5py.File(path, "r") as g:
        assert list(g["arducam"]) == ["cam0"]
        assert g["arducam/cam0/frames"].shape == (1,)


def test_the_preview_panel_takes_one_wrist_frame(tmp_path):
    color = [np.zeros((480, 640, 3), np.uint8) for _ in range(3)]
    gs = [np.zeros((480, 640, 3), np.uint8) for _ in range(2)]
    one = [np.full((480, 640, 3), 40, np.uint8)]
    panel = build_preview_panel(color, gs, gs, {}, False, 0, 0.0,
                                arducam_frames=one, arducam_labels=["wrist left"])
    from twm.viz import PANEL_H, PANEL_W, RS_THUMB_H
    assert panel.shape == (PANEL_H + RS_THUMB_H, PANEL_W, 3)


def test_registering_one_camera_writes_a_one_camera_config(tmp_path):
    out = tmp_path / "w.json"
    cams = register_wrist_cameras(
        out, devices=[VideoDevice("/dev/video10", "pci-0:2:1.0", "S", True, "1e45")],
        realsense_serials=(), gelsight_serials=(), allow_one=True)
    assert [c["slot"] for c in cams] == ["cam0"]
    assert json.loads(out.read_text())["cameras"][0]["id_path"] == "pci-0:2:1.0"


def test_registering_one_camera_still_refuses_by_default(tmp_path):
    """The pair is the normal rig; a single camera has to be asked for."""
    with pytest.raises(ArducamConfigError, match="allow-one"):
        register_wrist_cameras(
            tmp_path / "w.json",
            devices=[VideoDevice("/dev/video10", "pci-0:2:1.0", "S", True, "1e45")],
            realsense_serials=(), gelsight_serials=())
