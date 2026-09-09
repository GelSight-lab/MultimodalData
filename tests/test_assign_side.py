"""Naming a side by unplugging the other camera.

Two identical cameras cannot be told apart in a preview — they report the
same serial and look the same. Unplugging one and naming the other is the
procedure that actually works on the bench, so it is a command rather than a
note.
"""
import json

import pytest

from twm.sensor_camera import (ArducamConfigError, VideoDevice,
                               assign_side_from_connected)

PORT1, PORT2 = "pci-0:1:1.0", "pci-0:2:1.0"


def cfg(tmp_path, positions=("unknown", "unknown")):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"cameras": [
        {"slot": "cam0", "id_path": PORT1, "position": positions[0],
         "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"},
        {"slot": "cam1", "id_path": PORT2, "position": positions[1],
         "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"}]}))
    return p


def dev(path):
    return VideoDevice("/dev/video12", path, "200901010001", True, "1e45")


def sides(path):
    return {c["id_path"]: c["position"] for c in json.loads(path.read_text())["cameras"]}


def test_the_connected_camera_takes_the_named_side_and_the_other_takes_the_opposite(tmp_path):
    p = cfg(tmp_path)
    assign_side_from_connected(p, "left", devices=[dev(PORT1)],
                               realsense_serials=(), gelsight_serials=())
    assert sides(p) == {PORT1: "left", PORT2: "right"}


def test_naming_the_connected_one_right_puts_left_on_the_other(tmp_path):
    p = cfg(tmp_path)
    assign_side_from_connected(p, "right", devices=[dev(PORT2)],
                               realsense_serials=(), gelsight_serials=())
    assert sides(p) == {PORT1: "left", PORT2: "right"}


def test_two_connected_cameras_are_refused(tmp_path):
    """With both plugged in there is nothing to distinguish them."""
    p = cfg(tmp_path)
    with pytest.raises(ArducamConfigError, match="exactly one"):
        assign_side_from_connected(p, "left", devices=[dev(PORT1), dev(PORT2)],
                                   realsense_serials=(), gelsight_serials=())
    assert sides(p) == {PORT1: "unknown", PORT2: "unknown"}, "a refusal must not write"


def test_a_camera_that_is_not_in_the_config_is_refused(tmp_path):
    p = cfg(tmp_path)
    with pytest.raises(ArducamConfigError, match="pci-0:9:1.0"):
        assign_side_from_connected(p, "left", devices=[dev("pci-0:9:1.0")],
                                   realsense_serials=(), gelsight_serials=())


def test_a_one_camera_config_just_names_that_camera(tmp_path):
    p = tmp_path / "one.json"
    p.write_text(json.dumps({"cameras": [
        {"slot": "cam0", "id_path": PORT1, "position": "unknown",
         "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG"}]}))
    assign_side_from_connected(p, "left", devices=[dev(PORT1)],
                               realsense_serials=(), gelsight_serials=())
    assert sides(p) == {PORT1: "left"}


def test_an_unknown_side_name_is_refused(tmp_path):
    p = cfg(tmp_path)
    with pytest.raises(ArducamConfigError, match="left"):
        assign_side_from_connected(p, "middle", devices=[dev(PORT1)],
                                   realsense_serials=(), gelsight_serials=())
