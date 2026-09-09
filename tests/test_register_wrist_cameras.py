"""Writing the wrist-camera config from what is actually plugged in.

Hand-editing it does not survive contact with the rig: these cameras report
the same serial, so identity is the USB port, and the port changes whenever
someone re-plugs. The registration refuses rather than guessing, because a
config that silently names one camera twice records one camera twice.
"""
import json

import pytest

from twm.sensor_camera import (ArducamConfigError, VideoDevice,
                               register_wrist_cameras)

GEL = ("2DUPB53G", "2BKRDTAD")


def dev(node, path, serial, capture=True, vendor="1e45"):
    return VideoDevice(f"/dev/video{node}", path, serial, capture, vendor)


def rig(*extra):
    """The rig's own cameras, which are never wrist cameras."""
    return [
        dev(0, "pci-0:4:1.0", "143523020603", vendor="8086"),   # RealSense
        dev(2, "pci-0:4:1.0", "143523020603", vendor="8086"),   # its second node
        dev(14, "pci-0:3:1.0", "219523020530", vendor="8086"),
        dev(6, "pci-0:12.3:1.0", GEL[0]),
        dev(8, "pci-0:12.4:1.0", GEL[1]),
        dev(9, "pci-0:12.4:1.0", GEL[1], capture=False),
        *extra,
    ]


def test_two_unclaimed_cameras_are_written_keyed_by_port(tmp_path):
    out = tmp_path / "wrist.json"
    cams = register_wrist_cameras(
        out, devices=rig(dev(12, "pci-0:1:1.0", "200901010001"),
                         dev(10, "pci-0:2:1.0", "200901010001")),
        realsense_serials=("143523020603", "219523020530"), gelsight_serials=GEL)
    assert [c["slot"] for c in cams] == ["cam0", "cam1"]
    assert [c["id_path"] for c in cams] == ["pci-0:1:1.0", "pci-0:2:1.0"]
    assert all("serial" not in c for c in cams), "a shared serial identifies neither"
    saved = json.loads(out.read_text())["cameras"]
    assert [c["id_path"] for c in saved] == ["pci-0:1:1.0", "pci-0:2:1.0"]
    assert saved[0]["controls"]["exposure_dynamic_framerate"] == 0


def test_one_camera_is_refused_and_says_what_it_found(tmp_path):
    """Exactly the state the rig was in: one of the pair unplugged."""
    with pytest.raises(ArducamConfigError) as exc:
        register_wrist_cameras(tmp_path / "w.json",
                               devices=rig(dev(10, "pci-0:2:1.0", "200901010001")),
                               realsense_serials=("143523020603", "219523020530"),
                               gelsight_serials=GEL)
    assert "1" in str(exc.value) and "/dev/video10" in str(exc.value)
    assert not (tmp_path / "w.json").exists(), "a refusal must not write a file"


def test_three_candidates_are_refused_rather_than_truncated(tmp_path):
    with pytest.raises(ArducamConfigError, match="3"):
        register_wrist_cameras(tmp_path / "w.json",
                               devices=rig(dev(12, "pci-0:1:1.0", "A"),
                                           dev(10, "pci-0:2:1.0", "B"),
                                           dev(11, "pci-0:5:1.0", "C")),
                               realsense_serials=("143523020603", "219523020530"),
                               gelsight_serials=GEL)


def test_an_existing_left_right_assignment_survives_a_re_registration(tmp_path):
    out = tmp_path / "wrist.json"
    out.write_text(json.dumps({"cameras": [
        {"slot": "cam0", "id_path": "old-path", "position": "left"},
        {"slot": "cam1", "id_path": "other", "position": "right"}]}))
    cams = register_wrist_cameras(
        out, devices=rig(dev(12, "pci-0:1:1.0", "X"), dev(10, "pci-0:2:1.0", "Y")),
        realsense_serials=("143523020603", "219523020530"), gelsight_serials=GEL)
    assert [c["position"] for c in cams] == ["left", "right"]


def test_a_realsense_is_excluded_by_vendor_even_with_an_unknown_serial(tmp_path):
    """librealsense reports one serial and the USB layer another, so the
    serial list cannot be relied on to exclude them."""
    cams = register_wrist_cameras(
        tmp_path / "w.json",
        devices=[dev(30, "pci-0:7:1.0", "totally-unknown", vendor="8086"),
                 dev(12, "pci-0:1:1.0", "S1"), dev(10, "pci-0:2:1.0", "S2")],
        realsense_serials=(), gelsight_serials=GEL)
    assert [c["id_path"] for c in cams] == ["pci-0:1:1.0", "pci-0:2:1.0"]
