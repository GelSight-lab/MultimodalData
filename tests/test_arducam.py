import json

import pytest

from twm.sensor_camera import (
    ArducamConfigError,
    CameraSlot,
    VideoDevice,
    resolve_slots,
    validate_config,
)


def _raw_config(position0="unknown", position1="unknown"):
    return {
        "cameras": [
            {
                "slot": "cam0",
                "id_path": "usb-A",
                "position": position0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "pixel_format": "MJPG",
            },
            {
                "slot": "cam1",
                "id_path": "usb-B",
                "position": position1,
                "width": 640,
                "height": 480,
                "fps": 30,
                "pixel_format": "MJPG",
            },
        ]
    }


def test_validate_config_accepts_two_unknown_positions():
    slots = validate_config(_raw_config())

    assert slots == (
        CameraSlot("cam0", "usb-A", "unknown", 640, 480, 30, "MJPG"),
        CameraSlot("cam1", "usb-B", "unknown", 640, 480, 30, "MJPG"),
    )


def test_validate_config_accepts_complete_left_right_mapping():
    slots = validate_config(_raw_config("left", "right"))

    assert [slot.position for slot in slots] == ["left", "right"]


@pytest.mark.parametrize(
    "change, message",
    [
        (("position", "left", "left"), "positions"),
        (("position", "left", "unknown"), "both unknown"),
        (("position", "top", "unknown"), "position"),
        (("id_path", "usb-A", "usb-A"), "id_path"),
        (("slot", "cam0", "cam0"), "cam0 and cam1"),
    ],
)
def test_validate_config_rejects_ambiguous_identity_or_mapping(change, message):
    field, value0, value1 = change
    raw = _raw_config()
    raw["cameras"][0][field] = value0
    raw["cameras"][1][field] = value1

    with pytest.raises(ArducamConfigError, match=message):
        validate_config(raw)


def test_validate_config_applies_capture_defaults():
    raw = {"cameras": [
        {"slot": "cam0", "id_path": "usb-A", "position": "unknown"},
        {"slot": "cam1", "id_path": "usb-B", "position": "unknown"},
    ]}

    slots = validate_config(raw)

    assert (slots[0].width, slots[0].height, slots[0].fps, slots[0].pixel_format) == (
        640, 480, 30, "MJPG"
    )


def test_resolve_slots_uses_path_and_ignores_duplicate_serial_and_metadata_nodes():
    slots = validate_config(_raw_config())
    devices = [
        VideoDevice("/dev/video6", "usb-A", "SN001", True),
        VideoDevice("/dev/video7", "usb-A", "SN001", False),
        VideoDevice("/dev/video10", "usb-B", "SN001", True),
        VideoDevice("/dev/video11", "usb-B", "SN001", False),
    ]

    resolved = resolve_slots(slots, devices)

    assert [camera.device for camera in resolved] == ["/dev/video6", "/dev/video10"]
    assert [camera.reported_serial for camera in resolved] == ["SN001", "SN001"]


def test_resolve_slots_reports_missing_path_with_inventory():
    slots = validate_config(_raw_config())
    devices = [VideoDevice("/dev/video6", "usb-A", "SN001", True)]

    with pytest.raises(ArducamConfigError, match=r"usb-B.*video6"):
        resolve_slots(slots, devices)


def test_resolve_slots_rejects_multiple_capture_nodes_for_path():
    slots = validate_config(_raw_config())
    devices = [
        VideoDevice("/dev/video6", "usb-A", "SN001", True),
        VideoDevice("/dev/video8", "usb-A", "SN001", True),
        VideoDevice("/dev/video10", "usb-B", "SN001", True),
    ]

    with pytest.raises(ArducamConfigError, match=r"usb-A.*multiple"):
        resolve_slots(slots, devices)
