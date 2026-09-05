import json
import time

import h5py
import numpy as np

import pytest

from twm.sensor_camera import (
    ArducamConfigError,
    CameraSlot,
    VideoDevice,
    record_verification,
    resolve_slots,
    save_position_mapping,
    validate_config,
    verify_recording,
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


def test_save_position_mapping_atomically_assigns_or_clears_sides(tmp_path):
    path = tmp_path / "arducam.json"
    path.write_text(json.dumps(_raw_config()))

    save_position_mapping(path, left_slot="cam1")
    mapped = json.loads(path.read_text())
    assert [entry["position"] for entry in mapped["cameras"]] == ["right", "left"]

    save_position_mapping(path, left_slot=None)
    cleared = json.loads(path.read_text())
    assert [entry["position"] for entry in cleared["cameras"]] == [
        "unknown", "unknown"
    ]
    assert not list(tmp_path.glob("*.tmp"))


class FakeSensorCameraStream:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.index = 0
        self.stopped = False
        self.value = 30 if config.slot == "cam0" else 130

    def start(self, timeout=5.0):
        return self

    def get_frame_with_timestamp(self, timeout=0.5):
        self.index += 1
        image = np.full((480, 640, 3), self.value, np.uint8)
        image[:, self.index % 640, :] = self.value + 10
        return image, 1000.0 + self.index / 30.0

    def stop(self):
        self.stopped = True


class FrozenSlowSensorCameraStream(FakeSensorCameraStream):
    def get_frame_with_timestamp(self, timeout=0.5):
        self.index += 1
        x = np.arange(640, dtype=np.uint8)[None, :, None] % 31
        image = np.broadcast_to(x, (480, 640, 3)).copy() + self.value
        return image, 1000.0 + self.index / 16.0


def test_record_verification_uses_real_writer_and_reopens_valid_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_raw_config()))
    output = tmp_path / "verify.h5"
    devices = [
        VideoDevice("/dev/video6", "usb-A", "SN001", True),
        VideoDevice("/dev/video10", "usb-B", "SN001", True),
    ]

    report = record_verification(
        config_path=config_path,
        output_path=output,
        duration=0.12,
        devices=devices,
        stream_factory=FakeSensorCameraStream,
    )

    assert report["ok"] is True
    assert report["frame_count"] >= 3
    with h5py.File(output, "r") as f:
        assert f["arducam/cam0/frames"].shape[0] == report["frame_count"]
        assert f["arducam/cam1/frames"].shape[0] == report["frame_count"]


def test_record_verification_does_not_touch_sibling_episode_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_raw_config()))
    sibling = tmp_path / "episode_000.h5"
    sibling.write_bytes(b"unrelated recording")
    devices = [
        VideoDevice("/dev/video6", "usb-A", "SN001", True),
        VideoDevice("/dev/video10", "usb-B", "SN001", True),
    ]

    report = record_verification(
        config_path=config_path,
        output_path=tmp_path / "verify.h5",
        duration=0.12,
        devices=devices,
        stream_factory=FakeSensorCameraStream,
    )

    assert report["ok"] is True
    assert sibling.read_bytes() == b"unrelated recording"


def test_verify_recording_reports_black_and_identical_streams(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_raw_config()))
    output = tmp_path / "verify.h5"
    devices = [
        VideoDevice("/dev/video6", "usb-A", "SN001", True),
        VideoDevice("/dev/video10", "usb-B", "SN001", True),
    ]
    record_verification(
        config_path=config_path,
        output_path=output,
        duration=0.12,
        devices=devices,
        stream_factory=FakeSensorCameraStream,
    )
    with h5py.File(output, "r+") as f:
        f["arducam/cam0/frames"][:] = 0
        f["arducam/cam1/frames"][:] = 0

    report = verify_recording(output, requested_fps=30)

    assert report["ok"] is False
    assert report["checks"]["cam0_nonblack"] is False
    assert report["checks"]["streams_distinct"] is False


def test_verifier_rejects_under_rate_temporally_frozen_video(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_raw_config()))
    devices = [
        VideoDevice("/dev/video6", "usb-A", "SN001", True),
        VideoDevice("/dev/video10", "usb-B", "SN001", True),
    ]

    report = record_verification(
        config_path=config_path,
        output_path=tmp_path / "slow.h5",
        duration=0.12,
        devices=devices,
        stream_factory=FrozenSlowSensorCameraStream,
    )

    assert report["ok"] is False
    assert report["checks"]["cam0_cadence"] is False
    assert report["checks"]["cam0_temporal_change"] is False
