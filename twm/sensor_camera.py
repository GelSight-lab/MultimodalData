"""Configuration and stable discovery for TWM sensor-mounted cameras."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config") / "arducam.json"
VALID_POSITIONS = frozenset({"unknown", "left", "right"})


class ArducamConfigError(ValueError):
    """The camera configuration cannot identify exactly two capture devices."""


@dataclass(frozen=True)
class CameraSlot:
    slot: str
    id_path: str
    position: str = "unknown"
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "MJPG"


@dataclass(frozen=True)
class VideoDevice:
    device: str
    id_path: str
    reported_serial: str
    is_capture: bool


@dataclass(frozen=True)
class ResolvedCamera:
    config: CameraSlot
    device: str
    reported_serial: str

    @property
    def slot(self) -> str:
        return self.config.slot

    @property
    def id_path(self) -> str:
        return self.config.id_path

    @property
    def position(self) -> str:
        return self.config.position

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def fps(self) -> int:
        return self.config.fps

    @property
    def pixel_format(self) -> str:
        return self.config.pixel_format


def _positive_int(entry: Mapping, key: str, default: int) -> int:
    try:
        value = int(entry.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ArducamConfigError(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise ArducamConfigError(f"{key} must be a positive integer")
    return value


def validate_config(raw: Mapping) -> tuple[CameraSlot, CameraSlot]:
    """Validate and normalize the two-slot JSON representation."""
    entries = raw.get("cameras") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list) or len(entries) != 2:
        raise ArducamConfigError("configuration must contain exactly two cameras")

    slots = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArducamConfigError("each camera must be an object")
        slot = str(entry.get("slot", ""))
        id_path = str(entry.get("id_path", ""))
        position = str(entry.get("position", "unknown")).lower()
        if not id_path:
            raise ArducamConfigError("each camera needs a nonempty id_path")
        if position not in VALID_POSITIONS:
            raise ArducamConfigError(
                f"position must be one of {sorted(VALID_POSITIONS)}, got {position!r}"
            )
        pixel_format = str(entry.get("pixel_format", "MJPG")).upper()
        if len(pixel_format) != 4:
            raise ArducamConfigError("pixel_format must be a four-character V4L2 code")
        slots.append(CameraSlot(
            slot=slot,
            id_path=id_path,
            position=position,
            width=_positive_int(entry, "width", 640),
            height=_positive_int(entry, "height", 480),
            fps=_positive_int(entry, "fps", 30),
            pixel_format=pixel_format,
        ))

    if {slot.slot for slot in slots} != {"cam0", "cam1"}:
        raise ArducamConfigError("camera slots must be exactly cam0 and cam1")
    slots.sort(key=lambda item: item.slot)
    if len({slot.id_path for slot in slots}) != 2:
        raise ArducamConfigError("camera id_path values must be unique")
    positions = [slot.position for slot in slots]
    if positions != ["unknown", "unknown"] and set(positions) != {"left", "right"}:
        raise ArducamConfigError(
            "camera positions must be both unknown or exactly one left and one right"
        )
    return slots[0], slots[1]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[CameraSlot, CameraSlot]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArducamConfigError(f"cannot read camera configuration {path}: {exc}") from exc
    return validate_config(raw)


def enumerate_capture_devices(context=None) -> list[VideoDevice]:
    """Return V4L2 nodes and the udev identity needed for strict resolution."""
    if context is None:
        import pyudev
        context = pyudev.Context()
    devices = []
    for device in context.list_devices(subsystem="video4linux"):
        capabilities = str(device.get("ID_V4L_CAPABILITIES") or "")
        devices.append(VideoDevice(
            device=str(device.device_node),
            id_path=str(device.get("ID_PATH") or ""),
            reported_serial=str(device.get("ID_SERIAL_SHORT") or ""),
            is_capture=":capture:" in capabilities,
        ))
    return sorted(devices, key=lambda item: item.device)


def _inventory(devices: Sequence[VideoDevice]) -> str:
    return ", ".join(
        f"{d.device} path={d.id_path or '<none>'} capture={d.is_capture} "
        f"serial={d.reported_serial or '<none>'}"
        for d in devices
    ) or "<no video devices>"


def resolve_slots(
    slots: Iterable[CameraSlot], devices: Sequence[VideoDevice] | None = None
) -> tuple[ResolvedCamera, ResolvedCamera]:
    """Resolve each logical slot to exactly one capture node by udev ID_PATH."""
    slots = tuple(slots)
    devices = tuple(enumerate_capture_devices() if devices is None else devices)
    resolved = []
    for slot in slots:
        matches = [d for d in devices if d.is_capture and d.id_path == slot.id_path]
        if not matches:
            raise ArducamConfigError(
                f"{slot.slot} path {slot.id_path!r} was not found; inventory: "
                f"{_inventory(devices)}"
            )
        if len(matches) > 1:
            raise ArducamConfigError(
                f"{slot.slot} path {slot.id_path!r} matched multiple capture nodes; "
                f"inventory: {_inventory(devices)}"
            )
        device = matches[0]
        resolved.append(ResolvedCamera(slot, device.device, device.reported_serial))
    if len({camera.device for camera in resolved}) != len(resolved):
        raise ArducamConfigError("configured slots resolved to the same capture device")
    return tuple(resolved)  # type: ignore[return-value]
