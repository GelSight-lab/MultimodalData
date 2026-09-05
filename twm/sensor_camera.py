"""Configuration and stable discovery for TWM sensor-mounted cameras."""

from __future__ import annotations

import json
import os
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


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


def save_position_mapping(path: str | Path, left_slot: str | None) -> None:
    """Atomically assign physical sides, or clear both sides to unknown."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArducamConfigError(f"cannot read camera configuration {path}: {exc}") from exc
    validate_config(raw)
    if left_slot is not None and left_slot not in {"cam0", "cam1"}:
        raise ArducamConfigError("left_slot must be cam0, cam1, or None")
    for entry in raw["cameras"]:
        if left_slot is None:
            entry["position"] = "unknown"
        else:
            entry["position"] = "left" if entry["slot"] == left_slot else "right"
    validate_config(raw)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2) + "\n")
    os.replace(temporary, path)


def _sample_stats(dataset) -> tuple[float, float]:
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    samples = dataset[indices]
    return float(np.mean(samples)), float(np.var(samples))


def verify_recording(path: str | Path, requested_fps: float = 30.0,
                     expected_duration: float | None = None) -> dict:
    """Reopen a sensor-camera recording and evaluate its data invariants."""
    import h5py

    path = Path(path)
    checks = {}
    cameras = {}
    try:
        with h5py.File(path, "r") as f:
            counts = []
            first_frames = []
            for slot in ("cam0", "cam1"):
                frame_key = f"arducam/{slot}/frames"
                timestamp_key = f"arducam/{slot}/timestamps"
                present = frame_key in f and timestamp_key in f
                checks[f"{slot}_datasets"] = present
                if not present:
                    continue
                frames = f[frame_key]
                timestamps = np.asarray(f[timestamp_key][:], dtype=np.float64)
                count = int(len(frames))
                counts.append(count)
                checks[f"{slot}_nonempty"] = count > 0
                checks[f"{slot}_shape"] = frames.shape[1:] == (480, 640, 3)
                checks[f"{slot}_timestamp_count"] = len(timestamps) == count
                finite = len(timestamps) > 0 and bool(np.all(np.isfinite(timestamps)))
                monotonic = finite and bool(np.all(np.diff(timestamps) >= 0))
                unique_ts = np.unique(timestamps) if finite else np.array([])
                span = float(unique_ts[-1] - unique_ts[0]) if len(unique_ts) > 1 else 0.0
                cadence = ((len(unique_ts) - 1) / span if span > 0 else 0.0)
                checks[f"{slot}_timestamps_finite"] = finite
                checks[f"{slot}_timestamps_monotonic"] = monotonic
                checks[f"{slot}_timestamp_span"] = span > 0
                checks[f"{slot}_cadence"] = (
                    requested_fps * 0.85 <= cadence <= requested_fps * 1.15
                )
                if expected_duration is not None:
                    allowed_edge_loss = 2.0 / requested_fps
                    checks[f"{slot}_duration_coverage"] = (
                        span >= max(0.0, expected_duration - allowed_edge_loss)
                    )
                if count:
                    mean, variance = _sample_stats(frames)
                    first_frames.append(np.asarray(frames[0]))
                    hashes = [
                        zlib.crc32(np.asarray(frames[i]).tobytes())
                        for i in range(count)
                    ]
                    changed = sum(a != b for a, b in zip(hashes, hashes[1:]))
                    temporal_change_ratio = changed / max(1, count - 1)
                else:
                    mean, variance = 0.0, 0.0
                    temporal_change_ratio = 0.0
                checks[f"{slot}_nonblack"] = mean > 0.0 and variance > 0.0
                checks[f"{slot}_temporal_change"] = temporal_change_ratio >= 0.8
                cameras[slot] = {
                    "count": count,
                    "shape": list(frames.shape),
                    "dtype": str(frames.dtype),
                    "timestamp_first": float(timestamps[0]) if len(timestamps) else None,
                    "timestamp_last": float(timestamps[-1]) if len(timestamps) else None,
                    "timestamp_span_s": span,
                    "distinct_capture_timestamps": int(len(unique_ts)),
                    "capture_fps": cadence,
                    "sample_mean": mean,
                    "sample_variance": variance,
                    "temporal_change_ratio": temporal_change_ratio,
                    "attributes": {
                        key: (value.item() if hasattr(value, "item") else value)
                        for key, value in frames.parent.attrs.items()
                    },
                }
            checks["equal_frame_counts"] = len(counts) == 2 and counts[0] == counts[1]
            checks["streams_distinct"] = (
                len(first_frames) == 2
                and not np.array_equal(first_frames[0], first_frames[1])
            )
    except Exception as exc:  # diagnostic boundary: report instead of hiding output
        checks["file_readable"] = False
        return {
            "ok": False,
            "path": str(path),
            "frame_count": 0,
            "checks": checks,
            "cameras": cameras,
            "error": str(exc),
        }
    checks["file_readable"] = True
    return {
        "ok": all(checks.values()),
        "path": str(path),
        "frame_count": counts[0] if len(counts) == 2 and counts[0] == counts[1] else 0,
        "checks": checks,
        "cameras": cameras,
    }


def record_verification(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = "/tmp/twm_arducam_verification.h5",
    duration: float = 5.0,
    *,
    devices: Sequence[VideoDevice] | None = None,
    stream_factory=None,
    overwrite: bool = False,
) -> dict:
    """Record both cameras through the production writer, then validate HDF5."""
    from camera_stream.arducam_video_stream import ArducamVideoStream
    from twm.data_collection import HDF5Writer, create_episode_file

    if duration <= 0:
        raise ValueError("duration must be positive")
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slots = load_config(config_path)
    resolved = resolve_slots(slots, devices)
    factory = stream_factory or ArducamVideoStream
    streams = [factory(camera.config, camera.device) for camera in resolved]
    writer = None
    h5_file = None
    dropped_frames = 0
    working_dir = tempfile.TemporaryDirectory(
        dir=output_path.parent, prefix=".twm_arducam_"
    )
    created_path = None
    try:
        for stream in streams:
            stream.start(timeout=5.0)
        h5_file, created_path = create_episode_file(
            working_dir.name,
            0,
            [],
            [],
            slots[0].fps,
            task_name="arducam_verification",
            arducam_config=resolved,
            include_legacy=False,
        )
        writer = HDF5Writer()
        deadline = time.monotonic() + duration
        tick_dt = 1.0 / slots[0].fps
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            samples = [stream.get_frame_with_timestamp(timeout=0.5)
                       for stream in streams]
            writer.enqueue(
                h5_file, None, None, None, time.time(),
                arducam_frames=[sample[0] for sample in samples],
                arducam_timestamps=[sample[1] for sample in samples],
            )
            next_tick += tick_dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        writer.stop()
        dropped_frames = writer.dropped_frames
        writer = None
        h5_file.flush()
        h5_file.close()
        h5_file = None
        os.replace(created_path, output_path)
    finally:
        if writer is not None:
            writer.stop()
        if h5_file is not None:
            h5_file.close()
        for stream in streams:
            stream.stop()
        working_dir.cleanup()
    report = verify_recording(
        output_path,
        requested_fps=slots[0].fps,
        expected_duration=duration,
    )
    report["dropped_frames"] = dropped_frames
    report["checks"]["no_writer_drops"] = dropped_frames == 0
    report["ok"] = all(report["checks"].values())
    return report


def identify(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """Display both feeds and optionally persist a physical left/right mapping."""
    import cv2
    import numpy as np
    from camera_stream.arducam_video_stream import ArducamVideoStream

    config_path = Path(config_path)
    slots = load_config(config_path)
    resolved = resolve_slots(slots)
    streams = [ArducamVideoStream(camera.config, camera.device) for camera in resolved]
    left_slot = next((slot.slot for slot in slots if slot.position == "left"), None)
    try:
        for stream in streams:
            stream.start(timeout=5.0)
        while True:
            frames = [stream.get_frame() for stream in streams]
            tiles = []
            for camera, frame in zip(resolved, frames):
                tile = cv2.resize(frame, (640, 480))
                position = ("left" if camera.slot == left_slot else
                            "right" if left_slot is not None else "unknown")
                cv2.putText(
                    tile, f"{camera.slot}  {position}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
                cv2.putText(
                    tile, camera.id_path, (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
                )
                tiles.append(tile)
            panel = np.hstack(tiles)
            cv2.putText(
                panel, "0/1 = choose LEFT  u = unknown  s = save  q = quit",
                (12, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2,
            )
            cv2.imshow("TWM sensor-camera identification", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("0"), ord("1")):
                left_slot = f"cam{chr(key)}"
            elif key == ord("u"):
                left_slot = None
            elif key == ord("s"):
                save_position_mapping(config_path, left_slot)
                return
            elif key == ord("q"):
                return
    finally:
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    identify_parser = subparsers.add_parser("identify", help="preview and assign sides")
    identify_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    verify_parser = subparsers.add_parser("verify", help="record and validate both cameras")
    verify_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    verify_parser.add_argument("--duration", type=float, default=5.0)
    verify_parser.add_argument("--output", default="/tmp/twm_arducam_verification.h5")
    verify_parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "identify":
        identify(args.config)
        return 0
    report = record_verification(
        config_path=args.config,
        output_path=args.output,
        duration=args.duration,
        overwrite=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
