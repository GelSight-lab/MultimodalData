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
    id_path: str = ""
    position: str = "unknown"
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "MJPG"
    serial: str = ""          # preferred identity; id_path is the fallback
    # V4L2 buffer depth. NOT 1: with a single buffer the driver has nowhere to
    # put the next frame while the reader holds the current one, and drops it.
    # Measured on the generic USB wrist cameras — 17.8 fps at depth 1, 30.0 at
    # depth 2, against 30.02 fps from v4l2-ctl driving the same camera. The
    # extra frame of staleness it can cost is recorded, not assumed: every
    # frame carries the timestamp of its own grab.
    buffer_size: int = 2
    # V4L2 controls to apply on open, as (name, value) pairs. OpenCV can
    # set only a handful of properties; the ones that silently cost frames
    # live here. `exposure_dynamic_framerate=1` lets a camera trade frame
    # rate for exposure — measured at 7.6 fps in dim light on the generic
    # USB wrist cameras, against the 30 they advertise.
    controls: tuple = ()


@dataclass(frozen=True)
class VideoDevice:
    device: str
    id_path: str
    reported_serial: str
    is_capture: bool
    # USB identity. The RealSense cameras report a DIFFERENT serial at the USB
    # layer than librealsense does, so a serial cannot exclude them; the vendor
    # id can.
    vendor_id: str = ""
    model: str = ""


@dataclass(frozen=True)
class ResolvedCamera:
    config: CameraSlot
    device: str
    reported_serial: str
    device_id_path: str = ""

    @property
    def slot(self) -> str:
        return self.config.slot

    @property
    def serial(self) -> str:
        return self.config.serial or self.reported_serial

    @property
    def id_path(self) -> str:
        return self.device_id_path or self.config.id_path

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


def validate_config(raw: Mapping) -> tuple:
    """Validate and normalize the one- or two-slot JSON representation."""
    entries = raw.get("cameras") if isinstance(raw, Mapping) else None
    # One OR two. The pair is the normal rig, but a single wrist camera is a
    # real configuration — testing one before the mount exists, or carrying on
    # after one comes off — and every layer below handles the count it is
    # given rather than assuming two.
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise ArducamConfigError("configuration must contain one or two cameras")

    slots = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArducamConfigError("each camera must be an object")
        slot = str(entry.get("slot", ""))
        serial = str(entry.get("serial", "")).strip()
        id_path = str(entry.get("id_path", "")).strip()
        position = str(entry.get("position", "unknown")).lower()
        if not serial and not id_path:
            raise ArducamConfigError("each camera needs a nonempty serial or id_path")
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
            serial=serial,
            buffer_size=_positive_int(entry, "buffer_size", 2),
            controls=tuple((str(k), int(v)) for k, v in
                           (entry.get("controls") or {}).items()),
        ))

    expected = {"cam0"} if len(slots) == 1 else {"cam0", "cam1"}
    if {slot.slot for slot in slots} != expected:
        raise ArducamConfigError(
            "camera slots must be exactly "
            + ("cam0" if len(slots) == 1 else "cam0 and cam1"))
    slots.sort(key=lambda item: item.slot)
    serials = [s.serial for s in slots if s.serial]
    if len(serials) != len(set(serials)):
        raise ArducamConfigError("camera serial values must be unique")
    paths = [s.id_path for s in slots if s.id_path]
    if len(paths) != len(set(paths)):
        raise ArducamConfigError("camera id_path values must be unique")
    positions = [slot.position for slot in slots]
    # With two cameras the sides must be assigned together or not at all: one
    # named and one unknown is a half-finished mapping that reads as complete.
    # With one, any single value is a complete statement about it.
    ok = (len(slots) == 1 or positions == ["unknown", "unknown"]
          or set(positions) == {"left", "right"})
    if not ok:
        raise ArducamConfigError(
            "camera positions must be both unknown or exactly one left and one right"
        )
    return tuple(slots)


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
            vendor_id=str(device.get("ID_VENDOR_ID") or "").lower(),
            model=str(device.get("ID_MODEL") or ""),
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
) -> tuple:
    """Resolve each logical slot to exactly one capture node by udev serial
    (ID_SERIAL_SHORT), falling back to ID_PATH when the slot has no serial."""
    slots = tuple(slots)
    devices = tuple(enumerate_capture_devices() if devices is None else devices)
    resolved = []
    for slot in slots:
        if slot.serial:
            key, wanted = "serial", slot.serial
            matches = [d for d in devices if d.is_capture and d.reported_serial == wanted]
        else:
            key, wanted = "path", slot.id_path
            matches = [d for d in devices if d.is_capture and d.id_path == wanted]
        if not matches:
            raise ArducamConfigError(
                f"{slot.slot} {key} {wanted!r} was not found; inventory: {_inventory(devices)}")
        if len(matches) > 1:
            raise ArducamConfigError(
                f"{slot.slot} {key} {wanted!r} matched multiple capture nodes; "
                f"inventory: {_inventory(devices)}")
        device = matches[0]
        resolved.append(ResolvedCamera(slot, device.device, device.reported_serial,
                                       device.id_path))
    if len({camera.device for camera in resolved}) != len(resolved):
        raise ArducamConfigError("configured slots resolved to the same capture device")
    return tuple(resolved)  # type: ignore[return-value]


DEFAULT_WRIST_CONTROLS = {
    # Lets the camera trade frame rate for exposure. Measured at 7.6 fps in
    # room light on the generic USB pair, against the 30 they advertise.
    "exposure_dynamic_framerate": 0,
    # 50 Hz mains. 60 Hz anti-flicker over 50 Hz lighting beats at 10 Hz.
    "power_line_frequency": 1,
}


# Intel. Its cameras are capture nodes too, and their USB-layer serial differs
# from the one librealsense reports, so the vendor id is what excludes them.
REALSENSE_VENDOR_ID = "8086"


def _attached_realsense_serials() -> tuple:
    """Serials of the RealSense cameras librealsense can see, or () if the
    SDK is unavailable. They are capture nodes too and must not be mistaken
    for wrist cameras."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return ()
    return tuple(d.get_info(rs.camera_info.serial_number)
                 for d in rs.context().query_devices())


def register_wrist_cameras(path, *, devices=None, realsense_serials=None,
                           gelsight_serials=None, controls=None, allow_one=False):
    """Write the wrist-camera config from whatever is plugged in right now.

    A wrist camera is a capture node that is not a RealSense and not a
    GelSight. Identity is the USB PORT, never the serial: the generic pair
    both report 200901010001, so a serial would match both and resolving
    would refuse. The port changes whenever someone re-plugs, which is why
    this is a command rather than a file you edit.

    Refuses unless exactly two are left, naming what it found. Anything else
    is a rig that is not ready, and a config written from it would record one
    camera twice or the wrong camera once.
    """
    from twm.recorder.config import GELSIGHT_SERIALS

    path = Path(path)
    devices = tuple(enumerate_capture_devices() if devices is None else devices)
    rs_serials = set(_attached_realsense_serials() if realsense_serials is None
                     else realsense_serials)
    gel = set(GELSIGHT_SERIALS.values() if gelsight_serials is None else gelsight_serials)

    candidates, seen = [], set()
    for d in devices:
        if not d.is_capture:
            continue
        if d.vendor_id == REALSENSE_VENDOR_ID or d.reported_serial in rs_serials:
            continue
        if d.reported_serial in gel:
            continue
        if d.id_path in seen:          # a device's second capture node
            continue
        seen.add(d.id_path)
        candidates.append(d)
    candidates.sort(key=lambda d: d.id_path)

    want = "one or two" if allow_one else "exactly two"
    ok = (1 <= len(candidates) <= 2) if allow_one else (len(candidates) == 2)
    if not ok:
        found = ", ".join(f"{d.device} at {d.id_path}" for d in candidates) or "none"
        hint = ("" if allow_one else
                " Pass --allow-one to register a single camera on purpose.")
        raise ArducamConfigError(
            f"expected {want} wrist cameras, found {len(candidates)}: {found}. "
            f"Plug both in (and check they are not a RealSense or a GelSight), "
            f"then run this again. Nothing was written.{hint}")

    # Keyed by PORT, not by slot. Slots are handed out in port order, so
    # plugging in the second camera can move the first from cam0 to cam1 —
    # and a side carried across by slot would then name the new camera with
    # the old one's side.
    prior = {}
    try:
        for entry in json.loads(path.read_text()).get("cameras", []):
            prior[entry.get("id_path")] = entry
    except (OSError, json.JSONDecodeError):
        pass

    cams = []
    for slot, d in zip(("cam0", "cam1")[:len(candidates)], candidates):
        cams.append({
            "slot": slot,
            "id_path": d.id_path,
            # No serial: see the docstring. The one it reports is recorded in
            # the episode anyway, as `reported_serial`.
            "position": prior.get(d.id_path, {}).get("position", "unknown"),
            "width": 640, "height": 480, "fps": 30, "pixel_format": "MJPG",
            "controls": dict(DEFAULT_WRIST_CONTROLS if controls is None else controls),
        })
    # A side carried over for one camera and not the other is a half-finished
    # mapping that reads as complete, so a mixed result is cleared rather than
    # kept: adding the second camera means assigning the sides again.
    sides = [c["position"] for c in cams]
    if len(cams) == 2 and sides != ["unknown", "unknown"] and set(sides) != {"left", "right"}:
        for c in cams:
            c["position"] = "unknown"
    doc = {"_note": ("Written by `python -m twm.sensor_camera register`. Identity is "
                     "the USB port, not the serial: re-plugging into another port "
                     "means running that command again."),
           "cameras": cams}
    validate_config(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return cams


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
    from twm.recorder.frames import Tick
    from twm.recorder.schema import create_episode_file
    from twm.recorder.writer import EpisodeWriter, WriterOverloaded, queue_capacity_bytes

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
        writer = EpisodeWriter(
            capacity_bytes=queue_capacity_bytes(3.0, slots[0].fps,
                                                slots[0].width * slots[0].height * 3 * 2))
        deadline = time.monotonic() + duration
        tick_dt = 1.0 / slots[0].fps
        next_tick = time.monotonic()
        overloads = 0
        while time.monotonic() < deadline:
            samples = [stream.get_frame_with_timestamp(timeout=0.5)
                       for stream in streams]
            t = time.time()
            try:
                writer.submit(h5_file, Tick(
                    timestamp=t,
                    arducam=tuple(s[0] for s in samples),
                    arducam_ts=tuple(t if s[1] is None else float(s[1]) for s in samples)))
            except WriterOverloaded:
                overloads += 1
            next_tick += tick_dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        writer.drain()
        writer.stop()
        dropped_frames = overloads
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
                    tile, f"{camera.serial}  {camera.id_path}", (12, 55),
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


def _cmd_list() -> int:
    from twm.recorder.config import GELSIGHT_SERIALS
    rs = set(_attached_realsense_serials())
    gel = set(GELSIGHT_SERIALS.values())
    for d in sorted(enumerate_capture_devices(), key=lambda x: x.id_path):
        if not d.is_capture:
            continue
        kind = ("RealSense" if d.vendor_id == REALSENSE_VENDOR_ID
                or d.reported_serial in rs else
                "GelSight" if d.reported_serial in gel else "wrist candidate")
        print(f"  {d.device:14s} {d.id_path:34s} serial={d.reported_serial:14s} {kind}")
    return 0


def _cmd_register(out: str, allow_one: bool = False) -> int:
    import sys
    try:
        cams = register_wrist_cameras(out, allow_one=allow_one)
    except ArducamConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    for c in cams:
        print(f"  {c['slot']}: {c['id_path']}  position={c['position']}")
    print("Assign left/right once mounted: twm.sensor_camera identify")
    return 0


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
    register_parser = subparsers.add_parser(
        "register", help="write the wrist-camera config from what is plugged in")
    register_parser.add_argument("--out", default=str(DEFAULT_CONFIG_PATH.with_name("wrist_usb.json")))
    register_parser.add_argument("--allow-one", action="store_true",
                                 help="register a single camera; the pair is the default")
    subparsers.add_parser("list", help="show every capture device the machine sees")
    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    if args.command == "register":
        return _cmd_register(args.out, args.allow_one)
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
