"""SensorRig — owns the hardware. Starts in order, stops in reverse, grabs Ticks.

Drivers are injected so the rig (and everything above it) is testable
without cameras. `default_drivers()` imports the real stream classes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from twm.recorder.config import OT_TRACKERS, RecorderConfig
from twm.recorder.frames import COLOR_SHAPE, Tick

log = logging.getLogger("twm.recorder")


class DummyGelSight:
    """Stands in for a GelSight that failed to start: black frames, no timestamp."""

    def __init__(self, side: str):
        self.side = side
        self._frame = np.zeros(COLOR_SHAPE, np.uint8)

    def start(self, **kwargs):
        pass

    def stop(self):
        pass

    def get_frame(self, **kwargs):
        return self._frame

    def get_frame_with_timestamp(self, **kwargs):
        return self._frame, None


def frame_with_timestamp(stream) -> Tuple[np.ndarray, Optional[float]]:
    """(frame, capture_ts) from any stream; ts is None if it has no clock."""
    fn = getattr(stream, "get_frame_with_timestamp", None)
    if fn is not None:
        return fn()
    return stream.get_frame(), None


@dataclass(frozen=True)
class Drivers:
    realsense: Callable[..., Any]          # (serial=, fps=) -> stream
    gelsight: Callable[..., Any]           # (serial=, resolution=, name=) -> stream
    optitrack: Callable[[], Any]
    arducam: Callable[..., Any]            # (config, device) -> stream
    resolve_arducams: Callable[[Optional[Path]], Sequence[Any]]
    sleep: Callable[[float], None] = time.sleep


def default_drivers() -> Drivers:
    from camera_stream.arducam_video_stream import ArducamVideoStream
    from camera_stream.realsense_stream import RealsenseStream
    from camera_stream.usb_video_stream import USBVideoStream
    from optitrack.optitrack_stream import OptitrackStream
    from twm.sensor_camera import load_config, resolve_slots

    def resolve(path: Optional[Path]):
        return resolve_slots(load_config(path) if path else load_config())

    return Drivers(realsense=RealsenseStream, gelsight=USBVideoStream,
                   optitrack=OptitrackStream, arducam=ArducamVideoStream,
                   resolve_arducams=resolve)


def _stop_all(started: List[Any]) -> None:
    """Best-effort reverse-order stop; one failure never skips the others."""
    while started:
        resource = started.pop()
        try:
            resource.stop()
        except Exception as exc:
            log.warning("could not stop %r: %s", resource, exc)


class SensorRig:
    def __init__(self, realsense: Sequence[Any], gelsight_left, gelsight_right,
                 optitrack, arducam: Sequence[Any] = (),
                 arducam_config: Sequence[Any] = (),
                 trackers: Sequence[str] = OT_TRACKERS,
                 clock: Callable[[], float] = time.time):
        self.realsense = list(realsense)
        self.gelsight_left = gelsight_left
        self.gelsight_right = gelsight_right
        self.optitrack = optitrack
        self.arducam = list(arducam)
        self.arducam_config = tuple(arducam_config)
        self.trackers = tuple(trackers)
        self._clock = clock
        self._started: List[Any] = [*self.realsense, *self.arducam,
                                    gelsight_left, gelsight_right, optitrack]

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def open(cls, config: RecorderConfig, drivers: Optional[Drivers] = None) -> "SensorRig":
        """Start every sensor. On any failure (including Ctrl-C) stop what
        was started, in reverse, and re-raise."""
        drivers = drivers or default_drivers()
        started: List[Any] = []

        def start(resource, **kwargs):
            started.append(resource)          # registered before start(): a
            resource.start(**kwargs)          # failing start() still gets stop()
            return resource

        try:
            realsense = []
            for serial in config.realsense_serials:
                log.info("starting RealSense %s", serial)
                realsense.append(start(drivers.realsense(serial=serial, fps=config.fps)))
                drivers.sleep(0.5)            # stagger: USB bandwidth contention

            arducam, arducam_config = [], ()
            if config.use_arducam:
                arducam_config = tuple(drivers.resolve_arducams(config.arducam_config_path))
                for cam in arducam_config:
                    log.info("starting Arducam %s at %s", cam.slot, cam.device)
                    arducam.append(start(drivers.arducam(cam.config, cam.device),
                                         timeout=config.startup_timeout_s))

            gelsight: Dict[str, Any] = {}
            for side, serial in config.gelsight_serials.items():
                stream = drivers.gelsight(serial=serial, resolution=(640, 480), name=side)
                try:
                    start(stream)
                except Exception as exc:
                    log.warning("GelSight %s (serial %s) unavailable: %s — "
                                "recording black frames for this side", side, serial, exc)
                    started.remove(stream)
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    stream = DummyGelSight(side)
                gelsight[side] = stream

            log.info("starting OptiTrack")
            optitrack = start(drivers.optitrack())
        except BaseException:
            _stop_all(started)
            raise

        rig = cls(realsense, gelsight["left"], gelsight["right"], optitrack,
                  arducam, arducam_config)
        rig._started = started
        return rig

    def wait_ready(self, timeout_s: float, settle_s: float,
                   sleep: Callable[[float], None] = time.sleep) -> None:
        """Block until every sensor has produced a frame, then pump frames
        for `settle_s` so auto-exposure converges before the reference grab.
        Closes the rig and re-raises on failure."""
        try:
            for s in self.realsense:
                s.get_color_frame(timeout=timeout_s)
            for s in self.arducam:
                s.get_frame_with_timestamp(timeout=timeout_s)
            self.gelsight_left.get_frame()
            self.gelsight_right.get_frame()
            end = self._clock() + settle_s
            while self._clock() < end:
                for s in self.realsense:
                    s.get_color_frame()
                for s in self.arducam:
                    s.get_frame()
                self.gelsight_left.get_frame()
                self.gelsight_right.get_frame()
                sleep(0.02)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        _stop_all(self._started)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── per-tick access ──────────────────────────────────────────────────────
    def grab(self) -> Tick:
        """Snapshot every sensor and drain the OptiTrack buffers into one Tick."""
        color = tuple(s.get_color_frame() for s in self.realsense)
        depth = tuple(s.get_depth_frame() for s in self.realsense)
        gs = [frame_with_timestamp(s) for s in (self.gelsight_left, self.gelsight_right)]
        ard = [s.get_frame_with_timestamp() for s in self.arducam]
        t = self._clock()
        flush = getattr(self.optitrack, "flush_buffer", None)
        ot = {name: flush(name) for name in self.trackers} if flush else {}
        return Tick(
            timestamp=t,
            color=color, depth=depth,
            gelsight=tuple(f for f, _ in gs),
            gelsight_ts=tuple(t if ts is None else float(ts) for _, ts in gs),
            arducam=tuple(f for f, _ in ard),
            arducam_ts=tuple(t if ts is None else float(ts) for _, ts in ard),
            optitrack=ot,
        )

    def latest_poses(self) -> Dict[str, Any]:
        return {name: self.optitrack.get_latest_pose(name) for name in self.trackers}

    def arducam_labels(self) -> List[str]:
        return [f"{c.slot} {getattr(c, 'serial', '') or c.id_path} {c.position}"
                for c in self.arducam_config]
