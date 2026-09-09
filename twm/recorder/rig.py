"""SensorRig — owns the hardware. Starts in order, stops in reverse, grabs Ticks.

Drivers are injected so the rig (and everything above it) is testable
without cameras. `default_drivers()` imports the real stream classes.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from twm.recorder.config import OT_TRACKERS, RecorderConfig, realsense_position
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


class DummyOptitrack:
    """Stands in for OptitrackStream when recording without ROS: no poses, empty buffers."""

    def start(self):
        pass

    def stop(self):
        pass

    def get_latest_pose(self, name):
        return None

    def flush_buffer(self, name):
        return []

    def peek_frame_with_timestamp(self):
        return self._frame, None


def frame_with_timestamp(stream) -> Tuple[np.ndarray, Optional[float]]:
    """(frame, capture_ts) from any stream; ts is None if it has no clock."""
    fn = getattr(stream, "get_frame_with_timestamp", None)
    if fn is not None:
        return fn()
    return stream.get_frame(), None


@dataclass(frozen=True)
class Drivers:
    realsense: Callable[..., Any]          # (serial=, fps=, align=) -> stream
    gelsight: Callable[..., Any]           # (serial=, resolution=, name=) -> stream
    optitrack: Callable[[], Any]
    arducam: Callable[..., Any]            # (config, device, encoding=) -> stream
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


STALL_AFTER_S = 1.0        # a stream whose capture clock is older than this is restarted
SUPERVISOR_POLL_S = 0.25


def restart_stream(stream) -> None:
    """Restart a video stream the way its driver expects (blocking)."""
    restart = getattr(stream, "restart", None)
    if restart is not None:
        restart()
        return
    stream.stop()
    stream.start()


class SensorSupervisor:
    """Watches the capture clocks of the USB video streams and restarts a
    stalled one on its own thread, so the capture thread never blocks on a
    sensor. Streams without a capture clock (dummies) are not supervised.
    """

    def __init__(self, streams: Dict[str, Any], stall_after_s: float = STALL_AFTER_S,
                 poll_s: float = SUPERVISOR_POLL_S,
                 clock: Callable[[], float] = time.time):
        self._streams = dict(streams)
        self.stall_after_s = stall_after_s
        self.poll_s = poll_s
        self._clock = clock
        self._lock = threading.Lock()
        self._restarts: Dict[str, int] = {n: 0 for n in self._streams}
        self._restarting: set = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="SensorSupervisor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15.0)     # a driver restart sleeps ~3 s

    def alive(self) -> bool:
        return self._thread.is_alive()

    @staticmethod
    def _peek_ts(stream) -> Optional[float]:
        peek = getattr(stream, "peek_frame_with_timestamp", None)
        return None if peek is None else peek()[1]

    def _run(self) -> None:
        while not self._stop.is_set():
            now = self._clock()
            for name, stream in self._streams.items():
                ts = self._peek_ts(stream)
                if ts is None or now - ts <= self.stall_after_s:
                    continue
                with self._lock:
                    if name in self._restarting:
                        continue
                    self._restarting.add(name)
                log.warning("%s: no frame for %.1fs — restarting the stream", name, now - ts)
                try:
                    restart_stream(stream)
                except Exception as exc:
                    log.error("%s: restart failed: %s", name, exc)
                finally:
                    with self._lock:
                        self._restarts[name] += 1
                        self._restarting.discard(name)
            self._stop.wait(self.poll_s)

    def status(self) -> Dict[str, Dict[str, Any]]:
        now = self._clock()
        out = {}
        with self._lock:
            for name, stream in self._streams.items():
                ts = self._peek_ts(stream)
                out[name] = {"stale_s": (now - ts) if ts is not None else None,
                             "restarting": name in self._restarting,
                             "restarts": self._restarts[name]}
        return out


class SensorRig:
    def __init__(self, realsense: Sequence[Any], gelsight_left, gelsight_right,
                 optitrack, arducam: Sequence[Any] = (),
                 arducam_config: Sequence[Any] = (),
                 trackers: Sequence[str] = OT_TRACKERS,
                 clock: Callable[[], float] = time.time, arducam_encoding: str = "bgr8"):
        self.realsense = list(realsense)
        self.gelsight_left = gelsight_left
        self.gelsight_right = gelsight_right
        self.optitrack = optitrack
        self.arducam = list(arducam)
        self.arducam_config = tuple(arducam_config)
        # How the wrist frames in each Tick are encoded; the episode file
        # records it so a reader never has to guess from dataset shape.
        self._arducam_encoding = arducam_encoding
        self.trackers = tuple(trackers)
        self._clock = clock
        self._started: List[Any] = [*self.realsense, *self.arducam,
                                    gelsight_left, gelsight_right, optitrack]
        self._last: Dict[str, Tuple[np.ndarray, Optional[float]]] = {}
        self._supervisor: Optional[SensorSupervisor] = None

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
            for i, serial in enumerate(config.realsense_serials):
                log.info("starting RealSense cam%d %s (%s)%s", i, serial,
                         realsense_position(serial),
                         "" if config.align_depth else " [raw depth]")
                realsense.append(start(drivers.realsense(serial=serial, fps=config.fps,
                                                        align=config.align_depth)))
                drivers.sleep(0.5)            # stagger: USB bandwidth contention

            arducam, arducam_config = [], ()
            if config.use_arducam:
                arducam_config = tuple(drivers.resolve_arducams(config.arducam_config_path))
                for cam in arducam_config:
                    log.info("starting Arducam %s%s (%s) at %s", cam.slot,
                             f" {cam.serial}" if getattr(cam, "serial", "") else "",
                             cam.position, cam.device)
                    arducam.append(start(drivers.arducam(cam.config, cam.device,
                                                        encoding=config.arducam_encoding),
                                         timeout=config.startup_timeout_s))

            gelsight: Dict[str, Any] = {}
            for side, serial in config.gelsight_serials.items():
                log.info("starting GelSight %s %s", side, serial)
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

            if config.use_optitrack:
                log.info("starting OptiTrack")
                optitrack = start(drivers.optitrack())
            else:
                log.info("OptiTrack disabled")
                optitrack = DummyOptitrack()
        except BaseException:
            _stop_all(started)
            raise

        rig = cls(realsense, gelsight["left"], gelsight["right"], optitrack,
                  arducam, arducam_config,
                  arducam_encoding=config.arducam_encoding)
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
            for i, s in enumerate(self.arducam):
                frame, _ = s.get_frame_with_timestamp(timeout=timeout_s)
                self._check_arducam_encoding(i, frame)
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
        self.start_supervisor()

    # ── stalled-stream supervision ───────────────────────────────────────────
    def supervised_streams(self) -> Dict[str, Any]:
        streams = {"gelsight_left": self.gelsight_left, "gelsight_right": self.gelsight_right}
        for i, stream in enumerate(self.arducam):
            streams[f"arducam_cam{i}"] = stream
        return streams

    def start_supervisor(self, stall_after_s: float = STALL_AFTER_S,
                         poll_s: float = SUPERVISOR_POLL_S) -> None:
        if self._supervisor is not None:
            return
        self._supervisor = SensorSupervisor(self.supervised_streams(), stall_after_s, poll_s,
                                            clock=self._clock)
        self._supervisor.start()

    def supervisor_alive(self) -> bool:
        return self._supervisor is not None and self._supervisor.alive()

    def sensor_status(self) -> Dict[str, Dict[str, Any]]:
        if self._supervisor is not None:
            return self._supervisor.status()
        return {name: {"stale_s": None, "restarting": False, "restarts": 0}
                for name in self.supervised_streams()}

    def close(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
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
        gs = [self._peek_or_last(name, stream) for name, stream in
              (("gelsight_left", self.gelsight_left), ("gelsight_right", self.gelsight_right))]
        ard = [self._peek_or_last(f"arducam_cam{i}", stream) for i, stream in enumerate(self.arducam)]
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

    def _peek_or_last(self, name: str, stream) -> Tuple[np.ndarray, Optional[float]]:
        """Latest frame and its capture time without waiting. During a stall
        (or a restart) the last good frame is returned with its OLD capture
        time, so the file stays honest and the tick never blocks."""
        peek = getattr(stream, "peek_frame_with_timestamp", None)
        frame, ts = peek() if peek is not None else frame_with_timestamp(stream)
        if frame is None:
            last = self._last.get(name)
            if last is None:
                raise RuntimeError(f"{name}: no frame available yet")
            return last
        self._last[name] = (frame, ts)
        return frame, ts

    def latest_poses(self) -> Dict[str, Any]:
        return {name: self.optitrack.get_latest_pose(name) for name in self.trackers}

    def _check_arducam_encoding(self, index: int, frame) -> None:
        """The frames must be shaped the way the episode file says they are.

        The encoding is declared once, in the config, and reaches both the
        driver and the file header. A driver that ignores it writes raw pixels
        into a dataset labelled MJPEG, and nothing downstream can read them —
        so compare what actually arrived, once, at startup.
        """
        want = self._arducam_encoding
        got = "bgr8" if getattr(frame, "ndim", 0) == 3 else "mjpeg"
        if got != want:
            slot = (self.arducam_config[index].slot
                    if index < len(self.arducam_config) else f"#{index}")
            raise RuntimeError(
                f"arducam {slot}: the recorder is configured for {want} but the "
                f"driver produced {got} (shape {getattr(frame, 'shape', '?')}). "
                f"An episode would claim an encoding its frames do not have.")

    @property
    def arducam_encoding(self) -> str:
        return self._arducam_encoding

    def arducam_labels(self) -> List[str]:
        return [f"{c.slot} {getattr(c, 'serial', '') or c.id_path} {c.position}"
                for c in self.arducam_config]
