"""CaptureLoop — grabs a Tick at a strict rate on its own thread.

The GUI thread reads `latest()` at whatever rate it manages; recording
cadence never depends on cv2. While recording, every accepted tick goes to
the writer; anything that would break the 30 Hz timeline (writer overload
or fault, low disk, a stall between ticks) becomes a StopRequest that the
controller turns into a finalized, explicitly-invalid episode.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from twm.recorder.frames import Tick
from twm.recorder.writer import WriterFault, WriterOverloaded, WriterStats

log = logging.getLogger("twm.recorder")


@dataclass(frozen=True)
class StopRequest:
    kind: str      # overload | writer_fault | disk_low | capture_stall
    detail: str


@dataclass(frozen=True)
class CaptureSnapshot:
    tick: Tick
    gs_ref: Tuple[np.ndarray, ...]
    ot_poses: Dict[str, Any]
    recording: bool
    frame_count: int
    elapsed: float
    fps_meas: float
    writer: WriterStats
    stop_request: Optional[StopRequest]
    fatal_error: Optional[str] = None
    sensors: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingResult:
    h5_file: Any
    frame_count: int
    max_gap_s: float
    gap_count: int
    stop_request: Optional[StopRequest]


class CaptureLoop:
    def __init__(self, rig, writer, fps: int = 30, warmup_drop_frames: int = 10,
                 max_tick_gap_s: float = 0.5, report_every: int = 60,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.rig = rig
        self.writer = writer
        self.tick_dt = 1.0 / fps
        self.warmup_drop_frames = warmup_drop_frames
        self.max_tick_gap_s = max_tick_gap_s
        self.report_every = report_every
        self._clock = clock
        self._sleep = sleep

        self._lock = threading.Lock()
        self._latest: Optional[CaptureSnapshot] = None
        self._gs_ref: Optional[Tuple[np.ndarray, ...]] = None
        self._reset_ref = False
        self._recording = False
        self._h5_file = None
        self._episode_id = 0
        self._frame_count = 0
        self._start_t = 0.0
        self._warmup_remaining = 0
        self._last_recorded_ts: Optional[float] = None
        self._gap_count = 0
        self._max_gap = 0.0
        self._stop_request: Optional[StopRequest] = None

        self._tick_times = collections.deque(maxlen=30)
        self._grab_s = 0.0
        self._ticks = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CaptureLoop", daemon=True)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                log.warning("capture thread did not stop within 2s; "
                           "it may be blocked in a sensor read")

    # ── GUI-thread API (all non-blocking) ────────────────────────────────────
    def latest(self) -> Optional[CaptureSnapshot]:
        with self._lock:
            return self._latest

    def request_reset_ref(self) -> None:
        with self._lock:
            self._reset_ref = True

    def start_recording(self, h5_file) -> None:
        self.writer.reset_episode_stats()
        with self._lock:
            self._episode_id += 1
            self._h5_file = h5_file
            self._recording = True
            self._frame_count = 0
            self._start_t = self._clock()
            self._warmup_remaining = self.warmup_drop_frames
            self._last_recorded_ts = None
            self._gap_count = 0
            self._max_gap = 0.0
            self._stop_request = None

    def stop_recording(self) -> Optional[RecordingResult]:
        """Stop feeding the writer and hand the open file back. Returns None
        if no episode is open. Does not wait for the writer. Safe to call at
        any moment (an operator keypress), not only after seeing a published
        stop request — the capture thread applies stop requests to shared
        state inside the same lock acquisition that reads/writes them, so
        there is no window where this can see a torn or stale episode."""
        with self._lock:
            if self._h5_file is None:
                return None
            result = RecordingResult(self._h5_file, self._frame_count, self._max_gap,
                                     self._gap_count, self._stop_request)
            self._recording = False
            self._h5_file = None
            self._stop_request = None
        return result

    # ── capture thread ───────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            t_start = self._clock()
            try:
                tick = self.rig.grab()
                self._grab_s += self._clock() - t_start

                # Reading `recording`/`h5`, recording the tick (gap check,
                # writer.submit, frame_count, writer.check), and applying any
                # resulting StopRequest all happen under one lock acquisition.
                # That closes the window a separate later lock reacquisition
                # used to leave open: a stop_recording()+start_recording()
                # landing between "compute the StopRequest" and "apply it"
                # could return episode N with stop_request=None and then
                # kill episode N+1 at frame 0 with episode N's request. The
                # episode id is still checked and kept as a second, explicit
                # guard against that class of bug.
                with self._lock:
                    episode_id = self._episode_id
                    recording, h5 = self._recording, self._h5_file
                    if recording and self._warmup_remaining > 0:
                        self._warmup_remaining -= 1
                        recording = False
                    stop_request = self._record(h5, tick) if recording else None
                    if stop_request is not None:
                        if episode_id == self._episode_id:
                            self._recording = False
                            self._stop_request = stop_request
                        else:
                            log.warning(
                                "discarding stale %s stop request from episode %d "
                                "(now on episode %d): %s", stop_request.kind,
                                episode_id, self._episode_id, stop_request.detail)
                            stop_request = None

                if stop_request is not None:
                    log.error("ending episode: %s — %s", stop_request.kind, stop_request.detail)

                self._tick_times.append(tick.timestamp)
                fps_meas = ((len(self._tick_times) - 1)
                            / (self._tick_times[-1] - self._tick_times[0])
                            if len(self._tick_times) >= 2
                            and self._tick_times[-1] > self._tick_times[0] else 0.0)
                ot_poses = self.rig.latest_poses()
                stats = self.writer.stats()
                sensor_status = getattr(self.rig, "sensor_status", None)
                sensors = sensor_status() if sensor_status is not None else {}

                with self._lock:
                    if self._gs_ref is None or self._reset_ref:
                        self._gs_ref = tuple(g.copy() for g in tick.gelsight)
                        self._reset_ref = False
                    self._latest = CaptureSnapshot(
                        tick=tick, gs_ref=self._gs_ref, ot_poses=ot_poses,
                        sensors=sensors,
                        recording=self._recording, frame_count=self._frame_count,
                        elapsed=(tick.timestamp - self._start_t) if self._recording else 0.0,
                        fps_meas=fps_meas, writer=stats, stop_request=self._stop_request)

                self._ticks += 1
                if self._ticks % self.report_every == 0:
                    log.info("[%s] fps=%.1f grab=%.1fms queue=%.0f%% write=%.0f MB/s",
                             "REC" if recording else "IDLE", fps_meas,
                             self._grab_s / self.report_every * 1e3,
                             stats.fraction * 100, stats.mean_mb_s)
                    self._grab_s = 0.0

                remaining = self.tick_dt - (self._clock() - t_start)
                if remaining > 0:
                    self._sleep(remaining)
            except BaseException as exc:
                # Nothing above this point may kill the thread silently: a
                # dead thread with `fatal_error` never set means the GUI
                # keeps showing `recording=True` with a frame_count that will
                # never advance again, with no signal anyone could act on.
                # This covers rig.grab() as well as rig.latest_poses(),
                # writer.stats(), and anything in between.
                self._publish_fatal(f"{type(exc).__name__}: {exc}")
                return

    def _record(self, h5, tick: Tick) -> Optional[StopRequest]:
        """Must be called with self._lock held: submit(), the frame counter,
        and the gap bookkeeping all have to be atomic with the recording
        state a concurrent stop_recording() reads, or a stop_recording() that
        lands mid-tick can return a frame_count that does not match what
        reached the writer for that file, or race a fresh episode's
        `_last_recorded_ts = None` into a spurious capture_stall."""
        if self._last_recorded_ts is not None:
            gap = tick.timestamp - self._last_recorded_ts
            self._max_gap = max(self._max_gap, gap)
            if gap > self.max_tick_gap_s:
                self._gap_count += 1
                return StopRequest("capture_stall",
                                   f"{gap:.2f}s between ticks (limit {self.max_tick_gap_s}s)")
        try:
            self.writer.submit(h5, tick)
        except WriterOverloaded as exc:
            return StopRequest("overload", str(exc))
        except WriterFault as exc:
            return StopRequest("writer_fault", str(exc))
        self._last_recorded_ts = tick.timestamp
        self._frame_count += 1
        health = self.writer.check()
        return StopRequest(*health) if health else None

    def _publish_fatal(self, message: str) -> None:
        log.error("capture stopped: %s", message)
        with self._lock:
            prev = self._latest
            # stats() may be the failing dependency. Fatal publication must
            # use only the last snapshot or an explicitly unavailable value.
            stats = prev.writer if prev else WriterStats(
                queue_items=0, queue_bytes=0, capacity_bytes=0,
                peak_fraction=0.0, bytes_written=0, write_seconds=0.0,
                last_batch_ms=0.0, flushes=0, file_mb_s=0.0,
                disk_free_gb=None, overloaded_since=None,
                fault="writer stats unavailable")
            self._latest = CaptureSnapshot(
                tick=prev.tick if prev else Tick(self._clock()),
                gs_ref=self._gs_ref or (), ot_poses=prev.ot_poses if prev else {},
                sensors=prev.sensors if prev else {},
                recording=self._recording, frame_count=self._frame_count,
                elapsed=(self._clock() - self._start_t) if self._recording else 0.0,
                fps_meas=0.0, writer=stats,
                stop_request=self._stop_request, fatal_error=message)
        self._stop.set()
