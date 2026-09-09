"""Recorder — the episode state machine — and the cv2 operator loop.

Recorder has no GUI dependency; run() wires hardware, writer, capture and
the preview window together. Keys: s start, e end, r reset GelSight
reference, p toggle projection overlay, q quit.
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from twm.recorder.capture import CaptureLoop, CaptureSnapshot
from twm.recorder.config import RecorderConfig, parse_args
from twm.recorder.episode import VALID_ENDINGS, EpisodeStore, EpisodeSummary
from twm.recorder.frames import full_rig_tick_nbytes
from twm.recorder.monitor import health_line
from twm.recorder.preflight import (CheckResult, check_capture_alive, failures,
                                    format_report, run_episode_preflight,
                                    run_startup_preflight, stale_trackers)
from twm.recorder.rig import Drivers, SensorRig
from twm.recorder.schema import count_optitrack_samples, write_episode_attrs
from twm.recorder.writer import EpisodeWriter, WriterFault, queue_capacity_bytes

log = logging.getLogger("twm.recorder")


@dataclass
class Session:
    writer: EpisodeWriter
    capture: CaptureLoop
    store: EpisodeStore
    recorder: Recorder


def build_session(config: RecorderConfig, rig) -> Session:
    """The production wiring shared by the GUI and the headless soak.

    If anything after the writer is built raises, stop the writer before
    re-raising — the caller owns closing the rig itself."""
    tick_bytes = full_rig_tick_nbytes(len(rig.realsense), 2, len(rig.arducam))
    writer = EpisodeWriter(
        capacity_bytes=queue_capacity_bytes(config.writer.queue_seconds, config.fps, tick_bytes),
        batch_size=config.writer.batch_size, flush_interval_s=config.writer.flush_interval_s,
        overload_fraction=config.writer.overload_fraction,
        overload_sustained_s=config.writer.overload_sustained_s,
        min_free_gb=config.disk.min_free_gb)
    try:
        capture = CaptureLoop(rig, writer, fps=config.fps,
                              warmup_drop_frames=config.warmup_drop_frames,
                              max_tick_gap_s=config.writer.max_tick_gap_s)
        store = EpisodeStore(config.data_dir, config.task)
        return Session(writer, capture, store, Recorder(config, rig, writer, capture, store))
    except BaseException:
        writer.stop()
        raise


@dataclass
class OpenEpisode:
    num: int
    path: Path
    h5: Any


class Recorder:
    """Owns the open episode. Every transition happens on the caller's thread."""

    def __init__(self, config: RecorderConfig, rig, writer: EpisodeWriter,
                 capture: CaptureLoop, store: EpisodeStore,
                 clock: Callable[[], float] = time.time,
                 disk_usage: Callable[[str], Any] = shutil.disk_usage):
        self.config = config
        self.rig = rig
        self.writer = writer
        self.capture = capture
        self.store = store
        self._clock = clock
        self._disk_usage = disk_usage
        self._open: Optional[OpenEpisode] = None
        self.last_summary: Optional[EpisodeSummary] = None
        self._closed = False

    def _sensor_restart_counts(self) -> Dict[str, int]:
        status = getattr(self.rig, "sensor_status", None)
        if status is None:
            return {}
        return {name: int(st.get("restarts", 0)) for name, st in status().items()}

    def _sensor_restarts_since_start(self) -> Dict[str, int]:
        start = getattr(self, "_restarts_at_start", {})
        return {name: count - start.get(name, 0)
                for name, count in self._sensor_restart_counts().items()}

    @property
    def recording(self) -> bool:
        return self._open is not None

    def start_episode(self) -> List[CheckResult]:
        """Run the episode preflight and start recording. Returns the failed
        checks; an empty list means recording started."""
        if self._open is not None:
            return []
        now = self._clock()
        poses = self.rig.latest_poses()
        checks = list(run_episode_preflight(self.config, poses, self.writer.stats(),
                                            now, self._disk_usage))
        checks.append(check_capture_alive(self.capture.latest(),
                                          self.config.writer.max_tick_gap_s * 4, now))
        failed = failures(checks)
        if failed:
            log.error("cannot start episode:\n%s", format_report(failed))
            return failed
        num = self.store.next_episode_number()
        h5, path = self.store.create(num, self.config.realsense_serials,
                                     list(self.config.gelsight_serials.values()),
                                     self.config.fps,
                                     arducam_config=self.rig.arducam_config or None,
                                     n_realsense=len(self.config.realsense_serials),
                                     depth_aligned=self.config.align_depth,
                                     arducam_encoding=self.rig.arducam_encoding)
        self._open = OpenEpisode(num, path, h5)
        self._restarts_at_start = self._sensor_restart_counts()
        self.capture.start_recording(h5)
        log.info("recording episode %03d → %s", num, path)
        return []

    def end_episode(self, ended_by: str = "operator", reason: str = "") -> Optional[EpisodeSummary]:
        """Stop feeding, drain the writer, stamp validity, close, log."""
        if self._open is None:
            return None
        ep, self._open = self._open, None
        result = self.capture.stop_recording()
        frame_count = result.frame_count if result else 0
        if result and result.stop_request and ended_by in VALID_ENDINGS:
            ended_by, reason = result.stop_request.kind, result.stop_request.detail
        try:
            self.writer.drain()
        except WriterFault as exc:
            if ended_by in VALID_ENDINGS:
                ended_by, reason = "writer_fault", str(exc)
        stats = self.writer.stats()
        has_ot = False
        try:
            has_ot = count_optitrack_samples(ep.h5) > 0
        except Exception:
            pass
        summary = EpisodeSummary(
            episode_num=ep.num, path=ep.path, task=self.config.task,
            frame_count=frame_count, fps=self.config.fps,
            valid=ended_by in VALID_ENDINGS, ended_by=ended_by, reason=reason,
            max_tick_gap_s=result.max_gap_s if result else 0.0,
            gap_count=result.gap_count if result else 0,
            queue_peak_fraction=stats.peak_fraction,
            writer_mean_mb_s=stats.mean_mb_s, has_optitrack=has_ot,
            sensor_restarts=self._sensor_restarts_since_start())
        try:
            write_episode_attrs(ep.h5, summary.attrs())
        except Exception as exc:
            log.warning("could not write episode attrs: %s", exc)
        finally:
            try:
                ep.h5.close()
            except Exception as exc:
                log.warning("could not close %s cleanly: %s", ep.path, exc)
        self.store.log(summary)
        (log.error if not summary.valid else log.info)(summary.describe())
        self.last_summary = summary
        return summary

    def poll(self, snapshot: Optional[CaptureSnapshot]) -> Optional[EpisodeSummary]:
        """Call once per GUI tick. Ends the episode on capture events."""
        if snapshot is None or self._open is None:
            return None
        if snapshot.fatal_error:
            return self.end_episode("sensor_error", snapshot.fatal_error)
        if snapshot.stop_request:
            return self.end_episode(snapshot.stop_request.kind, snapshot.stop_request.detail)
        age = self._clock() - snapshot.tick.timestamp
        if age > self.config.writer.max_tick_gap_s * 4:
            return self.end_episode(
                "capture_stall", f"no tick for {age:.1f}s (capture thread not producing)")
        stale = stale_trackers(snapshot.ot_poses, self.config.active_sensors,
                               self.config.ot_watchdog_timeout_s, self._clock())
        if stale:
            return self.end_episode("watchdog", "OptiTrack " + ", ".join(stale))
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.end_episode("quit")
        except Exception:
            log.exception("could not finalize the open episode")
        self.capture.stop()
        self.writer.stop()
        self.rig.close()


# ── projection overlay (preview only) ────────────────────────────────────────

def load_projection(config: RecorderConfig) -> Optional[Dict[str, Any]]:
    """GelSight→camera overlay calibration, or None if unavailable."""
    if not config.show_projection:
        return None
    from twm.viz import CAM_CALIB_NAME, load_calibrations
    # The LIVE recorder always uses the current calibration; epochs only exist
    # for replaying the past. Resolved through calib_epoch rather than named
    # here, so renaming an epoch directory cannot leave this pointing at a path
    # that no longer exists -- which is what happened when the epoch
    # directories were renamed under a hard-coded path here, and the overlay
    # turned itself off with one warning line.
    from twm.calib_epoch import current_epoch
    epoch = "?"
    try:
        epoch, calib_dir = current_epoch()
        cam_calibs, gel_left, gel_right = load_calibrations(
            [calib_dir / CAM_CALIB_NAME[i] for i in range(3)],
            calib_dir / "T_gel_to_rigid_left.json",
            calib_dir / "T_gel_to_rigid_right.json")
    except Exception as exc:
        # ERROR, not warning: the overlay is how the operator confirms the rig
        # is calibrated before recording, so losing it silently costs a whole
        # session. Still not fatal -- nothing recorded depends on it.
        log.error("projection overlay disabled -- epoch %s: %s", epoch, exc)
        return None  # fallback-ok: the overlay is preview-only cosmetics; no calibration means no overlay, and nothing recorded depends on it
    cams = []
    for calib in cam_calibs:
        serial = calib["camera_serial"]
        if serial not in config.realsense_serials:
            log.warning("projection: camera %s not in realsense_serials, skipped", serial)
            continue
        cams.append({"index": config.realsense_serials.index(serial),
                     "T_mocap_to_cam": calib["T_mocap_to_cam"],
                     "intrinsics": calib["intrinsics"]})
    log.info("projection overlay: epoch %s, %d cameras calibrated (press p to toggle)",
             epoch, len(cams))
    return {"cams": cams, "gel_left": gel_left, "gel_right": gel_right} if cams else None


def _startup_preflight_ok(config: RecorderConfig, n_arducam: int) -> bool:
    """Run the startup checks. Low free disk refuses to start; the
    write-bandwidth self-test is advisory: it runs against the page cache
    and under-reads a slow HDD (54 ticks/s was measured minutes after a
    10-minute full-rig episode passed with a 24 % queue peak), and a real
    shortfall is caught at runtime by the writer's fail-fast anyway."""
    results = run_startup_preflight(config, n_arducam=n_arducam)
    log.info("startup preflight:\n%s", format_report(results))
    blocking = [r for r in failures(results) if r.name != "write_bandwidth"]
    advisory = [r for r in failures(results) if r.name == "write_bandwidth"]
    for r in advisory:
        log.warning("write_bandwidth below the %.2gx margin (%s); recording anyway — "
                    "watch the writer queue in the health line", config.disk.min_bandwidth_margin, r.detail)
    if blocking:
        log.error("startup preflight failed; fix the above and retry")
        return False
    return True


# ── operator loop ────────────────────────────────────────────────────────────

def run(config: RecorderConfig, drivers: Optional[Drivers] = None) -> int:
    n_arducam = 2 if config.use_arducam else 0
    log.info("task %s → %s", config.task, config.data_dir / config.task)
    if not _startup_preflight_ok(config, n_arducam):
        return 2

    rig = SensorRig.open(config, drivers)
    restore_logging()
    rig.wait_ready(config.startup_timeout_s, config.settle_s)
    log.info("all sensors ready")

    try:
        # Nothing built here has an owner yet (Recorder.close() below is
        # what usually stops the writer and closes the rig); if build_session
        # fails partway through, it stops any writer it managed to build,
        # and we close the rig ourselves before propagating.
        session = build_session(config, rig)
    except BaseException:
        rig.close()
        raise
    recorder, capture = session.recorder, session.capture
    try:
        capture.start()
        return _gui_loop(config, recorder, capture, rig, load_projection(config))
    finally:
        recorder.close()


class PreviewRenderer:
    """Two-rate preview: the base panel (thumbnails of the tick's frames,
    ~5 ms) is rebuilt only when a new tick arrives, at most `preview_fps`
    times a second; the projection overlay is redrawn on EVERY render onto a
    copy of that base, with the OptiTrack pose fetched *now* rather than the
    one sampled at the tick. The dot therefore tracks the sensor with the
    GUI's own latency (one 30 Hz frame) instead of tick + preview latency.

    `build(snap) -> panel`, `overlay(panel, ot_poses)`, `poses_now() -> dict`.
    The returned panel is always a fresh copy: callers may draw on it.
    """

    def __init__(self, build, overlay, poses_now, preview_fps: float,
                 clock: Callable[[], float] = time.time):
        self._build, self._overlay, self._poses_now = build, overlay, poses_now
        self._preview_dt = 1.0 / float(preview_fps)
        self._clock = clock
        self._base = None
        self._base_tick_ts = None
        self._base_t = -1e9

    def render(self, snap, show_overlay: bool):
        now = self._clock()
        tick_ts = snap.tick.timestamp
        if (self._base is None
                or (tick_ts != self._base_tick_ts and now - self._base_t >= self._preview_dt)):
            self._base = self._build(snap)
            self._base_tick_ts, self._base_t = tick_ts, now
        panel = self._base.copy()
        if show_overlay:
            self._overlay(panel, self._poses_now())
        return panel


def _gui_loop(config, recorder: Recorder, capture: CaptureLoop, rig,
              projection: Optional[Dict[str, Any]]) -> int:
    import cv2
    from twm.recorder.frames import decode_arducam
    from twm.viz import build_preview_panel, draw_projection_overlay

    log.info("controls: s start | e end | r reset diff ref | p projection | q quit")
    arducam_labels = rig.arducam_labels() or None
    show_projection = projection is not None
    gui_dt = 1.0 / float(getattr(config, 'gui_fps', 15))

    def draw_health(panel, snap):
        text, level = health_line(snap.writer, config.writer.warn_fraction,
                                  config.disk.min_free_gb, sensors=snap.sensors)
        color = {"ok": (80, 200, 80), "warn": (0, 200, 255), "fail": (0, 0, 255)}[level]
        cv2.putText(panel, text, (10, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)          # second line of the status strip

    def build(snap):
        tick = snap.tick
        return build_preview_panel(
            list(tick.color), list(tick.gelsight), list(snap.gs_ref), snap.ot_poses,
            snap.recording, snap.frame_count, snap.elapsed,
            snap.writer.queue_items, snap.fps_meas, task_name=config.task,
            # Decoded here, at the preview's own rate, rather than on the
            # 30 Hz capture thread: the wrist frames are stored as the camera
            # sent them and only the operator's screen needs pixels.
            arducam_frames=[decode_arducam(a) for a in tick.arducam] or None,
            arducam_labels=arducam_labels)

    def overlay(panel, ot_poses):
        draw_projection_overlay(panel, ot_poses, projection["cams"],
                                projection["gel_left"], projection["gel_right"])

    latest_snap = {"snap": None}

    def poses_now():
        try:
            return rig.latest_poses()
        except Exception as exc:  # fallback-ok: preview only; the tick's pose is at most one tick old
            log.debug("latest_poses failed (%s); using the tick's pose", exc)
            return latest_snap["snap"].ot_poses

    renderer = PreviewRenderer(build, overlay, poses_now, config.preview_fps)

    while True:
        t0 = time.time()
        snap = capture.latest()
        if snap is None:
            time.sleep(0.005)
            continue
        recorder.poll(snap)
        if snap.fatal_error:
            log.error("capture stopped: %s", snap.fatal_error)
            return 1

        latest_snap["snap"] = snap
        panel = renderer.render(snap, show_overlay=bool(show_projection and projection))
        draw_health(panel, snap)
        cv2.imshow("TWM Data Collection", panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not recorder.recording:
            failed = recorder.start_episode()
            if failed:
                log.error("press s again once these pass:\n%s", format_report(failed))
        elif key == ord("e") and recorder.recording:
            recorder.end_episode("operator")
        elif key == ord("r"):
            capture.request_reset_ref()
            log.info("GelSight diff reference reset (queued)")
        elif key == ord("p"):
            if projection:
                show_projection = not show_projection
                log.info("projection overlay %s", "ON" if show_projection else "OFF")
            else:
                log.info("projection overlay unavailable (no calibration loaded)")
        elif key == ord("q"):
            if recorder.recording:
                recorder.end_episode("quit")
            cv2.destroyAllWindows()
            return 0

        remaining = gui_dt - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)


def run_headless(config: RecorderConfig, duration_s: float, drivers: Optional[Drivers] = None,
                 poll_interval_s: float = 0.05, health_every_s: float = 10.0,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> int:
    """Record one timed episode with no GUI. 0 valid, 1 auto-ended, 2 refused."""
    n_arducam = 2 if config.use_arducam else 0
    if not _startup_preflight_ok(config, n_arducam):
        return 2
    rig = SensorRig.open(config, drivers)
    restore_logging()
    rig.wait_ready(config.startup_timeout_s, config.settle_s)
    try:
        session = build_session(config, rig)
    except BaseException:
        rig.close()
        raise
    recorder, capture = session.recorder, session.capture
    try:
        capture.start()
        snapshot_deadline = clock() + config.startup_timeout_s
        while capture.latest() is None:
            if clock() >= snapshot_deadline:
                log.error("capture thread published no snapshot within %.0fs",
                         config.startup_timeout_s)
                return 2
            sleep(poll_interval_s)
        failed = recorder.start_episode()
        if failed:
            log.error("soak refused:\n%s", format_report(failed))
            return 2
        t0 = clock()
        next_health = t0 + health_every_s
        while clock() - t0 < duration_s:
            snap = capture.latest()
            summary = recorder.poll(snap)
            if summary is not None:
                # end_episode (called inside recorder.poll) already logged
                # summary.describe() at the right level; don't repeat it.
                log.error("soak auto-ended after %.1fs", clock() - t0)
                log.info("episode file: %s", summary.path)
                return 1
            if clock() >= next_health:
                text, level = health_line(snap.writer, config.writer.warn_fraction,
                                          config.disk.min_free_gb, sensors=snap.sensors)
                log_fn = {"ok": log.info, "warn": log.warning, "fail": log.error}[level]
                log_fn("[%s] t=%.0fs frames=%d fps=%.1f | %s", level.upper(), clock() - t0,
                      snap.frame_count, snap.fps_meas, text)
                next_health += health_every_s
            sleep(poll_interval_s)
        summary = recorder.end_episode("operator")
        # end_episode already logged summary.describe(); avoid the duplicate.
        log.info("episode file: %s", summary.path)
        return 0 if summary.valid else 1
    finally:
        recorder.close()


_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"


def configure_logging() -> None:
    """Console logging for the recorder's entry points."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format=_LOG_FORMAT, datefmt="%H:%M:%S")


def restore_logging() -> None:
    """Re-apply console logging after a driver reconfigured Python logging.

    `rospy.init_node()` (OptiTrack) installs its own logging config: it
    replaces the root handlers with a ROS file handler and disables existing
    loggers, so every recorder message after "starting OptiTrack" would land
    in ~/.ros/log instead of the operator's terminal. Called once the rig is
    open, on both the GUI and the headless path.
    """
    root = logging.getLogger()
    logging.getLogger("twm.recorder").disabled = False
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    if not any(isinstance(h, (_ConsoleHandler, logging.StreamHandler))
               and getattr(h, "stream", None) in (sys.stdout, sys.stderr)
               for h in root.handlers):
        handler = _ConsoleHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)


class _ConsoleHandler(logging.StreamHandler):
    """A stdout handler that resolves sys.stdout at emit time, so it keeps
    working when the stream object is swapped (pytest capture, redirects)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def main(argv=None) -> int:
    configure_logging()
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
