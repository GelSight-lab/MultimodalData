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
                                     n_realsense=len(self.config.realsense_serials))
        self._open = OpenEpisode(num, path, h5)
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
            writer_mean_mb_s=stats.mean_mb_s, has_optitrack=has_ot)
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
    calib_dir = Path(__file__).resolve().parent.parent / "calibration" / "result"  # tactile-lag-exempt: the LIVE recorder always uses the current calibration; epochs only exist for replaying the past
    try:
        cam_calibs, gel_left, gel_right = load_calibrations(
            [calib_dir / CAM_CALIB_NAME[i] for i in range(3)],
            calib_dir / "T_gel_to_rigid_left.json",
            calib_dir / "T_gel_to_rigid_right.json")
    except Exception as exc:
        log.warning("projection overlay disabled (%s)", exc)
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
    log.info("projection overlay: %d cameras calibrated (press p to toggle)", len(cams))
    return {"cams": cams, "gel_left": gel_left, "gel_right": gel_right} if cams else None


# ── operator loop ────────────────────────────────────────────────────────────

def run(config: RecorderConfig, drivers: Optional[Drivers] = None) -> int:
    n_arducam = 2 if config.use_arducam else 0
    log.info("task %s → %s", config.task, config.data_dir / config.task)
    results = run_startup_preflight(config, n_arducam=n_arducam)
    log.info("startup preflight:\n%s", format_report(results))
    if failures(results):
        log.error("startup preflight failed; fix the above and retry")
        return 2

    rig = SensorRig.open(config, drivers)
    rig.wait_ready(config.startup_timeout_s, config.settle_s)
    log.info("all sensors ready")

    writer = None
    try:
        tick_bytes = full_rig_tick_nbytes(len(rig.realsense), 2, len(rig.arducam))
        writer = EpisodeWriter(
            capacity_bytes=queue_capacity_bytes(config.writer.queue_seconds, config.fps, tick_bytes),
            batch_size=config.writer.batch_size,
            flush_interval_s=config.writer.flush_interval_s,
            overload_fraction=config.writer.overload_fraction,
            overload_sustained_s=config.writer.overload_sustained_s,
            min_free_gb=config.disk.min_free_gb)
        capture = CaptureLoop(rig, writer, fps=config.fps,
                              warmup_drop_frames=config.warmup_drop_frames,
                              max_tick_gap_s=config.writer.max_tick_gap_s)
        store = EpisodeStore(config.data_dir, config.task)
        recorder = Recorder(config, rig, writer, capture, store)
    except BaseException:
        # Nothing built here has an owner yet (Recorder.close() below is
        # what usually stops the writer and closes the rig); if we fail
        # partway through, do that ourselves before propagating.
        if writer is not None:
            writer.stop()
        rig.close()
        raise
    try:
        capture.start()
        return _gui_loop(config, recorder, capture, rig, load_projection(config))
    finally:
        recorder.close()


def _gui_loop(config, recorder: Recorder, capture: CaptureLoop, rig,
              projection: Optional[Dict[str, Any]]) -> int:
    import cv2
    from twm.viz import build_preview_panel, draw_projection_overlay

    log.info("controls: s start | e end | r reset diff ref | p projection | q quit")
    arducam_labels = rig.arducam_labels() or None
    show_projection = projection is not None
    preview_dt, gui_dt = 1.0 / config.preview_fps, 1.0 / 30.0
    last_preview_t, panel = 0.0, None

    def draw_health(panel, snap):
        text, level = health_line(snap.writer, config.writer.warn_fraction,
                                  config.disk.min_free_gb)
        color = {"ok": (80, 200, 80), "warn": (0, 200, 255), "fail": (0, 0, 255)}[level]
        cv2.putText(panel, text, (8, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

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

        if panel is None or t0 - last_preview_t >= preview_dt:
            tick = snap.tick
            panel = build_preview_panel(
                list(tick.color), list(tick.gelsight), list(snap.gs_ref), snap.ot_poses,
                snap.recording, snap.frame_count, snap.elapsed,
                snap.writer.queue_items, snap.fps_meas, task_name=config.task,
                arducam_frames=list(tick.arducam) or None,
                arducam_labels=arducam_labels)
            if show_projection and projection:
                draw_projection_overlay(panel, snap.ot_poses, projection["cams"],
                                        projection["gel_left"], projection["gel_right"])
            draw_health(panel, snap)
            last_preview_t = t0
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
                panel = None
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


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
