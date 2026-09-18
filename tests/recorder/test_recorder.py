import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from twm.recorder import app as app_module
from twm.recorder.app import Recorder
from twm.recorder.capture import CaptureLoop
from twm.recorder.config import DiskConfig, RecorderConfig, WriterConfig
from twm.recorder.episode import EpisodeStore
from twm.recorder.frames import synthetic_tick
from twm.recorder.rig import Drivers
from twm.recorder.schema import append_ticks
from twm.recorder.writer import EpisodeWriter


class FakeRig:
    arducam_encoding = "bgr8"
    arducam_config = ()

    def __init__(self):
        self.poses = {"sensor_left": None, "sensor_right": None}
        self.base = synthetic_tick(0.0, seed=1)

    def grab(self):
        t = time.time()
        return type(self.base)(timestamp=t, color=self.base.color, depth=self.base.depth,
                               gelsight=self.base.gelsight, gelsight_ts=(t, t),
                               optitrack={"motherboard": [(t, [0] * 7)]})

    def latest_poses(self):
        return dict(self.poses)

    def fresh(self):
        now = time.time()
        self.poses = {k: (now, [0] * 7) for k in self.poses}

    def close(self):
        self.closed = True


class BlockingRig(FakeRig):
    """Simulates a capture thread parked forever inside a blocking sensor
    read (GelSight's restart loop in base_video_stream.py can do exactly
    this). `grab()` behaves like `FakeRig` until `arm()` is called; every
    call after that blocks on `release`, which the test must set in
    teardown so the capture thread does not leak into later tests."""

    def __init__(self):
        super().__init__()
        self._armed = threading.Event()
        self.release = threading.Event()

    def arm(self):
        self._armed.set()

    def grab(self):
        if self._armed.is_set():
            self.release.wait()
        return super().grab()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.005)
    raise AssertionError("timeout")


@pytest.fixture
def parts(tmp_path):
    cfg = RecorderConfig(task="t", data_dir=tmp_path, fps=60, warmup_drop_frames=0,
                         writer=WriterConfig(queue_seconds=0.5, batch_size=2,
                                             overload_sustained_s=0.2),
                         ot_watchdog_timeout_s=0.5)
    rig = FakeRig()
    gate = threading.Event()
    gate.set()

    def sink(f, ticks):
        gate.wait(5)
        append_ticks(f, ticks)

    tick_bytes = rig.grab().nbytes()
    writer = EpisodeWriter(capacity_bytes=tick_bytes * 4, batch_size=2,
                           overload_sustained_s=0.2, sink=sink)
    capture = CaptureLoop(rig, writer, fps=cfg.fps, warmup_drop_frames=0)
    store = EpisodeStore(tmp_path, "t", date="2026-09-05")
    rec = Recorder(cfg, rig, writer, capture, store,
                   disk_usage=lambda p: SimpleNamespace(free=500e9))
    capture.start()
    _wait(capture.latest)
    yield SimpleNamespace(cfg=cfg, rig=rig, writer=writer, capture=capture,
                          store=store, rec=rec, gate=gate)
    rec.close()


@pytest.fixture
def stalled(tmp_path):
    """Like `parts`, but with a `BlockingRig` and a small `max_tick_gap_s`
    so a stalled capture thread can be simulated (and the 4x threshold
    crossed) without a multi-second sleep. `ot_watchdog_timeout_s` is set
    high so the OptiTrack watchdog cannot fire first and mask the check
    under test."""
    cfg = RecorderConfig(task="t", data_dir=tmp_path, fps=60, warmup_drop_frames=0,
                         writer=WriterConfig(queue_seconds=0.5, batch_size=2,
                                             overload_sustained_s=0.2,
                                             max_tick_gap_s=0.05),
                         ot_watchdog_timeout_s=5.0)
    rig = BlockingRig()
    tick_bytes = rig.grab().nbytes()
    writer = EpisodeWriter(capacity_bytes=tick_bytes * 20, batch_size=2,
                           overload_sustained_s=0.2)
    capture = CaptureLoop(rig, writer, fps=cfg.fps, warmup_drop_frames=0,
                          max_tick_gap_s=cfg.writer.max_tick_gap_s)
    store = EpisodeStore(tmp_path, "t", date="2026-09-05")
    rec = Recorder(cfg, rig, writer, capture, store,
                   disk_usage=lambda p: SimpleNamespace(free=500e9))
    capture.start()
    _wait(capture.latest)
    yield SimpleNamespace(cfg=cfg, rig=rig, writer=writer, capture=capture,
                          store=store, rec=rec)
    rig.release.set()              # let a blocked grab() return so stop() can join
    rec.close()


def test_start_refuses_when_optitrack_is_silent(parts):
    fails = parts.rec.start_episode()
    assert [f.name for f in fails] == ["optitrack_fresh"]
    assert parts.rec.recording is False


def test_operator_episode_is_valid_and_logged(parts):
    parts.rig.fresh()
    assert parts.rec.start_episode() == []
    assert parts.rec.recording
    _wait(lambda: parts.capture.latest().frame_count >= 4)
    s = parts.rec.end_episode()
    assert s.valid and s.ended_by == "operator" and s.frame_count >= 4
    with h5py.File(s.path, "r") as f:
        assert f["timestamps"].shape[0] == s.frame_count
        assert bool(f["metadata"].attrs["valid"]) is True
        assert f["optitrack/motherboard/pose"].shape[0] > 0
    assert "ep_000" in parts.store.log_path.read_text()
    assert parts.rec.start_episode() == []          # a second episode numbers 001
    assert parts.rec.end_episode().episode_num == 1


def test_overload_auto_ends_as_invalid_without_losing_accepted_ticks(parts):
    parts.rig.fresh()
    parts.gate.clear()
    parts.rec.start_episode()
    snap = _wait(lambda: parts.capture.latest() if parts.capture.latest().stop_request else None)
    assert snap.stop_request.kind == "overload"
    parts.gate.set()
    s = parts.rec.poll(snap)
    assert s is not None and s.valid is False and s.ended_by == "overload"
    with h5py.File(s.path, "r") as f:
        assert f["timestamps"].shape[0] == s.frame_count == 4
        assert f["metadata"].attrs["invalid_reason"].startswith("overload:")
    assert "INVALID: overload" in parts.store.log_path.read_text()
    assert parts.rec.recording is False
    assert parts.rec.poll(parts.capture.latest()) is None   # nothing left to end


def test_watchdog_auto_ends_but_keeps_episode_valid(parts):
    parts.rig.fresh()
    parts.rec.start_episode()
    _wait(lambda: parts.capture.latest().frame_count >= 2)
    parts.rig.poses = {k: (time.time() - 5.0, [0] * 7) for k in parts.rig.poses}
    snap = _wait(lambda: parts.capture.latest()
                 if parts.capture.latest().ot_poses["sensor_left"][0] < time.time() - 4 else None)
    s = parts.rec.poll(snap)
    assert s.valid and s.ended_by == "watchdog" and "silent" in s.reason


def test_poll_ends_stalled_capture_thread_as_invalid_not_watchdog(stalled):
    """A capture thread parked in a blocking grab() publishes no new
    snapshot: poll() must catch that itself, before it ever reaches the
    OptiTrack watchdog, or a hung sensor gets finalized as a valid episode
    with OptiTrack blamed for it."""
    stalled.rig.fresh()
    assert stalled.rec.start_episode() == []
    _wait(lambda: stalled.capture.latest().frame_count >= 1)
    stalled.rig.arm()
    threshold = stalled.cfg.writer.max_tick_gap_s * 4
    time.sleep(threshold + 0.2)
    snap = stalled.capture.latest()
    s = stalled.rec.poll(snap)
    assert s is not None
    assert s.valid is False and s.ended_by == "capture_stall"
    assert "no tick for" in s.reason


def test_start_refuses_on_a_stale_capture_snapshot(stalled):
    """start_episode reads OptiTrack and disk live, so both pass even while
    the capture thread is stuck; it must separately refuse on a stale
    snapshot instead of starting a zero-tick recording."""
    stalled.rig.fresh()
    stalled.rig.arm()
    threshold = stalled.cfg.writer.max_tick_gap_s * 4
    time.sleep(threshold + 0.2)
    fails = stalled.rec.start_episode()
    assert [f.name for f in fails] == ["capture_alive"]
    assert stalled.rec.recording is False


def test_close_finalizes_open_episode_as_quit(parts):
    parts.rig.fresh()
    parts.rec.start_episode()
    _wait(lambda: parts.capture.latest().frame_count >= 1)
    parts.rec.close()
    s = parts.rec.last_summary
    assert s.ended_by == "quit" and s.valid
    assert parts.rig.closed
    with h5py.File(s.path, "r") as f:
        assert f["metadata"].attrs["ended_by"] == "quit"


def test_stats_failure_still_closes_and_marks_episode_invalid(parts, monkeypatch):
    parts.rig.fresh()
    assert parts.rec.start_episode() == []
    _wait(lambda: parts.capture.latest().frame_count >= 1)
    handle = parts.rec._open.h5

    def broken_stats():
        raise RuntimeError("stats unavailable")

    monkeypatch.setattr(parts.writer, "stats", broken_stats)
    summary = parts.rec.end_episode()
    assert not handle.id.valid
    assert not summary.valid
    assert summary.ended_by == "writer_fault"
    assert "stats unavailable" in summary.reason
    with h5py.File(summary.path, "r") as f:
        assert not f["metadata"].attrs["valid"]


class _Stream:
    """Minimal hardware double: starts, stops, hands back a fixed frame."""

    def __init__(self, log, name):
        self.log, self.name = log, name
        self.color = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.zeros((480, 640), np.uint16)

    def start(self, **kw):
        self.log.append(f"start {self.name}")

    def stop(self):
        self.log.append(f"stop {self.name}")

    def get_color_frame(self, **kw):
        return self.color

    def get_depth_frame(self, **kw):
        return self.depth

    def get_frame(self, **kw):
        return self.color

    def get_frame_with_timestamp(self, **kw):
        return self.color, None


class _Optitrack:
    def __init__(self, log):
        self.log = log

    def start(self):
        self.log.append("start optitrack")

    def stop(self):
        self.log.append("stop optitrack")

    def get_latest_pose(self, name):
        return None

    def flush_buffer(self, name):
        return []


def test_run_closes_rig_when_startup_fails_after_open(tmp_path, monkeypatch):
    """Regression for a device leak: EpisodeWriter/CaptureLoop/EpisodeStore/
    Recorder used to be constructed outside any try/finally around the open
    rig, so a failure there (e.g. --queue_seconds 0, which makes
    EpisodeWriter raise ValueError) left every sensor still open. `run()`
    must close the rig — and stop a writer it did manage to build — before
    propagating, and must never reach the GUI loop."""
    log = []
    fake_drivers = Drivers(
        realsense=lambda serial, fps, align=True: _Stream(log, f"rs {serial}"),
        gelsight=lambda serial, resolution, name: _Stream(log, f"gs {name}"),
        optitrack=lambda: _Optitrack(log),
        arducam=lambda config, device, encoding="bgr8": _Stream(log, f"ard {device}"),
        resolve_arducams=lambda path: [],
        sleep=lambda s: None,
    )
    cfg = RecorderConfig(
        task="t", data_dir=tmp_path, use_arducam=False, settle_s=0.0,
        realsense_serials=("A",), gelsight_serials={"left": "L", "right": "R"},
        writer=WriterConfig(queue_seconds=0.0),
        disk=DiskConfig(bandwidth_test_s=0.0))

    # The disk-space check in run_startup_preflight depends on the real
    # filesystem under tmp_path; stub it out so this test only exercises the
    # cleanup path under test, not an unrelated preflight failure.
    monkeypatch.setattr(app_module, "run_startup_preflight", lambda *a, **k: [])

    with pytest.raises(ValueError, match="capacity_bytes must be positive"):
        app_module.run(cfg, drivers=fake_drivers)

    assert log.count("stop rs A") == 1
    assert log.count("stop gs left") == 1
    assert log.count("stop gs right") == 1
    assert log.count("stop optitrack") == 1


def test_restore_logging_survives_ros_style_reconfiguration(capsys):
    """rospy.init_node() replaces the root logger's handlers with a ROS file
    handler, silencing every recorder message after OptiTrack starts. The
    recorder re-applies its own console logging once the rig is open."""
    import logging
    from twm.recorder.app import configure_logging, restore_logging
    configure_logging()
    root = logging.getLogger()
    for h in list(root.handlers):                 # what rospy does
        root.removeHandler(h)
    root.addHandler(logging.NullHandler())
    logging.getLogger("twm.recorder").disabled = True
    logging.getLogger("twm.recorder").info("lost")
    restore_logging()
    logging.getLogger("twm.recorder").info("visible again")
    out = capsys.readouterr().out
    assert "lost" not in out
    assert "visible again" in out


def test_episode_metadata_records_sensor_restarts_during_the_episode(parts):
    import json
    counts = {"gelsight_left": 3, "gelsight_right": 0}
    parts.rig.sensor_status = lambda: {n: {"stale_s": 0.0, "restarting": False, "restarts": c}
                                       for n, c in counts.items()}
    parts.rig.fresh()
    assert parts.rec.start_episode() == []
    counts["gelsight_left"] = 5                       # two restarts while recording
    _wait(lambda: parts.capture.latest().frame_count >= 2)
    s = parts.rec.end_episode()
    assert s.valid and s.sensor_restarts == {"gelsight_left": 2, "gelsight_right": 0}
    assert "restarts" in s.describe()
    with h5py.File(s.path, "r") as f:
        assert json.loads(f["metadata"].attrs["sensor_restarts"]) == {"gelsight_left": 2, "gelsight_right": 0}


def test_live_projection_loads_the_newest_calibration_epoch(caplog):
    """The overlay is preview-only and degrades to None on any failure, so a
    calibration directory it can no longer find disables it in silence."""
    import logging

    from twm.calib_epoch import CURRENT_EPOCH
    from twm.recorder.app import load_projection
    from twm.recorder.config import RecorderConfig

    with caplog.at_level(logging.INFO, logger="twm.recorder"):
        proj = load_projection(RecorderConfig(task="t"))
    assert proj is not None and len(proj["cams"]) == 3
    assert CURRENT_EPOCH in "\n".join(caplog.messages)
