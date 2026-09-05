import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from twm.recorder.app import Recorder
from twm.recorder.capture import CaptureLoop
from twm.recorder.config import RecorderConfig, WriterConfig
from twm.recorder.episode import EpisodeStore
from twm.recorder.frames import synthetic_tick
from twm.recorder.schema import append_ticks
from twm.recorder.writer import EpisodeWriter


class FakeRig:
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
