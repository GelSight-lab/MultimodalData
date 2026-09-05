import numpy as np
import pytest

from twm.recorder.config import RecorderConfig
from twm.recorder.frames import Tick
from twm.recorder.rig import Drivers, DummyGelSight, SensorRig


class Stream:
    def __init__(self, log, name, fail_start=False, value=1, ts=None):
        self.log, self.name, self.fail_start = log, name, fail_start
        self.color = np.full((480, 640, 3), value, np.uint8)
        self.depth = np.full((480, 640), value, np.uint16)
        self.ts = ts

    def start(self, **kwargs):
        self.log.append(f"start {self.name}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed")

    def stop(self):
        self.log.append(f"stop {self.name}")

    def get_color_frame(self, **kw):
        return self.color

    def get_depth_frame(self, **kw):
        return self.depth

    def get_frame(self, **kw):
        return self.color

    def get_frame_with_timestamp(self, **kw):
        return self.color, self.ts


class Optitrack:
    def __init__(self, log):
        self.log = log
        self.buffers = {"motherboard": [(1.0, [0] * 7)], "sensor_left": [],
                        "sensor_right": [(2.0, [1] * 7)]}

    def start(self):
        self.log.append("start optitrack")

    def stop(self):
        self.log.append("stop optitrack")

    def get_latest_pose(self, name):
        return (3.0, [0] * 7) if name == "motherboard" else None

    def flush_buffer(self, name):
        data, self.buffers[name] = self.buffers[name], []
        return data


class Cam:
    def __init__(self, slot):
        self.slot, self.id_path, self.position = slot, f"usb-{slot}", "unknown"
        self.config, self.device = {}, f"/dev/{slot}"


def drivers(log, gelsight_fail=(), arducam_fail=False):
    return Drivers(
        realsense=lambda serial, fps: Stream(log, f"rs {serial}"),
        gelsight=lambda serial, resolution, name: Stream(
            log, f"gs {name}", fail_start=name in gelsight_fail, ts=42.0),
        optitrack=lambda: Optitrack(log),
        arducam=lambda config, device: Stream(log, f"ard {device}",
                                              fail_start=arducam_fail, ts=7.0),
        resolve_arducams=lambda path: [Cam("cam0"), Cam("cam1")],
        sleep=lambda s: None,
    )


def config(**kw):
    kw.setdefault("realsense_serials", ("A", "B"))
    kw.setdefault("gelsight_serials", {"left": "L", "right": "R"})
    return RecorderConfig(task="t", **kw)


def test_open_starts_in_order_and_close_stops_in_reverse():
    log = []
    rig = SensorRig.open(config(), drivers(log))
    assert log == ["start rs A", "start rs B", "start ard /dev/cam0",
                   "start ard /dev/cam1", "start gs left", "start gs right",
                   "start optitrack"]
    log.clear()
    rig.close()
    rig.close()
    assert log == ["stop optitrack", "stop gs right", "stop gs left",
                   "stop ard /dev/cam1", "stop ard /dev/cam0", "stop rs B",
                   "stop rs A"]


def test_open_failure_stops_everything_started_including_failed_one():
    log = []
    with pytest.raises(RuntimeError, match="cam0 failed"):
        SensorRig.open(config(), drivers(log, arducam_fail=True))
    assert log == ["start rs A", "start rs B", "start ard /dev/cam0",
                   "stop ard /dev/cam0", "stop rs B", "stop rs A"]


def test_keyboard_interrupt_during_open_also_cleans_up():
    log = []
    d = drivers(log)
    d = Drivers(**{**d.__dict__, "optitrack": lambda: (_ for _ in ()).throw(KeyboardInterrupt())})
    with pytest.raises(KeyboardInterrupt):
        SensorRig.open(config(use_arducam=False), d)
    assert log[-4:] == ["stop gs right", "stop gs left", "stop rs B", "stop rs A"]


def test_missing_gelsight_falls_back_to_dummy_and_is_stopped():
    log = []
    rig = SensorRig.open(config(use_arducam=False), drivers(log, gelsight_fail={"right"}))
    assert isinstance(rig.gelsight_right, DummyGelSight)
    assert "stop gs right" in log         # the failed stream was released
    tick = rig.grab()
    assert tick.gelsight[1].max() == 0
    assert tick.gelsight_ts == (42.0, tick.timestamp)   # dummy has no capture time


def test_grab_builds_tick_and_drains_optitrack():
    log = []
    rig = SensorRig.open(config(), drivers(log))
    tick = rig.grab()
    assert isinstance(tick, Tick)
    assert len(tick.color) == 2 and len(tick.depth) == 2
    assert tick.arducam_ts == (7.0, 7.0)
    assert tick.optitrack["motherboard"] == [(1.0, [0] * 7)]
    assert rig.grab().optitrack["motherboard"] == []     # drained
    assert rig.latest_poses()["motherboard"] == (3.0, [0] * 7)
    assert rig.arducam_labels() == ["cam0 usb-cam0 unknown", "cam1 usb-cam1 unknown"]


def test_wait_ready_failure_closes_rig():
    log = []
    d = drivers(log)

    class Slow(Stream):
        def get_color_frame(self, **kw):
            raise TimeoutError("no frame")

    d = Drivers(**{**d.__dict__, "realsense": lambda serial, fps: Slow(log, f"rs {serial}")})
    rig = SensorRig.open(config(use_arducam=False), d)
    with pytest.raises(TimeoutError):
        rig.wait_ready(timeout_s=0.01, settle_s=0.0)
    assert log[-1] == "stop rs A"
