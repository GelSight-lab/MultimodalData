from types import SimpleNamespace
import h5py, numpy as np, pytest
from twm.recorder import app as app_module
from twm.recorder.config import DiskConfig, RecorderConfig, WriterConfig
from twm.recorder.rig import Drivers
from tests.recorder.test_rig import Stream, Optitrack, Cam   # reuse the fakes


def _arducam_slot(slot):
    """test_rig.Cam predates the width/height/fps/pixel_format metadata that
    schema.create_episode_file now stamps for every arducam slot; add them
    here rather than touching the shared fake."""
    cam = Cam(slot)
    cam.width, cam.height, cam.fps, cam.pixel_format = 640, 480, 30, "MJPG"
    return cam


def fake_drivers(log):
    return Drivers(
        realsense=lambda serial, fps: Stream(log, f"rs {serial}", value=int(serial[-1])),
        gelsight=lambda serial, resolution, name: Stream(log, f"gs {name}", value=7, ts=None),
        optitrack=lambda: Optitrack(log),
        arducam=lambda config, device: Stream(log, f"ard {device}", value=9, ts=None),
        resolve_arducams=lambda path: [_arducam_slot("cam0"), _arducam_slot("cam1")],
        sleep=lambda s: None)


def test_run_headless_records_a_valid_episode(tmp_path, monkeypatch):
    log = []
    cfg = RecorderConfig(task="soak", data_dir=tmp_path, fps=60, warmup_drop_frames=2,
                         realsense_serials=("1", "2"), use_optitrack=False, active_sensors=(),
                         settle_s=0.0, writer=WriterConfig(queue_seconds=1.0),
                         disk=DiskConfig(min_free_gb=0.0, bandwidth_test_s=0.0))
    monkeypatch.setattr(app_module.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=1e12))
    code = app_module.run_headless(cfg, duration_s=0.6, drivers=fake_drivers(log),
                                   health_every_s=0.2)
    assert code == 0
    files = sorted((tmp_path / "soak").rglob("episode_*.h5"))
    assert len(files) == 1
    with h5py.File(files[0], "r") as f:
        T = f["timestamps"].shape[0]
        assert 15 <= T <= 60 and f["realsense/cam1/color"].shape[0] == T
        assert bool(f["metadata"].attrs["valid"]) and f["metadata"].attrs["ended_by"] == "operator"
    assert log[-1].startswith("stop ")          # rig closed


def test_run_headless_returns_2_when_preflight_refuses(tmp_path):
    cfg = RecorderConfig(task="soak", data_dir=tmp_path, use_optitrack=False,
                         disk=DiskConfig(min_free_gb=1e9, bandwidth_test_s=0.0))
    assert app_module.run_headless(cfg, duration_s=0.1, drivers=fake_drivers([])) == 2


def test_run_headless_returns_2_when_capture_never_publishes(tmp_path, monkeypatch):
    """A capture thread parked in a blocking sensor read (rig.grab() never
    returns) must not hang the soak forever: the wait for the first
    snapshot is bounded by startup_timeout_s, and the rig/recorder must
    still be torn down cleanly on that timeout."""
    import threading
    import time as time_module

    from twm.recorder.rig import SensorRig

    log = []
    event = threading.Event()

    def blocking_grab(self):
        event.wait()
        raise RuntimeError("released for teardown")

    monkeypatch.setattr(SensorRig, "grab", blocking_grab)
    cfg = RecorderConfig(task="soak", data_dir=tmp_path, use_optitrack=False,
                         realsense_serials=("1", "2"), startup_timeout_s=0.3,
                         settle_s=0.0, disk=DiskConfig(min_free_gb=0.0, bandwidth_test_s=0.0))
    try:
        t0 = time_module.time()
        code = app_module.run_headless(cfg, duration_s=1.0, drivers=fake_drivers(log))
        elapsed = time_module.time() - t0
    finally:
        event.set()          # let the parked capture thread unwind and exit
    assert code == 2
    assert elapsed < 5.0
    assert any(entry.startswith("stop ") for entry in log)


def test_soak_rejects_non_positive_duration():
    from twm.recorder.__main__ import main
    with pytest.raises(SystemExit):
        main(["soak", "--task", "t", "--duration", "0"])
