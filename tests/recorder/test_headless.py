import time
from types import SimpleNamespace
import h5py, numpy as np, pytest
from twm.recorder import app as app_module
from twm.recorder.config import DiskConfig, RecorderConfig, WriterConfig
from twm.recorder.rig import Drivers
from twm.recorder.validate import validate_episode
from tests.recorder.test_rig import Stream, Optitrack, Cam   # reuse the fakes


def _arducam_slot(slot):
    """test_rig.Cam predates the width/height/fps/pixel_format metadata that
    schema.create_episode_file now stamps for every arducam slot; add them
    here rather than touching the shared fake."""
    cam = Cam(slot)
    cam.width, cam.height, cam.fps, cam.pixel_format = 640, 480, 30, "MJPG"
    return cam


class RealisticStream(Stream):
    """Like test_rig.Stream but with per-frame variation and a real
    wall-clock capture timestamp, instead of one frozen constant-color
    frame and ts=None.

    Two validator checks need this: `content` rejects a stream whose
    sampled frames all have zero variance or never change (the plain
    Stream fake returns the identical constant-fill frame every call), and
    `sensor_sync` rejects a stream whose timestamp is identical to the tick
    clock (the rig substitutes the tick's own clock when a stream reports
    ts=None — rig.py's `frame_with_timestamp` fallback — which is exactly
    what an unmeasured sensor clock looks like).

    A handful of noisy frames are precomputed once at construction and
    cycled through on each call — generating fresh noise on every call
    inside the real-time capture loop is slow enough (a few ms per 480x640x3
    array) to starve the loop and undercount ticks at fps=60."""

    # Independent per-channel counters, each incrementing by exactly one per
    # tick (a stream's color path — get_color_frame / get_frame /
    # get_frame_with_timestamp — is used exclusively by exactly one sensor
    # role, never two of them together, so this never double-advances).
    # N_VARIANTS is large relative to the ~20 evenly spaced samples the
    # validator's content check draws from a short episode, so a sampled
    # pair landing on the same variant by aliasing is very unlikely.
    _N_VARIANTS = 16

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        rng = np.random.default_rng()
        self._color_variants = [self._make_variant(self.color, rng)
                                for _ in range(self._N_VARIANTS)]
        self._depth_variants = [self._make_variant(self.depth, rng)
                                for _ in range(self._N_VARIANTS)]
        self._color_k = 0
        self._depth_k = 0

    @staticmethod
    def _make_variant(base, rng):
        noise = rng.integers(0, 32, base.shape, dtype=np.int32)
        hi = int(np.iinfo(base.dtype).max)
        return np.clip(base.astype(np.int32) + noise, 0, hi).astype(base.dtype)

    def _next_color(self):
        self._color_k += 1
        return self._color_variants[self._color_k % self._N_VARIANTS]

    def get_color_frame(self, **kw):
        return self._next_color()

    def get_depth_frame(self, **kw):
        self._depth_k += 1
        return self._depth_variants[self._depth_k % self._N_VARIANTS]

    def get_frame(self, **kw):
        return self._next_color()

    def get_frame_with_timestamp(self, **kw):
        return self._next_color(), time.time() - 0.02


def fake_drivers(log):
    return Drivers(
        realsense=lambda serial, fps: RealisticStream(log, f"rs {serial}", value=int(serial[-1])),
        gelsight=lambda serial, resolution, name: RealisticStream(log, f"gs {name}", value=7),
        optitrack=lambda: Optitrack(log),
        arducam=lambda config, device: RealisticStream(log, f"ard {device}", value=9),
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

    r = validate_episode(str(files[0]), fps=60, expected_duration=0.6, warmup_frames=2)
    assert r.ok, [c for c in r.checks if not c.ok]


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
