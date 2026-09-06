# TWM Recorder Soak Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless timed recording command and an HDF5 episode validator, plus the config flags needed to run the recorder on the attached rig without ROS and with any RealSense set.

**Architecture:** `run_headless` drives the same `Recorder`/`CaptureLoop`/`EpisodeWriter` objects as the GUI through a shared `build_session`. `validate_episode` reads a finished file and returns named `(ok, detail)` checks. Config gains `--realsense_serials` and `--no_optitrack`; the schema creates as many RealSense groups as serials.

**Tech Stack:** Python 3.9, numpy, h5py + hdf5plugin, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-twm-recorder-soak-test-design.md`

## Global Constraints

- Python 3.9 syntax (`from __future__ import annotations`; no `match`). Tests via `python -m pytest`. Work in `.worktrees/twm-arducam` on branch `feature/twm-soak-test`.
- `twm/recorder/` never imports hardware or cv2 at module import time; `logging.getLogger("twm.recorder")`, never `print`, except the CLI in `__main__.py` which prints its JSON report.
- Legacy callers of `create_episode_file` still get three `realsense/cam{i}` groups; the HDF5 layout of every existing dataset is unchanged; `tests/test_hdf5_writer.py` stays green unmodified.
- `Recorder`, `CaptureLoop`, `EpisodeWriter`, `SensorRig` public interfaces are unchanged except where a task says otherwise. Stop kinds, `VALID_ENDINGS`, and metadata attrs are unchanged.
- Commit after each task with the message given.

---

### Task 1: Rig flags — `--realsense_serials`, `--no_optitrack`, N RealSense groups

**Files:**
- Modify: `twm/recorder/config.py` (`RecorderConfig.use_optitrack`, parser flags, `parse_args`)
- Modify: `twm/recorder/rig.py` (`DummyOptitrack`, `SensorRig.open` uses it when `use_optitrack` is False)
- Modify: `twm/recorder/schema.py` (`create_episode_file(..., n_realsense=None)`; `append_ticks` iterates existing groups)
- Modify: `twm/recorder/episode.py` (`EpisodeStore.create(..., n_realsense=None)` passthrough)
- Modify: `twm/recorder/app.py` (`start_episode` passes `n_realsense=len(self.config.realsense_serials)`)
- Test: `tests/recorder/test_config.py`, `tests/recorder/test_rig.py`, `tests/recorder/test_schema.py` (append tests)

**Interfaces:**
- Produces: `RecorderConfig.use_optitrack: bool = True`; `parse_args(["--realsense_serials", "A,B"])` → `realsense_serials == ("A", "B")`; `parse_args(["--no_optitrack"])` → `use_optitrack False` and `active_sensors == ()`; `DummyOptitrack` with `start/stop/get_latest_pose(name) -> None/flush_buffer(name) -> []`; `create_episode_file(..., n_realsense: Optional[int] = None)` (None → 3); `count_realsense_groups(f) -> int`; `EpisodeStore.create(..., n_realsense=None)`.

- [ ] **Step 1: Failing tests**

Append to `tests/recorder/test_config.py`:
```python
def test_realsense_serials_and_no_optitrack_flags():
    cfg = parse_args(["--task", "t", "--realsense_serials", "143322063538,134322071848",
                      "--no_optitrack"])
    assert cfg.realsense_serials == ("143322063538", "134322071848")
    assert cfg.use_optitrack is False
    assert cfg.active_sensors == ()
    assert parse_args(["--task", "t"]).use_optitrack is True
```
Append to `tests/recorder/test_rig.py`:
```python
def test_no_optitrack_installs_dummy_and_never_calls_the_driver():
    log = []
    d = drivers(log)
    d = Drivers(**{**d.__dict__, "optitrack": lambda: (_ for _ in ()).throw(AssertionError("driver called"))})
    rig = SensorRig.open(config(use_arducam=False, use_optitrack=False), d)
    tick = rig.grab()
    assert tick.optitrack == {name: [] for name in rig.trackers}
    assert rig.latest_poses() == {name: None for name in rig.trackers}
    rig.close()
    assert "start optitrack" not in log and "stop optitrack" not in log
```
Append to `tests/recorder/test_schema.py`:
```python
def test_n_realsense_controls_group_count_and_append(tmp_path):
    from twm.recorder.schema import count_realsense_groups
    f, _ = create_episode_file(str(tmp_path), 3, ["A", "B"], ["L", "R"], 30, n_realsense=2)
    assert count_realsense_groups(f) == 2 and "realsense/cam2" not in f
    t = synthetic_tick(1.0, n_realsense=2)
    append_ticks(f, [t])
    assert f["realsense/cam1/color"].shape[0] == 1
    f.close()
    g, _ = create_episode_file(str(tmp_path), 4, ["A", "B", "C"], ["L", "R"], 30)
    assert count_realsense_groups(g) == 3        # legacy default
    g.close()
```

- [ ] **Step 2: Run, expect failures** — `python -m pytest tests/recorder/test_config.py tests/recorder/test_rig.py tests/recorder/test_schema.py -q` → the three new tests fail (`unrecognized arguments`, `AssertionError: driver called`, `TypeError: n_realsense`).

- [ ] **Step 3: Implement**

`config.py`: add field `use_optitrack: bool = True` to `RecorderConfig`; parser flags
```python
p.add_argument("--realsense_serials", default=None,
               help="Comma-separated RealSense serials to record (default: the rig's three).")
p.add_argument("--no_optitrack", action="store_true",
               help="Record without OptiTrack (no ROS needed); pose datasets stay empty.")
```
and in `parse_args`: `realsense_serials=tuple(s.strip() for s in a.realsense_serials.split(",") if s.strip()) if a.realsense_serials else REALSENSE_SERIALS`, `use_optitrack=not a.no_optitrack`, `active_sensors=() if a.no_optitrack else ACTIVE_SENSOR_CHOICES[a.active_sensors]`.

`rig.py`:
```python
class DummyOptitrack:
    """Stands in for OptitrackStream when recording without ROS: no poses, empty buffers."""
    def start(self): pass
    def stop(self): pass
    def get_latest_pose(self, name): return None
    def flush_buffer(self, name): return []
```
In `SensorRig.open`: `optitrack = start(drivers.optitrack()) if config.use_optitrack else DummyOptitrack()` (log "OptiTrack disabled" in the else branch; the dummy is not appended to `started`).

`schema.py`: `create_episode_file(..., arducam_config=None, include_legacy=True, n_realsense=None)`; `n = N_REALSENSE if n_realsense is None else int(n_realsense)`; loop `for i in range(n)`. Add
```python
def count_realsense_groups(f) -> int:
    return len([k for k in f["realsense"]]) if "realsense" in f else 0
```
and in `append_ticks` replace `for i in range(N_REALSENSE)` with `for i in range(count_realsense_groups(f))`.

`episode.py`: `EpisodeStore.create(..., arducam_config=None, n_realsense=None)` passes `n_realsense` through. `app.py::start_episode`: pass `n_realsense=len(self.config.realsense_serials)`.

- [ ] **Step 4: Run** — the three files plus `python -m pytest tests -q` all green.
- [ ] **Step 5: Commit** — `git commit -am "feat(recorder): --realsense_serials and --no_optitrack for partial rigs"` (add the test files).

---

### Task 2: Headless soak — `build_session`, `run_headless`, `python -m twm.recorder soak`

**Files:**
- Modify: `twm/recorder/app.py` (extract `build_session`, add `run_headless`)
- Modify: `twm/recorder/__main__.py` (`soak` subcommand)
- Test: `tests/recorder/test_headless.py`

**Interfaces:**
- Produces: `build_session(config, rig) -> Session` (dataclass: `writer, capture, store, recorder`), used by both `run` and `run_headless`; `run_headless(config, duration_s, drivers=None, poll_interval_s=0.05, health_every_s=10.0, clock=time.time, sleep=time.sleep) -> int` returning 0 (valid episode), 1 (auto-ended/invalid), 2 (preflight refused); it logs `summary.describe()` and `"episode file: <path>"`; `__main__` `soak` accepts every `run` flag plus `--duration` (float seconds, required).

- [ ] **Step 1: Failing test**
```python
# tests/recorder/test_headless.py
from types import SimpleNamespace
import h5py, numpy as np, pytest
from twm.recorder import app as app_module
from twm.recorder.config import DiskConfig, RecorderConfig, WriterConfig
from twm.recorder.rig import Drivers
from tests.recorder.test_rig import Stream, Optitrack, Cam   # reuse the fakes


def fake_drivers(log):
    return Drivers(
        realsense=lambda serial, fps: Stream(log, f"rs {serial}", value=int(serial[-1])),
        gelsight=lambda serial, resolution, name: Stream(log, f"gs {name}", value=7, ts=None),
        optitrack=lambda: Optitrack(log),
        arducam=lambda config, device: Stream(log, f"ard {device}", value=9, ts=None),
        resolve_arducams=lambda path: [Cam("cam0"), Cam("cam1")],
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
```
(The `Stream` fake in `test_rig.py` takes `value`/`ts`; if `test_rig.Stream.get_frame_with_timestamp` returns `ts=None`, the rig falls back to the tick time, which is what the validator later needs. If importing from `tests.recorder.test_rig` fails because `tests` is not a package on `sys.path`, copy the three fakes into `tests/recorder/fakes.py` and import from there in both files.)

- [ ] **Step 2: Run, expect** `AttributeError: module 'twm.recorder.app' has no attribute 'run_headless'`.

- [ ] **Step 3: Implement**

In `app.py`:
```python
@dataclass
class Session:
    writer: EpisodeWriter
    capture: CaptureLoop
    store: EpisodeStore
    recorder: Recorder


def build_session(config: RecorderConfig, rig) -> Session:
    """The production wiring shared by the GUI and the headless soak."""
    tick_bytes = full_rig_tick_nbytes(len(rig.realsense), 2, len(rig.arducam))
    writer = EpisodeWriter(
        capacity_bytes=queue_capacity_bytes(config.writer.queue_seconds, config.fps, tick_bytes),
        batch_size=config.writer.batch_size, flush_interval_s=config.writer.flush_interval_s,
        overload_fraction=config.writer.overload_fraction,
        overload_sustained_s=config.writer.overload_sustained_s,
        min_free_gb=config.disk.min_free_gb)
    capture = CaptureLoop(rig, writer, fps=config.fps,
                          warmup_drop_frames=config.warmup_drop_frames,
                          max_tick_gap_s=config.writer.max_tick_gap_s)
    store = EpisodeStore(config.data_dir, config.task)
    return Session(writer, capture, store, Recorder(config, rig, writer, capture, store))
```
Refactor `run()` to use it (keeping the existing try/except cleanup that closes the rig if building fails — `build_session` raising must still close the rig and stop a half-built writer; keep that logic where it is). Then:
```python
def run_headless(config: RecorderConfig, duration_s: float, drivers: Optional[Drivers] = None,
                 poll_interval_s: float = 0.05, health_every_s: float = 10.0,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> int:
    """Record one timed episode with no GUI. 0 valid, 1 auto-ended, 2 refused."""
    n_arducam = 2 if config.use_arducam else 0
    results = run_startup_preflight(config, n_arducam=n_arducam)
    log.info("startup preflight:\n%s", format_report(results))
    if failures(results):
        return 2
    rig = SensorRig.open(config, drivers)
    rig.wait_ready(config.startup_timeout_s, config.settle_s)
    try:
        session = build_session(config, rig)
    except BaseException:
        rig.close()
        raise
    recorder, capture = session.recorder, session.capture
    try:
        capture.start()
        while capture.latest() is None:
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
                log.error("soak auto-ended after %.1fs: %s", clock() - t0, summary.describe())
                log.info("episode file: %s", summary.path)
                return 1
            if clock() >= next_health:
                text, level = health_line(snap.writer, config.writer.warn_fraction,
                                          config.disk.min_free_gb)
                log.info("[%s] t=%.0fs frames=%d fps=%.1f | %s", level.upper(), clock() - t0,
                         snap.frame_count, snap.fps_meas, text)
                next_health += health_every_s
            sleep(poll_interval_s)
        summary = recorder.end_episode("operator")
        log.info(summary.describe())
        log.info("episode file: %s", summary.path)
        return 0 if summary.valid else 1
    finally:
        recorder.close()
```
In `__main__.py`: add a `soak` branch that builds a parser from `twm.recorder.config.build_parser()` plus `--duration` (type float, required), calls `parse_args` on the remaining argv (drop `--duration` first — simplest: `p = build_parser(); p.add_argument("--duration", type=float, required=True); a = p.parse_args(argv[1:])`, then rebuild the config via `parse_args` on `argv[1:]` minus the duration pair; or expose `config_from_namespace(a)` in `config.py` and use it for both — do the latter, it is cleaner), configures logging like `main`, and `sys.exit(run_headless(cfg, a.duration))`.

- [ ] **Step 4: Run** `python -m pytest tests/recorder/test_headless.py -q -p no:cacheprovider` three times, then `python -m pytest tests -q`.
- [ ] **Step 5: Commit** — `feat(recorder): headless timed soak recording via build_session`.

---

### Task 3: Episode validator — `validate_episode`, `python -m twm.recorder validate`

**Files:**
- Create: `twm/recorder/validate.py`
- Modify: `twm/recorder/__main__.py` (`validate` subcommand: `<path> [--fps 30] [--expected-duration S] [--max-tick-gap 0.5] [--report out.json]`)
- Test: `tests/recorder/test_validate.py`

**Interfaces:**
- Produces: `Check(name, ok, detail)`; `ValidationReport(path, checks: List[Check], stats: Dict[str, float])` with `ok` property and `to_dict()`; `validate_episode(path, fps=30, expected_duration=None, max_tick_gap_s=0.5, sample_frames=20) -> ValidationReport`. Check names exactly: `metadata, shapes, tick_rate, duration, sensor_sync, content, optitrack, writer`. `stats` carries at least `T, median_dt, max_dt, late_fraction, sensor_lag_max_s (per stream), distinct_sensor_hz (per stream)`.

- [ ] **Step 1: Failing tests** — build episodes with a helper:
```python
# tests/recorder/test_validate.py
import numpy as np, h5py, pytest
from twm.recorder.frames import Tick, synthetic_tick
from twm.recorder.schema import append_ticks, create_episode_file, write_episode_attrs
from twm.recorder.validate import validate_episode


def make_episode(tmp_path, n=90, fps=30, gap_at=None, sensor_lag=0.02, frozen_stream=None,
                 valid=True, drop_last_gelsight=False):
    f, path = create_episode_file(str(tmp_path), 0, ["A"], ["L", "R"], fps, n_realsense=1)
    rng = np.random.default_rng(0)
    ticks = []
    t = 100.0
    for k in range(n):
        if gap_at is not None and k == gap_at:
            t += 1.0
        tk = synthetic_tick(t, seed=k, n_realsense=1)
        gs = tk.gelsight if frozen_stream != "gelsight" else synthetic_tick(0, seed=0, n_realsense=1).gelsight
        ticks.append(Tick(t, color=tk.color, depth=tk.depth, gelsight=gs,
                          gelsight_ts=(t - sensor_lag, t - sensor_lag),
                          optitrack={"motherboard": [(t, [0, 0, 0, 0, 0, 0, 1])]}))
        t += 1.0 / fps
    append_ticks(f, ticks)
    if drop_last_gelsight:
        f["gelsight/left/frames"].resize(n - 1, axis=0)
    write_episode_attrs(f, {"valid": valid, "invalid_reason": "" if valid else "overload: x",
                            "ended_by": "operator" if valid else "overload", "frame_count": n,
                            "gap_count": 0, "queue_peak_fraction": 0.1, "writer_mean_mb_s": 150.0,
                            "max_tick_gap_s": 0.04, "duration_s": n / fps})
    f.close()
    return path


def test_good_episode_passes_every_check(tmp_path):
    r = validate_episode(make_episode(tmp_path), fps=30, expected_duration=3.0)
    assert r.ok, [c for c in r.checks if not c.ok]
    assert r.stats["T"] == 90 and abs(r.stats["median_dt"] - 1 / 30) < 1e-3


@pytest.mark.parametrize("kw, failing", [
    (dict(gap_at=40), "tick_rate"),
    (dict(drop_last_gelsight=True), "shapes"),
    (dict(frozen_stream="gelsight"), "content"),
    (dict(sensor_lag=0.4), "sensor_sync"),
    (dict(valid=False), "metadata"),
    (dict(n=30), "duration"),
])
def test_defects_are_named(tmp_path, kw, failing):
    r = validate_episode(make_episode(tmp_path, **kw), fps=30, expected_duration=3.0)
    assert not r.ok
    assert failing in [c.name for c in r.checks if not c.ok]
```
(For `n=30`, expected 3 s at 30 fps needs ≥ 87 frames → `duration` fails.)

- [ ] **Step 2: Run, expect** `ModuleNotFoundError: twm.recorder.validate`.

- [ ] **Step 3: Implement** `validate.py` per the spec table. Structure: one small function per check taking the open file and a `stats` dict; `validate_episode` opens the file read-only, runs them in order, catches an exception inside any check and records it as `Check(name, False, f"{type}: {exc}")` so one bad dataset never hides the others. Frame datasets to inspect: every `realsense/cam*/color|depth`, `gelsight/*/frames`, `arducam/*/frames` that exists. Sensor streams for `sensor_sync`: every `gelsight/*/timestamps` and `arducam/*/timestamps`; lag = `timestamps[:] - sensor_ts[:]`; distinct rate = `len(np.unique(sensor_ts)) / (last_tick - first_tick)`; require ≥ 10 Hz only when T > fps (skip for tiny files, detail "too short to judge"). `content`: indices `np.linspace(0, T-1, sample_frames).astype(int)`; per stream compute `std` of each sampled frame (> 1.0) and whether consecutive samples differ (`np.array_equal` False for ≥ half); depth additionally `np.count_nonzero > 0`. `optitrack`: samples across trackers; "no samples" → ok True with detail. `writer`: from metadata attrs. `to_dict()` is JSON-serializable (cast numpy scalars).

`__main__.py`: `validate` subcommand prints `json.dumps(report.to_dict(), indent=2)`, writes it to `--report` if given, exits `0 if report.ok else 1`.

- [ ] **Step 4: Run** the new tests, then `python -m pytest tests -q`, then a smoke: record a 2-second headless episode into `/tmp/twm_soak_smoke` with `python -m twm.recorder soak --task smoke --duration 2 --data_dir /tmp/twm_soak_smoke --no_optitrack --realsense_serials 143322063538 --no_bandwidth_test` ONLY if the hardware is free (it is a real recording on the attached cameras; skip with a note if any device is busy), then `python -m twm.recorder validate` on the produced file and paste the JSON into the report.
- [ ] **Step 5: Commit** — `feat(recorder): episode validator for format, sync, and drop checks`.
