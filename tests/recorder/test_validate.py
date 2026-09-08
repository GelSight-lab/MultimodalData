import json
import logging

import numpy as np, h5py, pytest

from twm.recorder.frames import Tick, synthetic_tick
from twm.recorder.schema import append_ticks, create_episode_file, write_episode_attrs
from twm.recorder.validate import validate_episode


def make_episode(tmp_path, n=90, fps=30, gap_at=None, sensor_lag=0.02, lag_overrides=None,
                 frozen_stream=None, valid=True, drop_last_gelsight=False, frozen_ticks=None):
    """`lag_overrides` maps tick index -> gelsight lag (seconds) for that
    tick only, so a test can inject a lag spike on a handful of ticks while
    the rest stay at `sensor_lag` (e.g. "2% of ticks are 0.4s late")."""
    lag_overrides = lag_overrides or {}
    frozen_at = 100.0 - sensor_lag
    f, path = create_episode_file(str(tmp_path), 0, ["A"], ["L", "R"], fps, n_realsense=1)
    rng = np.random.default_rng(0)
    ticks = []
    t = 100.0
    for k in range(n):
        if gap_at is not None and k == gap_at:
            t += 1.0
        tk = synthetic_tick(t, seed=k, n_realsense=1)
        gs = tk.gelsight if frozen_stream != "gelsight" else synthetic_tick(0, seed=0, n_realsense=1).gelsight
        lag = lag_overrides.get(k, sensor_lag)
        gs_ts = t - lag
        if frozen_ticks is not None and frozen_ticks[0] <= k < frozen_ticks[0] + frozen_ticks[1]:
            gs_ts = frozen_at                        # the sensor clock stops: a stall/restart
        else:
            frozen_at = gs_ts
        ticks.append(Tick(t, color=tk.color, depth=tk.depth, gelsight=gs,
                          gelsight_ts=(gs_ts, t - lag),
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
    (dict(sensor_lag=0.0), "sensor_sync"),
    (dict(valid=False), "metadata"),
    (dict(n=30), "duration"),
])
def test_defects_are_named(tmp_path, kw, failing):
    r = validate_episode(make_episode(tmp_path, **kw), fps=30, expected_duration=3.0)
    assert not r.ok
    assert failing in [c.name for c in r.checks if not c.ok]


# ── controller ruling: sensor_sync tolerates <1% late samples, but never a
#    sample past the drivers' own max_age/max_no_update_time contract ──────

def test_sensor_sync_fails_when_2pct_of_ticks_are_late(tmp_path):
    n = 100
    overrides = {k: 0.4 for k in range(2)}          # 2 % of ticks, 0.4s lag
    r = validate_episode(make_episode(tmp_path, n=n, lag_overrides=overrides), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["sensor_sync"].ok
    assert "2" in names["sensor_sync"].detail
    assert r.stats["sensor_late_count.gelsight/left"] == 2


def test_sensor_sync_passes_when_only_half_a_percent_of_ticks_are_late(tmp_path):
    n = 200
    overrides = {0: 0.4}                            # 0.5 % of ticks, 0.4s lag
    r = validate_episode(make_episode(tmp_path, n=n, lag_overrides=overrides), fps=30)
    names = {c.name: c for c in r.checks}
    assert names["sensor_sync"].ok, names["sensor_sync"].detail
    assert r.stats["sensor_late_count.gelsight/left"] == 1


def test_a_brief_stall_beyond_the_driver_contract_is_a_resolved_outage(tmp_path):
    """A sensor clock frozen for 0.53 s (16 ticks) is what a brief stall looks
    like once the recorder stops blocking on sensors; it is reported as one
    outage on that side and, because it resolved, does not fail the episode."""
    r = validate_episode(make_episode(tmp_path, n=200, frozen_ticks=(5, 16)), fps=30)
    names = {c.name: c for c in r.checks}
    assert names["sensor_sync"].ok, names["sensor_sync"].detail
    assert "1 outage" in names["sensor_sync"].detail
    assert r.stats["sensor_outage_count.gelsight/left"] == 1
    assert r.stats["sensor_outage_count.gelsight/right"] == 0


# ── Critical 1: unreadable files never crash the validator ──────────────────

def test_validate_reports_instead_of_crashing_on_zero_byte_file(tmp_path):
    path = tmp_path / "empty.h5"
    path.write_bytes(b"")
    r = validate_episode(str(path), fps=30)
    assert not r.ok
    names = {c.name: c for c in r.checks}
    assert not names["metadata"].ok
    assert names["metadata"].detail  # carries the exception text
    for name in ("shapes", "tick_rate", "duration", "sensor_sync",
                "content", "optitrack", "writer"):
        assert not names[name].ok
        assert names[name].detail == "file not readable"
    json.dumps(r.to_dict())  # never raises


def test_validate_reports_instead_of_crashing_on_missing_timestamps(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as f:
        del f["timestamps"]
    r = validate_episode(str(path), fps=30)
    assert not r.ok
    json.dumps(r.to_dict())  # never raises


# ── Important 1: duration subtracts warm-up, and warns when skipped ─────────

def test_duration_check_subtracts_warmup_frames(tmp_path):
    path = make_episode(tmp_path, n=289, fps=30)
    r = validate_episode(str(path), fps=30, expected_duration=10.0, warmup_frames=10)
    names = {c.name: c for c in r.checks}
    assert names["duration"].ok, names["duration"].detail


def test_duration_check_fails_below_warmup_adjusted_requirement(tmp_path):
    path = make_episode(tmp_path, n=280, fps=30)
    r = validate_episode(str(path), fps=30, expected_duration=10.0, warmup_frames=10)
    names = {c.name: c for c in r.checks}
    assert not names["duration"].ok


def test_duration_check_skips_and_warns_without_expected_duration(tmp_path, caplog):
    path = make_episode(tmp_path)
    with caplog.at_level(logging.WARNING, logger="twm.recorder.validate"):
        r = validate_episode(str(path), fps=30)
    names = {c.name: c for c in r.checks}
    assert names["duration"].ok
    assert "skipped" in names["duration"].detail
    assert any("duration" in rec.message.lower() for rec in caplog.records)


# ── Important 3: missing failing-path tests ──────────────────────────────────

def test_writer_check_fails_on_high_queue_peak_fraction(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as f:
        f["metadata"].attrs["queue_peak_fraction"] = 0.6
    r = validate_episode(str(path), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["writer"].ok


def test_writer_check_fails_on_missing_attr(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as f:
        del f["metadata"].attrs["writer_mean_mb_s"]
    r = validate_episode(str(path), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["writer"].ok
    assert "writer_mean_mb_s" in names["writer"].detail


def test_optitrack_check_fails_on_out_of_span_timestamps(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as f:
        ts = f["optitrack/motherboard/timestamps"][:]
        ts[0] -= 5.0
        f["optitrack/motherboard/timestamps"][:] = ts
    r = validate_episode(str(path), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["optitrack"].ok


def test_optitrack_check_fails_on_decreasing_timestamps(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as f:
        ts = f["optitrack/motherboard/timestamps"][:]
        ts[0], ts[1] = ts[1], ts[0]
        f["optitrack/motherboard/timestamps"][:] = ts
    r = validate_episode(str(path), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["optitrack"].ok


def test_exception_in_one_check_does_not_abort_the_others(tmp_path, monkeypatch):
    import twm.recorder.validate as validate_mod

    def boom(f, stats):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(validate_mod, "check_writer", boom)
    r = validate_mod.validate_episode(str(make_episode(tmp_path)), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["writer"].ok
    assert "kaboom" in names["writer"].detail
    assert names["metadata"].ok
    assert names["shapes"].ok


def test_good_report_is_json_serializable(tmp_path):
    r = validate_episode(make_episode(tmp_path), fps=30, expected_duration=3.0)
    text = json.dumps(r.to_dict())
    parsed = json.loads(text)
    assert parsed["ok"] is True


# ── Important 5: shapes fails on a stream metadata promised but doesn't have ─

def test_shapes_check_fails_when_promised_stream_is_missing(tmp_path):
    path = make_episode(tmp_path)
    with h5py.File(path, "r+") as f:
        del f["gelsight/right"]
    r = validate_episode(str(path), fps=30)
    names = {c.name: c for c in r.checks}
    assert not names["shapes"].ok
    assert "gelsight/right" in names["shapes"].detail


def test_sensor_outage_is_reported_and_tolerated_when_short(tmp_path):
    """A GelSight stall + driver restart freezes its capture clock for a few
    seconds; the recorder keeps ticking with the last frame and its old
    timestamp. That is an outage to report, not a reason to fail the episode."""
    r = validate_episode(make_episode(tmp_path, n=300, frozen_ticks=(100, 18)),
                         fps=30, expected_duration=10.0)
    sync = next(c for c in r.checks if c.name == "sensor_sync")
    assert sync.ok, sync.detail
    assert "1 outage" in sync.detail and "gelsight/left" in sync.detail
    assert r.stats["sensor_outage_count.gelsight/left"] == 1
    # an 18-tick (0.6 s) freeze: the outage spans the ticks with lag > 250 ms (~0.35 s, 3.7 %)
    assert 0.25 < r.stats["sensor_outage_longest_s.gelsight/left"] < 0.45
    assert r.stats["sensor_outage_count.gelsight/right"] == 0


@pytest.mark.parametrize("frozen, why", [
    ((100, 50), "outages cover more than 5% of the episode"),
    ((270, 30), "the sensor never came back before the episode ended"),
])
def test_sensor_outage_fails_when_long_or_unresolved(tmp_path, frozen, why):
    r = validate_episode(make_episode(tmp_path, n=300, frozen_ticks=frozen), fps=30, expected_duration=10.0)
    sync = next(c for c in r.checks if c.name == "sensor_sync")
    assert not sync.ok, why
    assert "gelsight/left" in sync.detail
