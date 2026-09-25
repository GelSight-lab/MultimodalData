"""Per-stream integrity: every stream's frame rate is consistent and no
frames were lost, reported stream by stream."""
import numpy as np
import pytest

from twm.recorder.frames import Tick, synthetic_tick
from twm.recorder.schema import append_ticks, create_episode_file, write_episode_attrs
from twm.recorder.integrity import check_integrity


def build(tmp_path, n=120, fps=30, missed_tick_at=None, gelsight_native_hz=None,
          gelsight_skip_at=None, optitrack=True):
    """Ticks at `fps`. `missed_tick_at`: one tick interval doubled (a lost
    tick). `gelsight_native_hz`: the GelSight produces new frames at that
    rate, holding the last frame on the other ticks. `gelsight_skip_at`:
    the GelSight misses one of its own frames at that tick index."""
    f, path = create_episode_file(str(tmp_path), 0, ["A"], ["L", "R"], fps, n_realsense=1)
    ticks, t = [], 100.0
    gs_period = 1.0 / (gelsight_native_hz or fps)
    next_gs = t
    gs_ts = t - 0.02
    for k in range(n):
        if missed_tick_at is not None and k == missed_tick_at:
            t += 1.0 / fps
        if t >= next_gs - 1e-9:
            skips = gelsight_skip_at if isinstance(gelsight_skip_at, tuple) else (gelsight_skip_at,)
            if k not in skips:
                gs_ts = t - 0.02
            next_gs += gs_period
        tk = synthetic_tick(t, seed=k, n_realsense=1)
        ot = {"sensor_left": [(t, [0, 0, 0, 0, 0, 0, 1])]} if optitrack else {}
        ticks.append(Tick(t, color=tk.color, depth=tk.depth, gelsight=tk.gelsight,
                          gelsight_ts=(gs_ts, gs_ts), optitrack=ot))
        t += 1.0 / fps
    append_ticks(f, ticks)
    write_episode_attrs(f, {"valid": True, "invalid_reason": "", "ended_by": "operator",
                            "frame_count": n, "gap_count": 0})
    f.close()
    return path


def by_name(report):
    return {s.name: s for s in report.streams}


def test_clean_episode_is_ok_and_reports_every_stream(tmp_path):
    r = check_integrity(build(tmp_path))
    s = by_name(r)
    assert r.ok
    assert {"timestamps", "realsense/cam0/color", "realsense/cam0/depth",
            "gelsight/left", "gelsight/right", "optitrack/sensor_left"} <= set(s)
    assert s["timestamps"].lost == 0
    assert s["timestamps"].rate_hz == pytest.approx(30, abs=0.1)
    assert s["gelsight/left"].native_hz == pytest.approx(30, abs=0.5)
    assert s["gelsight/left"].held == 0 and s["gelsight/left"].lost == 0
    assert s["optitrack/sensor_left"].rate_hz == pytest.approx(30, abs=0.1)


def test_missed_tick_is_counted_as_a_lost_tick(tmp_path):
    r = check_integrity(build(tmp_path, missed_tick_at=40))
    t = by_name(r)["timestamps"]
    assert t.lost == 1 and not t.ok and not r.ok
    assert t.max_dt_ms == pytest.approx(66.7, abs=1.0)
    assert any("timestamps" in p for p in r.problems)


def test_slow_sensor_holds_frames_but_loses_none(tmp_path):
    r = check_integrity(build(tmp_path, gelsight_native_hz=15))
    g = by_name(r)["gelsight/left"]
    assert g.native_hz == pytest.approx(15, abs=0.5)
    assert g.held == pytest.approx(60, abs=2)
    assert g.lost == 0 and g.ok


def test_sensor_skipping_one_of_its_own_frames_is_counted_but_tolerated(tmp_path):
    r = check_integrity(build(tmp_path, gelsight_skip_at=50))
    g = by_name(r)["gelsight/left"]
    assert g.lost == 1 and g.ok and r.ok          # within the 2-frame floor


def test_sensor_skipping_several_frames_fails(tmp_path):
    r = check_integrity(build(tmp_path, gelsight_skip_at=(30, 50, 70)))
    g = by_name(r)["gelsight/left"]
    assert g.lost == 3 and not g.ok and not r.ok
    assert any("gelsight/left" in p for p in r.problems)


def test_burst_jitter_is_not_loss(tmp_path):
    """Two tracker packets delivered together (0 ms then 20 ms) is jitter, not loss."""
    import h5py
    path = build(tmp_path)
    with h5py.File(path, "a") as f:
        n = 300
        t = 100.0 + np.repeat(np.arange(n // 2) * 0.02, 2) + np.tile([0.0, 0.0002], n // 2)
        f["optitrack/sensor_left/timestamps"].resize(n, axis=0)
        f["optitrack/sensor_left/timestamps"][:] = t
        f["optitrack/sensor_left/pose"].resize(n, axis=0)
    o = by_name(check_integrity(path))["optitrack/sensor_left"]
    assert o.lost == 0 and o.ok
    assert o.rate_hz == pytest.approx(100, abs=1)


def test_sensor_jitter_at_tick_rate_is_not_loss(tmp_path):
    """A 30 Hz sensor sampled by the 30 Hz tick with +-12 ms timestamp jitter
    delivers one new frame per tick; nothing is lost."""
    import h5py
    path = build(tmp_path)
    with h5py.File(path, "a") as f:
        ts = f["timestamps"][:]
        rng = np.random.default_rng(1)
        f["gelsight/left/timestamps"][:] = ts - 0.02 + rng.uniform(-0.012, 0.012, len(ts))
    g = by_name(check_integrity(path))["gelsight/left"]
    assert g.held == 0 and g.lost == 0 and g.ok


def test_a_long_tracker_gap_within_budget_is_a_warning(tmp_path):
    import h5py
    path = build(tmp_path, n=600)
    with h5py.File(path, "a") as f:
        t = f["optitrack/sensor_left/timestamps"][:]
        t[300:] += 0.1                                 # 100 ms dropout = 3 samples
        f["optitrack/sensor_left/timestamps"][:] = t
    r = check_integrity(path)
    o = by_name(r)["optitrack/sensor_left"]
    assert o.lost == 3 and o.ok                        # 3 of 600 = 0.5 %, within budget
    assert any("sensor_left" in w and "gap" in w for w in r.warnings)


def test_short_frame_dataset_is_flagged(tmp_path):
    import h5py
    path = build(tmp_path)
    with h5py.File(path, "a") as f:
        f["realsense/cam0/depth"].resize(100, axis=0)
    r = check_integrity(path)
    d = by_name(r)["realsense/cam0/depth"]
    assert d.count == 100 and not d.ok and not r.ok


def test_empty_tracker_is_a_warning_not_a_failure(tmp_path):
    r = check_integrity(build(tmp_path, optitrack=False))
    o = by_name(r)["optitrack/sensor_left"]
    assert o.count == 0 and o.ok
    assert any("optitrack/sensor_left" in w for w in r.warnings)
    assert r.ok


def test_table_and_dict_render(tmp_path):
    r = check_integrity(build(tmp_path, missed_tick_at=40))
    text = r.table()
    assert "timestamps" in text and "LOST" in text
    d = r.to_dict()
    assert d["ok"] is False and d["streams"][0]["name"] == "timestamps"
