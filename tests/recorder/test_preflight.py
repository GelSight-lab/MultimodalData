from types import SimpleNamespace

import pytest

from twm.recorder.config import RecorderConfig, WriterConfig
from twm.recorder.preflight import (check_disk_free, check_optitrack_fresh,
                                    check_write_bandwidth, check_writer_idle,
                                    failures, format_report,
                                    run_episode_preflight, stale_trackers)
from twm.recorder.writer import WriterStats


def usage(free_gb):
    return lambda path: SimpleNamespace(free=free_gb * 1e9)


def stats(items=0, fault=None):
    return WriterStats(items, 0, 100, 0.0, 0, 0.0, 0.0, 0, 0.0, None, None, fault)


def test_disk_free_uses_nearest_existing_parent(tmp_path):
    missing = tmp_path / "task" / "2026-09-05"
    ok = check_disk_free(missing, 50.0, usage(239.0))
    assert ok.ok and "239.0 GB" in ok.detail
    bad = check_disk_free(missing, 50.0, usage(12.5))
    assert not bad.ok and "12.5 GB" in bad.detail and "50" in bad.detail


def test_optitrack_freshness_reports_missing_and_stale_bodies():
    poses = {"sensor_left": (100.0, [0] * 7), "sensor_right": None}
    r = check_optitrack_fresh(poses, ("sensor_left", "sensor_right"), 2.0, now=101.0)
    assert not r.ok and "sensor_right: no data yet" in r.detail
    r = check_optitrack_fresh(poses, ("sensor_left",), 2.0, now=103.5)
    assert not r.ok and "3.5s old" in r.detail
    assert check_optitrack_fresh(poses, ("sensor_left",), 2.0, now=101.0).ok
    assert stale_trackers(poses, ("sensor_left", "sensor_right"), 10.0, now=111.0) == \
        ["sensor_left silent 11.0s"]        # watchdog: None is not stale


def test_writer_idle_requires_empty_queue_and_no_fault():
    assert check_writer_idle(stats()).ok
    assert not check_writer_idle(stats(items=3)).ok
    assert "disk" in check_writer_idle(stats(fault="OSError: disk")).detail


def test_write_bandwidth_measures_real_writer(tmp_path):
    r = check_write_bandwidth(tmp_path, fps=30, seconds=0.3, margin=0.01,
                              writer_config=WriterConfig())
    assert r.ok, r.detail
    assert "ticks/s" in r.detail
    assert list(tmp_path.iterdir()) == []          # temp file removed
    skipped = check_write_bandwidth(tmp_path, 30, 0.0, 1.5, WriterConfig())
    assert skipped.ok and "skipped" in skipped.detail


def test_episode_preflight_collects_every_failure(tmp_path):
    cfg = RecorderConfig(task="t", data_dir=tmp_path)
    results = run_episode_preflight(cfg, {"sensor_left": None, "sensor_right": None},
                                    stats(items=1), now=0.0, disk_usage=usage(1.0))
    names = [r.name for r in failures(results)]
    assert names == ["disk_free", "optitrack_fresh", "writer_idle"]
    report = format_report(results)
    assert "FAIL disk_free" in report and "FAIL writer_idle" in report
