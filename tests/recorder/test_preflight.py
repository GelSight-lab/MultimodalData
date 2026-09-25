from types import SimpleNamespace

import pytest

from twm.recorder.config import DiskConfig, RecorderConfig, WriterConfig
from twm.recorder.preflight import (check_disk_free, check_optitrack_fresh,
                                    check_write_bandwidth, check_writer_idle,
                                    failures, format_report,
                                    run_episode_preflight, run_startup_preflight,
                                    stale_trackers)
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


def test_optitrack_freshness_with_no_active_bodies_reports_disabled():
    r = check_optitrack_fresh({}, (), 2.0, now=0.0)
    assert r.ok
    assert r.detail == "OptiTrack disabled (no active bodies)"


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


def test_write_bandwidth_reports_failure_instead_of_raising(tmp_path, monkeypatch):
    def boom(f, ticks):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("twm.recorder.preflight.append_ticks", boom)
    r = check_write_bandwidth(tmp_path, fps=30, seconds=0.3, margin=0.01,
                              writer_config=WriterConfig())
    assert not r.ok
    assert "disk full (simulated)" in r.detail
    assert list(tmp_path.iterdir()) == []          # temp file removed


def test_episode_preflight_collects_every_failure(tmp_path):
    cfg = RecorderConfig(task="t", data_dir=tmp_path)
    results = run_episode_preflight(cfg, {"sensor_left": None, "sensor_right": None},
                                    stats(items=1), now=0.0, disk_usage=usage(1.0))
    names = [r.name for r in failures(results)]
    assert names == ["disk_free", "optitrack_fresh", "writer_idle"]
    report = format_report(results)
    assert "FAIL disk_free" in report and "FAIL writer_idle" in report


def test_startup_preflight_skips_bandwidth_test_when_disk_check_fails(tmp_path):
    """The bandwidth self-test writes real data (~fps * bandwidth_test_s
    ticks, hundreds of MB at full-rig size) to `data_dir`. Running it after
    disk_free has already failed can itself blow past a nearly-full disk;
    it must be skipped, not attempted, and nothing may be written."""
    cfg = RecorderConfig(task="t", data_dir=tmp_path,
                         disk=DiskConfig(min_free_gb=50.0, bandwidth_test_s=0.2))
    results = run_startup_preflight(cfg, disk_usage=usage(1.0))
    assert [r.name for r in results] == ["disk_free", "write_bandwidth"]
    assert not results[0].ok
    assert not results[1].ok
    assert "skipped" in results[1].detail
    assert list(tmp_path.iterdir()) == []


def test_startup_preflight_measures_the_configured_realsense_count(tmp_path, monkeypatch):
    """The self-test must build its synthetic file and ticks for the rig
    actually configured (`config.realsense_serials`), not always the
    legacy three — otherwise a 2-camera config's synthetic ticks would
    mismatch a 3-group synthetic file and the self-test would measure the
    wrong rig (or crash on the mismatch)."""
    import twm.recorder.preflight as preflight_mod

    captured = {}
    real_create = preflight_mod.create_episode_file

    def spy(*args, **kwargs):
        captured["n_realsense"] = kwargs.get("n_realsense")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(preflight_mod, "create_episode_file", spy)
    cfg = RecorderConfig(task="t", data_dir=tmp_path, realsense_serials=("A", "B"),
                         disk=DiskConfig(min_free_gb=0.0, bandwidth_test_s=0.2))
    results = run_startup_preflight(cfg, disk_usage=usage(1e6))
    bw = results[1]
    assert bw.ok, bw.detail
    assert captured["n_realsense"] == 2
    # n_arducam defaults to 0 here, so the arducam scale factor is 1 and the
    # "need" figure (fps × margin) does not depend on the RealSense count.
    assert f"need {cfg.fps * cfg.disk.min_bandwidth_margin:.1f}" in bw.detail
