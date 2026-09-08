from twm.recorder.monitor import health_level, health_line, minutes_remaining
from twm.recorder.writer import WriterStats


def stats(**kw):
    base = dict(queue_items=0, queue_bytes=0, capacity_bytes=1000, peak_fraction=0.0,
                bytes_written=0, write_seconds=0.0, last_batch_ms=0.0, flushes=0,
                file_mb_s=0.0, disk_free_gb=None, overloaded_since=None, fault=None)
    base.update(kw)
    return WriterStats(**base)


def test_levels():
    assert health_level(stats(), 0.25, 50.0) == "ok"
    assert health_level(stats(queue_bytes=300), 0.25, 50.0) == "warn"
    assert health_level(stats(disk_free_gb=20.0), 0.25, 50.0) == "fail"
    assert health_level(stats(fault="x"), 0.25, 50.0) == "fail"


def test_minutes_remaining_needs_a_rate():
    assert minutes_remaining(stats(disk_free_gb=60.0)) is None
    assert minutes_remaining(stats(disk_free_gb=60.0, file_mb_s=100.0)) == 10.0


def test_health_line_text():
    text, level = health_line(stats(queue_bytes=120, bytes_written=200e6,
                                    write_seconds=1.0, disk_free_gb=239.0,
                                    file_mb_s=100.0), 0.25, 50.0)
    assert text == "writer 12% | 200 MB/s | disk 239 GB (~40 min) | OK"
    assert level == "ok"
    text, _ = health_line(stats(fault="OSError: No space"), 0.25, 50.0)
    assert text.endswith("| FAULT: OSError: No space")


def test_health_line_names_stale_sensors_and_warns():
    sensors = {"gelsight_left": {"stale_s": 3.2, "restarting": True, "restarts": 2},
               "gelsight_right": {"stale_s": 0.03, "restarting": False, "restarts": 0},
               "arducam_cam0": {"stale_s": None, "restarting": False, "restarts": 0}}
    text, level = health_line(stats(bytes_written=100e6, write_seconds=1.0), 0.25, 50.0, sensors=sensors)
    assert "gelsight_left STALE 3.2s (restarting, 2 restarts)" in text
    assert "gelsight_right" not in text and level == "warn"
    text, level = health_line(stats(), 0.25, 50.0, sensors={"gelsight_left": {"stale_s": 0.1, "restarting": False, "restarts": 0}})
    assert "STALE" not in text and level == "ok"
