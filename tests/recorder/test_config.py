from pathlib import Path

from twm.recorder.config import (DATA_DIR, FPS, RecorderConfig, WriterConfig,
                                 parse_args)


def test_parse_args_defaults_match_module_constants():
    cfg = parse_args(["--task", "pouring"])
    assert cfg.task == "pouring"
    assert cfg.data_dir == DATA_DIR
    assert cfg.fps == FPS
    assert cfg.active_sensors == ("sensor_left", "sensor_right")
    assert cfg.use_arducam is True
    assert cfg.show_projection is True
    assert cfg.writer == WriterConfig()
    assert cfg.disk.bandwidth_test_s == 2.0


def test_parse_args_maps_active_sensors_and_flags():
    cfg = parse_args(["--task", "t", "--active_sensors", "right",
                      "--no_projection", "--no_arducam", "--no_bandwidth_test",
                      "--data_dir", "/tmp/x", "--queue_seconds", "1.5",
                      "--min_free_gb", "7"])
    assert cfg.active_sensors == ("sensor_right",)
    assert cfg.show_projection is False
    assert cfg.use_arducam is False
    assert cfg.disk.bandwidth_test_s == 0.0
    assert cfg.data_dir == Path("/tmp/x")
    assert cfg.writer.queue_seconds == 1.5
    assert cfg.disk.min_free_gb == 7.0


def test_realsense_serials_and_no_optitrack_flags():
    cfg = parse_args(["--task", "t", "--realsense_serials", "143322063538,134322071848",
                      "--no_optitrack"])
    assert cfg.realsense_serials == ("143322063538", "134322071848")
    assert cfg.use_optitrack is False
    assert cfg.active_sensors == ()
    assert parse_args(["--task", "t"]).use_optitrack is True


def test_bandwidth_margin_flag():
    assert parse_args(["--task", "t", "--bandwidth_margin", "1.1"]
                      ).disk.min_bandwidth_margin == 1.1


def test_config_is_frozen_and_has_tick_dt():
    cfg = RecorderConfig(task="t", fps=25)
    assert abs(cfg.tick_dt - 0.04) < 1e-9
    try:
        cfg.fps = 10
    except Exception:
        return
    raise AssertionError("RecorderConfig must be frozen")


def test_post_init_forces_empty_active_sensors_when_optitrack_disabled():
    """Constructing RecorderConfig directly (not via config_from_namespace)
    with use_optitrack=False but a stale active_sensors must still end up
    with active_sensors == () — the watchdog and preflight must never see
    active bodies to check when OptiTrack isn't running."""
    cfg = RecorderConfig(task="t", use_optitrack=False,
                         active_sensors=("sensor_left", "sensor_right"))
    assert cfg.active_sensors == ()
    # Untouched when OptiTrack is enabled.
    cfg2 = RecorderConfig(task="t", use_optitrack=True,
                          active_sensors=("sensor_left",))
    assert cfg2.active_sensors == ("sensor_left",)


def test_raw_depth_flag_turns_off_the_sdk_alignment():
    assert parse_args(["--task", "t"]).align_depth is True
    assert parse_args(["--task", "t", "--raw_depth"]).align_depth is False
