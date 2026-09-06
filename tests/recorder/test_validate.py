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
