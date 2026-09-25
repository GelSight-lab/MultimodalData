import numpy as np
import pytest

from twm.recorder.frames import Tick, full_rig_tick_nbytes, synthetic_tick


def test_nbytes_counts_every_array_and_optitrack_samples():
    t = synthetic_tick(1.0, seed=0, n_realsense=3, n_gelsight=2, n_arducam=2)
    per_color = 480 * 640 * 3
    per_depth = 480 * 640 * 2
    assert t.nbytes() == 3 * per_color + 3 * per_depth + 2 * per_color + 2 * per_color
    assert full_rig_tick_nbytes(3, 2, 2) == t.nbytes()
    with_ot = Tick(1.0, optitrack={"motherboard": [(1.0, [0] * 7)] * 4})
    assert with_ot.nbytes() == 4 * Tick.OPTITRACK_SAMPLE_BYTES


def test_tick_rejects_mismatched_timestamp_lengths():
    frame = np.zeros((480, 640, 3), np.uint8)
    with pytest.raises(ValueError):
        Tick(1.0, gelsight=(frame, frame), gelsight_ts=(1.0,))
    with pytest.raises(ValueError):
        Tick(1.0, arducam=(frame,), arducam_ts=())


def test_synthetic_tick_is_deterministic_and_not_flat():
    a = synthetic_tick(5.0, seed=3)
    b = synthetic_tick(5.0, seed=3)
    np.testing.assert_array_equal(a.color[0], b.color[0])
    assert a.color[0].std() > 5
    assert a.gelsight_ts == (5.0, 5.0)
