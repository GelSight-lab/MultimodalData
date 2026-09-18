import numpy as np
import pytest

from twm.force_recovery.run_episode_mp4 import evaluate_fresh_rows


def test_mp4_rows_map_directly_and_duplicate_rows_reuse_force():
    frames = [np.full((2, 2, 3), value, np.uint8) for value in (1, 2, 3, 4)]
    calls = []

    def evaluate(frame):
        calls.append(int(frame[0, 0, 0]))
        value = float(frame[0, 0, 0])
        return value, value + 10, value + 20, value + 30

    out, source = evaluate_fresh_rows(frames, np.array([True, False, True, False]), evaluate)

    assert calls == [1, 3]
    np.testing.assert_array_equal(source, [0, 0, 2, 2])
    np.testing.assert_array_equal(out["force_normal_n"], [1, 1, 3, 3])
    np.testing.assert_array_equal(out["max_depth_mm"], [31, 31, 33, 33])


def test_mp4_decode_must_match_parquet_row_count():
    frames = [np.zeros((2, 2, 3), np.uint8)]
    with pytest.raises(ValueError, match="1 frames.*2 parquet rows"):
        evaluate_fresh_rows(frames, np.array([True, True]), lambda frame: (0, 0, 0, 0))
