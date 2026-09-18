import numpy as np
import pyarrow as pa

from twm.scripts.export_old_motherboard_actions import build_action_episode


def _pose(x):
    return [float(x), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _tables(n=4):
    raw = pa.table({
        "frame_idx": pa.array(range(n), type=pa.int64()),
        "sensor_left_pose": pa.array([_pose(i) for i in range(n)]),
        "sensor_right_pose": pa.array([_pose(10 + i) for i in range(n)]),
        "tag": pa.array(["raw"] * n),
    })
    candidate = raw
    candidate = candidate.set_column(
        candidate.schema.get_field_index("sensor_left_pose"),
        "sensor_left_pose", pa.array([_pose(100 + i) for i in range(n)]))
    candidate = candidate.set_column(
        candidate.schema.get_field_index("sensor_right_pose"),
        "sensor_right_pose", pa.array([_pose(200 + i) for i in range(n)]))
    left_conf = ([0, 3, 2, 1] + [0] * n)[:n]
    right_conf = ([0, 0, 3, 0] + [0] * n)[:n]
    candidate = candidate.append_column(
        "pose_left_repair_confidence", pa.array(left_conf, type=pa.uint8()))
    candidate = candidate.append_column(
        "pose_right_repair_confidence", pa.array(right_conf, type=pa.uint8()))
    candidate = candidate.append_column(
        "pose_left_valid", pa.array(([True, True, False, False] + [True] * n)[:n]))
    candidate = candidate.append_column(
        "pose_right_valid", pa.array([True] * n))
    return raw, candidate


def _native(valid, repaired):
    return {
        "valid": np.asarray(valid, bool),
        "repaired": np.asarray(repaired, bool),
    }


def test_builds_t_row_lerobot_action_without_changing_raw_columns():
    raw, candidate = _tables()
    out = build_action_episode(
        raw, candidate,
        _native([True, False, True], [True, True, False]),
        _native([True, True, True], [False, True, False]),
        [],
    )

    assert out.num_rows == 4
    assert out.schema.field("action").type == pa.list_(pa.float32(), 14)
    assert out["action"][0].as_py() == [_pose(101), _pose(201)][0] + _pose(201)
    assert out["action_valid_left"].to_pylist() == [True, False, True, False]
    assert out["action_valid_right"].to_pylist() == [True, True, True, False]
    assert out["action_repaired_left"].to_pylist() == [True, True, False, False]
    assert out["action_repair_confidence_left"].to_pylist() == [3, 3, 2, 0]
    assert out["sensor_left_pose"].equals(raw["sensor_left_pose"])
    assert out["sensor_right_pose"].equals(raw["sensor_right_pose"])
    assert out["tag"].equals(raw["tag"])
    assert out["sensor_left_pose_repaired"].to_pylist() == [
        _pose(100 + i) for i in range(4)]


def test_terminal_row_is_not_labeled_as_lost_track():
    raw, candidate = _tables()
    out = build_action_episode(
        raw, candidate,
        _native([True, True, True], [False, False, False]),
        _native([True, True, True], [False, False, False]),
        [],
    )

    assert out["action_valid_left"][-1].as_py() is False
    assert out["action_lost_track_left"][-1].as_py() is False


def test_unresolved_transition_is_invalid_and_labeled_lost_track():
    raw, candidate = _tables()
    out = build_action_episode(
        raw, candidate,
        _native([True, False, True], [False, False, False]),
        _native([True, True, True], [False, False, False]),
        [{"side": "left", "start": 1, "end": 2, "confidence": "LOW"}],
    )

    assert out["action_valid_left"].to_pylist() == [True, False, True, False]
    assert out["action_lost_track_left"].to_pylist() == [False, True, False, False]


def test_only_events_longer_than_sixty_frames_are_long_lost_track():
    raw, candidate = _tables(65)
    native = _native([False] * 64, [False] * 64)
    out = build_action_episode(
        raw, candidate, native, native,
        [
            {"side": "left", "start": 0, "end": 59, "confidence": "LOW"},
            {"side": "right", "start": 2, "end": 62, "confidence": "LOW"},
        ],
    )

    assert not any(out["action_long_lost_track_left"].to_pylist())
    marked = out["action_long_lost_track_right"].to_pylist()
    assert marked[1:63] == [True] * 62
    assert marked[0] is False and marked[63] is False and marked[64] is False
