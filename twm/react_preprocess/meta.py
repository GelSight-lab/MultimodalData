"""Per-frame parquet: poses, contact metrics, and tactile validity flags.

Row ``i`` corresponds to frame ``i`` of every MP4 in the same episode.

New in this version: ``tactile_{left,right}_is_new``. A tactile row is only a
fresh sensor reading when that flag is True — the GelSight Mini tops out at
18.75 fps while rows are written at 30 Hz, so some duplication is unavoidable,
and legacy recordings duplicated far more (see ``contact.NewFrameTracker``).
Train tactile dynamics on the flagged rows, not on all of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# THE task -> task_index mapping. APPEND ONLY: these ints are published inside
# every parquet, so renumbering a task silently relabels every copy already
# downloaded, with nothing on the reader's side to notice.
#
# It lived in four places. Two were stale — `dataset_prep` had no `rope` and
# read the map with `.get(task, 0)`, so a rope episode published through it was
# stamped task_index=0, which is motherboard.
TASK_INDEX = {"motherboard": 0, "pushT": 1, "rope": 2, "toy": 3}


def task_index(task: str) -> int:
    """`task`'s published index, or KeyError.

    Raises rather than defaulting: a default is how rope became motherboard,
    and a wrong int here is indistinguishable from a right one downstream.
    """
    try:
        return TASK_INDEX[task]
    except KeyError:
        raise KeyError(
            f"{task!r} has no published task_index. Add it to TASK_INDEX with "
            f"the NEXT free integer — never reuse or renumber, the existing "
            f"ints are already inside published parquets.") from None


# Columns added after the initial release, in the order they should appear.
TACTILE_FLAG_COLUMNS = ("tactile_left_is_new", "tactile_right_is_new")


def build_table(source, tactile: dict, object_pose: np.ndarray | None = None) -> pa.Table:
    """Assemble the per-frame table for one episode.

    source: EpisodeSource; tactile: {side: TactileResult}
    """
    T = source.T
    left, right = tactile["left"], tactile["right"]
    cols = {
        "frame_idx": np.arange(T, dtype=np.int32),
        "timestamp": source.trimmed_cam_ts.astype(np.float64),
        "sensor_left_pose": list(source.pose_left),
        "sensor_right_pose": list(source.pose_right),
        "tactile_left_intensity": left.intensity,
        "tactile_left_area": left.area,
        "tactile_left_mixed": left.mixed,
        "tactile_right_intensity": right.intensity,
        "tactile_right_area": right.area,
        "tactile_right_mixed": right.mixed,
        "tactile_left_is_new": left.is_new,
        "tactile_right_is_new": right.is_new,
        "source_h5_frame": (np.arange(T) + source.trim).astype(np.int32),
    }
    if object_pose is not None:
        cols["object_pose"] = list(object_pose)
    return pa.table(cols)


def write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def add_index_columns(table: pa.Table, task: str, task_index: int,
                      episode: str, episode_index: int) -> pa.Table:
    """Attach the LeRobot-style task/episode/frame index columns."""
    n = table.num_rows
    additions = {
        "task": pa.array([task] * n, pa.string()),
        "task_index": pa.array(np.full(n, task_index, np.int64)),
        "episode": pa.array([episode] * n, pa.string()),
        "episode_index": pa.array(np.full(n, episode_index, np.int64)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
    }
    for name, arr in additions.items():
        if name in table.column_names:
            table = table.set_column(table.schema.get_field_index(name), name, arr)
        else:
            table = table.append_column(name, arr)
    return table


def backfill_is_new(table: pa.Table, left_is_new: np.ndarray,
                    right_is_new: np.ndarray) -> pa.Table:
    """Add/replace the tactile validity flags on an existing parquet."""
    for name, values in zip(TACTILE_FLAG_COLUMNS, (left_is_new, right_is_new)):
        arr = pa.array(np.asarray(values, dtype=bool))
        if name in table.column_names:
            table = table.set_column(table.schema.get_field_index(name), name, arr)
        else:
            table = table.append_column(name, arr)
    return table
