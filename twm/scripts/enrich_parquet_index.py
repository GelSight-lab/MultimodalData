"""Add the LeRobot index columns to every parquet in a task's staging tree.

    task          (str)   e.g. "motherboard"
    task_index    (int64) 0=motherboard, 1=pushT, 2=rope
    episode       (str)   "<date>/episode_NNN"
    episode_index (int64) 0-based within the task, by sorted episode key
    frame_index   (int64) row position within the episode

Non-destructive: every other column is kept, and re-running replaces the five
rather than appending duplicates.

This used to write its own copy of the columns as **int32**, while every
published parquet carries int64 -- two writers of one schema, disagreeing.
Both the shape and the numbering now come from one place each:
`react_preprocess.meta.add_index_columns` and `backfill_index_columns`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_index_columns import TASK_INDEX, episode_indices  # noqa: E402

from twm.react_preprocess.meta import add_index_columns  # noqa: E402

STAGE = Path("/media/yxma/Disk1/twm/release")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_INDEX))
    ap.add_argument("--stage", default=str(STAGE))
    a = ap.parse_args()
    meta_root = Path(a.stage) / a.task / "meta"
    parquets = sorted(meta_root.rglob("episode_*.parquet"))
    idx = episode_indices(f"{p.parent.name}/{p.stem}" for p in parquets)
    for p in parquets:
        key = f"{p.parent.name}/{p.stem}"
        t = add_index_columns(pq.read_table(str(p)), a.task,
                              TASK_INDEX[a.task], key, idx[key])
        tmp = p.with_suffix(".parquet.tmp")
        pq.write_table(t, str(tmp))
        os.replace(tmp, p)
    print(f"[enrich] {a.task}: {len(parquets)} parquet enriched "
          f"(task/task_index/episode/episode_index/frame_index)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
