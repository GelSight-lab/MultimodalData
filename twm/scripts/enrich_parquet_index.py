"""Add task-aware index columns to each episode parquet (LeRobot-aligned):
    task          (str)   e.g. "motherboard"
    task_index    (int)   0=motherboard, 1=pushT
    episode       (str)   "<date>/episode_NNN"
    episode_index (int)   0-based within task, by sorted episode key
    frame_index   (int)   = frame_idx (LeRobot name alias)

Non-destructive: keeps all existing columns. Re-writes the parquet in place
in the release staging dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STAGE = Path("/media/yxma/Disk1/twm/release")
TASK_INDEX = {"motherboard": 0, "pushT": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_INDEX))
    args = ap.parse_args()
    task = args.task
    meta_root = STAGE / task / "meta"
    parquets = sorted(meta_root.rglob("episode_*.parquet"))
    # episode_index by sorted episode key
    keys = []
    for p in parquets:
        date = p.parent.name
        keys.append(f"{date}/{p.stem}")
    ep_index = {k: i for i, k in enumerate(sorted(set(keys)))}

    for p in parquets:
        ep_key = f"{p.parent.name}/{p.stem}"
        tbl = pq.read_table(p)
        T = tbl.num_rows
        # drop if re-running
        for c in ("task", "task_index", "episode", "episode_index", "frame_index"):
            if c in tbl.column_names:
                tbl = tbl.drop([c])
        frame_idx = np.array(tbl.column("frame_idx").to_pylist(), np.int32)
        tbl = tbl.append_column("task", pa.array([task] * T))
        tbl = tbl.append_column("task_index", pa.array(np.full(T, TASK_INDEX[task], np.int32)))
        tbl = tbl.append_column("episode", pa.array([ep_key] * T))
        tbl = tbl.append_column("episode_index", pa.array(np.full(T, ep_index[ep_key], np.int32)))
        tbl = tbl.append_column("frame_index", pa.array(frame_idx))
        pq.write_table(tbl, str(p))
    print(f"[enrich] {task}: {len(parquets)} parquet enriched "
          f"(task/task_index/episode/episode_index/frame_index)", flush=True)


if __name__ == "__main__":
    main()
