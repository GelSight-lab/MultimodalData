"""Add the manipulated-object OptiTrack pose (`optitrack/motherboard` body)
as an `object_pose` column to each episode's release parquet.

Only tasks/episodes where the object body has samples get the column
(motherboard: all 32; pushT: object body empty -> object_pose = NaNs and
episodes.jsonl object_tracked=false). The 2026-05-19 world-frame offset is
applied to object_pose too, matching sensor poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

H5_ROOTS = {"motherboard": Path("/media/yxma/Disk1/twm/data/motherboard"),
            "pushT": Path("/media/yxma/Disk1/twm/data/pushT")}
STAGE = Path("/media/yxma/Disk1/twm/release")
WORLD_OFFSET = {("motherboard", "2026-05-19"): (0.23, 0.0, 0.175)}


def cam_align(cam_ts, ot_ts, ot_pose):
    if len(ot_ts) == 0:
        return None
    idx = np.clip(np.searchsorted(ot_ts, cam_ts), 0, len(ot_ts) - 1)
    idxm = np.clip(idx - 1, 0, len(ot_ts) - 1)
    pm = np.abs(ot_ts[idxm] - cam_ts) < np.abs(ot_ts[idx] - cam_ts)
    return ot_pose[np.where(pm, idxm, idx)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(H5_ROOTS))
    args = ap.parse_args()
    task = args.task
    meta_root = STAGE / task / "meta"
    eps_jsonl = STAGE / task / "episodes.jsonl"
    rows = [json.loads(l) for l in eps_jsonl.read_text().splitlines()]
    rows_by_key = {r["episode"]: r for r in rows}

    n_tracked = 0
    for det in sorted(meta_root.rglob("*._detect.pt")):
        date = det.parent.name
        ep_stem = det.name.replace("._detect.pt", "")
        ep_key = f"{date}/{ep_stem}"
        pq_path = meta_root / date / f"{ep_stem}.parquet"
        d = torch.load(str(det), weights_only=False, map_location="cpu")
        trim = int(d["_contact_meta"]["trim_offset"])
        T = int(d["timestamps"].shape[0])
        off = WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))

        h5 = H5_ROOTS[task] / date / f"{ep_stem}.h5"
        obj = None
        try:
            with h5py.File(str(h5), "r") as f:
                ot_ts = f["optitrack/motherboard/timestamps"][:]
                ot_pose = f["optitrack/motherboard/pose"][:]
                cam_ts = f["timestamps"][trim:trim + T]
            if len(ot_ts) > 0:
                obj = cam_align(cam_ts, ot_ts, ot_pose)
                obj = obj.copy()
                obj[:, 0] += off[0]; obj[:, 1] += off[1]; obj[:, 2] += off[2]
        except Exception:
            obj = None

        tbl = pq.read_table(pq_path)
        if "object_pose" in tbl.column_names:
            tbl = tbl.drop(["object_pose"])
        if obj is not None:
            col = list(obj)
            n_tracked += 1
            tracked = True
        else:
            col = [[float("nan")] * 7 for _ in range(T)]
            tracked = False
        tbl = tbl.append_column("object_pose", pa.array(col))
        pq.write_table(tbl, str(pq_path))
        if ep_key in rows_by_key:
            rows_by_key[ep_key]["object_tracked"] = tracked
        print(f"  {ep_key}: object_tracked={tracked} (T={T})", flush=True)

    with open(eps_jsonl, "w") as f:
        for r in sorted(rows, key=lambda r: r["episode"]):
            f.write(json.dumps(r) + "\n")
    print(f"[object_pose] {task}: {n_tracked}/{len(rows)} episodes have object tracking", flush=True)


if __name__ == "__main__":
    main()
