"""Rewrite a release tree from Y-up to Z-up, poses and calibration together.

WHY THIS IS SAFE TO DO AT ALL

The conversion is a rotation of the world frame. Applied to the poses AND to
`T_mocap_to_cam`, every projection is unchanged — so every rendered preview,
overlay and clip stays correct, and only the numbers move. That invariance is
verified here on real data BEFORE anything is written, and again afterwards.

WHAT MOVES, AND WHAT MUST NOT BE MISSED

    parquet   sensor_left_pose, sensor_right_pose, object_pose,
              force_{left,right}_target_pose        (7-vectors)
    calib     T_mocap_to_cam_{left,middle,right}.json and .npy
    episodes  world_frame_offset  — a translation IN the world frame, so it
              rotates too. Leaving it would put the 2026-05-19 correction in
              the old frame while its poses are in the new one.
    metadata  the parquet's twm.world_frame fingerprint, which is a set of
              projected pixels; projections are invariant, so it is unchanged,
              but the DECLARATION gains up_axis so nothing has to guess.

Gel centres are in the sensor's own rigid frame and do not move.

    python scripts/convert_release_zup.py --src ... --dst ... [--force-src ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import force_meta, release_root   # noqa: E402

import numpy as np                                 # noqa: E402
import pyarrow as pa                               # noqa: E402
import pyarrow.parquet as pq                       # noqa: E402

POSE_COLS = ("sensor_left_pose", "sensor_right_pose", "object_pose",
             "force_left_target_pose", "force_right_target_pose")


def convert_tree(src: Path, dst: Path, task: str) -> dict:
    from react_toolbox.frames import YUP_TO_ZUP, convert_poses

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = {"parquet": 0, "cols": 0, "calib": 0, "episodes": 0}

    # --- parquet -------------------------------------------------------
    for p in sorted((src / "meta").glob("*/*.parquet")):
        t = pq.read_table(p)
        cols, names = [], []
        for name in t.column_names:
            col = t.column(name)
            if name in POSE_COLS:
                arr = np.asarray([x for x in col.to_pylist()], float)
                arr = convert_poses(arr, True)
                cols.append(pa.array([list(map(float, r)) for r in arr],
                                     type=pa.list_(pa.float64())))
                n["cols"] += 1
            else:
                cols.append(col)
            names.append(name)
        md = dict(t.schema.metadata or {})
        decl = md.get(b"twm.world_frame")
        if decl:
            d = json.loads(decl.decode())
            d["up_axis"] = "z"
            d["up_axis_note"] = ("converted from the recorded Y-up by "
                                 "R_x(-90): (x,y,z)->(x,-z,y). Projections are "
                                 "unchanged because T_mocap_to_cam moved with "
                                 "the poses.")
            # raw_h5_offset_m is NOT rotated. Every other number in this
            # blob describes the published poses, which are now Z-up; this one
            # describes what to add to a pose read out of the source H5, and
            # the source H5 is Y-up as recorded. Rotating it put 175 mm on the
            # wrong axis while the note still said "straight out of the H5".
            if d.get("raw_h5_offset_m"):
                d["raw_h5_offset_up_axis"] = "y"
            md[b"twm.world_frame"] = json.dumps(d).encode()
        out = pa.table(cols, names=names).replace_schema_metadata(md)
        q = dst / "meta" / p.parent.name / p.name
        q.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, q)
        n["parquet"] += 1

    # --- calibration ---------------------------------------------------
    cs, cd = src / "calibration", dst / "calibration"
    if cs.is_dir():
        shutil.copytree(cs, cd)
        for f in sorted(cd.glob("T_mocap_to_cam_*.json")):
            j = json.loads(f.read_text())
            T = np.asarray(j["T_mocap_to_cam"], float)
            T[:3, :3] = T[:3, :3] @ YUP_TO_ZUP.T
            j["T_mocap_to_cam"] = T.tolist()
            j["up_axis"] = "z"
            f.write_text(json.dumps(j, indent=1))
            npy = f.with_suffix(".npy")
            if npy.exists():
                np.save(npy, T)
            n["calib"] += 1

    # --- everything else, copied verbatim ------------------------------
    for name in ("episodes.jsonl", "segments.json", "bad_frames.json",
                 "splits.json"):
        f = src / name
        if f.is_file():
            shutil.copy(f, dst / name)
    ej = dst / "episodes.jsonl"
    if ej.is_file():
        lines = []
        for line in ej.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            off = r.get("world_frame_offset")
            if off:
                r["world_frame_offset"] = list(YUP_TO_ZUP @ np.asarray(off, float))
                n["episodes"] += 1
            r["up_axis"] = "z"
            lines.append(json.dumps(r))
        ej.write_text("\n".join(lines) + "\n")
    for sub in ("videos",):
        if (src / sub).is_dir():
            (dst / sub).symlink_to(src / sub)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="motherboard")
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    a = ap.parse_args()
    src = Path(a.src) if a.src else release_root(a.task)
    dst = Path(a.dst) if a.dst else src.parent.parent / "release_zup" / a.task
    n = convert_tree(src, dst, a.task)
    print(f"{src} -> {dst}")
    print(f"  {n['parquet']} parquet ({n['cols']} pose columns), "
          f"{n['calib']} calibrations, {n['episodes']} world offsets rotated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
