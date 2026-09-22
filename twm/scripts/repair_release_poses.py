"""Repair short OptiTrack excursions in a release tree's poses.

    python twm/scripts/repair_release_poses.py [--tree ...] [--since D] [--apply]

OptiTrack occasionally solves the wrong marker correspondence for a frame or
three. Measured across the published window on 2026-09-17: 681 of 1,004,118
frame-sides (0.068%) in 67 segments, the worst rotating 168.7 degrees out and
back while its neighbours sit 1.0 degrees apart. `detect_pose_teleports` finds
none of them -- it requires translation AND rotation, and these are rotation
only. See `react_preprocess.pose_repair`.

`force_<side>_target_pose` is recomputed, never left as it was: it is the pose
displaced along `R(q) @ gel_axis`, so both terms depend on the quaternion this
script just changed. Leaving it is the same defect one step downstream, where
it is harder to see.

Force values are untouched. They are measured from gel images and have nothing
to do with where the rig thought the sensor was.

Dry by default: rewriting published poses should have to be asked for twice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.react_preprocess.pose_repair import find_excursions, repair_poses  # noqa: E402

SIDES = ("left", "right")


def rows_for(uncut_source_frames, segment_source_frames):
    """Row indices into the uncut episode for a segment's frames.

    `source_h5_frame` is the RAW H5 frame number. It equals the row index only
    when trim == 0, which happens to hold for 45 of 46 motherboard episodes --
    so using it directly as an index looked correct everywhere until
    2026-05-11/episode_017, whose uncut frames run 19228..33738.
    """
    u = np.asarray(uncut_source_frames, np.int64)
    s = np.asarray(segment_source_frames, np.int64)
    order = np.argsort(u)
    pos = np.searchsorted(u[order], s)
    ok = pos < len(u)
    idx = np.where(ok, order[np.clip(pos, 0, len(u) - 1)], 0)
    bad = ~ok | (u[idx] != s)
    if bad.any():
        raise ValueError(
            f"source frame {s[bad][:5].tolist()} not in the uncut episode "
            f"({u.min()}..{u.max()}) -- refusing to align by position")
    return idx


def _list_col(a: np.ndarray) -> pa.ListArray:
    T, D = a.shape
    off = pa.array(np.arange(T + 1, dtype=np.int32) * D, type=pa.int32())
    return pa.ListArray.from_arrays(
        off, pa.array(np.ascontiguousarray(a, np.float64).ravel(),
                      type=pa.float64()))


def repair_parquet(path: Path, task: str) -> dict:
    """Repair one parquet in place. Returns what was changed."""
    from scipy.spatial.transform import Rotation as R
    from twm.force_recovery.dexforce import gel_axis

    t = pq.read_table(str(path))
    names, cols = list(t.column_names), list(t.columns)
    spans_all, n_frames, touched = {}, 0, {}

    for side in SIDES:
        pcol = f"sensor_{side}_pose"
        if pcol not in names:
            continue
        pose = np.array(t[pcol].to_pylist(), float)
        # `mark` is what says a frame CHANGED. Counting excursion spans alone
        # missed every flicker repair -- those frames are picked individually,
        # not as a span -- so a parquet whose only damage was flicker was
        # reported as "0 frames" and never written.
        fixed, spans, interp, skipped = repair_poses(pose, mark=True)
        if not interp.any():
            continue
        spans_all[side] = {"spans": spans, "flicker": int(interp.sum()) - sum(n for _, n in spans),
                           "skipped": skipped}
        n_frames += int(interp.sum())
        touched[side] = interp
        cols[names.index(pcol)] = _list_col(fixed)

        # The target is the pose displaced along R(q) @ gel_axis; both terms
        # moved, so it is recomputed rather than carried forward.
        tcol, pencol = f"force_{side}_target_pose", f"force_{side}_penetration_mm"
        if tcol in names and pencol in names:
            pen = np.asarray(t[pencol].to_numpy(), float)
            nh = np.einsum("nij,j->ni",
                           R.from_quat(fixed[:, 3:]).as_matrix(),
                           gel_axis(task, side))
            tgt = fixed.copy()
            move = pen > 0
            tgt[move, :3] += (pen[move, None] / 1000.0) * nh[move]
            cols[names.index(tcol)] = _list_col(tgt)

    if not spans_all:
        return {"frames": 0, "spans": {}}
    # pose_interpolated: the union over sides, so a reader sees which ROWS
    # carry an invented pose without having to know which sensor it was.
    if touched:
        import pyarrow as _pa
        u = np.zeros(t.num_rows, bool)
        for v in touched.values():
            u |= v
        if "pose_interpolated" in names:
            cols[names.index("pose_interpolated")] = _pa.array(u)
        else:
            names.append("pose_interpolated"); cols.append(_pa.array(u))
        out = pa.table(cols, names=names).replace_schema_metadata(
            dict(t.schema.metadata or {}))
        pq.write_table(out, str(path), compression="zstd")
    return {"frames": n_frames, "spans": spans_all}


def main() -> int:
    import twm.pipeline_stages as PS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", action="append", default=None,
                    help="release tree to repair (repeatable). Default: the "
                         "cut tree, which is what ships.")
    ap.add_argument("--task", action="append", default=None,
                    help="only this task (repeatable). Default: all.")
    ap.add_argument("--since", default=PS.SCOPE_SINCE,
                    help="skip dates before this. Pass '' for every date, "
                         "including the pre-2026-09 archives.")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    trees = [Path(t) for t in (a.tree or [PS.RELEASE_CUT])]

    total_seg = total_fr = 0
    record = {}
    for tree in trees:
        for f in sorted(tree.glob("*/meta/*/*.parquet")):
            date = f.parent.name
            if a.since and date < a.since:
                continue
            task = f.parts[-4]
            if a.task and task not in a.task:
                continue
            if not a.apply:
                pose_cols = [c for c in pq.read_schema(f).names
                             if c.startswith("sensor_") and c.endswith("_pose")]
                n = 0
                tb = pq.read_table(str(f), columns=pose_cols)
                for c in pose_cols:
                    n += sum(k for _, k in
                             find_excursions(np.array(tb[c].to_pylist(), float)))
                if n:
                    total_seg += 1; total_fr += n
                    print(f"  {task}/{date}/{f.stem}: {n} frame(s)")
                continue
            r = repair_parquet(f, task)
            if r["frames"]:
                total_seg += 1; total_fr += r["frames"]
                record[f"{task}/{date}/{f.stem}"] = r["spans"]
                print(f"  {task}/{date}/{f.stem}: repaired {r['frames']} frame(s)")

    print(f"\n{total_fr} frame(s) across {total_seg} segment(s)")
    if not a.apply:
        print("(dry run — pass --apply to rewrite the poses)")
        return 0
    for tree in trees:
        p = Path(tree) / "pose_repair.json"
        prior = {}
        if p.is_file():
            try: prior = json.loads(p.read_text())
            except ValueError: prior = {}
        prior.update({k: v for k, v in record.items()})
        p.write_text(json.dumps(prior, indent=1, sort_keys=True))
        print(f"recorded in {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
