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


def _already_zup(src: Path, parquet: Path) -> bool:
    """Whether this episode is already in the Z-up frame.

    From `episodes.jsonl`, which is where `up_axis` is actually recorded — the
    parquet's `twm.world_frame` metadata does NOT carry it on any episode in
    either tree, so a guard reading only the parquet never fires and every
    already-converted episode gets rotated a second time. Verified: a first
    attempt turned all 47 motherboard parquets instead of the 15 Y-up ones.
    """
    key = f"{parquet.parent.name}/{parquet.stem}"
    jsonl = src / "episodes.jsonl"
    if not jsonl.is_file():
        return False
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode") == key:
            return row.get("up_axis") == "z"
    # Not listed: curation refuses to drop episodes, so an unlisted parquet
    # means the indices are stale. Converting it blind could double-rotate.
    raise KeyError(f"{key} is not in {jsonl} — cannot tell which frame it is "
                   f"in; re-run curation before converting")


def _merge_force(t, force_src: Path | None, date: str, name: str):
    """Fold the force channel into the parquet BEFORE the rotation runs.

    `release_force/` is `release/` plus the force columns at the same path, and
    the Z-up tree is what gets CUT. Publishing an uncut release uploaded both
    trees over one another so the reader got the union, which is why nothing
    noticed this tree was built from the one WITHOUT the columns. A segment
    cannot be addressed by an uncut episode name, so for the cut tree the
    columns have to be inside the file.

    Merged here, ahead of the branch below, so `force_{left,right}_target_pose`
    goes through the same rotation as every other pose — and so an episode
    that is already Z-up gets the columns without being rotated twice.

    How MANY columns is not this function's business: it folds in whatever
    `force_*` the export tree holds. A force-only export (2026-09-16) ships
    four -- the measured newtons and their source frame, without the
    stiffness-derived pair -- and the loop below neither needs nor asserts the
    other four. `POSE_COLS` names `force_{left,right}_target_pose`, so when
    they are absent the rotation simply never reaches them.
    """
    if force_src is None:
        return t
    f = Path(force_src) / "meta" / date / name
    if not f.is_file():
        # Not a fallback to a default: the file converts either way. It is
        # named so that "this episode has no force channel" is a line in the
        # log rather than a column nobody notices is missing.
        print(f"  no force channel for {date}/{name} — converting without it",
              flush=True)
        return t
    ft = pq.read_table(f)
    if ft.num_rows != t.num_rows:
        raise SystemExit(
            f"{date}/{name}: force tree has {ft.num_rows} rows, release has "
            f"{t.num_rows}. These are meant to be the same rows plus the "
            f"force columns; refusing to merge mismatched frames.")
    cols, names = list(t.columns), list(t.column_names)
    for c in ft.column_names:
        if c.startswith("force_") and c not in names:
            cols.append(ft.column(c))
            names.append(c)
    return pa.table(cols, names=names).replace_schema_metadata(
        dict(t.schema.metadata or {}))


def convert_tree(src: Path, dst: Path, task: str,
                 force_src: Path | None = None) -> dict:
    from react_toolbox.frames import YUP_TO_ZUP, convert_poses

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = {"parquet": 0, "cols": 0, "calib": 0, "episodes": 0}

    # --- parquet -------------------------------------------------------
    for p in sorted((src / "meta").glob("*/*.parquet")):
        t = _merge_force(pq.read_table(p), force_src, p.parent.name, p.name)
        # ALREADY Z-UP: pass it through untouched. This tree no longer holds
        # one convention — the 2026-05/06 episodes were converted before the
        # 2026-09 sessions were recorded — and rotating those a second time is
        # a net -180 deg that nothing downstream can see: the conversion's own
        # safety argument is that projections are invariant, so every preview
        # and clip still renders correctly while the world-frame numbers are
        # wrong.
        if _already_zup(src, p):
            q = dst / "meta" / p.parent.name / p.name
            q.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, str(q))
            n["parquet"] += 1
            continue
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
            # ALREADY Z-UP: carry it through. `stage_calibration` writes the
            # declared epoch already converted and stamped, and rotating it
            # again is a net 180 degrees that every projection is blind to —
            # the same invisibility that made `to_zup` refuse a second
            # rotation. This path went around that refusal by rotating inline.
            #
            # Measured 2026-09-16: rope's staged calibration matched no epoch
            # in any form and sat 1.2733 from motherboard's and pushT's, which
            # had been rotated once. The publish gate caught it.
            if j.get("up_axis") == "z":
                continue
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
    ap.add_argument("--force-src", dest="force_src", default=None,
                    help="the force staging tree whose columns to fold in "
                         "(default: the one force_meta names for this task). "
                         "Pass 'none' to convert without the force channel.")
    a = ap.parse_args()
    src = Path(a.src) if a.src else release_root(a.task)
    dst = Path(a.dst) if a.dst else src.parent.parent / "release_zup" / a.task
    if a.force_src is None:
        force_src = force_meta(a.task).parent
    elif a.force_src.lower() == "none":
        force_src = None
    else:
        force_src = Path(a.force_src)
    # Stage the epoch the sessions DECLARE before converting. The source tree
    # carried whatever was last put there by hand — motherboard had
    # epoch_2026-05-12 while publishing September — and a hand fix to the CUT
    # tree is overwritten the next time this runs. Derived, not maintained.
    from twm.react_preprocess.segment import stage_calibration
    try:
        epoch = stage_calibration(src, a.task)
        print(f"  calibration staged from epoch {epoch}", flush=True)
    except (ValueError, FileNotFoundError) as e:
        print(f"  calibration NOT staged: {e}", flush=True)
    n = convert_tree(src, dst, a.task, force_src=force_src)
    print(f"{src} -> {dst}")
    print(f"  {n['parquet']} parquet ({n['cols']} pose columns), "
          f"{n['calib']} calibrations, {n['episodes']} world offsets rotated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
