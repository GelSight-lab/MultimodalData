"""Validate and export old_data V8 recovered from lossy released tactile video.

The base tree is a pinned remote snapshot (or a newer action-updated tree).
All nonforce columns and metadata are retained. Existing force-derived fields
are rebuilt using their previously declared stiffness; new episodes receive
only force and source-frame columns. This module never uploads anything.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .react_calib import CALIBRATION_NAME
from .run_episode import FIELDS, PIPELINE_VERSION
from .run_episode_mp4 import OLD_SCOPES

SOURCE_FORMAT = "release_tactile_mp4_h264"
SIDES = ("left", "right")


def columns_equal(left: pa.ChunkedArray, right: pa.ChunkedArray) -> bool:
    """Arrow treats NaN != NaN; retained missing mocap samples must compare equal."""
    if left.equals(right):
        return True
    if left.type != right.type or len(left) != len(right):
        return False
    a, b = np.asarray(left.to_pylist()), np.asarray(right.to_pylist())
    return (a.dtype.kind in "fciu" and b.dtype.kind in "fciu"
            and np.array_equal(a, b, equal_nan=True))


def tables_equal(left: pa.Table, right: pa.Table) -> bool:
    return (left.schema.equals(right.schema, check_metadata=True)
            and all(columns_equal(left[name], right[name]) for name in left.column_names))


def scoped_paths(root: Path) -> list[Path]:
    return sorted(path for task, dates in OLD_SCOPES.items() for date in dates
                  for path in root.glob(f"{task}/meta/{date}/episode_*.parquet"))


def download_base(root: Path, repo: str = "yxma/React") -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.repo_info(repo, repo_type="dataset").sha
    paths = [p for p in api.list_repo_files(repo, repo_type="dataset", revision=revision)
             if any(p.startswith(f"old_data/{task}/meta/{date}/episode_")
                    for task, dates in OLD_SCOPES.items() for date in dates)
             and p.endswith(".parquet")]
    if len(paths) != 36:
        raise ValueError(f"Expected 36 old_data episodes, found {len(paths)}")

    def fetch(name):
        src = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
        dst = root / name.removeprefix("old_data/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {"path": name, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(fetch, paths))
    manifest = {"repo": repo, "revision": revision, "files": files}
    (root / "remote_base.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_side(path: Path, table: pa.Table, side: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"source_format": SOURCE_FORMAT, "pipeline_version": PIPELINE_VERSION,
                "force_calibration": CALIBRATION_NAME,
                "alignment": "video_frame_i_equals_parquet_row_i", "lossy_input": True,
                "absolute_force_validated_on_react": False}
    for key, expected in required.items():
        if key not in data or data[key].item() != expected:
            raise ValueError(f"{path}: invalid {key}")
    rows = len(table)
    fresh = np.asarray(table[f"tactile_{side}_is_new"].to_numpy(), bool)
    expected_source = np.maximum.accumulate(np.where(fresh, np.arange(rows), 0))
    source = data["source_frame"]
    if source.shape != (rows,) or not np.array_equal(source, expected_source):
        raise ValueError(f"{path}: source-frame alignment mismatch")
    duplicate = np.flatnonzero(~fresh[1:]) + 1
    for key in FIELDS:
        values = data[key]
        if values.shape != (rows,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{path}: invalid {key} values/rows")
        if not np.array_equal(values[duplicate], values[duplicate - 1]):
            raise ValueError(f"{path}: duplicate rows are not held for {key}")
    ceiling = float(data["force_calibration_ceiling_n"])
    if not np.isfinite(ceiling) or ceiling <= 0 or data["force_normal_n"].max() > ceiling + 1e-5:
        raise ValueError(f"{path}: calibration ceiling mismatch")
    refs = data["reference_rows"]
    if refs.ndim != 1 or not len(refs) or (refs < 0).any() or (refs >= rows).any() or not fresh[refs].all():
        raise ValueError(f"{path}: invalid reference rows")
    diag = {**required, "side": side, "rows": rows,
            "fresh_rows": int(fresh.sum()), "duplicate_rows": len(duplicate),
            "force_max_n": float(data["force_normal_n"].max()),
            "force_p95_n": float(np.percentile(data["force_normal_n"], 95)),
            "force_calibration_ceiling_n": ceiling,
            "ceiling_rows": int(np.count_nonzero(data["force_normal_n"] >= ceiling - 1e-5)),
            "zero_force_rows": int(np.count_nonzero(data["force_normal_n"] == 0)),
            "reference_rows": refs.tolist(),
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return data, diag


def update_table(table: pa.Table, force_root: Path, task: str, date: str,
                 episode: str) -> tuple[pa.Table, dict]:
    if len(table.column_names) != len(set(table.column_names)):
        raise ValueError("Base table contains duplicate column names")
    known = {f"force_{s}_{suffix}" for s in SIDES for suffix in
             ("normal_n", "source_frame", "penetration_mm", "target_pose")}
    unknown = {name for name in table.column_names if name.startswith("force_")} - known
    if unknown:
        raise ValueError(f"Unrecognized existing force columns: {sorted(unknown)}")
    previous = json.loads((table.schema.metadata or {}).get(b"twm.force_export", b"{}"))
    updated = table
    diagnostics = []
    for side in SIDES:
        data, diag = validate_side(force_root / task / date / f"{episode}_{side}.npz", table, side)
        force = data["force_normal_n"].astype(np.float32)
        columns = {f"force_{side}_normal_n": pa.array(force),
                   f"force_{side}_source_frame": pa.array(data["source_frame"].astype(np.int32))}
        derived = [f"force_{side}_{name}" for name in ("penetration_mm", "target_pose")]
        presence = [name in table.column_names for name in derived]
        stiffness = None
        if any(presence):
            if not all(presence):
                raise ValueError("Incomplete existing derived force columns")
            stiffness = previous.get("stiffness_n_per_mm")
            if stiffness is None or not np.isfinite(stiffness) or stiffness <= 0:
                raise ValueError("Existing derived force columns lack a valid stiffness")
            from .export_force_columns import press_direction, _list_column
            pose = np.asarray(table[f"sensor_{side}_pose"].to_pylist(), dtype=np.float64)
            target = pose.copy()
            penetration = force.astype(np.float64) / stiffness
            contact = force > 0
            target[contact, :3] += (penetration[contact, None] / 1000) * press_direction(task, side, pose)[contact]
            columns[derived[0]] = pa.array(penetration.astype(np.float32))
            columns[derived[1]] = _list_column(target).cast(table.schema.field(derived[1]).type)
        for name, array in columns.items():
            field_meta = {"twm.source": "twm.force_recovery.run_episode_mp4",
                          "twm.source_format": SOURCE_FORMAT, "twm.lossy_input": "true",
                          "twm.source_video": f"old_data/{task}/videos/{date}/{episode}/tactile_{side}.mp4",
                          "twm.pipeline_version": str(PIPELINE_VERSION),
                          "twm.force_calibration": CALIBRATION_NAME}
            if name.endswith("source_frame"):
                field_meta.update({"twm.units": "index", "twm.desc":
                    "Zero-based decoded released tactile MP4 frame index; video frame i equals parquet row i. "
                    "Duplicate tactile rows repeat the preceding evaluated frame index. Input is lossy H.264."})
            elif name.endswith("normal_n"):
                field_meta.update({"twm.units": "N", "twm.desc":
                    "Estimated V8 normal force from lossy H.264 tactile input; nonnegative; held on duplicate rows. "
                    "Absolute force is not validated on React; not bit-equivalent to raw-H5 reconstruction."})
            else:
                field_meta.update({"twm.stiffness_n_per_mm": str(stiffness),
                    "twm.units": "mm" if name.endswith("penetration_mm") else "m,m,m,quat(xyzw)",
                    "twm.desc": "Recomputed from MP4-recovered V8 force using the previously declared assumed "
                    "controller stiffness; F/k virtual deflection, not measured gel indentation."})
            field = pa.field(name, array.type, metadata=field_meta)
            if name in updated.column_names:
                updated = updated.set_column(updated.schema.get_field_index(name), field, array)
            else:
                updated = updated.append_column(field, array)
        diagnostics.append({**diag, "stiffness_n_per_mm": stiffness,
                            "columns": list(columns),
                            "source_video": f"old_data/{task}/videos/{date}/{episode}/tactile_{side}.mp4"})
    header = {"generator": "twm.force_recovery.finalize_mp4_recovery",
              "pipeline_version": PIPELINE_VERSION, "force_calibration": [CALIBRATION_NAME],
              "source_format": SOURCE_FORMAT, "lossy_input": True,
              "alignment": "video_frame_i_equals_parquet_row_i",
              "absolute_force_validated_on_react": False,
              "raw_h5_equivalent": False,
              "stiffness_n_per_mm": diagnostics[0]["stiffness_n_per_mm"],
              "press_direction": "R(q_row) @ [0,-1,0] (sensor body -Y; existing pushT target convention)",
              "limitations": "Recovered from released lossy H.264 because May raw H5 was deleted; "
                  "compression can materially change force estimates. Existing controller targets are "
                  "recomputed at their previous stiffness; absolute React force accuracy is unvalidated."}
    metadata = dict(table.schema.metadata or {})
    metadata[b"twm.force_export"] = json.dumps(header).encode()
    updated = updated.replace_schema_metadata(metadata)
    for name in table.column_names:
        if name not in known and (not columns_equal(updated[name], table[name]) or
                                 not updated.schema.field(name).equals(table.schema.field(name), check_metadata=True)):
            raise AssertionError(f"Nonforce column changed: {name}")
    return updated, {**header, "task": task, "date": date, "episode": episode,
                     "rows": len(table), "sides": diagnostics}


def export_all(base_root: Path, force_root: Path, output: Path,
               alignment_root: Path | None = None) -> dict:
    paths = scoped_paths(base_root)
    if len(paths) != 36:
        raise ValueError(f"Expected 36 source episodes, found {len(paths)}")
    summaries = []
    for path in paths:
        task, _, date, name = path.relative_to(base_root).parts
        table = pq.read_table(path)
        if alignment_root is not None:
            import av
            original = pq.read_table(alignment_root / path.relative_to(base_root))
            for key in ("frame_idx", "timestamp", "tactile_left_is_new", "tactile_right_is_new"):
                if not table[key].equals(original[key]):
                    raise ValueError(f"{path}: base/input alignment differs: {key}")
            for side in SIDES:
                video = alignment_root / task / "videos" / date / path.stem / f"tactile_{side}.mp4"
                with av.open(str(video)) as container:
                    if container.streams.video[0].frames != len(table):
                        raise ValueError(f"{video}: video/parquet frame count mismatch")
        updated, summary = update_table(table, force_root, task, date, path.stem)
        dst = output / path.relative_to(base_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, dst, compression="zstd")
        reread = pq.read_table(dst)
        if not tables_equal(reread, updated):
            raise AssertionError(f"Parquet roundtrip mismatch: {dst}")
        dst.with_suffix(".force.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(f"verified {task}/{date}/{name}: {len(table)} rows", flush=True)
    manifest = {"episodes": summaries, "n_episodes": len(summaries),
                "n_sensor_sides": 2 * len(summaries),
                "total_rows": sum(x["rows"] for x in summaries),
                "all_nonforce_columns_preserved": True,
                "source_format": SOURCE_FORMAT, "lossy_input": True}
    base_manifest = base_root / "remote_base.json"
    if base_manifest.exists():
        manifest["remote_base"] = json.loads(base_manifest.read_text())
    (output / "force_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--download-base", action="store_true")
    parser.add_argument("--force-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alignment-root", type=Path)
    args = parser.parse_args()
    if args.download_base:
        print(json.dumps(download_base(args.base_root), indent=2))
    if args.output:
        if args.force_root is None:
            parser.error("--output requires --force-root")
        manifest = export_all(args.base_root, args.force_root, args.output, args.alignment_root)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("episodes", "remote_base")}, indent=2))


if __name__ == "__main__":
    main()
