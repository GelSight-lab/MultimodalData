"""Export old motherboard parquet with repaired T-row LeRobot actions.

Raw release columns are preserved. Candidate poses and action validity come
from the provenance-safe mocap repair tree; unresolved events stay invalid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TASK = "motherboard"
DATES = ("2026-05-10", "2026-05-11", "2026-05-19")
EXPECTED_EPISODES = 32
SIDES = ("left", "right")


def _fixed_float32(values: np.ndarray, width: int) -> pa.Array:
    flat = pa.array(np.asarray(values, np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _append(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    if name in table.column_names:
        raise ValueError(f"refusing to replace existing column {name!r}")
    return table.append_column(name, values)


def _long_transition_mask(events: Sequence[Mapping[str, object]], side: str,
                          n_rows: int) -> np.ndarray:
    result = np.zeros(n_rows, dtype=bool)
    for event in events:
        if str(event.get("side")) != side:
            continue
        start, end = int(event["start"]), int(event["end"])
        if end - start + 1 <= 60:
            continue
        first = max(0, start - 1)
        last = min(n_rows - 2, end)
        if last >= first:
            result[first:last + 1] = True
    return result


def build_action_episode(raw: pa.Table, candidate: pa.Table,
                         native_left: Mapping[str, np.ndarray],
                         native_right: Mapping[str, np.ndarray],
                         events: Sequence[Mapping[str, object]]) -> pa.Table:
    """Return raw columns plus repaired pose/action columns, aligned to T rows."""
    if raw.num_rows != candidate.num_rows:
        raise ValueError("raw and candidate row counts differ")
    n = raw.num_rows
    if n < 1:
        raise ValueError("empty episode")
    repaired_pose = {}
    native = {"left": native_left, "right": native_right}
    for side in SIDES:
        repaired_pose[side] = np.asarray(
            candidate[f"sensor_{side}_pose"].to_pylist(), dtype=np.float32)
        if repaired_pose[side].shape != (n, 7):
            raise ValueError(f"candidate {side} pose must have shape ({n}, 7)")
        for key in ("valid", "repaired"):
            if np.asarray(native[side][key]).shape != (n - 1,):
                raise ValueError(f"native {side} {key} must have {n - 1} rows")

    target = np.concatenate([
        np.concatenate([repaired_pose["left"][1:],
                        repaired_pose["right"][1:]], axis=1),
        np.concatenate([repaired_pose["left"][-1:],
                        repaired_pose["right"][-1:]], axis=1),
    ], axis=0)
    out = raw
    for side in SIDES:
        out = _append(out, f"sensor_{side}_pose_repaired",
                      _fixed_float32(repaired_pose[side], 7))
    out = _append(out, "action", _fixed_float32(target, 14))

    for side in SIDES:
        valid = np.r_[np.asarray(native[side]["valid"], bool), False]
        repaired = np.r_[np.asarray(native[side]["repaired"], bool), False]
        pose_conf = np.asarray(
            candidate[f"pose_{side}_repair_confidence"].to_numpy(),
            dtype=np.uint8)
        confidence = np.zeros(n, dtype=np.uint8)
        if n > 1:
            confidence[:-1] = np.maximum(pose_conf[:-1], pose_conf[1:])
        event_overlap = np.zeros(n, dtype=bool)
        for event in events:
            if str(event.get("side")) != side:
                continue
            start, end = int(event["start"]), int(event["end"])
            first, last = max(0, start - 1), min(n - 2, end)
            if last >= first:
                event_overlap[first:last + 1] = True
        lost = (~valid) & event_overlap
        long_lost = _long_transition_mask(events, side, n) & lost
        out = _append(out, f"action_valid_{side}", pa.array(valid))
        out = _append(out, f"action_repaired_{side}", pa.array(repaired))
        out = _append(out, f"action_repair_confidence_{side}",
                      pa.array(confidence, type=pa.uint8()))
        out = _append(out, f"action_lost_track_{side}", pa.array(lost))
        out = _append(out, f"action_long_lost_track_{side}",
                      pa.array(long_lost))
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_native(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
    try:
        pq.write_table(table, temp, compression="zstd")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _columns_equal(left: pa.ChunkedArray, right: pa.ChunkedArray) -> bool:
    if left.type != right.type or len(left) != len(right):
        return False
    if left.equals(right):
        return True
    try:
        if (pa.types.is_list(left.type) or pa.types.is_large_list(left.type)
                or pa.types.is_fixed_size_list(left.type)):
            a = np.asarray(left.to_pylist(), dtype=float)
            b = np.asarray(right.to_pylist(), dtype=float)
        elif pa.types.is_floating(left.type):
            a = np.asarray(left.to_numpy(zero_copy_only=False), dtype=float)
            b = np.asarray(right.to_numpy(zero_copy_only=False), dtype=float)
        else:
            return False
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def export_tree(release: Path, candidate_root: Path, output: Path) -> dict:
    release, candidate_root, output = map(Path, (release, candidate_root, output))
    sources = [p for date in DATES
               for p in sorted((release / TASK / "meta" / date).glob("episode_*.parquet"))]
    if len(sources) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} episodes, found {len(sources)}")
    episodes = []
    totals = {"valid_left": 0, "valid_right": 0,
              "repaired_left": 0, "repaired_right": 0,
              "lost_left": 0, "lost_right": 0,
              "long_left": 0, "long_right": 0}
    for source in sources:
        date, name = source.parent.name, source.name
        candidate = candidate_root / TASK / "meta" / date / name
        event_path = candidate_root / TASK / "repair_events" / date / f"{source.stem}.json"
        if not candidate.is_file() or not event_path.is_file():
            raise FileNotFoundError(f"missing candidate artifacts for {date}/{source.stem}")
        raw_table, candidate_table = pq.read_table(source), pq.read_table(candidate)
        event_doc = json.loads(event_path.read_text())
        events = event_doc.get("events", [])
        native = {
            side: _load_native(candidate_root / TASK / "actions_native" / date /
                               f"{source.stem}_{side}.npz")
            for side in SIDES
        }
        table = build_action_episode(raw_table, candidate_table,
                                     native["left"], native["right"], events)
        for field in raw_table.schema:
            if field.name not in table.column_names or table.schema.field(field.name) != field:
                raise ValueError(f"original schema changed: {date}/{name}:{field.name}")
            if not _columns_equal(raw_table[field.name], table[field.name]):
                raise ValueError(f"original values changed: {date}/{name}:{field.name}")
        dest = output / "meta" / date / name
        _atomic_parquet(table, dest)
        event_dest = output / "action_repair_events" / date / f"{source.stem}.json"
        event_dest.parent.mkdir(parents=True, exist_ok=True)
        event_dest.write_bytes(event_path.read_bytes())
        stats = {}
        for side in SIDES:
            for key, col in (("valid", f"action_valid_{side}"),
                             ("repaired", f"action_repaired_{side}"),
                             ("lost", f"action_lost_track_{side}"),
                             ("long", f"action_long_lost_track_{side}")):
                value = int(np.asarray(table[col].to_numpy()).sum())
                stats[f"{key}_{side}"] = value
                totals[f"{key}_{side}"] += value
        episodes.append({"date": date, "episode": source.stem,
                         "rows": table.num_rows, "source_sha256": _sha256(source),
                         "parquet_sha256": _sha256(dest), **stats})
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "dates": list(DATES),
        "episodes": episodes,
        "totals": totals,
        "action": "next-frame repaired left pose (7) + right pose (7)",
        "terminal_action_valid": False,
        "long_lost_track_min_frames": 61,
    }
    (output / "action_update_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_tree(args.release, args.candidate, args.output)
    print(f"episodes={len(manifest['episodes'])} errors=0")
    print(json.dumps(manifest["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
