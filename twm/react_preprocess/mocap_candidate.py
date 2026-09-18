"""Immutable-input writer for mocap repair candidate trees.

Production release trees are read-only inputs.  Every build is pinned to a
manifest of exact parquet digests and writes augmented metadata plus action
sidecars beneath a separate review root.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .mocap_repair import Confidence, TaskGate, repair_pose_stream
from .repaired_actions import ActionSeries, actions_fps15, native_actions


SIDES = ("left", "right")
MANIFEST_SCHEMA_VERSION = 1


class SourceChangedError(RuntimeError):
    """A manifest-pinned input no longer has its recorded content."""


class CandidateVerificationError(RuntimeError):
    """Candidate provenance or derived actions disagree with their contract."""


@dataclass(frozen=True)
class ManifestFile:
    path: str
    bytes: int
    rows: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class InputManifest:
    source_root: str
    tasks: tuple[str, ...]
    files: tuple[ManifestFile, ...]
    created_utc: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_root": self.source_root,
            "tasks": list(self.tasks),
            "created_utc": self.created_utc,
            "digest": self.digest,
            "files": [asdict(item) for item in self.files],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InputManifest":
        if int(value.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported input manifest schema")
        return cls(
            source_root=str(value["source_root"]),
            tasks=tuple(map(str, value["tasks"])),
            files=tuple(ManifestFile(**item) for item in value["files"]),
            created_utc=str(value["created_utc"]),
            digest=str(value["digest"]),
        )


@dataclass(frozen=True)
class CandidateEpisode:
    task: str
    date: str
    episode: str
    parquet: Path
    events: Path
    repaired_frames: int


@dataclass(frozen=True)
class CandidateBuild:
    root: Path
    manifest_digest: str
    episodes: tuple[CandidateEpisode, ...]


@dataclass(frozen=True)
class VerificationReport:
    episodes: int
    errors: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _manifest_identity(source_root: Path, tasks: tuple[str, ...],
                       files: tuple[ManifestFile, ...]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_root": str(source_root),
        "tasks": list(tasks),
        "files": [{
            "path": item.path,
            "bytes": item.bytes,
            "rows": item.rows,
            "sha256": item.sha256,
        } for item in files],
    }


def _validate_manifest(manifest: InputManifest) -> None:
    identity = _manifest_identity(
        Path(manifest.source_root).resolve(), manifest.tasks, manifest.files)
    expected = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if manifest.digest != expected:
        raise ValueError(
            f"input manifest digest is invalid: {manifest.digest} != {expected}")


def snapshot_inputs(source_root: Path,
                    tasks: Iterable[str]) -> InputManifest:
    """Snapshot the exact parquet set used by one candidate run."""
    source_root = Path(source_root).resolve()
    task_tuple = tuple(sorted(dict.fromkeys(str(task) for task in tasks)))
    files: list[ManifestFile] = []
    for task in task_tuple:
        for path in sorted((source_root / task / "meta").glob("*/*.parquet")):
            stat = path.stat()
            files.append(ManifestFile(
                path=path.relative_to(source_root).as_posix(),
                bytes=stat.st_size,
                rows=pq.ParquetFile(path).metadata.num_rows,
                mtime_ns=stat.st_mtime_ns,
                sha256=_sha256(path),
            ))
    files_tuple = tuple(sorted(files, key=lambda item: item.path))
    identity = _manifest_identity(source_root, task_tuple, files_tuple)
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return InputManifest(
        source_root=str(source_root),
        tasks=task_tuple,
        files=files_tuple,
        created_utc=datetime.now(timezone.utc).isoformat(),
        digest=digest,
    )


def save_manifest(manifest: InputManifest, path: Path) -> Path:
    """Persist a manifest, refusing to replace a different snapshot."""
    _validate_manifest(manifest)
    path = Path(path)
    if path.is_file():
        prior = InputManifest.from_dict(json.loads(path.read_text()))
        _validate_manifest(prior)
        if prior.digest != manifest.digest:
            raise ValueError(
                f"manifest path already contains a different snapshot: {path}")
    _atomic_json(path, manifest.to_dict())
    return path


def load_manifest(path: Path) -> InputManifest:
    manifest = InputManifest.from_dict(json.loads(Path(path).read_text()))
    _validate_manifest(manifest)
    return manifest


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp, path)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _pose_list_array(value: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(value, dtype=np.float64).tolist(),
                    type=pa.list_(pa.float64()))


def _set_column(table: pa.Table, name: str, value: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, value)
    return table.append_column(name, value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _event_dict(event: Any) -> dict[str, Any]:
    return _jsonable({
        "event_id": event.event_id,
        "side": event.bout.side,
        "start": event.bout.start,
        "end": event.bout.end,
        "kind": event.bout.kind,
        "seed_frames": list(event.bout.seed_frames),
        "left_context_end": event.bout.left_context_end,
        "right_context_start": event.bout.right_context_start,
        "confidence": Confidence(event.confidence).name,
        "method": event.method,
        "replaced_frames": list(event.replaced_frames),
        "evidence": event.evidence,
    })


def _write_actions(path: Path, actions: ActionSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_text = np.asarray(["|".join(ids) for ids in actions.event_ids],
                            dtype=np.str_)
    with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        np.savez_compressed(
            stream,
            values=np.asarray(actions.values, np.float32),
            valid=np.asarray(actions.valid, bool),
            repaired=np.asarray(actions.repaired, bool),
            event_ids=event_text,
            start_rows=np.asarray(actions.start_rows, np.int32),
            end_rows=np.asarray(actions.end_rows, np.int32),
            rotation_deg=np.asarray(actions.rotation_deg, np.float32),
            translation_mm=np.asarray(actions.translation_mm, np.float32),
        )
    os.replace(temp, path)


class CandidateWriter:
    """Build candidate metadata pinned to one immutable input manifest."""

    def __init__(self, source_root: Path, output_root: Path,
                 manifest: InputManifest, *,
                 task_gates: dict[str, TaskGate] | None = None,
                 production_roots: Iterable[Path] = ()):
        self.source_root = Path(source_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.manifest = manifest
        _validate_manifest(manifest)
        self.task_gates = dict(task_gates or {})
        inferred = {
            self.source_root,
            self.source_root.parent / "release_zup",
            self.source_root.parent / "release_cut",
        }
        inferred.update(Path(path).resolve() for path in production_roots)
        for root in inferred:
            if (_is_within(self.output_root, root)
                    or _is_within(root, self.output_root)):
                raise ValueError(
                    f"candidate output overlaps production root: "
                    f"{self.output_root} vs {root}")
        if Path(manifest.source_root).resolve() != self.source_root:
            raise ValueError("manifest source_root does not match writer source")

    def _verify_source(self, item: ManifestFile) -> Path:
        path = self.source_root / item.path
        if not path.is_file():
            raise SourceChangedError(f"source missing after manifest: {item.path}")
        stat = path.stat()
        actual = _sha256(path)
        if actual != item.sha256:
            raise SourceChangedError(
                f"source digest changed after manifest: {item.path} "
                f"(bytes {item.bytes} -> {stat.st_size})")
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != item.rows:
            raise SourceChangedError(
                f"source row count changed after manifest: {item.path}")
        return path

    def _known_gaps(self, task: str, date: str,
                    episode: str) -> dict[str, list[tuple[int, int]]]:
        path = self.source_root / task / "pose_gaps.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text())
        row = payload.get(f"{date}/{episode}", {})
        return {side: [tuple(map(int, gap)) for gap in row.get(side, [])]
                for side in SIDES}

    def _write_parquet_atomic(self, table: pa.Table, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}.",
                suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
        try:
            pq.write_table(table, temp, compression="zstd")
            os.replace(temp, output)
        finally:
            if temp.exists():
                temp.unlink()

    def _write_episode(self, item: ManifestFile) -> CandidateEpisode:
        source = self._verify_source(item)
        rel = Path(item.path)
        task, marker, date, filename = rel.parts
        if marker != "meta":
            raise ValueError(f"manifest path is not task/meta/date: {item.path}")
        episode = Path(filename).stem
        table = pq.read_table(source)
        output_table = table
        gaps = self._known_gaps(task, date, episode)
        events: list[dict[str, Any]] = []
        repaired_total = 0
        gate = self.task_gates.get(task, TaskGate())
        for side in SIDES:
            pose_name = f"sensor_{side}_pose"
            if pose_name not in table.column_names:
                continue
            raw_pose = np.asarray(table[pose_name].to_pylist(), dtype=float)
            result = repair_pose_stream(
                raw_pose, side, gaps.get(side, ()), task_gate=gate)
            output_table = _set_column(
                output_table, pose_name, _pose_list_array(result.pose))
            output_table = _set_column(
                output_table, f"pose_{side}_repaired",
                pa.array(result.repaired, type=pa.bool_()))
            output_table = _set_column(
                output_table, f"pose_{side}_repair_confidence",
                pa.array(result.confidence, type=pa.uint8()))
            output_table = _set_column(
                output_table, f"pose_{side}_valid",
                pa.array(result.valid, type=pa.bool_()))
            output_table = _set_column(
                output_table, f"pose_{side}_repair_event_id",
                pa.array(result.event_id.astype(str), type=pa.string()))
            repaired_total += int(result.repaired.sum())
            events.extend(_event_dict(event) for event in result.events)
            native = native_actions(
                result.pose, result.valid, result.repaired, result.event_id)
            half = actions_fps15(result.pose, native)
            _write_actions(
                self.output_root / task / "actions_native" / date
                / f"{episode}_{side}.npz", native)
            _write_actions(
                self.output_root / task / "actions_fps15" / date
                / f"{episode}_{side}.npz", half)

        output_table = output_table.replace_schema_metadata(
            dict(table.schema.metadata or {}))
        output = self.output_root / item.path
        self._write_parquet_atomic(output_table, output)
        event_path = (self.output_root / task / "repair_events" / date
                      / f"{episode}.json")
        _atomic_json(event_path, {
            "schema_version": 1,
            "manifest_digest": self.manifest.digest,
            "source_path": item.path,
            "source_sha256": item.sha256,
            "task": task,
            "date": date,
            "episode": episode,
            "events": sorted(events, key=lambda event: event["event_id"]),
        })
        return CandidateEpisode(
            task, date, episode, output, event_path, repaired_total)

    def write_all(self) -> CandidateBuild:
        """Verify the complete snapshot, then build only its declared files."""
        for item in self.manifest.files:
            self._verify_source(item)
        manifest_path = self.output_root / "input_manifest.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if prior.get("digest") != self.manifest.digest:
                raise ValueError(
                    "candidate root contains a different input manifest digest")
        _atomic_json(manifest_path, self.manifest.to_dict())
        _atomic_json(self.output_root / "build_config.json", {
            "schema_version": 1,
            "manifest_digest": self.manifest.digest,
            "task_gates": {
                task: asdict(self.task_gates.get(task, TaskGate()))
                for task in self.manifest.tasks
            },
        })
        episodes = tuple(self._write_episode(item) for item in self.manifest.files)
        _atomic_json(self.output_root / "build_summary.json", {
            "schema_version": 1,
            "manifest_digest": self.manifest.digest,
            "episodes": [{
                "task": episode.task,
                "date": episode.date,
                "episode": episode.episode,
                "parquet": episode.parquet.relative_to(self.output_root).as_posix(),
                "events": episode.events.relative_to(self.output_root).as_posix(),
                "repaired_frames": episode.repaired_frames,
            } for episode in episodes],
        })
        return CandidateBuild(self.output_root, self.manifest.digest, episodes)


def _load_candidate(candidate_root: Path) -> tuple[InputManifest, dict[str, TaskGate]]:
    candidate_root = Path(candidate_root).resolve()
    manifest_path = candidate_root / "input_manifest.json"
    config_path = candidate_root / "build_config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError("candidate root is missing manifest or build config")
    manifest = InputManifest.from_dict(json.loads(manifest_path.read_text()))
    config = json.loads(config_path.read_text())
    if config.get("manifest_digest") != manifest.digest:
        raise ValueError("build config manifest digest mismatch")
    gates = {task: TaskGate(**value)
             for task, value in config.get("task_gates", {}).items()}
    return manifest, gates


def _event_index(candidate_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(Path(candidate_root).glob("*/repair_events/*/*.json")):
        payload = json.loads(path.read_text())
        for event in payload.get("events", []):
            event_id = str(event["event_id"])
            if event_id in index:
                raise ValueError(f"duplicate event in candidate tree: {event_id}")
            index[event_id] = (path, event)
    return index


def _canonical_decisions(payload: dict[str, Any],
                         manifest: InputManifest,
                         events: dict[str, tuple[Path, dict[str, Any]]]) -> list[dict[str, str]]:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported decisions schema")
    if payload.get("manifest_digest") != manifest.digest:
        raise ValueError("decisions manifest digest mismatch")
    allowed = {"keep_raw", "accept_repair", "invalidate", "unsure"}
    seen: set[str] = set()
    out = []
    for row in payload.get("decisions", []):
        event_id = str(row.get("event_id", ""))
        decision = str(row.get("decision", ""))
        if event_id in seen:
            raise ValueError(f"duplicate decision for event: {event_id}")
        if event_id not in events:
            raise ValueError(f"unknown event in decisions: {event_id}")
        if decision not in allowed:
            raise ValueError(f"invalid decision {decision!r} for {event_id}")
        seen.add(event_id)
        out.append({"event_id": event_id, "decision": decision})
    return sorted(out, key=lambda row: row["event_id"])


def _finite_unit_pose(pose: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(pose[:, 3:], axis=1)
    return np.isfinite(pose).all(axis=1) & (norm > 1e-12)


def apply_decisions(candidate_root: Path,
                    payload: dict[str, Any]) -> CandidateBuild:
    """Replay canonical review decisions from the immutable source snapshot."""
    candidate_root = Path(candidate_root).resolve()
    manifest, gates = _load_candidate(candidate_root)
    events_before = _event_index(candidate_root)
    decisions = _canonical_decisions(payload, manifest, events_before)

    # Rebuild the undecided baseline first.  This makes decision application
    # independent of whatever decision was applied on the previous run.
    writer = CandidateWriter(
        Path(manifest.source_root), candidate_root, manifest,
        task_gates=gates)
    build = writer.write_all()
    events = _event_index(candidate_root)
    grouped: dict[Path, list[dict[str, str]]] = {}
    for row in decisions:
        path, _ = events[row["event_id"]]
        grouped.setdefault(path, []).append(row)

    for event_path, rows in grouped.items():
        event_payload = json.loads(event_path.read_text())
        by_id = {event["event_id"]: event
                 for event in event_payload.get("events", [])}
        task = str(event_payload["task"])
        date = str(event_payload["date"])
        episode = str(event_payload["episode"])
        candidate_path = candidate_root / task / "meta" / date / f"{episode}.parquet"
        source_path = Path(manifest.source_root) / event_payload["source_path"]
        table = pq.read_table(candidate_path)
        source = pq.read_table(source_path)
        side_state: dict[str, dict[str, np.ndarray]] = {}
        for side in SIDES:
            pose_name = f"sensor_{side}_pose"
            if pose_name not in table.column_names:
                continue
            side_state[side] = {
                "pose": np.asarray(table[pose_name].to_pylist(), float),
                "raw": np.asarray(source[pose_name].to_pylist(), float),
                "repaired": np.asarray(
                    table[f"pose_{side}_repaired"], bool).copy(),
                "confidence": np.asarray(
                    table[f"pose_{side}_repair_confidence"], np.uint8).copy(),
                "valid": np.asarray(
                    table[f"pose_{side}_valid"], bool).copy(),
                "event_id": np.asarray(
                    table[f"pose_{side}_repair_event_id"].to_pylist(), object),
            }
        for row in rows:
            event = by_id[row["event_id"]]
            side = str(event["side"])
            state = side_state[side]
            selected = np.arange(int(event["start"]), int(event["end"]) + 1)
            decision = row["decision"]
            if decision == "accept_repair":
                state["valid"][selected] = _finite_unit_pose(state["pose"][selected])
            else:
                state["pose"][selected] = state["raw"][selected]
                state["repaired"][selected] = False
                if decision == "keep_raw":
                    state["confidence"][selected] = Confidence.NONE
                    state["valid"][selected] = _finite_unit_pose(state["raw"][selected])
                elif decision == "invalidate":
                    state["confidence"][selected] = Confidence.LOW
                    state["valid"][selected] = False
                else:  # unsure
                    state["valid"][selected] = False
            event["decision"] = decision

        output_table = table
        for side, state in side_state.items():
            output_table = _set_column(
                output_table, f"sensor_{side}_pose", _pose_list_array(state["pose"]))
            output_table = _set_column(
                output_table, f"pose_{side}_repaired",
                pa.array(state["repaired"], type=pa.bool_()))
            output_table = _set_column(
                output_table, f"pose_{side}_repair_confidence",
                pa.array(state["confidence"], type=pa.uint8()))
            output_table = _set_column(
                output_table, f"pose_{side}_valid",
                pa.array(state["valid"], type=pa.bool_()))
            native = native_actions(
                state["pose"], state["valid"], state["repaired"],
                state["event_id"])
            half = actions_fps15(state["pose"], native)
            _write_actions(
                candidate_root / task / "actions_native" / date
                / f"{episode}_{side}.npz", native)
            _write_actions(
                candidate_root / task / "actions_fps15" / date
                / f"{episode}_{side}.npz", half)
        writer._write_parquet_atomic(output_table, candidate_path)
        _atomic_json(event_path, event_payload)

    _atomic_json(candidate_root / "decisions.json", {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "decisions": decisions,
    })
    return build


def _rows_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    same = left == right
    if np.issubdtype(left.dtype, np.floating):
        same |= np.isnan(left) & np.isnan(right)
    return np.all(same, axis=1)


def _verify_action_file(path: Path, expected: ActionSeries,
                        errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing action sidecar: {path}")
        return
    try:
        with np.load(path, allow_pickle=False) as data:
            for name, value in (
                    ("values", expected.values), ("valid", expected.valid),
                    ("repaired", expected.repaired),
                    ("start_rows", expected.start_rows),
                    ("end_rows", expected.end_rows)):
                if name not in data.files or not np.array_equal(data[name], value):
                    errors.append(f"action mismatch {path}: {name}")
            event_text = np.asarray(["|".join(ids) for ids in expected.event_ids],
                                    dtype=np.str_)
            if "event_ids" not in data.files or not np.array_equal(
                    data["event_ids"], event_text):
                errors.append(f"action mismatch {path}: event_ids")
    except Exception as exc:
        errors.append(f"unreadable action sidecar {path}: {exc}")


def verify_candidate(candidate_root: Path) -> VerificationReport:
    """Verify source immutability, declared changes, and action derivations."""
    candidate_root = Path(candidate_root).resolve()
    manifest, gates = _load_candidate(candidate_root)
    writer = CandidateWriter(
        Path(manifest.source_root), candidate_root, manifest,
        task_gates=gates)
    errors: list[str] = []
    for item in manifest.files:
        try:
            source_path = writer._verify_source(item)
        except SourceChangedError as exc:
            errors.append(str(exc)); continue
        candidate_path = candidate_root / item.path
        if not candidate_path.is_file():
            errors.append(f"missing candidate parquet: {item.path}"); continue
        try:
            source = pq.read_table(source_path)
            candidate = pq.read_table(candidate_path)
        except Exception as exc:
            errors.append(f"unreadable candidate {item.path}: {exc}"); continue
        if source.num_rows != candidate.num_rows:
            errors.append(f"row count mismatch: {item.path}"); continue
        for name in source.column_names:
            if name.startswith("sensor_") and name.endswith("_pose"):
                continue
            if name not in candidate.column_names or not source[name].equals(candidate[name]):
                errors.append(f"source column changed: {item.path}:{name}")
        rel = Path(item.path)
        task, _, date, filename = rel.parts
        episode = Path(filename).stem
        for side in SIDES:
            pose_name = f"sensor_{side}_pose"
            required = [pose_name, f"pose_{side}_repaired",
                        f"pose_{side}_repair_confidence", f"pose_{side}_valid",
                        f"pose_{side}_repair_event_id"]
            if pose_name not in source.column_names:
                continue
            if any(name not in candidate.column_names for name in required):
                errors.append(f"missing repair columns: {item.path}:{side}"); continue
            raw = np.asarray(source[pose_name].to_pylist(), float)
            pose = np.asarray(candidate[pose_name].to_pylist(), float)
            repaired = np.asarray(candidate[f"pose_{side}_repaired"], bool)
            valid = np.asarray(candidate[f"pose_{side}_valid"], bool)
            event_id = np.asarray(
                candidate[f"pose_{side}_repair_event_id"].to_pylist(), object)
            changed = ~_rows_equal(raw, pose)
            if np.any(changed & ~repaired):
                errors.append(f"undeclared pose change: {item.path}:{side}")
            if np.any(repaired & (event_id == "")):
                errors.append(f"repaired pose missing event id: {item.path}:{side}")
            finite_unit = _finite_unit_pose(pose)
            if np.any(valid & ~finite_unit):
                errors.append(f"valid pose is nonfinite/nonunit: {item.path}:{side}")
            norm = np.linalg.norm(pose[valid, 3:], axis=1)
            if len(norm) and not np.allclose(norm, 1.0, atol=1e-6):
                errors.append(f"valid quaternion is not unit: {item.path}:{side}")
            native = native_actions(pose, valid, repaired, event_id)
            half = actions_fps15(pose, native)
            _verify_action_file(
                candidate_root / task / "actions_native" / date
                / f"{episode}_{side}.npz", native, errors)
            _verify_action_file(
                candidate_root / task / "actions_fps15" / date
                / f"{episode}_{side}.npz", half, errors)
    report = VerificationReport(len(manifest.files), tuple(errors))
    if errors:
        raise CandidateVerificationError("; ".join(errors))
    return report
