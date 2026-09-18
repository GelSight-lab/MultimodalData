#!/usr/bin/env python3
"""Audit, calibrate, build, review, and verify mocap repair candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from twm.react_preprocess.mocap_candidate import (
    CandidateBuild,
    CandidateWriter,
    InputManifest,
    VerificationReport,
    apply_decisions,
    load_manifest,
    save_manifest,
    snapshot_inputs,
    verify_candidate,
)
from twm.react_preprocess.mocap_repair import (
    TaskGate,
    detect_bouts,
)
from twm.scripts.benchmark_mocap_repair import (
    BenchmarkReport,
    benchmark_task,
)


TASKS = ("motherboard", "pushT", "rope", "toy")
DEFAULT_SOURCE = Path("/media/yxma/Disk1/twm/release")
DEFAULT_OUTPUT = Path(
    "/media/yxma/Disk1/twm/review/mocap_repair_2026-09-17/candidate_release")


def _atomic_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp, path)
    return path


def _gap_map(source_root: Path, task: str) -> dict[str, Any]:
    path = source_root / task / "pose_gaps.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _known_gaps(gaps: dict[str, Any], date: str, episode: str,
                side: str) -> list[tuple[int, int]]:
    return [tuple(map(int, row)) for row in
            gaps.get(f"{date}/{episode}", {}).get(side, [])]


def _load_pinned_manifest(candidate_root: Path) -> InputManifest:
    path = Path(candidate_root) / "input_manifest.json"
    if not path.is_file():
        raise ValueError(f"run audit first; missing {path}")
    return load_manifest(path)


def run_audit(source_root: Path = DEFAULT_SOURCE,
              candidate_root: Path = DEFAULT_OUTPUT,
              tasks: Iterable[str] = TASKS) -> InputManifest:
    """Freeze input membership and report metadata-only anomaly bouts."""
    source_root = Path(source_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    manifest = snapshot_inputs(source_root, tasks)
    # Constructor performs the production-root overlap safety check before the
    # audit writes even its first JSON file.
    CandidateWriter(source_root, candidate_root, manifest)
    save_manifest(manifest, candidate_root / "input_manifest.json")

    counts: dict[str, Counter] = {task: Counter() for task in manifest.tasks}
    frames: dict[str, int] = {task: 0 for task in manifest.tasks}
    gap_cache = {task: _gap_map(source_root, task) for task in manifest.tasks}
    for item in manifest.files:
        task, _, date, filename = Path(item.path).parts
        episode = Path(filename).stem
        table = pq.read_table(source_root / item.path)
        for side in ("left", "right"):
            name = f"sensor_{side}_pose"
            if name not in table.column_names:
                continue
            pose = np.asarray(table[name].to_pylist(), float)
            bouts = detect_bouts(
                pose, side,
                _known_gaps(gap_cache[task], date, episode, side))
            for bout in bouts:
                counts[task][bout.kind] += 1
                frames[task] += bout.end - bout.start + 1
    _atomic_json(candidate_root / "audit_summary.json", {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "source_root": str(source_root),
        "episode_count": len(manifest.files),
        "tasks": {task: {
            "episodes": sum(Path(item.path).parts[0] == task
                            for item in manifest.files),
            "bouts": dict(sorted(counts[task].items())),
            "candidate_frames": frames[task],
        } for task in manifest.tasks},
    })
    return manifest


def _clean_fragments(manifest: InputManifest, task: str,
                     context: int = 15) -> list[np.ndarray]:
    source_root = Path(manifest.source_root)
    gaps = _gap_map(source_root, task)
    fragments: list[np.ndarray] = []
    for item in manifest.files:
        if Path(item.path).parts[0] != task:
            continue
        _, _, date, filename = Path(item.path).parts
        episode = Path(filename).stem
        table = pq.read_table(source_root / item.path)
        for side in ("left", "right"):
            name = f"sensor_{side}_pose"
            if name not in table.column_names:
                continue
            pose = np.asarray(table[name].to_pylist(), float)
            blocked = ~np.isfinite(pose).all(axis=1)
            for bout in detect_bouts(
                    pose, side, _known_gaps(gaps, date, episode, side)):
                lo = max(0, bout.start - context)
                hi = min(len(pose), bout.end + context + 1)
                blocked[lo:hi] = True
            starts = np.flatnonzero(~blocked & np.r_[True, blocked[:-1]])
            ends = np.flatnonzero(~blocked & np.r_[blocked[1:], True]) + 1
            for start, end in zip(starts, ends):
                if end - start >= 40:
                    fragments.append(pose[start:end].copy())
    return fragments


def run_benchmarks(candidate_root: Path = DEFAULT_OUTPUT, *,
                   anomaly_lengths: Iterable[int] = (1, 5, 20, 60),
                   max_intervals: int = 300,
                   seed: int = 0) -> dict[str, BenchmarkReport]:
    """Calibrate one immutable, explicit confidence gate per task."""
    candidate_root = Path(candidate_root).resolve()
    manifest = _load_pinned_manifest(candidate_root)
    # Preflight source digests before treating any source pose as clean truth.
    verifier = CandidateWriter(
        Path(manifest.source_root), candidate_root, manifest)
    for item in manifest.files:
        verifier._verify_source(item)
    reports: dict[str, BenchmarkReport] = {}
    benchmark_root = candidate_root / "benchmarks"
    for task in manifest.tasks:
        fragments = _clean_fragments(manifest, task)
        if not fragments:
            raise ValueError(f"no clean pose fragments available for {task}")
        report = benchmark_task(
            fragments, task=task, anomaly_lengths=anomaly_lengths,
            max_intervals=max_intervals, seed=seed)
        reports[task] = report
        _atomic_json(benchmark_root / f"{task}.json", report.to_dict())
    _atomic_json(candidate_root / "task_gates.json", {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "tasks": {task: asdict(report.gate)
                  for task, report in sorted(reports.items())},
    })
    return reports


def _load_task_gates(candidate_root: Path,
                     manifest: InputManifest) -> dict[str, TaskGate]:
    path = Path(candidate_root) / "task_gates.json"
    if not path.is_file():
        raise ValueError(f"run benchmark first; missing {path}")
    payload = json.loads(path.read_text())
    if payload.get("manifest_digest") != manifest.digest:
        raise ValueError("task gates manifest digest mismatch")
    missing = set(manifest.tasks) - set(payload.get("tasks", {}))
    if missing:
        raise ValueError(f"task gates missing tasks: {sorted(missing)}")
    return {task: TaskGate(**payload["tasks"][task])
            for task in manifest.tasks}


def run_build(candidate_root: Path = DEFAULT_OUTPUT, *,
              task_gates: dict[str, TaskGate] | None = None) -> CandidateBuild:
    candidate_root = Path(candidate_root).resolve()
    manifest = _load_pinned_manifest(candidate_root)
    gates = task_gates or _load_task_gates(candidate_root, manifest)
    missing = set(manifest.tasks) - set(gates)
    if missing:
        raise ValueError(f"task gates missing tasks: {sorted(missing)}")
    writer = CandidateWriter(
        Path(manifest.source_root), candidate_root, manifest,
        task_gates=gates)
    build = writer.write_all()
    _atomic_json(candidate_root / "candidate_summary.json",
                 summarize_candidate(candidate_root))
    return build


def run_verify(candidate_root: Path = DEFAULT_OUTPUT) -> VerificationReport:
    report = verify_candidate(candidate_root)
    _atomic_json(Path(candidate_root) / "verification.json", {
        "schema_version": 1,
        "episodes": report.episodes,
        "errors": list(report.errors),
    })
    return report


def summarize_candidate(candidate_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    candidate_root = Path(candidate_root).resolve()
    manifest = _load_pinned_manifest(candidate_root)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "tasks": {},
    }
    for task in manifest.tasks:
        row = {
            "events": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "repaired_frames": 0,
            "native_repaired_actions": 0,
            "native_recovered_valid_actions": 0,
            "fps15_repaired_actions": 0,
            "fps15_recovered_valid_actions": 0,
            "unresolved_events": 0,
        }
        for path in sorted((candidate_root / task / "repair_events").glob("*/*.json")):
            for event in json.loads(path.read_text()).get("events", []):
                confidence = str(event["confidence"]).upper()
                row["events"].setdefault(confidence, 0)
                row["events"][confidence] += 1
                row["repaired_frames"] += len(event.get("replaced_frames", []))
                if confidence in {"MEDIUM", "LOW"} and event.get("decision") != "accept_repair":
                    row["unresolved_events"] += 1
        for rate, repaired_key, recovered_key in (
                ("actions_native", "native_repaired_actions",
                 "native_recovered_valid_actions"),
                ("actions_fps15", "fps15_repaired_actions",
                 "fps15_recovered_valid_actions")):
            for path in sorted((candidate_root / task / rate).glob("*/*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    repaired = np.asarray(data["repaired"], bool)
                    valid = np.asarray(data["valid"], bool)
                    row[repaired_key] += int(repaired.sum())
                    row[recovered_key] += int((repaired & valid).sum())
        summary["tasks"][task] = row
    return summary


def run_review(candidate_root: Path = DEFAULT_OUTPUT, *, render: bool = False,
               review_root: Path | None = None,
               high_sample_rate: float = 0.1,
               sample_seed: int = 0) -> dict[str, Any]:
    from twm.scripts.build_pose_review import build_review

    candidate_root = Path(candidate_root).resolve()
    manifest = _load_pinned_manifest(candidate_root)
    source_root = Path(manifest.source_root)
    review_root = (Path(review_root).resolve() if review_root is not None
                   else candidate_root.parent / "review_package")
    results = {}
    for task in manifest.tasks:
        results[task] = build_review(
            source_root / task, review_root / task, render=render,
            candidate_task_root=candidate_root / task,
            high_sample_rate=high_sample_rate,
            sample_seed=sample_seed)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    audit.add_argument("--task", action="append", choices=TASKS)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--max-intervals", type=int, default=300)
    bench.add_argument("--seed", type=int, default=0)
    sub.add_parser("build")
    review = sub.add_parser("review")
    review.add_argument("--render", action="store_true")
    review.add_argument("--review-root", type=Path)
    review.add_argument("--high-sample-rate", type=float, default=0.1)
    review.add_argument("--seed", type=int, default=0)
    decisions = sub.add_parser("apply-decisions")
    decisions.add_argument("decisions", type=Path)
    sub.add_parser("verify")
    sub.add_parser("summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = run_audit(args.source, args.output, args.task or TASKS)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif args.command == "benchmark":
        result = run_benchmarks(
            args.output, max_intervals=args.max_intervals, seed=args.seed)
        print(json.dumps({task: report.to_dict()
                          for task, report in result.items()}, indent=2,
                         sort_keys=True))
    elif args.command == "build":
        result = run_build(args.output)
        print(f"episodes={len(result.episodes)} manifest={result.manifest_digest}")
    elif args.command == "review":
        result = run_review(
            args.output, render=args.render, review_root=args.review_root,
            high_sample_rate=args.high_sample_rate, sample_seed=args.seed)
        print(json.dumps({task: len(value["review_events"])
                          for task, value in result.items()}, indent=2,
                         sort_keys=True))
    elif args.command == "apply-decisions":
        apply_decisions(args.output, json.loads(args.decisions.read_text()))
        print(json.dumps(summarize_candidate(args.output), indent=2,
                         sort_keys=True))
    elif args.command == "verify":
        report = run_verify(args.output)
        print(f"episodes={report.episodes} errors={len(report.errors)}")
    elif args.command == "summary":
        print(json.dumps(summarize_candidate(args.output), indent=2,
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

