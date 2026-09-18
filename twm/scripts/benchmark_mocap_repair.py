#!/usr/bin/env python3
"""Calibrate mocap repair confidence with deterministic clean-data masking."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from twm.react_preprocess.mocap_repair import TaskGate, repair_pose_stream


@dataclass(frozen=True)
class MetricSummary:
    median: float
    p95: float
    count: int


@dataclass(frozen=True)
class BenchmarkReport:
    task: str
    seed: int
    intervals: int
    anomaly_lengths: tuple[int, ...]
    metrics: dict[str, MetricSummary]
    gate: TaskGate

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "task": self.task,
            "seed": self.seed,
            "intervals": self.intervals,
            "anomaly_lengths": list(self.anomaly_lengths),
            "metrics": {name: asdict(self.metrics[name])
                        for name in sorted(self.metrics)},
            "gate": asdict(self.gate),
        }


@dataclass(frozen=True)
class _Candidate:
    episode: int
    start: int
    length: int
    motion: float


def _validate_episode(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"clean episode must have shape (T, 7), got {pose.shape}")
    return pose


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(
        (Rotation.from_quat(a) * Rotation.from_quat(b).inv()).magnitude())


def _motion_score(pose: np.ndarray, start: int, length: int) -> float:
    lo, hi = start - 1, start + length
    trans = np.linalg.norm(pose[hi, :3] - pose[lo, :3]) * 1000.0
    rot = float(_rotation_angle_deg(
        pose[hi:hi + 1, 3:], pose[lo:lo + 1, 3:])[0])
    return float(trans + rot)


def _candidate_order(episodes: list[np.ndarray], lengths: tuple[int, ...],
                     rng: np.random.Generator) -> list[_Candidate]:
    candidates = []
    context = 15
    for episode_index, pose in enumerate(episodes):
        for length in lengths:
            for start in range(context, len(pose) - context - length):
                rows = pose[start - context:start + length + context]
                norm = np.linalg.norm(rows[:, 3:], axis=1)
                if not np.isfinite(rows).all() or np.any(norm <= 1e-12):
                    continue
                candidates.append(_Candidate(
                    episode_index, start, length,
                    _motion_score(pose, start, length)))
    candidates.sort(key=lambda row: (row.motion, row.episode, row.start, row.length))
    bins = [list(chunk) for chunk in np.array_split(
        np.asarray(candidates, dtype=object), 3) if len(chunk)]
    for rows in bins:
        rng.shuffle(rows)
    ordered: list[_Candidate] = []
    while any(bins):
        for rows in bins:
            if rows:
                ordered.append(rows.pop())
    return ordered


def _select_disjoint(candidates: list[_Candidate], count: int) -> list[_Candidate]:
    used: dict[int, list[tuple[int, int]]] = {}
    out = []
    for candidate in candidates:
        end = candidate.start + candidate.length
        overlaps = any(candidate.start < prior_end and prior_start < end
                       for prior_start, prior_end in used.get(candidate.episode, []))
        if overlaps:
            continue
        used.setdefault(candidate.episode, []).append((candidate.start, end))
        out.append(candidate)
        if len(out) >= count:
            break
    return out


def _inject_wrong_branch(pose: np.ndarray, rows: np.ndarray) -> np.ndarray:
    observed = pose.copy()
    observed[rows, :3] += np.array([0.029, -0.003, 0.0])
    wrong = Rotation.from_euler("yx", [100.0, 8.0], degrees=True)
    observed[rows, 3:] = (
        wrong * Rotation.from_quat(pose[rows, 3:])).as_quat()
    return observed


def _native_errors(candidate: np.ndarray, truth: np.ndarray,
                   transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth_delta = truth[transitions + 1, :3] - truth[transitions, :3]
    candidate_delta = candidate[transitions + 1, :3] - candidate[transitions, :3]
    trans = np.linalg.norm(candidate_delta - truth_delta, axis=1) * 1000.0
    truth_relative = (Rotation.from_quat(truth[transitions + 1, 3:])
                      * Rotation.from_quat(truth[transitions, 3:]).inv())
    candidate_relative = (Rotation.from_quat(candidate[transitions + 1, 3:])
                          * Rotation.from_quat(candidate[transitions, 3:]).inv())
    rot = np.degrees((candidate_relative * truth_relative.inv()).magnitude())
    return trans, rot


def _summary(values: list[float]) -> MetricSummary:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return MetricSummary(float("inf"), float("inf"), 0)
    return MetricSummary(
        median=float(np.median(data)),
        p95=float(np.percentile(data, 95)),
        count=int(len(data)),
    )


def benchmark_task(
    clean_episodes: Iterable[np.ndarray],
    *,
    task: str,
    anomaly_lengths: Iterable[int] = (1, 5, 20, 60),
    max_intervals: int = 300,
    seed: int = 0,
) -> BenchmarkReport:
    """Mask held-out clean intervals and score reconstruction/action errors."""
    episodes = [_validate_episode(pose) for pose in clean_episodes]
    lengths = tuple(sorted({int(length) for length in anomaly_lengths
                            if int(length) > 0}))
    if not episodes:
        raise ValueError("at least one clean episode is required")
    if not lengths:
        raise ValueError("at least one positive anomaly length is required")
    if max_intervals <= 0:
        raise ValueError("max_intervals must be positive")
    rng = np.random.default_rng(int(seed))
    selected = _select_disjoint(
        _candidate_order(episodes, lengths, rng), int(max_intervals))

    translation: list[float] = []
    orientation: list[float] = []
    native_translation: list[float] = []
    native_rotation: list[float] = []
    gate_for_reconstruction = TaskGate(True, max(lengths))
    for index, interval in enumerate(selected):
        truth = episodes[interval.episode]
        all_rows = np.arange(interval.start, interval.start + interval.length)
        if index % 2:
            corrupt_rows = all_rows[::2]
            known_gaps: list[tuple[int, int]] = []
        else:
            corrupt_rows = all_rows
            known_gaps = [(interval.start, interval.length)]
        observed = _inject_wrong_branch(truth, corrupt_rows)
        result = repair_pose_stream(
            observed, "left", known_gaps,
            task_gate=gate_for_reconstruction)
        pose_error = np.linalg.norm(
            result.pose[all_rows, :3] - truth[all_rows, :3], axis=1) * 1000.0
        rot_error = _rotation_angle_deg(
            result.pose[all_rows, 3:], truth[all_rows, 3:])
        translation.extend(map(float, pose_error))
        orientation.extend(map(float, rot_error))
        lo = max(0, interval.start - 1)
        hi = min(len(truth) - 1, interval.start + interval.length)
        transitions = np.arange(lo, hi, dtype=int)
        trans_action, rot_action = _native_errors(
            result.pose, truth, transitions)
        native_translation.extend(map(float, trans_action))
        native_rotation.extend(map(float, rot_action))

    metrics = {
        "translation_mm": _summary(translation),
        "orientation_deg": _summary(orientation),
        "native_translation_mm": _summary(native_translation),
        "native_rotation_deg": _summary(native_rotation),
    }
    high_enabled = (
        len(selected) >= 100
        and metrics["translation_mm"].median <= 2.0
        and metrics["translation_mm"].p95 <= 10.0
        and metrics["orientation_deg"].median <= 1.0
        and metrics["orientation_deg"].p95 <= 5.0
        and metrics["native_translation_mm"].p95 <= 5.0
        and metrics["native_rotation_deg"].p95 <= 3.0)
    gate = TaskGate(
        high_confidence_enabled=bool(high_enabled),
        validated_max_frames=max(lengths) if high_enabled else 0)
    return BenchmarkReport(
        task=str(task), seed=int(seed), intervals=len(selected),
        anomaly_lengths=lengths, metrics=metrics, gate=gate)

