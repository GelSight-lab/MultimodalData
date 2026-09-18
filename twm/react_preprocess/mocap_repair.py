"""Conservative, provenance-preserving OptiTrack trajectory repair.

This module starts by grouping discontinuity seeds into complete tracking-loss
bouts.  Detection is deliberately separate from reconstruction: a surprising
edge is evidence to inspect, never by itself permission to replace a pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RepairConfig:
    """Physical and temporal gates expressed at the native 30 fps rate."""

    rotation_seed_deg: float = 30.0
    coupled_rotation_deg: float = 12.0
    coupled_translation_mm: float = 20.0
    branch_agreement_deg: float = 12.0
    branch_context_frames: int = 15
    stable_frames: int = 10


@dataclass(frozen=True)
class AnomalyBout:
    """One complete interval requiring reconstruction or review."""

    side: str
    start: int
    end: int
    kind: str
    seed_frames: tuple[int, ...]
    left_context_end: int | None
    right_context_start: int | None


def _unit_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(norm > 1e-12, norm, 1.0)


def _quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _unit_quaternion(a), _unit_quaternion(b)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def transition_metrics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sign-insensitive rotation degrees and translation millimetres."""
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must have shape (T, 7), got {pose.shape}")
    if len(pose) < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    finite = np.isfinite(pose).all(axis=1)
    rot = _quaternion_angle_deg(pose[:-1, 3:], pose[1:, 3:])
    trans = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) * 1000.0
    invalid = ~(finite[:-1] & finite[1:])
    rot[invalid] = np.nan
    trans[invalid] = np.nan
    return rot, trans


def _branch_support(pose: np.ndarray, rows: set[int], config: RepairConfig) -> dict[int, float]:
    """Fraction of nearby finite observations agreeing with each candidate row."""
    q = np.asarray(pose, dtype=float)[:, 3:]
    finite = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 1e-12)
    out: dict[int, float] = {}
    for row in sorted(rows):
        if not finite[row]:
            out[row] = -1.0
            continue
        lo = max(0, row - config.branch_context_frames)
        hi = min(len(q), row + config.branch_context_frames + 1)
        neighbours = np.arange(lo, hi)
        neighbours = neighbours[(neighbours != row) & finite[lo:hi]]
        if not len(neighbours):
            out[row] = 1.0
            continue
        angle = _quaternion_angle_deg(
            np.repeat(q[row][None], len(neighbours), axis=0), q[neighbours])
        out[row] = float(np.mean(angle <= config.branch_agreement_deg))
    return out


def _discontinuity_seed_frames(pose: np.ndarray, config: RepairConfig) -> set[int]:
    rot, trans = transition_metrics(pose)
    edges = np.where(
        (rot > config.rotation_seed_deg)
        | ((rot > config.coupled_rotation_deg)
           & (trans > config.coupled_translation_mm))
    )[0]
    if not len(edges):
        return set()
    endpoints = {int(i) for edge in edges for i in (edge, edge + 1)}
    support = _branch_support(pose, endpoints, config)
    seeds: set[int] = set()
    for edge in edges:
        left, right = int(edge), int(edge + 1)
        ls, rs = support[left], support[right]
        # The alternate OptiTrack branch has less local temporal support.  A
        # tie remains a review seed on the post-transition side; reconstruction
        # later requires two trustworthy anchors before changing anything.
        seeds.add(left if ls < rs else right)
    return seeds


def detect_bouts(
    pose: np.ndarray,
    side: str,
    known_gaps: Iterable[tuple[int, int]] = (),
    *,
    config: RepairConfig | None = None,
) -> list[AnomalyBout]:
    """Group discontinuities and declared gaps into complete anomaly bouts.

    Seeds separated by fewer than ``stable_frames`` clean native frames are
    one bout.  Consequently a returning A/B flicker is represented once rather
    than as a list of unrelated >30 degree transitions.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must have shape (T, 7), got {pose.shape}")
    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    cfg = config or RepairConfig()
    n_rows = len(pose)
    known: set[int] = set()
    for start, count in known_gaps:
        start, count = int(start), int(count)
        if count > 0:
            known.update(range(max(0, start), min(n_rows, start + count)))
    nonfinite = set(np.where(~np.isfinite(pose).all(axis=1))[0].astype(int))
    discontinuity = _discontinuity_seed_frames(pose, cfg)
    seeds = sorted(known | nonfinite | discontinuity)
    if not seeds:
        return []

    groups: list[list[int]] = [[seeds[0]]]
    for row in seeds[1:]:
        if row - groups[-1][-1] <= cfg.stable_frames:
            groups[-1].append(row)
        else:
            groups.append([row])

    bouts: list[AnomalyBout] = []
    for group in groups:
        start, end = group[0], group[-1]
        group_set = set(group)
        if group_set <= known and group_set:
            kind = "known_gap"
        elif group_set & known:
            kind = "gap_with_discontinuity"
        elif group_set & nonfinite:
            kind = "nonfinite"
        else:
            kind = "branch_discontinuity"
        left = start - 1 if start > 0 else None
        right = end + 1 if end + 1 < n_rows else None
        bouts.append(AnomalyBout(
            side=side,
            start=start,
            end=end,
            kind=kind,
            seed_frames=tuple(group),
            left_context_end=left,
            right_context_start=right,
        ))
    return bouts

