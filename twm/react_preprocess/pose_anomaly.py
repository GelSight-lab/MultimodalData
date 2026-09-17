"""Explainable, conservative classification of OptiTrack pose discontinuities.

The detector deliberately separates a *candidate* (a physically surprising
single transition) from a tracking error.  A transition is automatically
repairable only when a second discontinuity returns to a compatible trajectory
and leaves trustworthy observations on both sides.  Unpaired changes are kept
for review; magnitude alone is not enough to rewrite a real motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


ROT_CANDIDATE_DEG = 30.0
TRANS_CANDIDATE_MM = 50.0
ROT_WITH_TRANSLATION_DEG = 12.0
RETURN_ROT_DEG = 12.0
RETURN_TRANS_MM = 20.0
EXIT_TRANS_MM = 10.0
MAX_RETURN_FRAMES = 40
MAX_REPAIR_FRAMES = 15


@dataclass(frozen=True)
class PoseEvent:
    side: str
    start: int
    end: int
    kind: str
    repairable: bool
    confidence: float
    bad_frames: tuple[int, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["bad_frames"] = list(self.bad_frames)
        return out


def _unit(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(_unit(a) * _unit(b), axis=-1)).clip(-1.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def transition_metrics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation degrees and translation millimetres for every transition."""
    pose = np.asarray(pose, float)
    if len(pose) < 2:
        return np.empty(0), np.empty(0)
    rot = _angle_deg(pose[:-1, 3:], pose[1:, 3:])
    trans = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) * 1000.0
    return rot, trans


def _in_gap(frame: int, gaps: list[tuple[int, int]]) -> bool:
    return any(start - 1 <= frame <= start + n for start, n in gaps)


def _return_event(pose: np.ndarray, side: str, transition: int,
                  rot: np.ndarray, trans: np.ndarray) -> PoseEvent | None:
    """Find the inverse boundary that closes one excursion."""
    start = transition + 1
    if transition == 0 or start >= len(pose) - 1:
        return None
    last_exit = min(len(pose) - 2, transition + MAX_RETURN_FRAMES)
    before = pose[transition]
    for exit_t in range(start, last_exit + 1):
        after = pose[exit_t + 1]
        return_rot = float(_angle_deg(before[None, 3:], after[None, 3:])[0])
        return_trans = float(np.linalg.norm(after[:3] - before[:3]) * 1000.0)
        exit_is_jump = (rot[exit_t] > ROT_CANDIDATE_DEG
                        or trans[exit_t] > EXIT_TRANS_MM)
        if not (exit_is_jump and return_rot < RETURN_ROT_DEG
                and return_trans < RETURN_TRANS_MM):
            continue
        end = exit_t
        n = end - start + 1
        evidence = {
            "jump_in_rotation_deg": float(rot[transition]),
            "jump_in_translation_mm": float(trans[transition]),
            "jump_out_rotation_deg": float(rot[exit_t]),
            "jump_out_translation_mm": float(trans[exit_t]),
            "return_rotation_deg": return_rot,
            "return_translation_mm": return_trans,
            "duration_frames": n,
        }
        return PoseEvent(
            side, start, end, "returning_excursion",
            n <= MAX_REPAIR_FRAMES, 0.98 if n <= MAX_REPAIR_FRAMES else 0.90,
            tuple(range(start, end + 1)), evidence)
    return None


def _merge_returning(events: list[PoseEvent]) -> list[PoseEvent]:
    """Combine nearby non-contiguous excursions into one A/B flicker event."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (e.start, e.end))
    groups: list[list[PoseEvent]] = [[events[0]]]
    for event in events[1:]:
        if event.start - groups[-1][-1].end <= 3:
            groups[-1].append(event)
        else:
            groups.append([event])
    out = []
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
            continue
        bad = tuple(i for event in group for i in event.bad_frames)
        ev = dict(group[0].evidence)
        ev["alternations"] = len(group)
        # Translation between the two branches is represented by the largest
        # incoming displacement, not by quaternion distance alone.
        ev["branch_translation_mm"] = max(
            float(e.evidence["jump_in_translation_mm"]) for e in group)
        out.append(PoseEvent(
            group[0].side, group[0].start, group[-1].end,
            "branch_flicker", all(e.repairable for e in group), 0.99,
            bad, ev))
    return out


def detect_pose_events(pose: np.ndarray, side: str,
                       known_gaps: Iterable[tuple[int, int]] = ()) -> list[PoseEvent]:
    """Classify surprising pose transitions without changing the pose stream."""
    pose = np.asarray(pose, float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must have shape (T, 7), got {pose.shape}")
    gaps = [(int(i), int(n)) for i, n in known_gaps]
    rot, trans = transition_metrics(pose)
    candidate = np.where((rot > ROT_CANDIDATE_DEG)
                         | ((rot > ROT_WITH_TRANSLATION_DEG)
                            & (trans > TRANS_CANDIDATE_MM)))[0]

    returning: list[PoseEvent] = []
    consumed: set[int] = set()
    for transition in candidate:
        transition = int(transition)
        if transition in consumed or _in_gap(transition, gaps):
            continue
        event = _return_event(pose, side, transition, rot, trans)
        if event is None:
            continue
        returning.append(event)
        # Both boundaries, plus candidates inside the bad span, belong to it.
        consumed.update(range(transition, event.end + 1))

    out = _merge_returning(returning)
    for transition in candidate:
        transition = int(transition)
        if transition in consumed or _in_gap(transition, gaps):
            continue
        edge = transition == 0 or transition == len(pose) - 2
        coupled = trans[transition] >= 20.0 and rot[transition] >= 60.0
        if edge:
            kind, confidence = "edge_discontinuity", 0.55
        elif coupled:
            kind, confidence = "persistent_branch", 0.80
        else:
            kind, confidence = "plausible_motion", 0.35
        out.append(PoseEvent(
            side, transition + 1, transition + 1, kind, False, confidence,
            (transition + 1,), {
                "jump_rotation_deg": float(rot[transition]),
                "jump_translation_mm": float(trans[transition]),
                "has_return_within_frames": False,
            }))

    for start, n in gaps:
        if n <= 0:
            continue
        out.append(PoseEvent(
            side, start, start + n - 1, "long_gap", False, 1.0,
            tuple(range(start, start + n)), {
                "duration_frames": n,
                "source": "pose_gaps.json",
            }))
    return sorted(out, key=lambda e: (e.start, e.end, e.kind))
