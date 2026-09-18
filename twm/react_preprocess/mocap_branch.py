"""Conservative body-frame branch proposals, separate from training admission.

A pose is a body-to-world transform T. Marker symmetry changes it to T @ B,
so correction must be right multiplication by inverse(B), including the body
translation lever arm. Subtracting a world offset destroys genuine motion.

Paired boundaries supply two independent estimates of B using only immediate
outer context. This never interpolates the interior of a persistent interval.
Even matching return pairs are hypotheses: real abrupt out-and-back motion can
be observationally indistinguishable. All results therefore remain MEDIUM;
method/task-specific validation and any human decision belong to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BranchConfig:
    rotation_seed_deg: float = 30.0
    coupled_rotation_deg: float = 12.0
    coupled_translation_mm: float = 20.0
    context_frames: int = 5
    max_interval_frames: int = 900
    transform_rotation_tolerance_deg: float = 3.0
    transform_translation_tolerance_mm: float = 5.0
    boundary_rotation_tolerance_deg: float = 3.0
    boundary_translation_tolerance_mm: float = 5.0
    max_step_rotation_deg: float = 30.0
    max_step_translation_mm: float = 50.0


@dataclass(frozen=True)
class BranchCandidate:
    start: int
    end: int
    pose: np.ndarray
    branch_rotation_xyzw: np.ndarray
    branch_translation_m: np.ndarray
    evidence: dict[str, float | int | bool | str]
    confidence: str = 'MEDIUM'


def _predict_next(context: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """One-step prediction; reversed context yields a backward prediction."""
    rotations = Rotation.from_quat(context[:, 3:])
    angular = np.median((rotations[1:]*rotations[:-1].inv()).as_rotvec(), axis=0)
    velocity = np.median(np.diff(context[:, :3], axis=0), axis=0)
    return context[-1, :3] + velocity, Rotation.from_rotvec(angular)*rotations[-1]


def _transform(expected: tuple[np.ndarray, Rotation], observed: np.ndarray):
    xyz, rotation = expected
    return rotation.inv().apply(observed[:3]-xyz), rotation.inv()*Rotation.from_quat(observed[3:])


def find_branch_candidates(
    pose: np.ndarray,
    known_gaps: Iterable[tuple[int, int]] = (),
    *,
    trusted: np.ndarray | None = None,
    config: BranchConfig | None = None,
) -> list[BranchCandidate]:
    """Propose complete paired-return intervals without changing input poses.

    ``trusted`` optionally identifies independently trusted outer anchors.
    Without it, anchors are only screened for finite, contiguous, quiet local
    observations and evidence records that limitation. Declared gaps and any
    nonfinite/zero-quaternion rows exclude both anchors and branch interiors.
    Only adjacent boundaries are paired: nested/chattering ambiguous branches
    are deliberately left for a richer decoder or human review.
    """
    raw = np.asarray(pose, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError('pose must have shape (T, 7)')
    cfg = config or BranchConfig()
    if cfg.context_frames < 2 or cfg.max_interval_frames < 1:
        raise ValueError('context_frames must be >=2 and max_interval_frames >=1')
    finite = np.isfinite(raw).all(axis=1) & (np.linalg.norm(raw[:, 3:], axis=1) > 1e-12)
    available = finite.copy()
    for start, count in known_gaps:
        if count > 0:
            available[max(0, int(start)):max(0, min(len(raw), int(start)+int(count)))] = False
    anchor_ok = available.copy()
    if trusted is not None:
        trusted = np.asarray(trusted, dtype=bool)
        if trusted.shape != (len(raw),):
            raise ValueError('trusted must have shape (T,)')
        anchor_ok &= trusted
    if len(raw) < 2*cfg.context_frames + 1:
        return []
    step_rot = np.full(len(raw)-1, np.nan)
    pairs = np.flatnonzero(finite[:-1] & finite[1:])
    if len(pairs):
        step_rot[pairs] = np.degrees((Rotation.from_quat(raw[pairs+1, 3:])*
                                      Rotation.from_quat(raw[pairs, 3:]).inv()).magnitude())
    step_trans = np.linalg.norm(np.diff(raw[:, :3], axis=0), axis=1)*1000
    edges = np.flatnonzero((step_rot > cfg.rotation_seed_deg) |
                          ((step_rot > cfg.coupled_rotation_deg) &
                           (step_trans > cfg.coupled_translation_mm)))
    candidates: list[BranchCandidate] = []
    used_until = -1
    for entry, exit_edge in zip(edges[:-1], edges[1:]):
        start, end = int(entry+1), int(exit_edge)
        left, right = start-cfg.context_frames, end+1+cfg.context_frames
        # A return boundary already used to close a proposal cannot also open
        # its inverse over the clean interval leading to the next occurrence.
        if start <= used_until+1 or left < 0 or right > len(raw):
            continue
        if end-start+1 > cfg.max_interval_frames or not available[start:end+1].all():
            continue
        if not anchor_ok[left:start].all() or not anchor_ok[end+1:right].all():
            continue
        # Contexts that cross another jump cannot establish either branch.
        context_edges = np.r_[np.arange(left, start-1), np.arange(end+1, right-1)]
        if (np.any(step_rot[context_edges] > cfg.coupled_rotation_deg) or
                np.any(step_trans[context_edges] > cfg.max_step_translation_mm)):
            continue
        expected_entry = _predict_next(raw[left:start])
        expected_exit = _predict_next(raw[end+1:right][::-1])
        entry_t, entry_r = _transform(expected_entry, raw[start])
        exit_t, exit_r = _transform(expected_exit, raw[end])
        transform_rot = float(np.degrees((entry_r.inv()*exit_r).magnitude()))
        transform_trans = float(np.linalg.norm(entry_t-exit_t)*1000)
        if (transform_rot > cfg.transform_rotation_tolerance_deg or
                transform_trans > cfg.transform_translation_tolerance_mm):
            continue
        branch_r = Rotation.from_quat(np.stack([entry_r.as_quat(), exit_r.as_quat()])).mean()
        branch_t = (entry_t+exit_t)/2
        corrected_r = Rotation.from_quat(raw[start:end+1, 3:])*branch_r.inv()
        corrected_t = raw[start:end+1, :3]-corrected_r.apply(branch_t)
        corrected = np.column_stack((corrected_t, corrected_r.as_quat()))
        boundary_trans = max(float(np.linalg.norm(corrected_t[i]-pred[0])*1000)
                             for i, pred in ((0, expected_entry), (-1, expected_exit)))
        boundary_rot = max(float(np.degrees((pred[1].inv()*corrected_r[i]).magnitude()))
                           for i, pred in ((0, expected_entry), (-1, expected_exit)))
        joined = np.vstack((raw[start-1], corrected, raw[end+1]))
        joined_r = Rotation.from_quat(joined[:, 3:])
        max_rot = float(np.max(np.degrees((joined_r[1:]*joined_r[:-1].inv()).magnitude())))
        max_trans = float(np.max(np.linalg.norm(np.diff(joined[:, :3], axis=0), axis=1)*1000))
        if (boundary_trans > cfg.boundary_translation_tolerance_mm or
                boundary_rot > cfg.boundary_rotation_tolerance_deg or
                max_rot > cfg.max_step_rotation_deg or
                max_trans > cfg.max_step_translation_mm):
            continue
        candidates.append(BranchCandidate(start, end, corrected, branch_r.as_quat(), branch_t, {
            'method': 'paired_body_branch',
            'duration_frames': end-start+1,
            'bidirectional_transform_agrees': True,
            'transform_rotation_disagreement_deg': transform_rot,
            'transform_translation_disagreement_mm': transform_trans,
            'boundary_prediction_rotation_max_deg': boundary_rot,
            'boundary_prediction_translation_max_mm': boundary_trans,
            'reconstructed_max_step_rotation_deg': max_rot,
            'reconstructed_max_step_translation_mm': max_trans,
            'left_anchor': start-1,
            'right_anchor': end+1,
            'context_frames': cfg.context_frames,
            'independent_anchor_mask_supplied': trusted is not None,
            'physical': True,
            'benchmark_validated': False,
        }))
        used_until = end
    # Support is restricted to this supplied recording/body stream; unrelated
    # epochs or rigid bodies must never be pooled implicitly.
    for index, candidate in enumerate(candidates):
        r = Rotation.from_quat(candidate.branch_rotation_xyzw)
        support = 0
        for other in candidates:
            rotation_error = np.degrees((r.inv()*Rotation.from_quat(other.branch_rotation_xyzw)).magnitude())
            translation_error = np.linalg.norm(candidate.branch_translation_m-other.branch_translation_m)*1000
            if (rotation_error <= cfg.transform_rotation_tolerance_deg and
                    translation_error <= cfg.transform_translation_tolerance_mm):
                support += 1
        candidates[index] = replace(candidate, evidence={**candidate.evidence,
                                                        'matching_return_pairs': support})
    return candidates
