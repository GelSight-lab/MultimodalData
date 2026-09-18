"""Conservative, provenance-preserving OptiTrack trajectory repair.

This module starts by grouping discontinuity seeds into complete tracking-loss
bouts.  Detection is deliberately separate from reconstruction: a surprising
edge is evidence to inspect, never by itself permission to replace a pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RepairConfig:
    """Physical and temporal gates expressed at the native 30 fps rate."""

    rotation_seed_deg: float = 30.0
    coupled_rotation_deg: float = 12.0
    coupled_translation_mm: float = 20.0
    branch_agreement_deg: float = 12.0
    branch_context_frames: int = 15
    stable_frames: int = 10
    context_frames: int = 15
    translation_inlier_mm: float = 20.0
    rotation_inlier_deg: float = 12.0
    max_high_frames: int = 60
    max_step_translation_mm: float = 50.0
    max_step_rotation_deg: float = 30.0
    max_prediction_translation_mm: float = 10.0
    max_prediction_rotation_deg: float = 5.0


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


class Confidence(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class TaskGate:
    """Held-out clean-data evidence allowed to enable automatic training."""

    high_confidence_enabled: bool = False
    validated_max_frames: int = 0


@dataclass(frozen=True)
class RepairEvent:
    event_id: str
    bout: AnomalyBout
    confidence: Confidence
    method: str
    replaced_frames: tuple[int, ...]
    evidence: dict[str, float | int | bool | str]


@dataclass(frozen=True)
class PoseRepairResult:
    pose: np.ndarray
    repaired: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    event_id: np.ndarray
    events: tuple[RepairEvent, ...]


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
    # A declared gap already owns its entry and return boundaries.  Counting
    # the first clean row after it as another seed makes an N-row gap appear
    # to be N+1 rows and can incorrectly cross a confidence-duration gate.
    gap_boundaries: set[int] = set()
    for start, count in known_gaps:
        gap_boundaries.update(range(max(0, int(start) - 1),
                                    min(n_rows, int(start) + int(count) + 1)))
    discontinuity -= gap_boundaries
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


def _finite_pose_rows(pose: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(pose[:, 3:], axis=1)
    return np.isfinite(pose).all(axis=1) & (q_norm > 1e-12)


def _context_rows(finite: np.ndarray, start: int, end: int,
                  count: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.flatnonzero(finite[:start])[-count:]
    right = np.flatnonzero(finite[end + 1:])[:count] + end + 1
    return left, right


def _median_linear_velocity(xyz: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if len(rows) < 2:
        return np.zeros(3, dtype=float)
    consecutive = np.diff(rows) == 1
    delta = np.diff(xyz[rows], axis=0)[consecutive]
    return np.median(delta, axis=0) if len(delta) else np.zeros(3, dtype=float)


def _median_angular_velocity(quat: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if len(rows) < 2:
        return np.zeros(3, dtype=float)
    pairs = np.flatnonzero(np.diff(rows) == 1)
    if not len(pairs):
        return np.zeros(3, dtype=float)
    first = Rotation.from_quat(quat[rows[pairs]])
    second = Rotation.from_quat(quat[rows[pairs + 1]])
    return np.median((second * first.inv()).as_rotvec(), axis=0)


def _hermite(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray,
             v1: np.ndarray, duration: int, offsets: np.ndarray) -> np.ndarray:
    s = (offsets / float(duration))[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(
        (Rotation.from_quat(a) * Rotation.from_quat(b).inv()).magnitude())


def _predict_bout(pose: np.ndarray, bout: AnomalyBout, left: np.ndarray,
                  right: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Boundary-constrained SE(3) prediction and independent check evidence."""
    lo, hi = int(left[-1]), int(right[0])
    rows = np.arange(bout.start, bout.end + 1)
    duration = hi - lo
    offsets = rows - lo
    xyz = pose[:, :3]
    quat = _unit_quaternion(pose[:, 3:])
    v0 = _median_linear_velocity(xyz, left)
    v1 = _median_linear_velocity(xyz, right)
    # Both effectively stationary: the minimum-assumption path is linear.
    if max(np.linalg.norm(v0), np.linalg.norm(v1)) < 0.0005:
        v0 = v1 = (xyz[hi] - xyz[lo]) / duration
    pred_xyz = _hermite(xyz[lo], xyz[hi], v0, v1, duration, offsets)

    r0, r1 = Rotation.from_quat(quat[lo]), Rotation.from_quat(quat[hi])
    end_rotvec = (r1 * r0.inv()).as_rotvec()
    w0 = _median_angular_velocity(quat, left)
    w1 = _median_angular_velocity(quat, right)
    pred_rotvec = _hermite(
        np.zeros(3), end_rotvec, w0, w1, duration, offsets)
    pred_quat = (Rotation.from_rotvec(pred_rotvec) * r0).as_quat()

    forward_xyz = xyz[lo] + offsets[:, None] * v0
    backward_xyz = xyz[hi] - (hi - rows)[:, None] * v1
    pred_trans_disagreement = float(np.max(
        np.linalg.norm(forward_xyz - backward_xyz, axis=1) * 1000.0))
    forward_q = (Rotation.from_rotvec(offsets[:, None] * w0) * r0).as_quat()
    backward_q = (
        Rotation.from_rotvec(-(hi - rows)[:, None] * w1) * r1).as_quat()
    pred_rot_disagreement = float(np.max(
        _rotation_error_deg(forward_q, backward_q)))
    pred = np.concatenate([pred_xyz, pred_quat], axis=1)
    evidence = {
        "left_anchor": lo,
        "right_anchor": hi,
        "left_context_frames": int(len(left)),
        "right_context_frames": int(len(right)),
        "prediction_translation_max_mm": pred_trans_disagreement,
        "prediction_rotation_max_deg": pred_rot_disagreement,
    }
    return pred, evidence


def _trajectory_is_physical(pose: np.ndarray, config: RepairConfig) -> tuple[bool, float, float]:
    rot, trans = transition_metrics(pose)
    finite = np.isfinite(rot) & np.isfinite(trans)
    if not finite.all():
        return False, float("inf"), float("inf")
    max_rot = float(np.max(rot, initial=0.0))
    max_trans = float(np.max(trans, initial=0.0))
    return (max_rot <= config.max_step_rotation_deg
            and max_trans <= config.max_step_translation_mm), max_rot, max_trans


def _residuals(observed: np.ndarray,
               predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(observed).all(axis=1)
    trans = np.full(len(observed), np.inf)
    rot = np.full(len(observed), np.inf)
    trans[finite] = np.linalg.norm(
        observed[finite, :3] - predicted[finite, :3], axis=1) * 1000.0
    rot[finite] = _rotation_error_deg(
        observed[finite, 3:], predicted[finite, 3:])
    return trans, rot, finite


def _refit_from_inliers(predicted: np.ndarray, observed: np.ndarray,
                        inlier: np.ndarray, offsets: np.ndarray,
                        duration: int) -> np.ndarray:
    """Fit a boundary-zero cubic correction without moving either anchor."""
    if int(inlier.sum()) < 2:
        return predicted
    s = offsets / float(duration)
    basis = np.stack([s * (1.0 - s),
                      s * (1.0 - s) * (2.0 * s - 1.0)], axis=1)
    design = basis[inlier]
    xyz_residual = observed[inlier, :3] - predicted[inlier, :3]
    xyz_coef, *_ = np.linalg.lstsq(design, xyz_residual, rcond=None)
    corrected_xyz = predicted[:, :3] + basis @ xyz_coef

    measured_r = Rotation.from_quat(observed[inlier, 3:])
    predicted_r = Rotation.from_quat(predicted[inlier, 3:])
    rot_residual = (measured_r * predicted_r.inv()).as_rotvec()
    rot_coef, *_ = np.linalg.lstsq(design, rot_residual, rcond=None)
    correction = Rotation.from_rotvec(basis @ rot_coef)
    corrected_q = (correction * Rotation.from_quat(predicted[:, 3:])).as_quat()
    return np.concatenate([corrected_xyz, corrected_q], axis=1)


def _robust_upper(values: np.ndarray, floor: float, cap: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return float(min(cap, max(floor, centre + 6.0 * 1.4826 * mad)))


def _local_inlier_thresholds(pose: np.ndarray, left: np.ndarray,
                             right: np.ndarray,
                             config: RepairConfig) -> tuple[float, float]:
    """Estimate clean local jitter, bounded by global physical residual caps."""
    trans_accel: list[np.ndarray] = []
    rot_accel: list[np.ndarray] = []
    for rows in (left, right):
        runs = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
        for run in runs:
            if len(run) < 3:
                continue
            xyz = pose[run, :3]
            trans_accel.append(
                np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=1) * 1000.0)
            rotations = Rotation.from_quat(_unit_quaternion(pose[run, 3:]))
            velocity = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
            rot_accel.append(np.degrees(
                np.linalg.norm(np.diff(velocity, axis=0), axis=1)))
    trans_values = np.concatenate(trans_accel) if trans_accel else np.empty(0)
    rot_values = np.concatenate(rot_accel) if rot_accel else np.empty(0)
    return (
        _robust_upper(trans_values, 3.0, config.translation_inlier_mm),
        _robust_upper(rot_values, 2.0, config.rotation_inlier_deg),
    )


def _robust_branch_fit(observed: np.ndarray, initial: np.ndarray,
                       forced: np.ndarray, offsets: np.ndarray, duration: int,
                       translation_threshold_mm: float,
                       rotation_threshold_deg: float) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                           int, bool]:
    """Alternate branch classification and trajectory correction up to 5 times."""
    predicted = initial.copy()
    previous: np.ndarray | None = None
    converged = False
    iterations = 0
    for iterations in range(1, 6):
        trans, rot, finite = _residuals(observed, predicted)
        inlier = (finite
                  & (trans <= translation_threshold_mm)
                  & (rot <= rotation_threshold_deg)
                  & ~forced)
        if previous is not None and np.array_equal(inlier, previous):
            converged = True
            break
        previous = inlier.copy()
        predicted = _refit_from_inliers(
            predicted, observed, inlier, offsets, duration)
    trans, rot, finite = _residuals(observed, predicted)
    inlier = (finite
              & (trans <= translation_threshold_mm)
              & (rot <= rotation_threshold_deg)
              & ~forced)
    if previous is not None and np.array_equal(inlier, previous):
        converged = True
    return predicted, inlier, trans, rot, iterations, converged


def repair_pose_stream(
    pose: np.ndarray,
    side: str,
    known_gaps: Iterable[tuple[int, int]] = (),
    *,
    task_gate: TaskGate | None = None,
    config: RepairConfig | None = None,
) -> PoseRepairResult:
    """Return candidate poses and explicit side-specific repair provenance.

    HIGH candidates are valid for training.  MEDIUM candidates contain the
    proposed reconstruction but remain invalid pending a review decision.
    LOW events preserve the measured poses and remain invalid.
    """
    raw = np.asarray(pose, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"pose must have shape (T, 7), got {raw.shape}")
    cfg = config or RepairConfig()
    gate = task_gate or TaskGate()
    gap_list = [(int(start), int(count)) for start, count in known_gaps]
    bouts = detect_bouts(raw, side, gap_list, config=cfg)
    finite = _finite_pose_rows(raw)
    candidate = raw.copy()
    repaired = np.zeros(len(raw), dtype=bool)
    confidence = np.full(len(raw), Confidence.NONE, dtype=np.uint8)
    valid = finite.copy()
    event_id = np.full(len(raw), "", dtype=object)
    events: list[RepairEvent] = []
    known_rows = {
        row for start, count in gap_list
        for row in range(max(0, start), min(len(raw), start + count))
    }

    for bout in bouts:
        rows = np.arange(bout.start, bout.end + 1)
        left, right = _context_rows(finite, bout.start, bout.end,
                                    cfg.context_frames)
        eid = f"{side}:{bout.start}-{bout.end}:{bout.kind}"
        event_id[rows] = eid
        if not len(left) or not len(right):
            confidence[rows] = Confidence.LOW
            valid[rows] = False
            events.append(RepairEvent(
                eid, bout, Confidence.LOW, "unrepaired_missing_anchor", (), {
                    "duration_frames": int(len(rows)),
                    "left_context_frames": int(len(left)),
                    "right_context_frames": int(len(right)),
                }))
            continue

        predicted, evidence = _predict_bout(raw, bout, left, right)
        observed = raw[rows]
        forced = np.array([int(row) in known_rows for row in rows])
        lo = int(left[-1]); hi = int(right[0])
        trans_threshold, rot_threshold = _local_inlier_thresholds(
            raw, left, right, cfg)
        predicted, inlier, trans_residual, rot_residual, iterations, converged = (
            _robust_branch_fit(observed, predicted, forced, rows - lo,
                               hi - lo, trans_threshold, rot_threshold))
        replace = ~inlier | forced
        classified_outlier = replace & ~forced
        if classified_outlier.any():
            outlier_separation = np.maximum(
                trans_residual[classified_outlier] / trans_threshold,
                rot_residual[classified_outlier] / rot_threshold)
            branch_unambiguous = bool(np.min(outlier_separation) >= 1.5)
        else:
            branch_unambiguous = True
        proposed = raw.copy()
        proposed[rows[replace]] = predicted[replace]
        physical, max_rot, max_trans = _trajectory_is_physical(
            proposed[max(0, bout.start - 1):min(len(raw), bout.end + 2)], cfg)
        evidence.update({
            "duration_frames": int(len(rows)),
            "observed_inlier_frames": int(inlier.sum()),
            "replaced_frames": int(replace.sum()),
            "translation_residual_p95_mm": float(np.percentile(trans_residual, 95)),
            "rotation_residual_p95_deg": float(np.percentile(rot_residual, 95)),
            "reconstructed_max_step_translation_mm": max_trans,
            "reconstructed_max_step_rotation_deg": max_rot,
            "physical": bool(physical),
            "fit_iterations": int(iterations),
            "fit_converged": bool(converged),
            "translation_inlier_threshold_mm": trans_threshold,
            "rotation_inlier_threshold_deg": rot_threshold,
            "branch_unambiguous": branch_unambiguous,
        })
        extreme_prediction_disagreement = (
            evidence["prediction_translation_max_mm"]
            > 3.0 * cfg.max_prediction_translation_mm
            or evidence["prediction_rotation_max_deg"]
            > 3.0 * cfg.max_prediction_rotation_deg)
        if not physical or extreme_prediction_disagreement:
            confidence[rows] = Confidence.LOW
            valid[rows] = False
            evidence["rejection"] = (
                "nonphysical_reconstruction" if not physical
                else "incompatible_two_sided_prediction")
            events.append(RepairEvent(
                eid, bout, Confidence.LOW, "unrepaired_ambiguous_branch", (),
                evidence))
            continue
        two_full_contexts = (len(left) >= cfg.context_frames
                             and len(right) >= cfg.context_frames)
        prediction_agrees = (
            evidence["prediction_translation_max_mm"]
            <= cfg.max_prediction_translation_mm
            and evidence["prediction_rotation_max_deg"]
            <= cfg.max_prediction_rotation_deg)
        high = (
            gate.high_confidence_enabled
            and len(rows) <= min(cfg.max_high_frames, gate.validated_max_frames)
            and two_full_contexts and prediction_agrees and physical
            and branch_unambiguous and bool(replace.any()))
        tier = Confidence.HIGH if high else Confidence.MEDIUM
        candidate[rows[replace]] = predicted[replace]
        repaired[rows[replace]] = True
        confidence[rows] = tier
        valid[rows] = high
        events.append(RepairEvent(
            eid, bout, tier, "robust_se3_hermite",
            tuple(rows[replace].astype(int)), evidence))

    return PoseRepairResult(
        pose=candidate,
        repaired=repaired,
        confidence=confidence,
        valid=valid,
        event_id=event_id,
        events=tuple(events),
    )
