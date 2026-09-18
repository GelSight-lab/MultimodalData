"""Repair-aware action construction from candidate OptiTrack poses.

Actions are 9-D ``[dx, dy, dz, relative_rotation_6d]``.  Translation is in
metres.  Rotation uses the first two columns of the world-frame relative
rotation matrix, concatenated column by column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


MAX_ROTATION_DEG = 30.0
MAX_TRANSLATION_MM = 50.0


@dataclass(frozen=True)
class ActionSeries:
    values: np.ndarray
    valid: np.ndarray
    repaired: np.ndarray
    event_ids: tuple[tuple[str, ...], ...]
    start_rows: np.ndarray
    end_rows: np.ndarray
    rotation_deg: np.ndarray
    translation_mm: np.ndarray


def _check_pose_arrays(pose: np.ndarray, pose_valid: np.ndarray,
                       pose_repaired: np.ndarray,
                       pose_event_id: np.ndarray) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must have shape (T, 7), got {pose.shape}")
    valid = np.asarray(pose_valid, dtype=bool)
    repaired = np.asarray(pose_repaired, dtype=bool)
    event_id = np.asarray(pose_event_id, dtype=object)
    for name, value in (("pose_valid", valid), ("pose_repaired", repaired),
                        ("pose_event_id", event_id)):
        if value.shape != (len(pose),):
            raise ValueError(f"{name} must have shape ({len(pose)},), got {value.shape}")
    return pose, valid, repaired, event_id


def _event_union(*values: object) -> tuple[str, ...]:
    out: set[str] = set()
    for value in values:
        if isinstance(value, str):
            if value:
                out.add(value)
        elif value is not None:
            out.update(str(item) for item in value if str(item))
    return tuple(sorted(out))


def _relative_values(pose: np.ndarray, start: np.ndarray,
                     end: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(start)
    values = np.zeros((n, 9), dtype=np.float32)
    values[:, 3:] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rotation_deg = np.full(n, np.inf, dtype=float)
    translation_mm = np.full(n, np.inf, dtype=float)
    endpoints_finite = np.isfinite(pose[start]).all(axis=1) & np.isfinite(pose[end]).all(axis=1)
    q0_norm = np.linalg.norm(pose[start, 3:], axis=1)
    q1_norm = np.linalg.norm(pose[end, 3:], axis=1)
    endpoints_finite &= (q0_norm > 1e-12) & (q1_norm > 1e-12)
    usable = np.flatnonzero(endpoints_finite)
    if len(usable):
        s, e = start[usable], end[usable]
        delta = pose[e, :3] - pose[s, :3]
        q0 = pose[s, 3:] / np.linalg.norm(pose[s, 3:], axis=1, keepdims=True)
        q1 = pose[e, 3:] / np.linalg.norm(pose[e, 3:], axis=1, keepdims=True)
        relative = Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()
        matrix = relative.as_matrix()
        rot6d = np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=1)
        values[usable, :3] = delta.astype(np.float32)
        values[usable, 3:] = rot6d.astype(np.float32)
        rotation_deg[usable] = np.degrees(relative.magnitude())
        translation_mm[usable] = np.linalg.norm(delta, axis=1) * 1000.0
    return values, rotation_deg, translation_mm, endpoints_finite


def native_actions(pose: np.ndarray, pose_valid: np.ndarray,
                   pose_repaired: np.ndarray,
                   pose_event_id: np.ndarray) -> ActionSeries:
    """Build each native transition ``k -> k+1`` with explicit validity."""
    pose, pose_valid, pose_repaired, pose_event_id = _check_pose_arrays(
        pose, pose_valid, pose_repaired, pose_event_id)
    start = np.arange(max(0, len(pose) - 1), dtype=np.int32)
    end = start + 1
    values, rotation_deg, translation_mm, finite = _relative_values(
        pose, start, end)
    valid = (pose_valid[start] & pose_valid[end] & finite
             & (rotation_deg <= MAX_ROTATION_DEG)
             & (translation_mm <= MAX_TRANSLATION_MM))
    repaired = pose_repaired[start] | pose_repaired[end]
    event_ids = tuple(_event_union(pose_event_id[a], pose_event_id[b])
                      for a, b in zip(start, end))
    return ActionSeries(values, valid, repaired, event_ids, start, end,
                        rotation_deg, translation_mm)


def actions_fps15(pose: np.ndarray, native: ActionSeries) -> ActionSeries:
    """Build ``k -> k+2`` actions; validity is AND and repair is OR."""
    pose = np.asarray(pose, dtype=float)
    expected_native = max(0, len(pose) - 1)
    if len(native.valid) != expected_native:
        raise ValueError(
            f"native has {len(native.valid)} rows for {len(pose)} poses; "
            f"expected {expected_native}")
    start = np.arange(0, max(0, len(pose) - 2), 2, dtype=np.int32)
    end = start + 2
    values, rotation_deg, translation_mm, _ = _relative_values(pose, start, end)
    first = start
    second = start + 1
    valid = native.valid[first] & native.valid[second]
    repaired = native.repaired[first] | native.repaired[second]
    event_ids = tuple(_event_union(native.event_ids[a], native.event_ids[b])
                      for a, b in zip(first, second))
    return ActionSeries(values, valid, repaired, event_ids, start, end,
                        rotation_deg, translation_mm)

