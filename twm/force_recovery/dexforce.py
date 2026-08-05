"""Method 2 — force-informed position targets (DexForce-style).

The dataset's actions are computed from tracked sensor pose, so demonstrated
pose equals achieved pose and the force channel of a stiffness controller —
``F = k (target - actual)`` — is identically zero. DexForce
(arXiv:2501.10356) closes that gap without changing the action type: the
measured (here: estimated) contact force is converted into a *virtual
target* displaced past the contact surface,

    p_target = p_observed + (F_n / k) * n_hat

where ``n_hat`` is the pressing direction (the sensor surface normal, in
world frame from the OptiTrack quaternion) and ``k`` the stiffness an
impedance controller would run with at deployment. A policy trained on
these targets reproduces the demonstrated force through the controller,
with no force interface required; in free space ``F_n = 0`` and the target
is exactly the observed pose, so no-contact behaviour is untouched.

Frame convention: sensor pose is (x, y, z, qx, qy, qz, qw) in the world
frame; the gel faces along the sensor's local ``SENSOR_NORMAL_LOCAL`` axis.
The virtual target must move the *mount* toward the surface, i.e. along the
pressing direction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Impedance controller stiffness the targets are computed for. 300 N/m is a
# soft-to-moderate Cartesian stiffness (Franka-class arms run 150-600 N/m in
# compliant tasks). The penetration is F/k, so with forces around 0.05-1.2 N
# offsets stay in the 0.2-4 mm range — small enough to be safe, large
# enough to be learnable.
STIFFNESS_N_PER_M = 300.0

# The pressing direction in the rigid-body frame is NOT a coordinate axis:
# the rig's dual-ball calibration measures it as ``gel_axis_in_rigid``
# (pose-to-pose consistency ~1 degree), pointing outward through the gel —
# verified against the same file's geometry (gel_center = gelball - 5 mm *
# axis) and by the sign of the approach velocity at force onsets, which the
# naive [0, 0, 1] guess got wrong.
CALIB_ROOT = Path(__file__).resolve().parent.parent / "calibration"
CALIB_DIRS = {"pushT": "result", "motherboard": "result backup"}


def gel_axis(task: str, side: str) -> np.ndarray:
    """Calibrated pressing direction in the sensor's rigid-body frame."""
    path = CALIB_ROOT / CALIB_DIRS[task] / f"T_gel_to_rigid_{side}.json"
    axis = np.asarray(json.loads(path.read_text())["gel_axis_in_rigid"], np.float64)
    return axis / np.linalg.norm(axis)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw quaternions -> (N, 3, 3) rotation matrices."""
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = (q / np.maximum(n, 1e-12)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


@dataclass
class ForceInformedActions:
    target_pos: np.ndarray        # (T, 3) virtual targets, world frame
    penetration_m: np.ndarray     # (T,) displacement magnitude F/k
    normal_world: np.ndarray      # (T, 3) pressing direction in world
    stiffness: float

    @property
    def offset(self) -> np.ndarray:
        """(T, 3) target minus observed position."""
        return self.penetration_m[:, None] * self.normal_world


def force_informed_targets(pose: np.ndarray, force_n: np.ndarray,
                           normal_local: np.ndarray,
                           stiffness: float = STIFFNESS_N_PER_M,
                           ) -> ForceInformedActions:
    """Convert observed poses + estimated normal force into virtual targets.

    pose: (T, 7) x,y,z,qx,qy,qz,qw world-frame sensor poses
    force_n: (T,) estimated normal force, newtons (>= 0)
    normal_local: pressing direction in the rigid-body frame — use
        ``gel_axis(task, side)``, not a guessed coordinate axis
    """
    pose = np.asarray(pose, np.float64)
    force_n = np.clip(np.asarray(force_n, np.float64), 0.0, None)
    R = quat_to_matrix(pose[:, 3:7])
    normal_world = R @ np.asarray(normal_local, np.float64)
    penetration = force_n / stiffness
    return ForceInformedActions(
        target_pos=pose[:, :3] + penetration[:, None] * normal_world,
        penetration_m=penetration,
        normal_world=normal_world,
        stiffness=stiffness,
    )


def roundtrip_force(actions: ForceInformedActions,
                    observed_pos: np.ndarray) -> np.ndarray:
    """Force an impedance controller at ``stiffness`` would exert if the
    arm sat at the observed pose with these targets — the consistency
    check ``k * ||target - observed|| == F_estimated``."""
    return actions.stiffness * np.linalg.norm(
        actions.target_pos - np.asarray(observed_pos), axis=-1)
