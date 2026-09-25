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

# Impedance controller stiffness the targets are computed for. THE single
# definition of this quantity in the codebase — `pipeline.STIFFNESS_N_PER_MM`
# is derived from it, not declared again, because a stiffness that disagrees
# between the exported dataset column and the site figure is a silent lie
# about what the action means.
#
# SETTLED BY MEASUREMENT, as the previous note here said it should be.
#
# It read: "1000 N/m = 1 N/mm is the project's declared starting point ... the
# top of that band exceeds the 4.25 mm gel thickness, i.e. at high force this k
# commands a target further past the surface than the gel could ever be
# compressed — an argument for raising k, to be settled by the measured
# penetration distribution rather than by taste." The distribution is now
# measured over the whole release (480,080 force samples, 72 sides):
#
#     k [N/mm]   max penetration   rows past the 4.25 mm gel
#       1.00        7.870 mm              14.98%
#       1.85        4.254 mm               2.22%
#       2.00        3.935 mm               0.00%
#
# The binding constraint is max |F| = 7.870 N, which needs k >= 1.852 N/mm for
# the deepest commanded target to stay inside the gel. 2.0 N/mm clears it with
# margin and is still low-mid for Franka-class arms (~150-3000 N/m).
#
# This is an ASSUMPTION about the environment either way — raising it does not
# make it measured. What changed is that 1.0 was measurably WRONG: it commanded
# a target past the surface further than the gel can compress on 15% of rows,
# and `export_force_columns.verify` now fails rather than reports if that ever
# returns.
STIFFNESS_N_PER_M = 2000.0

# The pressing direction in the rigid-body frame is NOT a coordinate axis:
# the rig's dual-ball calibration measures it as ``gel_axis_in_rigid``
# (pose-to-pose consistency ~1 degree), pointing outward through the gel —
# verified against the same file's geometry (gel_center = gelball - 5 mm *
# axis) and by the sign of the approach velocity at force onsets, which the
# naive [0, 0, 1] guess got wrong.
# The task -> calibration-epoch mapping has exactly one home: calib_epoch.
# This module used to carry its own copy (the mapping existed in five files),
# which is how every motherboard preview shipped with pushT's extrinsics.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twm.calib_epoch import calib_dir as _calib_dir  # noqa: E402


# GelSight Mini's face is normal to the sensor body's y, so a compression
# acts along local -y. That is the default here.
GEL_AXIS_SOURCE_DEFAULT = "body_y"


def gel_axis(task: str, side: str, source: str | None = None) -> np.ndarray:
    """Pressing direction in the sensor's rigid-body frame.

    `source="body_y"` (the default) returns local -y: the GelSight Mini's
    sensing face is normal to the body's y axis, so a compression acts along
    -y.

    `source="dual_ball"` returns the calibrated `gel_axis_in_rigid`, kept
    because it is what the published files contain and what earlier results
    used.

    WHY -Y IS THE DEFAULT. `gel_axis_in_rigid` is
    normalize(gelball_centre - refball_centre): the line between two
    calibration ball centres 57 mm apart, from three poses. It never measured
    the gel surface. That line is the surface normal only if the fixture held
    both balls along it.

    Pressing hard (>6 N) on a level board, where the gel normal must point
    near world -z, measured over 38k frames:

        left   dual_ball  7.1 deg off      body_y  25.6 deg off
        right  dual_ball 18.1 deg off      body_y   7.7 deg off

    The two sensors disagree about which is better, so one calibration is
    wrong; the right one also carries `depth_offset_mm = 0.0` where the left
    carries -5.0, i.e. its ball centre was never backed off by a ball radius
    to reach the gel surface. Two independent signs pointing at the same
    file.

    Kinematics cannot arbitrate: sum(R_i) over contact frames has
    singular-value ratio 1.09 (left) and 1.04 (right), so the axis is not
    identifiable from motion, and the concentration score differs by 0.013
    between the two candidates. The board-normal comparison above is the only
    measurement with any power, and it is why the physical geometry wins.
    """
    src = source or GEL_AXIS_SOURCE_DEFAULT
    if src == "body_y":
        return np.array([0.0, -1.0, 0.0])
    if src != "dual_ball":
        raise ValueError(f"gel_axis: unknown source {src!r}; "
                         f"expected 'body_y' or 'dual_ball'")
    path = _calib_dir(task) / f"T_gel_to_rigid_{side}.json"
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
    normal_local = np.asarray(normal_local, np.float64)
    normal_local = normal_local / np.linalg.norm(normal_local)
    normal_world = R @ normal_local
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
