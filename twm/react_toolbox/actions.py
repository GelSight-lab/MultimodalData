"""Derive action targets from the handheld sensor poses.

React is a handheld dataset — there is no robot command. Actions are derived
from the OptiTrack 6-DoF sensor poses stored per frame as
[x, y, z, qx, qy, qz, qw] — position in METRES, quaternion SCALAR-LAST, the
`scipy.spatial.transform.Rotation.from_quat` convention. An earlier version of
this docstring said "wxyz"; the code was always scalar-last, so the text was
the error, but a reader who trusted it would have swapped w into x.

Rotation deltas here are WORLD-FRAME: dq = q[i+1] * q[i]^-1, integrated as
q[i+1] = dq * q[i]. Round-trip verified to 6e-9 m and 1.4e-5 deg.
"""
from __future__ import annotations

import numpy as np


def next_state_action(poses):
    """Absolute next-frame pose as the action (last frame repeats).

    poses: (T, 7) -> action: (T, 7), action[i] = poses[i+1], action[-1]=poses[-1].
    Matches the convention used by the React-lerobot export.
    """
    poses = np.asarray(poses, np.float32)
    return np.concatenate([poses[1:], poses[-1:]], axis=0)


def delta_pose_action(poses):
    """Frame-to-frame delta: translation diff + relative rotation (quat).

    Returns (T, 7): [dx,dy,dz, dqx,dqy,dqz,dqw], last row zero-translation +
    identity rotation. Quaternions are SCALAR-LAST throughout, and the rotation
    delta is world-frame: dq = q[i+1] * q[i]^-1.
    """
    p = np.asarray(poses, np.float64)
    T = p.shape[0]
    out = np.zeros((T, 7), np.float64)
    dt = p[1:, :3] - p[:-1, :3]
    out[:-1, :3] = dt
    q0 = _norm(p[:-1, 3:]); q1 = _norm(p[1:, 3:])
    out[:-1, 3:] = _quat_mul(q1, _quat_conj(q0))
    out[-1, 3:] = [0, 0, 0, 1]
    return out.astype(np.float32)


def integrate_delta(p0, deltas):
    """Inverse of delta_pose_action: recover absolute poses from p0 + deltas."""
    p0 = np.asarray(p0, np.float64)
    out = [p0.copy()]
    cur = p0.copy()
    for d in np.asarray(deltas, np.float64)[:-1]:
        nxt = np.empty(7)
        nxt[:3] = cur[:3] + d[:3]
        nxt[3:] = _quat_mul(d[3:], _norm(cur[3:]))
        out.append(nxt); cur = nxt
    return np.asarray(out, np.float32)


def _norm(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _quat_conj(q):
    c = q.copy(); c[..., :3] *= -1; return c   # scalar-last [x,y,z,w]


def _quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)
