"""Which way is up — declared, and converted as one piece or not at all.

WHAT THE DATA ACTUALLY IS

OptiTrack records **Y-up, right-handed**. Measured on this release: the table
normal in world coordinates is (0.053, 0.997, 0.056), i.e. 4.4 degrees off +y,
and both the pose rotations and `T_mocap_to_cam` have determinant +1.

That convention was documented nowhere. Robotics code overwhelmingly assumes
Z-up, so a reader taking `pose[2]` as height gets a horizontal coordinate and
nothing complains — the numbers are plausible, the plots look fine, and the
error only shows up as a model that never learns which way gravity points.

THE CONVERSION, AND THE TRAP IN IT

Y-up to Z-up preserving handedness is a rotation of -90 degrees about x:

    (x, y, z)  ->  (x, -z, y)

There are two right-handed candidates; this is the one that sends UP to **+z**
rather than -z. The other, R_x(+90), is equally right-handed and puts the table
normal at (0.053, 0.056, -0.997) — upside down, det still +1, and no assertion
about handedness would catch it.

THE TRAP: a world-frame rotation must be applied to the POSES **and** to
`T_mocap_to_cam` together. Applied to one only, every projection moves and
nothing raises. So this module converts a BUNDLE — poses and cameras — and
`scripts/test_frames.py` asserts that projections are bit-identical afterwards.
Invariance is the test: if a pixel moves, the conversion was half-applied.

The gel centre is in the sensor's own RIGID frame and is therefore untouched.
"""
from __future__ import annotations

import numpy as np

UP_AXIS_RECORDED = "y"
UP_AXIS_ROBOTICS = "z"

# (x, y, z)_yup -> (x, -z, y)_zup.  det = +1, and it sends +y to +z.
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])
ZUP_TO_YUP = YUP_TO_ZUP.T


def _R(to_zup: bool) -> np.ndarray:
    return YUP_TO_ZUP if to_zup else ZUP_TO_YUP


def convert_poses(poses7, to_zup: bool = True) -> np.ndarray:
    """Rotate a pose array into the other up-convention. (N, 7) or (7,).

    Positions rotate; orientations PRE-multiply, because this is a change of
    the world frame, not a motion of the body.
    """
    from scipy.spatial.transform import Rotation

    p = np.atleast_2d(np.asarray(poses7, float)).copy()
    M = _R(to_zup)
    ok = np.isfinite(p).all(1) & (np.linalg.norm(p[:, 3:7], axis=1) > 0.5)
    p[ok, :3] = p[ok, :3] @ M.T
    p[ok, 3:7] = (Rotation.from_matrix(M) * Rotation.from_quat(p[ok, 3:7])).as_quat()
    return p[0] if np.ndim(poses7) == 1 else p


def convert_calibration(cal: dict, to_zup: bool = True) -> dict:
    """The same rotation applied to every camera extrinsic. Returns a copy.

    `T_mocap_to_cam` maps mocap -> camera. Rotating the mocap frame by M means
    the new matrix is `T @ M^-1`, so that `T_new @ (M @ x) == T @ x` and every
    projection is unchanged. Getting this inverse backwards is the other way to
    break it silently; the invariance test catches that too.
    """
    M = _R(to_zup)
    out = {k: v for k, v in cal.items()}
    out["cams"] = {}
    for name, c in cal["cams"].items():
        T = np.asarray(c["T_mocap_to_cam"], float).copy()
        T[:3, :3] = T[:3, :3] @ M.T           # M^-1 == M.T for a rotation
        out["cams"][name] = {**c, "T_mocap_to_cam": T}
    return out


def to_zup(poses7, cal: dict):
    """Convert a pose array and its calibration together. Returns (poses, cal).

    Use this rather than the two halves: the whole point is that they move as
    one piece.
    """
    return convert_poses(poses7, True), convert_calibration(cal, True)
