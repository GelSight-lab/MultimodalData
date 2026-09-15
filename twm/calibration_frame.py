"""Which world a calibration is expressed in — declared, never inferred.

React's published parquet are Z-up. The per-epoch calibration directories
shipped beside them are Y-up and carry no `up_axis` key, so pairing them is
silent: the two forms of the same 2026-09-09 solve differ by 1.2733 in matrix
norm, the projected point stays inside the frame, and nothing raises. Tactile
readings are unaffected; the view-frame action is wrong by R_x(90).

The conversion is not a guess. On the published files, for all three tasks:

    data/validation/calibration/T_mocap_to_cam_left.json
      == data/<task>/calibration/epoch_2026-09-09/T_mocap_to_cam_left.json @ R^-1

`require_zup` exists so that a file which does not say what it is can never
again be used by accident. "No declaration" means stop, not "probably the one
I want" — the same rule the pipeline applies to every other resolver.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The world-frame rotation the release applies: (x, y, z) -> (x, -z, y).
YUP_TO_ZUP_4 = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], float)

NOTE = ("converted from the recorded Y-up by R_x(-90): (x,y,z)->(x,-z,y). "
        "T_mocap_to_cam moved with the poses, so every projection is "
        "unchanged; the view-frame action is not.")


def to_zup(path) -> np.ndarray:
    """Rotate one calibration JSON in place and declare the result.

    Refuses a file that already declares Z-up: the second rotation is
    invisible — projections stay in frame and the file still claims to be
    right — which is exactly how the first one went unnoticed.
    """
    path = Path(path)
    d = json.loads(path.read_text())
    if d.get("up_axis") == "z":
        raise ValueError(f"{path} already declares up_axis=z; rotating it "
                         f"again would be a net -180 deg that nothing "
                         f"downstream can see")
    T = np.asarray(d["T_mocap_to_cam"], float) @ np.linalg.inv(YUP_TO_ZUP_4)
    d["T_mocap_to_cam"] = T.tolist()
    d["up_axis"] = "z"
    d["up_axis_note"] = NOTE
    path.write_text(json.dumps(d, indent=1))
    return T


def require_zup(path):
    """(T_mocap_to_cam, intrinsics) for a calibration that SAYS it is Z-up."""
    path = Path(path)
    d = json.loads(path.read_text())
    up = d.get("up_axis")
    if up is None:
        raise ValueError(
            f"{path} does not declare up_axis. React's parquet are Z-up; this "
            f"file is most likely the Y-up original, and pairing them raises "
            f"nothing while putting the view-frame action out by R_x(90). "
            f"Convert it with twm.calibration_frame.to_zup.")
    if up != "z":
        raise ValueError(
            f"{path} declares up_axis={up!r}, but React's parquet are Z-up. "
            f"Convert it with twm.calibration_frame.to_zup.")
    return np.asarray(d["T_mocap_to_cam"], float), d["intrinsics"]
