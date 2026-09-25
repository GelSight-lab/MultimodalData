"""The Z-up conversion must not rotate a calibration that already says z.

`stage_calibration` writes the declared epoch into `release/<task>/` already
converted and stamped `up_axis: z`. `convert_release_zup` then copies the
directory and rotates every `T_mocap_to_cam_*.json` again — unconditionally,
without reading the flag it itself writes.

Net effect: 180 degrees. Projections are unchanged, which is exactly why
nothing catches it downstream, and it is the reason `to_zup` refuses a second
rotation. This path went around that refusal by rotating inline.

Measured 2026-09-16: rope's staged calibration matched no epoch in any form
and differed from motherboard's and pushT's — which were copied by hand and
rotated once — by 1.2733, the Y-to-Z distance. The publish gate caught it:

    rope/2026-09-11 declares epoch 2026-09-09, but the left calibration
    shipped differs from it by 1.273

Both halves are tested, because skipping the rotation entirely would break
every tree that still stages a Y-up calibration.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The script resolves `react_paths` and `react_toolbox` relative to its own
# directory, the way it is run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm"))

from twm.calibration_frame import YUP_TO_ZUP_4


def _calib(d, T, up_axis=None):
    d.mkdir(parents=True, exist_ok=True)
    for cam in ("left", "middle", "right"):
        j = {"T_mocap_to_cam": np.asarray(T).tolist(),
             "intrinsics": {"fx": 1, "fy": 1, "ppx": 0, "ppy": 0}}
        if up_axis:
            j["up_axis"] = up_axis
        (d / f"T_mocap_to_cam_{cam}.json").write_text(json.dumps(j))
    return d


def _tree(root, up_axis, T):
    src = root / "release" / "rope"
    (src / "meta" / "2026-09-16").mkdir(parents=True)
    (src / "episodes.jsonl").write_text(json.dumps(
        {"episode": "2026-09-16/episode_000", "date": "2026-09-16",
         "up_axis": "z"}) + "\n")
    _calib(src / "calibration", T, up_axis)
    return src


def test_a_calibration_declaring_z_is_carried_through_unchanged(tmp_path):
    import twm.scripts.convert_release_zup as Z
    T = np.eye(4); T[0, 3] = 42.0
    src = _tree(tmp_path, "z", T)
    dst = tmp_path / "zup" / "rope"
    Z.convert_tree(src, dst, "rope")
    got = np.array(json.loads(
        (dst / "calibration/T_mocap_to_cam_left.json").read_text())["T_mocap_to_cam"])
    assert np.allclose(got, T), (
        "a calibration already in the Z-up frame was rotated again — a net "
        "180 degrees that every projection is blind to")


def test_a_calibration_with_no_declaration_is_still_converted(tmp_path):
    """The other half: a Y-up staging must still be rotated, or every tree
    that has not been through stage_calibration ships the wrong frame."""
    import twm.scripts.convert_release_zup as Z
    T = np.eye(4); T[1, 3] = 100.0
    src = _tree(tmp_path, None, T)
    dst = tmp_path / "zup" / "rope"
    Z.convert_tree(src, dst, "rope")
    d = json.loads((dst / "calibration/T_mocap_to_cam_left.json").read_text())
    assert d["up_axis"] == "z"
    assert not np.allclose(np.array(d["T_mocap_to_cam"]), T), \
        "a Y-up calibration was passed through without conversion"


def test_the_npy_follows_the_json(tmp_path):
    """Both are published; a reader taking the .npy must not get a different
    frame from one taking the .json."""
    import twm.scripts.convert_release_zup as Z
    T = np.eye(4); T[0, 3] = 42.0
    src = _tree(tmp_path, "z", T)
    np.save(src / "calibration" / "T_mocap_to_cam_left.npy", T)
    dst = tmp_path / "zup" / "rope"
    Z.convert_tree(src, dst, "rope")
    j = np.array(json.loads(
        (dst / "calibration/T_mocap_to_cam_left.json").read_text())["T_mocap_to_cam"])
    n = np.load(dst / "calibration" / "T_mocap_to_cam_left.npy")
    assert np.allclose(j, n)
