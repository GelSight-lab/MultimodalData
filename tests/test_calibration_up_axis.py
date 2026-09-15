"""A calibration that does not say which world it is in cannot be paired safely.

React's parquet are Z-up. The per-epoch calibration directories published
beside them — `data/{motherboard,pushT,rope}/calibration/epoch_2026-09-09/` —
are Y-up and carry NO `up_axis` key. Only `validation`'s was converted.

Pairing them is silent. Measured on the hub 2026-09-14: the Y-up and Z-up
forms of the same 2026-09-09 solve differ by 1.2733 in matrix norm, so the
projected point stays inside the frame and nothing raises. Tactile is
unaffected. The view-frame action is wrong by R_x(90).

Proven on the published files, all three tasks:

    data/validation/calibration/T_mocap_to_cam_left.json
        == data/<task>/calibration/epoch_2026-09-09/T_mocap_to_cam_left.json @ R^-1

so the conversion is not a guess. Two things follow: the conversion must be
applied, and an UNLABELLED calibration must never again be usable by accident.
"""
import json

import numpy as np
import pytest

from twm.calibration_frame import YUP_TO_ZUP_4, require_zup, to_zup


def _calib(path, T, *, up_axis=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {"T_mocap_to_cam": np.asarray(T).tolist(),
         "intrinsics": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0,
                        "width": 640, "height": 480}}
    if up_axis:
        d["up_axis"] = up_axis
    path.write_text(json.dumps(d))
    return path


def test_the_conversion_is_the_one_the_published_pair_proves(tmp_path):
    """`validation` == `epoch` @ R^-1 on the hub. Reproduce that exactly."""
    rng = np.random.default_rng(0)
    T_y = np.eye(4)
    T_y[:3, :3] = rng.normal(size=(3, 3))
    T_y[:3, 3] = [158.4, 284.9, 685.7]
    p = _calib(tmp_path / "T_mocap_to_cam_left.json", T_y)
    to_zup(p)
    got = np.array(json.loads(p.read_text())["T_mocap_to_cam"])
    assert np.allclose(got, T_y @ np.linalg.inv(YUP_TO_ZUP_4))


def test_the_converted_file_says_so(tmp_path):
    p = _calib(tmp_path / "T_mocap_to_cam_left.json", np.eye(4))
    to_zup(p)
    d = json.loads(p.read_text())
    assert d["up_axis"] == "z"
    assert "up_axis_note" in d, "a converted file has to record what was done"


def test_converting_twice_is_refused(tmp_path):
    """The second rotation is invisible: projections stay in frame, and the
    file already claims to be Z-up."""
    p = _calib(tmp_path / "T_mocap_to_cam_left.json", np.eye(4))
    to_zup(p)
    with pytest.raises(ValueError, match="already"):
        to_zup(p)


def test_an_unlabelled_calibration_is_refused_not_assumed(tmp_path):
    """The whole defect in one line: no declaration must mean 'stop', never
    'probably the one I want'."""
    p = _calib(tmp_path / "T_mocap_to_cam_left.json", np.eye(4))
    with pytest.raises(ValueError, match="up_axis"):
        require_zup(p)


def test_a_y_up_calibration_is_refused_by_name(tmp_path):
    p = _calib(tmp_path / "T_mocap_to_cam_left.json", np.eye(4), up_axis="y")
    with pytest.raises(ValueError, match="y"):
        require_zup(p)


def test_a_z_up_calibration_passes_and_hands_back_its_matrix(tmp_path):
    p = _calib(tmp_path / "T_mocap_to_cam_left.json", np.eye(4), up_axis="z")
    T, K = require_zup(p)
    assert np.allclose(T, np.eye(4)) and K["width"] == 640
