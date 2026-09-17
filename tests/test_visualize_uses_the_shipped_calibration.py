"""Playing a PUBLISHED episode must use the calibration it shipped with.

`twm.visualize` resolved calibration from the repo epoch, whose
`T_mocap_to_cam` is in the rig's Y-up frame. Published poses are Z-up.
`convert_release_zup` rotates the poses AND the calibration together --
verified: the shipped matrix equals the repo one composed with the Y->Z
rotation -- so the two are only interchangeable in matched pairs:

    Y-up poses x repo calibration      correct (the preview path, from H5)
    Z-up poses x shipped calibration   correct
    Z-up poses x repo calibration      WRONG  <- what the viewer did

The translation is identical in both, so the error is a rotation: the overlay
drifts rather than jumping, which is why it read as "looks off" rather than as
obviously broken.

The release tree ships the matching calibration beside the data, so that is
what a published episode should use. An explicit --cam_calib still wins.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from twm.release_episode import shipped_calibration


def _calib_dir(root, task, *, complete=True):
    d = Path(root) / task / "calibration"
    d.mkdir(parents=True)
    for name in ("left", "middle", "right"):
        (d / f"T_mocap_to_cam_{name}.json").write_text(json.dumps(
            {"T_mocap_to_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
             "intrinsics": {}, "camera_serial": name, "up_axis": "z"}))
    (d / "T_gel_to_rigid_left.json").write_text("{}")
    if complete:
        (d / "T_gel_to_rigid_right.json").write_text("{}")
    return d


def test_the_shipped_calibration_is_found(tmp_path):
    _calib_dir(tmp_path, "motherboard")
    cams, gl, gr = shipped_calibration(tmp_path, "motherboard")
    assert len(cams) == 3
    assert Path(gl).name == "T_gel_to_rigid_left.json"
    assert Path(gr).name == "T_gel_to_rigid_right.json"


def test_a_tree_without_calibration_returns_nothing(tmp_path):
    """Then the caller falls back to the repo epoch, as before."""
    (tmp_path / "motherboard").mkdir(parents=True)
    assert shipped_calibration(tmp_path, "motherboard") is None


def test_a_half_present_calibration_is_refused(tmp_path):
    """Mixing a shipped camera matrix with a repo gel file would pair a Z-up
    extrinsic with a Y-up one -- worse than using neither."""
    _calib_dir(tmp_path, "motherboard", complete=False)
    assert shipped_calibration(tmp_path, "motherboard") is None


def test_the_real_cut_tree_ships_a_usable_set():
    """Guards against the release dropping calibration silently."""
    import twm.pipeline_stages as PS
    got = shipped_calibration(PS.RELEASE_CUT, "motherboard")
    if got is None:
        pytest.skip("no local release_cut tree")
    cams, gl, gr = got
    assert len(cams) == 3
    for p in cams:
        d = json.loads(Path(p).read_text())
        assert d["up_axis"] == "z", f"{p} is not the Z-up matrix"
        assert "intrinsics" in d and "camera_serial" in d
