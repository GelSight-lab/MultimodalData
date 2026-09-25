"""The calibration shipped beside a session must be the epoch that session declares.

`CALIB_SESSIONS` is the definition of which extrinsics a recording needs, and
the module is explicit that date order does not determine it: a session
declares its epoch and nothing infers it.

The tree that gets published carried whatever `release/` happened to hold —
verified 2026-09-14 to be `epoch_2026-05-12` in Z-up form, correct label and
all, shipped beside the 2026-09-11 and 2026-09-12 sessions. Between epochs:
|dT| = 53-64 mm, dR = 2.6-6.0 deg, which puts the projected sensor 35-73 px
off in a 640x480 view — visibly wrong, but shaped like a slightly
miscalibrated rig rather than like a bug.

Nothing checked it, because the two facts live in different places: the epoch
table in the toolbox, the file in a staging tree.
"""
import json

import numpy as np
import pytest

from twm.calibration_frame import YUP_TO_ZUP_4
import twm.scripts.build_release_publish as P


def _write(path, T, up_axis="z"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"T_mocap_to_cam": np.asarray(T).tolist(),
                                "up_axis": up_axis,
                                "intrinsics": {"fx": 1, "fy": 1, "ppx": 0,
                                               "ppy": 0}}))


def _tree(tmp_path, T_shipped):
    r = tmp_path / "release_cut" / "pushT"
    (r / "meta" / "2026-09-11").mkdir(parents=True)
    (r / "meta" / "2026-09-11" / "episode_000_seg00.parquet").write_bytes(b"")
    _write(r / "calibration" / "T_mocap_to_cam_left.json", T_shipped)
    return r


def _epochs(tmp_path, T_right):
    d = tmp_path / "epochs" / "epoch_2026-09-09"
    _write(d / "T_mocap_to_cam_left.json", T_right)
    return tmp_path / "epochs"


def test_shipping_a_different_epoch_is_caught(tmp_path):
    right = np.eye(4)
    wrong = np.eye(4)
    wrong[0, 3] = 60.0                       # 60 mm — a whole epoch apart
    bad = P.check_calibration_epoch(
        _tree(tmp_path, wrong), _epochs(tmp_path, right),
        {"2026-09-11": "2026-09-09"})
    assert bad and "2026-09-09" in bad[0]


def test_shipping_the_declared_epoch_passes(tmp_path):
    right = np.eye(4)
    assert P.check_calibration_epoch(
        _tree(tmp_path, right), _epochs(tmp_path, right),
        {"2026-09-11": "2026-09-09"}) == []


def test_a_published_date_with_no_declared_epoch_is_refused(tmp_path):
    """`session_epoch` raises for an undeclared session, so the dataset would
    ship data its own toolbox cannot resolve a calibration for."""
    bad = P.check_calibration_epoch(
        _tree(tmp_path, np.eye(4)), _epochs(tmp_path, np.eye(4)), {})
    assert bad and "2026-09-11" in bad[0]


def test_the_comparison_is_made_in_one_convention(tmp_path):
    """The epoch files on disk are Y-up; the shipped one is Z-up. Comparing
    them raw would call every correct pairing a mismatch."""
    y = np.eye(4)
    y[1, 3] = 100.0
    z = y @ np.linalg.inv(YUP_TO_ZUP_4)
    d = tmp_path / "epochs" / "epoch_2026-09-09"
    _write(d / "T_mocap_to_cam_left.json", y, up_axis=None)
    assert P.check_calibration_epoch(
        _tree(tmp_path, z), tmp_path / "epochs",
        {"2026-09-11": "2026-09-09"}) == []


def test_dates_outside_the_publish_window_are_not_judged(tmp_path):
    """The May sessions genuinely need epoch_2026-05-12 and the tree now ships
    2026-09-09 — a real 52 mm difference. They are not being published, so
    saying so only makes the gate un-passable for the week that is."""
    r = tmp_path / "release_cut" / "pushT"
    for date in ("2026-05-10", "2026-09-11"):
        (r / "meta" / date).mkdir(parents=True)
        (r / "meta" / date / "episode_000_seg00.parquet").write_bytes(b"")
    _write(r / "calibration" / "T_mocap_to_cam_left.json", np.eye(4))
    bad = P.check_calibration_epoch(
        r, _epochs(tmp_path, np.eye(4)),
        {"2026-05-10": "2026-05-12", "2026-09-11": "2026-09-09"},
        since="2026-09-10")
    assert bad == [], bad
