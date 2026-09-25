"""The staged calibration must come from the epoch the sessions declare.

`segment.copy_calibration` carries whatever `release/<task>/calibration/` holds
into the cut tree. Nothing put the right thing there. On 2026-09-16:

    release/motherboard  = epoch_2026-05-12   while publishing 2026-09-11/12
    release/pushT        = epoch_2026-06-26   while publishing 2026-09-10..15
    release/rope         = absent entirely

The publish gate catches the mismatch, but only at publish. A fix applied to
the CUT tree by hand held for one day and was overwritten the moment segment
re-ran and copied the stale source forward — which is what happened here.

So the staging is derived, not maintained: read the epoch the sessions declare,
take that epoch's files, convert to the Z-up frame the poses are in, and write
them. Idempotent, so a re-run repairs rather than drifts.
"""
import json

import numpy as np
import pytest

from twm.calibration_frame import YUP_TO_ZUP_4
from twm.react_preprocess.segment import stage_calibration


def _epoch_dir(root, name, t=None):
    d = root / f"epoch_{name}"
    d.mkdir(parents=True, exist_ok=True)
    T = np.eye(4) if t is None else np.asarray(t)
    for cam in ("left", "middle", "right"):
        (d / f"T_mocap_to_cam_{cam}.json").write_text(json.dumps(
            {"T_mocap_to_cam": T.tolist(),
             "intrinsics": {"fx": 1, "fy": 1, "ppx": 0, "ppy": 0}}))
    (d / "T_gel_to_rigid_left.json").write_text(json.dumps({"x": 1}))
    return d


@pytest.fixture
def tree(tmp_path):
    rel = tmp_path / "release" / "rope"
    (rel / "meta" / "2026-09-14").mkdir(parents=True)
    (rel / "meta" / "2026-09-14" / "episode_000.parquet").write_bytes(b"")
    epochs = tmp_path / "calibration"
    T = np.eye(4); T[1, 3] = 100.0
    _epoch_dir(epochs, "2026-09-09", T)
    _epoch_dir(epochs, "2026-05-12")
    return rel, epochs, T


def test_it_stages_the_declared_epoch(tree):
    rel, epochs, T = tree
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-09-09"})
    got = np.array(json.loads(
        (rel / "calibration/T_mocap_to_cam_left.json").read_text())["T_mocap_to_cam"])
    assert np.allclose(got, T @ np.linalg.inv(YUP_TO_ZUP_4))


def test_the_staged_files_declare_z_up(tree):
    rel, epochs, _ = tree
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-09-09"})
    d = json.loads((rel / "calibration/T_mocap_to_cam_left.json").read_text())
    assert d["up_axis"] == "z" and "up_axis_note" in d


def test_the_gel_transform_is_marked_as_not_a_world_frame(tree):
    """It is in the rigid body's own frame and does not rotate with the world."""
    rel, epochs, _ = tree
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-09-09"})
    d = json.loads((rel / "calibration/T_gel_to_rigid_left.json").read_text())
    assert "rigid" in str(d.get("up_axis", "")).lower()


def test_running_it_twice_repairs_rather_than_rotating_twice(tree):
    """The second rotation is invisible — projections stay in frame — so the
    stage has to be idempotent, not merely runnable again."""
    rel, epochs, T = tree
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-09-09"})
    first = (rel / "calibration/T_mocap_to_cam_left.json").read_text()
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-09-09"})
    assert (rel / "calibration/T_mocap_to_cam_left.json").read_text() == first


def test_a_stale_epoch_already_there_is_replaced(tree):
    """The exact failure: a correct calibration placed by hand, overwritten by
    a re-run that copied a stale source forward."""
    rel, epochs, T = tree
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-05-12"})
    stage_calibration(rel, "rope", epochs, sessions={"2026-09-14": "2026-09-09"})
    got = np.array(json.loads(
        (rel / "calibration/T_mocap_to_cam_left.json").read_text())["T_mocap_to_cam"])
    assert np.allclose(got, T @ np.linalg.inv(YUP_TO_ZUP_4))


def test_sessions_declaring_two_epochs_is_refused(tree):
    rel, epochs, _ = tree
    (rel / "meta" / "2026-05-11").mkdir(parents=True)
    (rel / "meta" / "2026-05-11" / "episode_000.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="more than one"):
        stage_calibration(rel, "rope", epochs,
                          sessions={"2026-09-14": "2026-09-09",
                                    "2026-05-11": "2026-05-12"}, since=None)
