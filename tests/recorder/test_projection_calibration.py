"""The live recorder can actually find the calibration it projects through.

This is the test that was missing when `calibration/result` was renamed to
`calibration/epoch_2026-09-09`. `load_projection` named its own directory,
catches every exception, and returns None -- so a path that stopped existing
did not fail, it just made the overlay stop appearing, with one warning line
in a log nobody reads mid-session. The overlay is how the operator confirms
the rig is calibrated BEFORE recording, so losing it silently costs a session.

These assert against the real repository calibration, not a fixture: the
failure being guarded is precisely "the real tree moved and the code did not".
"""
import pytest

import twm.calib_epoch as ce
from twm.calib_epoch import CURRENT_EPOCH, EPOCH_DIRS, current_epoch, current_epoch_dir
from twm.recorder.app import load_projection
from twm.recorder.config import RecorderConfig
from twm.viz import CAM_CALIB_NAME


def test_current_epoch_names_its_directory():
    assert current_epoch() == (CURRENT_EPOCH, EPOCH_DIRS[CURRENT_EPOCH])
    assert current_epoch_dir().is_dir()


def test_rolling_back_is_expressible_without_moving_files():
    """A new solve can be worse than the one it replaced; the way back is one
    constant, not a rename."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ce, "CURRENT_EPOCH", "2026-05-12")
        assert ce.current_epoch() == ("2026-05-12", EPOCH_DIRS["2026-05-12"])


def test_current_epoch_holds_every_file_the_overlay_needs():
    d = current_epoch_dir()
    missing = [n for n in list(CAM_CALIB_NAME.values())
               + ["T_gel_to_rigid_left.json", "T_gel_to_rigid_right.json"]
               if not (d / n).is_file()]
    assert not missing, f"{d} is missing {missing}"


def test_a_missing_epoch_raises_naming_the_directory():
    """The recorder must be told which directory is gone, by name."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ce, "CURRENT_EPOCH", "1999-01-01")
        mp.setitem(ce.EPOCH_DIRS, "1999-01-01", ce.REPO / "calibration" / "nope")
        with pytest.raises(FileNotFoundError, match="nope"):
            ce.current_epoch()


def test_an_unknown_epoch_name_raises():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ce, "CURRENT_EPOCH", "not-an-epoch")
        with pytest.raises(KeyError, match="not-an-epoch"):
            ce.current_epoch()


def test_load_projection_finds_the_rigs_three_cameras():
    """The end the operator sees: overlay configured, not silently None."""
    cfg = RecorderConfig(task="t")
    proj = load_projection(cfg)
    assert proj is not None, "projection overlay disabled -- see the ERROR log"
    assert len(proj["cams"]) == len(cfg.realsense_serials)
    assert proj["gel_left"] is not None and proj["gel_right"] is not None


def test_load_projection_is_off_only_when_asked():
    assert load_projection(RecorderConfig(task="t", show_projection=False)) is None
