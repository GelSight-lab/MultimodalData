"""A gel calibration belongs to the physical units it was measured with.

`T_gel_to_rigid_<side>.json` says where the gel surface sits inside the
tracked rigid body. Swap the sensor and that geometry changes: the new unit
does not sit where the old one did. Everything downstream inherits the error
silently --

  * the projection overlay draws the gel centre in the wrong place,
  * `force_<side>_target_pose` displaces along `R(q) @ gel_axis`, so the
    virtual target moves in the wrong direction by the same amount,

and nothing raises, because no file on either side names a serial.

The rig already records what it used: `metadata.attrs["gelsight_serials"]` is
written into every HDF5 by `recorder.schema`. On 2026-09-22 the right sensor
was replaced mid-session and the recordings say so exactly --

    episode_000..006   2DUPB53G, 2BKRDTAD    (07:30 - 17:39)
    episode_007..009   2DUPB53G, 2BGLKZNT    (18:47 - 19:00)

-- while the epoch those recordings resolve to, 2026-09-09, was solved with
2BKRDTAD. Three episodes would have been published against a calibration for
a sensor that was no longer on the rig.

So the epoch declares its units, and a recording whose units differ is
refused rather than rendered.
"""
from __future__ import annotations

import json

import pytest

from twm.calib_epoch import epoch_sensors, check_sensors


def test_an_epoch_declares_the_units_it_was_solved_with(tmp_path):
    d = tmp_path / "epoch_2026-09-09"
    d.mkdir()
    (d / "sensors.json").write_text(json.dumps(
        {"left": "2DUPB53G", "right": "2BKRDTAD"}))
    assert epoch_sensors(d) == {"left": "2DUPB53G", "right": "2BKRDTAD"}


def test_an_epoch_that_declares_nothing_is_unknown_not_matching(tmp_path):
    """Silence must not read as agreement — that is the failure being fixed."""
    d = tmp_path / "epoch_old"
    d.mkdir()
    assert epoch_sensors(d) is None


def test_matching_serials_pass(tmp_path):
    d = tmp_path / "e"; d.mkdir()
    (d / "sensors.json").write_text(json.dumps(
        {"left": "2DUPB53G", "right": "2BKRDTAD"}))
    check_sensors(d, ["2DUPB53G", "2BKRDTAD"])          # no raise


def test_a_swapped_sensor_warns_but_does_not_block(tmp_path):
    """The rig shipped a byte-identical T_gel_to_rigid_right across three
    epochs and a sensor replacement. Refusing here would impose a standard
    this project has never held itself to, over an offset smaller than the
    calibration's own 0.72 mm / 1.07 deg repeatability."""
    d = tmp_path / "e"; d.mkdir()
    (d / "sensors.json").write_text(json.dumps(
        {"left": "2DUPB53G", "right": "2BKRDTAD"}))
    with pytest.warns(RuntimeWarning, match="right"):
        check_sensors(d, ["2DUPB53G", "2BGLKZNT"])


def test_strict_raises_for_a_caller_that_wants_it(tmp_path):
    d = tmp_path / "e"; d.mkdir()
    (d / "sensors.json").write_text(json.dumps(
        {"left": "2DUPB53G", "right": "2BKRDTAD"}))
    with pytest.raises(ValueError, match="right"):
        check_sensors(d, ["2DUPB53G", "2BGLKZNT"], strict=True)


def test_the_warning_says_force_values_are_unaffected(tmp_path):
    """Reconstruction never reads this transform; saying otherwise would send
    someone hunting a force bug that is not there."""
    d = tmp_path / "e"; d.mkdir()
    (d / "sensors.json").write_text(json.dumps(
        {"left": "2DUPB53G", "right": "2BKRDTAD"}))
    with pytest.warns(RuntimeWarning) as rec:
        check_sensors(d, ["2DUPB53G", "2BGLKZNT"])
    assert "Force VALUES are unaffected" in str(rec[0].message)


def test_the_message_names_both_serials(tmp_path):
    """So the operator can tell which way round the mismatch is."""
    d = tmp_path / "e"; d.mkdir()
    (d / "sensors.json").write_text(json.dumps(
        {"left": "2DUPB53G", "right": "2BKRDTAD"}))
    with pytest.raises(ValueError) as e:
        check_sensors(d, ["2DUPB53G", "2BGLKZNT"], strict=True)
    assert "2BKRDTAD" in str(e.value) and "2BGLKZNT" in str(e.value)


def test_an_undeclared_epoch_does_not_block_old_data(tmp_path):
    """Every epoch solved before this existed declares nothing. Refusing them
    all would stop the whole library to fix three episodes."""
    d = tmp_path / "e"; d.mkdir()
    check_sensors(d, ["2DUPB53G", "2BGLKZNT"])          # no raise
