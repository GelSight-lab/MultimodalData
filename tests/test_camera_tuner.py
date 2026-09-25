"""The control model behind the live tuner.

Kept apart from the window so the part with rules in it can be tested: what
a key does, what a value clamps to, which controls have to be written before
which, and what gets saved.
"""
import pytest

from twm.camera_tuner import ControlModel, parse_control_ranges

V4L2_OUTPUT = """
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=5 value=5
                       contrast 0x00980901 (int)    : min=0 max=100 step=1 default=54 value=54
                          gamma 0x00980910 (int)    : min=100 max=500 step=1 default=250 value=250
                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=10000 step=1 default=166 value=166
"""


def test_ranges_are_read_from_the_camera_not_hardcoded():
    r = parse_control_ranges(V4L2_OUTPUT)
    assert r["brightness"] == (-64, 64, 1)
    assert r["exposure_time_absolute"] == (1, 10000, 1)
    assert "auto_exposure" in r


def model(**start):
    values = {"brightness": 5, "gamma": 250, "exposure_time_absolute": 330,
              "auto_exposure": 1, "contrast": 54}
    values.update(start)
    return ControlModel(values, parse_control_ranges(V4L2_OUTPUT))


def test_a_key_steps_its_control_and_reports_what_changed():
    m = model()
    assert m.press("e") == ("exposure_time_absolute", 340)
    assert m.values["exposure_time_absolute"] == 340
    assert m.press("E") == ("exposure_time_absolute", 330)


def test_a_step_clamps_to_the_camera_range_instead_of_going_out_of_bounds():
    m = model(brightness=63)
    assert m.press("b") == ("brightness", 64)
    assert m.press("b") == ("brightness", 64)      # already at max, stays
    m = model(brightness=-63)
    assert m.press("B") == ("brightness", -64)


def test_an_unbound_key_changes_nothing():
    m = model()
    before = dict(m.values)
    assert m.press("z") is None
    assert m.values == before


def test_exposure_steps_are_capped_by_the_frame_rate():
    """33 ms is the ceiling at 30 fps; past it the camera drops frames and the
    tuner would be showing a rate the recorder cannot use."""
    m = model(exposure_time_absolute=330)
    assert m.press("e") == ("exposure_time_absolute", 340)
    m = ControlModel({"exposure_time_absolute": 330, "auto_exposure": 1},
                     parse_control_ranges(V4L2_OUTPUT), fps=30)
    assert m.press("e") == ("exposure_time_absolute", 333)   # 33.3 ms
    assert m.press("e") == ("exposure_time_absolute", 333)   # and no further


def test_toggling_auto_exposure_flips_between_manual_and_auto():
    m = model(auto_exposure=1)
    assert m.press("a") == ("auto_exposure", 3)
    assert m.press("a") == ("auto_exposure", 1)


def test_the_saved_order_puts_each_automatic_before_the_value_it_locks():
    m = model()
    order = list(m.to_config())
    assert order.index("auto_exposure") < order.index("exposure_time_absolute")


def test_reset_returns_every_control_to_the_camera_default():
    """The defaults come from the camera, so reset is a real escape hatch
    rather than a second set of numbers to keep in step."""
    ranges = parse_control_ranges(V4L2_OUTPUT)
    defaults = {"brightness": 5, "gamma": 250, "exposure_time_absolute": 166}
    m = ControlModel({"brightness": 40, "gamma": 400, "exposure_time_absolute": 330},
                     ranges, defaults=defaults)
    m.reset()
    assert m.values["brightness"] == 5 and m.values["gamma"] == 250
