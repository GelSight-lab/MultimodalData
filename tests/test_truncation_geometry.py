"""Assertion-based, data-independent tests for the manual truncation verifier."""
import numpy as np
import pytest

from twm.force_recovery import calib_free as CF
from twm.force_recovery.visible_eval import visible


@pytest.mark.parametrize("cx,radius,expected", [(160, 30, True), (32, 33, False),
                                               (12, 40, False), (0, 60, False),
                                               (-30, 60, False)])
def test_contact_core_border_rule(cx, radius, expected):
    reference = np.zeros((240, 320, 3), np.float32)
    image = reference.copy()
    yy, xx = np.mgrid[:240, :320]
    image[(yy - 120) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 4 * CF.VALID_DI
    assert bool(visible(image, reference)) == expected
