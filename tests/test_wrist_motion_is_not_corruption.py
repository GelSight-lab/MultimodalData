"""A camera that moves with the hand must not read as a torn frame.

`thr = max(CAM_SPIKE_ABS, CAM_SPIKE_REL * median + 5)` was calibrated on the
three static scene cameras, whose frame-to-frame difference has a median near
1.1 and never exceeds 9.6 in a whole toy episode -- the 25.0 floor sits 23x
above their median and 2.6x above their worst frame, so scene motion cannot
reach it.

The wrist cameras ride the hand. Measured on toy/2026-09-17/episode_006:

    stream        median   p99    p99.9   max    threshold
    view_middle     1.08    4.7     6.0    9.6      25.0
    wrist_left      3.28   25.3    33.3   40.0      25.0
    wrist_right     6.81   34.3    51.8   64.8      32.2

The threshold lands ON the 99th percentile of the wrist streams instead of
above their maximum, so the fastest one percent of ordinary hand motion is
flagged. The distribution is smooth across it -- the maxima are only 1.6-2.0x
the threshold, where a real tear is an outlier far from the bulk -- so what is
being cut is the tail of the motion distribution, not a second population.

The cost is not marginal. Over toy's fourteen episodes this accounted for
9,923 of 15,608 bad frames, and removing it takes the publishable yield from
15.8% to 92.7% -- 13.0 minutes to 76.4. Three episodes went to zero segments
on it alone. The operator reviewed one by eye and found nothing wrong with it.

What must NOT change: a frame that differs from both of its neighbours by far
more than the stream's own motion is still a tear, on any camera.
"""
from __future__ import annotations

import numpy as np
import pytest

from twm.react_preprocess.detect import cam_burst_threshold


def _flagged(diffs, thr):
    hot = np.where(np.asarray(diffs) > thr)[0]
    return len(hot)


def test_a_static_camera_keeps_its_floor():
    """Median near zero must not collapse the threshold onto sensor noise."""
    assert cam_burst_threshold(0.0) >= 25.0
    assert cam_burst_threshold(1.1) >= 25.0


@pytest.mark.parametrize("median,worst", [
    (3.28, 40.0),   # toy/09-17/ep006 wrist_left
    (6.81, 64.8),   # toy/09-17/ep006 wrist_right
    (5.24, 47.1),   # toy/09-17/ep007 wrist_left
    (3.58, 60.2),   # toy/09-22/ep004 wrist_left   worst ratio measured, 16.8x
    (1.58, 25.5),   # pushT/09-17/ep005 wrist_left
    (1.85, 32.1),   # pushT/09-17/ep005 wrist_right 17.3x
])
def test_ordinary_wrist_motion_clears_the_threshold(median, worst):
    """Every wrist stream measured, at its worst frame, stays under."""
    assert cam_burst_threshold(median) > worst


def test_the_headroom_does_not_shrink_as_the_stream_moves_faster():
    """A high-baseline stream must not be held to a tighter standard than a
    static one. That asymmetry is the whole defect."""
    static = cam_burst_threshold(1.08) / 1.08
    wrist = cam_burst_threshold(6.81) / 6.81
    assert wrist >= static * 0.5, (
        f"wrist headroom {wrist:.1f}x vs static {static:.1f}x")
    # and it must still exceed the worst ratio anyone measured
    assert wrist > 17.3


def test_a_tear_is_still_a_tear_on_a_moving_camera():
    """An outlier far above the stream's own motion still trips it."""
    thr = cam_burst_threshold(6.81)
    assert 6.81 * 40 > thr           # a frame 40x the median is caught
    diffs = np.r_[np.full(200, 6.81), 6.81 * 40, np.full(200, 6.81)]
    assert _flagged(diffs, thr) == 1


def test_the_scene_cameras_still_flag_nothing_on_toy():
    """The measured scene-camera maximum stays below the threshold."""
    assert cam_burst_threshold(1.08) > 9.6
