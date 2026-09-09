"""The live preview's overlay follows the newest OptiTrack pose, not the
pose sampled at the tick, and is redrawn every GUI frame; the (expensive)
base panel is rebuilt only when a new tick arrives."""
from types import SimpleNamespace

import numpy as np

from twm.recorder.app import PreviewRenderer


def _snap(ts):
    return SimpleNamespace(tick=SimpleNamespace(timestamp=ts))


def make(preview_fps=30):
    calls = {"build": 0, "overlay": []}
    clock = {"t": 100.0}
    poses = {"n": 0}

    def build(snap):
        calls["build"] += 1
        return np.full((4, 4, 3), calls["build"], np.uint8)

    def overlay(panel, ot_poses):
        calls["overlay"].append(ot_poses["n"])
        panel[0, 0] = 255

    def poses_now():
        poses["n"] += 1
        return {"n": poses["n"]}

    r = PreviewRenderer(build, overlay, poses_now, preview_fps=preview_fps,
                        clock=lambda: clock["t"])
    return r, calls, clock


def test_overlay_is_redrawn_with_the_newest_pose_on_every_render():
    r, calls, clock = make()
    r.render(_snap(1.0), show_overlay=True)
    r.render(_snap(1.0), show_overlay=True)
    r.render(_snap(1.0), show_overlay=True)
    assert calls["build"] == 1                     # same tick: base panel reused
    assert calls["overlay"] == [1, 2, 3]           # each render asked for the pose *now*


def test_base_panel_is_rebuilt_when_a_new_tick_arrives():
    r, calls, clock = make(preview_fps=30)
    r.render(_snap(1.0), show_overlay=False)
    clock["t"] += 0.04
    r.render(_snap(1.0333), show_overlay=False)
    assert calls["build"] == 2


def test_base_panel_rebuild_is_rate_limited_to_preview_fps():
    r, calls, clock = make(preview_fps=10)
    r.render(_snap(1.0), show_overlay=False)
    clock["t"] += 0.04
    r.render(_snap(1.0333), show_overlay=False)     # new tick, but only 40 ms later
    assert calls["build"] == 1
    clock["t"] += 0.07
    r.render(_snap(1.0666), show_overlay=False)
    assert calls["build"] == 2


def test_render_never_draws_into_the_cached_base():
    r, calls, clock = make()
    a = r.render(_snap(1.0), show_overlay=True)
    b = r.render(_snap(1.0), show_overlay=False)
    assert a[0, 0, 0] == 255 and b[0, 0, 0] == 1    # the overlay did not leak into the base
    b[1, 1] = 9
    c = r.render(_snap(1.0), show_overlay=False)
    assert c[1, 1, 0] == 1                          # nor did the caller's health line
