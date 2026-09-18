"""The new entry point delegates every legacy layout and projection decision."""
import importlib
import subprocess
import sys

import numpy as np
import pytest

from twm.viz import build_preview_panel, draw_projection_overlay


def api():
    try:
        return importlib.import_module("twm.visualization")
    except ModuleNotFoundError:
        pytest.fail("The shared visualization API has not been implemented")


def inputs(wrists):
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(7)]
    return dict(color_frames=frames[:3], gs_frames=frames[3:5], gs_ref=frames[5:7],
                optitrack_poses={"sensor_left": (0., [0.05, 0.02, 0., 0., 0., 0., 1.])},
                recording=True, frame_count=12, elapsed=0.4, buf=2, fps=30.,
                task_name="example", arducam_frames=frames[:wrists] if wrists else None)


@pytest.mark.parametrize("wrists", [0, 1, 2])
def test_plain_preview_is_byte_identical_to_legacy(wrists):
    data = inputs(wrists)
    expected = build_preview_panel(**data)
    np.testing.assert_array_equal(api().render_preview(**data), expected)


@pytest.mark.parametrize("positional", [False, True])
def test_projection_matches_legacy_and_really_draws(positional):
    v = api()
    data = inputs(1)
    cam = {"index": 2, "T_mocap_to_cam": np.eye(4),
           "intrinsics": {"fx": 600., "fy": 600., "ppx": 320., "ppy": 240.}}
    cam["T_mocap_to_cam"][2, 3] = 800.
    options = dict(frozen_side="left", forces_n={"left": 3.},
                   targets_7={"left": [0.055, 0.02, 0., 0., 0., 0., 1.]},
                   press_axis={"left": np.array([1., 0., 0.])}, axis_len_mm=60.)
    projection = v.Projection([cam], np.zeros(3), np.zeros(3), **options)
    base = build_preview_panel(**data)
    expected = base.copy()
    draw_projection_overlay(expected, data["optitrack_poses"], [cam],
                            np.zeros(3), np.zeros(3), **options)
    assert not np.array_equal(base, expected)
    if positional:
        keys = ("color_frames", "gs_frames", "gs_ref", "optitrack_poses",
                "recording", "frame_count", "elapsed")
        args = [data.pop(key) for key in keys]
        actual = v.render_preview(*args, projection=projection, **data)
    else:
        actual = v.render_preview(projection=projection, **data)
    np.testing.assert_array_equal(actual, expected)


def test_independent_overlay_uses_explicit_latest_poses():
    v = api()
    data = inputs(0)
    base = v.render_preview(**data)
    cam = {"index": 1, "T_mocap_to_cam": np.eye(4),
           "intrinsics": {"fx": 600., "fy": 600., "ppx": 320., "ppy": 240.}}
    poses = {"sensor_left": (1., [0., 0., 0.8, 0., 0., 0., 1.])}
    projection = v.Projection([cam], np.zeros(3), np.zeros(3))
    actual = base.copy()
    expected = base.copy()
    draw_projection_overlay(expected, poses, [cam], np.zeros(3), np.zeros(3))
    assert v.draw_preview_overlay(actual, poses, projection) is None
    assert not np.array_equal(actual, base)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("pose_frame,calibration_frame", [
    ("y", "z"), ("unknown", "unknown"), ("y", None),
])
def test_projection_rejects_incompatible_or_unverifiable_world_frames(pose_frame, calibration_frame):
    with pytest.raises(ValueError, match="frame"):
        api().Projection([], np.zeros(3), np.zeros(3),
                         pose_world_frame=pose_frame,
                         calibration_world_frame=calibration_frame)


@pytest.mark.parametrize("world_frame", ["y", "z"])
def test_matching_world_frame_is_accepted_without_conversion(world_frame):
    projection = api().Projection([], np.zeros(3), np.zeros(3),
                                 pose_world_frame=world_frame,
                                 calibration_world_frame=world_frame)
    assert projection.pose_world_frame == world_frame


def test_visualization_import_does_not_load_hardware_backends():
    result = subprocess.run([sys.executable, "-c", """
import importlib.abc
import sys
class BlockHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'PySpin', 'serial', 'open3d', 'torch'}:
            raise AssertionError('Unexpected hardware/model import: ' + fullname)
sys.meta_path.insert(0, BlockHardware())
from twm.visualization import Renderer, render_preview
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
