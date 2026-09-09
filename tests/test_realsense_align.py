"""Offline depth→color alignment must reproduce librealsense's rs.align.

The intrinsics here use a power-of-two focal length and a half-integer
principal point so the deproject/project round trip is exact in float32 and
the tests can assert pixel positions rather than tolerances.
"""
import json

import numpy as np
import pytest

from twm.realsense_align import (
    DepthToColor,
    Intrinsics,
    align_depth_to_color,
    load_calibration,
    save_calibration,
)

K = Intrinsics(width=128, height=128, fx=128.0, fy=128.0, ppx=64.0, ppy=64.0,
               model="brown_conrady", coeffs=(0.0,) * 5)
COLOR_K = Intrinsics(**{**K.__dict__, "model": "inverse_brown_conrady"})
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def calib(rotation=IDENTITY, translation=(0.0, 0.0, 0.0), **kw):
    return DepthToColor(serial="TEST", depth=K, color=COLOR_K, rotation=rotation,
                        translation=translation, depth_scale=0.001, **kw)


def depth_with(pixels):
    """(128, 128) uint16 depth image; `pixels` maps (x, y) -> millimetres."""
    d = np.zeros((128, 128), np.uint16)
    for (x, y), mm in pixels.items():
        d[y, x] = mm
    return d


def test_a_depth_pixel_fills_the_two_by_two_block_its_corners_span():
    """rs.align maps the pixel's top-left and bottom-right corners and fills
    the rectangle between them, so identity extrinsics dilate by one pixel
    toward +x/+y. Matching that is what makes this a drop-in replacement."""
    out = align_depth_to_color(depth_with({(3, 4): 1000}), calib())
    assert out.shape == (128, 128) and out.dtype == np.uint16
    assert (out[4:6, 3:5] == 1000).all()
    assert out.sum() == 4 * 1000


def test_zero_depth_writes_nothing():
    out = align_depth_to_color(np.zeros((128, 128), np.uint16), calib())
    assert not out.any()


def test_the_nearest_surface_wins_where_two_depth_pixels_overlap():
    out = align_depth_to_color(depth_with({(3, 4): 1000, (4, 4): 500}), calib())
    assert out[4, 3] == 1000          # only the far pixel reaches here
    assert out[4, 4] == 500           # both reach here; the nearer one wins
    assert out[4, 5] == 500


def test_a_block_hanging_over_an_edge_is_clipped_not_dropped():
    """rs.align keeps the part of the quad that is inside the image; dropping
    the whole quad instead loses a couple of hundred border pixels a frame."""
    out = align_depth_to_color(depth_with({(127, 60): 1000, (60, 127): 1000}), calib())
    assert (out[60:62, 127] == 1000).all()      # right edge: one column survives
    assert (out[127, 60:62] == 1000).all()      # bottom edge: one row survives
    assert out.sum() == 4 * 1000                # and nothing else is written


def test_a_block_entirely_outside_the_image_writes_nothing():
    # 2 m of baseline at 1 m range shifts the block 256 px, off a 128 px image.
    out = align_depth_to_color(depth_with({(3, 4): 1000}), calib(translation=(2.0, 0.0, 0.0)))
    assert not out.any()


def test_translating_the_color_camera_shifts_the_depth_by_fx_times_t_over_z():
    # t = 20 mm along x, z = 1 m, fx = 128 px -> 2.56 px. The corners land on
    # 5.06 and 6.06, and ceil puts the block at x in [6, 7].
    out = align_depth_to_color(depth_with({(3, 4): 1000}), calib(translation=(0.02, 0.0, 0.0)))
    assert (out[4:6, 6:8] == 1000).all()
    assert out.sum() == 4 * 1000


def test_nearer_surfaces_shift_further_than_distant_ones():
    c = calib(translation=(0.02, 0.0, 0.0))
    near = align_depth_to_color(depth_with({(3, 4): 500}), c)      # 0.5 m -> 5.12 px
    far = align_depth_to_color(depth_with({(3, 4): 2000}), c)      # 2.0 m -> 1.28 px
    assert (near[4:6, 8:10] == 500).all()
    assert (far[4:6, 4:6] == 2000).all()


def test_distortion_coefficients_are_refused_rather_than_ignored():
    distorted = Intrinsics(**{**K.__dict__, "coeffs": (0.1, 0.0, 0.0, 0.0, 0.0)})
    with pytest.raises(NotImplementedError, match="distortion"):
        align_depth_to_color(depth_with({(3, 4): 1000}),
                             DepthToColor(serial="T", depth=distorted, color=COLOR_K,
                                          rotation=IDENTITY, translation=(0.0, 0.0, 0.0),
                                          depth_scale=0.001))


def test_a_depth_image_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="480"):
        align_depth_to_color(np.zeros((480, 640), np.uint16), calib())


def test_rotation_is_read_column_major_like_librealsense():
    """rs2_extrinsics stores the rotation column-major; reading it row-major
    transposes the rotation and silently mis-aligns every frame."""
    # 90 degrees about z, column-major: columns are (0,1,0) and (-1,0,0).
    c = calib(rotation=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(c.rotation_matrix(),
                               [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                               atol=1e-9)


def test_calibration_round_trips_through_json(tmp_path):
    c = calib()
    path = tmp_path / "TEST.json"
    save_calibration(c, path)
    back = load_calibration(path)
    assert back == c
    assert json.loads(path.read_text())["depth"]["fx"] == 128.0


def test_apply_to_episode_rewrites_raw_depth_and_flips_the_flag(tmp_path):
    """The conversion an episode recorded with --raw_depth needs before any
    consumer that assumes depth sits on the color grid."""
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    from twm.realsense_align import apply_to_episode, save_calibration
    from twm.recorder.frames import Tick, synthetic_tick
    from twm.recorder.schema import append_ticks, create_episode_file, write_episode_attrs

    calib_dir = tmp_path / "calib"
    # 640x480 to match the recorder's depth datasets.
    K640 = Intrinsics(width=640, height=480, fx=128.0, fy=128.0, ppx=64.0, ppy=64.0,
                      model="brown_conrady", coeffs=(0.0,) * 5)
    save_calibration(DepthToColor(serial="A", depth=K640, color=K640, rotation=IDENTITY,
                                  translation=(0.02, 0.0, 0.0), depth_scale=0.001),
                     calib_dir / "A.json")

    f, path = create_episode_file(str(tmp_path), 0, ["A"], ["L", "R"], 30, n_realsense=1,
                                  depth_aligned=False)
    ticks = [synthetic_tick(100.0 + k / 30, seed=k, n_realsense=1) for k in range(3)]
    append_ticks(f, ticks)
    write_episode_attrs(f, {"valid": True, "invalid_reason": "", "ended_by": "operator",
                            "frame_count": 3, "gap_count": 0})
    f.close()

    out = apply_to_episode(path, tmp_path / "aligned.h5", directory=calib_dir, progress=None)

    with h5py.File(path, "r") as raw, h5py.File(out, "r") as done:
        assert done["metadata"].attrs["depth_aligned"]
        assert not raw["metadata"].attrs["depth_aligned"]      # the input is untouched
        assert done["realsense/cam0/depth"].shape == raw["realsense/cam0/depth"].shape
        assert not np.array_equal(done["realsense/cam0/depth"][0],
                                  raw["realsense/cam0/depth"][0])
        # colour and the tick clock come through unchanged
        np.testing.assert_array_equal(done["realsense/cam0/color"][:], raw["realsense/cam0/color"][:])
        np.testing.assert_allclose(done["timestamps"][:], raw["timestamps"][:])


def test_apply_refuses_an_episode_whose_depth_is_already_aligned(tmp_path):
    import hdf5plugin  # noqa: F401

    from twm.realsense_align import apply_to_episode
    from twm.recorder.schema import create_episode_file

    f, path = create_episode_file(str(tmp_path), 1, ["A"], ["L", "R"], 30, n_realsense=1)
    f.close()
    with pytest.raises(ValueError, match="already"):
        apply_to_episode(path, tmp_path / "out.h5")
