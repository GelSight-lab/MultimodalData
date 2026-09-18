"""Display conversion and drawing must never alter recorded source arrays."""
import importlib

import cv2
import numpy as np
import pytest


def api():
    try:
        return importlib.import_module("twm.visualization")
    except ModuleNotFoundError:
        pytest.fail("The shared visualization API has not been implemented")


def test_color_formats_layout_and_source_ownership():
    v = api()
    rgb = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
    bgr = np.full((4, 2, 3), [40, 50, 60], dtype=np.uint8)
    result = v.Renderer(tile_width=6, tile_height=4, columns=2).render([
        v.Tile("rgb", rgb, color_format="RGB"), v.Tile("bgr", bgr),
        v.Tile("other", bgr, modality="tactile"),
    ])
    assert result.image.shape == (8, 12, 3)
    assert result.image.dtype == np.uint8
    np.testing.assert_array_equal(result.image[0, 0], [30, 20, 10])
    np.testing.assert_array_equal(result.image[0, 6], [40, 50, 60])
    assert [(t.name, t.x, t.y, t.width, t.height, t.missing) for t in result.layouts] == [
        ("rgb", 0, 0, 6, 4, False), ("bgr", 6, 0, 6, 4, False),
        ("other", 0, 4, 6, 4, False),
    ]
    assert result.layouts[2].modality == "tactile"
    assert not result.image[4:, 6:].any()
    result.image[:] = 255
    np.testing.assert_array_equal(rgb[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(bgr[0, 0], [40, 50, 60])


def test_overlays_receive_native_bgr_copy_in_order_and_cannot_cross_tiles():
    v = api()
    source = np.zeros((8, 12, 3), np.uint8)
    source[:] = [10, 20, 30]
    source.flags.writeable = False
    calls = []

    def mask(image):
        assert image.shape == (8, 12, 3)
        assert image.flags.c_contiguous
        np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
        calls.append("mask")
        cv2.rectangle(image, (-100, -100), (100, 100), (0, 255, 0), -1)

    def contact(image):
        assert (image == [0, 255, 0]).all()
        calls.append("contact")
        image[:] = [255, 0, 0]

    result = v.Renderer(tile_width=6, tile_height=4, columns=2).render([
        v.Tile("pose", source, color_format="RGB", overlays=(mask, contact)),
        v.Tile("untouched", source),
    ])
    assert calls == ["mask", "contact"]
    assert (result.image[:, :6] == [255, 0, 0]).all()
    assert (result.image[:, 6:] == [10, 20, 30]).all()
    assert (source == [10, 20, 30]).all()


def test_depth_uses_only_finite_values_and_does_not_modify_source():
    v = api()
    source = np.array([[1, 2, 3], [np.nan, np.inf, -np.inf]], dtype=float)
    before = source.copy()
    result = v.Renderer(tile_width=3, tile_height=2, columns=1).render([
        v.Tile("depth", source, modality="depth", color_format="depth"),
    ])
    np.testing.assert_array_equal(result.image[:, :, 0], [[0, 127, 255], [0, 0, 0]])
    np.testing.assert_array_equal(result.image[:, :, 0], result.image[:, :, 2])
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("source", [np.ones((2, 3)), np.full((2, 3), np.nan)])
def test_degenerate_depth_is_black(source):
    v = api()
    result = v.Renderer(tile_width=3, tile_height=2, columns=1).render([
        v.Tile("depth", source, color_format="depth"),
    ])
    assert not result.image.any()


def test_depth_excludes_nonpositive_values_but_tactile_height_is_signed():
    v = api()
    source = np.array([[-1., 0., 1., 2.]])
    renderer = v.Renderer(tile_width=4, tile_height=1, columns=1)
    depth = renderer.render([v.Tile("depth", source, modality="depth", color_format="depth")])
    height = renderer.render([v.Tile("height", source, modality="tactile_height", color_format="depth")])
    np.testing.assert_array_equal(depth.image[0, :, 0], [0, 0, 0, 255])
    np.testing.assert_array_equal(height.image[0, :, 0], [0, 85, 170, 255])


def test_explicit_depth_range_clips_and_handles_nonfinite_values():
    v = api()
    result = v.Renderer(tile_width=5, tile_height=1, columns=1).render([
        v.Tile("depth", np.array([[-1, 2, 3, 9, np.nan]]),
               color_format="depth", depth_range=(2, 4)),
    ])
    np.testing.assert_array_equal(result.image[0, :, 0], [0, 0, 127, 255, 0])


def test_missing_tiles_are_visible_and_have_explicit_metadata():
    v = api()
    def never_called(image):
        pytest.fail("Missing tiles must not run source-dependent overlays")
    result = v.Renderer(tile_width=200, tile_height=80, columns=1).render([
        v.Tile("left", None, modality="tactile", overlays=(never_called,)),
    ])
    assert result.layouts[0].missing
    assert result.image.any()


@pytest.mark.parametrize("kwargs", [
    {"tile_width": 0}, {"tile_height": -1}, {"columns": 0},
    {"columns": 1.5}, {"tile_width": True},
])
def test_invalid_layout_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        api().Renderer(**kwargs)


@pytest.mark.parametrize("image,kwargs", [
    (np.zeros((2, 2, 3), float), {}),
    (np.zeros((2, 2), np.uint8), {}),
    (np.zeros((0, 2, 3), np.uint8), {}),
    (np.zeros((2, 2, 3), np.uint8), {"color_format": "YUV"}),
    (np.zeros((2, 2, 3), np.uint8), {"color_format": "depth"}),
    (np.zeros((2, 2), complex), {"color_format": "depth"}),
    (np.zeros((2, 2)), {"color_format": "depth", "depth_range": (1, 1)}),
    (np.zeros((2, 2)), {"color_format": "depth", "depth_range": (2, 1)}),
    (np.zeros((2, 2)), {"color_format": "depth", "depth_range": (0, np.inf)}),
])
def test_invalid_tile_data_is_rejected(image, kwargs):
    v = api()
    with pytest.raises(ValueError):
        v.Renderer().render([v.Tile("invalid", image, **kwargs)])


def test_empty_render_is_rejected():
    with pytest.raises(ValueError, match="tile"):
        api().Renderer().render([])
