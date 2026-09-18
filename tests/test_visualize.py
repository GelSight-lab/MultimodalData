import unittest
import numpy as np
import pytest
from twm.viz import optitrack_at, build_preview_panel, STATUS_STRIP_H


@pytest.mark.parametrize("query,expected", [(2, 2), (1.4, 1), (1.6, 2), (.5, 1), (5, 3)])
def test_optitrack_nearest_timestamp(query, expected):
    lookup = {"tracker": (np.array([1., 2., 3.]),
                          np.array([[1, 0, 0, 0, 0, 0, 1]] * 3))}
    assert optitrack_at(lookup, query)["tracker"][0] == pytest.approx(expected)


def test_optitrack_missing_tracker():
    assert optitrack_at({"tracker": None}, 1.)["tracker"] is None


class TestSensorCameraPreview(unittest.TestCase):

    def _args(self):
        colors = [np.full((480, 640, 3), value, np.uint8)
                  for value in (10, 20, 30)]
        gels = [np.full((480, 640, 3), value, np.uint8)
                for value in (40, 50)]
        refs = [frame.copy() for frame in gels]
        return colors, gels, refs, {}

    def test_legacy_panel_shape_is_unchanged(self):
        panel = build_preview_panel(
            *self._args(), recording=False, frame_count=0, elapsed=0,
        )

        self.assertEqual(panel.shape, (480 + STATUS_STRIP_H, 1280, 3))

    def test_two_sensor_cameras_add_third_row_in_slot_order(self):
        cam0 = np.full((480, 640, 3), (11, 22, 33), np.uint8)
        cam1 = np.full((480, 640, 3), (44, 55, 66), np.uint8)
        panel = build_preview_panel(
            *self._args(), recording=False, frame_count=0, elapsed=0,
            arducam_frames=[cam0, cam1],
            arducam_labels=["cam0 usb-A unknown", "cam1 usb-B unknown"],
        )

        self.assertEqual(panel.shape, (720 + STATUS_STRIP_H, 1280, 3))
        np.testing.assert_array_equal(panel[600, 100], [11, 22, 33])
        np.testing.assert_array_equal(panel[600, 420], [44, 55, 66])
        np.testing.assert_array_equal(panel[600, 900], [0, 0, 0])

    def test_sensor_camera_preview_takes_one_or_two_frames(self):
        """One wrist camera is a valid rig — testing a single one before the
        mount exists, or carrying on after one comes off."""
        frame = np.zeros((480, 640, 3), np.uint8)
        for n in (1, 2):
            panel = build_preview_panel(
                [frame] * 3, [frame] * 2, [frame] * 2, {}, False, 0, 0.0,
                arducam_frames=[frame] * n,
                arducam_labels=["a", "b"][:n])
            self.assertEqual(panel.shape[1], 1280)
        with self.assertRaises(ValueError):
            build_preview_panel([frame] * 3, [frame] * 2, [frame] * 2, {},
                                False, 0, 0.0, arducam_frames=[frame] * 3)
