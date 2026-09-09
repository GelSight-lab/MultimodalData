import unittest
import numpy as np
import sys
import os
import hdf5plugin  # register against real h5py before the temporary import mocks
from unittest.mock import MagicMock

# Temporarily mock h5py and cv2 just for the import of twm.visualize,
# then restore the real modules so other tests (e.g. test_hdf5_writer) are unaffected.
_saved = {k: sys.modules.get(k) for k in ('h5py', 'cv2')}
sys.modules['h5py'] = MagicMock()
sys.modules['cv2'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from twm.visualize import optitrack_at

for k, v in _saved.items():
    if v is not None:
        sys.modules[k] = v
    elif k in sys.modules:
        del sys.modules[k]

# twm.visualize imported twm.viz while cv2 was mocked. Reload the canonical
# panel module with the real cv2 before exercising its pixel output below.
sys.modules.pop('twm.viz', None)
from twm.viz import build_preview_panel, STATUS_STRIP_H


class TestOptitrackAt(unittest.TestCase):

    def _make_lookup(self, timestamps, poses):
        return {"tracker": (np.array(timestamps), np.array(poses))}

    def test_exact_match(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 2.0)
        self.assertAlmostEqual(result["tracker"][0], 2.0)

    def test_nearest_before(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 1.4)
        self.assertAlmostEqual(result["tracker"][0], 1.0)

    def test_nearest_after(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 1.6)
        self.assertAlmostEqual(result["tracker"][0], 2.0)

    def test_before_first(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 0.5)
        self.assertAlmostEqual(result["tracker"][0], 1.0)

    def test_after_last(self):
        lookup = self._make_lookup([1.0, 2.0, 3.0], [[1, 0, 0, 0, 0, 0, 1]] * 3)
        result = optitrack_at(lookup, 5.0)
        self.assertAlmostEqual(result["tracker"][0], 3.0)

    def test_none_when_no_data(self):
        lookup = {"tracker": None}
        result = optitrack_at(lookup, 1.0)
        self.assertIsNone(result["tracker"])


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
