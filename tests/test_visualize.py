import unittest
import numpy as np
import sys
import os
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


if __name__ == '__main__':
    unittest.main()
