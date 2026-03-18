import unittest
import tempfile
import os
import numpy as np
import h5py

# We'll import the helpers directly from the script
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from twm_data_collection import create_episode_file, append_camera_frame, flush_optitrack_to_hdf5


class TestHDF5Writer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_create_episode_file_structure(self):
        """create_episode_file creates correct HDF5 group/dataset structure."""
        f, path = create_episode_file(
            date_dir=self.tmpdir,
            episode_num=0,
            realsense_serials=["AAA", "BBB", "CCC"],
            gelsight_serials=["2BGLKZNT", "2BKRDTAD"],
            fps=30,
        )
        f.close()

        with h5py.File(path, "r") as f:
            self.assertIn("timestamps", f)
            for i in range(3):
                self.assertIn(f"realsense/cam{i}/color", f)
                self.assertIn(f"realsense/cam{i}/depth", f)
            self.assertIn("gelsight/left/frames", f)
            self.assertIn("gelsight/right/frames", f)
            for name in ["motherboard", "sensor_left", "sensor_right"]:
                self.assertIn(f"optitrack/{name}/timestamps", f)
                self.assertIn(f"optitrack/{name}/pose", f)

    def test_append_camera_frame_grows_datasets(self):
        """append_camera_frame appends data and grows datasets by 1 each call."""
        f, path = create_episode_file(self.tmpdir, 1, ["A", "B", "C"], ["L", "R"], 30)

        color_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        depth_frames = [np.zeros((480, 640), dtype=np.uint16) for _ in range(3)]
        gs_frames    = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]

        append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp=1.0)
        append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp=2.0)

        self.assertEqual(f["timestamps"].shape[0], 2)
        self.assertEqual(f["realsense/cam0/color"].shape[0], 2)
        self.assertEqual(f["gelsight/left/frames"].shape[0], 2)
        f.close()

    def test_flush_optitrack_writes_poses(self):
        """flush_optitrack_to_hdf5 writes all buffered poses to HDF5."""
        f, path = create_episode_file(self.tmpdir, 2, ["A", "B", "C"], ["L", "R"], 30)

        optitrack_data = {
            "motherboard":  [(1.0, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
                             (1.1, [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])],
            "sensor_left":  [(1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
            "sensor_right": [],
        }

        flush_optitrack_to_hdf5(f, optitrack_data)

        self.assertEqual(f["optitrack/motherboard/pose"].shape, (2, 7))
        self.assertEqual(f["optitrack/sensor_left/pose"].shape, (1, 7))
        self.assertEqual(f["optitrack/sensor_right/pose"].shape[0], 0)
        f.close()

    def test_episode_filename_format(self):
        """create_episode_file produces correctly named file."""
        _, path = create_episode_file(self.tmpdir, 5, [], [], 30)
        self.assertTrue(path.endswith("episode_005.h5"))


if __name__ == '__main__':
    unittest.main()
