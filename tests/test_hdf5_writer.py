import unittest
import tempfile
import os
import numpy as np
import h5py

# We'll import the helpers directly from the script
from twm.data_collection import (
    HDF5Writer,
    append_camera_frame,
    append_camera_frames_batch,
    create_episode_file,
    flush_optitrack_to_hdf5,
)
from twm.sensor_camera import CameraSlot, ResolvedCamera


class TestHDF5Writer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _arducams(self):
        return (
            ResolvedCamera(
                CameraSlot("cam0", "usb-A", "unknown", 640, 480, 30, "MJPG"),
                "/dev/video6", "SN001",
            ),
            ResolvedCamera(
                CameraSlot("cam1", "usb-B", "unknown", 640, 480, 30, "MJPG"),
                "/dev/video10", "SN001",
            ),
        )

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
            self.assertNotIn("arducam", f)
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

    def test_create_episode_file_adds_configured_arducams_and_identity(self):
        f, path = create_episode_file(
            self.tmpdir, 6, ["A", "B", "C"], ["L", "R"], 30,
            arducam_config=self._arducams(),
        )
        f.close()

        with h5py.File(path, "r") as f:
            for i, path_id, device in ((0, "usb-A", "/dev/video6"),
                                       (1, "usb-B", "/dev/video10")):
                group = f[f"arducam/cam{i}"]
                self.assertEqual(group["frames"].shape, (0, 480, 640, 3))
                self.assertEqual(group["frames"].dtype, np.uint8)
                self.assertEqual(group["timestamps"].shape, (0,))
                self.assertEqual(group.attrs["usb_path"], path_id)
                self.assertEqual(group.attrs["device_at_recording"], device)
                self.assertEqual(group.attrs["reported_serial"], "SN001")
                self.assertEqual(group.attrs["position"], "unknown")
                self.assertEqual(group.attrs["pixel_format"], "MJPG")
            self.assertIn("arducam_config", f["metadata"].attrs)

    def test_append_arducam_only_batch_preserves_pixels_and_timestamps(self):
        f, _ = create_episode_file(
            self.tmpdir, 7, [], [], 30,
            arducam_config=self._arducams(), include_legacy=False,
        )
        cam0 = np.full((480, 640, 3), 11, np.uint8)
        cam1 = np.full((480, 640, 3), 29, np.uint8)

        append_camera_frames_batch(f, [
            (None, None, None, 10.0, None, [cam0, cam1], [9.90, 9.95]),
            (None, None, None, 11.0, None, [cam0 + 1, cam1 + 1], [None, None]),
        ])

        self.assertNotIn("realsense", f)
        self.assertNotIn("gelsight", f)
        np.testing.assert_array_equal(f["arducam/cam0/frames"][0], cam0)
        np.testing.assert_array_equal(f["arducam/cam1/frames"][1], cam1 + 1)
        np.testing.assert_allclose(f["arducam/cam0/timestamps"][:], [9.90, 11.0])
        np.testing.assert_allclose(f["arducam/cam1/timestamps"][:], [9.95, 11.0])
        np.testing.assert_allclose(f["timestamps"][:], [10.0, 11.0])
        f.close()

    def test_background_writer_stop_is_idempotent(self):
        writer = HDF5Writer()

        writer.stop()
        writer.stop()

        self.assertEqual(writer.queue_size, 0)


if __name__ == '__main__':
    unittest.main()
