"""`--cam_calib <task>` resolves a whole calibration epoch by task name."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from twm import calib_epoch
from twm.calib_epoch import resolve_calibration

CAM_FILES = ["T_mocap_to_cam_middle.json", "T_mocap_to_cam_left.json",
             "T_mocap_to_cam_right.json"]
GEL_FILES = ["T_gel_to_rigid_left.json", "T_gel_to_rigid_right.json"]


def _epoch(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir()
    for f in CAM_FILES + GEL_FILES:
        (d / f).write_text(json.dumps({"up_axis": "y"}))
    return d


class TestResolveCalibration(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self.mb = _epoch(root, "mb_epoch")
        self.pt = _epoch(root, "pt_epoch")
        patcher = patch.dict(calib_epoch.CALIB_DIRS,
                             {"motherboard": self.mb, "pushT": self.pt}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        env = patch.dict("os.environ", {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        import os
        os.environ.pop("REACT_CALIB", None)
        os.environ.pop("REACT_RELEASE", None)

    def test_task_name_resolves_all_five_files_of_that_epoch(self):
        cams, gl, gr = resolve_calibration(
            ["motherboard"], None, None, "/data/test/2026-09-08/episode_000.h5")
        self.assertEqual(cams, [str(self.mb / f) for f in CAM_FILES])
        self.assertEqual(gl, str(self.mb / GEL_FILES[0]))
        self.assertEqual(gr, str(self.mb / GEL_FILES[1]))

    def test_task_name_wins_over_the_task_in_the_path(self):
        cams, gl, _ = resolve_calibration(
            ["pushT"], None, None, "/data/motherboard/2026-05-12/episode_000.h5")
        self.assertTrue(cams[0].startswith(str(self.pt)))
        self.assertTrue(gl.startswith(str(self.pt)))

    def test_explicit_gel_path_is_kept_alongside_a_task_name(self):
        _, gl, gr = resolve_calibration(
            ["motherboard"], "/my/gel_left.json", None, "/data/test/e.h5")
        self.assertEqual(gl, "/my/gel_left.json")
        self.assertEqual(gr, str(self.mb / GEL_FILES[1]))

    def test_explicit_cam_paths_pass_through_and_gels_come_from_the_path_epoch(self):
        cams, gl, gr = resolve_calibration(
            ["/a.json", "/b.json"], None, None, "/data/pushT/e.h5")
        self.assertEqual(cams, ["/a.json", "/b.json"])
        self.assertEqual(gl, str(self.pt / GEL_FILES[0]))
        self.assertEqual(gr, str(self.pt / GEL_FILES[1]))

    def test_nothing_given_infers_the_epoch_from_the_path(self):
        cams, gl, gr = resolve_calibration(
            None, None, None, "/data/motherboard/e.h5")
        self.assertEqual(cams, [str(self.mb / f) for f in CAM_FILES])
        self.assertEqual(gl, str(self.mb / GEL_FILES[0]))

    def test_unknown_task_name_lists_the_known_ones(self):
        with self.assertRaises(KeyError) as ctx:
            resolve_calibration(["cube"], None, None, "/data/test/e.h5")
        self.assertIn("motherboard", str(ctx.exception))
        self.assertIn("pushT", str(ctx.exception))

    def test_unknown_task_in_path_and_nothing_given_still_raises(self):
        with self.assertRaises(KeyError):
            resolve_calibration(None, None, None, "/data/test/2026-09-08/e.h5")


if __name__ == "__main__":
    unittest.main()
