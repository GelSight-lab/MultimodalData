import numpy as np
import pytest

import twm.data_collection as dc
from twm.recorder.writer import WriterOverloaded


def test_facade_exports_legacy_names():
    assert isinstance(dc.REALSENSE_SERIALS, list) and len(dc.REALSENSE_SERIALS) == 3
    assert isinstance(dc.DATA_DIR, str)
    assert dc.FPS == 30
    for name in ("create_episode_file", "append_camera_frame", "append_camera_frames_batch",
                 "flush_optitrack_to_hdf5", "log_episode", "next_episode_number",
                 "make_preview", "make_optitrack_panel", "TRACKER_COLORS", "main"):
        assert callable(getattr(dc, name)) or name == "TRACKER_COLORS"


def test_hdf5writer_adapter_raises_instead_of_dropping(tmp_path):
    f, _ = dc.create_episode_file(str(tmp_path), 0, ["A", "B", "C"], ["L", "R"], 30)
    w = dc.HDF5Writer(maxsize=2, batch_size=1)
    color = [np.zeros((480, 640, 3), np.uint8)] * 3
    depth = [np.zeros((480, 640), np.uint16)] * 3
    gs = [np.zeros((480, 640, 3), np.uint8)] * 2
    overloaded = False
    for t in range(3):
        try:
            w.enqueue(f, color, depth, gs, float(t), gs_timestamps=[None, t - 0.01])
        except WriterOverloaded:
            overloaded = True
            break
    assert overloaded, "expected the tiny (maxsize=2) writer to overload before all 3 ticks fit"
    w.flush()
    w.stop()
    assert w.dropped_frames == 0
    assert f["timestamps"].shape[0] >= 1
    np.testing.assert_allclose(f["gelsight/left/timestamps"][0], 0.0)
    f.close()


def test_log_episode_writes_legacy_row(tmp_path):
    f, path = dc.create_episode_file(str(tmp_path / "t" / "d"), 4, [], [], 30)
    f.close()
    dc.log_episode(str(tmp_path), "t", 4, path, 60, 30, has_optitrack=False, notes="x")
    text = (tmp_path / "dataset_log.csv").read_text()
    assert "ep_004" in text and ",no," in text and text.strip().endswith(",x")
    assert dc.next_episode_number(str(tmp_path / "t" / "d")) == 5
