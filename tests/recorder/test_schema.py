import numpy as np
import h5py
import hdf5plugin
import pytest

from twm.recorder.frames import Tick, synthetic_tick
from twm.recorder.schema import (append_optitrack, append_ticks,
                                 count_optitrack_samples, create_episode_file,
                                 tick_from_legacy, write_episode_attrs)


@pytest.fixture
def episode(tmp_path):
    f, path = create_episode_file(str(tmp_path), 0, ["A", "B", "C"], ["L", "R"], 30,
                                  task_name="t")
    yield f, path
    if f.id.valid:
        f.close()


def test_append_ticks_writes_every_modality_and_optitrack(episode):
    f, _ = episode
    t0 = synthetic_tick(10.0, seed=0)
    t1 = Tick(11.0, color=t0.color, depth=t0.depth, gelsight=t0.gelsight,
              gelsight_ts=(10.9, 10.95),
              optitrack={"motherboard": [(10.5, [1, 2, 3, 0, 0, 0, 1])],
                         "sensor_left": [], "sensor_right": []})
    append_ticks(f, [t0, t1])

    np.testing.assert_allclose(f["timestamps"][:], [10.0, 11.0])
    assert f["realsense/cam2/color"].shape == (2, 480, 640, 3)
    assert f["realsense/cam2/depth"].shape == (2, 480, 640)
    np.testing.assert_array_equal(f["gelsight/right/frames"][1], t0.gelsight[1])
    np.testing.assert_allclose(f["gelsight/right/timestamps"][:], [10.0, 10.95])
    np.testing.assert_allclose(f["optitrack/motherboard/pose"][:],
                               [[1, 2, 3, 0, 0, 0, 1]])
    assert count_optitrack_samples(f) == 1


def test_append_ticks_skips_absent_groups(tmp_path):
    f, _ = create_episode_file(str(tmp_path), 1, [], [], 30, include_legacy=False)
    append_ticks(f, [Tick(1.0)])
    assert f["timestamps"].shape == (1,)
    assert "realsense" not in f
    f.close()


def test_append_ticks_with_empty_batch_is_noop(episode):
    f, _ = episode
    append_ticks(f, [])
    assert f["timestamps"].shape == (0,)


def test_tick_from_legacy_falls_back_to_tick_timestamp():
    frame = np.zeros((480, 640, 3), np.uint8)
    depth = np.zeros((480, 640), np.uint16)
    t = tick_from_legacy(([frame] * 3, [depth] * 3, [frame, frame], 7.0,
                          [None, 6.9]))
    assert t.gelsight_ts == (7.0, 6.9)
    assert t.arducam == ()
    t2 = tick_from_legacy((None, None, None, 7.0, None, [frame, frame], [None, None]))
    assert t2.color == () and t2.arducam_ts == (7.0, 7.0)


def test_write_episode_attrs_round_trips(episode):
    f, path = episode
    write_episode_attrs(f, {"valid": False, "invalid_reason": "overload",
                            "frame_count": 3, "max_tick_gap_s": 0.7})
    f.close()
    with h5py.File(path, "r") as g:
        m = g["metadata"].attrs
        assert bool(m["valid"]) is False
        assert m["invalid_reason"] == "overload"
        assert int(m["frame_count"]) == 3
        assert abs(float(m["max_tick_gap_s"]) - 0.7) < 1e-9


def test_n_realsense_controls_group_count_and_append(tmp_path):
    from twm.recorder.schema import count_realsense_groups
    f, _ = create_episode_file(str(tmp_path), 3, ["A", "B"], ["L", "R"], 30, n_realsense=2)
    assert count_realsense_groups(f) == 2 and "realsense/cam2" not in f
    t = synthetic_tick(1.0, n_realsense=2)
    append_ticks(f, [t])
    assert f["realsense/cam1/color"].shape[0] == 1
    f.close()
    g, _ = create_episode_file(str(tmp_path), 4, ["A", "B", "C"], ["L", "R"], 30)
    assert count_realsense_groups(g) == 3        # legacy default
    g.close()


def test_arducam_groups_record_configured_serial(tmp_path):
    from twm.sensor_camera import CameraSlot, ResolvedCamera
    cams = (ResolvedCamera(CameraSlot("cam0", serial="TWML0001", position="left"),
                           "/dev/video10", "TWML0001", "usb-0:12.2"),
            ResolvedCamera(CameraSlot("cam1", serial="TWMR0001", position="right"),
                           "/dev/video6", "TWMR0001", "usb-0:12.1"))
    f, path = create_episode_file(str(tmp_path), 0, [], [], 30, arducam_config=cams,
                                  include_legacy=False)
    f.close()
    with h5py.File(path, "r") as g:
        assert g["arducam/cam0"].attrs["serial"] == "TWML0001"
        assert g["arducam/cam0"].attrs["usb_path"] == "usb-0:12.2"
        assert g["arducam/cam1"].attrs["position"] == "right"
        import json
        meta = json.loads(g["metadata"].attrs["arducam_config"])
        assert [m["serial"] for m in meta] == ["TWML0001", "TWMR0001"]


def test_frame_datasets_use_lz4_bitshuffle(tmp_path):
    f, _ = create_episode_file(str(tmp_path), 9, ["A"], ["L", "R"], 30, n_realsense=1)
    plist = f["gelsight/left/frames"].id.get_create_plist()
    filters = [plist.get_filter(i) for i in range(plist.get_nfilters())]
    blosc = [fl for fl in filters if fl[0] == hdf5plugin.BLOSC_ID]
    assert blosc, filters
    cd_values = blosc[0][2]
    assert cd_values[4] == 5                                   # clevel
    assert cd_values[5] == hdf5plugin.Blosc.BITSHUFFLE          # shuffle mode
    f.close()
