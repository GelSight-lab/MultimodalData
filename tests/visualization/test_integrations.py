"""Exercise actual HDF5 panel iteration with optional wrist streams."""
import h5py
import numpy as np
import pytest

from twm.scripts import build_episode_previews as previews


@pytest.mark.parametrize("slots", [[], ["cam0"], ["cam1"], ["cam0", "cam1"]])
def test_batch_preview_optional_wrist_streams(monkeypatch, tmp_path, slots):
    path = tmp_path / "task" / "2026-09-18" / "episode_000.h5"
    path.parent.mkdir(parents=True)
    frames = np.zeros((2, 480, 640, 3), np.uint8)
    with h5py.File(path, "w") as source:
        source["timestamps"] = [0., 1 / 30]
        for cam in range(3):
            source[f"realsense/cam{cam}/color"] = frames
        for side in ("left", "right"):
            source[f"gelsight/{side}/frames"] = frames
        for slot in slots:
            source[f"arducam/{slot}/frames"] = frames

    monkeypatch.setattr(previews, "_parquet_trim_and_rows", lambda *a: (0, 2))
    monkeypatch.setattr(previews, "_flagged_intervals", lambda *a: [])
    monkeypatch.setattr(previews, "load_optitrack", lambda *a: {})
    monkeypatch.setattr(previews, "gel_lag_frames", lambda *a: 0)
    monkeypatch.setattr(previews, "_gel_source_frames", lambda *a: {})
    monkeypatch.setattr(previews, "preview_reference", lambda *a: frames[0])
    monkeypatch.setattr(previews, "calib_describe", lambda *a: "synthetic")
    monkeypatch.setattr(previews, "load_forces", lambda *a: {})
    monkeypatch.setattr(previews, "preview_targets", lambda *a, **kw: {})
    monkeypatch.setattr(previews, "episode_wrist_gamma", lambda *a: 1.)
    panels = list(previews.iter_preview_panels(path, 2 / 30, [], None, None))
    assert len(panels) == 2
    assert panels[0].shape == ((768 if slots else 528), 1280, 3)
