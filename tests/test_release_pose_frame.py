"""Which axis convention a release parquet's poses are in must be declared.

The renderer works in the recorded Y-up frame. Handing it a Z-up pose put the
DexForce target 515 mm away; treating a Y-up pose as Z-up does the same thing
in reverse, and that is what shipped for 2026-09-09: `curate` wrote those
rows without `up_axis`, so the lookup fell through to the task's CALIBRATION
file, which says "z" about the extrinsics and nothing about the poses. The
sensor axes stayed right — they come from the H5 — and only the force-induced
target moved, which is a defect that looks like a calibration error.
"""
import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from twm.scripts.build_episode_previews import _release_poses  # noqa: E402


def _tree(tmp_path, up_axis, in_calibration="z"):
    root = tmp_path / "motherboard"
    (root / "meta" / "2026-09-09").mkdir(parents=True)
    t = pa.table({"sensor_left_pose": [[0.1, 0.2, 0.3, 0, 0, 0, 1]] * 4,
                  "sensor_right_pose": [[0.4, 0.5, 0.6, 0, 0, 0, 1]] * 4})
    pq.write_table(t, root / "meta" / "2026-09-09" / "episode_000.parquet")
    row = {"episode": "2026-09-09/episode_000", "date": "2026-09-09", "n_frames": 4}
    if up_axis is not None:
        row["up_axis"] = up_axis
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (root / "calibration").mkdir()
    (root / "calibration" / "T_mocap_to_cam_middle.json").write_text(
        json.dumps({"up_axis": in_calibration}))
    return root


def _patched(monkeypatch, root):
    import react_preprocess.config as cfg
    monkeypatch.setattr(cfg, "STAGE_ROOT", root.parent)


def test_a_declared_y_up_release_is_used_as_is(tmp_path, monkeypatch):
    _patched(monkeypatch, _tree(tmp_path, "y"))
    out = _release_poses("motherboard", "2026-09-09", "episode_000")
    np.testing.assert_allclose(out["left"][0][:3], [0.1, 0.2, 0.3])


def test_a_declared_z_up_release_is_rotated_into_the_renderer_frame(tmp_path, monkeypatch):
    _patched(monkeypatch, _tree(tmp_path, "z"))
    out = _release_poses("motherboard", "2026-09-09", "episode_000")
    assert not np.allclose(out["left"][0][:3], [0.1, 0.2, 0.3])


def test_an_undeclared_convention_raises_instead_of_reading_the_calibration(
        tmp_path, monkeypatch):
    """The calibration file's up_axis describes the extrinsics. Using it as a
    proxy for the poses is how a Y-up parquet got rotated as if Z-up."""
    _patched(monkeypatch, _tree(tmp_path, None, in_calibration="z"))
    with pytest.raises(ValueError, match="up_axis"):
        _release_poses("motherboard", "2026-09-09", "episode_000")
