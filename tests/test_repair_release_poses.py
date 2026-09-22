"""Applying the pose repair to a release must keep everything else exact.

`force_<side>_target_pose` is the observed pose displaced along the pressing
direction, and BOTH terms depend on the quaternion -- the direction is
`R(q) @ gel_axis`. So repairing a pose without recomputing the target leaves a
target built on the glitched orientation, which is the same defect one step
downstream and harder to see.

Everything that is not a pose or a target must come through bit-identical: the
force values themselves are measured from gel images and have nothing to do
with where the rig thought the sensor was.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
import repair_release_poses as RRP  # noqa: E402


def _episode(path, n=60, glitch_at=30):
    ang = np.linspace(0, 0.6, n)
    q = R.from_euler("z", ang).as_quat()
    xyz = np.stack([np.linspace(0, 0.3, n), np.zeros(n), np.zeros(n)], 1)
    pose = np.concatenate([xyz, q], 1)
    bad = pose.copy()
    bad[glitch_at, 3:] = (R.from_euler("x", np.pi / 2)
                          * R.from_quat(pose[glitch_at, 3:])).as_quat()
    force = np.linspace(0, 6, n).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "source_h5_frame": np.arange(n, dtype=np.int32),
        "sensor_left_pose": [list(map(float, r)) for r in bad],
        "sensor_right_pose": [list(map(float, r)) for r in pose],
        "force_left_normal_n": force,
        "force_right_normal_n": force,
        "force_left_penetration_mm": force / 2,
        "force_right_penetration_mm": force / 2,
        "force_left_target_pose": [list(map(float, r)) for r in bad],
        "force_right_target_pose": [list(map(float, r)) for r in pose],
        "force_left_source_frame": np.arange(n, dtype=np.int32),
        "force_right_source_frame": np.arange(n, dtype=np.int32),
    }), str(path))
    return path, pose


def test_the_glitched_frame_is_repaired(tmp_path):
    p, clean = _episode(tmp_path / "e.parquet")
    r = RRP.repair_parquet(p, "motherboard")
    assert r["frames"] == 1
    got = np.array(pq.read_table(str(p))["sensor_left_pose"].to_pylist())
    a = R.from_quat(got[30, 3:]); b = R.from_quat(clean[30, 3:])
    assert np.degrees((a.inv() * b).magnitude()) < 1.0


def test_the_target_pose_is_recomputed_not_left_stale(tmp_path):
    p, _ = _episode(tmp_path / "e.parquet")
    RRP.repair_parquet(p, "motherboard")
    t = pq.read_table(str(p))
    pose = np.array(t["sensor_left_pose"].to_pylist())
    tgt = np.array(t["force_left_target_pose"].to_pylist())
    pen = t["force_left_penetration_mm"].to_numpy()
    d = np.linalg.norm(tgt[:, :3] - pose[:, :3], axis=1) * 1000.0
    contact = t["force_left_normal_n"].to_numpy() > 0
    assert np.allclose(d[contact], pen[contact], atol=1e-6)


def test_the_target_quaternion_follows_the_repaired_pose(tmp_path):
    p, _ = _episode(tmp_path / "e.parquet")
    RRP.repair_parquet(p, "motherboard")
    t = pq.read_table(str(p))
    pose = np.array(t["sensor_left_pose"].to_pylist())
    tgt = np.array(t["force_left_target_pose"].to_pylist())
    assert np.allclose(tgt[:, 3:], pose[:, 3:])


def test_the_force_values_are_untouched(tmp_path):
    """Force comes from gel images; where the rig thought the sensor was has
    nothing to do with it."""
    p, _ = _episode(tmp_path / "e.parquet")
    before = pq.read_table(str(p))["force_left_normal_n"]
    RRP.repair_parquet(p, "motherboard")
    assert pq.read_table(str(p))["force_left_normal_n"].equals(before)


def test_untouched_columns_are_bit_identical(tmp_path):
    p, _ = _episode(tmp_path / "e.parquet")
    before = pq.read_table(str(p))
    RRP.repair_parquet(p, "motherboard")
    after = pq.read_table(str(p))
    keep = [c for c in before.column_names
            if "pose" not in c and "penetration" not in c]
    for c in keep:
        assert before[c].equals(after[c]), c


def test_a_clean_side_is_not_rewritten(tmp_path):
    p, _ = _episode(tmp_path / "e.parquet")
    before = pq.read_table(str(p))["sensor_right_pose"]
    RRP.repair_parquet(p, "motherboard")
    assert pq.read_table(str(p))["sensor_right_pose"].equals(before)


def test_running_it_twice_changes_nothing_the_second_time(tmp_path):
    p, _ = _episode(tmp_path / "e.parquet")
    RRP.repair_parquet(p, "motherboard")
    mid = pq.read_table(str(p))
    r2 = RRP.repair_parquet(p, "motherboard")
    assert r2["frames"] == 0
    assert pq.read_table(str(p)).equals(mid)


def test_a_segment_maps_by_source_frame_not_by_position(tmp_path):
    """`source_h5_frame` is the RAW H5 frame number, not a row index.

    They coincide only when trim == 0, which is true of 45 of 46 motherboard
    episodes -- so indexing the uncut pose array with source_h5_frame appeared
    to work everywhere. 2026-05-11/episode_017 has trim = 19228 (its uncut
    source_h5_frame runs 19228..33738), and there the same indexing runs off
    the end. It was never right; it was hidden.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "twm" / "scripts"))
    import repair_release_poses as RRP

    uncut_sf = np.arange(19228, 19228 + 100)
    seg_sf = np.arange(19300, 19320)
    idx = RRP.rows_for(uncut_sf, seg_sf)
    assert idx.tolist() == list(range(72, 92))
    assert uncut_sf[idx].tolist() == seg_sf.tolist()


def test_a_segment_frame_outside_the_episode_refuses(tmp_path):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "twm" / "scripts"))
    import repair_release_poses as RRP
    with pytest.raises(ValueError, match="source frame"):
        RRP.rows_for(np.arange(0, 100), np.array([50, 500]))
