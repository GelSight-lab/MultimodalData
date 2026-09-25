"""The tree that gets published has to carry the calibration it is expressed in.

`convert_release_zup` copies `calibration/` into the Z-up tree, so the Z-up
poses and the Z-up `T_mocap_to_cam` travel together — verified on real data at
0.00000 px of projection difference across all three cameras.

`segment` does not copy it. So `release_cut`, the tree the chain actually
publishes, has no `calibration/` at all, and the publish runs with
`--no_delete`: the Hub keeps whatever an earlier uncut publish left there. On
2026-09-14 that was the Y-up matrix — and its JSON is labelled `up_axis: z`,
so nothing about the file says it is the wrong one. Z-up poses through that
matrix land 487-543 px off, half a frame.

Anyone computing an action from the published data reads those two files
together. They have to describe the same world.
"""
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import twm.react_preprocess.segment as S
import twm.scripts.build_release_publish as P


def _calib(root, up_axis="z"):
    d = root / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    for side in ("left", "middle", "right"):
        (d / f"T_mocap_to_cam_{side}.json").write_text(json.dumps(
            {"T_mocap_to_cam": np.eye(4).tolist(), "up_axis": up_axis,
             "intrinsics": {"fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0}}))
    for side in ("left", "right"):
        (d / f"T_gel_to_rigid_{side}.json").write_text(json.dumps({"x": 1}))
    return d


def test_the_cut_tree_carries_the_calibration_of_the_tree_it_was_cut_from(
        tmp_path):
    src, dst = tmp_path / "zup", tmp_path / "cut"
    _calib(src / "pushT")
    (src / "pushT" / "meta").mkdir(parents=True, exist_ok=True)
    S.copy_calibration(src / "pushT", dst / "pushT")
    got = sorted(p.name for p in (dst / "pushT/calibration").glob("*.json"))
    assert got == sorted(p.name for p in (src / "pushT/calibration").glob("*.json"))


def test_a_calibration_already_there_is_replaced_not_merged(tmp_path):
    """A stale file left from an earlier convention is exactly the failure
    this exists to prevent; it must not survive because nothing overwrote it."""
    src, dst = tmp_path / "zup", tmp_path / "cut"
    _calib(src / "pushT", up_axis="z")
    stale = _calib(dst / "pushT", up_axis="y")
    (stale / "T_mocap_to_cam_leftover.json").write_text("{}")
    S.copy_calibration(src / "pushT", dst / "pushT")
    d = dst / "pushT/calibration"
    assert not (d / "T_mocap_to_cam_leftover.json").exists(), \
        "a file with no counterpart in the source survived the copy"
    assert json.loads((d / "T_mocap_to_cam_left.json").read_text())["up_axis"] == "z"


def test_publishing_a_tree_with_no_calibration_is_refused(tmp_path):
    """--no_delete means the Hub keeps the old one. Uploading poses without
    the matrix they are expressed in silently re-points every projection."""
    stage = tmp_path / "release_cut"
    (stage / "pushT" / "meta" / "2026-09-12").mkdir(parents=True)
    pq.write_table(pa.table({"frame_idx": np.arange(2, dtype=np.int32)}),
                   str(stage / "pushT/meta/2026-09-12/episode_000_seg00.parquet"))
    bad = P.check_calibration_present(stage, ("pushT",))
    assert bad and "calibration" in bad[0]


def test_a_tree_with_its_calibration_passes(tmp_path):
    stage = tmp_path / "release_cut"
    (stage / "pushT" / "meta" / "2026-09-12").mkdir(parents=True)
    _calib(stage / "pushT")
    assert P.check_calibration_present(stage, ("pushT",)) == []
