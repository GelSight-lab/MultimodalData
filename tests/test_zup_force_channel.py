"""The Z-up tree is what gets CUT, so it has to carry the force channel.

The force estimator writes a second staging tree, `release_force/`, whose
parquet are `release/`'s plus eight columns at the same paths. Publishing an
UNCUT release uploads both trees over one another and the reader ends up with
the union — which is why nothing noticed that `convert_release_zup` reads
`release/`, the tree WITHOUT the columns.

Publishing the CUT tree breaks that. The Hub holds `episode_003_seg00`; the
force tree holds `episode_003`; no overlay can address a segment from an
uncut name. The force columns have to be inside the segment parquet, which
means inside the Z-up parquet it was cut from.

Measured 2026-09-14, after the Z-up tree was rebuilt: 24 of the 25 published
pushT/2026-09-12 segments carry force, and `episode_003_seg00` — the one cut
after the rebuild — does not. Every later cut would have joined it.

`force_{left,right}_target_pose` is a POSE. Merging the force columns must
put them through the same rotation as every other pose column, or the force
channel arrives in the frame the rest of the file just left.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# The script resolves `react_paths` and `react_toolbox` relative to its own
# directory, the way it is run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm"))

import twm.scripts.convert_release_zup as Z  # noqa: E402

FORCE_SCALARS = ("force_left_normal_n", "force_right_normal_n")
TARGET = "force_left_target_pose"


def _tree(root, task, date, ep, *, up_axis, n=6, with_force=False):
    """A release tree of one episode, and its episodes.jsonl declaration."""
    md = root / task / "meta" / date
    md.mkdir(parents=True, exist_ok=True)
    pose = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]] * n
    cols = {"frame_idx": np.arange(n, dtype=np.int32),
            "sensor_left_pose": pose, "sensor_right_pose": pose,
            "object_pose": pose}
    if with_force:
        cols["force_left_normal_n"] = np.linspace(0, 3, n)
        cols["force_right_normal_n"] = np.linspace(3, 0, n)
        cols[TARGET] = pose
        cols["force_right_target_pose"] = pose
    t = pa.table(cols).replace_schema_metadata(
        {b"twm.world_frame": json.dumps({"up_axis": up_axis}).encode()})
    pq.write_table(t, str(md / f"{ep}.parquet"))
    (root / task / "episodes.jsonl").write_text(json.dumps(
        {"episode": f"{date}/{ep}", "date": date, "up_axis": up_axis}) + "\n")
    (root / task / "calibration").mkdir(parents=True, exist_ok=True)
    return root / task


@pytest.fixture
def trees(tmp_path):
    src = _tree(tmp_path / "release", "pushT", "2026-09-12", "episode_003",
                up_axis="y")
    force = _tree(tmp_path / "release_force", "pushT", "2026-09-12",
                  "episode_003", up_axis="y", with_force=True)
    return src, force, tmp_path / "zup" / "pushT"


def _out(dst):
    return pq.read_table(str(dst / "meta/2026-09-12/episode_003.parquet"))


def test_the_zup_parquet_carries_the_force_columns(trees):
    src, force, dst = trees
    Z.convert_tree(src, dst, "pushT", force_src=force)
    names = _out(dst).column_names
    for c in FORCE_SCALARS:
        assert c in names, f"{c} was dropped — the cut tree loses the channel"


def test_the_force_values_survive_the_merge_unchanged(trees):
    """A scalar force is frame-independent; the rotation must not touch it."""
    src, force, dst = trees
    Z.convert_tree(src, dst, "pushT", force_src=force)
    got = np.asarray(_out(dst)["force_left_normal_n"].to_pylist(), float)
    assert np.allclose(got, np.linspace(0, 3, 6))


def test_the_force_target_pose_is_rotated_with_every_other_pose(trees):
    """It is a POSE. Copying it raw lands the force channel in the Y-up frame
    the rest of the file has just left."""
    src, force, dst = trees
    Z.convert_tree(src, dst, "pushT", force_src=force)
    t = _out(dst)
    tgt = np.asarray(t[TARGET].to_pylist(), float)
    sensor = np.asarray(t["sensor_left_pose"].to_pylist(), float)
    assert np.allclose(tgt[:, :3], sensor[:, :3]), \
        "the force target pose did not move with the sensor pose it mirrors"
    assert not np.allclose(tgt[:, :3], [0.1, 0.2, 0.3]), \
        "the force target pose was copied through unrotated"


def test_an_already_zup_episode_gains_the_columns_without_rotating_twice(
        tmp_path):
    """The 2026-05/06 episodes were converted before the 2026-09 sessions were
    recorded. They still need the channel; they must not be rotated again."""
    src = _tree(tmp_path / "release", "pushT", "2026-05-11", "episode_000",
                up_axis="z")
    force = _tree(tmp_path / "release_force", "pushT", "2026-05-11",
                  "episode_000", up_axis="z", with_force=True)
    dst = tmp_path / "zup" / "pushT"
    Z.convert_tree(src, dst, "pushT", force_src=force)
    t = pq.read_table(str(dst / "meta/2026-05-11/episode_000.parquet"))
    assert "force_left_normal_n" in t.column_names
    assert np.allclose(np.asarray(t[TARGET].to_pylist(), float)[:, :3],
                       [0.1, 0.2, 0.3]), "an already-Z-up pose was rotated"


def test_an_episode_with_no_force_counterpart_is_named_not_dropped_silently(
        tmp_path, capsys):
    """'No silent fallback in resolvers' is a pipeline invariant. An episode
    the estimator has not reached still converts — but it says so."""
    src = _tree(tmp_path / "release", "pushT", "2026-09-12", "episode_009",
                up_axis="y")
    force = tmp_path / "release_force" / "pushT"
    force.mkdir(parents=True)
    dst = tmp_path / "zup" / "pushT"
    Z.convert_tree(src, dst, "pushT", force_src=force)
    out = capsys.readouterr().out
    assert "episode_009" in out and "force" in out.lower(), \
        f"the missing force channel was not reported: {out!r}"
