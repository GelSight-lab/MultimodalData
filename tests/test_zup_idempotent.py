"""Converting a tree twice must not rotate anything twice.

`convert_tree` rotates every pose column of every parquet under `meta/`. That
was correct when a release tree held one coordinate convention: everything was
recorded Y-up and the whole tree moved together.

The tree no longer does. 2026-05/06 episodes are already Z-up (32 motherboard,
4 pushT) and the 2026-09 sessions are Y-up (15 and 22). Rotating the whole tree
now turns the older half by R_x(-90) a SECOND time, for a net -180 deg.

Nothing would catch it downstream. The conversion's own safety argument is that
projections are invariant — poses and `T_mocap_to_cam` move together — so every
preview, overlay and clip still renders correctly no matter how many times it
is applied. Only the numbers are wrong, and only against the world frame.

So the guard has to be here: a parquet that already declares Z-up passes
through untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The script resolves `react_paths` and `react_toolbox` relative to its own
# directory, the way it is run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm"))


def _tree(tmp_path, up_axis, pose=(1.0, 2.0, 3.0)):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / "src" / "meta" / "2026-05-10"
    d.mkdir(parents=True)
    n = 4
    t = pa.table({
        "sensor_left_pose": [[*pose, 0.0, 0.0, 0.0, 1.0]] * n,
        "sensor_right_pose": [[*pose, 0.0, 0.0, 0.0, 1.0]] * n,
    })
    md = {b"twm.world_frame": json.dumps({"up_axis": up_axis}).encode()}
    pq.write_table(t.replace_schema_metadata(md), str(d / "episode_000.parquet"))
    (tmp_path / "src" / "calibration").mkdir(parents=True)
    # up_axis lives HERE, not in the parquet metadata — that is the whole
    # reason the first guard never fired.
    (tmp_path / "src" / "episodes.jsonl").write_text(json.dumps(
        {"episode": "2026-05-10/episode_000", "date": "2026-05-10",
         "up_axis": up_axis}) + "\n")
    return tmp_path / "src", tmp_path / "dst"


def _poses(path):
    import pyarrow.parquet as pq
    t = pq.read_table(str(path))
    return np.asarray(t["sensor_left_pose"].to_pylist(), float)


def test_a_yup_parquet_is_rotated(tmp_path):
    from convert_release_zup import convert_tree
    src, dst = _tree(tmp_path, "y")
    convert_tree(src, dst, "motherboard")
    out = _poses(dst / "meta/2026-05-10/episode_000.parquet")
    # R_x(-90): (x, y, z) -> (x, -z, y)
    np.testing.assert_allclose(out[0, :3], [1.0, -3.0, 2.0], atol=1e-9)


def test_a_parquet_already_declaring_zup_is_left_alone(tmp_path):
    """The whole point: a second pass must be a no-op, not another rotation."""
    from convert_release_zup import convert_tree
    src, dst = _tree(tmp_path, "z")
    convert_tree(src, dst, "motherboard")
    out = _poses(dst / "meta/2026-05-10/episode_000.parquet")
    np.testing.assert_allclose(out[0, :3], [1.0, 2.0, 3.0], atol=1e-9)


def test_converting_the_output_again_changes_nothing(tmp_path):
    from convert_release_zup import convert_tree
    src, dst = _tree(tmp_path, "y")
    convert_tree(src, dst, "motherboard")
    once = _poses(dst / "meta/2026-05-10/episode_000.parquet")
    twice_dst = tmp_path / "dst2"
    convert_tree(dst, twice_dst, "motherboard")
    twice = _poses(twice_dst / "meta/2026-05-10/episode_000.parquet")
    np.testing.assert_allclose(once, twice, atol=1e-9)
