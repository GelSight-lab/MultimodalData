"""A force-only parquet already HAS its force channel.

`force_overlay_plan` decides which episodes still need the force columns
merged in. It asked `set(FORCE_COLUMNS) <= names` -- all eight -- so a parquet
exported with `--force-only` reads as "force not yet inline", and the gate goes
looking for a force-stage file to overlay on top of one that is already
complete.

The question the gate is actually asking is "does this file carry the force
channel", and the answer is the MEASURED half. The derived half is a control
policy that may legitimately be absent.

`curation.force_flag` asks the same question with a set intersection and was
already right; this is the one place that asked for the superset.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.dataset_layout import FORCE_MEASURED, FORCE_DERIVED


def _write(path, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 4
    t = {"frame_idx": np.arange(n, dtype=np.int32)}
    for c in cols:
        t[c] = (pa.array([[0.0] * 7] * n) if c.endswith("_pose")
                else np.zeros(n, np.float32))
    pq.write_table(pa.table(t), str(path))


@pytest.fixture
def stage(tmp_path):
    root = tmp_path / "release_cut"
    _write(root / "rope" / "meta" / "2026-09-16" / "episode_000.parquet",
           FORCE_MEASURED)                      # force-only: already complete
    _write(root / "rope" / "meta" / "2026-09-16" / "episode_001.parquet",
           list(FORCE_MEASURED) + list(FORCE_DERIVED))    # full export
    _write(root / "rope" / "meta" / "2026-09-16" / "episode_002.parquet", [])
    return root


def test_a_force_only_episode_is_not_queued_for_overlay(stage, tmp_path):
    from twm.scripts.build_release_publish import force_overlay_plan
    files, missing = force_overlay_plan(stage, tmp_path / "force_stage",
                                        ("rope",), since="2026-09-10")
    queued = {Path(p).stem for p in files} | {str(m) for m in missing}
    assert not any("episode_000" in q for q in queued), (
        "a force-only parquet was treated as still needing its force channel")


def test_a_full_export_is_also_left_alone(stage, tmp_path):
    from twm.scripts.build_release_publish import force_overlay_plan
    files, missing = force_overlay_plan(stage, tmp_path / "force_stage",
                                        ("rope",), since="2026-09-10")
    queued = {Path(p).stem for p in files} | {str(m) for m in missing}
    assert not any("episode_001" in q for q in queued)


def test_an_episode_with_no_force_at_all_is_still_noticed(stage, tmp_path):
    from twm.scripts.build_release_publish import force_overlay_plan
    files, missing = force_overlay_plan(stage, tmp_path / "force_stage",
                                        ("rope",), since="2026-09-10")
    blob = " ".join(list(map(str, files)) + list(map(str, missing)))
    assert "episode_002" in blob, (
        "an episode with no force columns was silently accepted")


from pathlib import Path  # noqa: E402
