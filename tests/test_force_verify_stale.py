"""A force file whose source episode is gone must be named, not crashed on.

`release_force/` is derived from `release/`: same rows, eight more columns.
When an episode leaves the release tree — 2026-09-09's pushT session was
republished as `data/validation` and removed from `release/pushT` — its force
copy stays behind, and `verify` walks the force tree and opens the release
parquet beside each file:

    FileNotFoundError: release/pushT/meta/2026-09-09/episode_000.parquet

The export itself had already succeeded: 91 episodes, 182 sensor-sides,
857,184 rows written. Its own verification then died on one orphan, taking the
stage's exit code with it and stopping every stage behind it.

Crashing is the wrong answer, and so is skipping quietly. A file with no
source is a real finding — it is the residue of a move nobody finished — so it
is reported by name and the rest of the verification runs.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import twm.force_recovery.export_force_columns as EX


def _pq(path, n=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"frame_idx": np.arange(n, dtype=np.int32)}),
                   str(path))


@pytest.fixture
def trees(tmp_path, monkeypatch):
    force, rel = tmp_path / "force", tmp_path / "release"
    _pq(force / "pushT/meta/2026-09-11/episode_000.parquet")
    _pq(rel / "pushT/meta/2026-09-11/episode_000.parquet")
    _pq(force / "pushT/meta/2026-09-09/episode_000.parquet")   # orphan
    monkeypatch.setattr(EX, "EXPORT_ROOT", force, raising=False)
    monkeypatch.setattr(EX, "STAGE_ROOT", rel)
    return force, rel


def test_an_orphan_is_reported_by_name(trees):
    force, rel = trees
    orphans = EX.orphan_force_files(force, rel)
    assert orphans == ["pushT/2026-09-09/episode_000"]


def test_an_episode_with_its_source_is_not_an_orphan(trees):
    force, rel = trees
    assert "pushT/2026-09-11/episode_000" not in EX.orphan_force_files(force, rel)


def test_no_orphans_is_an_empty_list_not_a_crash(tmp_path, monkeypatch):
    force, rel = tmp_path / "f", tmp_path / "r"
    _pq(force / "rope/meta/2026-09-14/episode_000.parquet")
    _pq(rel / "rope/meta/2026-09-14/episode_000.parquet")
    assert EX.orphan_force_files(force, rel) == []
