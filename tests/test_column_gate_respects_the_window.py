"""The column-loss gate must ask about the files the run will overwrite.

`scoped_patterns` applies `--since` to every parquet upload, the force overlay
skips dates below it, and the only deletes are legacy `.pt` paths outside
`data/`. So a run with `--since D` cannot touch a published parquet dated
before D.

`check_no_column_loss` walked the whole task tree anyway. On 2026-09-22 that
refused a publish of pushT/2026-09-17 by reporting 68 segments from 09-10
through 09-15 -- files the run would neither upload nor delete. Their local
copies genuinely hold 29 columns against 60 published, because the action and
controller-target channel is produced by a different branch, so the finding is
true about the tree and false about the upload.

That matters beyond the inconvenience: the gate's own docstring says it asks
"for every parquet it is about to overwrite", and a gate that blocks the right
answer gets bypassed with --skip-gate, which is how the previous one stopped
being run at all.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import twm.scripts.build_release_publish as BP


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """One task, two dates: an old published one and a new unpublished one."""
    stage = tmp_path / "cut"
    for date, cols in (("2026-09-10", ["frame_idx"]),
                       ("2026-09-17", ["frame_idx"])):
        d = stage / "pushT" / "meta" / date
        d.mkdir(parents=True)
        pq.write_table(pa.table({c: np.arange(4, dtype=np.int64) for c in cols}),
                       str(d / "episode_000_seg00.parquet"))
    monkeypatch.setattr(BP, "STAGE", stage)
    monkeypatch.setattr(BP, "FORCE_STAGE", tmp_path / "force_absent")

    published = {"data/pushT/meta/2026-09-10/episode_000_seg00.parquet",
                 "data/pushT/meta/2026-09-17/episode_000_seg00.parquet"}

    class _Api:
        def repo_info(self, *a, **k):
            sib = [types.SimpleNamespace(rfilename=p) for p in published]
            return types.SimpleNamespace(siblings=sib)

    # the published file carries a column the local tree lacks
    rich = tmp_path / "published.parquet"
    pq.write_table(pa.table({"frame_idx": np.arange(4, dtype=np.int64),
                             "action": np.arange(4, dtype=np.int64)}), str(rich))
    monkeypatch.setattr(BP, "hf_hub_download", lambda *a, **k: str(rich),
                        raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(rich))
    return _Api()


def test_without_a_window_both_dates_are_reported(staged):
    bad = BP.check_no_column_loss(staged, ["pushT"])
    assert len(bad) == 2, bad


def test_a_window_excludes_the_dates_the_run_cannot_touch(staged):
    bad = BP.check_no_column_loss(staged, ["pushT"], since="2026-09-17")
    assert len(bad) == 1, bad
    assert "2026-09-17" in bad[0]
    assert "2026-09-10" not in bad[0]


def test_the_window_still_catches_loss_inside_it(staged):
    """Scoping must narrow the question, never soften it."""
    bad = BP.check_no_column_loss(staged, ["pushT"], since="2026-09-10")
    assert len(bad) == 2, bad
