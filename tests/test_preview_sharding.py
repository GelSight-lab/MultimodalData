"""Splitting one task's previews across processes must not double-render.

A preview takes ~70 s and the renderer sits at half a core, so the way to use
an idle machine is more processes. Two processes over the same job list race:
`build_task` skips an output that already EXISTS, which is false until the
other process finishes writing, so both start the same episode and the loser's
ffmpeg writes over a file the winner is still producing.

Sharding by position makes the lists disjoint before anything runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
import build_release_previews as BRP  # noqa: E402


def test_shards_partition_the_jobs():
    jobs = [{"episode": f"e{i}"} for i in range(23)]
    got = [BRP.shard(jobs, i, 4) for i in range(4)]
    assert sum(len(g) for g in got) == len(jobs)
    seen = [j["episode"] for g in got for j in g]
    assert sorted(seen) == sorted(j["episode"] for j in jobs)
    assert len(set(seen)) == len(seen)          # nothing rendered twice


def test_one_shard_is_everything():
    jobs = [{"episode": f"e{i}"} for i in range(5)]
    assert BRP.shard(jobs, 0, 1) == jobs


def test_shards_stay_balanced():
    jobs = [{"episode": f"e{i}"} for i in range(23)]
    sizes = sorted(len(BRP.shard(jobs, i, 4)) for i in range(4))
    assert sizes[-1] - sizes[0] <= 1
