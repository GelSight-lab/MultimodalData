"""Batch worker: process a slice of the release episodes.

    python -m force_recovery.batch_worker <worker_id> <n_workers>

Episodes are interleaved across workers (worker w takes indices w, w+n, ...)
so both make steady progress regardless of episode length. Existing npz
outputs from the CURRENT pipeline are skipped via a version stamp.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .run_episode import OUT_ROOT, STAGE_ROOT, process_side

# Defined by the module that WRITES the npz, so a stamp cannot go missing
# by writing one through a different entry point. Re-exported here because
# `export_force_columns` has always imported it from this module.
from .run_episode import PIPELINE_VERSION  # noqa: F401


def all_episodes() -> list[tuple[str, str, str]]:
    jobs = []
    for task in ("pushT", "motherboard"):
        meta = STAGE_ROOT / task / "meta"
        for p in sorted(meta.rglob("episode_*.parquet")):
            jobs.append((task, p.parent.name, p.stem))
    return jobs


def is_done(task: str, date: str, ep: str, side: str) -> bool:
    p = OUT_ROOT / task / date / f"{ep}_{side}.npz"
    if not p.exists():
        return False
    try:
        with np.load(p) as z:
            return int(z.get("pipeline_version", 0)) >= PIPELINE_VERSION
    except Exception:
        return False


def main() -> int:
    import os

    worker, n_workers = int(sys.argv[1]), int(sys.argv[2])
    jobs = all_episodes()[worker::n_workers]
    if os.environ.get("REVERSE"):
        # extra workers eat the same slice from the tail; the version-stamp
        # skip makes the meeting point cost at most one duplicated episode
        jobs = jobs[::-1]
    print(f"[worker {worker}] {len(jobs)} episodes", flush=True)
    failures = 0
    for task, date, ep in jobs:
        for side in ("left", "right"):
            if is_done(task, date, ep, side):
                print(f"[worker {worker}] skip {task}/{date}/{ep} {side}",
                      flush=True)
                continue
            t0 = time.time()
            try:
                m = process_side(task, date, ep, side)
                # `process_side` stamps the version itself now. This used to
                # decompress and recompress the entire npz to insert one int.
                print(f"[worker {worker}] {task}/{date}/{ep} {side}: "
                      f"max={m['force_max_n']:.2f}N "
                      f"thr={m['contact_threshold_mm']*1e3:.0f}um "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as exc:                     # noqa: BLE001
                failures += 1
                print(f"[worker {worker}] FAIL {task}/{date}/{ep} {side}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    print(f"[worker {worker}] DONE ({failures} failures)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
