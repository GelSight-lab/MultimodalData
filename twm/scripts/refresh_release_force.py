"""Push refreshed force columns into the cut release. No re-encoding.

    python twm/scripts/refresh_release_force.py --task rope [--apply]

For every published segment on or after `--since`, takes the force columns
from the uncut Z-up episode it was cut from and writes them into the segment
parquet. The videos are not opened, so a segment does not gain a generation of
H.264 loss for a change that never touched a pixel.

Dry by default: a run that rewrites published parquet should have to be asked
for twice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.pipeline_stages import RELEASE_CUT, RELEASE_ZUP, SCOPE_SINCE, TASKS  # noqa: E402
from twm.react_preprocess.refresh_force import refresh_segment  # noqa: E402

SEG = "_seg"


def plan(task: str, since: str, cut: Path, zup: Path) -> list[tuple[Path, Path]]:
    """(published parquet, source episode parquet) for everything in scope.

    A published unit is NOT always named `_segNN`. `cut_episode` copies an
    episode verbatim when its only span covers the whole recording -- "it has
    nothing to cut" -- so `episode_000.parquet` sits beside
    `episode_001_seg00.parquet`. Globbing only `*_seg*` skipped 3 of
    motherboard's 25 published units, which would have shipped stale force
    values while everything around them was refreshed.
    """
    out = []
    for seg in sorted((cut / task / "meta").rglob("episode_*.parquet")):
        date = seg.parent.name
        if since and date < since:
            continue                      # out of scope; not this run's business
        episode = seg.stem.split(SEG)[0]  # a whole-episode unit maps to itself
        src = zup / task / "meta" / date / f"{episode}.parquet"
        out.append((seg, src))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=TASKS, action="append", default=None)
    ap.add_argument("--since", default=SCOPE_SINCE)
    ap.add_argument("--cut", type=Path, default=RELEASE_CUT)
    ap.add_argument("--zup", type=Path, default=RELEASE_ZUP)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    tasks = a.task or list(TASKS)
    missing, done, n_rows = [], 0, 0
    for task in tasks:
        jobs = plan(task, a.since, a.cut, a.zup)
        print(f"[{task}] {len(jobs)} segment(s) on or after {a.since}")
        for seg, src in jobs:
            if not src.is_file():
                missing.append(f"{task}/{seg.parent.name}/{seg.stem}: "
                               f"no source episode {src}")
                continue
            if not a.apply:
                continue
            r = refresh_segment(seg, src)
            done += 1
            n_rows += r["rows"]
    if missing:
        for m in missing[:10]:
            print("  MISSING:", m)
        print(f"refusing: {len(missing)} segment(s) have no source episode")
        return 1
    if not a.apply:
        print("\n(dry run — pass --apply to rewrite the segment parquets)")
        return 0
    print(f"\nrefreshed {done} segment(s), {n_rows} rows; no video touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
