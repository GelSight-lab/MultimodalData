"""Render the release preview panels.

Thin adapter: `react_preprocess.previews` decides *what* to render (which
episodes, which calibration, trim offset, world-frame offset, output path) and
`build_episode_previews` does the drawing — the 1280x480 panel of 3 cams +
OptiTrack + GelSight raw/diff + projection overlay. The renderer stays here
rather than in the package because it needs rig-local calibration that the
release does not ship.

    python scripts/build_release_previews.py --task motherboard [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_episode_previews as BEP
from react_preprocess import previews


def make_renderer(task: str, clip_s: float, speed: float,
                  force_root: Path | None = None):
    """Bind the renderer to one task's calibration epoch.

    `_load_proj_calibs(task)` resolves the epoch through `calib_epoch` and
    refuses a wrong one. The old version monkeypatched a module-level
    `CALIB_DIR` — the exact global whose one-value-for-all-tasks default put
    pushT's extrinsics under every motherboard preview.
    """
    # Per SESSION, not per task. Binding one calibration to the whole task is
    # the same defect the docstring above describes, one level down: motherboard
    # spans the 2026-05-12 and 2026-09-09 epochs, so a task-wide binding renders
    # one of them through the other's extrinsics.
    _by_date: dict = {}

    def _calibs(date: str):
        if date not in _by_date:
            _by_date[date] = BEP._load_proj_calibs(task, date)
        return _by_date[date]

    def render(job: dict) -> None:
        project_cams, glc, grc, _ = _calibs(job["date"])
        dx, dy, dz = job["world_offset"]
        # The builder reads the trim from the release parquet itself, so this
        # used to monkeypatch the sidecar reader it consulted instead. That
        # hook is gone; assert the two agree rather than silently letting one
        # win — a plan and a renderer disagreeing about where an episode
        # starts is precisely the pushT pre-roll defect.
        want = int(job["trim_offset"])
        got, _ = BEP._parquet_trim_and_rows(
            job["task"], job["date"], job["episode"], job.get("parquet"))
        if got != want:
            raise ValueError(
                f"{job['date']}/{job['episode']}: plan says trim {want}, "
                f"release parquet says {got}")
        # Passed explicitly, always. `build_one_preview` falls back to its
        # module-level FORCE_ROOT -- the production tree -- when this is None,
        # and a preview whose force numbers came from the wrong tree looks
        # exactly like one that didn't. That is the runbook's "never render v8
        # labels over v7 values", and the force writer's own environment does
        # not reach this process.
        BEP.build_one_preview(job["h5"], job["out"], clip_s, speed,
                              project_cams, glc, grc, dx=dx, dy=dy, dz=dz,
                              window_start=want, force_root=force_root)

    return render, len(BEP._load_proj_calibs(task)[0])


def shard(jobs: list[dict], index: int, count: int) -> list[dict]:
    """The `index`-th of `count` disjoint slices, interleaved.

    Interleaved rather than contiguous because the plan is ordered by date and
    episode, and a recording's cost tracks its length: contiguous blocks hand
    one worker a run of long episodes and another a run of short ones.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard {index} of {count} is out of range")
    return jobs[index::count]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(previews.CALIB_DIRS))
    ap.add_argument("--clip-s", type=float, default=previews.CLIP_SECONDS)
    ap.add_argument("--speed", type=float, default=previews.SPEED)
    ap.add_argument("--stage-root", default=None,
                    help="tree whose published videos drive the plan. Defaults "
                         "to STAGE_ROOT; point it at release_cut to render a "
                         "preview per PUBLISHED segment rather than per source "
                         "episode")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help="render only the I-th of N disjoint slices, so N "
                         "processes can share one task without rendering the "
                         "same episode twice (the renderer sits at half a core)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-root", default=None,
                    help="force-recovery tree the overlays read (default: the "
                         "production tree). Point this at a candidate run's "
                         "force/ directory when reprocessing, or the preview "
                         "labels a new estimate with the old values.")
    args = ap.parse_args()

    from pathlib import Path as _P
    stage = _P(args.stage_root) if args.stage_root else previews.STAGE_ROOT
    i, n = (int(x) for x in args.shard.split("/"))
    all_jobs = list(previews.plan(args.task, stage))
    jobs = shard(all_jobs, i, n)
    print(f"[previews] {args.task}: {len(jobs)} of {len(all_jobs)} episodes "
          f"(shard {i}/{n}), calib={previews.CALIB_DIRS[args.task].name}",
          flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['date']}/{j['episode']} trim={j['trim_offset']} "
                  f"world_offset={j['world_offset']}")
        return 0

    froot = _P(args.force_root) if args.force_root else None
    render, n_cams = make_renderer(args.task, args.clip_s, args.speed,
                                   force_root=froot)
    print(f"[previews] projection cameras: {n_cams}, "
          f"force={froot or 'production default'}", flush=True)

    results = previews.build_task(args.task, render, stage_root=stage,
                                  overwrite=args.overwrite, jobs=jobs)
    for r in results:
        detail = (f"({r['bytes']/1024:.0f} KB)" if r["status"] == "OK"
                  else r.get("error", ""))
        print(f"  {r['date']}/{r['episode']}: {r['status']} {detail}", flush=True)

    ok = sum(r["status"] == "OK" for r in results)
    skipped = sum(r["status"] == "SKIP" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)
    print(f"[previews] done — {ok} rendered, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
