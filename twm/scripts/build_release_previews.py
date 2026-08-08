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


def make_renderer(task: str, clip_s: float, speed: float):
    """Bind the renderer to one task's calibration epoch.

    `_load_proj_calibs(task)` resolves the epoch through `calib_epoch` and
    refuses a wrong one. The old version monkeypatched a module-level
    `CALIB_DIR` — the exact global whose one-value-for-all-tasks default put
    pushT's extrinsics under every motherboard preview.
    """
    project_cams, glc, grc = BEP._load_proj_calibs(task)

    def render(job: dict) -> None:
        dx, dy, dz = job["world_offset"]
        # The builder reads the trim from the release parquet itself, so this
        # used to monkeypatch the sidecar reader it consulted instead. That
        # hook is gone; assert the two agree rather than silently letting one
        # win — a plan and a renderer disagreeing about where an episode
        # starts is precisely the pushT pre-roll defect.
        want = int(job["trim_offset"])
        got, _ = BEP._parquet_trim_and_rows(
            job["task"], job["date"], job["episode"])
        if got != want:
            raise ValueError(
                f"{job['date']}/{job['episode']}: plan says trim {want}, "
                f"release parquet says {got}")
        BEP.build_one_preview(job["h5"], job["out"], clip_s, speed,
                              project_cams, glc, grc, dx=dx, dy=dy, dz=dz)

    return render, len(project_cams)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(previews.CALIB_DIRS))
    ap.add_argument("--clip-s", type=float, default=previews.CLIP_SECONDS)
    ap.add_argument("--speed", type=float, default=previews.SPEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = list(previews.plan(args.task))
    print(f"[previews] {args.task}: {len(jobs)} episodes, "
          f"calib={previews.CALIB_DIRS[args.task].name}", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['date']}/{j['episode']} trim={j['trim_offset']} "
                  f"world_offset={j['world_offset']}")
        return 0

    render, n_cams = make_renderer(args.task, args.clip_s, args.speed)
    print(f"[previews] projection cameras: {n_cams}", flush=True)

    results = previews.build_task(args.task, render, overwrite=args.overwrite)
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
