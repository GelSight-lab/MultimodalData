"""Command-line entry point.

    python -m react_preprocess build --task pushT [--date D] [--with-depth]
    python -m react_preprocess audit --root /path/to/data/pushT
    python -m react_preprocess backfill-flags --root /path/to/data/pushT
    python -m react_preprocess verify-flags --root /path/to/data/pushT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backfill
from .config import H5_ROOTS, STAGE_ROOT
from .h5io import discover
from .pipeline import build_episode


def cmd_build(args) -> int:
    root = H5_ROOTS[args.task]
    paths = discover(args.task, root, args.date, args.episodes)
    if not paths:
        print(f"no source recordings under {root}", file=sys.stderr)
        return 1
    print(f"[build] {args.task}: {len(paths)} episodes -> {STAGE_ROOT}", flush=True)
    failures = 0
    for p in paths:
        report = build_episode(p, args.task, force=args.force,
                               with_depth=args.with_depth,
                               encode_video=not args.meta_only)
        print(f"  {report}", flush=True)
        failures += report.status == "FAIL"
    print(f"[build] done ({failures} failed)")
    return 1 if failures else 0


def _resolve_roots(args) -> list[Path]:
    if args.root:
        return [Path(args.root)]
    return [STAGE_ROOT / t for t in (H5_ROOTS if not args.task else [args.task])]


def cmd_audit(args) -> int:
    for root in _resolve_roots(args):
        reports = backfill.process_tree(root, dry_run=True)
        if not reports:
            print(f"[audit] {root}: no parquet found")
            continue
        summary = backfill.aggregate(reports)
        print(f"[audit] {root}")
        print(f"  episodes={summary['episodes']} rows={summary['rows']}")
        print(f"  unique tactile frames = {summary['unique_tactile']} "
              f"({summary['duplicate_ratio']*100:.1f}% duplicated)")
        print(f"  effective tactile rate = {summary['effective_fps']:.1f} fps "
              f"(rows written at 30 fps)")
        print(f"  longest frozen run    = {summary['max_repeat_run']} frames "
              f"({summary['max_repeat_run']/30:.2f} s)")
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"summary": summary, "episodes": reports}, indent=2))
            print(f"  wrote {args.json}")
    return 0


def cmd_backfill(args) -> int:
    for root in _resolve_roots(args):
        reports = backfill.process_tree(root, dry_run=args.dry_run)
        if not reports:
            print(f"[backfill] {root}: no parquet found")
            continue
        summary = backfill.aggregate(reports)
        verb = "would update" if args.dry_run else "updated"
        print(f"[backfill] {verb} {len(reports)} parquet under {root}")
        print(f"  duplicate ratio {summary['duplicate_ratio']*100:.1f}% "
              f"-> effective {summary['effective_fps']:.1f} fps")
    return 0


def cmd_verify(args) -> int:
    """Check the recovered flags.

    Against source H5 this is exact. Against the published MP4s it can only be
    approximate, because H.264 is lossy: a duplicated source frame does not
    decode back to identical pixels.
    """
    task = args.task or "pushT"
    root = Path(args.root) if args.root else STAGE_ROOT / task
    pqs = sorted(root.rglob("meta/**/episode_*.parquet"))[: args.limit_episodes]
    if not pqs:
        print(f"no parquet under {root}", file=sys.stderr)
        return 1

    bad = 0
    for pq_path in pqs:
        date, ep = pq_path.parent.name, pq_path.stem
        if args.against == "h5":
            h5 = H5_ROOTS[task] / date / f"{ep}.h5"
            if not h5.exists():
                print(f"  {ep}: source H5 missing, skipped")
                continue
            search = range(args.shift_min, args.shift_max + 1) if args.shift is None else None
            res = backfill.verify_against_h5(pq_path, h5, args.side, args.frames,
                                             shift=args.shift, search=search)
            ok = res["mismatches"] == 0
            bad += not ok
            how = "detected" if res["shift_detected"] else "given"
            print(f"  {ep}: {'OK      ' if ok else 'MISMATCH'} "
                  f"n={res['compared']} mismatches={res['mismatches']} "
                  f"tactile_shift={res['shift']:+d} ({how}) "
                  f"unique proxy/source = {res['proxy_unique']}/{res['source_unique']}")
        else:
            video = root / "videos" / date / ep / f"tactile_{args.side}.mp4"
            if not video.exists():
                print(f"  {ep}: no video, skipped")
                continue
            res = backfill.verify_against_video(pq_path, video, args.frames, args.tol)
            ok = res["mismatches"] == 0
            bad += not ok
            print(f"  {ep}: {'OK      ' if ok else 'MISMATCH'} "
                  f"n={res['compared']} mismatches={res['mismatches']} "
                  f"unique proxy/video = {res['proxy_unique']}/{res['video_unique']} "
                  f"MAD dup<={res['mad_duplicate_max']:.2f} new>={res['mad_new_min']:.2f}")
    print(f"[verify] against {args.against}: {len(pqs)} episodes, {bad} mismatching")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="react_preprocess")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="source H5 -> published release")
    b.add_argument("--task", required=True, choices=sorted(H5_ROOTS))
    b.add_argument("--date")
    b.add_argument("--episodes", nargs="*")
    b.add_argument("--force", action="store_true")
    b.add_argument("--with-depth", action="store_true")
    b.add_argument("--meta-only", action="store_true",
                   help="recompute parquet without re-encoding video")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("audit", help="report tactile duplication")
    a.add_argument("--root")
    a.add_argument("--task", choices=sorted(H5_ROOTS))
    a.add_argument("--json")
    a.set_defaults(func=cmd_audit)

    f = sub.add_parser("backfill-flags", help="add tactile_*_is_new to parquet")
    f.add_argument("--root")
    f.add_argument("--task", choices=sorted(H5_ROOTS))
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(func=cmd_backfill)

    v = sub.add_parser("verify-flags", help="check flags against ground truth")
    v.add_argument("--root")
    v.add_argument("--task", choices=sorted(H5_ROOTS))
    v.add_argument("--against", choices=("h5", "video"), default="h5",
                   help="h5 = exact (source pixels); video = approximate (lossy)")
    v.add_argument("--side", choices=("left", "right"), default="left")
    v.add_argument("--frames", type=int, default=600)
    v.add_argument("--tol", type=float, default=1.0,
                   help="mean-abs-diff threshold when checking against video")
    v.add_argument("--shift", type=int, default=None,
                   help="known baked-in tactile shift; omit to auto-detect")
    v.add_argument("--shift-min", type=int, default=0)
    v.add_argument("--shift-max", type=int, default=20)
    v.add_argument("--limit-episodes", type=int, default=2)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
