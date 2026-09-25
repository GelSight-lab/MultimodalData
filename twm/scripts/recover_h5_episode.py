"""CLI over ``react_preprocess.repair`` for recovering a crashed recording.

All of the reasoning — how the damage is diagnosed, how the chunk index is
rebuilt from orphaned B-tree leaves, and above all when a recovered file may
NOT become a published episode — lives in that module, because the pipeline
runs the same code automatically. This file only parses arguments.

    python scripts/recover_h5_episode.py diagnose --src <broken.h5>
    python scripts/recover_h5_episode.py index    --src <broken.h5> --out <idx.json>
    python scripts/recover_h5_episode.py write    --src <broken.h5> --idx <idx.json> \
        --ref <healthy.h5> --out <recovered.h5> [--limit N]
    python scripts/recover_h5_episode.py verify   --src <broken.h5> --idx <idx.json> \
        --out <recovered.h5> [--samples 200] [--ref <healthy.h5>]
    python scripts/recover_h5_episode.py auto     --src <broken.h5> [--ref <healthy.h5>]

`auto` is what the pipeline calls: diagnose, recover if the damage is the one
validated signature, verify, and then report whether the result is publishable.
It usually is not — see the module docstring for why that is the point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.react_preprocess import repair                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["diagnose", "index", "write", "verify",
                                     "auto"])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--idx", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N frames — runs the whole path on a "
                         "handful of frames so it is proven before an hour "
                         "of copying rather than after")
    args = ap.parse_args()

    if args.mode == "diagnose":
        print(repair.diagnose(args.src))
        return 0

    if args.mode == "index":
        def progress(scanned, hits):
            print(f"  scanned {scanned/1e9:6.1f} GB  TREE hits {hits}",
                  flush=True)
        idx = repair.build_chunk_index(args.src, progress)
        print(f"[recover] {idx['n_chains']} dataset chains")
        for ci, v in idx["chains"].items():
            print(f"  chain{ci}: rank={v['ndim']} chunks={v['n']}")
        repair.save_index(idx, args.out)
        print(f"[recover] wrote {args.out}")
        return 0

    if args.mode == "auto":
        path, note = repair.ensure_readable(args.src, reference=args.ref)
        print(f"[recover] {note}")
        if path is None:
            return 1
        ok, why = repair.release_eligibility(path)
        print(f"[recover] publishable={ok} — {why}")
        return 0

    idx = repair.load_index(args.idx)
    if args.mode == "write":
        repair.write_recovered(args.src, idx, args.ref, args.out,
                               limit=args.limit)
        return 0

    problems = repair.verify_against_source(args.src, idx, args.out,
                                            samples=args.samples)
    for p in problems[:20]:
        print("  " + p)
    print(f"[verify] byte fidelity: {len(problems)} problem(s)")
    if args.ref:
        ident = repair.verify_stream_identity(args.out, args.ref)
        print(f"[verify] stream identity: "
              f"{'all streams match their names' if ident['ok'] else ident['misassigned']}")
        if not ident["ok"]:
            return 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
