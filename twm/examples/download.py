"""Download helper for the React dataset — depth is optional.

Examples
--------
    # Core only (RGB + tactile + poses), no depth (~4.4 GB)
    python examples/download.py --no-depth

    # Everything including depth (~37 GB)
    python examples/download.py

    # One task, no depth
    python examples/download.py --task motherboard --no-depth

    # Depth too, only the middle camera depth
    python examples/download.py --depth-cams middle

    # Choose where to put it
    python examples/download.py --no-depth --local-dir ./react
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO = "yxma/React"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default=None,
                    help="Limit to one task (e.g. motherboard, pushT). Default: all.")
    ap.add_argument("--no-depth", action="store_true",
                    help="Skip the ~33 GB depth tree (RGB + tactile + poses only).")
    ap.add_argument("--depth-cams", nargs="*", default=None,
                    choices=["left", "middle", "right"],
                    help="If downloading depth, restrict to these cameras "
                         "(e.g. --depth-cams middle). Default: all 3.")
    ap.add_argument("--local-dir", default=None,
                    help="Target directory (default: HF cache).")
    args = ap.parse_args()

    allow = None
    ignore = []

    if args.task:
        allow = [f"data/{args.task}/*", "README.md", "tasks.json", "examples/*"]

    if args.no_depth:
        ignore.append("*/depth/*")
    elif args.depth_cams:
        # keep only requested depth cams: ignore the others
        for cam in ("left", "middle", "right"):
            if cam not in args.depth_cams:
                ignore.append(f"*/depth/*/depth_{cam}.mkv")

    print(f"Downloading {REPO}"
          f"{' task='+args.task if args.task else ''}"
          f"{' (no depth)' if args.no_depth else ''}"
          f"{' depth_cams='+','.join(args.depth_cams) if args.depth_cams else ''} ...")
    path = snapshot_download(
        REPO, repo_type="dataset",
        allow_patterns=allow, ignore_patterns=ignore or None,
        local_dir=args.local_dir,
    )
    print(f"Done -> {path}")


if __name__ == "__main__":
    main()
