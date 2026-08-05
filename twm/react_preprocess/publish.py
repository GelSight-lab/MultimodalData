"""Publish the release — and the code that produced it — to the Hub.

The dataset ships both halves of its own tooling so a download is
self-contained and reproducible:

    toolbox/      react_toolbox     — consume the data
    preprocess/   react_preprocess  — rebuild it from source recordings

Git stays the source of truth for both; this module mirrors them onto the
dataset repo (and into the local staging tree, so the code sits next to the
data on disk too).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import HF_REPO, STAGE_ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent                    # twm/

CODE_TREES = {
    "preprocess": SOURCE_ROOT / "react_preprocess",
    "toolbox": SOURCE_ROOT / "react_toolbox",
}

SKIP = {"__pycache__", ".pytest_cache", ".DS_Store"}


def _files(tree: Path) -> list[Path]:
    return sorted(p for p in tree.rglob("*")
                  if p.is_file() and not any(s in p.parts for s in SKIP)
                  and p.suffix in {".py", ".md", ".txt", ".json", ".toml"})


def stage_code(stage_root: Path = STAGE_ROOT) -> dict[str, int]:
    """Copy the code trees next to the data in the local staging area."""
    counts = {}
    for name, tree in CODE_TREES.items():
        if not tree.exists():
            continue
        dest = Path(stage_root) / name
        if dest.exists():
            shutil.rmtree(dest)
        n = 0
        for src in _files(tree):
            out = dest / src.relative_to(tree)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            n += 1
        counts[name] = n
    return counts


def code_operations(trees: dict[str, Path] | None = None) -> list:
    """CommitOperationAdd entries mirroring the code trees onto the repo."""
    from huggingface_hub import CommitOperationAdd

    ops = []
    for name, tree in (trees or CODE_TREES).items():
        if not tree.exists():
            continue
        for src in _files(tree):
            ops.append(CommitOperationAdd(
                path_in_repo=f"{name}/{src.relative_to(tree).as_posix()}",
                path_or_fileobj=str(src)))
    return ops


def parquet_operations(stage_root: Path, tasks=("motherboard", "pushT")) -> list:
    """CommitOperationAdd entries for every per-episode parquet."""
    from huggingface_hub import CommitOperationAdd

    ops = []
    for task in tasks:
        meta = Path(stage_root) / task / "meta"
        for src in sorted(meta.rglob("episode_*.parquet")):
            ops.append(CommitOperationAdd(
                path_in_repo=f"data/{task}/meta/{src.parent.name}/{src.name}",
                path_or_fileobj=str(src)))
    return ops


def publish(ops: list, message: str, repo: str = HF_REPO, dry_run: bool = False):
    """Push one commit to the dataset repo."""
    if dry_run or not ops:
        return {"committed": 0, "dry_run": True,
                "paths": [o.path_in_repo for o in ops][:20]}
    from huggingface_hub import HfApi

    HfApi().create_commit(repo_id=repo, repo_type="dataset",
                          operations=ops, commit_message=message)
    return {"committed": len(ops), "dry_run": False}
