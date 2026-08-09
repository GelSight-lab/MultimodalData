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


def check_no_column_loss(ops: list, repo: str = HF_REPO) -> list[str]:
    """A parquet may not replace a published one that has columns it lacks.

    The six force columns are not produced by this pipeline. `build` writes 19
    columns; `force_recovery.export_force_columns` reads those, adds
    `force_{left,right}_{normal_n,penetration_mm,target_pose}`, and
    `upload_force_columns` publishes the 25-column superset to the SAME path.
    So the published file is not the file this module last wrote, and
    `parquet_operations(STAGE_ROOT)` targets all 36 of them.

    Re-publishing after any rebuild would therefore have silently deleted the
    force channel from a dataset whose README documents it, leaving 36 files
    that look completely normal. Nothing about a 19-column parquet says a
    column is missing.

    The schema is read through HfFileSystem, which range-reads the parquet
    footer instead of downloading the file.
    """
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    problems = []
    for op in ops:
        if not op.path_in_repo.endswith(".parquet"):
            continue
        remote = f"datasets/{repo}/{op.path_in_repo}"
        try:
            with fs.open(remote, "rb") as fh:
                published = set(pq.read_schema(fh).names)
        except FileNotFoundError:
            continue                      # new file, nothing to lose
        except Exception as exc:          # noqa: BLE001
            problems.append(f"{op.path_in_repo}: cannot read the published "
                            f"schema to compare against ({exc}) — refusing "
                            f"rather than overwriting blind")
            continue
        local = set(pq.read_schema(str(op.path_or_fileobj)).names)
        lost = sorted(published - local)
        if lost:
            problems.append(
                f"{op.path_in_repo}: would drop {lost} — the published file "
                f"has columns this one does not. Re-run "
                f"force_recovery.export_force_columns and publish the "
                f"superset instead")
    return problems


def publish(ops: list, message: str, repo: str = HF_REPO, dry_run: bool = False,
            check_columns: bool = True):
    """Push one commit to the dataset repo."""
    if check_columns:
        problems = check_no_column_loss(ops, repo)
        if problems:
            raise SystemExit("refusing to publish — column loss:\n  "
                             + "\n  ".join(problems[:10]))
    if dry_run or not ops:
        return {"committed": 0, "dry_run": True,
                "paths": [o.path_in_repo for o in ops][:20]}
    from huggingface_hub import HfApi

    HfApi().create_commit(repo_id=repo, repo_type="dataset",
                          operations=ops, commit_message=message)
    return {"committed": len(ops), "dry_run": False}
