"""Add the LeRobot index columns to the already-published cut segments.

Every folder published before the cut release carries `task`, `task_index`,
`episode`, `episode_index` and `frame_index` inside the parquet. The 79
segments of the cut release carry none of them, so a loader keyed on the
column works on the old folders and silently returns nothing on the new ones.

`react_preprocess.meta.add_index_columns` is the authority for what those
columns are. `enrich_parquet_index.py` grew a second copy that wrote them as
int32 while every published parquet has int64; it now delegates here rather
than diverge again.

`task_index` is APPEND-ONLY. The ints live inside every parquet already
downloaded, so renumbering a task relabels someone's local copy with no way to
notice.

The local parquet is replaced through a temp file and an atomic rename: the
preview renderer reads these same files for their trim offsets, and a reader
that catches a half-written parquet fails in a way that looks like corruption.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twm.react_preprocess.meta import TASK_INDEX, add_index_columns  # noqa: E402

CUT_ROOT = Path("/media/yxma/Disk1/twm/release_cut")
REPO = "yxma/React"


def episode_indices(keys) -> dict[str, int]:
    """0-based index per `<date>/<episode>` key, sorted, over the whole task.

    Sorted, not encounter order, so the numbering is a property of the folder
    rather than of whatever order the filesystem handed back. Zero-padded
    `segNN` means plain string sort already puts seg09 before seg10.
    """
    return {k: i for i, k in enumerate(sorted(set(keys)))}


def published(task: str, api) -> list[str]:
    """The `<date>/<episode>` keys this task actually publishes."""
    out = []
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        p = f.split("/")
        if (len(p) == 5 and p[0] == "data" and p[1] == task
                and p[2] == "meta" and p[4].endswith(".parquet")):
            out.append(f"{p[3]}/{p[4][:-len('.parquet')]}")
    return sorted(out)


def upload_patterns(keys) -> list[str]:
    """Exactly the files this folder publishes -- no globs.

    The local tree is not the published tree. `meta/**/*.parquet` swept up
    motherboard's 2026-09-09 segments, which belong to `data/validation`, and
    re-created them under `data/motherboard` as parquets with no videos beside
    them. A pattern that names each published key cannot do that.
    """
    return [f"meta/{k}.parquet" for k in keys]


def rewrite(task: str, keys: list[str], dry_run: bool) -> list[Path]:
    idx = episode_indices(keys)
    touched = []
    for key in keys:
        date, ep = key.split("/")
        path = CUT_ROOT / task / "meta" / date / f"{ep}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{task}: {key} is published but not local")
        t = pq.read_table(str(path))
        have = [c for c in ("task", "task_index", "episode", "episode_index",
                            "frame_index") if c in t.column_names]
        t = add_index_columns(t, task, TASK_INDEX[task], key, idx[key])
        if not dry_run:
            tmp = path.with_suffix(".parquet.tmp")
            pq.write_table(t, str(tmp), compression="zstd")
            os.replace(tmp, path)          # atomic: no reader sees a partial file
        touched.append(path)
        if len(touched) <= 3 or len(touched) == len(keys):
            print(f"    {key:34s} ep_index={idx[key]:3d} 行={t.num_rows:6d} "
                  f"原有索引列={have or '无'}")
    return touched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", choices=list(TASK_INDEX),
                    help="repeatable; default all three")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    for task in (a.task or list(TASK_INDEX)):
        keys = published(task, api)
        print(f"  [{task}] {len(keys)} 个已发布 parquet")
        rewrite(task, keys, a.dry_run)
        if a.dry_run or a.no_upload:
            continue
        api.upload_folder(repo_id=REPO, repo_type="dataset",
                          folder_path=str(CUT_ROOT / task),
                          path_in_repo=f"data/{task}",
                          allow_patterns=upload_patterns(keys),
                          commit_message=f"{task}: parquets gain the LeRobot index "
                                         f"columns (task/task_index/episode/"
                                         f"episode_index/frame_index)")
        print(f"  [{task}] 已上传")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
