"""Rebuild a task's published index files from the set of published episodes.

    python twm/scripts/project_release_index.py --task pushT [--add-date 2026-09-12]
                                                [--check] [--upload]

`episodes.jsonl`, `segments.json`, `bad_frames.json`, `segment_provenance.json`
and `splits.json` sit beside the data on the Hub. Each is a per-episode body
plus a summary, and the local copies are ACCUMULATED: every `segment` run
rewrites them in place, so they carry whatever the last invocation happened to
see and whatever the local tree happens to hold.

That is how `data/motherboard/segment_provenance.json` came to be published
claiming `"episodes": 0, "kept_minutes": 0.0` beside a body listing 14
segments -- the summary described a run that had nothing left to cut -- and how
the local pushT index came to list 11 segments from 2026-06-18 that the release
does not publish at all.

This script derives each file instead: the key set is what the Hub actually
serves (plus `--add-date`, for the day being published in the same pass), and
`react_preprocess.index_projection` computes each document from it.

`--check` compares without writing, so it can run as a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twm.react_preprocess import index_projection as P  # noqa: E402
from twm.react_preprocess.meta import TASK_INDEX  # noqa: E402

CUT_ROOT = Path("/media/yxma/Disk1/twm/release_cut")
REPO = "yxma/React"

# name -> (loader, projector). `splits.json` is projected only if present:
# rope has never had one and motherboard's covers 42 of its 47 episodes.
FILES = {
    "episodes.jsonl": ("jsonl", P.project_episodes),
    "segments.json": ("json", P.project_segments),
    "bad_frames.json": ("json", P.project_bad_frames),
    "segment_provenance.json": ("json", P.project_provenance),
    "splits.json": ("json", P.project_splits),
}


def published_keys(task: str, api) -> list[str]:
    """`<date>/<episode>` for every parquet the Hub serves under this task."""
    out = []
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        p = f.split("/")
        if (len(p) == 5 and p[0] == "data" and p[1] == task
                and p[2] == "meta" and p[4].endswith(".parquet")):
            out.append(f"{p[3]}/{p[4][: -len('.parquet')]}")
    return sorted(out)


def local_keys(task: str, date: str) -> list[str]:
    d = CUT_ROOT / task / "meta" / date
    return sorted(f"{date}/{p.stem}" for p in d.glob("*.parquet"))


def load(path: Path, kind: str):
    if kind == "jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())


def run(task: str, keys: list[str], check: bool) -> tuple[list[Path], list[str]]:
    root = CUT_ROOT / task
    written, notes = [], []
    for name, (kind, project) in FILES.items():
        path = root / name
        if not path.is_file():
            notes.append(f"{name}: 本地没有，跳过")
            continue
        doc = load(path, kind)
        out = project(doc, keys)
        if name == "episodes.jsonl":
            before, after = len(doc), len(out)
        elif name == "segment_provenance.json":
            before, after = len(doc["segments"]), len(out["segments"])
            notes.append(f"  provenance summary: {json.dumps(out['summary'], ensure_ascii=False)}")
        elif name == "segments.json":
            before, after = doc["n_segments"], out["n_segments"]
        elif name == "bad_frames.json":
            before, after = len(doc["episodes"]), len(out["episodes"])
        else:
            before, after = len(doc.get("episodes", {})), len(out.get("episodes", {}))
            miss = P.uncovered(keys, doc)
            if miss:
                notes.append(f"  splits.json 未覆盖 {len(miss)} 个已发布 episode "
                             f"(ReactVideoDataset 会把它们静默算作 train): "
                             f"{miss[:4]}{'...' if len(miss) > 4 else ''}")
        print(f"  {name:26s} {before:4d} -> {after:4d} 条")
        if not check:
            (P.write_jsonl if kind == "jsonl" else P.write_json)(path, out)
            written.append(path)
    return written, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_INDEX))
    ap.add_argument("--add-date", action="append", default=[],
                    help="a date being published in this same pass; its local "
                         "parquets count as published")
    ap.add_argument("--check", action="store_true", help="report, do not write")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    keys = set(published_keys(a.task, api))
    on_hub = len(keys)
    for d in a.add_date:
        keys |= set(local_keys(a.task, d))
    keys = sorted(keys)
    print(f"[{a.task}] Hub 上 {on_hub} 个 + 本次新增 {len(keys) - on_hub} 个 "
          f"= {len(keys)} 个已发布 episode")
    written, notes = run(a.task, keys, a.check)
    for n in notes:
        print(n)
    if a.upload and written:
        api.upload_folder(repo_id=REPO, repo_type="dataset",
                          folder_path=str(CUT_ROOT / a.task),
                          path_in_repo=f"data/{a.task}",
                          allow_patterns=[p.name for p in written],
                          commit_message=f"{a.task}: index files derived from the "
                                         f"{len(keys)} published episodes")
        print(f"[{a.task}] 已上传 {len(written)} 个索引文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
