"""Publish the task-first video release to HF and remove the old .pt release.

Uploads:
  data/<task>/{calibration,videos,meta(parquet only),previews,
               bad_frames.json,segments.json,episodes.jsonl}
  tasks.json, README.md, examples/react_video_dataset.py
Deletes (old single-task .pt release):
  episodes/, segments/, bad_frames.json, segments.json,
  freeze_intervals.json, figures/episode_previews/, metadata/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationDelete

REPO = "yxma/React"
STAGE = Path("/media/yxma/Disk1/twm/release")
REPO_ROOT = Path("/home/yxma/MultimodalData/twm")
TASKS = ("motherboard", "pushT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_delete", action="store_true")
    args = ap.parse_args()
    api = HfApi()

    # 1. Upload each task's data/ (exclude _detect.pt sidecars)
    for task in TASKS:
        src = STAGE / task
        print(f"[publish] uploading data/{task}/ ...", flush=True)
        if not args.dry_run:
            api.upload_folder(
                repo_id=REPO, repo_type="dataset",
                folder_path=str(src),
                path_in_repo=f"data/{task}",
                ignore_patterns=["*._detect.pt"],
                commit_message=f"Publish {task} video release (MP4+parquet, 640x480) + curation + previews",
            )

    # 2. Upload top-level metadata / loader / README
    print("[publish] uploading tasks.json, README, loader ...", flush=True)
    if not args.dry_run:
        from huggingface_hub import CommitOperationAdd
        ops = [
            CommitOperationAdd("tasks.json", "/tmp/tasks_v2.json"),
            CommitOperationAdd("README.md",
                               str(REPO_ROOT / "docs/superpowers/specs/README_v2_release.md")),
            CommitOperationAdd("examples/react_video_dataset.py",
                               str(REPO_ROOT / "examples/react_video_dataset.py")),
        ]
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message="Multi-task video release: tasks.json + README + ReactVideoDataset loader")

    # 3. Delete old .pt release paths
    if not args.no_delete:
        print("[publish] deleting old .pt release paths ...", flush=True)
        files = api.list_repo_files(REPO, repo_type="dataset")
        stale = [f for f in files if (
            f.startswith("episodes/") or f.startswith("segments/")
            or f in ("bad_frames.json", "segments.json", "freeze_intervals.json")
            or f.startswith("figures/episode_previews/")
            or f.startswith("metadata/")
            or f.startswith("examples/react_window_dataset")
            or f.startswith("examples/react_segment_dataset")
            or f.startswith("examples/demo_react")
            or f.startswith("examples/play_react_pt")
        )]
        print(f"[publish] {len(stale)} stale files to delete", flush=True)
        if not args.dry_run and stale:
            ops = [CommitOperationDelete(path_in_repo=f) for f in stale]
            # batch deletes (HF handles large op lists)
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message="Remove superseded single-task .pt release (episodes/, segments/, old previews, root JSONs)")
    print("[publish] done", flush=True)


if __name__ == "__main__":
    main()
