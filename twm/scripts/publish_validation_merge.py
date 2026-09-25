"""Publish the merged `data/validation` in three commits, delete last.

The pushT episode's videos and depth already exist in the repo as LFS objects,
so they are *copied server-side* rather than re-uploaded: 2.2 GB that never
crosses the wire, and a copy that cannot differ from the original because no
bytes are re-encoded. Its parquet is NOT copied -- renumbering rewrote the
`episode`/`episode_index` columns, so the staged file is uploaded instead.

The old path is removed in a separate, LAST commit, after the copy is verified
present. A delete that shares a commit with the thing replacing it has no state
in between to check, so a bad copy would take the original with it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = "yxma/React"
OLD = "data/pushT_2026-09-09"
NEW = "data/validation"
DATE = "2026-09-09"
OLD_EP, NEW_EP = "episode_000", "episode_003"
STREAMS = ("view_left", "view_middle", "view_right", "tactile_left",
           "tactile_right", "wrist_left", "wrist_right")
DEPTHS = ("depth_left", "depth_middle", "depth_right")


def copy_ops():
    from huggingface_hub import CommitOperationCopy
    ops = []
    for s in STREAMS:
        ops.append(CommitOperationCopy(
            src_path_in_repo=f"{OLD}/videos/{DATE}/{OLD_EP}/{s}.mp4",
            path_in_repo=f"{NEW}/videos/{DATE}/{NEW_EP}/{s}.mp4"))
    for d in DEPTHS:
        ops.append(CommitOperationCopy(
            src_path_in_repo=f"{OLD}/depth/{DATE}/{OLD_EP}/{d}.mkv",
            path_in_repo=f"{NEW}/depth/{DATE}/{NEW_EP}/{d}.mkv"))
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="/media/yxma/Disk1/twm/publish/validation")
    ap.add_argument("--step", choices=["copy", "upload", "delete", "all"],
                    default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    stage = Path(a.stage)

    def have(prefix: str) -> set[str]:
        return {f for f in api.list_repo_files(REPO, repo_type="dataset")
                if f.startswith(prefix)}

    if a.step in ("copy", "all"):
        ops = copy_ops()
        print(f"[copy] {len(ops)} 个 LFS 对象 {OLD}/{OLD_EP} -> {NEW}/{NEW_EP}")
        if not a.dry_run:
            api.create_commit(REPO, repo_type="dataset", operations=ops,
                              commit_message=f"validation: {OLD_EP} of pushT 2026-09-09 "
                                             f"becomes {NEW_EP} (server-side copy)")
        print("[copy] 完成")

    if a.step in ("upload", "all"):
        files = sorted(p for p in stage.rglob("*") if p.is_file()
                       and p.name != "_manifest.json")
        mb = sum(p.stat().st_size for p in files) / 1e6
        print(f"[upload] {len(files)} 个文件 {mb:.0f} MB")
        # Every staged episode needs its preview. NEW_EP is the one exception
        # and a permanent one: the panel is rendered from the source recording
        # and that H5 is deleted, so waiting for it would never end.
        staged = sorted(p.stem for p in (stage / "meta" / DATE).glob("*.parquet"))
        missing = [e for e in staged if e != NEW_EP
                   and not (stage / "previews" / DATE / f"{e}.mp4").is_file()]
        if missing:
            print(f"[upload] 拒绝：缺少预览 {missing}", file=sys.stderr)
            return 1
        if not a.dry_run:
            api.upload_folder(repo_id=REPO, repo_type="dataset",
                              folder_path=str(stage), path_in_repo=NEW,
                              ignore_patterns=["_manifest.json"],
                              commit_message="validation: add the 2026-09-09 17:20/17:41 "
                                             "motherboard segments, renumbered 004-008; "
                                             "merge the indices")
        print("[upload] 完成")

    if a.step in ("delete", "all"):
        landed = have(f"{NEW}/videos/{DATE}/{NEW_EP}/")
        if len(landed) != len(STREAMS):
            print(f"[delete] 拒绝：{NEW_EP} 只有 {len(landed)}/{len(STREAMS)} 个视频",
                  file=sys.stderr)
            return 1
        old = have(f"{OLD}/")
        print(f"[delete] {len(old)} 个文件 {OLD}/  （{NEW_EP} 已确认落地）")
        if not a.dry_run:
            api.delete_folder(path_in_repo=OLD, repo_id=REPO, repo_type="dataset",
                              commit_message=f"drop {OLD}: moved into {NEW} as {NEW_EP}")
        print("[delete] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
