"""Upload previews as they finish, in parallel with the renders.

A preview takes ~70 s and there are dozens; waiting for the last one before
sending the first leaves an hour of finished work sitting on the disk.

Two guards, because either failure publishes something broken:

* **Done-ness.** The renderer writes with `-movflags +faststart`, so ffmpeg
  rewrites the file at the end to move the moov atom to the front. Until that
  happens the file cannot be probed at all -- which makes a header probe an
  exact test for "ffmpeg has finished", and a cheap one: it reads the header,
  it does not decode.

* **Membership.** The local tree is not the published tree. motherboard's
  2026-09-09 previews are local, but those episodes live in `data/validation`
  under different numbers; sending them to `data/motherboard` would publish
  previews for episodes that folder does not contain.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CUT_ROOT = Path("/media/yxma/Disk1/twm/release_cut")
REPO = "yxma/React"
TASKS = ("motherboard", "pushT", "rope")


def is_finished(path: Path) -> bool:
    """True once ffmpeg has written the moov atom, i.e. the file is complete."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return r.returncode == 0 and float(r.stdout.strip()) > 0
    except ValueError:
        return False


def pending(local: set[str], remote: set[str], publishes: set[str]) -> list[str]:
    """`<date>/<episode>` keys ready to send: rendered, published, not yet up."""
    return sorted((local & publishes) - remote)


def publishes(task: str, api) -> set[str]:
    """The keys this folder publishes, from its parquets."""
    out = set()
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        p = f.split("/")
        if (len(p) == 5 and p[0] == "data" and p[1] == task
                and p[2] == "meta" and p[4].endswith(".parquet")):
            out.add(f"{p[3]}/{p[4][:-len('.parquet')]}")
    return out


def remote_previews(task: str, api) -> set[str]:
    out = set()
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        p = f.split("/")
        if (len(p) == 5 and p[0] == "data" and p[1] == task
                and p[2] == "previews" and p[4].endswith(".mp4")):
            out.add(f"{p[3]}/{p[4][:-len('.mp4')]}")
    return out


def local_finished(task: str) -> set[str]:
    root = CUT_ROOT / task / "previews"
    return {f"{p.parent.name}/{p.stem}" for p in sorted(root.glob("*/*.mp4"))
            if is_finished(p)} if root.is_dir() else set()


def round_once(api, dry_run: bool) -> tuple[int, int]:
    """One pass over every task. Returns (uploaded, still outstanding)."""
    sent = outstanding = 0
    for task in TASKS:
        pub = publishes(task, api)
        keys = pending(local_finished(task), remote_previews(task, api), pub)
        outstanding += len(pub) - len(remote_previews(task, api)) - len(keys)
        if not keys:
            continue
        print(f"  [{task}] 上传 {len(keys)} 个: {', '.join(k.split('/')[-1] for k in keys[:4])}"
              f"{' ...' if len(keys) > 4 else ''}", flush=True)
        if not dry_run:
            api.upload_folder(
                repo_id=REPO, repo_type="dataset",
                folder_path=str(CUT_ROOT / task), path_in_repo=f"data/{task}",
                allow_patterns=[f"previews/{k}.mp4" for k in keys],
                commit_message=f"{task}: previews for {len(keys)} published segments")
        sent += len(keys)
    return sent, outstanding


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="keep polling while the renders run")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    total = 0
    while True:
        sent, outstanding = round_once(api, a.dry_run)
        total += sent
        print(f"  [{time.strftime('%H:%M:%S')}] 本轮 {sent}，累计 {total}，"
              f"尚未渲染 {outstanding}", flush=True)
        if not a.watch or (outstanding <= 0 and sent == 0):
            break
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
