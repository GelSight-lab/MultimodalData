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
from twm.pipeline_stages import TASKS  # one list; copies are how a task falls out

CUT_ROOT = Path("/media/yxma/Disk1/twm/release_cut")
REPO = "yxma/React"
TASKS = TASKS


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


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def changed(local: dict[str, str], remote: dict[str, str]) -> list[str]:
    """Keys whose local bytes differ from what the Hub holds.

    `--resend` asked "is it already there", which re-sent all 61 previews
    every round -- 533 uploads -- and, once the renderer was fixed, could not
    tell an old preview from a new one anyway. Comparing the content answers
    both: a re-rendered file differs, an untouched one does not.

    The Hub stores these as LFS, whose oid IS the sha256 of the content, so no
    download is needed to compare.
    """
    return sorted(k for k, sha in local.items() if remote.get(k) != sha)


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


def remote_previews(task: str, api) -> dict[str, str]:
    """`<date>/<episode>` -> sha256 of what the Hub holds."""
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    out = {}
    for sib in info.siblings:
        p = sib.rfilename.split("/")
        if (len(p) == 5 and p[0] == "data" and p[1] == task
                and p[2] == "previews" and p[4].endswith(".mp4")):
            sha = sib.lfs.sha256 if sib.lfs else None
            out[f"{p[3]}/{p[4][:-len('.mp4')]}"] = sha
    return out


def local_finished(task: str) -> set[str]:
    root = CUT_ROOT / task / "previews"
    return {f"{p.parent.name}/{p.stem}" for p in sorted(root.glob("*/*.mp4"))
            if is_finished(p)} if root.is_dir() else set()


def round_once(api, dry_run: bool, resend: bool = False) -> tuple[int, int]:
    """One pass over every task. Returns (uploaded, still outstanding)."""
    sent = outstanding = 0
    for task in TASKS:
        pub = publishes(task, api)
        up = remote_previews(task, api)
        ready = local_finished(task) & pub
        local_sha = {k: sha256(CUT_ROOT / task / "previews" / f"{k}.mp4")
                     for k in ready}
        keys = changed(local_sha, up)
        outstanding += len(pub) - len(ready)
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
    ap.add_argument("--resend", action="store_true",
                    help="re-upload previews already on the Hub — for when the "
                         "renderer itself was wrong, not just incomplete")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    total = 0
    while True:
        sent, outstanding = round_once(api, a.dry_run, a.resend)
        total += sent
        print(f"  [{time.strftime('%H:%M:%S')}] 本轮 {sent}，累计 {total}，"
              f"尚未渲染 {outstanding}", flush=True)
        if not a.watch or (outstanding <= 0 and sent == 0):
            break
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
