"""Upload the probe clip page and the test-set page to the Space.

These two pages were built locally and uploaded by hand, which is the same
gap the READMEs had: a step nobody but the author can repeat, and no record
of which local directory became which URL. Both are on the Space as
`/probes/` and `/testset/`, and both are rebuilt by scripts in this
directory:

    python scripts/build_probe_clips.py      ->  probes/
    python scripts/build_testset_page.py     ->  testset/
    python scripts/publish_probe_pages.py

Refuses to upload a page older than the probe package it claims to show:
re-uploading stale HTML after regenerating the data is exactly how a page
starts describing something that is no longer there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_paths import out_root, testset_root                  # noqa: E402

SPACE_ID = "yxma/react-force-recovery"
PAGES = (("probe_clips", "probes"), ("testset_page", "testset"))


def stale() -> list[str]:
    """Pages that predate the probe package they render."""
    src = testset_root() / "manifest.json"
    if not src.exists():
        return [f"{src} missing — build the probe package first"]
    t = src.stat().st_mtime
    bad = []
    for local, _ in PAGES:
        idx = out_root(local) / "index.html"
        if not idx.exists():
            bad.append(f"{local}/index.html missing — rebuild it")
        elif idx.stat().st_mtime < t:
            bad.append(f"{local}/index.html is older than the probe package "
                       f"it renders — rebuild it before publishing")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bad = stale()
    if bad:
        print("refusing to publish:")
        for b in bad:
            print("  ", b)
        return 1

    from huggingface_hub import HfApi
    api = HfApi()
    for local, remote in PAGES:
        d = out_root(local)
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  {d}  ->  {SPACE_ID}:/{remote}/  ({n} files)", flush=True)
        if a.dry_run:
            continue
        api.upload_folder(folder_path=str(d), repo_id=SPACE_ID,
                          repo_type="space", path_in_repo=remote,
                          commit_message=f"rebuild {remote}/ from the probe "
                                         f"package")
    print("\n" + ("dry run — nothing uploaded" if a.dry_run
                  else f"https://{SPACE_ID.replace('/', '-')}.static.hf.space"
                       f"/probes/index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
