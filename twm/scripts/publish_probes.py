"""Publish everything the probe pipeline produces.

These two pages were built locally and uploaded by hand, which is the same
gap the READMEs had: a step nobody but the author can repeat, and no record
of which local directory became which URL. Both are on the Space as
`/probes/` and `/testset/`, and both are rebuilt by scripts in this
directory:

    python scripts/build_probe_testset.py     ->  dataset test_sets/probes_v1/
    python scripts/build_probe_clips.py       ->  space  probes/
    python scripts/build_testset_page.py      ->  space  testset/
    python scripts/build_sim.py               ->  space  sim/
    python scripts/publish_probes.py

Deletes remote files under `test_sets/probes_v1/` that the local package no
longer contains. A re-sample changes which runs exist, and an upload that only
adds leaves the old ones sitting there looking current -- the orphan-preview
failure, again.

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
DATASET = "yxma/React"
TESTSET_PATH = "test_sets/probes_v1"
PAGES = (("probe_clips", "probes"), ("testset_page", "testset"),
         ("sim", "sim"))


def stale() -> list[str]:
    """Pages that predate the probe package they render."""
    src = testset_root() / "manifest.json"
    if not src.exists():
        return [f"{src} missing — build the probe package first"]
    t = src.stat().st_mtime
    bad = []
    for local, _ in PAGES:
        if local == "sim":
            # built from the release, not from the probe package, so the
            # probe manifest's mtime says nothing about it. Only require that
            # it exists -- claiming to check its freshness against an
            # unrelated file would be worse than not checking.
            if not (out_root(local) / "index.html").exists():
                bad.append(f"{local}/index.html missing — run build_sim.py")
            continue
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

    # 1. the package itself, to the dataset
    src = testset_root()
    have = {str(q.relative_to(src)) for q in src.rglob("*") if q.is_file()}
    remote_files = [f[len(TESTSET_PATH) + 1:] for f in
                    api.list_repo_files(DATASET, repo_type="dataset")
                    if f.startswith(TESTSET_PATH + "/")]
    orphans = sorted(set(remote_files) - have)
    print(f"  {src}  ->  {DATASET}:/{TESTSET_PATH}/  ({len(have)} files, "
          f"{len(orphans)} orphaned remotely)", flush=True)
    for o in orphans[:8]:
        print(f"      orphan: {o}")
    if not a.dry_run:
        api.upload_folder(folder_path=str(src), repo_id=DATASET,
                          repo_type="dataset", path_in_repo=TESTSET_PATH,
                          commit_message="rebuild the probe test set")
        if orphans:
            from huggingface_hub import CommitOperationDelete
            api.create_commit(
                repo_id=DATASET, repo_type="dataset",
                operations=[CommitOperationDelete(
                    path_in_repo=f"{TESTSET_PATH}/{o}") for o in orphans],
                commit_message=f"drop {len(orphans)} probe files the rebuild "
                               f"no longer produces")

    # 2. the two pages, to the Space
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
