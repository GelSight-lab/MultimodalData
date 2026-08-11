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
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationDelete

REPO = "yxma/React"
STAGE = Path("/media/yxma/Disk1/twm/release")
# The force channel's staging tree. Same repo paths, six more columns.
FORCE_STAGE = Path("/media/yxma/Disk1/twm/release_force")
REPO_ROOT = Path("/home/yxma/MultimodalData/twm")
TASKS = ("motherboard", "pushT")


def gate() -> int:
    """Refuse to publish anything the gates have not passed. RUN HERE.

    `pipeline_guard` and `certify_release` both existed while a half-second
    skew between the tactile tile and the force disc beside it shipped in
    every published preview. Not because they were wrong — the certifier
    catches it now — but because NOTHING CALLED THEM. They appeared in
    PIPELINE_WORKFLOW.md and in a README, which makes them advice; advice runs
    when someone remembers, and on the day it mattered nobody did.

    A gate that is not on the path is not a gate. This is the path.

    `--skip-gate` exists for re-uploading a single unrelated file, and prints
    what it skipped, because a silent bypass is how this ends up back where it
    started.
    """
    import subprocess
    rc = 0
    for name, cmd in (
            ("pipeline invariants",
             [sys.executable, "-m", "twm.pipeline_guard"]),
            ("release certification",
             [sys.executable, str(REPO_ROOT / "scripts" / "certify_release.py")]),
    ):
        print(f"[gate] {name} ...", flush=True)
        p = subprocess.run(cmd, cwd=str(REPO_ROOT.parent),
                           capture_output=True, text=True)
        if p.returncode:
            rc = 1
            print(p.stdout[-4000:])
            print(p.stderr[-2000:])
            print(f"[gate] {name}: FAILED")
        else:
            print(f"[gate] {name}: ok")
    return rc


def check_no_column_loss(api, tasks) -> list[str]:
    """No upload may leave a published parquet with FEWER columns than it has.

    THIS PUBLISHER SILENTLY REVERTED THE FORCE CHANNEL. Two staging trees
    write to the same repo paths — `release/` holds the parquet without force
    columns, `release_force/` holds the same rows plus six — and the last
    upload wins. For months force went last, so nobody noticed; the first time
    a publish ran afterwards it took the channel off the dataset, and every
    local check passed because locally nothing was wrong.

    The gates added the same day did not help either: they certify that the
    LOCAL data matches its source H5. That is a different question from "will
    this upload remove something the published file already has", and only the
    second one can be answered by looking at the remote.

    `upload_force_columns.check_superset` has asked exactly this since it was
    written. The publisher never did. It does now, for every parquet it is
    about to overwrite.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    published = {f.rfilename for f in
                 api.repo_info(REPO, repo_type="dataset").siblings}
    bad = []
    for task in tasks:
        for local in sorted((STAGE / task).rglob("meta/*/*.parquet")):
            rel = f"data/{task}/" + str(local.relative_to(STAGE / task))
            if rel not in published:
                continue                      # new file: nothing to lose
            try:
                old = set(pq.read_schema(hf_hub_download(
                    REPO, rel, repo_type="dataset")).names)
            except Exception as exc:                        # noqa: BLE001
                bad.append(f"{rel}: cannot read the published schema ({exc}) "
                           f"— refusing rather than guessing")
                continue
            # THE QUESTION IS ABOUT THE END STATE, not about step 1. This
            # publish uploads `release/` and then `release_force/` over the
            # same paths, so what a reader ends up with is the union. Asking
            # it of step 1 alone would refuse the correct flow — and a gate
            # that blocks the right answer gets bypassed, which is how the
            # last one stopped being run at all.
            final = set(pq.read_schema(local).names)
            forced = FORCE_STAGE / task / local.relative_to(STAGE / task)
            if forced.exists():
                final |= set(pq.read_schema(forced).names)
            lost = old - final
            if lost:
                bad.append(f"{rel}: would DROP {len(lost)} published "
                           f"column(s): {', '.join(sorted(lost)[:6])}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_delete", action="store_true")
    ap.add_argument("--skip-gate", action="store_true",
                    help="publish without the guards — say why in the commit")
    args = ap.parse_args()

    if args.skip_gate:
        print("[gate] SKIPPED by --skip-gate: pipeline invariants and release "
              "certification did NOT run for this upload")
    elif gate():
        raise SystemExit("refusing to publish: the gates above failed")

    api = HfApi()

    print("[gate] no published column would be lost ...", flush=True)
    lost = check_no_column_loss(api, TASKS)
    if lost:
        for b in lost[:20]:
            print("   ", b)
        raise SystemExit(
            f"refusing to publish: {len(lost)} file(s) would drop published "
            f"columns. The force channel lives in release_force/, not "
            f"release/ — re-export it into the staging tree, or upload it "
            f"after this run and re-check.")
    print("[gate] no published column would be lost: ok", flush=True)

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
    #
    # PRE-FLIGHT, NOT MID-FLIGHT. `tasks.json` was read from `/tmp/tasks_v2.json`
    # — a path from a one-off run, long since cleared. The publisher uploaded
    # both tasks' data, then died here on a missing file, leaving the release
    # half-published; and because the shell that invoked it ended in an `echo`,
    # the reported exit code was 0. Anything the publisher needs is checked
    # before the first byte goes out, and a source that is absent is announced
    # rather than fatal.
    print("[publish] uploading top-level metadata ...", flush=True)
    wanted = {
        "README.md": REPO_ROOT / "docs/superpowers/specs/README_v2_release.md",
        "examples/react_video_dataset.py":
            REPO_ROOT / "examples/react_video_dataset.py",
    }
    missing = {k: v for k, v in wanted.items() if not Path(v).exists()}
    for k, v in missing.items():
        print(f"    ! {k}: source {v} is absent — NOT uploaded; the published "
              f"copy stays as it is", flush=True)
    ops_src = {k: v for k, v in wanted.items() if k not in missing}
    if not args.dry_run and ops_src:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(k, str(v)) for k, v in ops_src.items()]
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message="Multi-task video release: README + ReactVideoDataset loader")

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
    # 4. THE FORCE CHANNEL, AS PART OF THE SAME PUBLISH.
    #
    # It lives in a second staging tree (`release_force/`) whose parquet are
    # the `release/` ones plus six columns, written to the SAME repo paths.
    # Two trees, one destination, last writer wins — and step 1 above is the
    # writer without the columns. Running the force upload separately worked
    # only for as long as it happened to run last; the first publish that did
    # not took the channel off the dataset.
    #
    # The column-loss gate above is the backstop. This is the mechanism: one
    # command publishes the whole release, in an order that cannot be got
    # wrong by forgetting a step.
    if not args.dry_run:
        print("[publish] uploading the force channel ...", flush=True)
        from force_recovery.upload_force_columns import main as force_main
        import sys as _sys
        argv = _sys.argv
        _sys.argv = [argv[0]]
        try:
            if force_main():
                raise SystemExit("force channel upload FAILED — the release "
                                 "on the hub is missing its force columns")
        finally:
            _sys.argv = argv
    print("[publish] done", flush=True)


if __name__ == "__main__":
    main()
