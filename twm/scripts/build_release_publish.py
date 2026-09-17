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
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationDelete

# `pipeline_stages._publish` runs this file BY PATH, so sys.path[0] is
# twm/scripts/ and the repo root is not importable. Without this line the
# import below raises ModuleNotFoundError before argparse is even built, and
# the publish stage dies the moment it runs. The same lesson is already
# written 380 lines down, against the force-channel import; it was never
# applied to the module-level import added later.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The scope this pipeline publishes, declared once, in the scheduler. The gate
# below certifies AT it; a gate that re-typed the date would drift from the
# run it is gating.
from twm.pipeline_stages import SCOPE_SINCE

REPO = "yxma/React"
STAGE = Path("/media/yxma/Disk1/twm/release")
# The force channel's staging tree. Same repo paths, six more columns.
FORCE_STAGE = Path("/media/yxma/Disk1/twm/release_force")
# The toolbox users actually run. Published from the repo, not from a
# staging copy, so a fix cannot be published and forgotten locally.
TOOLBOX_SRC = Path("/home/yxma/MultimodalData/twm/react_toolbox")
REPO_ROOT = Path("/home/yxma/MultimodalData/twm")
# Imported, not restated: two lists that can disagree WILL, and the failure
# is a task built and cut but never uploaded, with nothing saying so.
from twm.pipeline_stages import TASKS  # noqa: E402


def gate(src, since: str = SCOPE_SINCE, task: str | None = None,
         skip_align: bool = False) -> int:
    """Refuse to publish anything the gates have not passed. RUN HERE.

    `src` is REQUIRED and has no default. The gate once called the certifier
    with no arguments at all, so it certified the default (uncut) tree over
    every date back to May while the upload beneath it carried `release_cut`
    and a one-week scope — 43 minutes and 211 GB spent checking something
    other than what shipped. Naming the tree is not optional here.

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
             [sys.executable, str(REPO_ROOT / "scripts" / "certify_release.py"),
              "--src", str(src), "--since", since]
             + (["--task", task] if task else [])
             + (["--skip-align"] if skip_align else [])),
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


def scoped_patterns(task_root, since: str) -> list[str]:
    """Which paths of one task's tree go to the dataset's main revision.

    main carries one week. The cut tree holds every date ever built because
    the older sessions still have to go somewhere — a separate branch — so the
    window is applied HERE, at the upload, and not by deleting anything.

    `calibration/` is not dated: it describes the frame the poses are
    expressed in, and the poses being uploaded need it.
    """
    root = Path(task_root)
    pats = ["calibration/*"]
    for sub in ("meta", "videos"):
        d = root / sub
        if not d.is_dir():
            continue
        for date in sorted(p.name for p in d.iterdir() if p.is_dir()):
            if date >= since:
                pats.append(f"{sub}/{date}/*")
                pats.append(f"{sub}/{date}/**")
    p = root / "previews"
    if p.is_dir():
        pats.append("previews/**")
    return pats


def scoped_index(task_root, since: str, out_dir) -> list[str]:
    """Write the four index files filtered to the window being uploaded.

    The files and the index have to describe the same set. A full index beside
    a windowed file set leaves the Hub claiming episodes it does not hold; a
    windowed index beside a full file set leaves episodes published but
    unlisted, which `ReactVideoDataset._split_filter` reads as TRAIN. Both are
    the same defect from opposite directions, so both halves are derived from
    one predicate.
    """
    root, out = Path(task_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = lambda key: str(key).split("/")[0] >= since
    written = []

    src = root / "episodes.jsonl"
    if src.is_file():
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        (out / "episodes.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows if keep(r["episode"])))
        written.append("episodes.jsonl")

    for name, field in (("bad_frames.json", "episodes"),
                        ("splits.json", "episodes")):
        src = root / name
        if not src.is_file():
            continue
        d = json.loads(src.read_text())
        # Everything that is NOT the episode map is the record of how the
        # episodes were judged — thresholds, the seed, the policy. Filtering
        # the episodes must not drop it.
        d[field] = {k: v for k, v in d.get(field, {}).items() if keep(k)}
        (out / name).write_text(json.dumps(d, indent=1))
        written.append(name)

    src = root / "segments.json"
    if src.is_file():
        d = json.loads(src.read_text())
        segs = [s for s in d.get("segments", []) if keep(s["source_episode"])]
        d["segments"] = segs
        if "n_segments" in d:
            d["n_segments"] = len(segs)
        (out / "segments.json").write_text(json.dumps(d, indent=1))
        written.append("segments.json")
    return written


def check_calibration_epoch(task_root, epoch_root, sessions,
                            since: str | None = None) -> list[str]:
    """The calibration shipped beside a session must be the epoch it declares.

    `CALIB_SESSIONS` is the definition of which extrinsics a recording needs,
    and the module is explicit that date order does not determine it. The
    published tree carried whatever the build staged: on 2026-09-14 that was
    `epoch_2026-05-12`, correctly converted and correctly labelled, shipped
    beside the September sessions. Between epochs |dT| = 53-64 mm, which puts
    the projected sensor 35-73 px off — shaped like a slightly miscalibrated
    rig, not like a bug.

    The epoch files on disk are Y-up and the shipped one is Z-up, so the
    comparison is made after converting whichever side declares itself Y-up
    (or declares nothing, which for those directories is the same thing).
    """
    import numpy as np
    from twm.calibration_frame import YUP_TO_ZUP_4
    task_root, epoch_root = Path(task_root), Path(epoch_root)
    dates = sorted({p.name for p in (task_root / "meta").iterdir() if p.is_dir()}) \
        if (task_root / "meta").is_dir() else []
    bad = []
    for date in dates:
        if since and date < since:
            continue          # not being published; its epoch is not this run's claim
        # ASK THE RESOLVER. `session_epoch` is the definition, and since
        # 2026-09-15 it answers for an undeclared session dated on or after
        # CURRENT_EPOCH. Reading CALIB_SESSIONS directly re-decided a question
        # the resolver had already been taught — and refused two sessions that
        # resolve fine.
        if sessions is not None:
            epoch = sessions.get(date)
        else:
            from twm.calib_epoch import session_epoch
            try:
                epoch = session_epoch(task_root.name, date)
            except KeyError:
                epoch = None
        if epoch is None:
            bad.append(f"{date}: no calibration epoch resolves for it — the "
                       f"dataset would ship data its own toolbox raises on")
            continue
        for cam in ("left", "middle", "right"):
            shipped = task_root / "calibration" / f"T_mocap_to_cam_{cam}.json"
            want = epoch_root / f"epoch_{epoch}" / f"T_mocap_to_cam_{cam}.json"
            if not shipped.is_file() or not want.is_file():
                continue
            a = json.loads(shipped.read_text())
            b = json.loads(want.read_text())
            A = np.asarray(a["T_mocap_to_cam"], float)
            B = np.asarray(b["T_mocap_to_cam"], float)
            if b.get("up_axis") != "z":
                B = B @ np.linalg.inv(YUP_TO_ZUP_4)
            if a.get("up_axis") != "z":
                A = A @ np.linalg.inv(YUP_TO_ZUP_4)
            if not np.allclose(A, B, atol=1e-6):
                bad.append(
                    f"{date} declares epoch {epoch}, but the {cam} calibration "
                    f"shipped differs from it by {np.abs(A - B).max():.3f} — "
                    f"a different epoch puts the projected sensor 35-73 px off")
    return bad


def check_calibration_present(stage, tasks) -> list[str]:
    """Every task being published must ship the calibration its poses use.

    `--no_delete` means an absent directory is not an absent directory on the
    Hub: it is the PREVIOUS one, from whichever tree was published last. The
    only way that is safe is if it is never absent.
    """
    bad = []
    for task in tasks:
        d = Path(stage) / task / "calibration"
        n = len(list(d.glob("*.json"))) if d.is_dir() else 0
        if not n:
            bad.append(f"{task}: no calibration/ in the tree being published "
                       f"({d}) — with --no_delete the Hub would keep the one "
                       f"an earlier publish left, in whatever convention that "
                       f"was")
    return bad


def _force_declared_absent(task_root, key: str) -> bool:
    """Does `episodes.jsonl` say this unit ships without the force channel?"""
    p = Path(task_root) / "episodes.jsonl"
    if not p.is_file():
        return False
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("episode") == key:
            return row.get("force") is False
    return False


def force_overlay_plan(stage, force_stage, tasks, since: str = SCOPE_SINCE):
    """Which force-tree files may be uploaded over the tree being published.

    Step 4 uploads `release_force/` to the same repo paths because the UNCUT
    `release/` tree it was built from carries no force columns. That is an
    overlay only while the published names match: the force tree holds
    `episode_003`, and a cut publish holds `episode_003_seg00`. Uploaded
    wholesale it does not overlay anything — it ADDS the uncut recording next
    to the segments cut from it, and a reader summing that folder counts the
    same frames twice.

    So a file is uploadable only if the tree being published has a file at the
    SAME relative path that is missing the columns. Anything else is either
    already carrying them (the Z-up conversion folds them in) or unreachable.

    Returns (files, missing): what to upload, and the published parquet that
    need the columns and that no overlay can reach — the only place a cut
    segment's missing force channel can be caught.
    """
    import pyarrow.parquet as _pq
    # The MEASURED half answers "does this file carry the force channel".
    # Asking for the full superset treated a --force-only export as still
    # needing an overlay, and sent the gate looking for a force-stage file to
    # merge onto a parquet that was already complete.
    from twm.dataset_layout import FORCE_MEASURED
    stage, force_stage = Path(stage), Path(force_stage)
    files, missing = [], []
    for task in tasks:
        for local in sorted((stage / task / "meta").rglob("episode_*.parquet")):
            date = local.parent.name
            if since and date < since:
                continue          # not this run's business
            if set(FORCE_MEASURED) <= set(_pq.read_schema(local).names):
                continue          # already inline
            forced = force_stage / task / local.relative_to(stage / task)
            if forced.is_file():
                files.append(forced)
            elif _force_declared_absent(stage / task, f"{date}/{local.stem}"):
                # DECLARED absent. The operator paused force estimation on
                # 2026-09-16 pending a new algorithm, and `episodes.jsonl`
                # records `force: false`. An absence written down is a fact a
                # reader can act on; refusing it would block a correct publish.
                continue
            else:
                missing.append(
                    f"{task}/{date}/{local.stem}: no force columns, no "
                    f"force-tree file at the same path, and episodes.jsonl "
                    f"does not declare `force: false` — an undeclared absence "
                    f"is the residue of a run that half-finished")
    return files, missing


def check_no_column_loss(api, tasks, published_only: bool = False,
                         allow_dropping=frozenset()) -> list[str]:
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

    # ONE function decides what counts as loss, shared with
    # react_preprocess.publish. There were two implementations of this gate and
    # only THIS one is on the real publish path, so a withdrawal declared to
    # the other would have been approved nowhere that mattered.
    from twm.react_preprocess.publish import lost_columns

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
                    REPO, rel, repo_type="dataset",
                    force_download=published_only)).names)
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
            if published_only:
                # AFTER the run: `old` was just re-downloaded, so it IS the
                # published state. Compare it against what the two staging
                # trees together say the file should have.
                want = set(pq.read_schema(local).names)
                forced = FORCE_STAGE / task / local.relative_to(STAGE / task)
                if forced.exists():
                    want |= set(pq.read_schema(forced).names)
                # Reversed on purpose: afterwards the question is what the
                # published file is still MISSING, so a withdrawn column must
                # not be reported as missing either.
                lost = lost_columns(want, old, allow_dropping=allow_dropping)
            else:
                final = set(pq.read_schema(local).names)
                forced = FORCE_STAGE / task / local.relative_to(STAGE / task)
                if forced.exists():
                    final |= set(pq.read_schema(forced).names)
                lost = lost_columns(old, final, allow_dropping=allow_dropping)
            if lost:
                bad.append(f"{rel}: would DROP {len(lost)} published "
                           f"column(s): {', '.join(sorted(lost)[:6])}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="tree to publish (default: the uncut STAGE). The real "
                         "chain passes the CUT tree: the Hub holds segments, "
                         "and uploading the uncut tree beside them puts "
                         "episode_001 and episode_001_seg00 in one folder with "
                         "the same frames counted twice.")
    ap.add_argument("--since", default=SCOPE_SINCE,
                    help="certify only recordings on or after this date "
                         "(default: the scheduler's SCOPE_SINCE)")
    ap.add_argument("--task", choices=TASKS, default=None,
                    help="publish only this task (default: all of them). The "
                         "certification is scoped to it too, so a task that "
                         "has finished does not wait on one that has not.")
    ap.add_argument("--withdraw-column", action="append", default=[],
                    metavar="NAME",
                    help="a published column this run deliberately stops "
                         "shipping. Repeatable. Naming it IS the audit trail: "
                         "the column-loss gate refuses every drop that is not "
                         "named, because the defect it exists for (a publisher "
                         "silently reverting the force channel) also looked "
                         "like simply writing fewer columns.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_delete", action="store_true")
    ap.add_argument("--skip-align", action="store_true",
                    help="run every gate EXCEPT the per-frame alignment "
                         "certification. For re-publishing bytes a completed "
                         "certification already covered — say which run, in "
                         "the commit. Not for new data.")
    ap.add_argument("--skip-gate", action="store_true",
                    help="publish without the guards — say why in the commit")
    args = ap.parse_args()

    # Resolve the tree FIRST: the gate certifies the tree being published,
    # so it cannot run before that tree is known.
    global STAGE
    if args.src:
        STAGE = Path(args.src)
        print(f"[publish] source tree: {STAGE}", flush=True)

    if args.skip_gate:
        print("[gate] SKIPPED by --skip-gate: pipeline invariants and release "
              "certification did NOT run for this upload")
    elif gate(STAGE, args.since, task=args.task, skip_align=args.skip_align):
        raise SystemExit("refusing to publish: the gates above failed")

    api = HfApi()

    withdraw = frozenset(args.withdraw_column)
    if withdraw:
        print(f"[gate] declared withdrawal of {len(withdraw)} published "
              f"column(s): {', '.join(sorted(withdraw))}", flush=True)

    print("[gate] no published column would be lost ...", flush=True)
    tasks = (args.task,) if args.task else TASKS
    lost = check_no_column_loss(api, tasks, allow_dropping=withdraw)
    if lost:
        for b in lost[:20]:
            print("   ", b)
        raise SystemExit(
            f"refusing to publish: {len(lost)} file(s) would drop published "
            f"columns. The force channel lives in release_force/, not "
            f"release/ — re-export it into the staging tree, or upload it "
            f"after this run and re-check.")
    print("[gate] no published column would be lost: ok", flush=True)

    print("[gate] the calibration travels with the poses ...", flush=True)
    bad_cal = check_calibration_present(STAGE, tasks)
    if bad_cal:
        for b in bad_cal:
            print("   ", b)
        raise SystemExit(
            "refusing to publish: the tree has no calibration to publish "
            "beside its poses")
    print("[gate] the calibration travels with the poses: ok", flush=True)

    print("[gate] the calibration is the epoch the sessions declare ...", flush=True)
    sys.path.insert(0, str(REPO_ROOT))
    from react_toolbox.calib_epoch import CALIB_SESSIONS
    wrong = []
    for task in tasks:
        # None, not a pre-filtered dict: the gate resolves each date through
        # `session_epoch`, which knows the current-epoch default.
        wrong += [f"{task}/{m}" for m in check_calibration_epoch(
            STAGE / task, REPO_ROOT / "calibration", None, since=args.since)]
    if wrong:
        for w in wrong[:20]:
            print("   ", w)
        raise SystemExit(
            "refusing to publish: the calibration shipped is not the epoch the "
            "published sessions declare")
    print("[gate] the calibration is the epoch the sessions declare: ok", flush=True)

    # 1. Upload each task's data/ (exclude _detect.pt sidecars)
    import tempfile
    for task in tasks:
        src = STAGE / task
        pats = scoped_patterns(src, args.since)
        print(f"[publish] uploading data/{task}/ (on or after {args.since}; "
              f"{len([p for p in pats if p.startswith('meta/')]) // 2} date(s)) ...",
              flush=True)
        if not args.dry_run:
            api.upload_folder(
                repo_id=REPO, repo_type="dataset",
                folder_path=str(src),
                path_in_repo=f"data/{task}",
                allow_patterns=pats,
                ignore_patterns=["*._detect.pt", "*._camscan.json"],
                commit_message=f"Publish {task} video release (MP4+parquet, 640x480) + curation + previews",
            )
            # The indices, filtered to the same window. Uploading the tree's
            # full index beside a windowed file set would have the Hub claim
            # episodes it does not hold.
            with tempfile.TemporaryDirectory() as td:
                names = scoped_index(src, args.since, td)
                api.upload_folder(
                    repo_id=REPO, repo_type="dataset",
                    folder_path=td, path_in_repo=f"data/{task}",
                    allow_patterns=names,
                    commit_message=f"Publish {task} indices for {args.since}+",
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

    # 2b. THE TOOLBOX. It was published once and then never again by this
    # script, so a fix to it could not reach users through the normal path.
    # `toolbox/calibration.py` shipped for months returning a gel centre of
    # [0, 0, 0] — projecting the rigid-body origin, 21-36 px off — and the fix
    # sat in git with no way out. A release pipeline that cannot publish its
    # own toolbox will publish a stale one indefinitely.
    print("[publish] uploading toolbox/ ...", flush=True)
    if not TOOLBOX_SRC.is_dir():
        print(f"    ! toolbox source {TOOLBOX_SRC} is absent — NOT uploaded",
              flush=True)
    elif not args.dry_run:
        api.upload_folder(repo_id=REPO, repo_type="dataset",
                          folder_path=str(TOOLBOX_SRC),
                          path_in_repo="toolbox",
                          ignore_patterns=["__pycache__/*", "*.pyc"],
                          commit_message="toolbox: projection rendering + "
                                         "world-frame check; gel centre fix")

    # 2c. THE SCRIPTS AND THE USAGE DOC. Same story as the toolbox above, one
    # directory over: uploaded once by hand and never again by this script, so
    # `USAGE.md` on the Hub went on saying "Up is +y" after the release was
    # rotated to Z-up, and the published test scripts drifted five checks
    # behind the repo. Nothing looked wrong -- the files were all there.
    #
    # Only files ALREADY published are refreshed. Deciding here which scripts
    # belong in the release would put that list in two places; this keeps the
    # published set as it is and only stops it going stale.
    print("[publish] refreshing published scripts + USAGE.md ...", flush=True)
    published = set(api.list_repo_files(REPO, repo_type="dataset"))
    pairs = [(f, REPO_ROOT / f) for f in sorted(published)
             if f.startswith("scripts/")]
    pairs += [(f, REPO_ROOT / src) for f, src in
              (("USAGE.md", "docs/USAGE.md"),) if f in published]
    gone = [f for f, src in pairs if not src.exists()]
    for f in gone:
        print(f"    ! {f} is published but has no source in this repo — "
              f"left as it is", flush=True)
    live = [(f, src) for f, src in pairs if src.exists()]
    if not args.dry_run and live:
        from huggingface_hub import CommitOperationAdd as _Add
        api.create_commit(
            repo_id=REPO, repo_type="dataset",
            operations=[_Add(f, str(src)) for f, src in live],
            commit_message="refresh the published scripts and USAGE.md from "
                           "the repo")
    print(f"[publish] {len(live)} script/doc files refreshed", flush=True)

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
    overlay, unreachable = force_overlay_plan(STAGE, FORCE_STAGE, tasks,
                                              args.since)
    if unreachable:
        for m in unreachable[:20]:
            print("   ", m)
        raise SystemExit(
            f"refusing to publish: {len(unreachable)} published parquet have "
            f"no force columns and no overlay can reach them. Re-run the Z-up "
            f"conversion (it folds the columns in) and re-cut those episodes.")
    if not overlay:
        print("[publish] force channel: already inline in every published "
              "parquet in scope — the overlay has nothing to add", flush=True)
    elif not args.dry_run:
        print("[publish] uploading the force channel ...", flush=True)
        # `force_recovery` lives beside this script's parent, not on the path
        # a bare `python scripts/...` sets up. The first version imported it
        # inside the function and died with ModuleNotFoundError AFTER step 1
        # had already replaced the parquet — taking the force channel off the
        # dataset for the second time in one session.
        sys.path.insert(0, str(REPO_ROOT.parent))
        from twm.force_recovery.upload_force_columns import main as force_main
        argv = sys.argv
        sys.argv = [argv[0]]
        try:
            rc = force_main()
        finally:
            sys.argv = argv
        if rc:
            raise SystemExit("force channel upload FAILED — the release on "
                             "the hub is missing its force columns")

        # VERIFY THE END STATE FROM THE REMOTE. The pre-flight check asks
        # whether the FINAL state loses a column, and computes that final
        # state as the union of the two staging trees — i.e. it trusts step 4
        # to run. Step 4 then crashed on an import, the check having already
        # said "ok", and the columns went. A gate that credits a step which
        # has not happened yet is not a gate. This one reads what is actually
        # published, after everything has been uploaded.
        # THE TOOLBOX, READ BACK. Same reasoning as the columns: the only
        # trustworthy statement about a published artifact is one fetched from
        # where it is published. `toolbox/calibration.py` was wrong on the hub
        # for months while the repo copy was fine.
        print("[publish] verifying the published toolbox ...", flush=True)
        import hashlib
        from huggingface_hub import hf_hub_download
        drift = []
        for lp in sorted(TOOLBOX_SRC.rglob("*.py")):
            rel = "toolbox/" + str(lp.relative_to(TOOLBOX_SRC))
            if "__pycache__" in rel:
                continue
            try:
                rp = hf_hub_download(REPO, rel, repo_type="dataset",
                                     force_download=True)
            except Exception as exc:                        # noqa: BLE001
                drift.append(f"{rel}: not on the hub ({type(exc).__name__})")
                continue
            if (hashlib.sha256(Path(rp).read_bytes()).digest()
                    != hashlib.sha256(lp.read_bytes()).digest()):
                drift.append(f"{rel}: published copy differs from the repo")
        if drift:
            for d in drift[:12]:
                print("   ", d)
            raise SystemExit(
                f"PUBLISHED TOOLBOX IS WRONG: {len(drift)} file(s) differ "
                f"from {TOOLBOX_SRC}.")
        print("[publish] published toolbox verified: ok", flush=True)

        print("[publish] verifying the published columns ...", flush=True)
        still = check_no_column_loss(api, TASKS, published_only=True,
                                     allow_dropping=withdraw)
        if still:
            for b in still[:20]:
                print("   ", b)
            raise SystemExit(
                f"PUBLISHED STATE IS WRONG: {len(still)} file(s) on the hub "
                f"lost columns during this run. Re-run "
                f"`python -m force_recovery.upload_force_columns`.")
        print("[publish] published columns verified: ok", flush=True)
    print("[publish] done", flush=True)


if __name__ == "__main__":
    main()
