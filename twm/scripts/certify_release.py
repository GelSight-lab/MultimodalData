"""Certify the published release: modalities aligned, training view clean.

Run before publishing anything. Exits non-zero on the first failure of
either half.

Why a certifier and not the existing checks
-------------------------------------------
Both halves already had a check, and both were weaker than they looked.

* `react_preprocess verify-flags` defaults to ``--limit-episodes 2
  --frames 600 --side left``: 1,200 of ~200,000 rows, one sensor, two
  episodes. It also scored an episode OK on ``mismatches == 0`` *after
  searching for the best shift* — so an episode with the WRONG shift baked
  in passes, because the search simply finds whichever constant offset
  exists. A self-fulfilling check reports the data it is given.

* Curation wrote intervals and segments, and nothing ever re-read the
  parquet to confirm that what a training run loads is actually free of the
  thing the intervals claim to have removed.

So: full coverage, both sensors, expected shift asserted (not discovered),
and the clean segments re-measured from the published parquet rather than
trusted.

    python scripts/certify_release.py                 # both tasks
    python scripts/certify_release.py --task pushT
    python scripts/certify_release.py --align-frames 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twm.react_preprocess import backfill                          # noqa: E402
from twm.react_preprocess.curation import BAD_KEYS                 # noqa: E402
from twm.react_preprocess.detect import (EPS_POSE_BIT,             # noqa: E402
                                         FREEZE_THRESHOLD_S, TAU_INTENSITY)
from twm.tactile_align import LEGACY_SHIFT, RIG_FIXED_DATE         # noqa: E402

RELEASE = Path("/media/yxma/Disk1/twm/release")
SINCE: str | None = None        # set by --since; sessions before it are out of scope
H5_ROOT = Path("/media/yxma/Disk1/twm/data")
FPS = 30


# ── half 1: multimodal alignment ────────────────────────────────────────────
def expected_shift(date: str) -> int:
    """Frames the release must have shifted tactile by, for this date.

    From the dataset's own declaration (tasks.json -> tactile_latency): every
    recording up to and including 2026-06-18 carries the V4L2 buffer lag; the
    rig was fixed 2026-06-27.
    """
    return LEGACY_SHIFT if date <= "2026-06-18" else 0


def published_episodes(task: str) -> set[str]:
    """`<date>/<episode>` keys already on the Hub for this task.

    An episode listed here has been published before; if its parquet is
    unchanged it asserts nothing new, so a missing source is a gap in the
    EVIDENCE, not a defect in the data.
    """
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files("yxma/React", repo_type="dataset")
    except Exception:                                        # noqa: BLE001
        return set()          # offline: treat nothing as published (strict)
    out = set()
    for f in files:
        parts = f.split("/")
        # data/<task>/meta/<date>/<episode>.parquet — five parts.
        if len(parts) == 5 and parts[0] == "data" and parts[1] == task \
                and parts[2] == "meta" and parts[-1].endswith(".parquet"):
            out.add(f"{parts[3]}/{parts[-1][:-len('.parquet')]}")
    return out


def source_recording(parquet: Path) -> str:
    """Which recording this parquet's rows came from.

    READ, not derived from the filename. The cut renames its output —
    `episode_000` becomes `episode_000_seg00`, `_seg01`, … — and deriving the
    H5 name from the stem sent this looking for `episode_000_seg00.h5`, which
    has never existed. It reported the source as missing while
    `episode_000.h5` sat on the disk at 14.9 GB, and 5 segments failed the
    gate for it. The cut writes `source_episode` into every row so this does
    not have to guess.

    A parquet without the column is an uncut episode, which IS its own
    recording.
    """
    try:
        t = pq.read_table(str(parquet), columns=["source_episode"])
        if t.num_rows:
            return str(t["source_episode"][0].as_py()).rsplit("/", 1)[-1]
    except (KeyError, OSError, ValueError):
        pass
    return parquet.stem


def certify_alignment(task: str, frames: int) -> list[str]:
    """Every episode, both sensors: tactile rows sit where they should.

    The published `tactile_{side}_is_new` flags are compared against the
    bit-exact "differs from predecessor" truth computed from the SOURCE H5
    pixels — the only reference that is not itself lossy. The shift is
    ASSERTED, not searched: a constant misalignment is invisible to a search.
    """
    errs, warns = [], []
    already = published_episodes(task)
    for p in sorted((RELEASE / task / "meta").rglob("episode_*.parquet")):
        date, ep = p.parent.name, p.stem
        if SINCE and date < SINCE:
            continue
        h5 = H5_ROOT / task / date / f"{source_recording(p)}.h5"
        if not h5.exists():
            # 1.19 TB of raw HDF5 was deleted 2026-09-09; for those sessions
            # this condition is permanent. Failing on it means the gate can
            # never pass again for the task, which is how a gate turns into a
            # --skip-gate habit. An episode already published asserts nothing
            # new, so this is a gap in the evidence, not a defect — said, not
            # swallowed. Anything NEW still fails.
            line = (f"{task}/{date}/{ep}: source H5 missing — alignment "
                    f"cannot be certified, only assumed")
            (warns if f"{date}/{ep}" in already else errs).append(line)
            continue
        want = expected_shift(date)
        for side in ("left", "right"):
            try:
                r = backfill.verify_against_h5(p, h5, side, frames, shift=want)
            except Exception as e:                    # noqa: BLE001
                errs.append(f"{task}/{date}/{ep} {side}: {type(e).__name__}: {e}")
                continue
            if r["mismatches"]:
                errs.append(
                    f"{task}/{date}/{ep} {side}: {r['mismatches']}/"
                    f"{r['compared']} tactile rows disagree with the source at "
                    f"the declared shift {want:+d} (rig fixed {RIG_FIXED_DATE})")
    return errs, warns


# ── half 2: curation ────────────────────────────────────────────────────────
def _overlap(a1, b1, a2, b2) -> bool:
    return a1 <= b2 and a2 <= b1


def certify_curation(task: str) -> list[str]:
    root = RELEASE / task
    bf = json.loads((root / "bad_frames.json").read_text())["episodes"]
    seg = json.loads((root / "segments.json").read_text())["segments"]
    rows = {r["episode"]: r for r in
            (json.loads(l) for l in
             (root / "episodes.jsonl").read_text().splitlines() if l.strip())}
    by_ep: dict[str, list] = {}
    for s in seg:
        by_ep.setdefault(s["source_episode"], []).append(s)

    errs = []
    # (a) bookkeeping: no clean segment may touch any flagged interval
    for key, rep in bf.items():
        flagged = [(a, b, k) for k in BAD_KEYS for a, b in rep.get(k, [])]
        for s in by_ep.get(key, []):
            sa, sb = s["frame_range"]
            for a, b, k in flagged:
                if _overlap(sa, sb, a, b):
                    errs.append(f"{task}/{key}: clean segment [{sa},{sb}] "
                                f"overlaps {k} [{a},{b}]")
        row = rows.get(key)
        if row is None:
            errs.append(f"{task}/{key}: curated but absent from episodes.jsonl")
        elif row["n_segments"] != len(by_ep.get(key, [])):
            errs.append(f"{task}/{key}: episodes.jsonl claims "
                        f"{row['n_segments']} segments, segments.json has "
                        f"{len(by_ep.get(key, []))}")
    # The scope applies HERE too. `certify_alignment` skips dates before
    # SINCE; this half did not, so a run pointed at one week still demanded a
    # curation record for every May and June segment in the tree — 55
    # problems, none of them about the data being published.
    published = {f"{p.parent.name}/{p.stem}"
                 for p in (root / "meta").rglob("episode_*.parquet")
                 if not (SINCE and p.parent.name < SINCE)}
    for missing in sorted(published - set(bf)):
        errs.append(f"{task}/{missing}: published but never curated")

    # (b) remeasurement: re-derive the scalar detectors INSIDE each clean span
    min_run = int(round(FREEZE_THRESHOLD_S * FPS))
    for key, segs in sorted(by_ep.items()):
        date, ep = key.split("/")
        t = pq.read_table(root / "meta" / date / f"{ep}.parquet",
                          columns=["sensor_left_pose", "sensor_right_pose",
                                   "tactile_left_intensity",
                                   "tactile_right_intensity"])
        pose = {s: np.array(t[f"sensor_{s}_pose"].to_pylist())
                for s in ("left", "right")}
        inten = {s: t[f"tactile_{s}_intensity"].to_numpy()
                 for s in ("left", "right")}
        active = rows[key].get("active_sensors", ["left", "right"])
        for s in segs:
            sa, sb = s["frame_range"]
            for side in active:
                same = np.all(np.abs(np.diff(pose[side][sa:sb + 1], axis=0))
                              < EPS_POSE_BIT, axis=1)
                run = best = 0
                for v in same:
                    run = run + 1 if v else 0
                    best = max(best, run)
                if best + 1 >= min_run:
                    errs.append(f"{task}/{key} [{sa},{sb}] {side}: frozen "
                                f"action — {best + 1} bit-identical poses "
                                f"survive in a clean segment")
                hot = int((inten[side][sa:sb + 1] > TAU_INTENSITY).sum())
                if hot:
                    errs.append(f"{task}/{key} [{sa},{sb}] {side}: {hot} "
                                f"tactile frames above tau survive curation")
    return errs


def certify_previews(task: str, sample: int) -> list[str]:
    """The preview renderer pairs the right tactile frame with the right force.

    Delegates to `scripts/test_preview_alignment`, which MEASURES it — the
    reference's contact fraction against an independent estimate of the free
    gel, and the cross-correlation lag between the displayed contact signal
    and the displayed force. A static guard cannot do this: the defect it
    replaces imported the lag constant from its single source and applied it
    in the wrong place, so every text-level check passed.
    """
    import io
    import contextlib
    import test_preview_alignment as TPA

    TPA.RESULTS.clear()
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = [argv[0], "--task", task, "--sample", str(sample)]
    try:
        with contextlib.redirect_stdout(buf):
            TPA.main()
    finally:
        sys.argv = argv
    return [f"{name}: {ev}" for ok, name, ev in TPA.RESULTS if not ok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="tree to certify (default: the uncut RELEASE). Must be "
                         "the tree that will be PUBLISHED — certifying one and "
                         "uploading another means the checks answered about "
                         "files nobody ships.")
    ap.add_argument("--since", default=None,
                    help="only certify sessions on or after this date. The "
                         "pre-2026-09 sessions are published, their source H5 "
                         "is deleted, and they are out of scope — certifying "
                         "them can only ever fail.")
    ap.add_argument("--task", choices=("motherboard", "pushT"))
    ap.add_argument("--align-frames", type=int, default=2000,
                    help="rows per episode/side compared against source H5")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--preview-sample", type=int, default=4,
                    help="episodes per task for the derived-artifact check")
    args = ap.parse_args()

    global RELEASE, SINCE
    SINCE = args.since
    if args.src:
        RELEASE = Path(args.src)
        print(f"[certify] tree: {RELEASE}", flush=True)
    tasks = [args.task] if args.task else ["motherboard", "pushT"]
    total = 0
    for task in tasks:
        for name, errs in (
                ("alignment", ([], []) if args.skip_align
                 else certify_alignment(task, args.align_frames)),
                # THE PICTURES, NOT ONLY THE DATA. Both halves above certify
                # the published parquet against its source H5. Nothing
                # certified the artifacts DRAWN from it, and that is where the
                # half-second skew between a tactile tile and the force disc
                # beside it lived, through every publish, until a reader
                # watching the videos reported it.
                ("curation", (certify_curation(task), [])),
                ("preview alignment", (certify_previews(task, args.preview_sample), []))):
            errs, warns = errs if isinstance(errs, tuple) else (errs, [])
            total += len(errs)
            print(f"[{'FAIL' if errs else 'ok'}] {task} {name}"
                  + (f": {len(errs)} problem(s)" if errs else ""))
            for e in errs[:40]:
                print("   ", e)
            if len(errs) > 40:
                print(f"    ... and {len(errs) - 40} more")
            if warns:
                print(f"    ({len(warns)} already-published episode(s) whose "
                      f"source recording is deleted — evidence gap, not a defect)")
    print(f"\ncertify: {len(tasks)} task(s), {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
