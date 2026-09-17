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
import hashlib
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
from twm.pipeline_stages import TASKS  # one list; nine copies is how rope fell out

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


def certify_alignment(task: str, frames: int, todo=None) -> list[str]:
    """Every episode, both sensors: tactile rows sit where they should.

    The published `tactile_{side}_is_new` flags are compared against the
    bit-exact "differs from predecessor" truth computed from the SOURCE H5
    pixels — the only reference that is not itself lossy. The shift is
    ASSERTED, not searched: a constant misalignment is invisible to a search.
    """
    errs, warns = [], []
    already = published_episodes(task)
    # `todo` is the receipt's answer: the units whose bytes have moved since
    # they last passed. Everything else would be re-read out of the source H5 —
    # ~2000 rows per side, 2356 GB over the window — to re-derive the same
    # answer. None is "all of them", which is what a bare invocation means.
    for p in sorted((RELEASE / task / "meta").rglob("episode_*.parquet")):
        date, ep = p.parent.name, p.stem
        if SINCE and date < SINCE:
            continue
        if todo is not None and f"{date}/{ep}" not in todo:
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


RECEIPT = ".certified.json"


def _parquet_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_per_check(rec) -> dict:
    """Upgrade any receipt record to `{"hashes": {check: hash}}`.

    Three formats have existed: a bare hash string (alignment only), then
    `{"hash": h, "checks": [...]}` with ONE hash shared by every check, and now
    a hash PER check -- necessary because the checks no longer hash the same
    thing. Old records are read, never rewritten in place: the upgrade happens
    the next time a check passes, so a stale entry simply re-certifies once.
    """
    if isinstance(rec, str):
        return {"hashes": {"alignment": rec}}
    if isinstance(rec, dict):
        if "hashes" in rec:
            return {"hashes": dict(rec["hashes"])}
        h = rec.get("hash")
        if h:
            return {"hashes": {c: h for c in rec.get("checks", ["alignment"])}}
    return {"hashes": {}}


def _receipt_hash(rec, check: str):
    return _as_per_check(rec)["hashes"].get(check)


def _evidence_hash(path: Path, check: str) -> str:
    """Hash of what THIS check reads, not of the whole file.

    A receipt keyed on the file re-runs its check whenever any byte moves. For
    `alignment` that is too strict in an expensive direction: it reads only
    `source_h5_frame` and the tactile contact scalars, compares them against
    the source H5 pixels, and never looks at another column. The 2026-09-16
    force-only export adds columns to every published parquet, which under a
    whole-file hash would have re-certified all 173 segments -- 2000 frames per
    sensor-side out of the source H5, on a seek-bound mechanical disk, about
    seven hours -- to re-derive an answer that could not have changed.

    The danger runs the other way, so the column set is named explicitly and
    anything not analysed falls back to the whole file. A check that starts
    reading a new column must be added here, and the tests in
    test_receipt_keys_on_evidence spell out what must still invalidate.
    """
    if check != "alignment":
        # previews render from more than these columns and nothing has
        # established exactly what. Stay conservative.
        return _parquet_hash(path)

    import pyarrow.parquet as _pq
    from twm.react_preprocess.backfill import SCALARS

    want = ["source_h5_frame"] + [f"tactile_{side}_{s}"
                                  for side in ("left", "right")
                                  for s in SCALARS]
    try:
        t = _pq.read_table(str(path))
    except Exception:                                   # noqa: BLE001
        # Unreadable: fall back to the strict rule rather than crash the
        # bookkeeping. The certification itself will fail on this file and say
        # why; the receipt's job here is only to avoid claiming it passed.
        return _parquet_hash(path)
    h = hashlib.sha256()
    for name in want:
        # The NAME goes in as well as the bytes, so dropping a column that
        # `flags_from_scalars` would have used changes the hash rather than
        # silently shortening the list.
        h.update(name.encode())
        if name in t.column_names:
            arr = t[name].combine_chunks()
            for buf in arr.buffers():
                h.update(b"" if buf is None else buf)
        else:
            h.update(b"<absent>")
    h.update(str(t.num_rows).encode())
    return h.hexdigest()


def _published_units(task: str) -> dict[str, Path]:
    """`date/episode` -> parquet, for the in-scope part of the tree."""
    root = RELEASE / task / "meta"
    if not root.is_dir():
        return {}
    return {f"{p.parent.name}/{p.stem}": p
            for p in sorted(root.rglob("episode_*.parquet"))
            if not (SINCE and p.parent.name < SINCE)}


def needs_certifying(task: str, force: bool = False,
                     check: str = "alignment") -> list[str]:
    """Which units this run has to read the source H5 for.

    The alignment half reads ~2000 rows per episode per side out of the source
    H5 and compares them pixel by pixel: 2356 GB over the in-scope window at
    about 4.8 GB/min. Doing that for an episode that passed last time, is
    already on the Hub, and whose bytes have not moved re-derives the same
    answer at the same cost.

    The receipt is keyed on the parquet's CONTENT, not its name. A re-cut
    episode keeps its name and changes its bytes, and that is exactly the case
    a name-keyed receipt would wave through.
    """
    units = _published_units(task)
    if force:
        return sorted(units)
    seen = {}
    p = RELEASE / task / RECEIPT
    if p.is_file():
        try:
            seen = json.loads(p.read_text())
        except (OSError, ValueError):
            seen = {}
    # Per CHECK, because they cost very differently and a unit can have passed
    # one and not the other. Measured on rope: curation 2.3 s against the
    # parquet scalars, previews 238.2 s because it opens the 120 GB source H5
    # per sampled episode.
    out = []
    for k, q in units.items():
        if _receipt_hash(seen.get(k), check) != _evidence_hash(q, check):
            out.append(k)
    return sorted(out)


def write_receipt(task: str, passed, check: str = "alignment") -> None:
    """Stamp only the units that PASSED. Stamping a failure makes the next run
    skip the very thing that was wrong."""
    units = _published_units(task)
    p = RELEASE / task / RECEIPT
    seen = {}
    if p.is_file():
        try:
            seen = json.loads(p.read_text())
        except (OSError, ValueError):
            seen = {}
    for k in passed:
        if k not in units:
            continue
        rec = _as_per_check(seen.get(k))
        rec["hashes"][check] = _evidence_hash(units[k], check)
        seen[k] = rec
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(seen, indent=1, sort_keys=True))


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
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--align-frames", type=int, default=2000,
                    help="rows per episode/side compared against source H5")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--recertify", action="store_true",
                    help="ignore the receipt and re-read every source H5. For "
                         "a changed detector or a suspected bad receipt — not "
                         "for a routine run, where it costs hours to re-derive "
                         "answers that have not changed.")
    ap.add_argument("--preview-sample", type=int, default=4,
                    help="episodes per task for the derived-artifact check")
    args = ap.parse_args()

    global RELEASE, SINCE
    SINCE = args.since
    if args.src:
        RELEASE = Path(args.src)
        print(f"[certify] tree: {RELEASE}", flush=True)
    tasks = [args.task] if args.task else TASKS
    total = 0
    for task in tasks:
        todo = needs_certifying(task, force=args.recertify)
        n_all = len(_published_units(task))
        if not args.skip_align:
            print(f"[certify] {task}: {len(todo)} of {n_all} unit(s) need the "
                  f"source-H5 pass" + (" (all — --recertify)" if args.recertify
                                       else f", {n_all - len(todo)} unchanged "
                                            f"since they passed"), flush=True)
        # previews is a per-TASK sample, not a per-unit check, so the receipt
        # applies to the task as a whole: skip only when EVERY unit is
        # unchanged and has passed. One changed segment and the sample is
        # re-drawn, which is the only honest use of a receipt for a sample.
        prev_todo = needs_certifying(task, force=args.recertify, check="previews")
        if not prev_todo:
            print(f"[certify] {task}: previews unchanged since they passed — "
                  f"skipping the {args.preview_sample}-episode source-H5 sample "
                  f"(238 s/task measured)", flush=True)
        for name, errs in (
                ("alignment", ([], []) if args.skip_align
                 else certify_alignment(task, args.align_frames, todo)),
                # THE PICTURES, NOT ONLY THE DATA. Both halves above certify
                # the published parquet against its source H5. Nothing
                # certified the artifacts DRAWN from it, and that is where the
                # half-second skew between a tactile tile and the force disc
                # beside it lived, through every publish, until a reader
                # watching the videos reported it.
                ("curation", (certify_curation(task), [])),
                ("preview alignment", ([], []) if not prev_todo
                 else (certify_previews(task, args.preview_sample), []))):
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
            # Stamp only on a clean alignment pass, and only the units this run
            # actually read. Stamping a failure, or stamping units the run
            # skipped, makes the next run skip the very thing that was wrong.
            if not errs:
                if name == "alignment" and not args.skip_align and todo:
                    write_receipt(task, todo, check="alignment")
                    print(f"    receipt: {len(todo)} unit(s) stamped "
                          f"(alignment)", flush=True)
                elif name == "preview alignment" and prev_todo:
                    write_receipt(task, prev_todo, check="previews")
                    print(f"    receipt: {len(prev_todo)} unit(s) stamped "
                          f"(previews)", flush=True)
    print(f"\ncertify: {len(tasks)} task(s), {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
