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


def certify_alignment(task: str, frames: int) -> list[str]:
    """Every episode, both sensors: tactile rows sit where they should.

    The published `tactile_{side}_is_new` flags are compared against the
    bit-exact "differs from predecessor" truth computed from the SOURCE H5
    pixels — the only reference that is not itself lossy. The shift is
    ASSERTED, not searched: a constant misalignment is invisible to a search.
    """
    errs = []
    for p in sorted((RELEASE / task / "meta").rglob("episode_*.parquet")):
        date, ep = p.parent.name, p.stem
        h5 = H5_ROOT / task / date / f"{ep}.h5"
        if not h5.exists():
            errs.append(f"{task}/{date}/{ep}: source H5 missing — alignment "
                        f"cannot be certified, only assumed")
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
    return errs


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
    for missing in sorted({f"{p.parent.name}/{p.stem}"
                           for p in (root / "meta").rglob("episode_*.parquet")}
                          - set(bf)):
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
    ap.add_argument("--task", choices=("motherboard", "pushT"))
    ap.add_argument("--align-frames", type=int, default=2000,
                    help="rows per episode/side compared against source H5")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--preview-sample", type=int, default=4,
                    help="episodes per task for the derived-artifact check")
    args = ap.parse_args()

    tasks = [args.task] if args.task else ["motherboard", "pushT"]
    total = 0
    for task in tasks:
        for name, errs in (
                ("alignment", [] if args.skip_align
                 else certify_alignment(task, args.align_frames)),
                ("curation", certify_curation(task)),
                # THE PICTURES, NOT ONLY THE DATA. Both halves above certify
                # the published parquet against its source H5. Nothing
                # certified the artifacts DRAWN from it, and that is where the
                # half-second skew between a tactile tile and the force disc
                # beside it lived, through every publish, until a reader
                # watching the videos reported it.
                ("preview alignment", certify_previews(task, args.preview_sample))):
            total += len(errs)
            print(f"[{'FAIL' if errs else 'ok'}] {task} {name}"
                  + (f": {len(errs)} problem(s)" if errs else ""))
            for e in errs[:40]:
                print("   ", e)
            if len(errs) > 40:
                print(f"    ... and {len(errs) - 40} more")
    print(f"\ncertify: {len(tasks)} task(s), {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
