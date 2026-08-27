"""Is a session fully registered? Answers before you discover it is not.

Adding a session touches six places, and a prose checklist gets skipped. Each
omission has a characteristic silent failure:

  calibration epoch missing  -> a wrong-epoch projection looks like a slightly
                                miscalibrated rig, not like a bug. It shipped
                                that way once: 35-73 px off.
  episodes.jsonl missing     -> world_offset_m has nothing to read, so poses in
                                a redefined frame are used uncorrected
  world residual missing     -> a consumer assumes zero error
  bad_frames / segments      -> dropouts train as if they were data
  splits.json stale          -> the new session is entirely in train, and the
                                probe set can draw start frames the model saw
  not validated              -> nothing above is checked

    python scripts/check_session_ready.py --task motherboard
    python scripts/check_session_ready.py --task motherboard --date 2026-05-19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import release_root   # noqa: E402

RELEASE = release_root()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="motherboard")
    ap.add_argument("--date", default=None, help="check one session; default all")
    a = ap.parse_args()
    root = RELEASE / a.task
    rows, problems = [], []

    def note(ok, item, detail):
        rows.append((ok, item, detail))
        if not ok:
            problems.append(item)

    # 1 — a calibration epoch is declared for the task, and it is the right one
    try:
        from twm.calib_epoch import calib_dir, check_epoch, epoch_of
        d = calib_dir(a.task)
        try:
            check_epoch(a.task)
            note(True, "calibration epoch declared", f"{epoch_of(a.task)} at {d.name}")
        except ValueError as e:
            note(False, "calibration epoch declared", str(e)[:90])
    except Exception as e:
        note(False, "calibration epoch declared", f"{type(e).__name__}: {e}")

    # 2 — episodes.jsonl covers every episode on disk
    ep_file = root / "episodes.jsonl"
    on_disk = sorted({f"{p.parent.name}/{p.stem}"
                      for p in (root / "meta").glob("*/*.parquet")})
    if a.date:
        on_disk = [e for e in on_disk if e.startswith(a.date)]
    listed = set()
    if ep_file.is_file():
        listed = {json.loads(l)["episode"]
                  for l in ep_file.read_text().splitlines() if l.strip()}
    miss = [e for e in on_disk if e not in listed]
    note(not miss and bool(on_disk), "episodes.jsonl covers every episode",
         f"{len(on_disk)} on disk, {len(miss)} unlisted"
         + (f": {miss[:3]}" if miss else ""))

    # 3 — each episode declares a world offset (even if zero)
    try:
        from twm.calib_epoch import world_offset_m
        bad = []
        for e in on_disk:
            dt, ep = e.split("/")
            try:
                world_offset_m(a.task, dt, ep)
            except Exception as ex:
                bad.append(f"{e}: {type(ex).__name__}")
        note(not bad, "every episode declares a world offset",
             f"{len(on_disk)} episodes readable" + (f"; {bad[:2]}" if bad else ""))
    except Exception as e:
        note(False, "every episode declares a world offset", str(e)[:80])

    # 4 — the world residual is declared per date, so a consumer can bound error
    from twm.calib_epoch import world_residual
    dates = sorted({e.split("/")[0] for e in on_disk})
    nores = [d_ for d_ in dates if not world_residual(a.task, d_)]
    note(not nores, "every session publishes a world residual",
         f"{len(dates)} sessions: " + ", ".join(
             f"{d_}(tilt {world_residual(a.task, d_).get('tilt_deg')})" for d_ in dates)
         + (f"; missing {nores}" if nores else ""))

    # 5 — curation artefacts exist and cover the episodes
    for name in ("bad_frames.json", "segments.json"):
        f = root / name
        if not f.is_file():
            note(False, f"{name} present", "missing"); continue
        j = json.loads(f.read_text())
        cov = set(j["episodes"]) if name == "bad_frames.json" else \
            {s["source_episode"] for s in j["segments"]}
        m = [e for e in on_disk if e not in cov]
        note(not m, f"{name} covers every episode",
             f"{len(cov)} covered, {len(m)} uncovered" + (f": {m[:3]}" if m else ""))

    # 6 — splits.json exists and is current
    sf = root / "splits.json"
    if not sf.is_file():
        note(False, "splits.json present", "missing — run scripts/build_splits.py")
    else:
        sp = json.loads(sf.read_text())
        m = [e for e in on_disk if e not in sp["episodes"]]
        st = sp["stats"]
        note(not m, "splits.json covers every episode",
             f"test {st['test_fraction']*100:.1f}% guard {st['guard_fraction']*100:.1f}%, "
             f"{len(m)} uncovered" + (f": {m[:3]}" if m else "")
             + f"; rebuild with scripts/build_splits.py" if m else
             f"test {st['test_fraction']*100:.1f}% guard {st['guard_fraction']*100:.1f}%, "
             f"{st['n_test_intervals']} intervals")

    w = max(len(x) for _, x, _ in rows)
    print()
    for ok, item, detail in rows:
        print(f"  [{'ok' if ok else 'TODO'}] {item:<{w}}  {detail}")
    print(f"\nsession readiness: {len(rows)} items, {len(problems)} incomplete")
    if problems:
        print("\nnext: see docs/USAGE.md section 8 — 'Adding a session or a task'")
    else:
        print("run scripts/validate_all.py, then scripts/test_frame_consistency.py")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
