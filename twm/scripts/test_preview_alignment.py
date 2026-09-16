"""The preview panel shows the frame and the number that belong together.

Two defects, both found by a reader watching the published videos rather than
by anything in this repo, and both the same shape: a fact about the data
implemented a second time in the renderer instead of imported.

  1  THE REFERENCE FRAME. `run_episode._reference_rows` builds a pool — the 15
     lowest-intensity fresh rows, at least a second apart — and the force
     channel differences against the median of twelve of them. The preview
     used `sample_idx[0]`: whatever frame the recording happened to start on.
     Measured against an independent estimate of the free gel (the per-pixel
     median over the whole episode), on motherboard/2026-05-10:

         episode          preview ref     pool ref
         episode_001         0.21%          0.11%
         episode_002         0.25%          0.12%
         episode_004        12.98%          0.05%

     as a fraction of pixels already in contact. episode_004 starts with the
     gel pressed, so its entire diff panel was referenced against a press.

  2  THE FORCE ROW. In one panel the tactile tile shows gelsight frame i+15
     (`gel_at`, the documented legacy lag) while the force disc showed the
     force of gelsight frame i, because `row_for_h5_frame` subtracted
     LEGACY_SHIFT a second time. Cross-correlating the displayed contact
     signal against the displayed force put the peak at -16 frames on all
     three episodes, and at 0 with the extra subtraction removed.

     The mapping is: row r of the release parquet corresponds to CAMERA frame
     trim + r. `run_episode` adds LEGACY_SHIFT on top of that only to reach
     the GELSIGHT frame that row was computed from. Adding it changes which
     tactile image you read; it does not change which row a camera frame is.

    python scripts/test_preview_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# parents[2] is the REPO ROOT. `import twm.X` needs the directory that
# CONTAINS the twm package, not the package itself -- inserting parents[1]
# (twm/) leaves every `from twm.calib_epoch import ...` unresolvable, which
# is why eight verifiers under this directory could not run at all.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def sample_episodes(task: str, n: int, seed: int = 0):
    """(date, episode) pairs spread across the task's recording dates.

    NOT a fixed list. The two defects here were found on one date, and a test
    pinned to that date would prove nothing about the other five — the
    reference defect in particular is data-dependent (it fires only where a
    recording starts with the gel already pressed, which is why two of the
    three original episodes looked clean). Spread the sample over dates and
    the class of defect has nowhere uniform to hide.
    """
    import numpy as np

    from force_recovery.run_episode import DATA_ROOT, STAGE_ROOT
    dates = sorted(d.name for d in (STAGE_ROOT / task / "meta").iterdir()
                   if d.is_dir())
    pairs = []
    for d in dates:
        for pqf in sorted((STAGE_ROOT / task / "meta" / d).glob("*.parquet")):
            if (DATA_ROOT / task / d / f"{pqf.stem}.h5").exists():
                pairs.append((d, pqf.stem))
    if not pairs:
        return []
    rng = np.random.default_rng(seed)
    # one per date first, then fill at random — a sample that happened to draw
    # five episodes of one session would repeat the original blind spot
    by_date, rest = {}, []
    for d, e in pairs:
        (by_date.setdefault(d, e), rest.append((d, e)))
    picked = [(d, e) for d, e in by_date.items()][:n]
    pool = [p for p in rest if p not in picked]
    if len(picked) < n and pool:
        idx = rng.permutation(len(pool))[:n - len(picked)]
        picked += [pool[int(i)] for i in idx]
    return picked


def target_verdict(ev_tgt, tgt_bad, force_declared: bool):
    """(ok, evidence) for the virtual-target check.

    The target is `observed_pose + (f / K) * n_hat`. With no force there is
    nothing to compute, and reporting that as a FAILURE blocked the publish of
    a task deliberately built without force while a new estimator is written.

    "No evidence" is not "evidence of wrong" — the same distinction the force
    overlay gate draws between a declared absence and an undeclared one. But
    a task that DECLARES force and still produces no target is a broken
    overlay, and that stays a failure.
    """
    if not ev_tgt:
        if force_declared:
            return False, ("force is declared but no target was computed on "
                           "any sample — the overlay is not drawing it")
        return True, ("n/a: no force channel in this task, so there is no "
                      "virtual target to label")
    return (not any(tgt_bad),
            "displayed target gap vs displayed contact: " + ", ".join(ev_tgt))


def _force_declared(root, task: str) -> bool:
    """Does any published unit of this task declare `force: true`?

    Read from episodes.jsonl rather than from the presence of a column, so a
    task pausing force estimation says so once in the index instead of every
    consumer inferring it.
    """
    import json as _json
    from pathlib import Path as _P
    p = _P(root) / task / "episodes.jsonl"
    if not p.is_file():
        return True                     # unknown: keep the strict reading
    for line in p.read_text().splitlines():
        if line.strip() and _json.loads(line).get("force") is True:
            return True
    return False


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import h5py
    import pyarrow.parquet as pq

    from force_overlay import row_for_h5_frame
    from force_recovery.lut_calibration import crop
    from force_recovery.run_episode import (DATA_ROOT, STAGE_ROOT,
                                            _reference_rows)
    from twm.tactile_align import LEGACY_SHIFT

    # 1 — THE ROW MAPPING, on arithmetic alone. Row r was computed from
    # gelsight frame trim + r + LEGACY_SHIFT, and that gelsight frame is
    # paired with CAMERA frame trim + r. So the camera frame that row r
    # annotates is trim + r, and the inverse must return r.
    trim, n_rows = 7, 500
    bad = [r for r in (0, 1, 137, 499)
           if row_for_h5_frame(trim + r, trim, n_rows) != r]
    check(not bad, "a camera frame maps back to its own row",
          f"{len(bad)} of 4 round-trips wrong"
          + (f"; row {bad[0]} came back as "
             f"{row_for_h5_frame(trim + bad[0], trim, n_rows)}" if bad else ""))

    # 2 / 3 — ON THE REAL EPISODES, sampled across dates.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="motherboard")
    ap.add_argument("--sample", type=int, default=4)
    args, _ = ap.parse_known_args()
    TASK = args.task
    episodes = sample_episodes(TASK, args.sample)
    if not episodes:
        check(False, "the preview references a free gel",
              f"UNVERIFIED: no {TASK} episode with both parquet and H5 here")
        episodes = []

    ref_bad, lag_bad, ev_ref, ev_lag = [], [], [], []
    tgt_bad, ev_tgt = [], []
    for DATE, ep in episodes:
        pqt = STAGE_ROOT / TASK / "meta" / DATE / f"{ep}.parquet"
        h5p = DATA_ROOT / TASK / DATE / f"{ep}.h5"
        t = pq.read_table(str(pqt))
        inten = np.asarray(t["tactile_left_intensity"].to_numpy())
        isnew = np.asarray(t["tactile_left_is_new"].to_numpy())
        trim = int(np.asarray(t["source_h5_frame"].to_numpy())[0])
        rows = _reference_rows(inten, isnew)
        npz = (Path("/media/yxma/Disk1/twm/force_recovery") / TASK / DATE
               / f"{ep}_left.npz")
        if not npz.exists():
            continue
        force = np.load(npz)["force_normal_n"].astype(float)

        with h5py.File(str(h5p), "r") as f:
            fr = f["gelsight/left/frames"]
            n = len(fr)

            def gel(i):
                return crop(fr[min(int(i), n - 1)]).astype(np.float32)

            pool = np.median(np.stack(
                [gel(trim + int(r) + LEGACY_SHIFT) for r in rows[:12]]), 0)
            # An independent estimate of the free gel: the per-pixel median
            # over the whole recording. It is not either candidate, so it can
            # referee between them.
            wide = np.linspace(0, len(inten) - 1, 64).astype(int)
            truth = np.median(np.stack(
                [gel(trim + int(r) + LEGACY_SHIFT) for r in wide]), 0)

            # THE REFERENCE THE RENDERER ACTUALLY USES.
            import build_episode_previews as BEP
            pick = getattr(BEP, "preview_reference", None)
            if pick is None:
                ref = gel(trim + LEGACY_SHIFT)          # today: first frame
                how = "first sampled frame"
            else:
                ref = crop(np.asarray(
                    pick(f, "left", TASK, DATE, ep)).astype(np.float32))
                how = "preview_reference()"
            in_contact = float((np.abs(ref - truth).max(axis=2) > 8.0).mean())
            ok = in_contact <= 0.01
            ref_bad.append(not ok)
            ev_ref.append(f"{DATE[5:]}/{ep[-3:]} {in_contact*100:.2f}%")

            # THE FORCE THE RENDERER ACTUALLY OVERLAYS, against the tactile
            # tile it actually shows.
            # WHICH gel frame the renderer shows for camera frame i is not
            # restated here: `_gel_source_frames` reads it from the force npz,
            # which records the frame each row's force was computed from. The
            # old `i + LEGACY_SHIFT` is the pre-2026-06-27 index rule, and on
            # every timestamped recording it put this correlation at -12
            # frames while the renderer was at 0 — a checker failing the data
            # for not matching a mapping the data never used.
            src_col = BEP._gel_source_frames(TASK, DATE, ep).get("left")

            def shown_gel(i, _c=src_col):
                if _c is None:
                    return int(i) + LEGACY_SHIFT
                r = row_for_h5_frame(int(i), trim, len(_c))
                return int(_c[r]) if r is not None else int(i)

            cam = np.arange(0, min(len(force), n - LEGACY_SHIFT - 1), 4)
            sig = np.array([(np.abs(gel(shown_gel(i)) - pool
                                    ).max(axis=2) > 8.0).mean() for i in cam])
            shown = np.array([
                force[r] if (r := row_for_h5_frame(int(i), trim, len(force)))
                is not None else np.nan for i in cam])
            # AND THE VIRTUAL TARGET, measured the same way. It is drawn from
            # the same row as the disc, but "same row by construction" is an
            # argument; this is a measurement. The quantity is the target's
            # displacement from the observed pose — which is force/k, so it
            # must track the displayed contact signal at lag 0 exactly as the
            # force does.
            from force_overlay import load_targets as _lt
            poses = BEP._release_poses(TASK, DATE, ep)
            tg = _lt(TASK, DATE, ep, Path("/media/yxma/Disk1/twm/force_recovery"),
                     poses).get("left")
            if tg is not None and "left" in poses:
                gap = np.linalg.norm(tg[:, :3] - poses["left"][:len(tg), :3],
                                     axis=1) * 1000.0
                shown_t = np.array([
                    gap[r] if (r := row_for_h5_frame(int(i), trim, len(gap)))
                    is not None else np.nan for i in cam])
            else:
                shown_t = np.full(len(cam), np.nan)

        m = np.isfinite(sig) & np.isfinite(shown)
        a, b = sig[m] - sig[m].mean(), shown[m] - shown[m].mean()
        c = np.correlate(a, b, "full")
        k = np.arange(-len(b) + 1, len(a))
        w = np.abs(k) <= 15
        lag = int(k[w][np.argmax(c[w])]) * 4
        lag_bad.append(abs(lag) > 4)
        ev_lag.append(f"{DATE[5:]}/{ep[-3:]} {lag:+d}f")
        mt = np.isfinite(sig) & np.isfinite(shown_t)
        if mt.sum() > 50 and shown_t[mt].std() > 1e-9:
            at, bt = sig[mt] - sig[mt].mean(), shown_t[mt] - shown_t[mt].mean()
            ct = np.correlate(at, bt, "full")
            kt = np.arange(-len(bt) + 1, len(at))
            wt = np.abs(kt) <= 15
            lt_ = int(kt[wt][np.argmax(ct[wt])]) * 4
            tgt_bad.append(abs(lt_) > 4)
            ev_tgt.append(f"{DATE[5:]}/{ep[-3:]} {lt_:+d}f")

    # STATE THE COVERAGE. The sampler skips dates with no episodes on this
    # disk, which is correct (2026-05-15 was emptied when its orphans were
    # deleted) and invisible — a check that quietly covers 3 of 4 dates reads
    # exactly like one that covered all 4.
    from force_recovery.run_episode import STAGE_ROOT as _SR
    all_dates = sorted(d.name for d in (_SR / TASK / "meta").iterdir()
                       if d.is_dir() and any(d.glob("*.parquet")))
    seen_dates = sorted({d for d, _ in episodes})
    missed = [d for d in all_dates if d not in seen_dates]
    check(not any(ref_bad), "the preview references a free gel",
          f"{len(episodes)} episodes over {len(seen_dates)}/{len(all_dates)} "
          f"populated dates"
          + (f" (not sampled: {', '.join(missed)})" if missed else "")
          + "; pixels already in contact: " + ", ".join(ev_ref) + " (want <=1%)")
    check(not any(lag_bad), "the force disc labels the tile beside it",
          "displayed force vs displayed contact: " + ", ".join(ev_lag))
    _declared = _force_declared(_SR, TASK)
    _ok, _ev = target_verdict(ev_tgt, tgt_bad, _declared)
    check(_ok, "the virtual target labels the tile beside it", _ev)

    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    n_bad = sum(not ok for ok, _, _ in RESULTS)
    print(f"\npreview alignment: {len(RESULTS)} checks, {n_bad} failing")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
