"""Recompute every React episode's force channel, into a STAGING directory.

The published channel is not overwritten. It is compared first — the last two
pipeline changes both produced a plausible-looking force curve that was wrong
in a way only a side-by-side showed (a relative contact floor put the median at
1.585 N where the published channel reads 0.000, and called 67% of frames a
contact against 47%). `process_side` takes an explicit `out_dir` for exactly
this reason.

    python -m force_recovery.reprocess_react run      # -> STAGING
    python -m force_recovery.reprocess_react compare  # staged vs published
    python -m force_recovery.reprocess_react promote  # replace, after compare

`promote` refuses unless every staged episode exists and carries the current
calibration name, so a half-finished run cannot half-replace the release.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from .run_episode import OUT_ROOT

STAGING = OUT_ROOT / "_reprocess_staging"
REPORT = OUT_ROOT / "feature_cache" / "reprocess_react.json"
WORKERS = int(os.environ.get("REPROCESS_WORKERS", "4"))


def episodes() -> list[tuple[str, str, str, str]]:
    """(task, date, episode, side) for every published npz."""
    out = []
    for p in sorted(OUT_ROOT.glob("*/*/*_left.npz")) + \
            sorted(OUT_ROOT.glob("*/*/*_right.npz")):
        if STAGING.name in p.parts:
            continue
        side = "left" if p.name.endswith("_left.npz") else "right"
        ep = p.name[: -len(f"_{side}.npz")]
        out.append((p.parts[-3], p.parts[-2], ep, side))
    return sorted(set(out))


def _one(job):
    task, date, ep, side = job
    from .run_episode import process_side
    dest = STAGING / task / date
    try:
        m = process_side(task, date, ep, side, out_dir=dest)
        return {"job": job, "ok": True, "force_max_n": m["force_max_n"],
                "calibration": m["force_calibration"]}
    except Exception as exc:                                   # noqa: BLE001
        return {"job": job, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def cmd_run() -> int:
    from concurrent.futures import ProcessPoolExecutor
    jobs = episodes()
    print(f"[reprocess] {len(jobs)} sides -> {STAGING} ({WORKERS} workers)",
          flush=True)
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(_one, jobs), 1):
            (done if r["ok"] else failed).append(r)
            if r["ok"]:
                print(f"  {i:3d}/{len(jobs)} {'/'.join(r['job'])} "
                      f"max {r['force_max_n']:.2f} N", flush=True)
            else:
                print(f"  {i:3d}/{len(jobs)} {'/'.join(r['job'])} FAILED "
                      f"{r['error']}", flush=True)
    print(f"\n[reprocess] {len(done)} ok, {len(failed)} failed")
    for f in failed:
        print(f"  FAILED {'/'.join(f['job'])}: {f['error']}")
    return 1 if failed else 0


def cmd_compare() -> int:
    from scipy.stats import spearmanr
    rows = []
    for task, date, ep, side in episodes():
        pub = OUT_ROOT / task / date / f"{ep}_{side}.npz"
        new = STAGING / task / date / f"{ep}_{side}.npz"
        if not new.exists():
            rows.append({"ep": f"{task}/{date}/{ep}/{side}", "staged": False})
            continue
        a = np.load(pub, allow_pickle=True)["force_normal_n"]
        b = np.load(new, allow_pickle=True)["force_normal_n"]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        rows.append({
            "ep": f"{task}/{date}/{ep}/{side}", "staged": True, "n": int(n),
            "rho": float(spearmanr(a, b).statistic),
            "mad": float(np.abs(a - b).mean()),
            "old_max": float(a.max()), "new_max": float(b.max()),
            "old_contact": float((a > 0.1).mean()),
            "new_contact": float((b > 0.1).mean())})
    ok = [r for r in rows if r.get("staged")]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rows, indent=1))
    if not ok:
        print("nothing staged")
        return 1
    rho = np.array([r["rho"] for r in ok])
    mad = np.array([r["mad"] for r in ok])
    oc = np.array([r["old_contact"] for r in ok])
    nc = np.array([r["new_contact"] for r in ok])
    print(f"[compare] {len(ok)}/{len(rows)} staged")
    print(f"  rank agreement with the published channel: "
          f"min {rho.min():.3f}  median {np.median(rho):.3f}  max {rho.max():.3f}")
    print(f"  mean |difference|: median {np.median(mad):.3f} N  "
          f"worst {mad.max():.3f} N")
    print(f"  contact fraction: published {oc.mean()*100:.1f}%  "
          f"staged {nc.mean()*100:.1f}%")
    worst = sorted(ok, key=lambda r: r["rho"])[:5]
    print("  five lowest-agreement episodes:")
    for r in worst:
        print(f"    {r['ep']:44s} rho {r['rho']:.3f}  mad {r['mad']:.3f} N  "
              f"contact {r['old_contact']*100:.0f}% -> {r['new_contact']*100:.0f}%")
    print(f"\n-> {REPORT}")
    return 0


def cmd_promote() -> int:
    """Replace the published npz with the staged ones. Refuses a partial run."""
    import shutil

    from .react_calib import CALIBRATION_NAME
    jobs = episodes()
    missing, wrong = [], []
    for task, date, ep, side in jobs:
        p = STAGING / task / date / f"{ep}_{side}.npz"
        if not p.exists():
            missing.append(f"{task}/{date}/{ep}/{side}")
            continue
        got = str(np.load(p, allow_pickle=True)["force_calibration"])
        if got != CALIBRATION_NAME:
            wrong.append(f"{task}/{date}/{ep}/{side}: {got}")
    if missing or wrong:
        print(f"refusing to promote: {len(missing)} missing, "
              f"{len(wrong)} on a stale calibration")
        for x in (missing + wrong)[:8]:
            print(f"  {x}")
        return 1
    for task, date, ep, side in jobs:
        src = STAGING / task / date / f"{ep}_{side}.npz"
        shutil.copy2(src, OUT_ROOT / task / date / f"{ep}_{side}.npz")
    print(f"promoted {len(jobs)} sides to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    raise SystemExit({"run": cmd_run, "compare": cmd_compare,
                      "promote": cmd_promote}[cmd]())
