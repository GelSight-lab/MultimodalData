"""Run every check in the repository and report one verdict.

There were 30 test scripts and no way to ask "is the dataset healthy". A suite
nobody can run in one command is a suite that rots: this session found a
published toolbox missing two modules, a release with no held-out split, and
two pages disagreeing about the same session — none of which any individual
check was wrong about, because none of them was looking.

Each script prints its own evidence and exits non-zero on failure. This runs
them, records duration, and prints the failures last so they are the thing you
see.

    python scripts/validate_all.py [--only substr] [--timeout 900]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Checks that need hardware, a recording session, or many minutes of rendering.
# Named, not silently skipped — an unexplained omission is how coverage rots.
SLOW = {
    "test_crash_leaves_readable_h5.py": "kills a live recorder; needs the rig",
    "test_latency.py": "needs a per-sensor GelSight recording",
    "test_poisson_edge.py": "full-release reconstruction sweep, ~20 min",
    "test_calibfree_ratio.py": "full-release reconstruction sweep, ~15 min",
    "test_calibfree_scale.py": "full-release reconstruction sweep, ~15 min",
    "test_mesh_shading.py": "renders every published mesh under xvfb",
    "test_mesh_uncropped.py": "renders every published mesh under xvfb",
    "test_boundary_rule.py": "full-release reconstruction sweep",
    "test_gel_bound.py": "full-release reconstruction sweep",
    "test_frame_consistency.py": "reads depth from 32 episodes x 3 cameras, ~10 min",
    "test_site.py": "drives the live Space in a browser; needs network",
    "test_published_toolbox.py": "downloads the toolbox from the Hub; needs network",
    "test_doc_references.py": "reads every published doc from the Hub; needs network",
    "test_reproducible.py": "clean-room rebuild from the Hub, ~300 MB download",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--include-slow", action="store_true")
    a = ap.parse_args()

    tests = sorted(HERE.glob("test_*.py"))
    if a.only:
        tests = [t for t in tests if a.only in t.name]
    rows, skipped, unloadable = [], [], []
    for t in tests:
        if t.name in SLOW and not a.include_slow:
            # SKIPPED IS NOT UNCHECKED. Six of these were broken at import —
            # `ModuleNotFoundError: force_recovery`, because they lacked the
            # sys.path bootstrap their siblings have — and the skip list hid it
            # for as long as it existed. Naming a script in a skip list does not
            # run it. Importing the module executes its top-level imports while
            # leaving main() alone, which is exactly the part that was rotting.
            imp = subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.path.insert(0, {str(HERE)!r}); "
                 f"import {t.stem}"],
                capture_output=True, text=True, timeout=120)
            if imp.returncode != 0:
                unloadable.append(
                    (t.name, (imp.stderr.strip().splitlines() or ["?"])[-1][:90]))
            skipped.append((t.name, SLOW[t.name]))
            continue
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, str(t)], capture_output=True,
                               text=True, timeout=a.timeout, cwd=HERE.parent)
            code, out = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            code, out = 124, f"TIMEOUT after {a.timeout}s"
        rows.append((t.name, code, time.time() - t0, out))
        mark = "ok  " if code == 0 else ("TIME" if code == 124 else "FAIL")
        print(f"  [{mark}] {t.name:<44} {time.time()-t0:6.1f}s", flush=True)

    bad = [r for r in rows if r[1] != 0]
    print(f"\n{len(rows)} checks run, {len(rows)-len(bad)} passing, {len(bad)} failing"
          f"{f', {len(skipped)} skipped as slow' if skipped else ''}"
          f"{f' ({len(unloadable)} of them DO NOT IMPORT)' if unloadable else ''}")
    if unloadable:
        print("\nskipped AND broken — these never ran and could not have:")
        for n, e in unloadable:
            print(f"   {n:<44} {e}")
    if skipped:
        print("\nskipped (run with --include-slow):")
        for n, why in skipped:
            print(f"   {n:<44} {why}")
    for name, code, dt, out in bad:
        print(f"\n{'='*70}\nFAIL  {name}  (exit {code})\n{'='*70}")
        tail = [l for l in out.splitlines() if l.strip()][-25:]
        print("\n".join(tail))
    return 1 if (bad or unloadable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
