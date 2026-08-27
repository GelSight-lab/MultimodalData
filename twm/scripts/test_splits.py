"""Held-out intervals are actually held out, and the guard is not decorative.

Before this, every motherboard episode was `split: train` — the release had no
held-out data at all, and the probe set's start frames were training frames.

The failure this guards against leaves no trace. A training window starting
shortly before a held-out interval still contains its frames; the metric just
comes out better and nothing says why. So the checks below enumerate ACTUAL
window starts rather than reasoning about the intervals.

    python scripts/test_splits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import release_root   # noqa: E402

import numpy as np                                             # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
REL = release_root("motherboard")


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from twm.splits import (assert_window_fits, build_splits, forbidden_starts,
                            test_starts)

    eps = [json.loads(l) for l in (REL / "episodes.jsonl").read_text().splitlines() if l.strip()]
    bad = json.loads((REL / "bad_frames.json").read_text())["episodes"]
    S = build_splits(eps, bad, seed=0)
    W = S["max_train_window"]

    # 1 — NO TRAIN WINDOW TOUCHES A TEST FRAME. Enumerated, not argued.
    leaks, n_win = [], 0
    for e in eps:
        key, N = e["episode"], e["n_frames"]
        info = S["episodes"][key]
        test = np.zeros(N, bool)
        for a, b in info["test"]:
            test[a:b + 1] = True
        forb = forbidden_starts(S, key, W)
        for s in range(0, N - W + 1):
            if any(lo <= s <= hi for lo, hi in forb):
                continue
            n_win += 1
            if test[s:s + W].any():
                leaks.append(f"{key}: window at {s}")
                if len(leaks) > 3:
                    break
    check(not leaks, "no admissible train window contains a held-out frame",
          f"{n_win} train windows of {W} frames enumerated across "
          f"{len(eps)} episodes, 0 touch a test interval"
          + (f"; leaks {leaks[:3]}" if leaks else ""))

    # 2 — AND THE GUARD IS LOAD-BEARING. Shrinking it to the interval alone
    #     must produce leaks, or the guard was never doing anything.
    naive = []
    for e in eps[:6]:
        key, N = e["episode"], e["n_frames"]
        info = S["episodes"][key]
        if info["whole"]:
            continue
        test = np.zeros(N, bool)
        for a, b in info["test"]:
            test[a:b + 1] = True
        for s in range(0, N - W + 1):
            if any(a <= s <= b for a, b in info["test"]):   # interval only
                continue
            if test[s:s + W].any():
                naive.append(s)
    check(len(naive) > 0, "the guard is load-bearing, not decorative",
          f"excluding only the intervals (no guard) leaks {len(naive)} windows "
          f"in the first 6 episodes; with the guard it is 0")

    # 3 — a too-long window is REFUSED, not silently allowed
    try:
        assert_window_fits(S, S["guard_frames"] + 2)
        raised = False
    except ValueError:
        raised = True
    ok_small = True
    try:
        assert_window_fits(S, W)
    except ValueError:
        ok_small = False
    check(raised and ok_small, "a window longer than the guard is refused",
          f"span {W} accepted, span {S['guard_frames']+2} raises")

    # 4 — deterministic, and it moves when the seed does
    a = build_splits(eps, bad, seed=0)
    b = build_splits(eps, bad, seed=1)
    same = a["episodes"] == S["episodes"]
    diff = sum(1 for k in a["episodes"]
               if a["episodes"][k]["test"] != b["episodes"][k]["test"])
    check(same and diff > len(eps) // 2,
          "the split is reproducible and seed-dependent",
          f"seed 0 reproduces exactly; seed 1 moves {diff}/{len(eps)} episodes")

    # 5 — no test interval sits on known-bad frames
    onbad = []
    for e in eps:
        key, N = e["episode"], e["n_frames"]
        if S["episodes"][key]["whole"]:
            continue
        m = np.zeros(N, bool)
        for k in ("intensity_spikes", "pose_teleports_L", "pose_teleports_R",
                  "ot_loss_L", "ot_loss_R"):
            for x, y in bad.get(key, {}).get(k, []):
                m[max(0, x):min(N, y + 1)] = True
        for x, y in S["episodes"][key]["test"]:
            if m[x:y + 1].any():
                onbad.append(f"{key}[{x},{y}]")
    check(not onbad, "held-out intervals avoid known-bad frames",
          f"{S['stats']['n_test_intervals']} intervals, none on flagged "
          f"dropouts" + (f"; {onbad[:2]}" if onbad else ""))

    # 6 — the numbers the docstring quotes are the numbers it produces
    st = S["stats"]
    check(0.10 <= st["test_fraction"] <= 0.15 and st["n_test_intervals"] > 100,
          "the split holds out a usable fraction",
          f"test {st['test_fraction']*100:.1f}%  guard "
          f"{st['guard_fraction']*100:.1f}%  train "
          f"{(1-st['test_fraction']-st['guard_fraction'])*100:.1f}%  over "
          f"{st['n_test_intervals']} intervals + {st['n_whole_test_episodes']} "
          f"whole episodes")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nsplits: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
