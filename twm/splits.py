"""Held-out INTERVALS carved from inside episodes, with a leak-proof guard.

WHY NOT HOLD OUT WHOLE EPISODES

There are 32 motherboard episodes and 194,445 frames. Holding out episodes
spends the scarce resource — episodes, and with them board layouts and lighting
— to buy independence that a short-horizon world model does not need: what it
must generalise over is dynamics within a scene, not scenes.

So the split is at INTERVAL level. Each episode contributes a few held-out
windows from its middle; the rest of that episode trains.

THE PART THAT LEAKS IF YOU GET IT WRONG

A training window starting shortly BEFORE a held-out interval still contains
its frames. Excluding only the interval is not enough. A window of span S
starting at s covers [s, s+S-1], so starts in [a-(S-1), b] must be rejected,
not just [a, b].

`guard_frames` is therefore `max_train_window - 1`, and it is RECORDED. A
loader using a longer window must FAIL rather than silently leak — see
`assert_window_fits`. That failure mode leaves no trace in any metric until
the numbers are suspiciously good.

Cost, measured over this release with a 64-frame training window:
    test 12.2%   guard 9.5%   train 78.4%   over 146 held-out intervals
The guard is what independence costs; naming it makes the price visible.

EPISODES TOO SHORT TO CARVE go entirely to test rather than being dropped,
which also yields a few wholly-unseen episodes for free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "react-splits/1.0"
TEST_INTERVAL_FRAMES = 160      # 128-frame probe horizon + 4 context + slack
MAX_TRAIN_WINDOW = 64           # frames; guard = this - 1
TARGET_TEST_FRACTION = 0.12


def _bad_mask(entry: dict, T: int) -> np.ndarray:
    m = np.zeros(T, bool)
    for k in ("intensity_spikes", "pose_teleports_L", "pose_teleports_R",
              "ot_loss_L", "ot_loss_R"):
        for a, b in entry.get(k, []):
            m[max(0, a):min(T, b + 1)] = True
    return m


def build_splits(episodes, bad=None, seed: int = 0,
                 test_len: int = TEST_INTERVAL_FRAMES,
                 max_train_window: int = MAX_TRAIN_WINDOW,
                 target: float = TARGET_TEST_FRACTION) -> dict:
    """Carve held-out intervals. Deterministic given `episodes` and `seed`.

    `bad` is the `bad_frames.json` episodes dict, so an interval is never
    placed on a stretch of dropouts — a test window full of tracking loss
    measures the rig, not the model.
    """
    guard = int(max_train_window) - 1
    rng = np.random.default_rng(seed)
    need = 2 * test_len + 2 * guard
    out, n_test, n_guard, n_tot = {}, 0, 0, 0

    for e in sorted(episodes, key=lambda x: x["episode"]):
        key, N = e["episode"], int(e["n_frames"])
        n_tot += N
        if N < need:
            out[key] = {"n_frames": N, "whole": "test",
                        "test": [[0, N - 1]], "guard": []}
            n_test += N
            continue
        k = max(1, int(round(N * target / test_len)))
        k = min(k, max(1, (N - test_len) // (test_len + 2 * guard)))
        bm = _bad_mask(bad.get(key, {}) if bad else {}, N)
        lo, hi = guard, N - test_len - guard
        iv = []
        for c in np.linspace(lo, hi, k + 2)[1:-1]:
            a0 = int(np.clip(round(c + rng.integers(-test_len // 2, test_len // 2 + 1)),
                             lo, hi))
            for shift in (0, *[s for d in range(1, test_len + 1) for s in (d, -d)]):
                s = int(np.clip(a0 + shift, lo, hi))
                b = s + test_len - 1
                if bm[s:b + 1].any():
                    continue
                if any(not (b + guard < p or s - guard > q) for p, q in iv):
                    continue
                iv.append([s, b])
                break
        iv.sort()
        out[key] = {"n_frames": N, "whole": None, "test": iv,
                    "guard": [[max(0, s - guard), min(N - 1, b + guard)]
                              for s, b in iv]}
        n_test += sum(b - a + 1 for a, b in iv)
        n_guard += 2 * guard * len(iv)

    return {
        "format": FORMAT, "seed": int(seed), "policy": "interval",
        "test_interval_frames": int(test_len),
        "max_train_window": int(max_train_window), "guard_frames": guard,
        "guard_note": ("a training window of span S starting at s covers "
                       "[s, s+S-1], so starts in [a-(S-1), b] must be rejected, "
                       "not just [a, b]. guard_frames = max_train_window - 1; a "
                       "loader using a longer window must rebuild the split."),
        "episodes": out,
        "stats": {"n_episodes": len(out), "n_frames": n_tot,
                  "n_test_frames": n_test, "n_guard_frames": n_guard,
                  "test_fraction": round(n_test / n_tot, 4),
                  "guard_fraction": round(n_guard / n_tot, 4),
                  "n_test_intervals": sum(len(v["test"]) for v in out.values()),
                  "n_whole_test_episodes": sum(1 for v in out.values() if v["whole"])},
    }


def load_splits(path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("format") != FORMAT:
        raise ValueError(f"expected {FORMAT}, got {d.get('format')!r}")
    return d


def assert_window_fits(splits: dict, window_span: int) -> None:
    """Refuse a training window the guard cannot cover."""
    if window_span - 1 > splits["guard_frames"]:
        raise ValueError(
            f"training window spans {window_span} frames but the split has "
            f"guard_frames={splits['guard_frames']} "
            f"(max_train_window={splits['max_train_window']}). A window this "
            f"long would overlap held-out intervals. Rebuild the split with "
            f"max_train_window >= {window_span}.")


def forbidden_starts(splits: dict, ep_key: str, window_span: int):
    """[(lo, hi)] inclusive start indices a TRAIN window may not begin at."""
    e = splits["episodes"].get(ep_key)
    if e is None:
        return []
    if e["whole"] == "test":
        return [(0, e["n_frames"] - 1)]
    return [(max(0, a - (window_span - 1)), b) for a, b in e["test"]]


def test_starts(splits: dict, ep_key: str, window_span: int):
    """[(lo, hi)] start indices whose whole window lies inside a test interval."""
    e = splits["episodes"].get(ep_key)
    if e is None:
        return []
    return [(a, b - window_span + 1) for a, b in e["test"]
            if b - a + 1 >= window_span]
