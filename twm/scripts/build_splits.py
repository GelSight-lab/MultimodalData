"""Write splits.json into a release tree.

    python scripts/build_splits.py --root /media/yxma/Disk1/twm/release/motherboard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.splits import (MAX_TRAIN_WINDOW, TARGET_TEST_FRACTION,   # noqa: E402
                        TEST_INTERVAL_FRAMES, build_splits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/yxma/Disk1/twm/release/motherboard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-len", type=int, default=TEST_INTERVAL_FRAMES)
    ap.add_argument("--max-train-window", type=int, default=MAX_TRAIN_WINDOW)
    ap.add_argument("--target", type=float, default=TARGET_TEST_FRACTION)
    a = ap.parse_args()
    root = Path(a.root)
    eps = [json.loads(l) for l in (root / "episodes.jsonl").read_text().splitlines() if l.strip()]
    bad = json.loads((root / "bad_frames.json").read_text())["episodes"]
    s = build_splits(eps, bad, seed=a.seed, test_len=a.test_len,
                     max_train_window=a.max_train_window, target=a.target)
    (root / "splits.json").write_text(json.dumps(s, indent=1))
    st = s["stats"]
    print(f"{root/'splits.json'}")
    print(f"  test {st['test_fraction']*100:.1f}%  guard {st['guard_fraction']*100:.1f}%"
          f"  train {(1-st['test_fraction']-st['guard_fraction'])*100:.1f}%")
    print(f"  {st['n_test_intervals']} intervals + {st['n_whole_test_episodes']} "
          f"whole-test episodes, guard {s['guard_frames']} frames "
          f"(max_train_window {s['max_train_window']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
