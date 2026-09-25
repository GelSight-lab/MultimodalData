"""What the PUBLISHED release actually carries, read from the release.

The results page carried the sentence "the published dataset still carries the
LUT column: switching it needs 36 episodes reprocessed" as literal HTML. It was
true when written. It stops being true the moment `reprocess_react promote`
runs, and nothing would have made it stop — a typed sentence about the state of
a directory cannot notice the directory changing.

So the sentence is generated from the directory. This module reads the
calibration string out of every published side and records what it found; the
page prints it. If half the release is promoted and half is not, that shows up
here as two calibrations instead of one, which is exactly the state a release
must never be shipped in.

    python -m force_recovery.release_channel
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "release_channel.json"


def survey() -> dict:
    from .react_calib import CALIBRATION_NAME
    from .reprocess_react import episodes

    jobs = episodes()
    names: Counter = Counter()
    frames = 0
    for task, date, ep, side in jobs:
        with np.load(OUT_ROOT / task / date / f"{ep}_{side}.npz",
                     allow_pickle=True) as d:
            names[str(d["force_calibration"])] += 1
            frames += int(len(d["force_normal_n"]))
    eps = {(t, d, e) for t, d, e, _ in jobs}
    top = names.most_common(1)[0][0] if names else ""
    return {"sides": len(jobs), "episodes": len(eps), "frames": frames,
            "calibrations": dict(names),
            "calibration": top,
            "uniform": len(names) == 1,
            "matches_current_code": top == CALIBRATION_NAME,
            "current_code_calibration": CALIBRATION_NAME}


def main() -> int:
    from .artifact_lock import one_writer
    with one_writer(OUT):
        rep = survey()
        OUT.write_text(json.dumps(rep, indent=1))
    print(f"  {rep['sides']} 侧 / {rep['episodes']} episode / "
          f"{rep['frames']} 帧")
    for k, v in rep["calibrations"].items():
        print(f"    {v:3d} 侧: {k}")
    print(f"  与当前代码一致: {rep['matches_current_code']}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
