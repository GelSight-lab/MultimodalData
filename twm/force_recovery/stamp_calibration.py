"""Backfill the calibration identity into force npz written before it was saved.

`run_episode` filtered str-valued metadata out of `np.savez_compressed`, so the
one field naming the map that produced the newtons — `force_calibration` — was
written to the log and nowhere else. The writer is fixed; this stamps the files
that already exist, which is far cheaper than re-running the reconstruction
(~200 s per sensor-side) purely to add a string.

It never touches `force_normal_n` and verifies as much, byte for byte: a
provenance backfill that could alter values would be worse than the gap.

    python -m force_recovery.stamp_calibration --dry-run
    python -m force_recovery.stamp_calibration
"""
from __future__ import annotations

import argparse

import numpy as np

from .react_calib import CALIBRATION_NAME
from .run_episode import OUT_ROOT

# Version -> the calibration that produced it. v4 is `react_calib`; v3 and
# earlier came from the pixel/mm mismatch and are NOT relabelled as anything
# trustworthy, because they are not.
CALIB_BY_VERSION = {
    4: CALIBRATION_NAME,
    3: "BROKEN: showcase._glowtact_calib, pixel-unit weights on mm features "
       "(end-to-end rho 0.143) -- do not use",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stamped = skipped = 0
    for p in sorted(OUT_ROOT.rglob("*.npz")):
        with np.load(p, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        ver = int(data["pipeline_version"]) if "pipeline_version" in data else 0
        if "force_calibration" in data:
            skipped += 1
            continue
        label = CALIB_BY_VERSION.get(ver)
        if label is None:
            print(f"  ? {p.name}: unknown pipeline_version {ver}, left alone")
            skipped += 1
            continue
        print(f"  + {p.relative_to(OUT_ROOT)}  v{ver}")
        if args.dry_run:
            stamped += 1
            continue
        before = data["force_normal_n"].copy()
        data["force_calibration"] = np.str_(label)
        np.savez_compressed(p, **data)
        with np.load(p, allow_pickle=True) as d2:
            assert np.array_equal(d2["force_normal_n"], before), p
            assert str(d2["force_calibration"]) == label, p
        stamped += 1
    print(f"\n{stamped} stamped, {skipped} already labelled or unknown"
          + ("  (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
