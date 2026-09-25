"""Put `raw_h5_offset_m` back in the convention its own note describes.

The Z-up conversion rotated every vector in the parquet's `twm.world_frame`
blob, including this one. Every other number there describes the PUBLISHED
poses, which are Z-up. This one describes what to add to a pose read straight
out of the source HDF5 -- and the source HDF5 is Y-up, as recorded. Rotating it
left 175 mm on the wrong axis under a note that still said "straight out of the
H5".

Idempotent: a file that already carries `raw_h5_offset_up_axis` is left alone.

    python scripts/fix_raw_offset_axis.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402
import pyarrow as pa                                            # noqa: E402
import pyarrow.parquet as pq                                    # noqa: E402

from react_paths import force_meta, release_root                # noqa: E402
from react_toolbox.frames import ZUP_TO_YUP                     # noqa: E402

NOTE = ("add this to a pose read straight out of the source H5, which is Y-up "
        "as recorded; the published poses already have it, expressed Z-up")


def fix(meta_dir: Path, dry: bool) -> tuple[int, int]:
    seen = done = 0
    for p in sorted(meta_dir.glob("*/*.parquet")):
        t = pq.read_table(p)
        md = dict(t.schema.metadata or {})
        raw = md.get(b"twm.world_frame")
        if not raw:
            continue
        d = json.loads(raw.decode())
        if "raw_h5_offset_m" not in d:
            continue
        seen += 1
        if d.get("raw_h5_offset_up_axis"):
            continue                       # already repaired
        v = np.asarray(ZUP_TO_YUP, float) @ np.asarray(d["raw_h5_offset_m"], float)
        d["raw_h5_offset_m"] = [float(x) for x in v]
        d["raw_h5_offset_up_axis"] = "y"
        d["raw_h5_note"] = NOTE
        done += 1
        if dry:
            continue
        md[b"twm.world_frame"] = json.dumps(d).encode()
        pq.write_table(t.replace_schema_metadata(md), p)
    return seen, done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    # EVERY task. This was hard-coded to "motherboard", so when pushT was
    # finally converted the repair that goes with it would have skipped it in
    # silence -- the same single-task blind spot that let pushT ship Y-up
    # beside a Z-up motherboard in the first place.
    tasks = sorted(q.name for q in release_root().iterdir()
                   if (q / "episodes.jsonl").exists())
    print(f"tasks: {', '.join(tasks)}")
    total = 0
    pairs = []
    for t in tasks:
        pairs.append((f"release/{t}", release_root(t) / "meta"))
        pairs.append((f"release_force/{t}", force_meta(t)))
    for name, d in pairs:
        if not d.is_dir():
            print(f"  {name}: {d} missing, skipped")
            continue
        seen, done = fix(d, a.dry_run)
        total += done
        print(f"  {name}: {seen} declarations, {done} "
              f"{'would be ' if a.dry_run else ''}repaired  ({d})")
    print(f"\n{total} files {'would change' if a.dry_run else 'repaired'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
