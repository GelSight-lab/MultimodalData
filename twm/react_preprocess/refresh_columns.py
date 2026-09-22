"""Bring an already-cut segment up to a wider episode schema. No re-encoding.

`refresh_force` does this for the `force_*` family. The same argument applies
whenever a stage adds columns to the uncut episode after the cut was made:
`segment` re-encodes all seven streams, which makes the cut stream a third
H.264 generation and costs hours, to change bytes that did not change.

The case this exists for: pushT/2026-09-17 shipped with 25 columns while every
other published date carries 60. The missing ones are pose-repair provenance
and the action channel, both metadata. Re-cutting to add them would re-encode
168 videos whose pixels are already correct.

Alignment is by `source_h5_frame`, never by position, for the reason
`refresh_force` gives: a positional slice is right for a segment whose episode
was not trimmed and silently wrong for one that was.

`sensor_*_pose` is REPLACED, not only added to. On pushT/2026-09-17 the cut
tree holds poses that were repaired in place before the convention settled;
the episode tree now holds the raw measurement with the repair beside it, and
a refresh that added the new columns while leaving the old ones would publish
two contradictory accounts of the same frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEY = "source_h5_frame"
# Written by `segment` and meaningless in the uncut episode, so they are never
# taken from it even when it happens to carry a column of the same name.
SEGMENT_OWNED = ("source_episode", "source_frame_idx")


def refresh_segment(segment_parquet: Path, episode_parquet: Path,
                    drop: tuple[str, ...] = ()) -> dict:
    """Rewrite `segment_parquet` from `episode_parquet`, keeping the cut.

    Every column the episode has is taken from it. Columns the segment owns
    are kept. Columns named in `drop` are removed -- for a column the new
    schema replaces rather than updates, such as a provenance flag whose
    meaning moved into a different column.
    """
    seg = pq.read_table(str(segment_parquet))
    ep = pq.read_table(str(episode_parquet))
    for t, name in ((seg, segment_parquet), (ep, episode_parquet)):
        if KEY not in t.column_names:
            raise ValueError(f"{name}: no {KEY}; cannot align without guessing")

    want = np.asarray(seg[KEY].to_numpy(), np.int64)
    have = np.asarray(ep[KEY].to_numpy(), np.int64)
    order = np.argsort(have)
    pos = np.searchsorted(have[order], want)
    bad = (pos >= len(have)) | (have[order][np.clip(pos, 0, len(have) - 1)] != want)
    if bad.any():
        raise ValueError(
            f"{segment_parquet.name}: source frame {want[bad][:5].tolist()} "
            f"not in {episode_parquet.name} — refusing to guess or zero-fill")
    take = pa.array(order[pos])

    from_ep = [c for c in ep.column_names
               if c not in SEGMENT_OWNED and c not in drop]
    names, cols = [], []
    for name, col in zip(seg.column_names, seg.columns):
        if name in drop or name in from_ep:
            continue                              # replaced below, or removed
        names.append(name); cols.append(col)
    for name in from_ep:
        names.append(name); cols.append(ep[name].take(take))

    meta = dict(ep.schema.metadata or {})
    meta.update(dict(seg.schema.metadata or {}))  # the cut's own keys win
    out = pa.table(cols, names=names).replace_schema_metadata(meta or None)
    pq.write_table(out, str(segment_parquet), compression="zstd")
    return {"rows": out.num_rows, "columns": len(out.column_names),
            "added": sorted(set(from_ep) - set(seg.column_names)),
            "dropped": sorted(set(drop) & set(seg.column_names))}
