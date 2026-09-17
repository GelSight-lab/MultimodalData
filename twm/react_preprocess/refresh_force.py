"""Update the force columns of an already-cut segment. Nothing else.

A force re-run changes the force columns and nothing about the pixels, so
re-cutting is the wrong tool: `segment` re-encodes all seven streams -- its own
docstring notes "This re-encodes, so a cut stream is a second H.264
generation", which makes a re-cut segment a THIRD -- and it costs hours. On
2026-09-17 a `--force` re-cut was started for exactly this and had to be
killed: it takes no date range, so it began re-encoding 2026-05 data that is
out of scope, and the kill truncated one stream.

This reads the uncut Z-up episode, takes the rows the segment actually holds,
and writes only the `force_*` columns into the existing segment parquet.
Videos are never opened.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORCE_PREFIX = "force_"

# The column both files carry that says WHICH tactile frame a row came from.
# Alignment is by this, never by position: a segment is a contiguous run of
# the episode, so a positional slice is usually right -- and silently wrong
# for exactly the episodes whose trim moved, which are the ones that matter.
KEY = "source_h5_frame"


def refresh_segment(segment_parquet: Path, episode_parquet: Path) -> dict:
    """Rewrite `segment_parquet`'s force columns from `episode_parquet`."""
    seg = pq.read_table(str(segment_parquet))
    ep = pq.read_table(str(episode_parquet))

    force_cols = [c for c in ep.column_names if c.startswith(FORCE_PREFIX)]
    if not force_cols:
        raise ValueError(f"{episode_parquet}: no force columns to copy")
    for t, name in ((seg, segment_parquet), (ep, episode_parquet)):
        if KEY not in t.column_names:
            raise ValueError(f"{name}: no {KEY}; cannot align without guessing")

    want = np.asarray(seg[KEY].to_numpy(), np.int64)
    have = np.asarray(ep[KEY].to_numpy(), np.int64)
    order = np.argsort(have)
    pos = np.searchsorted(have[order], want)
    bad = (pos >= len(have)) | (have[order][np.clip(pos, 0, len(have) - 1)] != want)
    if bad.any():
        missing = want[bad][:5].tolist()
        raise ValueError(
            f"{segment_parquet.name}: source frame {missing} not in "
            f"{episode_parquet.name} — refusing to guess or zero-fill")
    take = pa.array(order[pos])

    names, cols = [], []
    for name, col in zip(seg.column_names, seg.columns):
        if not name.startswith(FORCE_PREFIX):
            names.append(name); cols.append(col)
    for name in force_cols:                      # replaces AND adds
        names.append(name); cols.append(ep[name].take(take))

    out = pa.table(cols, names=names).replace_schema_metadata(
        dict(seg.schema.metadata or {}))
    pq.write_table(out, str(segment_parquet), compression="zstd")
    return {"rows": out.num_rows, "columns": force_cols}
