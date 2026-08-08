"""Add tactile validity flags to parquet files that were built before them.

The already-published release has no ``tactile_*_is_new`` columns, and
rebuilding it from source would mean re-encoding every video. Instead we
recover the flags from data already in the parquet.

Method — a repeated GelSight frame produces a bit-identical contact triple
(intensity, area, mixed), because all three are deterministic reductions of
the same pixels. So a row is a fresh reading exactly when its triple differs
from the previous row's.

This is a *proxy*, not a pixel comparison: two genuinely different frames
would have to agree in all three float32 reductions to be missed, which does
not happen in practice but is not impossible. ``verify_against_video()``
checks the proxy against real decoded frames on a sample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .contact import duplication_stats
from .meta import backfill_is_new

SCALARS = ("intensity", "area", "mixed")


def flags_from_scalars(table, side: str) -> np.ndarray:
    """Recover per-row 'this is a new tactile frame' flags for one side."""
    cols = [np.asarray(table[f"tactile_{side}_{s}"].to_numpy(), np.float64)
            for s in SCALARS if f"tactile_{side}_{s}" in table.column_names]
    if not cols:
        raise KeyError(f"parquet has no tactile_{side}_* contact columns")
    stacked = np.stack(cols, axis=1)
    is_new = np.ones(len(stacked), dtype=bool)
    if len(stacked) > 1:
        is_new[1:] = np.any(stacked[1:] != stacked[:-1], axis=1)
    return is_new


def process_parquet(path: Path, dry_run: bool = False) -> dict:
    """Backfill one parquet in place; returns its duplication stats."""
    table = pq.read_table(str(path))
    left = flags_from_scalars(table, "left")
    right = flags_from_scalars(table, "right")
    if not dry_run:
        pq.write_table(backfill_is_new(table, left, right), str(path))
    return {
        "path": str(path),
        "left": duplication_stats(left),
        "right": duplication_stats(right),
    }


def process_tree(root: Path, dry_run: bool = False) -> list[dict]:
    """Backfill every ``meta/**/episode_*.parquet`` under a task directory."""
    files = sorted(Path(root).rglob("meta/**/episode_*.parquet"))
    if not files:
        files = sorted(Path(root).rglob("episode_*.parquet"))
    return [process_parquet(p, dry_run) for p in files]


def aggregate(reports: list[dict]) -> dict:
    """Dataset-level duplication summary across episodes."""
    total = unique = 0
    worst = 0
    for rep in reports:
        for side in ("left", "right"):
            s = rep[side]
            total += s["n_frames"]
            unique += s["n_unique"]
            worst = max(worst, s["max_repeat_run"])
    ratio = 1.0 - unique / total if total else 0.0
    return {
        "episodes": len(reports),
        "rows": total,
        "unique_tactile": unique,
        "duplicate_ratio": ratio,
        "effective_fps": 30.0 * (1.0 - ratio),
        "max_repeat_run": worst,
    }


def _source_is_new(h5_path: Path, side: str, start: int, count: int) -> np.ndarray:
    """Bit-exact 'this frame differs from the previous one' over source pixels."""
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(str(h5_path), "r") as f:
        # Consecutive frames of ONE stream compared to each other (bit-exact
        # is_new truth); no cross-modal pairing, so the camera<->gel lag
        # cannot enter.
        block = f[f"gelsight/{side}/frames"][start:start + count]  # tactile-lag-exempt
    truth = np.ones(len(block), bool)
    for i in range(1, len(block)):
        truth[i] = not np.array_equal(block[i], block[i - 1])
    return truth


def verify_against_h5(parquet_path: Path, h5_path: Path, side: str = "left",
                      limit: int = 600, shift: int | None = None,
                      search: range | None = None) -> dict:
    """Ground-truth check of the flags against the source H5 pixels.

    The source frames are the only bit-exact reference — the published MP4s are
    H.264-encoded, so a duplicated frame does not decode back to identical
    pixels (use ``verify_against_video``, which compares with a tolerance).

    A published episode may have had a constant tactile latency shift baked in,
    in which case parquet row ``i`` corresponds to source frame
    ``trim + i + shift``. Pass ``shift``, or leave it None to search for the
    value that lines the two up — that search doubles as an integrity check
    that the intended correction really was applied.
    """
    table = pq.read_table(str(parquet_path))
    proxy = flags_from_scalars(table, side)[:limit]
    trim = int(np.asarray(table["source_h5_frame"].to_numpy())[0])
    n_req = len(proxy)

    candidates = ([shift] if shift is not None
                  else list(search) if search is not None else [0, 15])
    lo, hi = min(candidates), max(candidates)

    # Read the source span once and slide over it — re-reading per candidate
    # shift would multiply the (large) HDF5 traffic by len(candidates).
    start = max(0, trim + lo - 1)                 # one extra for a predecessor
    pad = (trim + lo) - start                     # 1 unless clamped at 0
    span = _source_is_new(h5_path, side, start, (hi - lo) + n_req + pad)

    def slice_truth(sh: int) -> np.ndarray:
        off = pad + (sh - lo)
        return span[off:off + n_req]

    # Row 0 is True by convention on both sides (neither has a predecessor
    # inside its own window), so it carries no evidence — compare from row 1.
    scored = []
    for sh in candidates:
        truth = slice_truth(sh)
        n = min(len(truth), n_req)
        if n < 2:
            continue
        scored.append((int((truth[1:n] != proxy[1:n]).sum()), n, sh))
    if not scored:
        raise ValueError(f"{parquet_path.name}: no overlap with source frames")

    mismatches, n, best = min(scored, key=lambda x: x[0])
    truth = slice_truth(best)
    return {
        "compared": int(n - 1),
        "mismatches": int(mismatches),
        "shift": int(best),
        "shift_detected": shift is None,
        "proxy_unique": int(proxy[1:n].sum()),
        "source_unique": int(truth[1:n].sum()),
    }


def verify_against_video(parquet_path: Path, video_path: Path,
                         limit: int = 300, tol: float = 1.0) -> dict:
    """Check the proxy against decoded video, comparing with a tolerance.

    H.264 is lossy, so a duplicated source frame still decodes to slightly
    different pixels. We therefore call two decoded frames "the same" when
    their mean absolute difference stays below `tol`, and report the observed
    separation so the threshold can be sanity-checked rather than trusted.
    """
    import av

    table = pq.read_table(str(parquet_path))
    side = "left" if "tactile_left" in video_path.name else "right"
    proxy = flags_from_scalars(table, side)[:limit]

    mad, prev = [], None
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= limit:
                break
            arr = frame.to_ndarray(format="rgb24").astype(np.float32)
            mad.append(0.0 if prev is None
                       else float(np.abs(arr - prev).mean()))
            prev = arr
    mad = np.asarray(mad)
    truth = mad > tol
    truth[0] = True

    n = min(len(truth), len(proxy))
    dup_mad = mad[1:n][~proxy[1:n]]
    new_mad = mad[1:n][proxy[1:n]]
    return {
        "compared": int(n),
        "mismatches": int((truth[:n] != proxy[:n]).sum()),
        "proxy_unique": int(proxy[:n].sum()),
        "video_unique": int(truth[:n].sum()),
        "mad_duplicate_max": float(dup_mad.max()) if len(dup_mad) else 0.0,
        "mad_new_min": float(new_mad.min()) if len(new_mad) else 0.0,
    }
