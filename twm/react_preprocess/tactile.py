"""Two-pass GelSight processing: reference selection, then scalars + encode.

Pass 1 finds the no-contact (p01) reference frame; pass 2 computes the contact
metrics against it, flags genuinely-new frames, and streams the video out.
Frames are read from HDF5 in blocks so memory stays bounded regardless of
episode length.

Both passes read through the ``TactileAlignment`` index map, so a timestamped
recording is resampled onto the camera clock here — not patched up later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CHUNK, P01_SMOOTH_WIN
from .contact import (NewFrameTracker, contact_metrics, duplication_stats,
                      l2_diff, pick_p01_reference)
from .encode import rgb_writer
from .h5io import TactileAlignment


@dataclass
class TactileResult:
    side: str
    intensity: np.ndarray
    area: np.ndarray
    mixed: np.ndarray
    is_new: np.ndarray
    ref_index: int              # index into the source H5 dataset
    stats: dict


def _gather(ds, indices: np.ndarray) -> np.ndarray:
    """Read `indices` (non-decreasing) from an H5 dataset in one contiguous slice."""
    lo, hi = int(indices[0]), int(indices[-1])
    span = ds[lo:hi + 1]
    return span[indices - lo]


def _blocks(total: int, size: int = CHUNK):
    for s in range(0, total, size):
        yield s, min(s + size, total)


def process_side(h5file, side: str, align: TactileAlignment, out_path: Path,
                 encode: bool = True) -> TactileResult:
    """Run both passes for one GelSight and write its MP4."""
    ds = h5file[f"gelsight/{side}/frames"]        # (N, H, W, 3) uint8 RGB
    idx_map = align.index_map
    T = len(idx_map)

    # ── pass 1: intensity vs the first frame -> smoothed argmin = p01 ────────
    ref0 = ds[0].astype(np.float32)
    rough = np.empty(T, np.float32)
    for s, e in _blocks(T):
        rough[s:e] = l2_diff(_gather(ds, idx_map[s:e]), ref0).mean(axis=(1, 2))
    p01_local = pick_p01_reference(rough, P01_SMOOTH_WIN)
    ref_index = int(idx_map[p01_local])
    reference = ds[ref_index]

    # ── pass 2: metrics against p01 + new-frame flags + encode ──────────────
    intensity = np.empty(T, np.float32)
    area = np.empty(T, np.float32)
    mixed = np.empty(T, np.float32)
    is_new = np.empty(T, bool)
    tracker = NewFrameTracker()

    writer = rgb_writer(out_path) if encode else None
    ctx = writer if writer is not None else _NullCtx()
    with ctx:
        for s, e in _blocks(T):
            block = _gather(ds, idx_map[s:e])                 # uint8 RGB
            i, a, m = contact_metrics(block, reference)
            intensity[s:e], area[s:e], mixed[s:e] = i, a, m
            is_new[s:e] = tracker.update(block)
            if writer is not None:
                writer.write(block[..., ::-1])                # RGB -> BGR

    return TactileResult(side, intensity, area, mixed, is_new, ref_index,
                         duplication_stats(is_new))


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
