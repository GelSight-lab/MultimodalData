"""Thin loading helpers for the React video-format dataset.

Decoded frames are RGB uint8 (T, H, W, 3) — standard decoder convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    import av
    _BACKEND = "av"
except Exception:
    import cv2
    _BACKEND = "cv2"


def load_video(mp4_path, frames=None):
    """Decode an MP4 to (N, H, W, 3) uint8 RGB.

    frames=None -> all frames; else an iterable of frame indices.
    """
    mp4_path = str(mp4_path)
    want = None if frames is None else sorted(set(int(i) for i in frames))
    if _BACKEND == "av":
        c = av.open(mp4_path)
        out = []
        idxset = set(want) if want is not None else None
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if idxset is None or i in idxset:
                out.append(fr.to_ndarray(format="rgb24"))
            if idxset is not None and i >= max(idxset):
                break
        c.close()
        return np.stack(out)
    cap = cv2.VideoCapture(mp4_path)
    out = []
    if want is None:
        ok, fr = cap.read()
        while ok:
            out.append(fr[..., ::-1]); ok, fr = cap.read()
    else:
        for i in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            out.append(fr[..., ::-1] if ok else np.zeros((480, 640, 3), np.uint8))
    cap.release()
    return np.stack(out)


def episode_paths(task_root, episode):
    """Resolve an episode key '<date>/episode_NNN' to its file paths."""
    root = Path(task_root)
    date, ep = episode.split("/")
    vd = root / "videos" / date / ep
    return {
        "view_left": vd / "view_left.mp4", "view_middle": vd / "view_middle.mp4",
        "view_right": vd / "view_right.mp4",
        "tactile_left": vd / "tactile_left.mp4", "tactile_right": vd / "tactile_right.mp4",
        "depth_dir": root / "depth" / date / ep,
        "parquet": root / "meta" / date / f"{ep}.parquet",
    }


def load_meta(parquet_path, columns=None):
    """Load per-frame metadata as a dict of numpy arrays."""
    tbl = pq.read_table(str(parquet_path), columns=columns)
    out = {}
    for c in tbl.column_names:
        col = tbl.column(c).to_pylist()
        out[c] = np.array(col)
    return out
