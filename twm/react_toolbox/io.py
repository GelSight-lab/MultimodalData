"""Thin loading helpers for the React video-format dataset.

Decoded frames are RGB uint8 (T, H, W, 3) — standard decoder convention.
"""
from __future__ import annotations

from pathlib import Path
from numbers import Integral

import numpy as np


def _video_backend():
    """Load decoder dependencies only when video is requested."""
    try:
        import av
        return "av", av
    except ImportError:
        import cv2
        return "cv2", cv2


def load_video(mp4_path, frames=None):
    """Decode an MP4 to (N, H, W, 3) uint8 RGB.

    ``frames=None`` decodes all frames; selections are sorted and deduplicated.
    An empty selection returns shape (0, 0, 0, 3) without opening the file.
    Negative/noninteger indices raise ValueError; unavailable requested frames
    raise IndexError. Neither backend fabricates black frames or drops requests.
    """
    mp4_path = str(mp4_path)
    want = None if frames is None else list(frames)
    if want is not None:
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
               or i < 0 for i in want):
            raise ValueError("frame indices must be nonnegative integers")
        want = sorted(set(int(i) for i in want))
        if not want:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
    idxset = None if want is None else set(want)
    last = None if want is None else want[-1]
    backend, decoder = _video_backend()
    out = []
    found = set()
    if backend == "av":
        container = decoder.open(mp4_path)
        try:
            if not container.streams.video:
                raise ValueError(f"No video stream in {mp4_path}")
            for i, frame in enumerate(container.decode(container.streams.video[0])):
                if idxset is None or i in idxset:
                    out.append(frame.to_ndarray(format="rgb24"))
                    found.add(i)
                if last is not None and i >= last:
                    break
        finally:
            container.close()
    else:
        cap = decoder.VideoCapture(mp4_path)
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot open video {mp4_path}")
            i = 0
            while last is None or i <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                if idxset is None or i in idxset:
                    out.append(frame[..., ::-1].copy())
                    found.add(i)
                i += 1
        finally:
            cap.release()
    if idxset is not None and idxset - found:
        raise IndexError(f"Unavailable frame indices in {mp4_path}: {sorted(idxset - found)}")
    if not out:
        raise ValueError(f"No decodable video frames in {mp4_path}")
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
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(parquet_path), columns=columns)
    out = {}
    for c in tbl.column_names:
        col = tbl.column(c).to_pylist()
        out[c] = np.array(col)
    return out
