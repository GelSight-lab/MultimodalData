"""Bounded-memory preview encoding with decode verification and atomic output.

The frame factory is replayed only if verification fails. It must yield the
same aligned BGR frames on each call; inference and publishing do not belong here.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
import math
import os
import subprocess
import tempfile

import cv2
import numpy as np


def decoded_frame_count(path: Path) -> int:
    """Count actual decoded frames without keeping decoded pixels in memory."""
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                return count
            count += 1
    finally:
        cap.release()


def _validate_frame(frame, shape=None):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("video frames must be uint8 BGR arrays")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
        raise ValueError("video frames must have nonempty (height, width, 3) shape")
    if shape is not None and frame.shape != shape:
        raise ValueError(f"frame shape changed: {frame.shape} != {shape}")


def _encode(path, frames, fps):
    first = next(frames, None)
    if first is None:
        raise ValueError("cannot encode an empty frame sequence")
    _validate_frame(first)
    height, width = first.shape[:2]
    shape = first.shape
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-profile:v", "high444",
        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", "-an", str(path),
    ]
    # A file avoids stderr pipe deadlocks and unbounded error buffering.
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors)
        count = 0
        try:
            stream = chain((first,), frames)
            del first
            for frame in stream:
                _validate_frame(frame, shape)
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                count += 1
            proc.stdin.close()
            rc = proc.wait()
            if rc:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg failed ({rc}): {errors.read(8192).decode(errors='replace')}")
        except BaseException:
            # Also reap on producer/validation errors or BrokenPipe from close().
            if proc.poll() is None:
                proc.kill()
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            proc.wait()
            raise
    return count


def write_video(path, frame_factory, *, fps=30.0, attempts=3):
    """Encode and verify, returning frame count; preserve existing output on failure.

    ``frame_factory()`` returns an iterable of identically sized BGR uint8 frames.
    Open sources inside a generator's ``with`` block so closing on failure releases
    them. Invalid input/encoder errors are not retried. A decode-count mismatch is.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".mp4", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        for _ in range(attempts):
            frames = iter(frame_factory())
            try:
                count = _encode(temporary, frames, fps)
            finally:
                close = getattr(frames, "close", None)
                if close is not None:
                    close()
            decoded = decoded_frame_count(temporary)
            if decoded == count:
                temporary.replace(path)
                return count
        raise RuntimeError(f"preview decodes to {decoded}/{count} frames after {attempts} attempts")
    finally:
        temporary.unlink(missing_ok=True)
