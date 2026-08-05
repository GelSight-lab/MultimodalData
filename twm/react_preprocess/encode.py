"""ffmpeg encoders for the published video streams.

RGB / tactile  -> H.264 yuv444p CRF18 (visually lossless, seekable)
depth          -> FFV1 gray16le      (mathematically lossless, millimetres)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import CRF, FPS, H, W


class VideoWriter:
    """Streaming ffmpeg writer. Use as a context manager and `write()` blocks."""

    def __init__(self, path: Path, pix_fmt="bgr24", codec="libx264",
                 width=W, height=H, fps=FPS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cmd = self._build(pix_fmt, codec, width, height, fps)
        self._proc = None

    def _build(self, pix_fmt, codec, width, height, fps):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", pix_fmt,
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if codec == "libx264":
            cmd += ["-c:v", "libx264", "-profile:v", "high444", "-preset", "medium",
                    "-crf", CRF, "-pix_fmt", "yuv444p", "-movflags", "+faststart"]
        elif codec == "ffv1":
            cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le"]
        else:
            raise ValueError(f"unknown codec {codec!r}")
        return cmd + ["-an", str(self.path)]

    def __enter__(self):
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        return self

    def write(self, block: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(block).tobytes())

    def __exit__(self, exc_type, exc, tb):
        if self._proc.stdin:
            self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0 and exc_type is None:
            raise RuntimeError(f"ffmpeg failed ({rc}) writing {self.path}")
        return False


def rgb_writer(path: Path) -> VideoWriter:
    """8-bit colour stream. Feed BGR blocks (ffmpeg's native order)."""
    return VideoWriter(path, pix_fmt="bgr24", codec="libx264")


def depth_writer(path: Path) -> VideoWriter:
    """16-bit depth stream, lossless. Feed uint16 millimetre blocks."""
    return VideoWriter(path, pix_fmt="gray16le", codec="ffv1")
