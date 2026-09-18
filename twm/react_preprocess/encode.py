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
        if pix_fmt not in ("bgr24", "gray16le"):
            raise ValueError(f"unsupported input pixel format {pix_fmt!r}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cmd = self._build(pix_fmt, codec, width, height, fps)
        self._proc = None
        self._frame_shape = ((height, width, 3) if pix_fmt == "bgr24"
                             else (height, width))
        self._dtype = np.dtype("uint8" if pix_fmt == "bgr24" else "<u2")

    def _build(self, pix_fmt, codec, width, height, fps):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", pix_fmt,
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if codec == "libx264":
            # `fast`, not `medium`. Measured on 300 real frames against the raw
            # pixels: medium 74 fps / PSNR 48.14 dB, fast 105 fps / 47.96 dB,
            # same file size. 0.18 dB for 40 % of the time. `veryfast` buys
            # 2.9x for 1.82 dB and was NOT taken — the gel gradients are the
            # force estimator's input and nothing shows 46 dB leaves them
            # intact. `ultrafast` is slower AND 3x larger: it drops motion
            # estimation, so the write cost overtakes the CPU it saves.
            cmd += ["-c:v", "libx264", "-profile:v", "high444", "-preset", "fast",
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
        block = np.asarray(block)
        if (block.ndim != len(self._frame_shape) + 1
                or block.shape[1:] != self._frame_shape or block.dtype != self._dtype):
            raise ValueError(
                f"{self.path}: expected {self._dtype} blocks shaped "
                f"(N, {', '.join(map(str, self._frame_shape))}), "
                f"got {block.dtype} {block.shape}")
        self._proc.stdin.write(np.ascontiguousarray(block).tobytes())

    def __exit__(self, exc_type, exc, tb):
        close_error = None
        wait_error = None
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception as error:
            close_error = error
        finally:
            try:
                rc = self._proc.wait()
            except Exception as error:
                wait_error = error
        # Ordinary cleanup failures must not hide the frame/encoding error
        # that caused us to leave the context. Process-control errors escape.
        if exc_type is None and wait_error is not None:
            raise RuntimeError(f"ffmpeg wait failed writing {self.path}") from wait_error
        if exc_type is None and (rc != 0 or close_error is not None):
            raise RuntimeError(f"ffmpeg failed ({rc}) writing {self.path}") from close_error
        return False


def rgb_writer(path: Path) -> VideoWriter:
    """8-bit colour stream. Feed BGR blocks (ffmpeg's native order)."""
    return VideoWriter(path, pix_fmt="bgr24", codec="libx264")


def depth_writer(path: Path) -> VideoWriter:
    """16-bit depth stream, lossless. Feed uint16 millimetre blocks."""
    return VideoWriter(path, pix_fmt="gray16le", codec="ffv1")
