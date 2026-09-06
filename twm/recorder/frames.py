"""Tick — one synchronized multimodal sample produced per capture cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

COLOR_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (480, 640)


@dataclass(frozen=True)
class Tick:
    """Everything captured in one 30 Hz cycle.

    Frames are the sensors' own copies (streams copy on read), so the
    writer may keep references without copying again. Per-sensor timestamps
    are the true capture time; the recorder tick time is `timestamp`.
    """
    OPTITRACK_SAMPLE_BYTES = 8 * 8  # 1 timestamp + 7 pose floats

    timestamp: float
    color: Tuple[np.ndarray, ...] = ()
    depth: Tuple[np.ndarray, ...] = ()
    gelsight: Tuple[np.ndarray, ...] = ()
    gelsight_ts: Tuple[float, ...] = ()
    arducam: Tuple[np.ndarray, ...] = ()
    arducam_ts: Tuple[float, ...] = ()
    optitrack: Mapping[str, Sequence[Tuple[float, Sequence[float]]]] = field(
        default_factory=dict)

    def __post_init__(self):
        if len(self.gelsight_ts) != len(self.gelsight):
            raise ValueError(f"{len(self.gelsight)} GelSight frames but "
                             f"{len(self.gelsight_ts)} timestamps")
        if len(self.arducam_ts) != len(self.arducam):
            raise ValueError(f"{len(self.arducam)} Arducam frames but "
                             f"{len(self.arducam_ts)} timestamps")

    def nbytes(self) -> int:
        arrays = (*self.color, *self.depth, *self.gelsight, *self.arducam)
        n = sum(int(a.nbytes) for a in arrays)
        n += self.OPTITRACK_SAMPLE_BYTES * sum(len(v) for v in self.optitrack.values())
        return n


def full_rig_tick_nbytes(n_realsense: int = 3, n_gelsight: int = 2,
                         n_arducam: int = 2) -> int:
    """Bytes of one tick for the given rig, without building one."""
    color = int(np.prod(COLOR_SHAPE))
    depth = int(np.prod(DEPTH_SHAPE)) * 2
    return n_realsense * (color + depth) + (n_gelsight + n_arducam) * color


def synthetic_tick(timestamp: float, seed: int = 0, n_realsense: int = 3,
                   n_gelsight: int = 2, n_arducam: int = 0) -> Tick:
    """A realistic-looking tick (gradient + noise) for benchmarks and tests.

    Pure noise defeats BLOSC and understates throughput; a flat frame
    overstates it. This sits in between, like a real scene.

    Under BITSHUFFLE, this synthetic noise still compresses noticeably
    worse than real camera frames: the write-bandwidth preflight measured
    ~170 ticks/s on real frames vs. ~104 ticks/s on this synthetic tick on
    the same NVMe drive (about 40% slower). The preflight's pass/fail bound
    is therefore pessimistic relative to what the real rig will sustain.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:COLOR_SHAPE[0], 0:COLOR_SHAPE[1]]
    base = ((xx * 255 // COLOR_SHAPE[1]) + (yy * 255 // COLOR_SHAPE[0])) // 2

    def color_frame():
        noise = rng.integers(0, 24, COLOR_SHAPE, dtype=np.uint8)
        return (base[..., None].astype(np.uint8) + noise).astype(np.uint8)

    def depth_frame():
        noise = rng.integers(0, 40, DEPTH_SHAPE, dtype=np.uint16)
        return (base.astype(np.uint16) * 8 + noise).astype(np.uint16)

    return Tick(
        timestamp=timestamp,
        color=tuple(color_frame() for _ in range(n_realsense)),
        depth=tuple(depth_frame() for _ in range(n_realsense)),
        gelsight=tuple(color_frame() for _ in range(n_gelsight)),
        gelsight_ts=tuple(timestamp for _ in range(n_gelsight)),
        arducam=tuple(color_frame() for _ in range(n_arducam)),
        arducam_ts=tuple(timestamp for _ in range(n_arducam)),
    )
