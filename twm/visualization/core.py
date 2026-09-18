"""Hardware-independent BGR tile rendering with local, ordered overlays."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Iterable, Literal, Optional

import cv2
import numpy as np


ColorFormat = Literal["BGR", "RGB", "depth"]
Overlay = Callable[[np.ndarray], None]


@dataclass(frozen=True)
class Tile:
    """One named source image; ``None`` explicitly represents missing data.

    Color sources must be H×W×3 uint8. ``depth`` sources are real H×W
    arrays, displayed as grayscale. Finite positive samples are valid depth;
    ``modality='tactile_height'`` also permits zero and negative heights.
    Invalid samples are black. An omitted depth range uses valid min/max;
    constant or wholly invalid sources are black.

    Overlays run in order on a native-size, contiguous BGR copy before
    resizing. They mutate that copy and return None; use closures for poses,
    masks, contacts, or tactile context. No callback receives another tile.
    """

    name: str
    image: Optional[np.ndarray]
    modality: str = "color"
    color_format: ColorFormat = "BGR"
    overlays: tuple[Overlay, ...] = ()
    depth_range: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class TileLayout:
    """Pixel bounds of one tile in the rendered BGR image."""

    name: str
    modality: str
    x: int
    y: int
    width: int
    height: int
    missing: bool


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray
    layouts: tuple[TileLayout, ...]


def _to_bgr(tile: Tile) -> np.ndarray:
    image = tile.image
    if tile.color_format not in ("BGR", "RGB", "depth"):
        raise ValueError(f"Unsupported color format: {tile.color_format!r}")
    if not isinstance(image, np.ndarray) or image.ndim < 2 or 0 in image.shape:
        raise ValueError(f"Tile {tile.name!r} requires a nonempty image array")
    if tile.color_format != "depth":
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Color images must be H×W×3 uint8 arrays")
        return (image[:, :, ::-1] if tile.color_format == "RGB" else image).copy(order="C")

    if image.ndim != 2 or image.dtype.kind not in "uif":
        raise ValueError("Depth images must be real numeric H×W arrays")
    values = image.astype(np.float64)
    valid = np.isfinite(values)
    if tile.modality != "tactile_height":
        valid &= values > 0
    gray = np.zeros(values.shape, np.uint8)
    if tile.depth_range is not None:
        limits = np.asarray(tile.depth_range, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("Depth range must contain two finite increasing limits")
        low, high = limits
    elif valid.any():
        low, high = values[valid].min(), values[valid].max()
    else:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if high > low:
        # Scale first so even large, finite signed height ranges cannot
        # overflow during subtraction. Invalid samples never enter the math.
        scale = max(abs(low), abs(high))
        samples = np.clip(values[valid], low, high) / scale
        normalized = (samples - low / scale) / (high / scale - low / scale)
        gray[valid] = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@dataclass(frozen=True)
class Renderer:
    """Render tiles in row-major order, stretching each to the given size.

    Empty cells in the last row are black. Missing tiles show a label and
    skip source-dependent overlays. The output always owns its pixel data.
    """

    tile_width: int = 320
    tile_height: int = 240
    columns: int = 3

    def __post_init__(self) -> None:
        for name in ("tile_width", "tile_height", "columns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def render(self, tiles: Iterable[Tile]) -> RenderResult:
        tiles = tuple(tiles)
        if not tiles:
            raise ValueError("At least one tile is required")
        rows = (len(tiles) + self.columns - 1) // self.columns
        panel = np.zeros((rows * self.tile_height, self.columns * self.tile_width, 3), np.uint8)
        layouts = []
        for index, tile in enumerate(tiles):
            x = (index % self.columns) * self.tile_width
            y = (index // self.columns) * self.tile_height
            missing = tile.image is None
            if missing:
                thumb = np.zeros((self.tile_height, self.tile_width, 3), np.uint8)
                cv2.putText(thumb, f"{tile.name}: missing", (8, min(24, self.tile_height - 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            else:
                native = _to_bgr(tile)
                for overlay in tile.overlays:
                    overlay(native)
                thumb = cv2.resize(native, (self.tile_width, self.tile_height),
                                   interpolation=cv2.INTER_NEAREST)
            panel[y:y + self.tile_height, x:x + self.tile_width] = thumb
            layouts.append(TileLayout(tile.name, tile.modality, x, y,
                                      self.tile_width, self.tile_height, missing))
        return RenderResult(panel, tuple(layouts))
