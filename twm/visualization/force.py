"""Pure force-result image helpers, independent of estimation and file I/O."""
from __future__ import annotations

import cv2
import numpy as np

from .core import Renderer, Tile

# Keep the signed colour difference ungained: gain clips the strongest contacts.
DIFF_GAIN = 1.0


def diff_rgb(img: np.ndarray, ref: np.ndarray, gain: float = DIFF_GAIN):
    """Signed RGB difference centred on neutral grey, preserving LED channels."""
    d = (np.asarray(img, np.float32) - np.asarray(ref, np.float32)) * gain
    return np.clip(d + 128.0, 0, 255).astype(np.uint8)


def diff_caption(prefix: str = "difference  dI = frame − ref",
                 gain: float = DIFF_GAIN) -> str:
    """Describe the actual display gain rather than duplicating it in labels."""
    return f"{prefix}  (colour)" if abs(gain - 1.0) < 1e-9 \
        else f"{prefix}  (×{gain:g}, colour)"


class ForceOverlay:
    """Fixed clip layout; draw the O(N) timeline once for all N frames."""

    def __init__(self, force: np.ndarray, *, maximum: float):
        self.force = np.asarray(force)
        self.maximum = maximum
        self.renderer = Renderer(tile_width=640, tile_height=480, columns=1)
        self.background = np.zeros((560, 640, 3), np.uint8)
        self.xs = np.linspace(40, 600, len(force)).astype(int)
        self.ys = (556 - 12 * self.force / maximum).astype(int)
        for i in range(1, len(self.xs)):
            cv2.line(self.background, (self.xs[i - 1], self.ys[i - 1]),
                     (self.xs[i], self.ys[i]), (140, 140, 140), 1)

    def render(self, image: np.ndarray, row: int) -> np.ndarray:
        """Compose a tactile RGB tile and the row's force in owned BGR pixels."""
        canvas = self.background.copy()
        canvas[:480] = self.renderer.render([
            Tile("tactile", image, modality="tactile", color_format="RGB")
        ]).image
        value = self.force[row]
        cv2.rectangle(canvas, (40, 520), (600, 540), (60, 60, 60), 1)
        cv2.rectangle(canvas, (40, 520),
                      (40 + int(560 * (value / self.maximum)), 540),
                      (30, 120, 240), -1)
        cv2.putText(canvas, f"F_n = {value:.3f} N", (40, 512),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.circle(canvas, (self.xs[row], self.ys[row]), 3, (30, 120, 240), -1)
        return canvas
