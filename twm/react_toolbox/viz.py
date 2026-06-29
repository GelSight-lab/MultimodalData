"""Visualization helpers — all return RGB uint8 (H, W, 3) images you can
imshow / save, no display side effects.
"""
from __future__ import annotations

import numpy as np

from .reference import difference, l2_diff
from .contact import contact_mask


def _colormap(gray01, name="viridis"):
    import cv2
    g = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    cmaps = {"viridis": cv2.COLORMAP_VIRIDIS, "jet": cv2.COLORMAP_JET,
             "magma": cv2.COLORMAP_MAGMA, "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)}
    bgr = cv2.applyColorMap(g, cmaps.get(name, cv2.COLORMAP_VIRIDIS))
    return bgr[..., ::-1].copy()   # BGR->RGB


def diff_heatmap(frame, reference, vmax=None, cmap="turbo"):
    """L2 difference-from-reference as a heatmap (contact intensity)."""
    d = l2_diff(frame, reference)
    vmax = vmax if vmax is not None else max(d.max(), 1e-6)
    return _colormap(d / vmax, cmap)


def contact_overlay(frame, reference, tau=8.0, color=(255, 0, 0), alpha=0.45):
    """Tint the contact region on top of the raw frame."""
    m = contact_mask(frame, reference, tau=tau)
    out = frame.copy().astype(np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def reference_compare(frame, reference):
    """Side-by-side [reference | frame | signed-diff] strip."""
    sd = difference(frame, reference, signed=True)
    return np.concatenate([reference, frame, sd], axis=1)


def depth_view(height_map, cmap="gray"):
    """Render a (H, W) height map as an RGB image.

    Default is **grayscale** (brighter = higher), the standard GelSight
    height-map convention (gsrobotics, GelSight Wedge, depth-recon papers).
    Pass cmap="turbo"/"jet"/"viridis" for a colormapped view instead.
    """
    h = height_map.astype(np.float32)
    rng = h.max() - h.min()
    norm = (h - h.min()) / (rng + 1e-6)
    if cmap in (None, "gray", "grey", "grayscale"):
        g = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)   # (H,W,3) gray RGB
    return _colormap(norm, cmap)


def height_to_pointcloud(height_map, stride=4, z_scale=1.0):
    """(H, W) height -> (N, 3) point cloud (x, y, z) for 3D rendering."""
    h = height_map[::stride, ::stride]
    ys, xs = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    return np.stack([xs.ravel(), ys.ravel(), (h * z_scale).ravel()], axis=1).astype(np.float32)
