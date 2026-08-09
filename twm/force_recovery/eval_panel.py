"""The one way an evaluation figure draws a reconstruction.

Every validation figure on the site shows the same three things about a
contact — what changed in the image, what depth was recovered, and what that
depth is as a surface — and until now each figure drew them its own way. The
MNIST panel showed depth as two heat maps and no mesh; the React depth panel
showed a greyscale `|frame - ref|` and one mesh; the marker-removal panel
showed something else again. A reader comparing two figures was comparing two
conventions.

So: one function per quantity, called by every figure.

    diff_rgb(img, ref)      signed colour difference    (visualize.diff_rgb)
    depth_heat(d)           depth as a heat map, mm
    mesh(d)                 Open3D surface              (showcase.mesh_view)

`mesh` and the camera it uses (`o3d_view.MESH_KW`) are imported, never
restated — the same law that renders the workbench renders the figures, so a
surface looks the same wherever it appears.

WHY A MESH AND NOT JUST A HEAT MAP. A heat map of depth answers "how deep";
it does not show whether the recovered surface is the object. The React panel
that started this showed a plausible heat map whose mesh was a mound with a
ridge — the failure was legible in the surface and invisible in the colours.
An evaluation figure that omits the mesh omits the failure mode.

Requires an X display for Open3D (`xvfb-run -a` off a desktop); `available()`
says so rather than letting a figure silently fall back to no mesh.
"""
from __future__ import annotations

import numpy as np

from .o3d_view import has_display
from .visualize import diff_caption, diff_rgb

__all__ = ["diff_rgb", "diff_caption", "depth_heat", "mesh", "available",
           "panel_row",
           "COLUMNS"]

# What every evaluation figure shows, in this order. Named so a figure states
# which columns it drops rather than quietly having fewer.
COLUMNS = ("input", "difference", "depth", "mesh")

DEPTH_CMAP = "inferno"


def available() -> bool:
    """Can Open3D render here? Figures must ask, not assume."""
    return has_display()


def depth_heat(d: np.ndarray) -> np.ndarray:
    """Depth as-is; the caller supplies the colour map via imshow."""
    return np.clip(np.asarray(d, np.float32), 0.0, None)


def mesh(d: np.ndarray, w: int = 320, h: int = 240,
         relative: bool = False) -> np.ndarray:
    """Open3D surface of a depth map — the site's one mesh renderer.

    `relative=True` normalises to the frame's own peak first. Required for any
    reconstruction whose scale is not calibrated: the renderer applies a FIXED
    z exaggeration, so an uncalibrated magnitude 3.5x the LUT's draws the same
    correct geometry as a vertical tower. That is what made the
    calibration-free surfaces look broken when they were not
    (`calib_free.RETURNS_MILLIMETRES`).

    Passing relative=True for a depth that IS in millimetres would throw away
    real information — the height difference between a light press and a hard
    one — so it is opt-in per caller, not the default.
    """
    from .showcase import mesh_tile
    d = np.clip(np.asarray(d, np.float32), 0.0, None)
    if relative:
        d = d / max(float(d.max()), 1e-12)
    return mesh_tile(d, w=w, h=h, bg=1.0)


def panel_row(ax_row, img, ref, depth, titles=None, mm_note: str = "") -> None:
    """Draw one contact across a row of four axes, in the site's convention.

    `ax_row` must have at least len(COLUMNS) axes. A figure that wants fewer
    columns slices COLUMNS itself, so the omission is visible in its code.
    """
    t = dict(zip(COLUMNS, titles or (
        "input frame",
        diff_caption(),
        f"depth [mm]{mm_note}",
        "3D reconstruction (Open3D mesh)")))
    cells = [
        (np.clip(np.asarray(img), 0, 255).astype(np.uint8), None, t["input"]),
        (diff_rgb(img, ref), None, t["difference"]),
        (depth_heat(depth), DEPTH_CMAP, t["depth"]),
        (mesh(depth) if available() else np.zeros((240, 320, 3), np.uint8),
         None, t["mesh"] if available() else "mesh — no display"),
    ]
    for ax, (data, cmap, title) in zip(ax_row, cells):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
