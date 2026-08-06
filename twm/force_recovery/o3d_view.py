"""Open3D mesh rendering of reconstructed gel surfaces.

Adapted from the visualizer in Yuxiang-Ma/ProbingPi
(`scripts/tactile/common/{figure,render}.py`): a full-resolution grid
TriangleMesh in mm with gradient-grey shading baked into the vertex colours,
rendered offscreen with `light_on = False` so the baked relief is exactly what
appears. Open3D's legacy viewer only offers a camera headlight, which hits this
camera-facing surface square on and washes every slope to the same white; the
paper look needs light raking across the surface.

Two environment facts, both measured here:

* Open3D 0.15.2's `Visualizer` SEGFAULTS with `visible=False` on this machine
  (no GL context). It works under `xvfb-run -a -s "-screen 0 1400x1000x24"`,
  which is how `render_depth_mesh` must be invoked. `has_display()` checks this
  so callers fail loudly instead of dumping core.
* The reconstruction being rendered is NOT noisy: high-frequency content in the
  depth field is 0.3% of peak depth (0.0065 mm on a 1.9 mm press). The
  speckle visible in gradient-domain panels is LUT bin quantisation (22.2% HF
  in the gradient field, 9.8% of contact pixels landing in unobserved bins that
  are nearest-filled); Poisson integration low-passes it away. `smooth_px`
  below is therefore cosmetic anti-aliasing for the render only and is
  deliberately NOT applied to the depth used for force features.
"""
from __future__ import annotations

import os

import numpy as np

from .lut_calibration import MM_PER_PIXEL

# Oblique paper camera (9DTact-style), shared with ProbingPi.
VIEW_FRONT = (0.0, 0.55, 0.84)
VIEW_UP = (0.0, -1.0, 0.0)
VIEW_ZOOM = 0.62


def has_display() -> bool:
    """Offscreen Open3D needs an X display here; see module docstring."""
    return bool(os.environ.get("DISPLAY"))


def gradient_shade(z: np.ndarray, contrast: float = 0.30,
                   smooth_px: int = 0) -> np.ndarray:
    """9DTact relief shading: normalised (dx+dy) mapped into a grey band."""
    import cv2

    dy, dx = np.gradient(z.astype(np.float64))
    shade = dx + dy
    if smooth_px >= 3:
        shade = cv2.GaussianBlur(shade, (smooth_px, smooth_px), 0)
    lo, hi = np.percentile(shade, [2, 98])
    if hi - lo > 1e-9:
        shade = np.clip((shade - lo) / (hi - lo), 0, 1)
    else:
        shade = np.full_like(shade, 0.5)
    return 0.5 + (shade - 0.5) * 2.0 * contrast + 0.12


def remove_halo_pedestal(depth_mm: np.ndarray, contact_frac: float = 0.25
                         ) -> np.ndarray:
    """Flatten the diffuse halo skirt that Poisson integration adds.

    The `|dI| > 8` valid mask is halo-dominated: around a shaped indenter it is
    a blob considerably larger than the true contact (measured on GlowTact —
    the mask outline is round even for star/triangle/quad, while the DEPTH
    contours at 0.3/0.6 of peak recover the shapes correctly). Integrating the
    halo gradients therefore raises a broad low pedestal under the shape, which
    an oblique 3D view renders as a rounded mound and hides the geometry.

    Estimate the pedestal as the median depth in the annulus just outside the
    contact contour and subtract it. Cosmetic: applied to figures only, never
    to the depth used for force features (those are calibrated per group, so a
    constant offset is absorbed by the fit anyway).
    """
    import cv2

    peak = float(depth_mm.max())
    if peak <= 1e-6:
        return depth_mm
    contact = (depth_mm > contact_frac * peak).astype(np.uint8)
    ring = cv2.dilate(contact, np.ones((21, 21), np.uint8)) - contact
    base = float(np.median(depth_mm[ring > 0])) if ring.any() else 0.0
    return np.maximum(depth_mm - base, 0.0)


def build_depth_mesh(depth_mm: np.ndarray, mm_per_px: float,
                     smooth_px: int = 5, z_scale: float = 1.0,
                     stride: int = 1, contrast: float = 0.30):
    """Full-grid mm-scale TriangleMesh with baked gradient-grey vertex colour.

    `z_scale=1` is true geometric scale. Triangle winding (a,b,c)/(b,d,c) puts
    the normals on the +z side facing the camera — the mirrored winding
    back-face-culls the whole surface (ProbingPi hit exactly this).
    """
    import cv2
    import open3d as o3d

    z = depth_mm.astype(np.float32)
    if smooth_px >= 3:
        z = cv2.GaussianBlur(z, (smooth_px, smooth_px), 0)
    shade = gradient_shade(z, contrast)
    if stride > 1:
        z = z[::stride, ::stride]
        shade = shade[::stride, ::stride]
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    scale = mm_per_px * stride
    # x mirrored: the view (front +y/+z, up -y) renders screen right as -x, so
    # the double mirror puts image left/right back where the raw frame has it.
    vertices = np.stack([(w - 1 - xs).ravel() * scale, ys.ravel() * scale,
                         z.ravel().astype(np.float64) * z_scale], axis=1)
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(
        np.concatenate([np.stack([a, b, c], axis=1),
                        np.stack([b, d, c], axis=1)]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.repeat(np.clip(shade, 0, 1).ravel()[:, None], 3, axis=1))
    mesh.compute_vertex_normals()
    return mesh


def render_mesh(mesh, width: int = 900, height: int = 700, bg: float = 1.0,
                zoom: float = VIEW_ZOOM, front=VIEW_FRONT, up=VIEW_UP
                ) -> np.ndarray:
    """Offscreen render -> RGB uint8. Requires a display (see has_display)."""
    import open3d as o3d

    if not has_display():
        raise RuntimeError(
            "Open3D offscreen rendering segfaults without a display here; "
            "run under: xvfb-run -a -s '-screen 0 1400x1000x24'")
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([bg, bg, bg])
    opt.light_on = False              # baked shading only
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_lookat(mesh.get_center())
    ctr.set_front(list(front))
    ctr.set_up(list(up))
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (np.clip(buf, 0, 1) * 255).astype(np.uint8)


def crop_to_content(rgb: np.ndarray, bg_tol: float = 0.02, pad: int = 8
                    ) -> np.ndarray:
    """Trim the uniform background border an Open3D render leaves around the
    mesh.

    Open3D fits the camera to the scene bounding SPHERE, so a flat wide pad
    renders with a large empty margin — embedded in a figure panel the mesh
    ends up occupying a fraction of the tile. Cropping to the non-background
    bounding box makes the surface fill its panel without changing the camera
    (which would also change the perspective between frames of a clip).
    """
    lum = rgb.mean(axis=2) / 255.0
    bg = float(np.median(np.concatenate(
        [lum[0], lum[-1], lum[:, 0], lum[:, -1]])))
    mask = np.abs(lum - bg) > bg_tol
    if not mask.any():
        return rgb
    ys, xs = np.nonzero(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, rgb.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, rgb.shape[1])
    return rgb[y0:y1, x0:x1]


def render_depth_mesh(depth_mm: np.ndarray, mm_per_px: float, **kw
                      ) -> np.ndarray:
    """depth map -> rendered RGB image, one call."""
    build_kw = {k: kw.pop(k) for k in
                ("smooth_px", "z_scale", "stride", "contrast") if k in kw}
    return render_mesh(build_depth_mesh(depth_mm, mm_per_px, **build_kw), **kw)


MESH_KW = dict(smooth_px=3, z_scale=1.6, front=(0.0, 0.32, 0.95), zoom=0.66)


def mesh_view_rgb(depth_mm: np.ndarray, width: int = 620, height: int = 500,
                  stride: int = 1) -> np.ndarray:
    """Depth (already pedestal-corrected) -> cropped Open3D mesh render.

    Kept here rather than in showcase.py so analysis scripts can render a mesh
    without importing the site-building module.
    """
    return crop_to_content(render_depth_mesh(
        np.clip(depth_mm, 0.0, None).astype(np.float32), MM_PER_PIXEL,
        width=width, height=height, stride=stride, **MESH_KW))
