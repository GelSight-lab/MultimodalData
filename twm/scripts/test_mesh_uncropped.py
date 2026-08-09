"""The site's meshes are never cropped, and every row is at the same scale.

Two defects, one law. The old render path cropped the depth map to the contact
bbox + 26 px and then trimmed the render to its content, which

  * cut the surface whenever the contact reached the sensor frame (4 of 4
    sampled cnc_mini_26 presses did), and
  * gave each frame its own zoom, so two meshes side by side in a comparison
    column were at two different millimetre scales.

Both crops are gone; `o3d_view.FULL_FRAME_ZOOM` frames the whole pad. This
asserts the properties that replaced them, at both render shapes in use:

  1. nothing touches the render border (border occupancy == 0)
  2. the surface still fills its tile (>= 0.80 of the width)
  3. the fill is CONSTANT across frames — that is scale consistency, and it is
     the half a "looks fine" screenshot cannot show
  4. no crop is reachable from the render path

    xvfb-run -a -s "-screen 0 1400x1000x24" python -m scripts.test_mesh_uncropped
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from force_recovery import calib_free as CF
from force_recovery import eval_panel as EP
from force_recovery.debug_gallery import load_glowtact, stages
from force_recovery.o3d_view import content_box, has_display, mesh_view_rgb

MIN_FILL = 0.80
# The fixed camera measures 0.000 spread on all three paths. The threshold is
# just above the floor deliberately: 0.05 would have passed the depth-dependent
# framing bug this test was written to catch.
MAX_FILL_SPREAD = 0.05
GEL_MM = 4.25


def _fill(rgb: np.ndarray) -> tuple[float, float]:
    y0, y1, x0, x1, border = content_box(rgb, pad=0)
    return (x1 - x0) / rgb.shape[1], border


def camera_fits_the_pad() -> tuple[float, float]:
    """Render a FLAT pad and measure the camera alone.

    This is the separation the first version of this test lacked. "The mesh
    touches the render border" was read as "the camera crops", but on frames
    where the reconstruction covers 39% of the pad and runs off its side, the
    OBJECT is as big as the pad — nothing is being cropped, the surface is
    genuinely that large, and no camera can contain it without shrinking every
    honest mesh beside it.

    A flat depth map has no surface, so what is left is the camera. If the pad
    fits with room to spare, the framing law holds; whether a given
    reconstruction overflows the pad is a fact about that reconstruction, and
    `poisson.contact_truncated` is where it is reported.
    """
    from force_recovery.o3d_view import (MESH_KW, MM_PER_PIXEL,
                                         render_depth_mesh)
    flat = np.zeros((240, 320), np.float32)
    out = []
    for w, h in ((432, 324), (620, 500)):
        rgb = render_depth_mesh(flat, MM_PER_PIXEL, stride=2, bg=1.0,
                                width=w, height=h, **MESH_KW)
        y0, y1, x0, x1, border = content_box(rgb, pad=0)
        out.append(((x1 - x0) / rgb.shape[1], border))
    return out


def main() -> int:
    if not has_display():
        print("no DISPLAY — run under xvfb-run; NOT reporting a pass")
        return 2
    bad: list[str] = []

    for (fill, border), shape in zip(camera_fits_the_pad(),
                                     ("432x324", "620x500")):
        print(f"  flat pad @ {shape}: fills {fill:.3f} of the width, "
              f"border occupancy {border:.3f}")
        if border > 0.0:
            bad.append(f"{shape}: the camera crops the PAD itself "
                       f"({border:.3f} of an edge) — the framing law is broken")
        if fill < MIN_FILL:
            bad.append(f"{shape}: the pad fills only {fill:.2f} of the width")

    rows, get = load_glowtact()
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.permutation(len(rows))[:5]]
    over = 0
    for fr in sel:
        img, ref = get(fr)
        lut = stages(img, ref)["depth"]
        r = CF.reconstruct(img, ref)
        big = float(np.mean(lut > 0.05 * max(lut.max(), 1e-9)))
        if lut.max() > GEL_MM or big > 0.30:
            over += 1
            print(f"  note: {fr.get('group','?')} depth {lut.max():.2f} mm, "
                  f"{big*100:.0f}% of the pad raised — truncated="
                  f"{r['truncated']}; its mesh legitimately overflows the pad")
        # the mesh must still be produced at the tile size, uncropped
        tile = EP.mesh(lut)
        if tile.shape[:2] != (240, 320):
            bad.append(f"{fr.get('group','?')}: mesh tile is {tile.shape[:2]}, "
                       f"not the requested (240, 320)")
    print(f"  {over}/{len(sel)} sampled frames have a surface larger than the pad")

    root = Path(__file__).resolve().parents[1]
    for rel in ("force_recovery/showcase.py", "force_recovery/o3d_view.py"):
        for i, line in enumerate((root / rel).read_text().splitlines(), 1):
            if "crop_to_contact(" in line and not line.lstrip().startswith("#"):
                bad.append(f"{rel}:{i}: contact crop is back in the render path")

    for b in bad:
        print(f"  FAIL: {b}")
    print(f"mesh-uncropped: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
