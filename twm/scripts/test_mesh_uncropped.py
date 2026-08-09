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

import numpy as np

from force_recovery import calib_free as CF
from force_recovery import eval_panel as EP
from force_recovery.debug_gallery import load_glowtact, stages
from force_recovery.o3d_view import content_box, has_display, mesh_view_rgb

MIN_FILL = 0.80
# The fixed camera measures 0.000 spread on all three paths. The threshold is
# just above the floor deliberately: 0.05 would have passed the depth-dependent
# framing bug this test was written to catch.
MAX_FILL_SPREAD = 0.01


def _fill(rgb: np.ndarray) -> tuple[float, float]:
    y0, y1, x0, x1, border = content_box(rgb, pad=0)
    return (x1 - x0) / rgb.shape[1], border


def main() -> int:
    if not has_display():
        print("no DISPLAY — run under xvfb-run; NOT reporting a pass")
        return 2
    bad: list[str] = []

    rows, get = load_glowtact()
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.permutation(len(rows))[:5]]

    fills: dict[str, list[float]] = {"tile-lut": [], "tile-cf": [], "rgb": []}
    for fr in sel:
        img, ref = get(fr)
        lut = stages(img, ref)["depth"]
        cf = CF.reconstruct(img, ref)["depth"]
        cases = [("tile-lut", EP.mesh(lut)),
                 ("tile-cf", EP.mesh(cf, relative=True)),
                 ("rgb", mesh_view_rgb(np.clip(lut, 0, None), stride=2))]
        for key, rgb in cases:
            f, border = _fill(rgb)
            fills[key].append(f)
            if border > 0.0:
                bad.append(f"{key} {fr.get('group','?')}: mesh touches the "
                           f"render border ({border:.3f} of an edge) — cropped")
            if f < MIN_FILL:
                bad.append(f"{key} {fr.get('group','?')}: surface fills only "
                           f"{f:.2f} of the tile width (< {MIN_FILL})")

    for key, v in fills.items():
        spread = max(v) - min(v)
        print(f"  {key:9s} fill {min(v):.3f}-{max(v):.3f}  spread {spread:.3f}")
        if spread > MAX_FILL_SPREAD:
            bad.append(f"{key}: fill varies {spread:.3f} across frames — the "
                       f"meshes in one column are at different scales")

    src = (EP.mesh.__module__, "force_recovery/showcase.py")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for p in ("force_recovery/showcase.py", "force_recovery/o3d_view.py"):
        t = (root / p).read_text()
        for i, line in enumerate(t.splitlines(), 1):
            if "crop_to_contact(" in line and not line.lstrip().startswith("#"):
                bad.append(f"{p}:{i}: contact crop is back in the render path")
    del src

    for b in bad:
        print(f"  FAIL: {b}")
    print(f"mesh-uncropped: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
