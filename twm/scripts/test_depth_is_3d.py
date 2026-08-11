"""Every depth map the site shows is a rendered surface, not a heatmap.

A depth map drawn as a colour-mapped image asks the reader to convert hue into
height in their head, and they cannot: `inferno` at 60% looks like `inferno` at
70%, so a shallow dent and a deep one read the same. The repo already has an
Open3D renderer used for the workbench meshes; the figures kept their own
`imshow(d, cmap=...)`.

Two checks, because either alone can pass while the page is still wrong:

  1  `eval_panel.mesh` — the site's ONE mesh renderer, the same call the
     workbench uses — really renders a surface. Sabotage it back to a colormap
     and this fails. No second renderer was added for the figures: a figure
     and the interactive view of the same frame must not be able to disagree.
  2  no figure module still calls `imshow` on a depth array. A helper nobody
     calls fixes nothing.

    xvfb-run -a -s "-screen 0 1400x1000x24" python scripts/test_depth_is_3d.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
ROOT = Path(__file__).resolve().parent.parent

# Every module that draws a figure the site publishes.
FIGURE_MODULES = [
    "force_recovery/error_analysis.py",
    "force_recovery/showcase.py",
    "force_recovery/sparsh_channel_fix.py",
    "force_recovery/site2_figures.py",
    "force_recovery/truncation_figure.py",
]
# `imshow` on any of these names is a depth map drawn flat.
DEPTH_NAMES = r"(?:d|dd|z|depth|depth_mm|dep)"


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from force_recovery.o3d_view import has_display

    # 1 — the renderer responds to GEOMETRY, not to value.
    #
    # Feed it a depth map that is exactly symmetric top-to-bottom. Any colormap
    # of it is symmetric too — a colormap is a per-pixel function of the value,
    # so it commutes with a flip. A surface viewed from a raised camera cannot
    # be: the near half of the cone occludes and shades differently from the
    # far half. Asymmetry in the output is therefore proof that a camera and a
    # light were involved.
    #
    # My first attempt compared a cone against `max - cone` "with identical
    # value histograms". They do not have identical histograms — the cone is
    # mostly zeros, so its inverse is mostly the peak value — and the check
    # failed on its own premise rather than on the code. Symmetry is checked
    # here on the input, in the assertion, so the premise cannot rot.
    if not has_display():
        check(False, "depth is rendered as a surface",
              "UNVERIFIED: no DISPLAY, run under xvfb-run")
    else:
        from force_recovery.eval_panel import mesh as depth_axis_rgb
        h, w = 161, 201
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
        cone = np.clip(1.0 - r / 60.0, 0, None) * 2.0
        sym = np.array_equal(cone, cone[::-1])
        a = depth_axis_rgb(cone).astype(float)
        flip = float(np.abs(a - a[::-1]).mean())
        check(sym and flip > 5.0, "depth is rendered as a surface",
              f"input flip-symmetric: {sym}; its render is not, by "
              f"{flip:.1f}/255 per pixel — a colormap commutes with a flip")

    # 2 — and every figure actually goes through it.
    stray = []
    for rel in FIGURE_MODULES:
        src = (ROOT / rel).read_text()
        for i, line in enumerate(src.splitlines(), 1):
            # BOTH forms. The second pattern was `\.imshow\(.*cmap=...` and it
            # missed `showcase.py`, which draws its depth through a local
            # `add(d, title, cmap="inferno")` helper rather than calling
            # imshow directly — a real flat depth map that the check reported
            # as clean. A sequential colormap is the tell wherever it appears,
            # so it is matched anywhere on the line now.
            if re.search(rf"\.imshow\(\s*{DEPTH_NAMES}\b", line) or \
               re.search(r"cmap\s*=\s*[\"'](?:inferno|magma|"
                         r"turbo|jet|viridis)[\"']", line):
                if "# flat-ok:" in line:
                    continue                 # an explicitly non-depth heatmap
                stray.append(f"{rel}:{i}")
    check(not stray, "no figure draws a depth map flat",
          f"{len(stray)} flat depth imshow"
          + (": " + ", ".join(stray) if stray else ""))

    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    bad = sum(not ok for ok, _, _ in RESULTS)
    print(f"\ndepth is 3D: {len(RESULTS)} checks, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
