"""Gate: freeze the layout defect classes we just fixed, so they can't return.

`python -m force_recovery.design_guard` exits non-zero on any violation.

Each rule exists because of a measured defect, not a preference:

1. PANEL WIDTH. Multi-stage figures were emitted as one long strip: the
   14-panel workbench rendered each panel 136 px inside its 1900 px column
   and its titles at ~6 px effective size; the 8-panel gallery gave 110 px
   and ~2.8 px. Two-row layouts doubled both. The rule asserts the figure's
   aspect ratio implies >= MIN_PANEL_PX per panel at the page column it is
   shown in, which is what forces a second row rather than naming "2 rows"
   directly — a 4-panel figure is legitimately one row.

2. TITLE LEGIBILITY. Derived, not independent: a title drawn at `pt` in a
   figure of width W shown in a column of width C renders at pt * C / W.
   Asserted >= MIN_TITLE_PX.

3. NO STALE SITE COPIES. The site keeps copies of figures generated
   elsewhere; twice now a regenerated figure did not reach `site/assets`
   and the page kept serving the old strip. Assert every site copy matches
   its source byte-for-byte where a source exists.

Bounds are deliberately just under what we now achieve, so a real
regression trips them and normal variation does not.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

from .run_episode import OUT_ROOT

SITE = OUT_ROOT / "site"
MIN_PANEL_PX = 180.0        # measured after fix: 220-271
MIN_TITLE_PX = 4.0          # measured after fix: 6.0-10.5; before: 2.8

# figure -> (page column width px, panel count, title point size in the figure)
FIGURES = {
    "assets/recon/glowtact_01.png": (1900, 14, 10.5),
    "assets/recon/glowtact_06.png": (1900, 14, 10.5),
    "assets/gallery/feats_04.png": (880, 8, 10.0),
    "assets/gallery/cnc_00.png": (880, 8, 10.0),
    "assets/debug/cnc_01.png": (880, 7, 11.0),
    "assets/debug/glowtact_04.png": (880, 7, 11.0),
}

# site copy -> source of truth
MIRRORS = {
    "assets/recon/glowtact_01.png":
        OUT_ROOT / "recon_study" / "glowtact_01.png",
    "assets/debug/cnc_01.png":
        OUT_ROOT / "site_assets" / "debug_gallery" / "cnc" / "sample_01.png",
}


def _rows_and_cols(w: int, h: int, n: int) -> tuple[int, int]:
    """Recover the grid from the aspect ratio, assuming 4:3 panels.

    A one-row strip of n 4:3 panels has aspect ~ n*4/3; two rows ~ n*2/3.
    We only need the implied columns-per-row, which is what sets panel width.
    """
    for rows in range(1, 5):
        cols = -(-n // rows)
        implied = (cols * 4) / (rows * 3)
        if abs(w / h - implied) / implied < 0.45:
            return rows, cols
    return 1, n


def main() -> int:
    bad = []
    for rel, (column, n, title_pt) in FIGURES.items():
        p = SITE / rel
        if not p.exists():
            bad.append(f"MISSING  {rel}")
            continue
        w, h = Image.open(p).size
        rows, cols = _rows_and_cols(w, h, n)
        panel_px = column / cols
        title_px = title_pt * column / w
        if panel_px < MIN_PANEL_PX:
            bad.append(f"NARROW   {rel}: {panel_px:.0f} px/panel "
                       f"({cols} cols x {rows} rows) < {MIN_PANEL_PX:.0f}")
        if title_px < MIN_TITLE_PX:
            bad.append(f"TINYTEXT {rel}: title renders {title_px:.1f} px "
                       f"< {MIN_TITLE_PX:.1f}")

    for rel, src in MIRRORS.items():
        dst = SITE / rel
        if not (src.exists() and dst.exists()):
            continue
        if hashlib.md5(src.read_bytes()).hexdigest() != \
                hashlib.md5(dst.read_bytes()).hexdigest():
            bad.append(f"STALE    {rel} differs from {src}")

    for line in bad:
        print(line)
    print(f"design guard: {len(FIGURES)} figures, {len(MIRRORS)} mirrors, "
          f"{len(bad)} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
