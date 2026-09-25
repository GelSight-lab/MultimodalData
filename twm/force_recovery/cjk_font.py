"""A CJK font for matplotlib, with the glyph coverage actually verified.

WHY THIS IS NOT JUST rcParams['font.family'] = 'Noto Sans CJK'

`recon_compare.png` shipped to the live site with its title rendered as a row
of tofu boxes. Setting the family had "worked" — matplotlib fell back
silently, drew U+FFFD for every Chinese character, and emitted a warning
nobody read. The figure looked fine to the code and wrong to every reader.

So the font is not selected, it is CHECKED: the chosen file's cmap is read
with fontTools and every character of every label is looked up. A missing
glyph raises here, at build time, instead of becoming a box in a PNG.

    from .cjk_font import use_cjk
    use_cjk(["深度", "标定无关"])        # raises if anything is uncoverable
"""
from __future__ import annotations

from pathlib import Path

# Preference order. The first one whose file exists AND covers the text wins.
CANDIDATES = ("Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK SC",
              "AR PL UMing CN", "WenQuanYi Zen Hei")


def _font_path(name: str) -> str | None:
    import matplotlib.font_manager as fm
    for f in fm.fontManager.ttflist:
        if f.name == name:
            return f.fname
    return None


def _covers(path: str, chars: set[str]) -> set[str]:
    """Characters in `chars` that the font at `path` has NO glyph for."""
    from fontTools.ttLib import TTCollection, TTFont

    fonts = (TTCollection(path).fonts if str(path).lower().endswith(".ttc")
             else [TTFont(path, fontNumber=0)])
    covered: set[int] = set()
    for f in fonts:
        for table in f["cmap"].tables:
            covered |= set(table.cmap.keys())
    return {c for c in chars if ord(c) not in covered}


def use_cjk(texts) -> str:
    """Point matplotlib at a font proven to cover `texts`. Returns its name."""
    import matplotlib

    chars = {c for t in texts for c in str(t) if not c.isascii()}
    tried = []
    for name in CANDIDATES:
        p = _font_path(name)
        if p is None:
            tried.append(f"{name}: not installed")
            continue
        missing = _covers(p, chars) if chars else set()
        if missing:
            tried.append(f"{name}: missing {''.join(sorted(missing))!r}")
            continue
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return name
    raise SystemExit(
        "no installed font covers these labels — a figure would ship tofu "
        "boxes, which is what this check exists to stop:\n  "
        + "\n  ".join(tried))
