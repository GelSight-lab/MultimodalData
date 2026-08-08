"""Contact sheets for EVERY video-corruption interval the curator flagged.

Two versions of the GelSight corruption detector were validated against a
hand-labelled set of five episodes, and both were falsified by the sixth: the
magenta-area rule flagged a probe tip at 0.0110 against a 0.0100 threshold, and
the magenta-onset rule that replaced it survived about an hour before the same
episode's touchdown frame stepped 0.0000 -> 0.0064.

The failure was not the thresholds. It was validating against the examples that
motivated the rule. So this script does not sample: it renders one sheet per
flagged interval over the whole release, so every flag can be looked at.

    python scripts/eyeball_flags.py [--task motherboard] [--out DIR]

Each sheet is 5 frames wide (interval start -4, start, middle, end, end +4) and
one row per tactile sensor, with the frame index and the measured row-fill
burned in. The two frames outside the interval are the control: if they look
identical to the ones inside, the flag is a false positive.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from react_preprocess import detect as D

RELEASE = Path("/media/yxma/Disk1/twm/release")
TILE_W, TILE_H = 320, 240


def _frames(mp4: Path, idx: list[int]) -> dict[int, Image.Image]:
    """Decode just the frames we need, once, in one pass."""
    out: dict[int, Image.Image] = {}
    want = sorted(set(i for i in idx if i >= 0))
    if not want:
        return out
    sel = "+".join(f"eq(n\\,{i})" for i in want)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-vf",
         f"select='{sel}',scale={TILE_W}:{TILE_H}", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    step = TILE_W * TILE_H * 3
    for k, i in enumerate(want):
        if (k + 1) * step > len(raw):
            break
        out[i] = Image.frombytes(
            "RGB", (TILE_W, TILE_H), raw[k * step:(k + 1) * step])
    return out


def sheet(task: str, date: str, ep: str, a: int, b: int, kind: str,
          out_dir: Path) -> Path | None:
    vd = RELEASE / task / "videos" / date / ep
    streams = sorted(vd.glob("tactile_*.mp4") if kind == "tactile_corruption"
                     else vd.glob("view_*.mp4"))
    if not streams:
        return None
    picks = [a - 4, a, (a + b) // 2, b, b + 4]
    rows = []
    for mp4 in streams:
        _, rowfill = D._video_stats(mp4)
        got = _frames(mp4, picks)
        tiles = []
        for i in picks:
            im = got.get(i)
            if im is None:
                im = Image.new("RGB", (TILE_W, TILE_H), (20, 20, 20))
            im = im.copy()
            dr = ImageDraw.Draw(im)
            inside = a <= i <= b
            rf = rowfill[i] if 0 <= i < len(rowfill) else float("nan")
            dr.text((6, 6), f"{mp4.stem} {i}", fill=(255, 255, 0))
            dr.text((6, 20), f"rowfill={rf:.3f}",
                    fill=(255, 80, 80) if inside else (150, 255, 150))
            if inside:
                dr.rectangle([0, 0, TILE_W - 1, TILE_H - 1],
                             outline=(255, 0, 0), width=3)
            tiles.append(im)
        rows.append(tiles)

    sh = Image.new("RGB", (TILE_W * len(picks), TILE_H * len(rows)))
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sh.paste(t, (c * TILE_W, r * TILE_H))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{task}_{date}_{ep}_{kind}_{a}-{b}.png"
    sh.save(p)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append",
                    choices=["motherboard", "pushT"])
    ap.add_argument("--out", default="_eyeball")
    args = ap.parse_args()
    tasks = args.task or ["motherboard", "pushT"]
    out_dir = Path(args.out)

    n = 0
    for task in tasks:
        bf = json.loads((RELEASE / task / "bad_frames.json").read_text())
        for key, fams in sorted(bf["episodes"].items()):
            date, ep = key.split("/")
            for kind in ("tactile_corruption", "cam_corruption"):
                for a, b in fams.get(kind, []):
                    p = sheet(task, date, ep, int(a), int(b), kind, out_dir)
                    print(f"  {p}" if p else f"  (no video) {key} {kind}",
                          flush=True)
                    n += 1
    print(f"[eyeball] {n} interval(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
